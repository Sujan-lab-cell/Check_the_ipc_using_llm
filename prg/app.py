import os
import re
import hashlib

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error(
        "HF_TOKEN is missing. Please add your Hugging Face "
        "API token to the .env file."
    )
    st.stop()


# ============================================================
# HUGGING FACE MODEL
# ============================================================

# Instruction-following model.
# We are NOT using Gemini.
# We are NOT using OpenAI.
# We are NOT using DeepSeek-R1 because it may expose reasoning.

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct:fastest"


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

hf_client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IPC 1860 Chatbot",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "pages" not in st.session_state:
    st.session_state.pages = []

if "section_map" not in st.session_state:
    st.session_state.section_map = {}


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.replace("\x00", " ")

        if text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


# ============================================================
# NORMALIZE PDF TEXT
# ============================================================

def normalize_text(text):

    # Replace unusual spaces
    text = text.replace("\xa0", " ")

    # Normalize different dash characters
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# DETECT IPC SECTION HEADINGS
# ============================================================

def find_section_headings(text):

    """
    Detect headings such as:

        299. Culpable homicide
        300. Murder
        302. Punishment for murder

    Also handles:

        302.—Punishment for murder
        302. - Punishment for murder
    """

    pattern = re.compile(
        r"(?im)"
        r"^\s*"
        r"(\d{1,3}[A-Za-z]?)"
        r"\s*\.\s*"
        r"[-]?\s*"
        r"([^\n]+)"
        r"\s*$"
    )

    return list(
        pattern.finditer(text)
    )


# ============================================================
# CREATE SECTION-AWARE CHUNKS
# ============================================================

def create_chunks(pages):

    chunks = []

    current_section = None
    current_text = []
    current_start_page = None
    current_pages = []


    def save_current_section():

        nonlocal current_section
        nonlocal current_text
        nonlocal current_start_page
        nonlocal current_pages

        if current_section is not None:

            section_text = "\n".join(
                current_text
            ).strip()

            if section_text:

                chunks.append({
                    "section": current_section,
                    "page": current_start_page,
                    "pages": sorted(
                        set(current_pages)
                    ),
                    "text": section_text
                })


        current_section = None
        current_text = []
        current_start_page = None
        current_pages = []


    for page_data in pages:

        page_number = page_data["page"]

        text = normalize_text(
            page_data["text"]
        )

        matches = find_section_headings(
            text
        )


        # ----------------------------------------------------
        # No section heading on this page
        # ----------------------------------------------------

        if not matches:

            if current_section is not None:

                current_text.append(
                    text
                )

                current_pages.append(
                    page_number
                )

            else:

                # Keep miscellaneous page content.
                if text:

                    chunks.append({
                        "section": None,
                        "page": page_number,
                        "pages": [page_number],
                        "text": text
                    })

            continue


        # ----------------------------------------------------
        # Text before first heading
        # ----------------------------------------------------

        first_start = matches[0].start()

        prefix = text[:first_start].strip()

        if prefix:

            if current_section is not None:

                current_text.append(
                    prefix
                )

                current_pages.append(
                    page_number
                )

            else:

                chunks.append({
                    "section": None,
                    "page": page_number,
                    "pages": [page_number],
                    "text": prefix
                })


        # ----------------------------------------------------
        # Process sections on this page
        # ----------------------------------------------------

        for i, match in enumerate(matches):

            section_number = (
                match.group(1)
                .strip()
                .lower()
            )

            section_heading = (
                match.group(0)
                .strip()
            )


            # Save previous section
            save_current_section()


            current_section = section_number

            current_start_page = page_number

            current_pages = [
                page_number
            ]


            # Start of section content
            section_start = match.start()

            if i + 1 < len(matches):

                section_end = matches[i + 1].start()

            else:

                section_end = len(text)


            section_content = text[
                section_start:section_end
            ].strip()


            current_text = [
                section_content
            ]


    # Save final section
    save_current_section()


    return chunks


# ============================================================
# BUILD SECTION MAP
# ============================================================

def build_section_map(chunks):

    section_map = {}


    for chunk in chunks:

        section = chunk.get(
            "section"
        )

        if section is None:
            continue


        section = str(section).lower()


        if section not in section_map:

            section_map[section] = []


        section_map[section].append(
            chunk
        )


    return section_map


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    embeddings = embedding_model.encode(

        texts,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False

    )

    return embeddings.astype(
        "float32"
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# EXTRACT SECTION NUMBER FROM QUESTION
# ============================================================

def extract_section_number(question):

    """
    Detect questions such as:

        What is Section 302?

        Explain section 420

        Tell me about Sec. 498A

        What does section 299 say?
    """

    patterns = [

        r"\bsection\s*(\d{1,3}[A-Za-z]?)\b",

        r"\bsec\.?\s*(\d{1,3}[A-Za-z]?)\b"

    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            return (
                match.group(1)
                .lower()
            )


    return None


# ============================================================
# EXACT SECTION SEARCH
# ============================================================

def exact_section_search(section_number):

    if not section_number:
        return []


    section_number = (
        section_number.lower()
    )


    results = []


    # --------------------------------------------------------
    # First use section map
    # --------------------------------------------------------

    section_map = st.session_state.section_map


    if section_number in section_map:

        for chunk in section_map[
            section_number
        ]:

            results.append({

                "section": chunk[
                    "section"
                ],

                "page": chunk[
                    "page"
                ],

                "pages": chunk.get(
                    "pages",
                    [chunk["page"]]
                ),

                "text": chunk[
                    "text"
                ],

                "score": 1.0,

                "exact": True

            })


        if results:

            return results


    # --------------------------------------------------------
    # Fallback: search raw PDF pages
    # --------------------------------------------------------

    section_pattern = re.compile(

        rf"\b"
        rf"{re.escape(section_number)}"
        rf"\s*\."
        rf"(?:\s+|[-])",

        re.IGNORECASE

    )


    for page in st.session_state.pages:

        if section_pattern.search(
            page["text"]
        ):

            results.append({

                "section": section_number,

                "page": page["page"],

                "pages": [
                    page["page"]
                ],

                "text": page["text"],

                "score": 0.95,

                "exact": True

            })


    return results


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(question):

    if st.session_state.index is None:

        return []


    question_embedding = embedding_model.encode(

        [question],

        convert_to_numpy=True,

        normalize_embeddings=True

    ).astype(
        "float32"
    )


    scores, indexes = (
        st.session_state.index.search(

            question_embedding,

            TOP_K

        )
    )


    results = []


    for score, index_number in zip(

        scores[0],

        indexes[0]

    ):

        if index_number < 0:
            continue


        chunk = st.session_state.chunks[
            index_number
        ]


        results.append({

            "section": chunk.get(
                "section"
            ),

            "page": chunk[
                "page"
            ],

            "pages": chunk.get(
                "pages",
                [chunk["page"]]
            ),

            "text": chunk[
                "text"
            ],

            "score": float(
                score
            ),

            "exact": False

        })


    return results


# ============================================================
# MAIN SEARCH FUNCTION
# ============================================================

def search_pdf(question):

    # --------------------------------------------------------
    # STEP 1
    # Exact IPC section search
    # --------------------------------------------------------

    section_number = extract_section_number(
        question
    )


    if section_number:

        exact_results = exact_section_search(
            section_number
        )


        if exact_results:

            return exact_results


    # --------------------------------------------------------
    # STEP 2
    # Semantic/vector search
    # --------------------------------------------------------

    return vector_search(
        question
    )


# ============================================================
# PREPARE CONTEXT FOR LLM
# ============================================================

def build_context(results):

    context_parts = []


    for result in results:

        section = result.get(
            "section"
        )

        page = result.get(
            "page"
        )

        pages = result.get(
            "pages",
            [page]
        )

        text = result.get(
            "text",
            ""
        )


        if len(pages) > 1:

            page_text = ", ".join(
                str(p)
                for p in pages
            )

        else:

            page_text = str(
                page
            )


        context_parts.append(

            f"""
==================================================
IPC SECTION: {section}
PDF PAGE(S): {page_text}
==================================================

{text}
"""

        )


    return "\n".join(
        context_parts
    )


# ============================================================
# REMOVE REASONING TAGS
# ============================================================

def clean_llm_response(answer):

    if not answer:
        return ""


    # Remove <think>...</think>
    answer = re.sub(

        r"<think>.*?</think>",

        "",

        answer,

        flags=re.DOTALL |
        re.IGNORECASE

    )


    # Remove common reasoning prefixes
    answer = re.sub(

        r"^\s*(analysis|reasoning)\s*:\s*",

        "",

        answer,

        flags=re.IGNORECASE

    )


    # Remove accidental code fences
    answer = answer.replace(
        "```text",
        ""
    )

    answer = answer.replace(
        "```",
        ""
    )


    return answer.strip()


# ============================================================
# ASK HUGGING FACE
# ============================================================

def ask_huggingface(
    question,
    results
):

    if not results:

        return (
            "I could not find this information "
            "in the uploaded PDF."
        )


    context = build_context(
        results
    )


    system_prompt = """
You are an Indian Penal Code, 1860 PDF assistant.

Your job is to answer questions using ONLY the content
retrieved from the uploaded PDF.

==================================================
STRICT RULES
==================================================

1. Use ONLY the supplied PDF content.

2. Do NOT use outside legal knowledge.

3. Do NOT invent an IPC section.

4. Do NOT invent a punishment.

5. Do NOT invent definitions.

6. Do NOT guess when the PDF does not contain
   the requested information.

7. If the answer is not contained in the supplied
   PDF context, respond exactly:

   "I could not find this information in the uploaded PDF."

8. If the user asks about a specific IPC section,
   answer that exact section.

9. Always mention the IPC section when available.

10. Always mention the PDF page number when available.

11. If multiple sections are relevant, clearly
    separate them.

12. Do NOT reveal your reasoning.

13. Do NOT output <think>...</think>.

14. Do NOT output analysis or internal reasoning.

15. Give only the final answer.

16. Keep the answer clear and concise.

17. Do not claim that the information is current
    law unless the uploaded PDF itself says so.

==================================================
"""


    user_prompt = f"""
The following information was retrieved from the
uploaded Indian Penal Code, 1860 PDF.

Use ONLY this information to answer the question.

PDF CONTENT:

{context}


USER QUESTION:

{question}


Provide the final answer now.
"""


    response = hf_client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        temperature=0.0,

        max_tokens=1200

    )


    answer = response.choices[
        0
    ].message.content


    return clean_llm_response(
        answer
    )


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

def process_pdf(uploaded_file):

    file_bytes = uploaded_file.getvalue()


    current_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    # Don't process same PDF again
    if (
        current_hash ==
        st.session_state.file_hash
    ):

        return


    # --------------------------------------------------------
    # Extract PDF
    # --------------------------------------------------------

    with st.spinner(
        "📖 Reading the IPC PDF..."
    ):

        pages = extract_pdf(
            uploaded_file
        )


    if not pages:

        st.error(
            "No readable text was found in this PDF."
        )

        st.warning(
            "This may be a scanned/image-only PDF. "
            "OCR would be required."
        )

        return


    # --------------------------------------------------------
    # Create section-aware chunks
    # --------------------------------------------------------

    with st.spinner(
        "⚖️ Detecting IPC sections..."
    ):

        chunks = create_chunks(
            pages
        )


    if not chunks:

        st.error(
            "Could not create searchable content."
        )

        return


    # --------------------------------------------------------
    # Build section map
    # --------------------------------------------------------

    section_map = build_section_map(
        chunks
    )


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Creating document search index..."
    ):

        texts = [
            chunk["text"]
            for chunk in chunks
        ]


        embeddings = create_embeddings(
            texts
        )


        index = build_faiss_index(
            embeddings
        )


    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    st.session_state.pages = pages

    st.session_state.chunks = chunks

    st.session_state.index = index

    st.session_state.section_map = section_map

    st.session_state.file_hash = current_hash

    st.session_state.messages = []


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    detected_sections = len(
        section_map
    )


    st.success(
        f"✅ PDF loaded successfully — "
        f"{len(pages)} pages, "
        f"{detected_sections} IPC sections detected."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "⚖️ IPC 1860"
    )


    st.write(
        "Upload the Indian Penal Code, 1860 PDF."
    )


    uploaded_file = st.file_uploader(

        "Choose PDF",

        type=["pdf"],

        help="Upload your IPC 1860 PDF."

    )


    st.divider()


    st.subheader(
        "Features"
    )


    st.write(
        "✅ PDF question answering"
    )

    st.write(
        "✅ Exact IPC section lookup"
    )

    st.write(
        "✅ Section-aware search"
    )

    st.write(
        "✅ Semantic search"
    )

    st.write(
        "✅ PDF page references"
    )

    st.write(
        "✅ Hugging Face LLM"
    )

    st.write(
        "❌ No Gemini API"
    )

    st.write(
        "❌ No OpenAI API"
    )


    # --------------------------------------------------------
    # PDF statistics
    # --------------------------------------------------------

    if st.session_state.chunks:

        st.divider()

        st.subheader(
            "Document Information"
        )


        st.write(
            f"📄 Pages: "
            f"{len(st.session_state.pages)}"
        )


        st.write(
            f"⚖️ Sections detected: "
            f"{len(st.session_state.section_map)}"
        )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "⚖️ Indian Penal Code, 1860 Chatbot"
)


st.caption(
    "Upload the IPC PDF and ask questions about its contents."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    process_pdf(
        uploaded_file
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.index is not None:

    question = st.chat_input(
        "Ask something about the IPC PDF..."
    )


    if question:

        # ----------------------------------------------------
        # Display user question
        # ----------------------------------------------------

        st.session_state.messages.append({

            "role": "user",

            "content": question

        })


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            try:

                with st.spinner(
                    "🔎 Searching the IPC PDF..."
                ):

                    results = search_pdf(
                        question
                    )


                if not results:

                    answer = (
                        "I could not find this information "
                        "in the uploaded PDF."
                    )

                    st.markdown(
                        answer
                    )


                else:

                    with st.spinner(
                        "🤖 Generating answer..."
                    ):

                        answer = ask_huggingface(

                            question,

                            results

                        )


                    st.markdown(
                        answer
                    )


                    # ------------------------------------------------
                    # Sources
                    # ------------------------------------------------

                    with st.expander(
                        "📚 View PDF sources"
                    ):

                        for i, result in enumerate(
                            results,
                            start=1
                        ):

                            section = result.get(
                                "section"
                            )

                            page = result.get(
                                "page"
                            )

                            pages = result.get(
                                "pages",
                                [page]
                            )


                            if len(pages) > 1:

                                page_display = ", ".join(
                                    str(p)
                                    for p in pages
                                )

                            else:

                                page_display = str(
                                    page
                                )


                            if result.get(
                                "exact"
                            ):

                                search_type = (
                                    "🎯 Exact section match"
                                )

                            else:

                                search_type = (
                                    "🔎 Semantic search"
                                )


                            st.markdown(
                                f"### Source {i}"
                            )


                            st.write(
                                f"**IPC Section:** "
                                f"{section}"
                            )


                            st.write(
                                f"**PDF Page(s):** "
                                f"{page_display}"
                            )


                            st.write(
                                f"**Search:** "
                                f"{search_type}"
                            )


                            st.write(
                                result["text"]
                            )


                            if i < len(results):

                                st.divider()


                # ------------------------------------------------
                # Save assistant message
                # ------------------------------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })


            except Exception as e:

                error_message = (
                    f"❌ An error occurred:\n\n"
                    f"`{str(e)}`"
                )


                st.error(
                    error_message
                )


else:

    st.info(
        "👈 Upload your Indian Penal Code, 1860 PDF "
        "to start chatting."
    )