"""Atlas Research — a user-friendly, multi-PDF RAG workspace."""

from __future__ import annotations

import hashlib
import html
import os
import uuid
from typing import Any

import fitz
import streamlit as st

from rag_pipeline import (
    delete_collection_if_exists,
    generate_answer,
    retrieve_relevant_chunks,
    store_chunks_in_chromadb,
)
from utils.pdf_processor import process_multiple_pdfs

st.set_page_config(
    page_title="Atlas Research | AI Research Workspace",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Remember the selected appearance when Streamlit reruns.
st.session_state.setdefault("atlas_dark_mode", False)

# A fresh visual direction: bright editorial workspace, soft lavender surfaces,
# confident indigo actions, mint status details, and generous spacing.
st.markdown(
    r"""
    <style>
    :root {
      --atlas-ink:#20243A;
      --atlas-muted:#626982;
      --atlas-line:#E3E6F2;
      --atlas-purple:#5A4CE6;
      --atlas-purple-dark:#4338CA;
      --atlas-lilac:#F0EEFF;
      --atlas-mint:#DDF8EF;
      --atlas-green:#167D62;
    }
    .stApp { background:linear-gradient(180deg,#F8F9FE 0%,#F4F5FB 70%,#F7F8FC 100%); color:var(--atlas-ink); }
    [data-testid="stAppViewContainer"] { background:transparent; }
    [data-testid="stHeader"] { background:rgba(248,249,254,.88); }
    [data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E7E9F3; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.15rem; }
    .block-container { max-width:1500px; padding-top:1.4rem; padding-bottom:2.5rem; }
    h1,h2,h3,h4,p,li,label { color:var(--atlas-ink); }
    h1,h2,h3 { letter-spacing:-.035em; }
    h1 { font-weight:780; }
    [data-testid="stMarkdownContainer"] p { line-height:1.55; }
    [data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--atlas-line); border-radius:17px; padding:1rem 1.1rem; box-shadow:0 5px 18px rgba(45,48,95,.035); }
    [data-testid="stMetricLabel"] p { color:var(--atlas-muted)!important; font-size:.82rem; font-weight:600; }
    [data-testid="stMetricValue"] { color:var(--atlas-ink)!important; font-weight:780; }
    [data-testid="stMetricDelta"] { color:var(--atlas-green)!important; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF; border:1px solid var(--atlas-line)!important; border-radius:18px!important; box-shadow:0 7px 22px rgba(45,48,95,.035); }
    div.stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:11px; font-weight:650; min-height:2.65rem; border:1px solid #DADDF1; transition:all .15s ease; }
    div.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:linear-gradient(100deg,#5547E5,#7465EE); color:#FFFFFF; border:1px solid #5547E5; box-shadow:0 5px 14px rgba(90,76,230,.18); }
    div.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background:#4338CA; border-color:#4338CA; box-shadow:0 7px 20px rgba(90,76,230,.22); transform:translateY(-1px); }
    div.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible { outline:3px solid #AAA4FF!important; outline-offset:2px!important; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { background:#FFFFFF!important; color:var(--atlas-ink)!important; border:1px solid #DDE1EF!important; border-radius:10px!important; }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#8075F4!important; box-shadow:0 0 0 2px rgba(90,76,230,.13)!important; }
    [data-testid="stFileUploader"] { border:1px dashed #B8B5EC; border-radius:14px; background:#F8F7FF; padding:.35rem; }
    [data-testid="stFileUploaderDropzone"] { background:transparent; border:0; }
    [data-testid="stTabs"] button { color:#70768E; font-weight:650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--atlas-purple-dark); border-bottom-color:var(--atlas-purple); }
    [data-testid="stExpander"] { border:1px solid var(--atlas-line); border-radius:13px; background:#FFFFFF; }
    [data-testid="stChatMessage"] { background:#FFFFFF; border:1px solid var(--atlas-line); border-radius:15px; padding:.8rem .9rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--atlas-line); border-radius:12px; overflow:hidden; }
    hr { border-color:#E6E8F2!important; }
    /* Sidebar */
    .atlas-brand { display:flex; align-items:center; gap:11px; padding:.1rem .15rem 1rem .15rem; }
    .atlas-mark { width:43px; height:43px; display:flex; align-items:center; justify-content:center; border-radius:14px; color:#FFF; font-size:1.42rem; font-weight:800; background:linear-gradient(140deg,#4D3BC9,#8174FF 75%,#6ADBCB); box-shadow:0 7px 17px rgba(90,76,230,.19); }
    .atlas-brand-title { font-size:1.13rem; font-weight:800; color:#22243A; letter-spacing:-.04em; line-height:1.2; }
    .atlas-brand-caption { color:#8A8FA6; font-size:.65rem; font-weight:750; letter-spacing:.13em; margin-top:4px; }
    .atlas-side-label { color:#666D84; text-transform:uppercase; letter-spacing:.13em; font-size:.66rem; font-weight:800; margin:.65rem 0 .35rem .2rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { color:#777D96; font-size:.76rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap:.22rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label { border-radius:10px; padding:.43rem .55rem; color:#4C526B; transition:background .12s ease; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover { background:#F2F1FF; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:#F0EEFF; color:#493DD2; font-weight:750; }
    .atlas-divider { height:1px; background:#EBECF4; margin:.85rem 0; }
    /* Main-page components */
    .atlas-eyebrow { text-transform:uppercase; color:#6358C5; letter-spacing:.15em; font-weight:800; font-size:.68rem; }
    .atlas-page-title { margin:.32rem 0 .45rem 0; font-size:2.05rem; line-height:1.1; font-weight:820; letter-spacing:-.055em; color:#20243A; }
    .atlas-page-subtitle { color:#69708A; font-size:.96rem; max-width:740px; line-height:1.55; }
    .atlas-hero { position:relative; overflow:hidden; background:linear-gradient(112deg,#4E43CC 0%,#675BE6 54%,#8B80FA 100%); color:#FFFFFF; padding:1.65rem 1.7rem; border-radius:23px; box-shadow:0 12px 27px rgba(83,68,211,.19); margin:.3rem 0 1.05rem 0; }
    .atlas-hero:after { content:""; position:absolute; width:260px; height:260px; border-radius:50%; right:-60px; top:-140px; border:1px solid rgba(255,255,255,.22); box-shadow:0 0 0 27px rgba(255,255,255,.045),0 0 0 57px rgba(255,255,255,.035); pointer-events:none; }
    .atlas-hero-kicker { color:#E0DEFF; font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .atlas-hero-title { margin:.52rem 0 .55rem 0; color:#FFFFFF; font-size:1.85rem; font-weight:820; letter-spacing:-.045em; line-height:1.16; max-width:700px; }
    .atlas-hero-text { color:#F0EFFF; font-size:.92rem; line-height:1.55; max-width:720px; }
    .atlas-pill { display:inline-block; padding:.33rem .63rem; border-radius:999px; font-size:.72rem; font-weight:750; color:#5143CA; background:#F0EEFF; border:1px solid #E3DFFF; }
    .atlas-pill-green { color:#18785D; background:#E1F8EF; border-color:#C7F0E1; }
    .atlas-pill-muted { color:#656B83; background:#F1F2F8; border-color:#E6E8F1; }
    .atlas-section-heading { font-size:1.08rem; font-weight:780; letter-spacing:-.025em; color:#242741; margin:0; }
    .atlas-section-caption { color:#687089; font-size:.78rem; margin-top:.2rem; }
    .atlas-document { border:1px solid #E5E7F2; background:#FFFFFF; border-radius:14px; padding:.85rem .9rem; margin:.45rem 0; }
    .atlas-doc-icon { width:40px; height:43px; flex:0 0 40px; display:flex; align-items:center; justify-content:center; border-radius:11px; background:#F0EEFF; color:#5547D5; font-size:1.15rem; border:1px solid #E4E0FF; }
    .atlas-doc-name { color:#292C46; font-size:.88rem; font-weight:750; overflow-wrap:anywhere; line-height:1.35; }
    .atlas-doc-meta { color:#6D748B; font-size:.73rem; line-height:1.5; margin-top:.18rem; }
    .atlas-empty { border:1px dashed #C9C8E9; border-radius:17px; padding:1.2rem; background:#FCFBFF; color:#70758D; }
    .atlas-empty-title { color:#30324A; font-weight:780; font-size:1rem; margin:.3rem 0; }
    .atlas-small-muted { color:#687089; font-size:.76rem; line-height:1.45; }
    .atlas-source-title { color:#3B3B5F; font-weight:750; font-size:.84rem; }
    .atlas-footer { color:#6E748B; text-align:center; padding:1.4rem 0 .25rem 0; font-size:.72rem; }
    @media(max-width:900px) { .block-container { padding-left:1rem; padding-right:1rem; } .atlas-page-title { font-size:1.65rem; } .atlas-hero-title { font-size:1.5rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# Apply dark theme overrides when Dark mode is enabled.
if st.session_state["atlas_dark_mode"]:
    st.markdown(
        """
        <style>
        /* Main application background */
        html {
            color-scheme: dark !important;
        }

        .stApp {
            background: linear-gradient(
                135deg,
                #0B1020 0%,
                #11182C 55%,
                #15152D 100%
            ) !important;
            color: #E8ECF8 !important;
        }

        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0.92) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #101729 !important;
            border-right: 1px solid #29334D !important;
        }

        [data-testid="stSidebar"] * {
            --atlas-ink: #E8ECF8;
            --atlas-muted: #A6B0C8;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #DCE3F5 !important;
        }

        /* Headings and ordinary text */
        h1, h2, h3, h4, p, li, label,
        [data-testid="stMarkdownContainer"] {
            color: #E8ECF8;
        }

        .atlas-page-title,
        .atlas-section-heading,
        .atlas-empty-title {
            color: #F1F3FF !important;
        }

        .atlas-page-subtitle,
        .atlas-section-caption,
        .atlas-small-muted,
        .atlas-doc-meta,
        .atlas-footer {
            color: #A5AEC7 !important;
        }

        .atlas-eyebrow {
            color: #B9B1FF !important;
        }

        /* Metrics and bordered cards */
        [data-testid="stMetric"],
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"],
        [data-testid="stChatMessage"] {
            background: #172137 !important;
            border-color: #303B59 !important;
            color: #E8ECF8 !important;
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.12) !important;
        }

        [data-testid="stMetricLabel"] p {
            color: #AAB4CE !important;
        }

        [data-testid="stMetricValue"] {
            color: #F4F5FF !important;
        }

        /* Document cards */
        .atlas-document {
            background: #1B2740 !important;
            border-color: #34415F !important;
        }

        .atlas-doc-name {
            color: #F1F4FF !important;
        }

        .atlas-doc-meta {
            color: #AAB4CC !important;
        }

        .atlas-doc-icon {
            background: #29274D !important;
            color: #C2BAFF !important;
            border-color: #464078 !important;
        }

        .atlas-empty {
            background: #172137 !important;
            border-color: #465174 !important;
            color: #BAC4DC !important;
        }

        /* Text inputs, text areas and selection controls */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #111A2D !important;
            color: #F0F3FF !important;
            border-color: #3A4664 !important;
        }

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: #909BB5 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: #151F35 !important;
            border-color: #655CD0 !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
        }

        /* Navigation */
        [data-testid="stSidebar"] [data-testid="stRadio"]
        div[role="radiogroup"] label {
            color: #C1C9DF !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"]
        div[role="radiogroup"] label:hover {
            background: #232744 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"]
        div[role="radiogroup"] label:has(input:checked) {
            background: #302B5A !important;
            color: #D3CDFF !important;
        }

        /* Tabs and expanders */
        [data-testid="stTabs"] button {
            color: #ABB5D0 !important;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #C7C0FF !important;
            border-bottom-color: #9286FF !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p {
            color: #E8ECF8 !important;
        }

        /* Data tables and separators */
        [data-testid="stDataFrame"] {
            border-color: #34415F !important;
        }

        hr,
        .atlas-divider {
            border-color: #303B59 !important;
            background-color: #303B59 !important;
        }

        /* Normal buttons. Keep primary buttons vibrant. */
        div.stButton > button:not([kind="primary"]),
        [data-testid="stFormSubmitButton"]
        button:not([kind="primary"]) {
            background: #202B43 !important;
            color: #E7EBF8 !important;
            border-color: #3A4664 !important;
        }

        div.stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"]
        button[kind="primary"] {
            color: #FFFFFF !important;
        }

        /* Alerts */
        [data-testid="stAlert"] {
            background: #18243A !important;
            color: #E8ECF8 !important;
        }

        /* Scrollbar */
        * {
            scrollbar-color: #465174 #111A2D;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        html {
            color-scheme: light !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def process_documents_cached(
    document_payloads: tuple[tuple[str, bytes], ...], chunk_size: int, chunk_overlap: int
):
    """Cache text extraction and chunking across normal Streamlit reruns."""
    return process_multiple_pdfs(document_payloads, chunk_size=chunk_size, overlap=chunk_overlap)


@st.cache_data(show_spinner=False)
def render_pdf_page(pdf_bytes: bytes, page_number: int) -> bytes:
    """Render one PDF page as a PNG preview."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if page_number < 1 or page_number > len(document):
            raise ValueError("Page number is outside this PDF.")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        return pixmap.tobytes("png")


def initialize_state() -> None:
    defaults = {
        "atlas_session_id": uuid.uuid4().hex[:12],
        "atlas_documents_hash": None,
        "atlas_collection": None,
        "atlas_indexed_hash": None,
        "atlas_indexed_chunk_size": None,
        "atlas_indexed_overlap": None,
        "atlas_chat_history": [],
        "atlas_search_results": [],
        "atlas_last_search": "",
        "atlas_tool_result": None,
        "atlas_tool_sources": [],
        "atlas_build_requested": False,
        "atlas_pending_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def request_index_build() -> None:
    st.session_state.atlas_build_requested = True


def go_to_research_chat() -> None:
    """Switch to the chat page safely before Streamlit recreates the widgets."""
    st.session_state.atlas_navigation = "Research chat"


def hash_documents(payloads: tuple[tuple[str, bytes], ...]) -> str:
    hasher = hashlib.sha256()
    for name, data in payloads:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(data).digest())
        hasher.update(b"\n")
    return hasher.hexdigest()


def make_collection_name(session_id: str, documents_hash: str) -> str:
    return f"atlas_{session_id}_{documents_hash[:12]}"


def groq_key_is_set() -> bool:
    if os.getenv("GROQ_API_KEY", "").strip():
        return True
    try:
        return bool(str(st.secrets.get("GROQ_API_KEY", "")).strip())
    except Exception:
        return False


def render_page_header(title: str, subtitle: str, indexed: bool) -> None:
    index_label = "Index ready" if indexed else "Index not built"
    index_class = "atlas-pill atlas-pill-green" if indexed else "atlas-pill atlas-pill-muted"
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;margin:.2rem 0 1.1rem 0">
          <div>
            <div class="atlas-eyebrow">ATLAS RESEARCH STUDIO&nbsp; / &nbsp;WORKSPACE</div>
            <h1 class="atlas-page-title">{title}</h1>
            <div class="atlas-page-subtitle">{subtitle}</div>
          </div>
          <div style="padding-top:.5rem"><span class="{index_class}">{index_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_card(summary: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="atlas-document">
          <div style="display:flex;gap:.75rem;align-items:center">
            <div class="atlas-doc-icon">▤</div>
            <div style="min-width:0;flex:1">
              <div class="atlas-doc-name">{html.escape(str(summary['source']))}</div>
              <div class="atlas-doc-meta">{summary['page_count']:,} text pages &nbsp;·&nbsp; {summary['word_count']:,} words &nbsp;·&nbsp; {summary['chunk_count']:,} chunks</div>
            </div>
            <span class="atlas-pill atlas-pill-green">PDF</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ask_research_question(question: str, collection_name: str, top_k: int, model_name: str):
    sources = retrieve_relevant_chunks(question, collection_name, top_k=top_k)
    if not sources:
        return "I couldn't find relevant evidence in the indexed documents for that question.", []
    answer = generate_answer(question, sources, model_name=model_name)
    return answer, sources


def show_evidence(sources: list[dict[str, Any]], *, expanded: bool = False) -> None:
    if not sources:
        return
    with st.expander(f"Sources and evidence · {len(sources)} passages", expanded=expanded):
        for rank, source in enumerate(sources, start=1):
            similarity = source.get("similarity")
            score_note = f" · similarity {similarity:.2f}" if isinstance(similarity, (float, int)) else ""
            st.markdown(
                f"**{rank}. {source.get('source', 'Unknown PDF')} · page {source.get('page_number', '?')}**{score_note}"
            )
            st.write(source.get("text", ""))
            if rank < len(sources):
                st.divider()


def render_chat(collection_name: str | None, is_indexed: bool, top_k: int, model_name: str) -> None:
    if not is_indexed or not collection_name:
        st.markdown(
            '<div class="atlas-empty"><div style="font-size:1.35rem">✳</div><div class="atlas-empty-title">Your research space is waiting</div><div>Upload text-based PDFs and build your index to ask questions with source citations.</div></div>',
            unsafe_allow_html=True,
        )
        return

    if not groq_key_is_set():
        st.warning("Your document index is ready, but answer generation needs a Groq API key. Add GROQ_API_KEY in your environment or Streamlit Secrets.")

    if not st.session_state.atlas_chat_history:
        with st.container(border=True):
            st.markdown("**Start with a question like these**")
            prompt_cols = st.columns(3)
            suggestions = [
                "Summarize the main findings.",
                "What limitations do the documents mention?",
                "Compare the key ideas across documents.",
            ]
            for col, prompt in zip(prompt_cols, suggestions):
                with col:
                    if st.button(prompt, key=f"suggest_{hashlib.md5(prompt.encode()).hexdigest()}", use_container_width=True):
                        st.session_state.atlas_pending_question = prompt
                        st.rerun()

    for entry in st.session_state.atlas_chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                show_evidence(entry.get("sources", []))

    with st.form("atlas_chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ask your document library",
            value=st.session_state.atlas_pending_question,
            placeholder="Ask a question across all indexed PDFs…",
            height=95,
            help="Questions are answered using retrieved PDF passages. Important claims should be checked against the cited source pages.",
        )
        submitted = st.form_submit_button("✦  Find evidence and answer", type="primary", use_container_width=True)

    if submitted:
        question = question.strip()
        st.session_state.atlas_pending_question = ""
        if not question:
            st.warning("Please enter a question.")
        elif not model_name:
            st.warning("Enter a Groq model ID in Settings.")
        elif not groq_key_is_set():
            st.error("GROQ_API_KEY is not configured. Your PDFs can be indexed, but Groq cannot generate an answer until a key is provided.")
        else:
            try:
                with st.spinner("Reading relevant passages and drafting a cited answer…"):
                    answer, sources = ask_research_question(question, collection_name, top_k, model_name)
                st.session_state.atlas_chat_history.append({"role": "user", "content": question})
                st.session_state.atlas_chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()
            except Exception as exc:
                st.error(f"I couldn't complete this request. Check your API key, model ID, and connection. Details: {exc}")

    st.caption("Answers are grounded in retrieved passages, not guaranteed to cover every page. Open the evidence panel to verify the source.")


initialize_state()

with st.sidebar:
    st.markdown(
        '<div class="atlas-brand"><div class="atlas-mark">✳</div><div><div class="atlas-brand-title">Atlas Research</div><div class="atlas-brand-caption">AI RESEARCH WORKSPACE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.toggle(
        "🌙 Dark mode",
        key="atlas_dark_mode",
        help="Switch between light and dark appearance.",
    )
    st.markdown('<div class="atlas-divider"></div><div class="atlas-side-label">Your workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace navigation",
        ["Overview", "Research chat", "Search evidence", "Document library", "Research tools", "Settings"],
        label_visibility="collapsed",
        key="atlas_navigation",
    )
    st.markdown('<div class="atlas-divider"></div><div class="atlas-side-label">Add documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="atlas_pdf_uploader",
        help="Select one or more text-based PDFs. Each file is limited by the upload-size setting.",
    )
    st.caption("Your PDFs are processed by this app. Avoid uploading confidential files to a public deployment.")
    st.markdown('<div class="atlas-divider"></div>', unsafe_allow_html=True)
    with st.expander("Index settings", expanded=False):
        chunk_size = int(st.slider("Chunk size (words)", 100, 1000, 500, 50))
        chunk_overlap = int(st.slider("Overlap (words)", 0, 400, 100, 25))
        retrieval_count = int(st.slider("Evidence passages per question", 1, 10, 5))
        model_name = st.text_input("Groq model ID", value="openai/gpt-oss-20b", help="Answer-generation model hosted by Groq.").strip()
    st.markdown('<div class="atlas-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="atlas-small-muted">Made for careful reading, evidence discovery, and cross-document research.</div>', unsafe_allow_html=True)

if chunk_overlap >= chunk_size:
    st.error("Chunk overlap must be smaller than chunk size. Adjust Index settings in the sidebar.")
    st.stop()

# Keep only stable (filename, bytes) payloads from Streamlit upload handles.
document_payloads = tuple((file.name, file.getvalue()) for file in (uploaded_files or []))
documents_hash = hash_documents(document_payloads) if document_payloads else ""

if document_payloads:
    try:
        with st.spinner("Preparing your PDF library…"):
            pages, chunks, document_summaries = process_documents_cached(document_payloads, chunk_size, chunk_overlap)
    except Exception as exc:
        st.error(f"One of the PDFs could not be processed. It may be damaged or password-protected. Details: {exc}")
        st.stop()
else:
    pages, chunks, document_summaries = [], [], []

# If the active document set changes, prevent old citations/history from appearing under the new library.
if documents_hash != st.session_state.atlas_documents_hash:
    previous_collection = st.session_state.atlas_collection
    if previous_collection:
        try:
            delete_collection_if_exists(previous_collection)
        except Exception:
            pass
    st.session_state.atlas_documents_hash = documents_hash
    st.session_state.atlas_collection = None
    st.session_state.atlas_indexed_hash = None
    st.session_state.atlas_chat_history = []
    st.session_state.atlas_search_results = []
    st.session_state.atlas_last_search = ""
    st.session_state.atlas_tool_result = None
    st.session_state.atlas_tool_sources = []

is_indexed = (
    bool(documents_hash)
    and st.session_state.atlas_indexed_hash == documents_hash
    and st.session_state.atlas_indexed_chunk_size == chunk_size
    and st.session_state.atlas_indexed_overlap == chunk_overlap
    and bool(st.session_state.atlas_collection)
)

# The callback sets this flag before a rerun, so indexing can happen before the page is rendered.
if st.session_state.atlas_build_requested:
    st.session_state.atlas_build_requested = False
    if not document_payloads:
        st.warning("Upload at least one PDF before building the research index.")
    elif not chunks:
        st.warning("No selectable text was found in the uploaded PDFs. Scanned PDFs require OCR, which is not included yet.")
    else:
        try:
            collection_name = make_collection_name(st.session_state.atlas_session_id, documents_hash)
            with st.spinner(f"Indexing {len(chunks):,} chunks from {len(document_summaries)} PDF(s)…"):
                collection = store_chunks_in_chromadb(chunks, collection_name, source_name="Atlas Research library")
            st.session_state.atlas_collection = collection.name
            st.session_state.atlas_indexed_hash = documents_hash
            st.session_state.atlas_indexed_chunk_size = chunk_size
            st.session_state.atlas_indexed_overlap = chunk_overlap
            st.session_state.atlas_chat_history = []
            st.session_state.atlas_search_results = []
            st.session_state.atlas_tool_result = None
            is_indexed = True
            st.toast(f"Research index ready · {collection.count():,} chunks", icon="✅")
        except Exception as exc:
            st.error(f"Index creation failed. Check dependencies and available memory. Details: {exc}")

# Count meaningful text pages and words, rather than the total pages in the PDF file.
page_count = len(pages)
word_count = sum(len(page["text"].split()) for page in pages)
chunk_count = len(chunks)
indexed_label = "Ready" if is_indexed else ("Needs indexing" if document_payloads else "No documents")

if page == "Overview":
    render_page_header(
        "A calmer way to research.",
        "Bring your PDFs together, ask better questions, and follow every answer back to its source.",
        is_indexed,
    )
    st.markdown(
        '<div class="atlas-hero"><div class="atlas-hero-kicker">Your AI research desk</div><h2 class="atlas-hero-title">Turn a stack of papers into a clearer point of view.</h2><div class="atlas-hero-text">Search across multiple PDFs, connect related ideas, and get answers with source-page citations — all in one focused workspace.</div></div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("PDF documents", f"{len(document_summaries):,}")
    metric_cols[1].metric("Text pages", f"{page_count:,}")
    metric_cols[2].metric("Searchable chunks", f"{chunk_count:,}")
    metric_cols[3].metric("Workspace status", indexed_label)

    st.write("")
    main_left, main_right = st.columns([1.18, .82], gap="large")
    with main_left:
        with st.container(border=True):
            st.markdown('<div class="atlas-section-heading">Your document library</div><div class="atlas-section-caption">Files ready to become searchable knowledge</div>', unsafe_allow_html=True)
            if document_summaries:
                for summary in document_summaries[:5]:
                    render_document_card(summary)
                if len(document_summaries) > 5:
                    st.caption(f"Showing 5 of {len(document_summaries)} documents. Open Document library to see all files.")
                button_cols = st.columns([1, 1.1])
                with button_cols[0]:
                    st.button(
                        "✦  Build research index" if not is_indexed else "↻  Rebuild index",
                        type="primary",
                        use_container_width=True,
                        on_click=request_index_build,
                        key="overview_build_index",
                    )
                with button_cols[1]:
                    st.caption("Indexing turns document text into vectors so similar passages can be found for your questions.")
                if is_indexed:
                    st.success("Your library is searchable. Open Research chat to start asking questions.")
                elif any(summary["chunk_count"] == 0 for summary in document_summaries):
                    st.warning("At least one PDF has no selectable text and may need OCR before it can be searched.")
            else:
                st.markdown(
                    '<div class="atlas-empty"><div style="font-size:1.65rem">📚</div><div class="atlas-empty-title">Start with your first PDF</div><div>Use the upload area in the left sidebar to add research papers, manuals, study notes, or reports. You can select several files together.</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown("**Good first test:** upload two short text-based PDFs and ask a question that needs both documents.")

    with main_right:
        with st.container(border=True):
            st.markdown('<div class="atlas-section-heading">Research assistant</div><div class="atlas-section-caption">A small set of useful actions to get started</div>', unsafe_allow_html=True)
            for icon, title, description in [
                ("⌕", "Find evidence", "Retrieve passages that match a question."),
                ("↔", "Compare ideas", "Explore agreements and differences across papers."),
                ("▤", "Create a summary", "Gather key findings with source references."),
            ]:
                st.markdown(
                    f'<div style="display:flex;gap:.75rem;align-items:flex-start;padding:.75rem 0;border-bottom:1px solid #EEF0F7"><div style="width:35px;height:35px;display:flex;align-items:center;justify-content:center;background:#F0EEFF;color:#5547D5;border-radius:10px;font-size:1.05rem;flex-shrink:0">{icon}</div><div><div style="font-weight:750;color:#30334D;font-size:.86rem">{title}</div><div style="color:#858BA2;font-size:.77rem;line-height:1.45;margin-top:.15rem">{description}</div></div></div>',
                    unsafe_allow_html=True,
                )
            st.write("")
            st.button(
                "Open research chat →",
                type="primary",
                use_container_width=True,
                disabled=not is_indexed,
                on_click=go_to_research_chat,
            )
            if not groq_key_is_set():
                st.caption("You'll need a Groq API key for generated answers. Indexing works without it.")
            else:
                st.caption("Your Groq key is configured. Never commit it to source control.")

elif page == "Research chat":
    render_page_header(
        "Ask the documents.",
        "Ask in plain language. Answers use retrieved passages and include filename/page citations for checking.",
        is_indexed,
    )
    if is_indexed:
        stat_cols = st.columns(3)
        stat_cols[0].metric("Documents in context", len(document_summaries))
        stat_cols[1].metric("Evidence passages", retrieval_count)
        stat_cols[2].metric("Conversation turns", len(st.session_state.atlas_chat_history) // 2)
        st.write("")
    with st.container(border=True):
        render_chat(st.session_state.atlas_collection, is_indexed, retrieval_count, model_name)

elif page == "Search evidence":
    render_page_header(
        "Find the right passage.",
        "Search your indexed PDFs by meaning, then review the source passage before relying on it.",
        is_indexed,
    )
    if not is_indexed:
        st.info("Upload your PDFs and build the research index before searching.")
    else:
        with st.container(border=True):
            st.markdown('<div class="atlas-section-heading">Search across your library</div><div class="atlas-section-caption">Try a specific question or a concept, not just a single keyword.</div>', unsafe_allow_html=True)
            with st.form("atlas_evidence_search_form", clear_on_submit=False):
                search_query = st.text_input(
                    "Search query",
                    value=st.session_state.atlas_last_search,
                    placeholder="e.g., What are the main limitations of this approach?",
                )
                search_clicked = st.form_submit_button("⌕  Find matching passages", type="primary", use_container_width=True)
            if search_clicked:
                if not search_query.strip():
                    st.warning("Enter a query before searching.")
                else:
                    try:
                        with st.spinner("Searching the document index…"):
                            result_chunks = retrieve_relevant_chunks(
                                search_query.strip(), st.session_state.atlas_collection, top_k=retrieval_count
                            )
                        st.session_state.atlas_last_search = search_query.strip()
                        st.session_state.atlas_search_results = result_chunks
                    except Exception as exc:
                        st.error(f"Search failed. Please rebuild the index if the problem persists. Details: {exc}")
            results = st.session_state.atlas_search_results
            if results:
                st.markdown(f"### {len(results)} matching passage(s)")
                for rank, item in enumerate(results, start=1):
                    with st.container(border=True):
                        score = item.get("similarity")
                        badge = f"Similarity {score:.2f}" if isinstance(score, (float, int)) else "Retrieved passage"
                        st.markdown(
                            f"**{rank}. {html.escape(str(item.get('source', 'Unknown PDF')))} · page {item.get('page_number', '?')}** &nbsp; <span class='atlas-pill'>{badge}</span>",
                            unsafe_allow_html=True,
                        )
                        st.write(item.get("text", ""))
                        st.caption(f"Chunk {item.get('chunk_id', '?')} · Check this passage in the original PDF before relying on it.")
            elif st.session_state.atlas_last_search:
                st.info("No results are currently available for the saved query. Submit a search to refresh them.")

elif page == "Document library":
    render_page_header(
        "Everything, in one place.",
        "Review your uploaded PDFs, inspect extracted text, and check which files have searchable content.",
        is_indexed,
    )
    if not document_summaries:
        st.info("Your library is empty. Upload PDFs from the sidebar to begin.")
    else:
        st.markdown(f"**{len(document_summaries)} document(s)** · {word_count:,} extracted words · {chunk_count:,} chunks")
        for summary in document_summaries:
            render_document_card(summary)
        st.divider()
        st.markdown("### Inspect a page")
        selected_index = st.selectbox(
            "Choose a PDF page",
            options=list(range(len(pages))),
            format_func=lambda i: f"{pages[i]['source']} · page {pages[i]['page_number']}",
            key=f"atlas_page_select_{documents_hash[:10]}",
        ) if pages else None
        if selected_index is not None:
            selected_page = pages[selected_index]
            page_left, page_right = st.columns([.85, 1.15], gap="large")
            source_payload = next(((name, data) for name, data in document_payloads if name == selected_page["source"]), None)
            with page_left:
                if source_payload:
                    try:
                        with st.spinner("Preparing page preview…"):
                            page_png = render_pdf_page(source_payload[1], selected_page["page_number"])
                        st.image(page_png, caption=f"{selected_page['source']} · page {selected_page['page_number']}", use_container_width=True)
                    except Exception as exc:
                        st.info(f"A preview couldn't be generated for this page: {exc}")
            with page_right:
                st.markdown("**Extracted text**")
                st.text_area("Page content", value=selected_page["text"], height=320, disabled=True, label_visibility="collapsed")
                st.caption(f"{len(selected_page['text'].split()):,} words on this page")

elif page == "Research tools":
    render_page_header(
        "Go beyond a quick answer.",
        "Use retrieved evidence to draft a first-pass summary or compare the main ideas across your selected PDFs.",
        is_indexed,
    )
    tool_left, tool_right = st.columns(2, gap="large")
    with tool_left:
        with st.container(border=True):
            st.markdown("### ✧ Research summary")
            st.write("Build a structured overview around a specific topic or the main findings in your documents.")
            summary_topic = st.text_input("Summary focus", placeholder="e.g., methods, findings, limitations", key="atlas_summary_topic")
            summary_clicked = st.button("Generate summary", type="primary", use_container_width=True, disabled=not is_indexed, key="atlas_summary_btn")
    with tool_right:
        with st.container(border=True):
            st.markdown("### ⇄ Compare documents")
            st.write("Ask for points of agreement, disagreement, and unique contributions across uploaded papers.")
            compare_topic = st.text_input("Comparison focus", placeholder="e.g., approach, results, limitations", key="atlas_compare_topic")
            compare_clicked = st.button("Compare documents", use_container_width=True, disabled=not is_indexed, key="atlas_compare_btn")
    if summary_clicked or compare_clicked:
        if not groq_key_is_set():
            st.error("Add GROQ_API_KEY before generating a research result.")
        else:
            topic = (summary_topic if summary_clicked else compare_topic).strip() or "the main findings and conclusions"
            question = (
                f"Create a concise research summary about {topic}. Organize it into key themes, findings, limitations, and open questions. Cite each important claim as [filename, p. N]. Use only retrieved evidence."
                if summary_clicked else
                f"Compare the uploaded documents about {topic}. Separate agreement, disagreement, and unique contributions. Cite each important claim as [filename, p. N]. If the retrieved evidence doesn't support a comparison, say so."
            )
            try:
                with st.spinner("Finding passages and preparing your research result…"):
                    sources = retrieve_relevant_chunks(question, st.session_state.atlas_collection, top_k=min(10, max(6, retrieval_count)))
                    answer = generate_answer(question, sources, model_name=model_name)
                st.session_state.atlas_tool_result = answer
                st.session_state.atlas_tool_sources = sources
                st.session_state.atlas_tool_title = "Research summary" if summary_clicked else "Document comparison"
            except Exception as exc:
                st.error(f"The research tool could not finish. Check the API key/model and try again. Details: {exc}")
    if st.session_state.atlas_tool_result:
        st.divider()
        st.markdown(f"### {st.session_state.get('atlas_tool_title', 'Research result')}")
        st.markdown(st.session_state.atlas_tool_result)
        show_evidence(st.session_state.atlas_tool_sources, expanded=False)
        st.caption("This result reflects the passages retrieved for the prompt. It is not guaranteed to exhaustively cover every document page.")

else:  # Settings
    render_page_header(
        "Your workspace, your way.",
        "Review retrieval settings, model configuration, and deployment guidance without exposing secrets.",
        is_indexed,
    )
    settings_left, settings_right = st.columns(2, gap="large")
    with settings_left:
        with st.container(border=True):
            st.markdown("### Retrieval configuration")
            st.write(f"**Embedding model**  \n`all-MiniLM-L6-v2`")
            st.write(f"**Chunk size**  \n{chunk_size:,} words")
            st.write(f"**Chunk overlap**  \n{chunk_overlap:,} words")
            st.write(f"**Passages per question**  \n{retrieval_count}")
            st.write(f"**Answer model**  \n`{model_name or 'Not configured'}`")
            st.write("**Vector database**  \nChromaDB cosine similarity")
            st.caption("Change these controls in the sidebar. If chunk size or overlap changes, rebuild the index to apply the new settings.")
    with settings_right:
        with st.container(border=True):
            st.markdown("### Connection & privacy")
            if groq_key_is_set():
                st.success("Groq API key detected. Its value is never shown here.")
            else:
                st.warning("Groq API key not found. Indexing and retrieval can work, but answer generation requires the key.")
            st.code('GROQ_API_KEY = "your-real-key"', language="toml")
            st.caption("For Streamlit Cloud, add this in Manage app → Settings → Secrets. For local use, set it as an environment variable.")
            st.markdown("**Privacy reminder**")
            st.write("Uploaded PDFs and extracted text are processed by the application server. Do not upload sensitive documents to an app you don't control.")
            st.markdown("**Index status**")
            st.write("Ready" if is_indexed else "Not built for the current files/settings")

st.markdown('<div class="atlas-footer">ATLAS RESEARCH &nbsp;·&nbsp; MULTI-PDF RAG &nbsp;·&nbsp; PYTHON / STREAMLIT / CHROMADB / GROQ</div>', unsafe_allow_html=True)
