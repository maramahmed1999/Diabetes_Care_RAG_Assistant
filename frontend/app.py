"""
Streamlit chat UI for the Diabetes RAG system.

Talks to the FastAPI backend over HTTP — run the backend first:
    uvicorn backend.main:app --reload --port 8000
then:
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Diabetes Care Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# Styling
# ----------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 2rem;
            max-width: 780px;
        }

        .app-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }

        .app-header .icon {
            font-size: 2.1rem;
        }

        .app-header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0;
            color: #0f172a;
        }

        .app-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }

        .stChatMessage {
            border-radius: 14px;
            padding: 0.4rem 0.2rem;
        }

        [data-testid="stChatMessageContent"] {
            font-size: 0.97rem;
            line-height: 1.55;
        }

        .source-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #0ea5e9;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.6rem;
        }

        .source-card .source-title {
            font-weight: 600;
            font-size: 0.88rem;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }

        .source-card .source-meta {
            font-size: 0.78rem;
            color: #64748b;
            margin-bottom: 0.4rem;
        }

        .source-card .source-text {
            font-size: 0.85rem;
            color: #334155;
            line-height: 1.5;
        }

        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .status-ok {
            background: #dcfce7;
            color: #166534;
        }

        .status-down {
            background: #fee2e2;
            color: #991b1b;
        }

        .stChatInput textarea {
            border-radius: 12px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------
# Backend helpers
# ----------------------------------------------------------------
def check_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask_question(question: str, top_k: int) -> dict:
    response = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    top_k = st.slider("Sources to retrieve", min_value=1, max_value=8, value=3)

    st.markdown("---")
    st.markdown("### 📡 Backend status")

    health = check_health()
    if health and health.get("index_ready"):
        st.markdown('<span class="status-pill status-ok">Connected & indexed</span>', unsafe_allow_html=True)
    elif health:
        st.markdown('<span class="status-pill status-down">Index not built yet</span>', unsafe_allow_html=True)
        st.caption("Run the ingestion + indexing scripts, then restart the API.")
    else:
        st.markdown('<span class="status-pill status-down">API unreachable</span>', unsafe_allow_html=True)
        st.caption(f"Couldn't reach {API_URL}. Is the FastAPI server running?")

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "This assistant answers strictly from ADA, NIDDK and WHO diabetes "
        "reference documents. It is not a substitute for medical advice."
    )

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="icon">🩺</div>
        <h1>Diabetes Care Assistant</h1>
    </div>
    <div class="app-subtitle">
        Ask a question and get an answer grounded in ADA, NIDDK and WHO diabetes guidelines.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🩺"):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 {len(message['sources'])} source(s) used"):
                for src in message["sources"]:
                    meta_bits = [
                        bit
                        for bit in [
                            src.get("organization"),
                            f"p.{src['page']}" if src.get("page") else None,
                        ]
                        if bit
                    ]
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-title">{src.get('document_title') or 'Source'}</div>
                            <div class="source-meta">{' · '.join(meta_bits)} · relevance {src['score']:.2f}</div>
                            <div class="source-text">{src['text'][:400]}{'…' if len(src['text']) > 400 else ''}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ----------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------
prompt = st.chat_input("Ask about diabetes care, nutrition, glycemic goals…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        if not health:
            st.error(f"Can't reach the API at {API_URL}. Please start the FastAPI backend first.")
        else:
            with st.spinner("Searching the guidelines and drafting an answer…"):
                try:
                    result = ask_question(prompt, top_k)
                    answer = result["answer"]
                    sources = result["sources"]
                except requests.RequestException as exc:
                    answer = f"Something went wrong talking to the API: {exc}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 {len(sources)} source(s) used"):
                    for src in sources:
                        meta_bits = [
                            bit
                            for bit in [
                                src.get("organization"),
                                f"p.{src['page']}" if src.get("page") else None,
                            ]
                            if bit
                        ]
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-title">{src.get('document_title') or 'Source'}</div>
                                <div class="source-meta">{' · '.join(meta_bits)} · relevance {src['score']:.2f}</div>
                                <div class="source-text">{src['text'][:400]}{'…' if len(src['text']) > 400 else ''}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
