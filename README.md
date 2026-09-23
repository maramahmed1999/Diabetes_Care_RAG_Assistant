# 🩺 Diabetes Care RAG Assistant

A retrieval-augmented question-answering system over diabetes care
guidelines (ADA, NIDDK, WHO). This project is a restructured version
of an original Colab notebook, split into a clean pipeline plus a
**FastAPI** backend and a **Streamlit** chat UI. No pipeline logic was
changed — extraction, cleaning, chunking, embedding and retrieval
behave exactly as in the notebook, just organized into separate,
reusable modules.

## How it works

```
PDFs  →  extract  →  clean  →  chunk  →  embed (dense + BM25)  →  Qdrant
                                                                     │
                                                       Streamlit ⇄ FastAPI
                                                        (chat UI)   (hybrid search + Ollama LLM)
```

1. **Extraction** — `src/ingestion/extract.py` reads each PDF with PyMuPDF, detects
   columns and reconstructs correct reading order.
2. **Cleaning** — `src/ingestion/clean.py` strips page numbers, fixes hyphenation,
   removes repeated headers/footers, and rebuilds readable paragraphs.
3. **Chunking** — `src/ingestion/chunk.py` splits cleaned text into ~1200-character,
   paragraph-aware chunks with 200-character overlap.
4. **Embedding & indexing** — `src/embedding/index.py` embeds each chunk with a dense
   model (`BAAI/bge-base-en-v1.5`) and a sparse BM25 model (`Qdrant/bm25`),
   then upserts both into a local Qdrant collection.
5. **Retrieval + generation** — `src/rag/retriever.py` runs a hybrid (dense + BM25,
   RRF-fused) search; `src/rag/generator.py` builds a grounded prompt and calls a
   local Ollama model (`phi3` by default) to answer strictly from the retrieved
   context.

## Project structure

```
rag_diabetes_app/
├── data/
│   ├── raw/                 # put your source PDFs here
│   └── processed/           # generated chunks (diabetes_chunks.jsonl)
├── src/
│   ├── config.py            # all paths, model names, constants
│   ├── ingestion/
│   │   ├── extract.py       # phase 1: PDF -> ordered text
│   │   ├── clean.py         # phase 2: cleaning + metadata
│   │   ├── chunk.py         # phase 3: chunking
│   │   └── pipeline.py      # orchestrates phases 1-3, writes JSONL
│   ├── embedding/
│   │   └── index.py         # phase 4: dense + BM25 embedding, Qdrant indexing
│   └── rag/
│       ├── retriever.py     # hybrid search
│       └── generator.py     # prompt building + Ollama call
├── backend/
│   ├── main.py               # FastAPI app (/health, /query)
│   └── schemas.py            # request/response models
├── frontend/
│   └── app.py                 # Streamlit chat UI
├── scripts/
│   ├── run_ingestion.py       # CLI: PDFs -> chunks.jsonl
│   └── build_index.py         # CLI: chunks.jsonl -> Qdrant index
├── requirements.txt
└── .env.example
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install & run Ollama (local LLM)

The generation step uses a local Ollama model, matching the original
notebook.

```bash
# install: see https://ollama.com/download
ollama serve                     # start the Ollama server
ollama pull phi3                 # or any model you configure in src/config.py
```

### 3. Add your source PDFs

Place the PDFs in `data/raw/` using these exact filenames (or edit
`PDF_FILES` in `src/config.py` to match your own):

```
data/raw/ada_section5.pdf
data/raw/ada_section6.pdf
data/raw/niddk_eating.pdf
data/raw/who_diabetes.pdf
```

### 4. Run the ingestion pipeline

```bash
python scripts/run_ingestion.py   # extract -> clean -> chunk -> data/processed/diabetes_chunks.jsonl
python scripts/build_index.py     # embed chunks -> local Qdrant index at data/qdrant_db/
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Check it's alive: `http://localhost:8000/health`

### 6. Start the frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and
start asking questions.

## Configuration

All tunables live in `src/config.py`:

| Setting | Purpose |
|---|---|
| `PDF_FILES` / `DOCUMENT_METADATA` | which PDFs to ingest and their citation metadata |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | chunking parameters |
| `DENSE_MODEL_NAME` / `SPARSE_MODEL_NAME` | embedding models |
| `OLLAMA_MODEL` | which local LLM to use for generation |
| `TOP_K` | default number of chunks retrieved per question |

The Streamlit app reads the backend URL from the `RAG_API_URL`
environment variable (see `.env.example`); it defaults to
`http://localhost:8000`.

## API reference

**`GET /health`**
```json
{ "status": "ok", "collection": "diabetes_collection", "index_ready": true }
```

**`POST /query`**
```json
// request
{ "question": "What are the glycemic goals for adults?", "top_k": 3 }

// response
{
  "answer": "…grounded answer citing [1], [2]…",
  "sources": [
    { "text": "...", "score": 0.83, "document_title": "...", "organization": "...", "page": 12, "chunk_id": "ada_section6_p12_c000" }
  ]
}
```

## Notes

- The Qdrant index is stored locally (embedded mode) at `data/qdrant_db/`.
  Because embedded Qdrant locks its storage directory, the retriever
  automatically works off a disposable copy of the index at query time,
  the same workaround used in the original notebook.
- Answers are strictly grounded in the retrieved documents — the
  system prompt instructs the LLM to say so explicitly when the
  answer isn't in the provided context.
- This assistant is for informational purposes only and is not a
  substitute for professional medical advice.
