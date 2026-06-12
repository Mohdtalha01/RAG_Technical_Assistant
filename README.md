# Self-Corrective RAG Technical Documentation Assistant

A robust, production-grade Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **Gradio**. It features a self-corrective workflow that evaluates retrieved documents, automatically rewrites queries, falls back to web searches if needed, and validates answers against hallucination checks before outputting responses.

The assistant is powered by local/cloud vector storage (**ChromaDB**) and uses free cloud-tier APIs (**Google Gemini** or **Groq**) for LLM reasoning and embeddings.

---

## 🛠️ System Architecture

The core of the application is a **LangGraph StateGraph** that orchestrates a self-corrective retrieval loop:

```
[User Question]
       │
       ▼
┌──────────────┐
│ analyze_query│ ◄──────────────────────────────┐
└──────┬───────┘                                │ (Hallucination
       │                                        │  detected &
       ▼                                        │  retries < 2)
┌──────────────┐                                │
│   retrieve   │ ◄──────────┐                   │
└──────┬───────┘            │                   │
       │                    │ (Irrelevant &     │
       ▼                    │  retries < 2)     │
┌──────────────┐            │                   │
│grade_document│            │                   │
└──────┬───────┘            │                   │
       │                    │                   │
       ├─► [All Irrelevant?]──►[rewrite_query]──┘
       │        │ (retries >= 2)
       │        ▼
       │  [web_search]
       │        │
       ▼        ▼
┌──────────────────┐
│     generate     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│check_hallucinate ├─► [Grounded?] ──► [END]
└──────────────────┘
```

### LangGraph Workflow Nodes:
1. **`analyze_query`**: Analyzes the query to classify the intent (`conceptual`, `how-to`, `troubleshooting`, `api_reference`, etc.). If conversation memory exists, it reformulates follow-up questions into standalone queries using the chat history.
2. **`retrieve`**: Searches the persistent local **ChromaDB** store using the expanded query.
3. **`grade_documents`**: A binary classification step where the LLM grades retrieved chunks as `"relevant"` or `"irrelevant"`. Irrelevant chunks are filtered out.
4. **`rewrite_query` (Self-Correction)**: If all chunks are irrelevant (or hallucination is detected) and retry count is < 2, the query is rewritten for optimized retrieval and looped back.
5. **`web_search` (Fallback)**: If no documents are relevant after 2 retries, it runs a free DuckDuckGo search fallback to get real-time context.
6. **`generate`**: Generates a concise answer based on context, with source file/URL citations.
7. **`check_hallucination`**: Assesses if the generated answer is grounded in the retrieved context. If not, it triggers a rewrite/re-retrieval loop (up to 2 retries).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Google Gemini API key (free on [Google AI Studio](https://aistudio.google.com/)) or a Groq API Key.

### 1. Setup Virtual Environment & Install Dependencies
Navigate to the project directory and run:
```bash
# Create virtual environment if you don't have one
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the `rag_assistant` root directory:
```env
# Google Gemini Key (Recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# OR Groq Key
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Fetch Documentation Corpus
Run the script to download the default FastAPI documentation pages:
```bash
python fetch_docs.py
```
This downloads markdown files to the `docs/` folder.

### 4. Index Documents
Run the ingestion script to split and index the documentation into the persistent Chroma DB:
```bash
python -m app.ingest
```

---

## 🏃 Running the Application

For the full experience, run both the FastAPI API backend and the Gradio Chat UI.

### Start the API Server
Start the FastAPI server on `http://127.0.0.1:8000`:
```bash
uvicorn app.main:app --reload
```

### Start the Gradio Chat UI
In a separate terminal (with virtual environment activated), run:
```bash
python app/ui.py
```
Open the local URL displayed (typically `http://127.0.0.1:7860`) in your browser to chat with the assistant, upload files, ingest URLs, and see the indexed documentation list!

---

## 📡 API Reference

You can view the interactive Swagger docs at `http://127.0.0.1:8000/docs` when the API is running.

### 1. POST `/query`
Execute a query through the RAG assistant.

**Request Body:**
```json
{
  "question": "How do I define path parameters?",
  "chat_history": [
    {"role": "user", "content": "What library is this for?"},
    {"role": "assistant", "content": "This documentation is for FastAPI."}
  ]
}
```

**Response:**
```json
{
  "answer": "You can declare path parameters in FastAPI using the same syntax as Python format strings: `/items/{item_id}`. The value of `item_id` will be passed to your function as the argument. You can also define path parameters with types.",
  "query_type": "how-to",
  "retries": 0,
  "sources": [
    {
      "content": "### Path Parameters\n\nYou can declare path \"parameters\" or \"variables\" with the same syntax used by Python-format strings...",
      "source": "tutorial_path-params.md"
    }
  ]
}
```

### 2. POST `/ingest`
Ingest files or URLs.

- **Option A: Ingest URL (Form Data)**
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest -F "url=https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body.md"
  ```
- **Option B: Upload File (Form Data)**
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest -F "file=@/path/to/my_doc.txt"
  ```
- **Option C: Re-Ingest Local docs/ Folder**
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest
  ```

### 3. GET `/documents`
List all indexed documents/sources currently loaded in Chroma DB and the local `docs/` folder.
```bash
curl http://127.0.0.1:8000/documents
```

### 4. POST `/feedback`
Submit thumbs up/down feedback on responses (persisted in local `feedbacks.json`).
```json
{
  "question": "How do I define path parameters?",
  "answer": "You can declare path parameters...",
  "feedback": "thumbs_up",
  "comment": "Perfect answer!"
}
```

---

## 🎨 Design Decisions & Tradeoffs

1. **Persistent ChromaDB Vector Store**: Instead of keeping embeddings in-memory (which gets lost on server restart), the application persists indexed documents to a `./chroma_db` directory.
2. **Google Generative AI Embeddings**: Replaced `sentence-transformers` with cloud-based `text-embedding-004` to avoid a massive ~1GB download of PyTorch and compile issues on macOS Python 3.13. It remains completely free within Gemini's free tier.
3. **Structured Grader Resiliency**: LLM grading nodes (`grade_documents` and `check_hallucination`) use structured outputs via Pydantic model schemas. However, they include catch-all `try-except` blocks that fall back to text completion parsers and safe defaults in case of API schema changes, rate limits, or transient connection errors.
4. **Standalone Question Reformulator**: Incorporates Gradio chat history by running a pre-retrieval LLM pass to resolve references and convert follow-up prompts into standalone queries before vector search.
5. **Gradio Multi-functional Layout**: Created a custom `gr.Blocks` dashboard UI containing a chat window alongside a dynamic sidebar to index URLs and upload files directly, providing a much cleaner UX.
