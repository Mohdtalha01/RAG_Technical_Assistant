import os
import json
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List, Optional
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

from .graph import app_graph
from .ingest import ingest_from_directory, ingest_text, ingest_from_url

app = FastAPI(title="RAG-Based Technical Documentation Assistant", description="FastAPI app handling RAG logic via LangGraph")

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str  # e.g., 'thumbs_up' or 'thumbs_down'
    comment: Optional[str] = None

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feedbacks.json")

def load_all_feedbacks() -> list:
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_feedback(feedback_data: dict) -> int:
    feedbacks = load_all_feedbacks()
    feedbacks.append(feedback_data)
    try:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedbacks, f, indent=4)
    except Exception as e:
        print(f"Error saving feedback: {e}")
    return len(feedbacks)

@app.post("/query")
async def query_assistant(request: QueryRequest):
    try:
        inputs = {
            "question": request.question,
            "chat_history": request.chat_history or []
        }
        # Invoke the state graph
        result = app_graph.invoke(inputs)
        return {
            "answer": result.get("generation"),
            "query_type": result.get("query_type"),
            "retries": result.get("retries", 0),
            "sources": [{"content": d.page_content, "source": d.metadata.get("source")} for d in result.get("documents", [])]
        }
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_docs(
    background_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    # Case 1: URL ingestion
    if url:
        background_tasks.add_task(ingest_from_url, url)
        return {"status": "ingestion of URL started in background", "source": url}
        
    # Case 2: File upload ingestion
    if file:
        try:
            contents = await file.read()
            text_content = contents.decode("utf-8", errors="ignore")
            
            # Save file to docs folder
            docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
            os.makedirs(docs_dir, exist_ok=True)
            save_path = os.path.join(docs_dir, file.filename)
            with open(save_path, "wb") as f:
                f.write(contents)
                
            background_tasks.add_task(ingest_text, text_content, file.filename)
            return {"status": "ingestion of uploaded file started in background", "filename": file.filename}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
            
    # Case 3: Default directory ingestion
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=400, detail="Docs directory does not exist and no URL or file was provided")
    background_tasks.add_task(ingest_from_directory, docs_dir)
    return {"status": "ingestion of local directory started in background"}

@app.get("/documents")
async def get_documents():
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    local_files = []
    if os.path.exists(docs_dir):
        local_files = [f for f in os.listdir(docs_dir) if f.endswith((".md", ".txt", ".html"))]
        
    db_sources = set()
    try:
        from .vectorstore import get_vectorstore
        vectorstore = get_vectorstore()
        for doc in vectorstore.docstore._dict.values():
            if doc.metadata and "source" in doc.metadata:
                # Exclude the default placeholder document source
                if doc.metadata["source"] != "system":
                    db_sources.add(doc.metadata["source"])
    except Exception as e:
        print(f"Error listing documents from vector store: {e}")

        
    all_sources = sorted(list(set(local_files).union(db_sources)))
    return {"documents": all_sources}

@app.post("/feedback")
async def post_feedback(feedback: FeedbackRequest):
    count = save_feedback(feedback.model_dump())
    return {"status": "feedback recorded successfully", "count": count}

