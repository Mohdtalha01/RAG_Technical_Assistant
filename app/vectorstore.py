import os
import json
import math
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store.json")

def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(a * a for a in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

class SimplePersistentVectorStore:
    """A pure Python persistent vector store using JSON and Gemini embeddings."""
    def __init__(self, persist_path: str, embeddings):
        self.persist_path = persist_path
        self.embeddings = embeddings
        self.data = []  # List of dicts: {"text": str, "metadata": dict, "embedding": list}
        self.load()

    def load(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                print(f"Loaded {len(self.data)} vectors from persistent store {self.persist_path}")
            except Exception as e:
                print(f"Error loading vector store: {e}. Starting fresh.")
                self.data = []

    def save(self):
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"Persisted {len(self.data)} vectors to {self.persist_path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def add_documents(self, documents):
        if not documents:
            return
        
        # Remove placeholders if present
        self.data = [item for item in self.data if item.get("metadata", {}).get("source") != "system"]
        
        import time
        batch_size = 20
        for idx in range(0, len(documents), batch_size):
            batch_docs = documents[idx : idx + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            print(f"Embedding batch {idx // batch_size + 1} of {(len(documents) - 1) // batch_size + 1}...")
            
            vectors = self.embeddings.embed_documents(texts)
            
            for doc, vector in zip(batch_docs, vectors):
                self.data.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": vector
                })
            
            # Save progress incrementally
            self.save()
            
            if idx + batch_size < len(documents):
                print("Sleeping 6 seconds to respect rate limits...")
                time.sleep(6)

    def similarity_search(self, query: str, k: int = 4):
        if not self.data:
            return []
            
        # Get query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Compute similarities
        scored_docs = []
        for item in self.data:
            sim = cosine_similarity(query_vector, item["embedding"])
            scored_docs.append((sim, item))
            
        # Sort by similarity score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k as Document objects
        results = []
        for score, item in scored_docs[:k]:
            results.append(Document(
                page_content=item["text"],
                metadata=item["metadata"]
            ))
        return results

def get_embeddings():
    """Return Google Generative AI embeddings model for free cloud vectorization."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)

def get_vectorstore():
    """Return SimplePersistentVectorStore instance."""
    return SimplePersistentVectorStore(persist_path=DB_PATH, embeddings=get_embeddings())




