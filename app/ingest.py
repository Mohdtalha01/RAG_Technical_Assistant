import os
import urllib.request
import ssl
from glob import glob
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .vectorstore import get_vectorstore

# Bypass SSL certificate verification on macOS Python installs
ssl._create_default_https_context = ssl._create_unverified_context

def ingest_text(text: str, source_name: str):
    """Ingest raw text into the vector store after chunking."""
    doc = Document(page_content=text, metadata={"source": source_name})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    splits = text_splitter.split_documents([doc])
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=splits)
    print(f"Ingested {len(splits)} chunks from source '{source_name}' and saved to vector store.")
    return len(splits)


def ingest_from_url(url: str):
    """Fetch content from a URL, clean it, and ingest it into the vector store."""
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get_content_type()
            raw_data = response.read()

        if url.endswith(".md") or url.endswith(".txt") or "text/plain" in content_type:
            clean_text = raw_data.decode("utf-8", errors="ignore")
        else:
            soup = BeautifulSoup(raw_data, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = "\n".join(chunk for chunk in chunks if chunk)

        source_name = url.split("/")[-1] or "url_source"
        if not source_name.endswith((".md", ".html", ".txt")):
            source_name = f"{source_name}.html"
            
        return ingest_text(clean_text, source_name)
    except Exception as e:
        print(f"Error ingesting from URL {url}: {e}")
        raise e

def ingest_from_directory(docs_dir: str):
    """Load, split, and ingest markdown files into the vector store."""
    docs = []
    files = glob(os.path.join(docs_dir, "*.md"))
    
    if not files:
        print(f"No markdown files found in {docs_dir}")
        return

    for file_path in files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(file_path)
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not docs:
        print("No documents loaded.")
        return

    print(f"Loaded {len(docs)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=splits)
    print("Ingestion complete. Vector store updated.")

if __name__ == "__main__":
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    ingest_from_directory(docs_dir)


