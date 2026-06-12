import gradio as gr
import requests
import os

API_URL = "http://127.0.0.1:8000"

def query_rag(question, history):
    chat_history = []
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            chat_history.append({"role": "user", "content": turn[0]})
            chat_history.append({"role": "assistant", "content": turn[1]})
        elif isinstance(turn, dict):
            chat_history.append({"role": turn.get("role"), "content": turn.get("content")})
        elif hasattr(turn, "role") and hasattr(turn, "content"):
            chat_history.append({"role": turn.role, "content": turn.content})
            
    try:
        payload = {
            "question": question,
            "chat_history": chat_history
        }
        response = requests.post(f"{API_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer generated.")
            query_type = data.get("query_type", "other")
            retries = data.get("retries", 0)
            
            # Format clean output
            info = f"\n\n*(Category: {query_type} | Retries: {retries})*"
            return answer + info
        else:
            return f"Error: HTTP {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the FastAPI server. Make sure it is running on http://127.0.0.1:8000"

def get_indexed_docs():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs = response.json().get("documents", [])
            if not docs:
                return "No documents indexed yet."
            return "\n".join([f"- {d}" for d in docs])
        return "Error loading documents."
    except Exception:
        return "Could not connect to FastAPI server."

def ingest_url(url):
    if not url:
        return "Please enter a valid URL."
    try:
        response = requests.post(f"{API_URL}/ingest", data={"url": url})
        if response.status_code == 200:
            return f"Ingestion started: {response.json().get('status')}"
        return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to server: {e}"

def ingest_file(file):
    if file is None:
        return "Please select a file to upload."
    try:
        filename = os.path.basename(file.name)
        with open(file.name, "rb") as f:
            files = {"file": (filename, f, "text/plain")}
            response = requests.post(f"{API_URL}/ingest", files=files)
        if response.status_code == 200:
            return f"Ingestion started: {response.json().get('status')}"
        return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to server: {e}"

# Modern Dark Purple/Indigo Theme
custom_theme = gr.themes.Default(
    primary_hue="purple",
    secondary_hue="indigo",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    button_primary_background_fill="linear-gradient(90deg, #8B5CF6, #6366F1)",
    button_primary_background_fill_hover="linear-gradient(90deg, #A78BFA, #818CF8)",
    button_primary_text_color="white",
    input_background_fill="*neutral_950",
    input_border_color="*neutral_800"
)

with gr.Blocks(theme=custom_theme, css="footer {visibility: hidden}") as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1E1B4B, #311042); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: #F3E8FF; margin: 0; font-size: 2.2rem; font-weight: 800;">Self-Corrective RAG Assistant</h1>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=query_rag,
                textbox=gr.Textbox(placeholder="Ask anything about the documentation...", container=False, scale=7),
            )
            
        with gr.Column(scale=1):
            with gr.Accordion("📚 Indexed Corpus", open=True):
                doc_list = gr.Markdown(value=get_indexed_docs())
                refresh_btn = gr.Button("🔄 Refresh Document List", size="sm")
                refresh_btn.click(fn=get_indexed_docs, outputs=doc_list)
                
            with gr.Accordion("📥 Ingest New Context", open=True):
                with gr.Tab("Ingest URL"):
                    url_input = gr.Textbox(placeholder="https://example.com/doc.md", label="URL to Markdown/HTML")
                    url_btn = gr.Button("Ingest URL", variant="primary")
                    url_status = gr.Markdown()
                    url_btn.click(fn=ingest_url, inputs=url_input, outputs=url_status)
                    
                with gr.Tab("Upload File"):
                    file_input = gr.File(label="Upload Markdown/Text File", file_types=[".md", ".txt"])
                    file_btn = gr.Button("Ingest File", variant="primary")
                    file_status = gr.Markdown()
                    file_btn.click(fn=ingest_file, inputs=file_input, outputs=file_status)

    gr.HTML("""
        <div style="text-align: center; padding: 16px; margin-top: 20px; border-top: 1px solid #3F3F46;">
            <p style="color: #A1A1AA; font-size: 0.9rem; margin: 0;">
                Made by <span style="color: #C084FC; font-weight: 600;">Mohd Talha</span>
            </p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch()
