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
        payload = {"question": question, "chat_history": chat_history}
        response = requests.post(f"{API_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer generated.")
            query_type = data.get("query_type", "other")
            retries = data.get("retries", 0)
            info = f"\n\n<span style='font-size:11px; color:#9CA3AF; background:#F3F4F6; padding:2px 8px; border-radius:20px; margin-right:4px;'>🏷 {query_type}</span><span style='font-size:11px; color:#9CA3AF; background:#F3F4F6; padding:2px 8px; border-radius:20px;'>↩ {retries} retries</span>"
            return answer + info
        else:
            return f"⚠️ Error: HTTP {response.status_code} — {response.text}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Could not connect to the FastAPI server. Make sure it is running on http://127.0.0.1:8000"


def get_indexed_docs():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs = response.json().get("documents", [])
            if not docs:
                return "*No documents indexed yet.*"
            return "\n".join([f"📄 `{d}`" for d in docs])
        return "⚠️ Error loading documents."
    except Exception:
        return "⚠️ Could not connect to FastAPI server."


def ingest_url(url):
    if not url:
        return "⚠️ Please enter a valid URL."
    try:
        response = requests.post(f"{API_URL}/ingest", data={"url": url})
        if response.status_code == 200:
            return f"✅ Ingestion started: {response.json().get('status')}"
        return f"⚠️ Error: {response.text}"
    except Exception as e:
        return f"⚠️ Error connecting to server: {e}"


def ingest_file(file):
    if file is None:
        return "⚠️ Please select a file to upload."
    try:
        filename = os.path.basename(file.name)
        with open(file.name, "rb") as f:
            files = {"file": (filename, f, "text/plain")}
            response = requests.post(f"{API_URL}/ingest", files=files)
        if response.status_code == 200:
            return f"✅ Ingestion started: {response.json().get('status')}"
        return f"⚠️ Error: {response.text}"
    except Exception as e:
        return f"⚠️ Error connecting to server: {e}"


# ── Clean White + Violet Theme ──────────────────────────────────────────────
custom_theme = gr.themes.Default(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    # Backgrounds
    body_background_fill="#FFFFFF",
    block_background_fill="#FFFFFF",
    block_border_color="#E5E7EB",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",

    # Panels / containers
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F9FAFB",

    # Inputs
    input_background_fill="#F9FAFB",
    input_border_color="#E5E7EB",
    input_border_width="1px",
    input_radius="8px",
    input_shadow="none",
    input_shadow_focus="0 0 0 2px #7C3AED22",
    input_border_color_focus="#7C3AED",

    # Buttons
    button_primary_background_fill="#7C3AED",
    button_primary_background_fill_hover="#6D28D9",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#7C3AED",
    button_primary_border_color_hover="#6D28D9",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F3EDFE",
    button_secondary_text_color="#374151",
    button_secondary_border_color="#E5E7EB",
    button_secondary_border_color_hover="#7C3AED",
    button_border_width="1px",
    button_small_radius="8px",
    button_large_radius="8px",

    # Text
    body_text_color="#111827",
    body_text_size="14px",
    block_title_text_size="13px",
    block_title_text_weight="500",
    block_label_text_color="#6B7280",
    block_label_text_size="12px",
    chatbot_text_size="14px",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
CSS = """
/* Import Inter from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');

/* Reset & base */
* { box-sizing: border-box; }
body, .gradio-container { background: #FFFFFF !important; font-family: 'Inter', sans-serif !important; }

/* Hide footer */
footer { visibility: hidden !important; }

/* Top-level container */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Header banner */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid #E5E7EB;
    background: #FFFFFF;
    margin-bottom: 0;
}
.app-header-left { display: flex; align-items: center; gap: 10px; }
.header-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #7C3AED;
    flex-shrink: 0;
}
.app-title { font-size: 15px; font-weight: 600; color: #111827; letter-spacing: -0.01em; margin: 0; }
.app-subtitle { font-size: 12px; color: #9CA3AF; margin: 0; }
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11.5px;
    color: #6B7280;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 4px 12px;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22C55E;
}

/* Layout grid */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 0;
    min-height: calc(100vh - 70px);
}

/* Chat area */
.chat-area { border-right: 1px solid #E5E7EB; }

/* Chatbot styling */
.chatbot {
    border: none !important;
    border-radius: 0 !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
}
.chatbot .message-wrap { padding: 16px 20px !important; gap: 14px !important; }

/* User bubble */
.chatbot .message.user {
    background: #7C3AED !important;
    color: #FFFFFF !important;
    border-radius: 12px 12px 2px 12px !important;
    padding: 10px 14px !important;
    font-size: 13.5px !important;
    box-shadow: none !important;
    border: none !important;
}

/* Bot bubble */
.chatbot .message.bot {
    background: #F9FAFB !important;
    color: #111827 !important;
    border-radius: 12px 12px 12px 2px !important;
    padding: 10px 14px !important;
    font-size: 13.5px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: none !important;
}

/* Chat input row */
.chat-input-row {
    border-top: 1px solid #E5E7EB;
    padding: 14px 20px;
    background: #FFFFFF;
}
.chat-input-row textarea {
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
    background: #F9FAFB !important;
    font-size: 13.5px !important;
    padding: 10px 14px !important;
    color: #111827 !important;
    resize: none !important;
    transition: border-color 0.15s !important;
}
.chat-input-row textarea:focus {
    border-color: #7C3AED !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px #7C3AED18 !important;
}

/* Sidebar */
.sidebar {
    background: #FAFAFA;
    border-left: 1px solid #E5E7EB;
}
.sidebar .block {
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Accordion */
.sidebar .gr-accordion {
    border: none !important;
    border-bottom: 1px solid #E5E7EB !important;
    border-radius: 0 !important;
    background: transparent !important;
}
.sidebar .gr-accordion > button {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #9CA3AF !important;
    padding: 14px 16px !important;
    background: transparent !important;
    border: none !important;
}

/* Document list */
.doc-list {
    padding: 4px 16px 14px !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
    color: #374151 !important;
}

/* Refresh button */
.refresh-btn button {
    width: 100% !important;
    font-size: 12px !important;
    color: #6B7280 !important;
    background: transparent !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 6px !important;
    transition: all 0.15s !important;
}
.refresh-btn button:hover {
    color: #7C3AED !important;
    border-color: #7C3AED !important;
    background: #F3EDFE !important;
}

/* Ingest inputs */
.ingest-input input, .ingest-input textarea {
    font-size: 12.5px !important;
    border: 1px solid #E5E7EB !important;
    background: #FFFFFF !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    color: #111827 !important;
}
.ingest-input input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px #7C3AED15 !important;
}

/* Ingest button */
.ingest-btn button {
    font-size: 12.5px !important;
    background: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
}
.ingest-btn button:hover {
    border-color: #7C3AED !important;
    color: #7C3AED !important;
    background: #F3EDFE !important;
}

/* File upload zone */
.file-upload .wrap {
    border: 1px dashed #D1D5DB !important;
    border-radius: 8px !important;
    background: #FAFAFA !important;
    transition: all 0.15s !important;
}
.file-upload .wrap:hover {
    border-color: #7C3AED !important;
    background: #F3EDFE !important;
}

/* Status markdown */
.status-msg { font-size: 12px !important; color: #6B7280 !important; margin-top: 6px !important; }

/* Tab styling */
.gr-tab-nav button {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 12px !important;
    background: transparent !important;
}
.gr-tab-nav button.selected {
    color: #7C3AED !important;
    border-bottom-color: #7C3AED !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #D1D5DB; }
"""

# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="RAG Assistant") as demo:

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-header-left">
            <div class="header-dot"></div>
            <div>
                <p class="app-title">RAG Assistant</p>
                <p class="app-subtitle">LangGraph · FastAPI · ChromaDB</p>
            </div>
        </div>
        <div class="status-indicator">
            <div class="status-dot"></div>
            Connected
        </div>
    </div>
    """)

    with gr.Row(elem_classes="main-grid"):

        # ── Left: Chat ──────────────────────────────────────────────────────
        with gr.Column(scale=3, elem_classes="chat-area"):
            gr.ChatInterface(
                fn=query_rag,
                chatbot=gr.Chatbot(
                    height=480,
                    show_label=False,
                    elem_classes="chatbot",
                    render_markdown=True,
                ),
                textbox=gr.Textbox(
                    placeholder="Ask anything about your indexed documents…",
                    container=False,
                    scale=7,
                    lines=1,
                    max_lines=4,
                    elem_classes="chat-input-row",
                    submit_btn="Send",
                ),
            )

        # ── Right: Sidebar ──────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="sidebar"):

            with gr.Accordion("📄  Indexed Corpus", open=True):
                doc_list = gr.Markdown(
                    value=get_indexed_docs(),
                    elem_classes="doc-list"
                )
                refresh_btn = gr.Button(
                    "↺  Refresh list",
                    size="sm",
                    variant="secondary",
                    elem_classes="refresh-btn"
                )
                refresh_btn.click(fn=get_indexed_docs, outputs=doc_list)

            with gr.Accordion("＋  Ingest Context", open=True):

                with gr.Tab("URL"):
                    url_input = gr.Textbox(
                        placeholder="https://example.com/docs.md",
                        label="",
                        show_label=False,
                        elem_classes="ingest-input"
                    )
                    url_btn = gr.Button(
                        "Ingest URL",
                        variant="secondary",
                        size="sm",
                        elem_classes="ingest-btn"
                    )
                    url_status = gr.Markdown(elem_classes="status-msg")
                    url_btn.click(fn=ingest_url, inputs=url_input, outputs=url_status)

                with gr.Tab("File"):
                    file_input = gr.File(
                        label="",
                        show_label=False,
                        file_types=[".md", ".txt"],
                        elem_classes="file-upload"
                    )
                    file_btn = gr.Button(
                        "Upload & Ingest",
                        variant="secondary",
                        size="sm",
                        elem_classes="ingest-btn"
                    )
                    file_status = gr.Markdown(elem_classes="status-msg")
                    file_btn.click(fn=ingest_file, inputs=file_input, outputs=file_status)


if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=CSS)