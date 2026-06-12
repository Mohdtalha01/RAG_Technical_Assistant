import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(temperature: float = 0) -> BaseChatModel:
    """Return the LLM instance based on available API keys (Groq or Gemini)."""
    if os.environ.get("GROQ_API_KEY"):
        # Llama-3.1 is fast and free on Groq
        return ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature)
    elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, google_api_key=api_key)
    else:
        # Default initialization, will fail at runtime if key isn't provided
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
