"""Model providers, chosen by environment: OpenAI by default.

EMBEDDINGS=google uses Gemini embeddings (GOOGLE_API_KEY). CHAT_MODEL picks the chat model; with
OPENAI_BASE_URL any OpenAI-compatible endpoint works.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def embeddings():
    if os.environ.get("EMBEDDINGS") == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.environ.get("EMBEDDINGS_MODEL", "models/text-embedding-004"))
    return OpenAIEmbeddings()


def chat():
    return ChatOpenAI(model=os.environ.get("CHAT_MODEL", "gpt-3.5-turbo"))
