"""Model providers, chosen by environment: OpenAI by default.

EMBEDDINGS=google uses Gemini embeddings (GOOGLE_API_KEY). CHAT_MODEL picks the chat model; with
OPENAI_BASE_URL any OpenAI-compatible endpoint works.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def embeddings():
    if os.environ.get("EMBEDDINGS") == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return _Paced(GoogleGenerativeAIEmbeddings(model=os.environ.get("EMBEDDINGS_MODEL", "models/gemini-embedding-001")))
    return OpenAIEmbeddings()


class _Paced:
    """Gemini's free tier limits tokens per minute: embed in batches, wait and retry on 429."""

    def __init__(self, inner, batch=100, pause_s=15.0):
        self.inner, self.batch, self.pause_s = inner, batch, pause_s

    def embed_documents(self, texts):
        import time
        out = []
        for i in range(0, len(texts), self.batch):
            for attempt in range(8):
                try:
                    out += self.inner.embed_documents(texts[i:i + self.batch])
                    break
                except Exception as exc:  # noqa: BLE001
                    if "429" not in str(exc) or attempt == 7:
                        raise
                    time.sleep(self.pause_s * (attempt + 2))
            print(f"embedded {min(i + self.batch, len(texts))}/{len(texts)}", flush=True)
            time.sleep(self.pause_s)
        return out

    def embed_query(self, text):
        return self.inner.embed_query(text)


def chat():
    return ChatOpenAI(model=os.environ.get("CHAT_MODEL", "gpt-3.5-turbo"))
