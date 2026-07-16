import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================
# NATIVE OPENROUTER EMBEDDING WRAPPER
# ============================================================
class CustomOpenRouterEmbeddings(Embeddings):
    """
    A custom wrapper that hits OpenRouter's embeddings endpoint natively,
    preventing OpenAI schema validation errors.
    """
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_embedding(self, text_list: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": text_list
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        # OpenRouter returns a standard 'data' block array containing embedding vectors
        response_data = response.json()
        
        # Safely extract the structural data values directly
        return [item["embedding"] for item in response_data["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._get_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._get_embedding([text])[0]


