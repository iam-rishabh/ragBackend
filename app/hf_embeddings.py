from typing import List

import requests
from langchain_core.embeddings import Embeddings

ROUTER_URL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"


class HFRouterEmbeddings(Embeddings):
    """Calls Hugging Face's current Inference Providers router for feature-extraction.

    Replaces langchain_huggingface's HuggingFaceEndpointEmbeddings, which still
    targets the decommissioned api-inference.huggingface.co host.
    """

    def __init__(self, model: str, api_token: str, timeout: int = 30):
        self.model = model
        self.url = ROUTER_URL.format(model=model)
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.timeout = timeout

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HuggingFace embedding request failed ({response.status_code}): {response.text}")
        return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]