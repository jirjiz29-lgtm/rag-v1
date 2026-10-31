"""HTTP client the Streamlit UI uses to talk to the FastAPI backend. Holds no secrets."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class APIError(Exception):
    pass


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = requests.request(method, f"{self.base_url}{path}", timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except requests.ConnectionError as exc:
            raise APIError(f"Cannot reach the backend at {self.base_url}. Is the API server running?") from exc
        except requests.Timeout as exc:
            raise APIError("The backend took too long to respond. Please try again.") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            if isinstance(detail, list):
                detail = "; ".join(d.get("message", str(d)) if isinstance(d, dict) else str(d) for d in detail)
            raise APIError(str(detail))
        return resp.json()

    def health(self) -> Dict:
        return self._request("GET", "/health", timeout=5)

    def settings(self) -> Dict:
        return self._request("GET", "/settings", timeout=10)

    def chat(self, question: str, history: List[Dict], **filters) -> Dict:
        payload = {"question": question, "history": history}
        payload.update({k: v for k, v in filters.items() if v not in (None, [], "")})
        return self._request("POST", "/chat", json=payload)

    def documents(self) -> List[Dict]:
        return self._request("GET", "/documents", timeout=15)

    def stats(self) -> Dict:
        return self._request("GET", "/stats", timeout=15)

    def upload(self, files: List[tuple], category: str) -> Dict:
        multipart = [("files", (name, data, mime or "application/octet-stream")) for name, data, mime in files]
        return self._request("POST", "/documents/upload", files=multipart, data={"category": category})

    def chunks(self, doc_id: str) -> Dict:
        return self._request("GET", f"/documents/{doc_id}/chunks", timeout=30)

    def delete(self, doc_id: str) -> Dict:
        return self._request("DELETE", f"/documents/{doc_id}", timeout=30)

    def reindex(self, doc_id: Optional[str] = None) -> Dict:
        path = f"/documents/{doc_id}/reindex" if doc_id else "/documents/reindex"
        return self._request("POST", path)

    def sync(self) -> Dict:
        return self._request("POST", "/documents/sync")
