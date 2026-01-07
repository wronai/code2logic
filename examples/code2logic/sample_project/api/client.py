from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Response:
    """Very small HTTP-like response placeholder."""

    status_code: int
    text: str


class APIClient:
    """Example client with async methods."""

    async def get(self, url: str) -> Response:
        """Fetch a URL."""
        return Response(status_code=200, text=f"GET {url}")

    async def post(self, url: str, data: dict) -> Response:
        """Send JSON-like data."""
        return Response(status_code=201, text=f"POST {url} {data}")
