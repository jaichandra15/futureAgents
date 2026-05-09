"""
Async Ollama API client for LLM interactions.

Supports both JSON responses and streaming for real-time output.
"""

import json
import os
from typing import AsyncGenerator

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")


class OllamaClient:
    """
    Asynchronous client for Ollama API.

    Provides methods for generating JSON responses and streaming outputs.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL
            model: Model name to use for generation
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)

    async def generate_json(self, prompt: str) -> dict:
        """
        Generate a JSON response from the model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            dict: Parsed JSON response from the model

        Raises:
            httpx.HTTPError: If the request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
        )
        response.raise_for_status()

        data = response.json()
        # Extract the response text from Ollama's response format
        response_text = data.get("response", "")

        # Parse the JSON from the response
        return json.loads(response_text)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the model.

        Yields chunks of text as the model generates them, suitable for
        Server-Sent Events (SSE) streaming to the frontend.

        Args:
            prompt: The prompt to send to the model

        Yields:
            str: Text chunks from the model response

        Raises:
            httpx.HTTPError: If the request fails
        """
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()

            # Stream response line by line (Ollama sends JSON objects)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    chunk_data = json.loads(line)
                    # Extract the text chunk from Ollama's response format
                    chunk_text = chunk_data.get("response", "")
                    if chunk_text:
                        yield chunk_text
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self.client.aclose()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
