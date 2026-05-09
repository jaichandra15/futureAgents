"""
Vector search tool for semantic document retrieval.

Uses Ollama embeddings (nomic-embed-text) to find similar document chunks
from the pgvector-enabled PostgreSQL database.
"""

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ollama_client import OllamaClient
from app.database.models import DocumentChunk


async def execute_vector_search(
    query: str, session: AsyncSession, ollama_client: OllamaClient
) -> str:
    """
    Execute vector search against document embeddings.

    Retrieves the top 3 most similar document chunks using cosine distance
    and returns their content formatted for use as RAG context.

    Args:
        query: The user's search query
        session: AsyncSession for database access
        ollama_client: OllamaClient for generating query embeddings

    Returns:
        str: Formatted context string with top 3 chunk results

    Raises:
        Exception: If embedding generation or database query fails
    """
    try:
        # Step 1: Generate embedding for the query using Ollama
        print(f"🔍 Generating embedding for query: {query[:50]}...")

        embedding_response = await ollama_client.client.post(
            f"{ollama_client.base_url}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": query,
            },
        )
        embedding_response.raise_for_status()

        embedding_data = embedding_response.json()
        query_embedding = embedding_data.get("embedding")

        if not query_embedding:
            return "❌ Failed to generate query embedding."

        print(f"✓ Generated embedding (dimension: {len(query_embedding)})")

        # Step 2: Query DocumentChunk table using cosine distance
        # pgvector's <=> operator provides cosine distance
        print("📚 Querying database for similar chunks...")

        stmt = (
            select(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(3)
        )

        result = await session.execute(stmt)
        chunks = result.scalars().all()

        if not chunks:
            return "⚠️ No relevant documents found in the database."

        print(f"✓ Found {len(chunks)} relevant chunks")

        # Step 3: Format results into context string
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"Source: {chunk.document_name}\nContent: {chunk.content}"
            )

        context = "\n\n".join(context_parts)

        print(f"✓ Vector search complete ({len(context)} chars)")
        return context

    except Exception as e:
        error_msg = f"❌ Vector search error: {str(e)}"
        print(error_msg)
        return error_msg
