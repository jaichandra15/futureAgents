"""
FastAPI router for Multimodal Offline RAG chat endpoint.

Implements intelligent routing with vector search, Text-to-SQL, and general chat
using Ollama LLM for streaming responses.
"""

from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agent_router import determine_route
from app.core.ollama_client import OllamaClient
from app.database.session import get_db
from app.tools.sql_tool import execute_sql_agent
from app.tools.vector_tool import execute_vector_search

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Chat endpoint with Server-Sent Events streaming.

    Accepts a query, routes it using intelligent intent detection,
    retrieves context from vector search or SQL execution, and streams
    the response from the Ollama LLM in real-time.

    Args:
        request: ChatRequest containing the user's query
        db: Database session injected via dependency

    Returns:
        StreamingResponse with event-stream media type for real-time UI updates

    Raises:
        HTTPException: If query is empty or processing fails
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events with streaming LLM response."""
        ollama_client = OllamaClient()
        try:
            print(f"\n📨 Chat request: {request.query[:60]}...")

            # Step 1: Determine the routing intent for this query
            route_decision = await determine_route(request.query, ollama_client)
            print(f"✓ Intent detected: {route_decision.intent}")

            # Step 2: Execute the appropriate tool based on intent
            context = ""
            if route_decision.intent == "vector_search":
                print("🔄 Executing vector search tool...")
                context = await execute_vector_search(request.query, db, ollama_client)

            elif route_decision.intent == "sql_query":
                print("🔄 Executing SQL agent...")
                context = await execute_sql_agent(request.query, db, ollama_client)

            else:  # general_chat
                context = "No external context needed."
                print("ℹ️ General chat mode (no external context)")

            # Step 3: Construct the final prompt with context
            final_prompt = f"""System: You are a helpful AI assistant. Answer the user's question using ONLY the following context. If the answer is not in the context, say you do not know.

Context:
{context}

User Question:
{request.query}"""

            print(f"✓ Context length: {len(context)} chars")
            print("🚀 Streaming LLM response...")

            # Step 4: Stream the response from Ollama
            async for chunk in ollama_client.generate_stream(final_prompt):
                # SSE format: data: <content>\n\n
                yield f"data: {chunk}\n\n"

            # Send completion signal
            yield "data: [DONE]\n\n"
            print("✓ Response streaming complete")

        except Exception as e:
            print(f"✗ Chat endpoint error: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: ERROR: {str(e)}\n\n"
        finally:
            await ollama_client.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable proxy buffering for streaming
            "Connection": "keep-alive",
        },
    )
