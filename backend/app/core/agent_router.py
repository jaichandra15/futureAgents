"""
Agent routing logic for query classification and intent detection.

Determines whether a query should be routed to vector search, SQL query, or general chat
using zero-shot prompting with the Ollama LLM.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from app.core.ollama_client import OllamaClient

# System prompt for routing decisions
ROUTING_SYSTEM_PROMPT = """You are a query router. Analyze the user's query and determine:
1. The PRIMARY INTENT: one of ["vector_search", "sql_query", "general_chat"]
2. Relevant TARGETS: list of entity names mentioned (e.g., ["documents", "financial_data"])

Guidelines:
- "vector_search": Query asks about documents, knowledge, information retrieval (e.g., "What does the document say about...?")
- "sql_query": Query asks about data, metrics, finance, revenue, numbers (e.g., "What was the revenue in Q1?")
- "general_chat": Conversational queries, greetings, open-ended questions

Return ONLY a valid JSON object with this exact structure:
{
    "intent": "vector_search|sql_query|general_chat",
    "targets": ["target1", "target2"]
}"""


class RouteDecision(BaseModel):
    """Decision output from the routing agent."""

    intent: Literal["vector_search", "sql_query", "general_chat"] = Field(
        ..., description="Primary intent of the query"
    )
    targets: List[str] = Field(default_factory=list, description="Relevant targets or entities")


async def determine_route(query: str, ollama_client: OllamaClient) -> RouteDecision:
    """
    Determine the route for a query using zero-shot LLM prompting.

    Args:
        query: The user's query string
        ollama_client: Ollama client for LLM inference

    Returns:
        RouteDecision: Structured routing decision with intent and targets

    Raises:
        ValueError: If the LLM response is not valid JSON or doesn't match schema
    """
    # Construct the prompt with system instructions and user query
    prompt = f"""{ROUTING_SYSTEM_PROMPT}

User query: "{query}"

Respond with ONLY the JSON object, no additional text."""

    try:
        # Call the LLM to get JSON response
        response_json = await ollama_client.generate_json(prompt)

        # Validate using Pydantic model
        route_decision = RouteDecision(**response_json)

        return route_decision

    except ValueError as e:
        # If JSON parsing failed, default to general_chat
        print(f"⚠ Failed to parse routing decision: {e}")
        return RouteDecision(intent="general_chat", targets=[])
