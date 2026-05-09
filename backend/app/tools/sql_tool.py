"""
Text-to-SQL tool for structured data queries.

Generates and executes SQL queries based on table schemas and user intent,
using the Ollama LLM for SQL generation.
"""

import json
import re

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ollama_client import OllamaClient
from app.database.models import TableSchema


async def execute_sql_agent(
    query: str, session: AsyncSession, ollama_client: OllamaClient
) -> str:
    """
    Execute Text-to-SQL agent for structured data queries.

    Fetches database schemas, generates a SELECT statement using the LLM,
    and executes it safely against the database.

    Args:
        query: The user's natural language query about data
        session: AsyncSession for database access
        ollama_client: OllamaClient for SQL generation

    Returns:
        str: Formatted SQL query results or error message

    Raises:
        Exception: If schema fetching or query execution fails
    """
    try:
        # Step 1: Fetch all table DDL schemas from database
        print("📋 Fetching table schemas...")

        stmt = select(TableSchema)
        result = await session.execute(stmt)
        schemas = result.scalars().all()

        if not schemas:
            return "⚠️ No table schemas available in database."

        # Build DDL context string
        ddl_context = "\n\n".join(
            [f"Table: {schema.table_name}\nDDL:\n{schema.ddl_schema}" for schema in schemas]
        )

        print(f"✓ Loaded {len(schemas)} table schemas")

        # Step 2: Create SQL generation prompt
        sql_prompt = f"""You are a PostgreSQL SQL expert. Generate a raw, executable SELECT statement based on the user's query.

IMPORTANT:
- Output ONLY the raw SQL statement
- NO markdown formatting (no backticks, no code blocks)
- NO explanation, NO other text
- Must be valid PostgreSQL syntax
- Use only the tables and columns defined below

Available Tables and Schemas:
{ddl_context}

User Query: {query}

Generate ONLY the SELECT statement:"""

        print("🤖 Generating SQL query...")

        # Step 3: Call LLM to generate SQL
        try:
            generated_sql = await ollama_client.generate_json(sql_prompt)

            # Extract SQL from potential JSON wrapper
            if isinstance(generated_sql, dict):
                sql_text = generated_sql.get("sql", generated_sql.get("query", str(generated_sql)))
            else:
                sql_text = str(generated_sql)

        except (json.JSONDecodeError, ValueError):
            # Fallback: use generate_stream for plain text response
            print("⚠️ JSON parsing failed, using streaming generation...")
            sql_text = ""
            async for chunk in ollama_client.generate_stream(sql_prompt):
                sql_text += chunk

        # Strip markdown formatting
        sql_text = re.sub(r"^```(sql)?\n?", "", sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r"\n?```$", "", sql_text)
        sql_text = sql_text.strip()

        print(f"✓ Generated SQL (length: {len(sql_text)})")
        print(f"  SQL: {sql_text[:100]}...")

        # Step 4: Execute the generated SQL safely
        try:
            result = await session.execute(text(sql_text))
            rows = result.fetchall()

            if not rows:
                return "ℹ️ Query executed successfully but returned no results."

            print(f"✓ Query executed, {len(rows)} rows returned")

        except Exception as e:
            error_msg = f"❌ SQL execution failed: {str(e)}\nGenerated SQL: {sql_text}"
            print(error_msg)
            return error_msg

        # Step 5: Format results into readable string
        # Get column names from result
        column_names = result.keys()

        # Convert rows to formatted output
        result_lines = [f"Query executed successfully. Returned {len(rows)} rows.\n"]
        result_lines.append("| " + " | ".join(str(col) for col in column_names) + " |")
        result_lines.append("|" + "|".join(["---" for _ in column_names]) + "|")

        for row in rows[:10]:  # Limit to first 10 rows for display
            formatted_row = [str(val) for val in row]
            result_lines.append("| " + " | ".join(formatted_row) + " |")

        if len(rows) > 10:
            result_lines.append(f"\n... and {len(rows) - 10} more rows")

        formatted_result = "\n".join(result_lines)
        print(f"✓ SQL agent complete ({len(formatted_result)} chars)")

        return formatted_result

    except Exception as e:
        error_msg = f"❌ SQL agent error: {str(e)}"
        print(error_msg)
        return error_msg
