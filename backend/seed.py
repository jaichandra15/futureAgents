"""
Database seeding script for Multimodal Offline RAG.

Populates PostgreSQL with mock data for Phase C testing:
- Financial records for Text-to-SQL demo
- Table schemas for SQL generation
- Document chunks with embeddings for vector search

Usage:
    python backend/seed.py
"""

import asyncio
import sys
from uuid import uuid4

from sqlalchemy import delete

from app.core.ollama_client import OllamaClient
from app.database.models import DocumentChunk, FinancialRecord, TableSchema
from app.database.session import AsyncSessionLocal, init_db


async def seed_database() -> None:
    """
    Seed the database with mock data for Phase C testing.

    Clears existing data, inserts new records, generates embeddings.
    """
    print("🌱 Starting database seed...\n")

    # Initialize database (creates pgvector extension and tables)
    await init_db()

    # Create Ollama client for embedding generation
    ollama_client = OllamaClient()

    try:
        async with AsyncSessionLocal() as session:
            # ─── Step 1: Clear existing data to prevent duplicates ──────────────
            print("🗑️  Clearing existing data...")

            await session.execute(delete(DocumentChunk))
            await session.execute(delete(TableSchema))
            await session.execute(delete(FinancialRecord))
            await session.commit()
            print("✓ Existing data cleared\n")

            # ─── Step 2: Insert FinancialRecord rows ───────────────────────────
            print("💰 Inserting financial records...")

            financial_records = [
                FinancialRecord(
                    project_name="Project Alpha",
                    quarter="Q1",
                    revenue=150000.0,
                ),
                FinancialRecord(
                    project_name="Project Alpha",
                    quarter="Q2",
                    revenue=200000.0,
                ),
                FinancialRecord(
                    project_name="Project Beta",
                    quarter="Q1",
                    revenue=95000.0,
                ),
            ]

            session.add_all(financial_records)
            await session.commit()
            print(f"✓ Inserted {len(financial_records)} financial records\n")

            # ─── Step 3: Insert TableSchema with FinancialRecord DDL ──────────
            print("📋 Inserting table schema...")

            financial_record_ddl = """
CREATE TABLE financial_records (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    revenue FLOAT NOT NULL
);
            """.strip()

            table_schema = TableSchema(
                table_name="financial_records",
                ddl_schema=financial_record_ddl,
            )

            session.add(table_schema)
            await session.commit()
            print("✓ Table schema inserted\n")

            # ─── Step 4: Create document chunk with embedding ──────────────────
            print("📄 Creating document chunk with embedding...")

            # Mock document text
            document_text = (
                "Acme Corp's Project Alpha focuses on next-generation AI routing. "
                "The CEO is Jane Doe."
            )

            # Step 4a: Generate embedding using Ollama
            print(f"🔄 Generating embedding for: '{document_text[:50]}...'")

            try:
                embedding_response = await ollama_client.client.post(
                    f"{ollama_client.base_url}/api/embeddings",
                    json={
                        "model": "nomic-embed-text",
                        "prompt": document_text,
                    },
                )
                embedding_response.raise_for_status()

                embedding_data = embedding_response.json()
                embedding_vector = embedding_data.get("embedding")

                if not embedding_vector:
                    print("❌ Failed to generate embedding from Ollama")
                    return

                print(
                    f"✓ Generated embedding (dimension: {len(embedding_vector)})\n"
                )

                # Step 4b: Insert DocumentChunk with embedding
                print("💾 Inserting document chunk...")

                doc_chunk = DocumentChunk(
                    id=uuid4(),
                    document_name="project_alpha_overview.md",
                    content=document_text,
                    embedding=embedding_vector,
                )

                session.add(doc_chunk)
                await session.commit()
                print("✓ Document chunk inserted\n")

            except Exception as e:
                print(f"❌ Error generating embedding: {e}\n")
                return

            # ─── Summary ────────────────────────────────────────────────────────
            print("✅ Database seed complete!")
            print("\nInserted data:")
            print("  • 3x FinancialRecord (revenues by project/quarter)")
            print("  • 1x TableSchema (financial_records DDL)")
            print("  • 1x DocumentChunk (with nomic-embed-text embedding)")
            print("\nReady for Phase C testing! 🚀\n")

    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        await ollama_client.close()


async def main() -> None:
    """Entry point for the seeding script."""
    try:
        await seed_database()
    except KeyboardInterrupt:
        print("\n⚠️  Seed interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Multimodal Offline RAG — Database Seeder\n")
    print("=" * 50)
    asyncio.run(main())
