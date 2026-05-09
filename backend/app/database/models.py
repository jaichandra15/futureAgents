"""
SQLAlchemy data models for Multimodal Offline RAG.

Uses SQLAlchemy 2.0 async declarative base with pgvector support
for semantic search on document embeddings.
"""

from typing import Optional
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()


class DocumentChunk(Base):
    """
    Represents a chunked document with its embedding vector.

    Stores text chunks extracted from ingested documents along with
    their vector embeddings for semantic search operations.
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Vector] = mapped_column(
        Vector(768), nullable=False, index=True
    )  # 768-dim for typical embeddings

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, document_name={self.document_name})>"


class TableSchema(Base):
    """
    Represents a database table schema for Text-to-SQL operations.

    Stores the DDL (Data Definition Language) schema for tables
    to enable the SQL agent to generate correct queries.
    """

    __tablename__ = "table_schemas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    ddl_schema: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"<TableSchema(table_name={self.table_name})>"


class FinancialRecord(Base):
    """
    Represents a financial record for demo queries.

    Example data table used to demonstrate Text-to-SQL capabilities
    on structured financial data like revenue by project and quarter.
    """

    __tablename__ = "financial_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)
    quarter: Mapped[str] = mapped_column(String(10), nullable=False)
    revenue: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"<FinancialRecord(project_name={self.project_name}, quarter={self.quarter}, revenue={self.revenue})>"
