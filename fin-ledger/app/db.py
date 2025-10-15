from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb  # type: ignore
import pandas as pd

from .config import get_settings


@dataclass(frozen=True)
class ParquetPaths:
    accounts: Path
    transactions: Path
    statements: Path  # metadata table, not PDFs
    documents: Path


def get_parquet_paths() -> ParquetPaths:
    settings = get_settings()
    base = settings.data_dir / "parquet"
    base.mkdir(parents=True, exist_ok=True)
    return ParquetPaths(
        accounts=base / "accounts.parquet",
        transactions=base / "transactions.parquet",
        statements=base / "statements.parquet",
        documents=base / "documents.parquet",
    )


def _empty_accounts_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "account_id",
            "name",
            "official_name",
            "mask",
            "type",
            "subtype",
            "iso_currency_code",
        ]
    )


def _empty_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transaction_id",
            "account_id",
            "date",
            "authorized_date",
            "name",
            "merchant_name",
            "amount",
            "iso_currency_code",
            "category",
            "pfc_primary",
            "pfc_detailed",
            "pending",
            "channel",
            "payment_channel",
            "mcc",
            "last_updated",
        ]
    )


def _empty_statements_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "account_id",
            "period",  # YYYY-MM
            "file_path",
            "byte_length",
            "sha256",
            "created_at",
        ]
    )


def _empty_documents_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "document_id",
            "title",
            "source",
            "content_hash",
            "content",
            "created_at",
        ]
    )


def ensure_parquet_files_exist() -> ParquetPaths:
    paths = get_parquet_paths()
    if not paths.accounts.exists():
        _empty_accounts_df().to_parquet(paths.accounts)
    if not paths.transactions.exists():
        _empty_transactions_df().to_parquet(paths.transactions)
    if not paths.statements.exists():
        _empty_statements_df().to_parquet(paths.statements)
    if not paths.documents.exists():
        _empty_documents_df().to_parquet(paths.documents)
    return paths


class DuckDB:
    def __init__(self) -> None:
        self.paths = ensure_parquet_files_exist()

    def connect(self) -> duckdb.DuckDBPyConnection:  # type: ignore
        con = duckdb.connect()
        # Expose parquet tables as views for concise queries
        con.execute(
            f"""
            CREATE OR REPLACE VIEW accounts AS SELECT * FROM read_parquet('{self.paths.accounts.as_posix()}');
            CREATE OR REPLACE VIEW transactions AS SELECT * FROM read_parquet('{self.paths.transactions.as_posix()}');
            CREATE OR REPLACE VIEW statements_meta AS SELECT * FROM read_parquet('{self.paths.statements.as_posix()}');
            CREATE OR REPLACE VIEW documents AS SELECT * FROM read_parquet('{self.paths.documents.as_posix()}');
            """
        )
        return con

    def query_df(self, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
        with self.connect() as con:
            if params is None:
                return con.execute(sql).fetch_df()
            return con.execute(sql, params).fetch_df()

    def query_one(self, sql: str, params: Iterable[Any] | None = None) -> Any:
        with self.connect() as con:
            if params is None:
                row = con.execute(sql).fetchone()
            else:
                row = con.execute(sql, params).fetchone()
            return row[0] if row else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
