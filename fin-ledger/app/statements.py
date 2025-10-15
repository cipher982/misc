from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from plaid import ApiClient, Configuration
from plaid.api import plaid_api
from plaid.model.statements_download_request import StatementsDownloadRequest

from .config import get_settings
from .db import DuckDB, ensure_parquet_files_exist, sha256_bytes, utc_now_iso


@dataclass(frozen=True)
class StatementMeta:
    account_id: str
    period: str  # YYYY-MM
    file_path: Path
    byte_length: int
    sha256: str


def _statements_dir() -> Path:
    base = Path(os.getenv("FIN_LEDGER_DATA_DIR", "./data")).resolve() / "statements"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _client() -> Optional[plaid_api.PlaidApi]:
    s = get_settings()
    if not s.plaid_client_id or not s.plaid_secret:
        return None
    configuration = Configuration(
        host={
            "sandbox": Configuration.Host.SANDBOX,
            "development": Configuration.Host.DEVELOPMENT,
            "production": Configuration.Host.PRODUCTION,
        }[s.plaid_env]
    )
    configuration.api_key["clientId"] = s.plaid_client_id
    configuration.api_key["secret"] = s.plaid_secret
    return plaid_api.PlaidApi(ApiClient(configuration))


def save_statement_pdf(account_id: str, yyyy_mm: str, content: bytes) -> StatementMeta:
    dir_ = _statements_dir() / account_id
    dir_.mkdir(parents=True, exist_ok=True)
    filename = f"{yyyy_mm}.pdf"
    path = dir_ / filename
    path.write_bytes(content)
    digest = sha256_bytes(content)
    meta = StatementMeta(
        account_id=account_id,
        period=yyyy_mm,
        file_path=path,
        byte_length=len(content),
        sha256=digest,
    )

    # Upsert into statements metadata parquet
    paths = ensure_parquet_files_exist()
    df = pd.DataFrame(
        [
            {
                "account_id": meta.account_id,
                "period": meta.period,
                "file_path": str(path),
                "byte_length": meta.byte_length,
                "sha256": meta.sha256,
                "created_at": utc_now_iso(),
            }
        ]
    )
    if paths.statements.exists() and paths.statements.stat().st_size > 0:
        existing = pd.read_parquet(paths.statements)
    else:
        existing = pd.DataFrame(columns=df.columns)
    combined = pd.concat([existing, df]).drop_duplicates(subset=["account_id", "period"], keep="last")
    combined.to_parquet(paths.statements, index=False)
    return meta


def download_statement(account_id: str, yyyy_mm: str) -> Optional[StatementMeta]:
    client = _client()
    s = get_settings()

    if client is None:
        # Mock: generate a tiny PDF-like file for local tests
        content = b"%PDF-1.4\n%Mock Statement\n"
        return save_statement_pdf(account_id, yyyy_mm, content)

    access_token = s.plaid_access_token or os.getenv("PLAID_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("PLAID_ACCESS_TOKEN missing. Run Link and exchange token.")

    # Plaid requires institution and supported period; in sandbox, limited
    request = StatementsDownloadRequest(access_token=access_token, account_id=account_id, statement_id=yyyy_mm)
    resp = client.statements_download(request)
    # Headers include Plaid-Content-Hash; capture if available
    content = resp.data
    meta = save_statement_pdf(account_id, yyyy_mm, content)
    return meta
