from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pandas as pd

from .db import ensure_parquet_files_exist, utc_now_iso


def _doc_id(title: str, content: str) -> str:
    return hashlib.sha256((title + "\n" + content).encode("utf-8")).hexdigest()


def ingest_document(title: str, content: str, source: str = "manual") -> str:
    doc_id = _doc_id(title, content)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    paths = ensure_parquet_files_exist()
    df = pd.DataFrame(
        [
            {
                "document_id": doc_id,
                "title": title,
                "source": source,
                "content_hash": content_hash,
                "content": content,
                "created_at": utc_now_iso(),
            }
        ]
    )

    if paths.documents.exists() and paths.documents.stat().st_size > 0:
        existing = pd.read_parquet(paths.documents)
    else:
        existing = pd.DataFrame(columns=df.columns)

    combined = pd.concat([existing, df]).drop_duplicates(subset=["document_id"], keep="last")
    combined.to_parquet(paths.documents, index=False)
    return doc_id


def search_documents(query: str, limit: int = 10) -> list[dict]:
    paths = ensure_parquet_files_exist()
    if not paths.documents.exists() or paths.documents.stat().st_size == 0:
        return []
    df = pd.read_parquet(paths.documents)
    # naive search: case-insensitive contains in title or content
    mask = df["title"].str.contains(query, case=False, na=False) | df["content"].str.contains(query, case=False, na=False)
    results = df[mask].copy()
    results["score"] = 1.0  # placeholder
    results = results.sort_values(by=["score", "created_at"], ascending=[False, False]).head(limit)
    return results.to_dict(orient="records")
