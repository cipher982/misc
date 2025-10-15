from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .db import DuckDB
from .plaid_sync import sync_all
from .statements import download_statement
from .documents import ingest_document, search_documents
from .benefits import compute_credit_usage


app = FastAPI(title="Fin Ledger", version="0.1.0")


class FeesResponse(BaseModel):
    count: int
    total: float


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {"ok": True, "env": s.plaid_env}


@app.post("/sync")
async def sync() -> dict:
    res = sync_all()
    return {
        "added": res.added,
        "modified": res.modified,
        "removed": res.removed,
        "accounts": res.accounts,
        "cursor": res.end_cursor,
    }


@app.get("/fees")
async def fees(since: str) -> FeesResponse:
    # Fees: simple heuristic on transaction name/category
    con = DuckDB().connect()
    sql = (
        "SELECT COUNT(*) AS c, COALESCE(SUM(amount),0) AS s FROM transactions "
        "WHERE date >= ? AND (LOWER(name) LIKE '%fee%' OR LOWER(name) LIKE '%interest%')"
    )
    c, s = con.execute(sql, [since]).fetchone()
    return FeesResponse(count=int(c or 0), total=float(s or 0.0))


@app.get("/renewal/{account_id}")
async def renewal(account_id: str) -> dict:
    con = DuckDB().connect()
    # Heuristic: find latest annual fee charge
    sql = (
        "SELECT date, amount FROM transactions WHERE account_id = ? "
        "AND LOWER(name) LIKE '%annual fee%' ORDER BY date DESC LIMIT 1"
    )
    row = con.execute(sql, [account_id]).fetchone()
    if not row:
        raise HTTPException(404, detail="No annual fee found")
    return {"last_renewal_date": row[0], "amount": float(row[1] or 0.0)}


@app.get("/spend")
async def spend(merchant: str, since: str) -> dict:
    con = DuckDB().connect()
    sql = (
        "SELECT COALESCE(SUM(amount),0) AS total, COUNT(*) AS cnt, MAX(date) AS last_date "
        "FROM transactions WHERE date >= ? AND LOWER(COALESCE(merchant_name, name)) LIKE '%' || LOWER(?) || '%'"
    )
    total, cnt, last_date = con.execute(sql, [since, merchant]).fetchone()
    return {"total": float(total or 0.0), "count": int(cnt or 0), "last_date": last_date}


@app.get("/statement/{account_id}/{yyyymm}")
async def statement(account_id: str, yyyymm: str) -> dict:
    yyyy_mm = f"{yyyymm[:4]}-{yyyymm[4:]}" if "-" not in yyyymm else yyyymm
    meta = download_statement(account_id, yyyy_mm)
    if not meta:
        raise HTTPException(404, detail="Statement not available")
    return {"account_id": meta.account_id, "period": meta.period, "file_path": str(meta.file_path), "sha256": meta.sha256}


class DocumentIngestRequest(BaseModel):
    title: str
    content: str
    source: Optional[str] = "manual"


@app.post("/documents/ingest")
async def documents_ingest(req: DocumentIngestRequest) -> dict:
    doc_id = ingest_document(req.title, req.content, req.source or "manual")
    return {"document_id": doc_id}


@app.get("/documents/search")
async def documents_search(q: str, limit: int = 10) -> dict:
    results = search_documents(q, limit)
    return {"results": results}


@app.get("/benefits/{product_slug}")
async def benefits(product_slug: str, year: int) -> dict:
    return compute_credit_usage(product_slug, year)
