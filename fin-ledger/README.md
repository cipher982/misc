# Fin Ledger

Headless finance data layer with DuckDB/Parquet and FastAPI.

- One storage truth: Parquet files under `data/parquet/` with DuckDB views.
- Sync from Plaid (mock mode if keys missing).
- Minimal API for agents: fees, renewal, spend, statements, documents, benefits.

## Quick start

1. Copy `.env.example` to `.env` and fill values (or leave blank for mock mode).
2. Install dependencies using uv:

```bash
uv pip install -e .
```

3. Run API:

```bash
uvicorn app.main:app --reload --port 8080
```

4. Sync data (mock mode if no Plaid keys):

```bash
curl -X POST http://localhost:8080/sync
```

5. Query endpoints:

```bash
curl "http://localhost:8080/fees?since=2025-01-01"
curl "http://localhost:8080/spend?merchant=Amazon&since=2025-01-01"
```

## Data layout

- `data/parquet/accounts.parquet`
- `data/parquet/transactions.parquet`
- `data/parquet/statements.parquet` (metadata)
- `data/statements/{account_id}/{yyyy-mm}.pdf`

## Notes

- Mock mode is used if `PLAID_CLIENT_ID` or `PLAID_SECRET` are missing.
- Do not commit `.env` or any statement PDFs.
- This is an educational scaffold and not a production system.
