from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from plaid import ApiClient, Configuration
from plaid.api import plaid_api
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.transactions_sync_request import TransactionsSyncRequest

from .config import get_settings
from .db import DuckDB, ensure_parquet_files_exist, utc_now_iso


@dataclass
class PlaidState:
    cursor: str | None


def _state_path() -> Path:
    settings = get_settings()
    state_dir = Path(os.getenv("FIN_LEDGER_DATA_DIR", "./data")).resolve() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "plaid_state.json"


def load_state() -> PlaidState:
    path = _state_path()
    if not path.exists():
        return PlaidState(cursor=None)
    data = json.loads(path.read_text())
    return PlaidState(cursor=data.get("cursor"))


def save_state(state: PlaidState) -> None:
    path = _state_path()
    path.write_text(json.dumps({"cursor": state.cursor}, indent=2))


@dataclass(frozen=True)
class SyncResult:
    added: int
    modified: int
    removed: int
    accounts: int
    end_cursor: str | None


class PlaidClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        configuration = Configuration(
            host={
                "sandbox": Configuration.Host.SANDBOX,
                "development": Configuration.Host.DEVELOPMENT,
                "production": Configuration.Host.PRODUCTION,
            }[self.settings.plaid_env]
        )
        configuration.api_key["clientId"] = self.settings.plaid_client_id or ""
        configuration.api_key["secret"] = self.settings.plaid_secret or ""
        self.client = plaid_api.PlaidApi(ApiClient(configuration))

    def create_link_token(self) -> str:
        user = LinkTokenCreateRequestUser(client_user_id="demo-user-123")
        request = LinkTokenCreateRequest(
            products=[Products("transactions"), Products("statements")],
            client_name="Fin Ledger",
            country_codes=["US"],
            language="en",
            user=user,
        )
        resp = self.client.link_token_create(request)
        return resp["link_token"]

    def exchange_public_token(self, public_token: str) -> str:
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        resp = self.client.item_public_token_exchange(request)
        return resp["access_token"]

    def get_accounts(self, access_token: str) -> list[dict]:
        resp = self.client.accounts_get(AccountsGetRequest(access_token=access_token))
        return [a.to_dict() for a in resp["accounts"]]

    def transactions_sync(self, access_token: str, cursor: str | None) -> dict:
        request = TransactionsSyncRequest(access_token=access_token, cursor=cursor)
        resp = self.client.transactions_sync(request)
        return resp.to_dict()


class MockPlaidClient:
    """Offline mock for development without real Plaid credentials."""

    def __init__(self) -> None:
        pass

    def create_link_token(self) -> str:
        return "mock-link-token"

    def exchange_public_token(self, public_token: str) -> str:
        return "mock-access-token"

    def get_accounts(self, access_token: str) -> list[dict]:
        return [
            {
                "account_id": "acc_mock_1",
                "name": "Mock Checking",
                "official_name": "Checking",
                "mask": "0001",
                "type": "depository",
                "subtype": "checking",
                "balances": {"iso_currency_code": "USD"},
            },
            {
                "account_id": "acc_mock_cc",
                "name": "Mock Credit Card",
                "official_name": "Visa",
                "mask": "4242",
                "type": "credit",
                "subtype": "credit card",
                "balances": {"iso_currency_code": "USD"},
            },
        ]

    def transactions_sync(self, access_token: str, cursor: str | None) -> dict:
        today = date.today()
        base_date = today - timedelta(days=120)
        txns = []
        for i in range(120):
            d = base_date + timedelta(days=i)
            txns.append(
                {
                    "transaction_id": f"txn_mock_{i}",
                    "account_id": "acc_mock_cc" if i % 3 else "acc_mock_1",
                    "date": d.isoformat(),
                    "authorized_date": None,
                    "name": "Amazon Marketplace" if i % 5 == 0 else "Generic Merchant",
                    "merchant_name": "Amazon" if i % 5 == 0 else None,
                    "amount": float((i % 20) + 1),
                    "iso_currency_code": "USD",
                    "category": "Shops",
                    "pfc_primary": None,
                    "pfc_detailed": None,
                    "pending": False,
                    "channel": "online",
                    "payment_channel": "card",
                    "mcc": "5311" if i % 5 == 0 else None,
                    "last_updated": utc_now_iso(),
                }
            )
        return {
            "added": txns,
            "modified": [],
            "removed": [],
            "next_cursor": "mock-cursor",
            "has_more": False,
        }


def _choose_client() -> PlaidClient | MockPlaidClient:
    s = get_settings()
    if not s.plaid_client_id or not s.plaid_secret:
        return MockPlaidClient()
    return PlaidClient()


def _upsert_accounts(df: pd.DataFrame) -> None:
    paths = ensure_parquet_files_exist()
    if paths.accounts.exists() and paths.accounts.stat().st_size > 0:
        existing = pd.read_parquet(paths.accounts)
    else:
        existing = pd.DataFrame(columns=df.columns)
    combined = pd.concat([existing, df]).drop_duplicates(subset=["account_id"], keep="last")
    combined.to_parquet(paths.accounts, index=False)


def _upsert_transactions(df: pd.DataFrame) -> None:
    paths = ensure_parquet_files_exist()
    if paths.transactions.exists() and paths.transactions.stat().st_size > 0:
        existing = pd.read_parquet(paths.transactions)
    else:
        existing = pd.DataFrame(columns=df.columns)
    combined = pd.concat([existing, df]).drop_duplicates(subset=["transaction_id"], keep="last")
    combined.to_parquet(paths.transactions, index=False)


def sync_all() -> SyncResult:
    client = _choose_client()
    settings = get_settings()

    access_token = settings.plaid_access_token or os.getenv("PLAID_ACCESS_TOKEN")
    if isinstance(client, PlaidClient) and not access_token:
        raise RuntimeError("PLAID_ACCESS_TOKEN missing. Run Link and exchange token.")

    state = load_state()

    # Accounts
    accounts = client.get_accounts(access_token or "mock")
    accounts_rows = [
        {
            "account_id": a["account_id"],
            "name": a.get("name"),
            "official_name": a.get("official_name"),
            "mask": a.get("mask"),
            "type": a.get("type"),
            "subtype": a.get("subtype"),
            "iso_currency_code": (a.get("balances") or {}).get("iso_currency_code"),
        }
        for a in accounts
    ]
    _upsert_accounts(pd.DataFrame(accounts_rows))

    # Transactions sync loop (single page for mock; loop for real client)
    total_added = total_modified = total_removed = 0
    end_cursor: str | None = state.cursor

    while True:
        resp = client.transactions_sync(access_token or "mock", state.cursor)
        added = resp.get("added", [])
        modified = resp.get("modified", [])
        removed = resp.get("removed", [])
        next_cursor = resp.get("next_cursor")
        has_more = bool(resp.get("has_more"))

        if added:
            _upsert_transactions(pd.DataFrame(added))
        if modified:
            _upsert_transactions(pd.DataFrame(modified))
        if removed:
            # Mark removed by setting amount = 0 and name tag; maintain history
            rem_df = pd.DataFrame(
                [
                    {"transaction_id": r.get("transaction_id"), "name": "REMOVED", "last_updated": utc_now_iso()}
                    for r in removed
                ]
            )
            _upsert_transactions(rem_df)

        total_added += len(added)
        total_modified += len(modified)
        total_removed += len(removed)
        end_cursor = next_cursor or state.cursor

        if not has_more:
            break
        state.cursor = next_cursor

    state.cursor = end_cursor
    save_state(state)

    return SyncResult(
        added=total_added, modified=total_modified, removed=total_removed, accounts=len(accounts_rows), end_cursor=end_cursor
    )
