from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    plaid_client_id: str | None
    plaid_secret: str | None
    plaid_env: str
    plaid_webhook_secret: str | None
    plaid_access_token: str | None
    data_dir: Path


def get_settings() -> Settings:
    data_dir = Path(os.getenv("FIN_LEDGER_DATA_DIR", "./data")).resolve()
    return Settings(
        plaid_client_id=os.getenv("PLAID_CLIENT_ID"),
        plaid_secret=os.getenv("PLAID_SECRET"),
        plaid_env=os.getenv("PLAID_ENV", "sandbox"),
        plaid_webhook_secret=os.getenv("PLAID_WEBHOOK_SECRET"),
        plaid_access_token=os.getenv("PLAID_ACCESS_TOKEN"),
        data_dir=data_dir,
    )
