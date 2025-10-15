from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb  # type: ignore
import pandas as pd
import yaml

from .config import get_settings
from .db import DuckDB


@dataclass(frozen=True)
class BenefitCredit:
    key: str
    name: str
    cap: float
    currency: str
    window: str  # calendar_year, cardmember_year, multi_year
    # In practice: parse match rules; here we just pass.


def load_benefits(product_slug: str) -> dict[str, Any]:
    base = Path(__file__).resolve().parent.parent / "benefits"
    path = base / f"{product_slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Benefits config not found: {path}")
    return yaml.safe_load(path.read_text())


def compute_credit_usage(product_slug: str, year: int) -> dict[str, Any]:
    cfg = load_benefits(product_slug)
    credits = cfg.get("credits", [])

    con = DuckDB().connect()
    # Basic example: sum transactions matching simple contains for memo/name
    # In real usage, you'd interpret MCC ranges and other filters.

    results: list[dict[str, Any]] = []
    for credit in credits:
        cap = float(credit.get("cap", 0.0))
        name = credit.get("name")
        key = credit.get("key")
        window = credit.get("window", "calendar_year")
        # naive matching placeholder: any transaction with merchant_name containing tokens
        tokens = []
        any_clause = credit.get("match", {}).get("any") or []
        for cond in any_clause:
            if "memo_contains" in cond:
                tokens.extend(cond["memo_contains"])  # treat as name contains for demo

        if tokens:
            predicates = " OR ".join(["LOWER(name) LIKE '%' || LOWER(?) || '%'" for _ in tokens])
            sql = (
                "SELECT COALESCE(SUM(amount),0) FROM transactions "
                "WHERE date >= ? AND date < ? AND (" + predicates + ")"
            )
            start = f"{year}-01-01"
            end = f"{year+1}-01-01"
            total = con.execute(sql, [start, end, *tokens]).fetchone()[0] or 0.0
        else:
            total = 0.0

        used = max(0.0, min(float(total), cap))
        results.append(
            {
                "key": key,
                "name": name,
                "cap": cap,
                "used": used,
                "remaining": max(0.0, cap - used),
                "currency": credit.get("currency", "USD"),
                "window": window,
            }
        )

    return {"product": cfg.get("product"), "issuer": cfg.get("issuer"), "year": year, "credits": results}
