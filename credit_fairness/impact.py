"""Translate fairness metrics into business and human impact for lending."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LendingEconomics:
    """Simple unit economics for a personal-loan book (illustrative values)."""

    interest_margin: float = 0.12  # net interest earned on a repaid loan, as a share of principal
    loss_given_default: float = 0.60  # share of principal lost when a borrower defaults


def portfolio_outcomes(df: pd.DataFrame, approve: np.ndarray, econ: LendingEconomics) -> dict:
    """Profit and default rate of the approved book."""
    approve = np.asarray(approve).astype(bool)
    loans = df["loan_amount_k"].values * 1000
    repaid = df["default"].values == 0
    profit = (loans * econ.interest_margin * (approve & repaid)).sum() - (
        loans * econ.loss_given_default * (approve & ~repaid)
    ).sum()
    return {
        "approval_rate": float(approve.mean()),
        "book_default_rate": float((~repaid[approve]).mean()) if approve.any() else float("nan"),
        "expected_profit": float(profit),
    }


def access_to_credit(df: pd.DataFrame, approve: np.ndarray, group_col: str) -> pd.DataFrame:
    """Per group: approvals, and creditworthy applicants who were wrongly denied."""
    approve = np.asarray(approve).astype(bool)
    creditworthy = df["default"].values == 0
    rows = []
    for g, idx in df.groupby(group_col).indices.items():
        a, c = approve[idx], creditworthy[idx]
        wrongly_denied = int((~a & c).sum())
        rows.append(
            {
                group_col: g,
                "applicants": len(idx),
                "approval_rate": a.mean(),
                "creditworthy_denied": wrongly_denied,
                "creditworthy_denied_per_1000": 1000 * wrongly_denied / len(idx),
                "credit_withheld_usd": float(df["loan_amount_k"].values[idx][~a & c].sum() * 1000),
            }
        )
    return pd.DataFrame(rows).set_index(group_col)
