"""Train credit models under several mitigation strategies and audit each one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rai_metrics as rm

from .data import FEATURES
from .impact import LendingEconomics, access_to_credit, portfolio_outcomes

GROUP = "ethnicity_group"
PRIVILEGED = "majority"
MAX_DEFAULT_PROB = 0.15  # approve if predicted probability of default is below this


@dataclass
class StrategyResult:
    name: str
    description: str
    approve: np.ndarray
    p_default: np.ndarray
    features: List[str]
    model: object
    fairness: pd.DataFrame = field(default=None)
    portfolio: Dict = field(default=None)
    access: pd.DataFrame = field(default=None)


def _fit(X, y, sample_weight=None):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y, logisticregression__sample_weight=sample_weight)
    return model


def split(df: pd.DataFrame, seed: int = 0):
    return train_test_split(df, test_size=0.3, random_state=seed, stratify=df["default"])


def run_strategies(train: pd.DataFrame, test: pd.DataFrame, econ: LendingEconomics) -> List[StrategyResult]:
    results: List[StrategyResult] = []
    no_proxy = [f for f in FEATURES if f != "neighbourhood_risk"]

    # 1. Baseline: "fairness through unawareness" (protected attribute excluded, proxy kept).
    m = _fit(train[FEATURES], train["default"])
    p = m.predict_proba(test[FEATURES])[:, 1]
    results.append(StrategyResult("baseline", "All features; protected attribute excluded",
                                  p < MAX_DEFAULT_PROB, p, FEATURES, m))

    # 2. Remove the proxy feature found by the proxy scan.
    m = _fit(train[no_proxy], train["default"])
    p = m.predict_proba(test[no_proxy])[:, 1]
    results.append(StrategyResult("drop_proxy", "Remove neighbourhood_risk (redlining proxy)",
                                  p < MAX_DEFAULT_PROB, p, no_proxy, m))

    # 3. Pre-processing: reweighing (Kamiran & Calders) on top of proxy removal.
    w = rm.reweighing_weights(train["default"], train[GROUP])
    m = _fit(train[no_proxy], train["default"], sample_weight=w)
    p = m.predict_proba(test[no_proxy])[:, 1]
    results.append(StrategyResult("reweighing", "Drop proxy + reweigh training data",
                                  p < MAX_DEFAULT_PROB, p, no_proxy, m))

    # 4. Post-processing: group-specific thresholds equalising opportunity.
    #    NOTE: uses the protected attribute at decision time, which can amount to
    #    disparate treatment under US ECOA. Shown for analysis, not as a recommendation.
    base = results[1]
    creditworthy = (test["default"] == 0).astype(int)
    target = float(np.mean((base.p_default < MAX_DEFAULT_PROB)[creditworthy == 1]))
    th = rm.group_thresholds_equal_opportunity(creditworthy, 1 - base.p_default, test[GROUP], target)
    approve = rm.apply_group_thresholds(1 - base.p_default, test[GROUP], th).astype(bool)
    results.append(StrategyResult("group_thresholds", "Equal-opportunity thresholds per group (analysis only)",
                                  approve, base.p_default, no_proxy, base.model))

    for r in results:
        audit(r, test, econ)
    return results


def audit(r: StrategyResult, test: pd.DataFrame, econ: LendingEconomics) -> None:
    # Favourable outcome = approval; "positive" ground truth = creditworthy (repaid).
    y_good = (test["default"] == 0).astype(int).values
    r.fairness = rm.fairness_report(y_good, r.approve.astype(int), test[GROUP].values,
                                    y_score=1 - r.p_default, privileged=PRIVILEGED)
    r.portfolio = portfolio_outcomes(test, r.approve, econ)
    r.access = access_to_credit(test, r.approve, GROUP)


def adverse_action_reasons(r: StrategyResult, test: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
    """Principal reasons for denial (ECOA / Reg B style) from the logistic model."""
    scaler = r.model.named_steps["standardscaler"]
    lr = r.model.named_steps["logisticregression"]
    Xs = scaler.transform(test[r.features])
    baseline = Xs[~r.approve].mean(axis=0) * 0  # mean of standardised training data = 0
    denied = np.where(~r.approve)[0][:n_examples]
    rows = []
    for i in denied:
        # Coefficients are for P(default): positive contribution pushes toward denial.
        reasons = rm.linear_reason_codes(lr.coef_[0], r.features, Xs[i], baseline, top_k=3, adverse_sign=+1)
        rows.append({"applicant_index": int(test.index[i]),
                     "p_default": round(float(r.p_default[i]), 3),
                     "principal_reasons": ", ".join(reasons)})
    return pd.DataFrame(rows)
