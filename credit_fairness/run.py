"""End-to-end Responsible AI audit of a credit-approval model.

Usage:
    python -m credit_fairness.run            # writes reports/ (Markdown + charts)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rai_metrics as rm

from .data import FEATURES, make_lending_data
from .impact import LendingEconomics
from .pipeline import GROUP, adverse_action_reasons, run_strategies, split


def _md(df: pd.DataFrame, fmt=".3f") -> str:
    try:
        return df.to_markdown(floatfmt=fmt)
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def main(out_dir: str = "reports", n: int = 20_000, seed: int = 7) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    econ = LendingEconomics()

    df = make_lending_data(n=n, seed=seed)
    train, test = split(df)
    results = run_strategies(train, test, econ)
    base = results[0]

    # ---- Summary across strategies
    summary = pd.DataFrame(
        [
            {
                "strategy": r.name,
                "description": r.description,
                "approval_rate": r.portfolio["approval_rate"],
                "book_default_rate": r.portfolio["book_default_rate"],
                "expected_profit_usd": r.portfolio["expected_profit"],
                "disparate_impact": r.fairness.loc["disparate_impact_ratio", "value"],
                "equal_opportunity_gap": r.fairness.loc["equal_opportunity_difference", "value"],
                "creditworthy_minority_denied": int(r.access.loc["minority", "creditworthy_denied"]),
            }
            for r in results
        ]
    ).set_index("strategy")
    summary.to_csv(out / "strategy_summary.csv")
    display = summary.drop(columns="description").copy()
    for c in ["approval_rate", "book_default_rate"]:
        display[c] = display[c].map("{:.1%}".format)
    display["expected_profit_usd"] = display["expected_profit_usd"].map("${:,.0f}".format)
    for c in ["disparate_impact", "equal_opportunity_gap"]:
        display[c] = display[c].map("{:.3f}".format)
    display["creditworthy_minority_denied"] = display["creditworthy_minority_denied"].map("{:,}".format)

    # ---- Transparency & robustness on the baseline
    proxy = rm.proxy_feature_scan(test[FEATURES], test[GROUP])
    importance = rm.permutation_importance(
        lambda d: base.model.predict_proba(d)[:, 1], test[FEATURES], test["default"], n_repeats=3
    )
    robustness = rm.noise_robustness(lambda d: base.model.predict_proba(d)[:, 1], test[FEATURES], test["default"])
    flip = rm.prediction_flip_rate(lambda d: base.model.predict_proba(d)[:, 1] < 0.15, test[FEATURES])
    psi = rm.population_stability_index(train["income_k"], test["income_k"])
    reasons = adverse_action_reasons(results[2], test)

    inter = rm.intersectional_groups(test[GROUP], test["gender"])
    inter_rates = pd.Series(rm.selection_rates(base.approve.astype(int), inter)).sort_values().to_frame("approval_rate")

    # ---- Charts
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(results))
    for i, g in enumerate(["majority", "minority"]):
        ax.bar(x + (i - 0.5) * 0.38, [r.access.loc[g, "approval_rate"] for r in results], 0.38, label=g)
    ax.set_xticks(x, [r.name for r in results])
    ax.set_ylabel("Approval rate")
    ax.set_title("Approval rate by group under each strategy")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "approval_by_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        ax.scatter(r.fairness.loc["disparate_impact_ratio", "value"], r.portfolio["expected_profit"] / 1e6, s=80)
        ax.annotate(r.name, (r.fairness.loc["disparate_impact_ratio", "value"], r.portfolio["expected_profit"] / 1e6),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(0.8, ls="--", color="grey", lw=1)
    ax.text(0.802, ax.get_ylim()[0], " four-fifths rule", color="grey", fontsize=8, va="bottom")
    ax.set_xlabel("Disparate impact ratio (higher = fairer)")
    ax.set_ylabel("Expected profit on test book ($M)")
    ax.set_title("Fairness vs profit trade-off")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fairness_profit_tradeoff.png", dpi=150)
    plt.close(fig)

    # ---- Report
    best = summary.drop(index="group_thresholds")
    report = f"""# Credit Approval: Responsible AI Audit Report

Synthetic portfolio of {n:,} applicants ({len(test):,} in the audit set). Decision rule: approve when predicted
probability of default < 15%. Protected attribute (`{GROUP}`) is never a model input.

## 1. Strategy comparison

{_md(display)}

![approval](approval_by_group.png)
![tradeoff](fairness_profit_tradeoff.png)

## 2. Baseline fairness assessment

{_md(base.fairness, '.4f')}

### Access to credit (baseline)

{_md(base.access, ',.2f')}

### Intersectional approval rates (ethnicity × gender, baseline)

{_md(inter_rates)}

## 3. Transparency

**Proxy scan.** How well each feature alone predicts group membership (strength 0 = no signal, 1 = perfect proxy):

{_md(proxy)}

**Permutation importance** (drop in AUC when the feature is shuffled):

{_md(importance, '.4f')}

**Adverse-action reason codes** (reweighing model, first denied applicants):

{_md(reasons)}

## 4. Robustness & monitoring

{_md(robustness, '.4f')}

* Decision flip rate under 5% input noise: **{flip:.2%}**
* Population Stability Index, income (train → test): **{psi:.4f}** (< 0.1 is stable)

## 5. Impact summary

* The baseline model denies **{int(base.access.loc['minority', 'creditworthy_denied'])}** creditworthy minority applicants
  ({base.access.loc['minority', 'creditworthy_denied_per_1000']:.0f} per 1,000 minority applicants), withholding
  **${base.access.loc['minority', 'credit_withheld_usd']:,.0f}** of credit from people who would have repaid.
* Disparate impact ratio moves from **{summary.loc['baseline', 'disparate_impact']:.2f}** (baseline) to
  **{best['disparate_impact'].max():.2f}** with the best attribute-blind strategy (`{best['disparate_impact'].idxmax()}`),
  with a profit change of
  **{(best['expected_profit_usd'].loc[best['disparate_impact'].idxmax()] - summary.loc['baseline', 'expected_profit_usd']) / abs(summary.loc['baseline', 'expected_profit_usd']):+.1%}**.
  {'It still fails the four-fifths rule, so further work (better features, alternative data, a fairness-constrained model) is needed.' if best['disparate_impact'].max() < 0.8 else 'This passes the four-fifths rule.'}
* `group_thresholds` uses the protected attribute at decision time. That equalises opportunity but can be
  unlawful disparate treatment in US consumer lending (ECOA / Reg B), so it is shown for analysis only.

_All data is synthetic. Figures illustrate methods and are not evidence about any real lender._
"""
    (out / "REPORT.md").write_text(report)
    print(summary.drop(columns="description").round(3).to_string())
    print(f"\nReport written to {out / 'REPORT.md'}")
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.out, a.n, a.seed)
