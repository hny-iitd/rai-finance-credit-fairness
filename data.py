"""Synthetic consumer-lending dataset with realistic, *structural* bias.

No real people are in this data. The generator injects two well-documented bias
mechanisms so the fairness metrics have something to find:

1. Historical disadvantage: the protected group has, on average, shorter credit
   histories and lower recorded income (e.g. from past exclusion from credit),
   even though their underlying repayment behaviour is identical.
2. Proxy variables: `neighbourhood_risk` (a postcode-level score) is strongly
   correlated with group membership, a digital echo of historical redlining.

3. Label bias: historical loans to the protected group carried worse terms
   (predatory pricing), inflating their recorded default rate. A model trained on
   these labels learns to penalise the group, mostly through the proxy.

Even so, most protected-group applicants repay, so a model that leans on the
biased signals denies creditworthy applicants from the protected group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "income_k",
    "debt_to_income",
    "credit_history_years",
    "delinquencies_2y",
    "loan_amount_k",
    "employment_years",
    "neighbourhood_risk",
]
SENSITIVE = ["ethnicity_group", "gender", "age_band"]


def make_lending_data(n: int = 20_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    group = rng.choice(["majority", "minority"], n, p=[0.72, 0.28])
    is_min = group == "minority"
    gender = rng.choice(["female", "male"], n, p=[0.49, 0.51])
    age = rng.integers(21, 75, n)
    age_band = pd.cut(age, [20, 30, 45, 60, 80], labels=["21-30", "31-45", "46-60", "61+"]).astype(str)

    # Latent repayment capacity: identical distribution for every group.
    capacity = rng.normal(0, 1, n)

    # Observable features = capacity + structural bias + noise.
    income_k = np.exp(3.9 + 0.35 * capacity - 0.18 * is_min - 0.06 * (gender == "female")
                      + rng.normal(0, 0.25, n))
    debt_to_income = np.clip(0.36 - 0.07 * capacity + 0.03 * is_min + rng.normal(0, 0.08, n), 0.02, 0.9)
    credit_history_years = np.clip(
        (age - 20) * 0.45 + 2.0 * capacity - 3.0 * is_min + rng.normal(0, 3, n), 0, 50
    )
    delinquencies_2y = rng.poisson(np.exp(-0.3 - 0.6 * capacity))
    loan_amount_k = np.clip(rng.lognormal(2.6, 0.5, n), 1, 150)
    employment_years = np.clip((age - 21) * 0.3 + capacity + rng.normal(0, 2.5, n), 0, 45)
    neighbourhood_risk = np.clip(0.35 + 0.30 * is_min - 0.04 * capacity + rng.normal(0, 0.12, n), 0, 1)

    # Recorded default depends on capacity and loan burden, plus a historical-pricing
    # effect: the protected group was offered worse loan terms in the past, which raised
    # their recorded default rate. This is label bias, not a difference in the people.
    logit = (-1.9 - 1.25 * capacity + 0.012 * (loan_amount_k - 15)
             + 0.55 * is_min + rng.normal(0, 0.4, n))
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame(
        {
            "income_k": income_k.round(1),
            "debt_to_income": debt_to_income.round(3),
            "credit_history_years": credit_history_years.round(1),
            "delinquencies_2y": delinquencies_2y,
            "loan_amount_k": loan_amount_k.round(1),
            "employment_years": employment_years.round(1),
            "neighbourhood_risk": neighbourhood_risk.round(3),
            "ethnicity_group": group,
            "gender": gender,
            "age_band": age_band,
            "default": default,
        }
    )
