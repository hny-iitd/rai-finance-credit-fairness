import numpy as np
import pytest

from credit_fairness.data import FEATURES, make_lending_data
from credit_fairness.impact import LendingEconomics, access_to_credit, portfolio_outcomes
from credit_fairness.pipeline import adverse_action_reasons, run_strategies, split
from credit_fairness.run import main


@pytest.fixture(scope="module")
def results():
    df = make_lending_data(n=6000, seed=1)
    train, test = split(df)
    return run_strategies(train, test, LendingEconomics()), test


def test_data_schema():
    df = make_lending_data(n=500)
    assert set(FEATURES) <= set(df.columns)
    assert df["default"].isin([0, 1]).all()
    assert df["neighbourhood_risk"].between(0, 1).all()


def test_baseline_is_biased(results):
    res, _ = results
    assert res[0].fairness.loc["disparate_impact_ratio", "value"] < 0.8


def test_mitigation_improves_fairness(results):
    res, _ = results
    by = {r.name: r for r in res}
    base_di = by["baseline"].fairness.loc["disparate_impact_ratio", "value"]
    assert by["drop_proxy"].fairness.loc["disparate_impact_ratio", "value"] > base_di
    assert by["group_thresholds"].fairness.loc["equal_opportunity_difference", "value"] < 0.05


def test_portfolio_outcomes_math():
    df = make_lending_data(n=10)
    df["loan_amount_k"] = 10.0
    df["default"] = [0] * 5 + [1] * 5
    approve = np.ones(10, bool)
    out = portfolio_outcomes(df, approve, LendingEconomics(interest_margin=0.1, loss_given_default=0.5))
    assert out["expected_profit"] == pytest.approx(5 * 1000 - 5 * 5000)
    assert out["book_default_rate"] == 0.5


def test_access_to_credit_counts(results):
    res, test = results
    acc = access_to_credit(test, res[0].approve, "ethnicity_group")
    assert acc["applicants"].sum() == len(test)


def test_reason_codes(results):
    res, test = results
    rc = adverse_action_reasons(res[2], test, n_examples=3)
    assert len(rc) == 3 and rc["principal_reasons"].str.len().gt(0).all()


def test_end_to_end_report(tmp_path):
    main(out_dir=str(tmp_path), n=4000, seed=3)
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "approval_by_group.png").exists()
