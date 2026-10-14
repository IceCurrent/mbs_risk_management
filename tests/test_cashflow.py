import numpy as np
import pandas as pd
import pytest

from cashflow import pool_cashflows, scheduled_balance_curve


def _amortise(balance: float, rate: float, term: int) -> np.ndarray:
    r = rate / 12
    payment = balance * r / (1 - (1 + r) ** -term)
    path = [balance]
    for _ in range(term):
        path.append(path[-1] * (1 + r) - payment)
    return np.maximum(np.array(path), 0.0)


def test_scheduled_balance_matches_amortisation_table():
    loans = pd.DataFrame({"current_upb": [300_000.0, 100_000.0], "current_interest_rate": [0.065, 0.06],
                          "months_remaining": [335, 120]})
    sched = scheduled_balance_curve(loans)
    expected = _amortise(300_000, 0.065, 335)
    expected[: 121] += _amortise(100_000, 0.06, 120)
    assert sched.shape == (336,)
    assert np.allclose(sched, expected, atol=1e-6)


def test_principal_returns_face_and_interest_uses_opening_balance(small_pool):
    sched = scheduled_balance_curve(small_pool)
    rng = np.random.default_rng(0)
    smm = rng.uniform(0, 0.02, size=(5, len(sched) - 1))
    flows = pool_cashflows(smm, sched, coupon=0.055)
    principal = flows["scheduled"] + flows["prepaid"]
    assert np.allclose(principal.sum(axis=1), small_pool["current_upb"].sum())
    assert np.allclose(flows["interest"], 0.055 / 12 * flows["opening"])
    assert np.allclose(flows["opening"][:, 1:], flows["closing"][:, :-1])


def test_zero_prepay_gives_scheduled_cash_flows_only(small_pool):
    sched = scheduled_balance_curve(small_pool)
    flows = pool_cashflows(np.zeros((1, len(sched) - 1)), sched, coupon=0.055)
    assert np.allclose(flows["prepaid"], 0.0)
    assert np.allclose(flows["scheduled"][0], sched[:-1] - sched[1:])
