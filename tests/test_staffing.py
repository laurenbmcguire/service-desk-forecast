import math

import numpy as np

from deskcast.staffing import agents_required, erlang_c, scheduled_fte, service_level, staffing_matrix


def _erlang_c_direct(n, a):
    top = a**n / math.factorial(n) * n / (n - a)
    return top / (sum(a**k / math.factorial(k) for k in range(n)) + top)


def test_erlang_c_matches_closed_form():
    for n, a in [(11, 10.0), (13, 10.0), (3, 1.2), (8, 5.5)]:
        assert math.isclose(erlang_c(n, a), _erlang_c_direct(n, a), rel_tol=1e-9)
    assert math.isclose(erlang_c(11, 10.0), 0.6821, abs_tol=1e-4)  # textbook value
    assert erlang_c(5, 6.0) == 1.0  # overloaded queue always waits


def test_agents_required_is_minimal_and_monotonic():
    lam, aht = 200.0, 3.0  # 10 Erlangs
    n = agents_required(lam, aht, target_sl=0.8, answer_seconds=20, max_occupancy=1.0)
    assert service_level(n, lam, aht, 20) >= 0.8
    assert service_level(n - 1, lam, aht, 20) < 0.8
    needs = [agents_required(x, 15.0) for x in (0.0, 2.0, 8.0, 20.0, 40.0)]
    assert needs == sorted(needs) and needs[0] == 1
    m = staffing_matrix(np.array([[0.0, 20.0], [40.0, 2.0]]), 15.0)
    assert m.shape == (2, 2) and m[1, 0] > m[0, 1]
    assert math.isclose(scheduled_fte(280, 0.3, 40), 10.0)
