"""Independent arithmetic and exhaustive checks for the mixed CV/HHI objective."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
from threadpoolctl import threadpool_limits

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
import joint_risk_portfolios as rw


def toy(scale=1.):
    mu = np.array([100., 90., 80., 70., 60.])*scale
    h = np.array([22., 25., 18., 12., 17.])*scale
    p = np.array([[.8,.7,0,.1,.4],[.2,.3,.2,0,.3],[0,0,.8,.4,.1],[0,0,0,.5,.2]])
    b = np.array([[2.,3.,1.,-1.,2.],[1.,-.5,2.,1.,-.2]])*scale
    diag = np.array([1.,3.,.5,2.,1.])*scale**2
    return rw.JointRisk(mu, h, b, diag, sparse.csc_matrix(p*mu), count=2)


def selections(obj, retention=.85):
    for pair in combinations(range(obj.n), obj.count):
        z = np.zeros(obj.n); z[list(pair)] = 1
        if rw.feasible(obj, z, retention): yield z


def test_direct_components_and_currency_invariance():
    o = toy()
    np.testing.assert_allclose(o.components(o.baseline), [1, 1])
    z = np.array([1., 0, 1, 0, 0])
    p = np.array([[.8,.7,0,.1,.4],[.2,.3,.2,0,.3],[0,0,.8,.4,.1],[0,0,0,.5,.2]])
    exposure = (p*o.means) @ z
    hhi = (exposure/exposure.sum()) @ (exposure/exposure.sum())
    assert o.components(z)[1]*o.base_hhi == pytest.approx(hhi)
    cov = np.array([[2.,3.,1.,-1.,2.],[1.,-.5,2.,1.,-.2]])
    sd = np.sqrt(np.sum((cov @ z)**2)+np.array([1.,3.,.5,2.,1.]) @ z)
    assert o.components(z)[0]*o.base_cv == pytest.approx(sd/(o.weekly_mean @ z))
    np.testing.assert_allclose(o.components(z), toy(1e6).components(z), rtol=1e-12)


@pytest.mark.parametrize('component', [0, 1])
def test_tangents_are_global_supports(component):
    o = toy(); z = np.array([.4,.7,.2,.1,.9])
    g, b = o.tangent(z, component)
    def numerator(x):
        return np.sqrt(x @ o.qmul(x)) if component == 0 else x @ o.hmul(x)
    assert g @ z+b == pytest.approx(numerator(z))
    for i in range(o.n):
        eps = np.zeros(o.n); eps[i] = 1e-6
        assert g[i] == pytest.approx((numerator(z+eps)-numerator(z-eps))/2e-6, rel=1e-6)
    for x in np.random.default_rng(26).uniform(0, 2, (30, o.n)):
        assert numerator(x) >= g @ x+b-1e-12


@pytest.mark.parametrize('weight', rw.WEIGHTS)
@pytest.mark.parametrize('exact_only', [False, True])
def test_optimizer_matches_exhaustive_search(weight, exact_only):
    o = toy()
    with threadpool_limits(limits=1):
        z, info, _ = rw.optimize(o, weight, .85, exact_only=exact_only,
                                budgets=[dict(cuts=60, nodes=2000)], relative_gap=1e-9)
    expected = min(o.value(z, weight) for z in selections(o))
    assert rw.feasible(o, z, .85)
    assert info['score']/100 == pytest.approx(expected, rel=1e-9)
    assert info['lower_bound']/100 <= expected+1e-8
    assert info['certified']


def test_relaxations_keep_every_improving_integer_portfolio():
    o = toy(); w = .5; r = .85
    dom = rw.domain(o, r)
    pts = list(selections(o, r))
    cuts = [[o.tangent(z, j) for z in pts] for j in range(2)]
    cuts.append([o.binary_variance_tangent(z) for z in pts])
    for builder in [rw.quick_problem, rw.exact_problem]:
        _, _, bounds, con = builder(o, r, w, o.baseline, (0., 0.), dom,
                                   pts if builder is rw.exact_problem else cuts)
        for z in pts:
            if o.value(z, w) > 1: continue
            c = o.components(z)
            x = (np.r_[z, z/(o.d@z), z/(o.u@z), 1/(o.d@z), 1/(o.u@z), c]
                 if builder is rw.exact_problem else np.r_[z, c])
            assert np.all(x >= bounds.lb-1e-10) and np.all(x <= bounds.ub+1e-10)
            v = con.A @ x
            assert np.all(v >= con.lb-1e-8) and np.all(v <= con.ub+1e-8)


def test_domain_bounds_and_swaps():
    o = toy(); lo, hi, a, b = rw.domain(o, .95)
    for z in selections(o, .95):
        assert np.all(lo <= z) and np.all(z <= hi)
        assert a <= o.d @ z <= b
    for w in rw.WEIGHTS:
        z = rw.swaps(o, o.baseline, w, .85)
        assert rw.feasible(o, z, .85) and o.value(z, w) <= 1+1e-12
        # Every possible single exchange is independently recomputed.
        for i in np.flatnonzero(z):
            for j in np.flatnonzero(1-z):
                x = z.copy(); x[i], x[j] = 0, 1
                if rw.feasible(o, x, .85): assert o.value(x, w) >= o.value(z, w)-1e-10


def test_reproducible_and_no_false_certificate():
    o = toy()
    a, ai, _ = rw.optimize(o, .5, .85, budgets=[dict(cuts=1, nodes=0)])
    b, bi, _ = rw.optimize(o, .5, .85, budgets=[dict(cuts=1, nodes=0)])
    np.testing.assert_array_equal(a, b)
    assert ai == bi
    assert ai['lower_bound'] <= ai['score']+1e-8
    if not ai['certified']: assert ai['status'] == 'bounded_best_found'


def test_zero_baseline_risk_rejected():
    with pytest.raises(ValueError, match='zero-CV'):
        rw.JointRisk([1, 2], [1, 2], np.zeros((2, 2)), [0, 0], sparse.diags([1, 2]), 1)


def test_continuous_cv_bound_and_exact_revenue_floor():
    o = toy()
    bound = rw.cv_relaxation_bound(o, .85)
    actual_minimum = min(o.components(z)[0] for z in selections(o))
    assert 0 <= bound <= actual_minimum+1e-7
    assert rw.cv_relaxation_bound(o, 1) == pytest.approx(1, abs=2e-7)
    z, info, _ = rw.optimize(o, .5, 1., budgets=[dict(cuts=5, nodes=100)])
    np.testing.assert_array_equal(z, o.baseline)
    assert info['certified']


@pytest.mark.parametrize('weight', [.25, .5, .75, 1.])
def test_strengthened_bounds_keep_all_true_improvers(weight):
    o = toy(); retention = .85
    pts = list(selections(o, retention))
    lower = np.min([o.components(z) for z in pts], axis=0)-1e-9
    exact_min = min(o.value(z, weight) for z in pts)
    dom = rw.domain(o, retention)
    cuts = [[o.tangent(z, j) for z in pts] for j in range(2)]
    cuts.append([o.binary_variance_tangent(z) for z in pts])
    for best in pts:
        for builder in [rw.quick_problem, rw.exact_problem]:
            _, _, bounds, con = builder(o, retention, weight, best, lower, dom,
                pts if builder is rw.exact_problem else cuts, total_lower_bound=exact_min-1e-9)
            for z in pts:
                if o.value(z, weight) > o.value(best, weight)+1e-12: continue
                c = o.components(z)
                if weight == 1: c[1] = 0
                x = (np.r_[z, z/(o.d@z), z/(o.u@z), 1/(o.d@z), 1/(o.u@z), c]
                     if builder is rw.exact_problem else np.r_[z, c])
                assert np.all(x >= bounds.lb-1e-9) and np.all(x <= bounds.ub+1e-9)
                v = con.A @ x
                assert np.all(v >= con.lb-1e-8) and np.all(v <= con.ub+1e-8)
    z, info, _ = rw.optimize(o, weight, retention, lower_components=lower,
        initial_lower_bound=exact_min-1e-8, budgets=[dict(cuts=20,nodes=200)], relative_gap=1e-9)
    assert info['score']/100 == pytest.approx(exact_min, rel=1e-9)


def test_binary_variance_cut_equals_diagonal_identity():
    o = toy()
    for z in selections(o):
        g, b = o.binary_variance_tangent(z)
        assert g@z+b == pytest.approx(z@o.qmul(z))
        for other in selections(o):
            assert other@o.qmul(other) >= g@other+b-1e-12
