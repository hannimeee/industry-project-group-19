"""Past-only 50/50 CV / customer-HHI portfolios at five revenue floors.

The two denominators are intentionally different: CV uses the 26-week mean,
whereas customer HHI allocates expanding mean monthly fees. Exact optimization
therefore uses TWO normalization transformations, not a substituted denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from threadpoolctl import threadpool_limits

import cash_flow_variability as cf
import customer_concentration as cc
from abs_momentum_comparison import require
from external_data_sources import save_json

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / 'tables/analysis/joint_risk_portfolios'
OUT = ROOT / 'tables/analysis/joint_risk_portfolios_50_50'
AS_OF = pd.Timestamp('2022-09-30').as_unit('ns')
WEIGHTS = [0., .25, .5, .75, 1.]
RETENTIONS = [1., .995, .99, .98, .95]
WEIGHT = .5
BUDGETS = [dict(cuts=30, nodes=200), dict(cuts=100, nodes=2000)]
REL_GAP = .001


class JointRisk:
    """Dimensionless CV and HHI, each equal to one at the revenue baseline."""
    def __init__(self, means, weekly_mean, factors, diagonal, exposure, count=100):
        self.means = np.asarray(means, float)
        self.weekly_mean = np.asarray(weekly_mean, float)
        self.count = count
        self.n = len(means)
        require(self.n >= count > 0 and np.all(self.means > 0)
                and np.all(self.weekly_mean > 0), 'Positive histories required')
        require(np.isfinite(self.means).all() and np.isfinite(self.weekly_mean).all(), 'Invalid means')
        self.baseline = np.zeros(self.n)
        self.baseline[np.lexsort((np.arange(self.n), -self.means))[:count]] = 1
        self.base_fees = float(self.means @ self.baseline)
        self.base_weekly_mean = float(self.weekly_mean @ self.baseline)
        self.u = self.means / self.base_fees
        self.d = self.weekly_mean / self.base_weekly_mean
        b, diag = np.asarray(factors, float), np.asarray(diagonal, float)
        require(b.shape[1] == self.n and len(diag) == self.n and (diag >= 0).all(), 'Invalid covariance')
        require(np.isfinite(b).all() and np.isfinite(diag).all(), 'Nonfinite covariance')
        sigma2 = float(np.sum((b @ self.baseline)**2) + diag @ self.baseline)
        require(sigma2 > 0, 'Undefined zero-CV baseline normalization')
        self.base_sd = np.sqrt(sigma2)
        self.base_cv = self.base_sd / self.base_weekly_mean
        self.b, self.diag = b / self.base_sd, diag / sigma2
        self.qdiag = np.sum(self.b**2, axis=0) + self.diag
        a = sparse.csc_matrix(exposure)
        require(a.shape[1] == self.n and np.isfinite(a.data).all() and (a.data >= 0).all(), 'Invalid exposure')
        np.testing.assert_allclose(np.asarray(a.sum(axis=0)).ravel(), self.means, rtol=1e-10, atol=1e-8)
        a = a[np.asarray(a.sum(axis=1)).ravel() > 0].tocsc()
        fees = np.asarray(a @ self.baseline).ravel()
        self.base_hhi = float(fees @ fees / self.base_fees**2)
        require(self.base_hhi > 0, 'Undefined HHI normalization')
        self.a = a / (self.base_fees * np.sqrt(self.base_hhi))
        self.hdiag = np.asarray(self.a.multiply(self.a).sum(axis=0)).ravel()
        self.cache = {}

    def qmul(self, z):
        return self.b.T @ (self.b @ z) + self.diag*z

    def hmul(self, z):
        return np.asarray(self.a.T @ (self.a @ z)).ravel()

    def hrow(self, i):
        if i not in self.cache:
            self.cache[i] = (self.a[:, i].T @ self.a).toarray().ravel()
        return self.cache[i]

    def components(self, z):
        q, h = float(z @ self.qmul(z)), float(z @ self.hmul(z))
        return np.array([np.sqrt(max(q, 0)) / (self.d @ z), h / (self.u @ z)**2])

    def value(self, z, weight):
        return float(np.array([weight, 1-weight]) @ self.components(z))

    def tangent(self, z, component):
        """Global numerator support: norm for CV, quadratic for HHI."""
        if component == 0:
            q = self.qmul(z)
            value = np.sqrt(max(float(z @ q), 0))
            return (q/value if value > 0 else np.zeros(self.n)), 0.
        q = self.hmul(z)
        return 2*q, -float(z @ q)

    def binary_variance_tangent(self, z):
        """For binary z, z_i^2=z_i: retain the covariance diagonal exactly."""
        projected = self.b @ z
        return 2*self.b.T @ projected+self.diag, -float(projected @ projected)


def feasible(obj, z, retention):
    return (z.shape == (obj.n,) and np.isin(z, [0, 1]).all() and z.sum() == obj.count
            and obj.means @ z >= retention*obj.base_fees - 1e-6)


def swaps(obj, initial, weight, retention, steps=200):
    """Exact best one-for-one exchanges for the specified mixed objective."""
    z = initial.copy()
    for _ in range(steps):
        inside, outside = np.flatnonzero(z), np.flatnonzero(1-z)
        if not len(outside):
            break
        u, d = float(obj.u @ z), float(obj.d @ z)
        unew = u-obj.u[inside, None]+obj.u[outside][None, :]
        dnew = d-obj.d[inside, None]+obj.d[outside][None, :]
        ok = (unew >= retention-1e-13) & (dnew > 0)
        values = np.zeros(ok.shape)
        if weight > 0:
            q = obj.qmul(z)
            cross = obj.b[:, inside].T @ obj.b[:, outside]
            new = (z @ q + 2*(q[outside][None, :]-q[inside, None])
                   + obj.qdiag[inside, None]+obj.qdiag[outside][None, :]-2*cross)
            values += weight*np.sqrt(np.maximum(new, 0))/dnew
        if weight < 1:
            q = obj.hmul(z)
            cross = np.vstack([obj.hrow(i)[outside] for i in inside])
            new = (z @ q + 2*(q[outside][None, :]-q[inside, None])
                   + obj.hdiag[inside, None]+obj.hdiag[outside][None, :]-2*cross)
            values += (1-weight)*np.maximum(new, 0)/unew**2
        values[~ok] = np.inf
        i, j = np.unravel_index(np.argmin(values), values.shape)
        old = obj.value(z, weight)
        if values[i, j] >= old-1e-12:
            break
        z[inside[i]], z[outside[j]] = 0., 1.
        np.testing.assert_allclose(obj.value(z, weight), values[i, j], rtol=1e-9, atol=1e-12)
        require(feasible(obj, z, retention), 'Swap violated the expected-fee floor')
    return z


def domain(obj, retention):
    """Only provably mandatory merchants are fixed; no arbitrary shortlist."""
    n, k = obj.n, obj.count
    lo, hi = np.zeros(n), np.ones(n)
    order = np.lexsort((np.arange(n), -obj.means))
    if n > k:
        top = order[:k]
        lo[top[obj.base_fees-obj.means[top]+obj.means[order[k]]
               < retention*obj.base_fees-1e-6]] = 1
        rest = order[k:]
        hi[rest[obj.base_fees-obj.means[top[-1]]+obj.means[rest]
                < retention*obj.base_fees-1e-6]] = 0
    bounds = list(zip(lo, hi))
    args = dict(A_ub=-obj.u[None, :], b_ub=[-retention],
                A_eq=np.ones((1, n)), b_eq=[k], bounds=bounds, method='highs',
                options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    low = linprog(obj.d, **args)
    high = linprog(-obj.d, **args)
    require(low.success and high.success, 'Weekly denominator bounds failed')
    # Continuous extrema enclose every binary feasible denominator. Pad solver tolerance.
    dl, du = float(low.fun)-1e-7, float(-high.fun)+1e-7
    require(dl > 0 and du >= dl and np.all(lo <= hi), 'Invalid denominator bounds')
    return lo, hi, dl, du


def cv_relaxation_bound(obj, retention, iterations=80):
    """Lower-bound pure CV by a continuous normalized norm relaxation.

    This is a bound, never a deployable fractional portfolio. Supporting-plane
    LPs under-estimate the convex norm; even an unfinished run is a valid bound.
    """
    n = obj.n
    zlo, zhi, dl, du = domain(obj, retention)
    eye = sparse.eye(n, format='csc')
    zero = sparse.csc_matrix((n, 1))
    fixed = sparse.vstack([
        sparse.hstack([eye, sparse.csc_matrix(-zhi[:, None]), zero]),
        sparse.hstack([-eye, sparse.csc_matrix(zlo[:, None]), zero]),
        sparse.csc_matrix(np.r_[-obj.u, retention, 0.][None, :])], format='csc')
    eq = sparse.csc_matrix(np.vstack([np.r_[obj.d, 0., 0.], np.r_[np.ones(n), -obj.count, 0.]]))
    x = obj.baseline/(obj.d @ obj.baseline)
    planes, rhs, bound = [], [], 0.
    for _ in range(iterations):
        g, b = obj.tangent(x, 0)
        planes.append(np.r_[g, 0., -1.]); rhs.append(-b)
        result = linprog(np.r_[np.zeros(n+1), 1.],
            A_ub=sparse.vstack([fixed, sparse.csc_matrix(np.vstack(planes))], format='csc'),
            b_ub=np.r_[np.zeros(2*n+1), rhs], A_eq=eq, b_eq=[1., 0.],
            bounds=[(0., None)]*n+[(1/du, 1/dl), (0., None)], method='highs',
            options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        require(result.success, 'Continuous CV bound failed: '+result.message)
        bound = max(bound, float(result.fun)-1e-8)
        x = result.x[:n]
        upper = np.sqrt(max(float(x @ obj.qmul(x)), 0))
        require(bound <= upper+1e-6, 'Invalid continuous CV bound')
        if upper-bound < 1e-5: break
    return max(bound, 0.)


def quick_problem(obj, retention, weight, best, lower_components, dom, cuts, total_lower_bound=0.):
    """Necessary conditions using norm/quadratic cuts and McCormick upper envelopes.

    CV numerator = CV_relative * weekly_denominator. For HHI, replace D^2 by
    its upper secant on [retention,1]. Upper McCormick envelopes then relax
    each product. Thus every genuinely improving binary portfolio is retained.
    """
    n = obj.n
    zlo, zhi, dl, du = dom
    w = np.array([weight, 1-weight])
    ub = obj.value(best, weight)
    lower = np.asarray(lower_components, float).copy()
    upper = np.full(2, np.inf)
    for j in range(2):
        if w[j] > 0:
            upper[j] = (ub-w[1-j]*lower[1-j])/w[j]+1e-8
            require(upper[j] >= lower[j]-1e-8, 'Invalid component bounds')
        else:
            lower[j], upper[j] = 0., 0.
    rows = [np.r_[np.ones(n), 0., 0.], np.r_[obj.means, 0., 0.], np.r_[np.zeros(n), w]]
    lows = [obj.count, retention*obj.base_fees, total_lower_bound]
    highs = [obj.count, np.inf, ub+1e-8]
    for j in range(2):
        if w[j] == 0:
            continue
        den = obj.d if j == 0 else (retention+1)*obj.u
        shift = 0. if j == 0 else -retention
        a, b = (dl, du) if j == 0 else (retention**2, 1.)
        for gradient, intercept in cuts[j]:
            for score_bound, den_bound in [(lower[j], b), (upper[j], a)]:
                row = np.r_[gradient-score_bound*den, 0., 0.]
                row[n+j] = -den_bound
                rows.append(row)
                lows.append(-np.inf)
                highs.append(-intercept+score_bound*(shift-den_bound))
    if weight > 0 and len(cuts) > 2:
        # On a <= CV <= b, CV^2 <= (a+b)*CV-a*b. This chord keeps the
        # ORIGINAL CV objective but permits much stronger binary variance cuts.
        # Both a and b are valid for every portfolio able to beat the incumbent.
        a, b = lower[0], upper[0]
        for gradient, intercept in cuts[2]:
            for gamma_bound, squared_den_bound in [(a*a, du*du), (b*b, dl*dl)]:
                row = np.r_[gradient-gamma_bound*(dl+du)*obj.d, 0., 0.]
                row[n] = -squared_den_bound*(a+b)
                rows.append(row); lows.append(-np.inf)
                highs.append(-intercept-squared_den_bound*a*b
                             -gamma_bound*dl*du-squared_den_bound*gamma_bound)
    return (np.r_[np.zeros(n), w], np.r_[np.ones(n), 0., 0.],
            Bounds(np.r_[zlo, lower], np.r_[zhi, upper]),
            LinearConstraint(sparse.csc_matrix(np.vstack(rows)), lows, highs))


def exact_problem(obj, retention, weight, best, lower_components, dom, points, total_lower_bound=0.):
    """Exact binary links for x=z/(d'z), v=z/(u'z); convex norm + quadratic.

    Variables: z[n], x[n], v[n], t, s, eta_cv, eta_hhi.
    Each support plane is globally valid for its convex normalized objective.
    """
    n = obj.n
    zlo, zhi, dl, du = dom
    size = 3*n+4
    t_index, s_index, cv_index, h_index = 3*n, 3*n+1, 3*n+2, 3*n+3
    rr, ccidx, vv, lows, highs = [], [], [], [], []
    def row(indices, values, low, high):
        number = len(lows)
        rr.extend([number]*len(indices)); ccidx.extend(indices); vv.extend(values)
        lows.append(low); highs.append(high)
    ids = np.arange(n)
    row(ids, np.ones(n), obj.count, obj.count)
    row(ids, obj.means, retention*obj.base_fees, np.inf)
    row(ids+n, obj.d, 1., 1.)
    row(ids+2*n, obj.u, 1., 1.)
    for offset, ti, lo, hi in [(n, t_index, 1/du, 1/dl), (2*n, s_index, 1., 1/retention)]:
        for i in range(n):
            row([i, i+offset], [-lo, 1.], 0., np.inf)
            row([i, i+offset], [-hi, 1.], -np.inf, 0.)
            row([i, i+offset, ti], [-hi, 1., -1.], -hi, np.inf)
            row([i, i+offset, ti], [-lo, 1., -1.], -np.inf, -lo)
    for z in points:
        if weight > 0:
            x = z/(obj.d @ z)
            g, b = obj.tangent(x, 0)
            row(np.r_[ids+n, cv_index], np.r_[g, -1.], -np.inf, -b)
            a = lower_components[0]
            upper_cv = (obj.value(best, weight)-(1-weight)*lower_components[1])/weight+1e-8
            if a+upper_cv > 0:
                # Perspective of the nonnegative diagonal: d_i*x_i^2/z_i.
                # Its support is valid even at z_i=x_i=0 (choose common t).
                t = 1/(obj.d @ z)
                gz = -obj.diag*t*t
                gx = 2*obj.b.T @ (obj.b @ x)+2*obj.diag*t
                value = float(x @ obj.qmul(x))
                denom = a+upper_cv
                row(np.r_[ids, ids+n, cv_index], np.r_[gz/denom, gx/denom, -1.], -np.inf,
                    float((gz@z+gx@x-value-a*upper_cv)/denom))
        if weight < 1:
            v = z/(obj.u @ z)
            g, b = obj.tangent(v, 1)
            row(np.r_[ids+2*n, h_index], np.r_[g, -1.], -np.inf, -b)
    w = np.array([weight, 1-weight])
    upper = obj.value(best, weight)+1e-8
    row([cv_index, h_index], w, total_lower_bound, upper)
    lower = np.asarray(lower_components, float).copy()
    if weight == 0: lower[0] = 0
    if weight == 1: lower[1] = 0
    bounds = Bounds(np.r_[zlo, np.zeros(2*n), 1/du, 1., lower],
                    np.r_[zhi, np.full(n, 1/dl), np.full(n, 1/retention), 1/dl, 1/retention, np.inf, np.inf])
    matrix = sparse.coo_matrix((vv, (rr, ccidx)), shape=(len(lows), size)).tocsc()
    return np.r_[np.zeros(size-2), w], np.r_[np.ones(n), np.zeros(size-n)], bounds, LinearConstraint(matrix, lows, highs)


def optimize(obj, weight, retention, seeds=(), lower_components=(0., 0.),
             budgets=None, relative_gap=REL_GAP, exact_only=False, progress=None,
             initial_lower_bound=0.):
    require(0 <= weight <= 1 and 0 < retention <= 1, 'Invalid preference or floor')
    budgets = BUDGETS if budgets is None else budgets
    candidates = [obj.baseline]+[np.asarray(z, float) for z in seeds if feasible(obj, np.asarray(z), retention)]
    best = min(candidates, key=lambda z: (obj.value(z, weight), tuple(np.flatnonzero(z)))).copy()
    best = swaps(obj, best, weight, retention)
    dom = domain(obj, retention)
    cuts, points, seen = [[], [], []], [], set()
    def add(z):
        key = tuple(np.flatnonzero(z))
        if key in seen: return False
        seen.add(key); points.append(z.copy())
        for j in range(2): cuts[j].append(obj.tangent(z, j))
        cuts[2].append(obj.binary_variance_tangent(z))
        return True
    for z in candidates+[best]: add(z)
    w = np.array([weight, 1-weight])
    lb = max(float(w @ np.asarray(lower_components)), initial_lower_bound)
    logs, force_exact = [], exact_only
    if progress is not None:
        ub = obj.value(best, weight)
        progress(best, dict(tier=0, iteration=-1, exact_normalization=False, status=-1,
                           nodes=0, score=100*ub, lower_bound=100*lb, gap=(ub-lb)/ub, seconds=0.))
    for tier, budget in enumerate(budgets, 1):
        for iteration in range(budget['cuts']):
            ub = obj.value(best, weight)
            if ub-lb <= relative_gap*ub or ub-lb <= 1e-6:
                break
            is_exact = force_exact or iteration == budget['cuts']-1
            builder = exact_problem if is_exact else quick_problem
            problem = builder(obj, retention, weight, best, lower_components, dom,
                              points if is_exact else cuts, total_lower_bound=lb)
            start = time.perf_counter()
            result = milp(problem[0], integrality=problem[1], bounds=problem[2], constraints=problem[3],
                          options=dict(node_limit=budget['nodes'], mip_rel_gap=min(1e-4, relative_gap/10),
                                       presolve=True, time_limit=60.))
            require(result.status in [0, 1] or 'Solution limit reached' in result.message, 'Joint MILP failed: '+result.message)
            bound = getattr(result, 'mip_dual_bound', None)
            if bound is not None and np.isfinite(bound): lb = max(lb, float(bound))
            candidate = None
            if result.x is not None:
                candidate = np.rint(result.x[:obj.n])
                require(np.max(np.abs(candidate-result.x[:obj.n])) < 1e-5 and feasible(obj, candidate, retention), 'Invalid joint incumbent')
                polished = swaps(obj, candidate, weight, retention)
                if (obj.value(polished, weight), tuple(np.flatnonzero(polished))) < (obj.value(best, weight), tuple(np.flatnonzero(best))):
                    best = polished
            ub = obj.value(best, weight)
            require(lb <= ub+1e-6, 'Invalid global joint lower bound')
            lb = min(lb, ub)
            record = dict(tier=tier, iteration=iteration, exact_normalization=is_exact, status=result.status,
                          nodes=getattr(result, 'mip_node_count', None), score=100*ub, lower_bound=100*lb,
                          gap=(ub-lb)/ub, seconds=time.perf_counter()-start)
            logs.append(record)
            if progress is not None: progress(best, record)
            added = add(best)
            if candidate is not None: added = add(candidate) or added
            if not added:
                if is_exact: break
                force_exact = True
        ub = obj.value(best, weight)
        if ub-lb <= relative_gap*ub or ub-lb <= 1e-6: break
    require(feasible(obj, best, retention), 'Final selection invalid')
    ub = obj.value(best, weight)
    certified = ub-lb <= relative_gap*ub or ub-lb <= 1e-6
    return best, dict(score=100*ub, lower_bound=100*lb, gap=(ub-lb)/ub,
                      certified=bool(certified), iterations=len(logs),
                      status='certified_gap' if certified else 'bounded_best_found'), pd.DataFrame(logs)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def previous_inventory():
    return {str(p.relative_to(OUT.parent)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(OUT.parent.rglob('*')) if p.is_file() and OUT not in p.parents}


def table(frame, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory/(name+'.parquet'), compression=None, index=False)
    frame.to_csv(directory/(name+'.csv'), index=False)


def load_snapshot():
    """Verify sources and rederive the features; only the chosen past window enters."""
    daily, monthly, merchants, audit = cf.load_audited_inputs()
    weekly = cf.weekly_panel(daily, merchants)
    del daily
    fresh, covariance = cf.features_at_cutoff(weekly, monthly, merchants, AS_OF)
    saved = pd.read_parquet(cf.OUT/'merchant_features.parquet')
    saved = saved.loc[saved.as_of.eq(AS_OF)].sort_values('merchant_abn').reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, saved, check_exact=False, rtol=1e-12, atol=1e-8)
    directory = cc.group_directory(AS_OF, '26w')
    freeze = json.loads((cc.OUT/'selection_freeze.json').read_text())
    exposure_path = directory/'exposure.parquet'
    for path in [exposure_path, cc.OUT/'selections.parquet', cc.OUT/'unscored_estimates.parquet']:
        require(file_hash(path) == freeze['files'][str(path.relative_to(cc.OUT))], 'Frozen concentration input changed')
    identities = pd.read_parquet(cc.OUT/'identities.parquet')
    require(file_hash(cc.OUT/'identities.parquet') == freeze['prepared_identities'], 'Customer identities changed')
    history = cc.load_exposure(directory, fresh.merchant_abn.to_numpy(), identities.consumer_id.to_numpy())
    dates = cf.risk_window(AS_OF)
    wide = weekly.loc[weekly.week_end.isin(dates)].pivot(index='week_end', columns='merchant_abn', values='gross_fees')
    wide = wide.reindex(index=dates, columns=fresh.merchant_abn)
    require(wide.notna().all().all(), 'Missing required week')
    np.testing.assert_allclose(np.asarray(history.fees.sum(axis=0)).ravel(), wide.sum().to_numpy(), rtol=1e-12, atol=1e-7)
    f = fresh.loc[fresh.eligible].reset_index(drop=True)
    ix = pd.Index(covariance.ids).get_indexer(f.merchant_abn)
    expected = cc.expected_exposure(history.subset(f.merchant_abn.to_numpy()), f.expected_monthly_fees.to_numpy())
    obj = JointRisk(f.expected_monthly_fees.to_numpy(), f.weekly_mean_fees.to_numpy(),
        np.sqrt(1-covariance.alpha)*covariance.centered[:, ix]/np.sqrt(len(dates)-1),
        covariance.alpha*covariance.variances[ix], expected.fees)
    x = wide.reindex(columns=f.merchant_abn).to_numpy()
    inputs = {str(p.relative_to(ROOT)): file_hash(p) for p in [
        exposure_path, cc.OUT/'identities.parquet', cc.OUT/'selections.parquet', cc.OUT/'unscored_estimates.parquet',
        cf.OUT/'merchant_features.parquet', cf.DAILY_OUT/'merchant_day.parquet',
        cf.DAILY_OUT/'daily_audit.json', cc.frontier.OUT/'selections.parquet']}
    inputs['archive_signatures'] = audit['archives']
    return obj, f, x, covariance.alpha, inputs


def portfolio_record(obj, f, x, z, retention, info):
    c = obj.components(z)
    weekly_fees = x @ z
    selected = f.loc[z.astype(bool)].sort_values(['expected_monthly_fees', 'merchant_abn'], ascending=[False, True]).copy()
    selected['retention'] = retention
    selected['rank_by_expected_fees'] = np.arange(1, obj.count+1)
    selected['in_revenue_baseline'] = selected.merchant_abn.isin(f.loc[obj.baseline.astype(bool), 'merchant_abn'])
    selected['variability_weight'], selected['concentration_weight'] = WEIGHT, 1-WEIGHT
    summary = dict(as_of=AS_OF, window='26w', retention=retention, merchants=int(z.sum()),
        variability_weight=WEIGHT, concentration_weight=1-WEIGHT,
        expected_monthly_fees=float(obj.means @ z), baseline_expected_monthly_fees=obj.base_fees,
        expected_fee_retention_pct=100*float(obj.u @ z),
        historical_weekly_mean=float(weekly_fees.mean()), sample_weekly_sd=float(weekly_fees.std(ddof=1)),
        estimated_weekly_sd=float(c[0]*obj.base_cv*(obj.weekly_mean @ z)),
        estimated_cv=float(c[0]*obj.base_cv), baseline_cv=obj.base_cv,
        estimated_hhi=float(c[1]*obj.base_hhi), baseline_hhi=obj.base_hhi,
        effective_customers=float(1/(c[1]*obj.base_hhi)),
        cv_relative=float(c[0]), hhi_relative=float(c[1]),
        cv_reduction_pct=100*(1-float(c[0])), hhi_reduction_pct=100*(1-float(c[1])),
        replaced_vs_baseline=int(obj.count-np.sum(z*obj.baseline)), **info)
    np.testing.assert_allclose(summary['score'], 50*sum(c), rtol=1e-12, atol=1e-9)
    return selected, summary


def predecessor_evidence(obj, f, inputs, alpha):
    """Reuse preserved solutions/bounds only for the identical inputs and score."""
    seeds, bounds, components, paths = [], {}, {}, []
    if not (PREDECESSOR/'solver_revision_1_stop.json').exists():
        return seeds, bounds, components, {}
    protocol = json.loads((PREDECESSOR/'protocol.json').read_text())
    require(protocol['inputs'] == inputs and protocol['as_of'] == str(AS_OF.date())
            and protocol['variability_weight'] == WEIGHT and protocol['concentration_weight'] == 1-WEIGHT
            and protocol['covariance_shrinkage'] == alpha, 'Predecessor inputs or objective changed')
    require(file_hash(PREDECESSOR/'solver_revision_1.py') == protocol['mathematical_module_sha256'],
            'Predecessor solver source not preserved')
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()
    paths = [PREDECESSOR/'protocol.json', PREDECESSOR/'solver_revision_1.py', PREDECESSOR/'solver_revision_1_stop.json']
    for r in RETENTIONS:
        d = PREDECESSOR/f'retention_{r:.3f}'
        if (d/'complete.json').exists():
            marker = json.loads((d/'complete.json').read_text())
            require(marker['signature'] == signature, 'Predecessor completed-case signature changed')
            for name, digest in marker['files'].items():
                require(file_hash(d/name) == digest, 'Predecessor completed case changed')
                paths.append(d/name)
            saved = pd.read_parquet(d/'selections.parquet')
            metrics = pd.read_parquet(d/'metrics.parquet').iloc[0]
            value, lower = float(metrics.score), float(metrics.lower_bound)
            paths.append(d/'complete.json')
        elif (d/'progress.json').exists():
            progress = json.loads((d/'progress.json').read_text())
            require(progress['signature'] == signature, 'Predecessor partial-case signature changed')
            saved = pd.read_parquet(d/'incumbent.parquet')
            value, lower = progress['score'], progress['lower_bound']
            paths += [d/'progress.json', d/'incumbent.parquet']
        else:
            continue
        z = f.merchant_abn.isin(saved.merchant_abn).to_numpy(float)
        require(feasible(obj, z, r), 'Predecessor portfolio no longer feasible')
        np.testing.assert_allclose(obj.value(z, WEIGHT)*100, value, rtol=1e-12, atol=1e-8)
        require(lower <= value+1e-7, 'Invalid predecessor global bound')
        seeds.append(z); bounds[r] = max(0., lower/100-1e-10)
        if (d/'component_bounds.json').exists():
            c = json.loads((d/'component_bounds.json').read_text())
            require(c['signature'] == signature, 'Predecessor component-bound signature changed')
            components[r] = [c['cv_continuous_lower_bound'], c['hhi_previous_certified_global_bound']]
            paths.append(d/'component_bounds.json')
    hashes = {str(p.relative_to(ROOT)):file_hash(p) for p in paths}
    return seeds, bounds, components, hashes


def run():
    import os
    from customer_concentration_resources import apply_worker_limits, system_memory
    limits = apply_worker_limits() if os.name == 'nt' else dict(maximum_numerical_workers=1)
    OUT.mkdir(parents=True, exist_ok=True)
    old = previous_inventory()
    with threadpool_limits(limits=1):
        print('Verifying audited histories and customer exposures.', flush=True)
        obj, f, x, alpha, inputs = load_snapshot()
        previous_seeds, previous_bounds, previous_components, previous_hashes = predecessor_evidence(obj, f, inputs, alpha)
        protocol = dict(version=2, as_of=str(AS_OF.date()), window='26w', retentions=RETENTIONS,
            variability_weight=WEIGHT, concentration_weight=1-WEIGHT,
            score='100 * (0.5 * shrunk_CV / baseline_CV + 0.5 * expected_fee_HHI / baseline_HHI)',
            expected_return='Expanding complete-month mean monthly gross fees',
            weeks_start=str(cf.risk_window(AS_OF)[0].date()-pd.Timedelta(days=6)),
            weeks_end=str(cf.risk_window(AS_OF)[-1].date()), eligible_merchants=obj.n,
            covariance_shrinkage=alpha, budgets=BUDGETS, milp_call_time_limit_seconds=60,
            relative_gap=REL_GAP, absolute_gap_baseline_fraction=1e-6,
            one_for_one_exchange_limit=200, tie_order='ascending numeric ABN',
            future_outcomes_used=False, snapshot_only=True, inputs=inputs,
            predecessor_evidence=previous_hashes,
            strengthened_bounds='binary covariance diagonal, conditional CV chord, perspective cuts, propagated global score bound',
            mathematical_module_sha256=file_hash(__file__), resource_limits=limits,
            versions={n: importlib.metadata.version(n) for n in ['numpy','pandas','scipy','pyarrow','scikit-learn']})
        signature = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()
        if (OUT/'protocol.json').exists():
            require(json.loads((OUT/'protocol.json').read_text()) == protocol, 'Protocol changed; do not mix runs')
        else:
            save_json(OUT/'protocol.json', protocol)
        source_selections = pd.concat([pd.read_parquet(cc.OUT/'selections.parquet'),
                                      pd.read_parquet(cc.frontier.OUT/'selections.parquet')], ignore_index=True)
        source_selections = source_selections.loc[source_selections.as_of.eq(AS_OF) & source_selections.window.eq('26w')]
        seeds = [f.merchant_abn.isin(g.merchant_abn).to_numpy(float)
                 for _, g in source_selections.groupby(['method','retention'])]
        seeds += previous_seeds
        old_est = pd.read_parquet(cc.OUT/'unscored_estimates.parquet')
        old_est = old_est.loc[old_est.as_of.eq(AS_OF) & old_est.window.eq('26w') & old_est.method.eq('shared')]
        all_s, all_e, all_logs = [], [], []
        for retention in RETENTIONS:
            directory = OUT/f'retention_{retention:.3f}'
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory/'complete.json'
            if marker.exists():
                record = json.loads(marker.read_text())
                require(record['signature'] == signature, 'Case provenance changed')
                for name, digest in record['files'].items(): require(file_hash(directory/name) == digest, 'Case file changed')
                selected, row = pd.read_parquet(directory/'selections.parquet'), pd.read_parquet(directory/'metrics.parquet')
                z = f.merchant_abn.isin(selected.merchant_abn).to_numpy(float)
                require(feasible(obj, z, retention), 'Saved case infeasible')
                np.testing.assert_allclose(obj.value(z, WEIGHT)*100, row.score.iloc[0], rtol=1e-12)
                print(f'Reused completed {100*retention:g}% case.', flush=True)
            else:
                if os.name == 'nt':
                    memory = system_memory()
                    require(memory['available_physical'] >= 2*2**30 and memory['available_commit'] >= 4*2**30,
                            'System memory low; completed cases are saved for resume')
                ranked = np.sort(obj.means)[::-1]
                if retention == 1 and ranked[obj.count-1]-ranked[obj.count] > 1e-6:
                    z = obj.baseline.copy()
                    info = dict(score=100., lower_bound=100., gap=0., certified=True, iterations=0,
                                status='unique_maximum_revenue_portfolio')
                    log = pd.DataFrame(columns=['tier','iteration','score','lower_bound','gap','seconds'])
                else:
                    print(f'Solving {100*retention:g}% floor at fixed 50/50 weights.', flush=True)
                    if retention in previous_components:
                        lower = previous_components[retention]
                    else:
                        cv_bound = cv_relaxation_bound(obj, retention)
                        saved_hhi_bound = float(old_est.loc[old_est.retention.eq(retention), 'lower_bound'].iloc[0])/obj.base_hhi if retention < 1 else 0.
                        lower = [cv_bound, max(0., saved_hhi_bound-1e-8)]
                    save_json(directory/'component_bounds.json', dict(cv_continuous_lower_bound=lower[0],
                        hhi_previous_certified_global_bound=lower[1], signature=signature))
                    def progress(current, record):
                        pd.DataFrame({'merchant_abn':f.loc[current.astype(bool),'merchant_abn']}).to_parquet(
                            directory/'incumbent.parquet', index=False, compression=None)
                        save_json(directory/'progress.json', dict(signature=signature, **record))
                        print(f"  tier {record['tier']} step {record['iteration']}: score {record['score']:.5f}; "
                              f"gap {100*record['gap']:.4f}%; last step {record['seconds']:.1f}s", flush=True)
                    warm = seeds.copy()
                    if (directory/'progress.json').exists():
                        previous = json.loads((directory/'progress.json').read_text())
                        require(previous['signature'] == signature, 'Interrupted case settings changed')
                        incumbent = pd.read_parquet(directory/'incumbent.parquet')
                        warm.append(f.merchant_abn.isin(incumbent.merchant_abn).to_numpy(float))
                    z, info, log = optimize(obj, WEIGHT, retention, seeds=warm, lower_components=lower,
                        initial_lower_bound=previous_bounds.get(retention, 0.), progress=progress)
                selected, summary = portfolio_record(obj, f, x, z, retention, info)
                row = pd.DataFrame([summary])
                table(selected, directory, 'selections'); table(row, directory, 'metrics'); table(log, directory, 'solver_log')
                files = {p.name: file_hash(p) for p in directory.iterdir() if p.suffix in ['.parquet','.csv']}
                save_json(marker, dict(signature=signature, files=files))
                print(f"Completed {100*retention:g}%: score {info['score']:.5f}; certified={info['certified']}", flush=True)
            seeds.append(z)
            all_s.append(selected); all_e.append(row)
            all_logs.append(pd.read_parquet(directory/'solver_log.parquet').assign(retention=retention))
        selection, metrics = pd.concat(all_s, ignore_index=True), pd.concat(all_e, ignore_index=True)
        table(selection, OUT, 'selections'); table(metrics, OUT, 'metrics')
        table(pd.concat(all_logs, ignore_index=True), OUT, 'solver_log')
        sets = {r:set(g.merchant_abn) for r,g in selection.groupby('retention')}
        overlaps = pd.DataFrame([dict(retention_a=a, retention_b=b, common_merchants=len(sets[a]&sets[b]),
                                    replaced_merchants=100-len(sets[a]&sets[b])) for a in RETENTIONS for b in RETENTIONS])
        table(overlaps, OUT, 'overlap')
        table(selection.groupby(['retention','category'], as_index=False).agg(
            merchants=('merchant_abn','size'), expected_monthly_fees=('expected_monthly_fees','sum')), OUT, 'category_composition')
        require(previous_inventory() == old, 'Earlier experiments were modified')
        save_json(OUT/'acceptance_checks.json', dict(five_portfolios=len(metrics)==5,
            unique_eligible_merchants=bool(selection.groupby('retention').merchant_abn.nunique().eq(100).all() and selection.eligible.all()),
            expected_fee_floors=bool(metrics.expected_fee_retention_pct.ge(metrics.retention*100-1e-8).all()),
            covariance_and_customer_reconciliation=True, fixed_weight=.5, no_future_outcomes=True,
            global_bounds_valid=bool(metrics.lower_bound.le(metrics.score+1e-6).all()),
            certified_portfolios=int(metrics.certified.sum()), previous_outputs_unchanged=True))
        save_json(OUT/'selection_freeze.json', dict(signature=signature, files={name:file_hash(OUT/name)
            for name in ['protocol.json','selections.parquet','metrics.parquet','overlap.parquet','solver_log.parquet']}))
    report()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = pd.read_parquet(OUT/'metrics.parquet').sort_values('retention', ascending=False)
    overlap = pd.read_parquet(OUT/'overlap.parquet')
    labels = [f'{100*r:g}%' for r in RETENTIONS]
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rev = metrics.expected_fee_retention_pct.to_numpy()
    axes[0].plot(rev, metrics.cv_relative*100, 'o-', label='Weekly CV')
    axes[0].plot(rev, metrics.hhi_relative*100, 'o-', label='Customer HHI')
    axes[0].set(xlabel='Expected monthly fees retained (%)', ylabel='Risk / revenue baseline (%)', title='Two risk components')
    axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].errorbar(rev, metrics.score, yerr=[metrics.score-metrics.lower_bound, np.zeros(len(metrics))], fmt='o-', capsize=4)
    axes[1].set(xlabel='Expected monthly fees retained (%)', ylabel='50/50 score (baseline = 100)', title='Best feasible scores and lower bounds')
    axes[1].grid(alpha=.2)
    mat = overlap.pivot(index='retention_a', columns='retention_b', values='replaced_merchants').reindex(index=RETENTIONS, columns=RETENTIONS)
    axes[2].imshow(mat, cmap='Blues', vmin=0)
    axes[2].set(xticks=range(5), yticks=range(5), xticklabels=labels, yticklabels=labels, title='Merchants replaced between portfolios')
    for i in range(5):
        for j in range(5): axes[2].text(j, i, str(mat.iloc[i,j]), ha='center', va='center',
                                     color='white' if mat.iloc[i,j] > .55*mat.to_numpy().max() else 'black')
    fig.suptitle('Fixed 50/50 CV + customer concentration | 30 September 2022 | 26 historical weeks')
    fig.tight_layout(); fig.savefig(OUT/'portfolio_comparison.png', dpi=170); plt.close(fig)
    lines = ['# Fixed 50/50 joint-risk portfolios', '',
        'Five selections of exactly 100 eligible merchants at 30 September 2022. Risk history: 28 March–25 September 2022.',
        'Expected return remains the expanding historical mean monthly gross fees. No future outcomes are used or scored.', '',
        'Score = 100 × [0.5 × (regularized weekly CV / revenue-baseline CV) + 0.5 × (expected-fee customer HHI / revenue-baseline HHI)].',
        'CV uses weekly standard deviation divided by the 26-week mean. HHI aggregates each customer across selected merchants.',
        'The standard-deviation numerator uses the existing raw-fee covariance shrinkage. This is a preference index, not a probability of loss or evidence of downside protection.', '',
        '| Minimum expected fees | Expected monthly fees (AUD) | Fees retained | CV reduction | HHI reduction | Score | Replacements vs baseline | Gap | Certified |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in metrics.itertuples():
        lines.append(f'| {100*r.retention:g}% | {r.expected_monthly_fees:,.2f} | {r.expected_fee_retention_pct:.3f}% | {r.cv_reduction_pct:+.3f}% | {r.hhi_reduction_pct:+.3f}% | {r.score:.4f} | {r.replaced_vs_baseline} | {100*r.gap:.4f}% | {r.certified} |')
    lines += ['', 'The 100% floor admits only the revenue-maximizing selection when the 100th and 101st expected fees differ. This is verified, not assumed.',
        'Other cases use exact one-for-one exchanges, globally valid norm/quadratic support planes, product relaxations, and an exact two-denominator normalized formulation. MILP dual bounds quantify optimization uncertainty.',
        'Certification requires at most a 0.1% relative score gap or an absolute gap of 0.0001 score points. An uncertified result is the best feasible solution found, not a proven optimum.',
        'Component bounds reuse the original global HHI bounds and a newly calculated continuous CV relaxation. Earlier single-objective portfolios are feasible starting candidates only.',
        'Both risk components and revenue are reported separately because one component can worsen while the mixed score improves.', '',
        '## Reproduce', '', '`python code/joint_risk_portfolios.py`',
        '`python -m pytest -p no:capture code/tests/test_joint_risk_portfolios.py -q`', '',
        'Completed cases are fingerprinted and reused on resume. There is one numerical worker, a 4 GiB committed-memory cap, a 20% CPU cap, and one BLAS thread. No prior experiment is modified.',
        'The abandoned 20-case weight-sensitivity grid was not run. These are five portfolios at one fixed weight.',
        'Solver revision 1 is preserved separately. Revision 2 strengthens bounds using the exact binary covariance diagonal and a valid chord bound for CV; the score and feasible portfolios are unchanged. Preserved solutions and bounds are reused only after input, source and objective checks.',
        'September has no complete future quarter in the audited data. These estimated trade-offs are not a new backtest, statistical significance claim, or calibrated loss forecast.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    report() if args.report_only else run()
