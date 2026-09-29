"""Portable CV/HHI optimization mathematics, extracted from the frozen study.

No dataset paths, dates or previously computed bounds are imported here.
"""
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

BUDGETS = [dict(cuts=30, nodes=200)]
REL_GAP = .001


def require(condition, message):
    """Reject invalid inputs rather than silently changing the objective."""
    if not condition:
        raise ValueError(message)


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


def component_relaxation_bound(obj, retention, component=0, iterations=20):
    """Lower-bound CV or HHI with continuous normalized supporting-plane LPs.

    This is a bound, never a deployable fractional portfolio. Supporting-plane
    LPs under-estimate the convex norm; even an unfinished run is a valid bound.
    """
    n = obj.n
    zlo, zhi, dl, du = domain(obj, retention)
    require(component in [0,1], 'Unknown risk component')
    denominator = obj.d if component == 0 else obj.u
    if component == 1:
        dl, du = retention, 1.
    eye = sparse.eye(n, format='csc')
    zero = sparse.csc_matrix((n, 1))
    fixed = sparse.vstack([
        sparse.hstack([eye, sparse.csc_matrix(-zhi[:, None]), zero]),
        sparse.hstack([-eye, sparse.csc_matrix(zlo[:, None]), zero]),
        sparse.csc_matrix(np.r_[-obj.u, retention, 0.][None, :])], format='csc')
    eq = sparse.csc_matrix(np.vstack([np.r_[denominator, 0., 0.], np.r_[np.ones(n), -obj.count, 0.]]))
    x = obj.baseline/(denominator @ obj.baseline)
    planes, rhs, bound = [], [], 0.
    for _ in range(iterations):
        g, b = obj.tangent(x, component)
        planes.append(np.r_[g, 0., -1.]); rhs.append(-b)
        result = linprog(np.r_[np.zeros(n+1), 1.],
            A_ub=sparse.vstack([fixed, sparse.csc_matrix(np.vstack(planes))], format='csc'),
            b_ub=np.r_[np.zeros(2*n+1), rhs], A_eq=eq, b_eq=[1., 0.],
            bounds=[(0., None)]*n+[(1/du, 1/dl), (0., None)], method='highs',
            options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        require(result.success, 'Continuous CV bound failed: '+result.message)
        bound = max(bound, float(result.fun)-1e-8)
        x = result.x[:n]
        upper = np.sqrt(max(float(x @ obj.qmul(x)), 0)) if component == 0 else float(x @ obj.hmul(x))
        require(bound <= upper+1e-6, 'Invalid continuous CV bound')
        if upper-bound < 1e-5: break
    return max(bound, 0.)


def cv_relaxation_bound(obj, retention, iterations=80):
    """Compatibility name for the independently tested CV relaxation."""
    return component_relaxation_bound(obj,retention,component=0,iterations=iterations)


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
             initial_lower_bound=0., time_limit=30., max_seconds=300.,
             patience=8, swap_steps=200):
    """Return a feasible selection and genuine bound, with explicit budget stops.

    The time budget is checked between operations; a running swap pass or solver
    call must finish. It is not a hard wall-clock or global-optimality guarantee.
    """
    require(0 <= weight <= 1 and 0 < retention <= 1, 'Invalid preference or floor')
    require(time_limit > 0 and max_seconds > 0 and patience >= 1
            and swap_steps >= 0 and relative_gap >= 0, 'Invalid search budget')
    started = time.perf_counter()
    stopped, stale = None, 0
    budgets = BUDGETS if budgets is None else budgets
    require(all(b['cuts'] >= 1 and b['nodes'] >= 0 for b in budgets), 'Invalid search tiers')
    candidates = [obj.baseline]+[np.asarray(z, float) for z in seeds if feasible(obj, np.asarray(z), retention)]
    best = min(candidates, key=lambda z: (obj.value(z, weight), tuple(np.flatnonzero(z)))).copy()
    best = swaps(obj, best, weight, retention, steps=swap_steps)
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
                           nodes=0, score=100*ub, lower_bound=100*lb, gap=(ub-lb)/max(ub,1e-15), seconds=0.))
    for tier, budget in enumerate(budgets, 1):
        for iteration in range(budget['cuts']):
            ub = obj.value(best, weight)
            if ub-lb <= relative_gap*ub or ub-lb <= 1e-6:
                break
            remaining = max_seconds-(time.perf_counter()-started)
            if remaining <= 0:
                stopped = 'time_budget'
                break
            previous_ub, previous_lb = ub, lb
            is_exact = force_exact or iteration == budget['cuts']-1
            builder = exact_problem if is_exact else quick_problem
            problem = builder(obj, retention, weight, best, lower_components, dom,
                              points if is_exact else cuts, total_lower_bound=lb)
            start = time.perf_counter()
            result = milp(problem[0], integrality=problem[1], bounds=problem[2], constraints=problem[3],
                          options=dict(node_limit=budget['nodes'], mip_rel_gap=min(1e-4, relative_gap/10),
                                       presolve=True, time_limit=min(time_limit, remaining)))
            require(result.status in [0, 1] or 'Solution limit reached' in result.message, 'Joint MILP failed: '+result.message)
            bound = getattr(result, 'mip_dual_bound', None)
            if bound is not None and np.isfinite(bound): lb = max(lb, float(bound))
            candidate = None
            if result.x is not None:
                candidate = np.rint(result.x[:obj.n])
                require(np.max(np.abs(candidate-result.x[:obj.n])) < 1e-5 and feasible(obj, candidate, retention), 'Invalid joint incumbent')
                polished = swaps(obj, candidate, weight, retention, steps=swap_steps)
                if (obj.value(polished, weight), tuple(np.flatnonzero(polished))) < (obj.value(best, weight), tuple(np.flatnonzero(best))):
                    best = polished
            ub = obj.value(best, weight)
            require(lb <= ub+1e-6, 'Invalid global joint lower bound')
            lb = min(lb, ub)
            record = dict(tier=tier, iteration=iteration, exact_normalization=is_exact, status=result.status,
                          nodes=getattr(result, 'mip_node_count', None), score=100*ub, lower_bound=100*lb,
                          gap=(ub-lb)/max(ub,1e-15), seconds=time.perf_counter()-start)
            logs.append(record)
            if progress is not None: progress(best, record)
            improved = previous_ub-ub > 1e-8 or lb-previous_lb > 1e-8
            stale = 0 if improved else stale+1
            if stale >= patience:
                stopped = 'stagnation'
                break
            if stale >= max(1, patience//2):
                force_exact = True
            added = add(best)
            if candidate is not None: added = add(candidate) or added
            if not added:
                if is_exact: break
                force_exact = True
        ub = obj.value(best, weight)
        if stopped or ub-lb <= relative_gap*ub or ub-lb <= 1e-6: break
    require(feasible(obj, best, retention), 'Final selection invalid')
    ub = obj.value(best, weight)
    certified = ub-lb <= relative_gap*ub or ub-lb <= 1e-6
    return best, dict(score=100*ub, lower_bound=100*lb, gap=(ub-lb)/max(ub,1e-15),
                      certified=bool(certified), iterations=len(logs),
                      status='certified_gap' if certified else 'bounded_best_found_'+(stopped or 'iteration_budget')), pd.DataFrame(logs)
