"""Past-only features, paired subset models and economic scoring for experiment 03."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from abs_momentum_comparison import MODEL_PARAMS, fit_component, require
from external_data_sources import ORIGINS, ROOT

INTERNAL = ['log_recent_sales', 'log_own_growth']
SOURCES = ['C', 'G', 'E']
ARMS = {'B': (), 'C': ('C',), 'G': ('G',), 'E': ('E',), 'CG': ('C', 'G'),
        'CE': ('C', 'E'), 'GE': ('G', 'E'), 'CGE': ('C', 'G', 'E')}
TREE_PARAMS = dict(loss='poisson', max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                   min_samples_leaf=100, l2_regularization=1, early_stopping=False, random_state=2026)
DEV_ORIGINS = pd.date_range('2021-11-30', '2022-06-30', freq='ME')


def horizon_end(origin, horizon):
    """Return the last observed day required to mature a training target."""
    return origin + pd.offsets.MonthEnd(3) if horizon == '3m' else origin + pd.offsets.MonthBegin(1) + pd.Timedelta(days=25)


def features_at(panel, merchants, external, origin):
    """Construct merchant inputs and eligibility using exactly six complete past months."""
    months = pd.date_range(end=origin.to_period('M').start_time, periods=6, freq='MS')
    history = panel.loc[panel.month.isin(months)]
    require(not history.duplicated(['merchant_abn', 'month']).any(), 'Duplicate history key')
    wide = history.pivot(index='merchant_abn', columns='month', values='sales').reindex(index=merchants.merchant_abn, columns=months)
    require(wide.notna().all().all(), 'Incomplete six-month merchant window')
    recent, previous = wide.iloc[:, 3:].sum(axis=1).to_numpy(), wide.iloc[:, :3].sum(axis=1).to_numpy()
    rows = merchants.copy().reset_index(drop=True)
    rows['origin'] = origin
    rows['recent_three_month_sales'], rows['previous_three_month_sales'] = recent, previous
    rows['recent_monthly_sales'] = recent / 3
    rows['own_growth'] = np.divide(recent, previous, out=np.full(len(rows), np.nan), where=previous > 0) - 1
    rows['log_recent_sales'] = np.log(np.where(recent > 0, recent / 3, np.nan))
    rows['log_own_growth'] = np.log(np.where((recent > 0) & (previous > 0), rows.own_growth + 1, np.nan))
    rows['eligibility_reason'] = np.select([~rows.valid_take_rate, previous <= 0, recent <= 0],
                                         ['invalid_take_rate', 'no_previous_sales', 'no_recent_sales'], default='eligible')
    rows['eligible'] = rows.eligibility_reason.eq('eligible')
    short = history.loc[history.month.isin(months[3:])].groupby('merchant_abn').sales_26.sum()
    rows['reference_26'] = short.reindex(rows.merchant_abn).to_numpy() / 3
    ext = external.loc[external.origin.eq(origin)]
    require(ext.category.is_unique, 'Duplicate category-origin external values')
    rows = rows.merge(ext, on=['category', 'origin'], how='left', validate='many_to_one')
    cutoff = origin.tz_localize('Australia/Sydney') + pd.Timedelta(days=1)
    for source in ['C', 'E']:
        col = source + '_published_at'
        if col in rows:
            require(rows.loc[rows[source].notna(), col].lt(cutoff).all(), 'Future ABS publication in merchant inputs')
    return rows


def build_features(panel, merchants, external):
    """Keep the feature artifact free of merchant outcome columns."""
    return pd.concat([features_at(panel.loc[panel.month.le(o)], merchants, external, o) for o in ORIGINS], ignore_index=True)


def target_at(panel, ids, origin, horizon):
    """Read only the specified complete target windows and retain zero-sales merchants."""
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3 if horizon == '3m' else 1, freq='MS')
    frame = panel.loc[panel.month.isin(months) & panel.merchant_abn.isin(ids)]
    require(not frame.duplicated(['merchant_abn', 'month']).any(), 'Duplicate target key')
    require(len(frame) == len(ids) * len(months), 'Incomplete target window')
    return frame.groupby('merchant_abn')['sales' if horizon == '3m' else 'sales_26'].sum().reindex(ids).to_numpy()


def training_at(panel, features, origin, horizon):
    """Exclude all immature targets before joining outcomes into the training matrix."""
    ends = features.origin.map(lambda d: horizon_end(d, horizon))
    train = features.loc[features.eligible & ends.le(origin)].copy()
    require(len(train) > 0 and train.origin.lt(origin).all(), 'Empty or immature training rows')
    history = panel.loc[panel.month.le(origin)]
    for day, group in train.groupby('origin'):
        train.loc[group.index, 'target_sales'] = target_at(history, group.merchant_abn, day, horizon)
    train['reference_sales'] = train.recent_three_month_sales if horizon == '3m' else train.reference_26
    train['target_shortfall'] = np.maximum(0, 0.5 * train.reference_sales - train.target_sales)
    return train.sort_values(['origin', 'merchant_abn']).reset_index(drop=True)


def deviance(actual, predicted):
    """Evaluate mean forecasts in fee thousands, with the zero-prediction limiting convention."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    require(len(actual) > 0 and np.isfinite(actual).all() and np.isfinite(predicted).all(), 'Invalid deviance inputs')
    return float(mean_tweedie_deviance(actual / 1000, np.maximum(predicted / 1000, 1e-12), power=1.5))


def forecast_metrics(actual, predicted):
    """Report mean-target loss together with familiar monetary accuracy and calibration metrics."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error, total = predicted - actual, float(np.sum(actual))
    return dict(merchants=len(actual), actual=float(total), predicted=float(np.sum(predicted)),
                mae=float(np.abs(error).mean()), wape_pct=float(np.abs(error).sum() / total * 100) if total > 0 else np.nan,
                bias_dollars=float(error.sum()), bias_pct=float(error.sum() / total * 100) if total > 0 else np.nan,
                deviance=deviance(actual, predicted))


@dataclass
class Design:
    """Share the exact fitted preprocessing across all arms at one cutoff."""
    train: np.ndarray
    test: np.ndarray
    names: list
    estimable: dict
    informative: dict
    observed: np.ndarray
    details: dict


def design_at(train, candidates, categories):
    """Scale continuous inputs, preserve category controls and use common missingness flags."""
    numeric = INTERNAL + SOURCES
    mean = train[numeric].mean().fillna(0)
    a, b = train[numeric].fillna(mean).to_numpy(float), candidates[numeric].fillna(mean).to_numpy(float)
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite model input')
    scaler = StandardScaler().fit(a)
    a, b = scaler.transform(a), scaler.transform(b)
    def controls(frame):
        cats = pd.Categorical(frame.category, categories=categories)
        require((cats.codes >= 0).all(), 'Unknown merchant category')
        return np.column_stack([np.eye(len(categories))[cats.codes], frame[SOURCES].isna().to_numpy(float)])
    a, b = np.column_stack([a, controls(train)]), np.column_stack([b, controls(candidates)])
    names = numeric + ['category:' + c for c in categories] + ['missing:' + s for s in SOURCES]
    estimable = {s: bool(train[s].notna().any() and train[s].std(ddof=0) > 1e-12) for s in SOURCES}
    informative = {s: bool((train.groupby('category')[s].agg(lambda x: x.max() - x.min()).fillna(0) > 1e-12).any()
                           and candidates[s].notna().any()) for s in SOURCES}
    return Design(a, b, names, estimable, informative, candidates[SOURCES].notna().to_numpy(),
                  dict(numeric=numeric, imputation_means=mean.tolist(), scaler_mean=scaler.mean_.tolist(),
                       scaler_scale=scaler.scale_.tolist(), training_rows=len(train),
                       training_origins=sorted(train.origin.dt.strftime('%Y-%m-%d').unique().tolist())))


def fit_model(design, train, candidates, arm, family, target):
    """Fit one fixed model; target scale and source subsets are explicit and auditable."""
    active = [s for s in ARMS[arm] if design.estimable[s]]
    columns = [i for i, name in enumerate(design.names) if name not in SOURCES or name in active]
    y = train['target_sales' if target == 'return' else 'target_shortfall'].to_numpy() / 1000
    info = dict(family=family, arm=arm, target=target, active=active, **design.details)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if not np.any(y > 0):
            prediction = np.zeros(len(candidates))
            info.update(iterations=0, constant_zero_target=True)
        else:
            model = TweedieRegressor(**MODEL_PARAMS) if family == 'tweedie' else HistGradientBoostingRegressor(**TREE_PARAMS)
            model.fit(design.train[:, columns], y)
            prediction = model.predict(design.test[:, columns]) * 1000
            require(np.isfinite(prediction).all() and (prediction >= 0).all(), 'Invalid model prediction')
            info.update(iterations=int(model.n_iter_))
            if family == 'tweedie':
                info.update(features=[design.names[i] for i in columns], coefficients=model.coef_.tolist(), intercept=float(model.intercept_))
    if target == 'risk':
        prediction = np.clip(prediction, 0, 0.5 * candidates.reference_sales.to_numpy())
    return prediction, info


def choose_family(history, origin, target, horizon):
    """Choose by baseline validation losses whose entire target period is already observed."""
    past = [r for r in history if r['target'] == target and horizon_end(r['origin'], horizon) <= origin]
    n = len(set(r['origin'] for r in past))
    if n < 2:
        return 'tweedie', dict(inner_origins=n, reason='fewer_than_two_matured_origins')
    means = {f: float(np.mean([r['deviance'] for r in past if r['family'] == f])) for f in ['tweedie', 'boosting']}
    winner = 'tweedie' if means['tweedie'] <= means['boosting'] else 'boosting'
    return winner, dict(inner_origins=n, losses=means, reason='minimum_past_baseline_deviance')


def subset_predictions(design, train, candidates, arms, families, base_predictions=None):
    """Route incomplete merchant observations to exactly the corresponding smaller subset model."""
    forecasts, details = {}, []
    key = {frozenset(v): k for k, v in ARMS.items()}
    for target in ['return', 'risk']:
        raw = {}
        for arm in arms:
            if arm == 'B' and base_predictions is not None:
                raw[arm] = base_predictions[(target, families[target])]
            else:
                raw[arm], info = fit_model(design, train, candidates, arm, families[target], target)
                details.append(info)
        for arm in arms:
            required = set(ARMS[arm])
            routes = [key[frozenset(s for i, s in enumerate(SOURCES) if s in required and observed[i] and design.estimable[s])]
                      for observed in design.observed]
            forecasts[(target, arm)] = np.array([raw[r][i] for i, r in enumerate(routes)])
        for source in SOURCES:
            missing = ~design.observed[:, SOURCES.index(source)]
            for arm in arms:
                if source in ARMS[arm]:
                    smaller = key[frozenset(set(ARMS[arm]) - {source})]
                    require(np.array_equal(forecasts[target, arm][missing], forecasts[target, smaller][missing]),
                            'Unavailable-source forecasts differ')
    return forecasts, details


def minimal_predictions(train, candidates):
    """Reproduce the prior internal-only Tweedie baseline, including its frozen coverage split."""
    mapping = pd.read_csv(ROOT / 'code/abs_category_mapping.csv', keep_default_na=False)
    matched = set(mapping.loc[mapping.series_id.ne(''), 'category'])
    pred = np.empty(len(candidates))
    for flag in [True, False]:
        a = train.loc[train.category.isin(matched).eq(flag)]
        mask = candidates.category.isin(matched).eq(flag)
        if mask.any():
            with threadpool_limits(limits=1):
                pred[mask], _ = fit_component(a, candidates.loc[mask], INTERNAL)
    return pred


def ranks(ids, scores):
    """Rank fees or utility descending with ascending numeric ABN as the deterministic tie break."""
    require(len(ids) >= 100 and len(set(ids)) == len(ids) and np.isfinite(scores).all(), 'Invalid selection pool')
    order = np.lexsort((np.asarray(ids), -np.asarray(scores)))
    result = np.empty(len(ids), dtype=int)
    result[order] = np.arange(1, len(ids) + 1)
    return result


def policy_frames(candidates, forecasts, arms, minimal=None, lambdas=(1.0,)):
    """Construct selections before outcomes; risk arms share baseline return forecasts."""
    frames = []
    cols = ['merchant_abn', 'category', 'origin', 'take_rate', 'reference_sales'] + SOURCES
    for arm in arms:
        for policy, risk_lambda in [('return', 0.0)] + [('risk', x) for x in lambdas]:
            frame = candidates[cols].copy()
            frame['arm'], frame['policy'], frame['risk_lambda'] = arm, policy, risk_lambda
            ret = forecasts['return', arm] if policy == 'return' else forecasts['return', 'B']
            frame['predicted_fees'] = ret * frame.take_rate / 100
            frame['predicted_shortfall'] = forecasts['risk', arm] * frame.take_rate / 100
            frame['score'] = frame.predicted_fees - risk_lambda * frame.predicted_shortfall
            frame['rank'] = ranks(frame.merchant_abn, frame.score)
            frame['selected'] = frame['rank'] <= 100
            frames.append(frame)
    for name, pred in [('repeat', candidates.reference_sales.to_numpy()), ('minimal', minimal)]:
        if pred is not None:
            frame = candidates[cols].copy()
            frame['arm'], frame['policy'], frame['risk_lambda'] = name, 'return', 0.0
            frame['predicted_fees'] = pred * frame.take_rate / 100
            frame['predicted_shortfall'] = np.nan
            frame['score'] = frame.predicted_fees
            frame['rank'] = ranks(frame.merchant_abn, frame.score)
            frame['selected'] = frame['rank'] <= 100
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_policies(predictions, panel, origin, horizon):
    """Attach realized outcomes after selection and independently reconcile fee attribution."""
    unique = predictions.drop_duplicates('merchant_abn')
    actual = target_at(panel, unique.merchant_abn, origin, horizon)
    outcomes = pd.DataFrame(dict(merchant_abn=unique.merchant_abn, actual_sales=actual))
    scored = predictions.merge(outcomes, on='merchant_abn', validate='many_to_one')
    scored['actual_fees'] = scored.actual_sales * scored.take_rate / 100
    scored['actual_shortfall'] = np.maximum(0, 0.5 * scored.reference_sales - scored.actual_sales) * scored.take_rate / 100
    scored['actual_utility'] = scored.actual_fees - scored.risk_lambda * scored.actual_shortfall
    rows, accuracy, changes = [], [], []
    for (arm, policy, lam), group in scored.groupby(['arm', 'policy', 'risk_lambda']):
        selected = group.loc[group.selected]
        require(len(selected) == 100 and selected.merchant_abn.is_unique, 'Selection must contain exactly 100 merchants')
        totals = selected[['actual_fees', 'actual_shortfall', 'actual_utility']].sum()
        direct = target_at(panel, selected.merchant_abn, origin, horizon) * selected.take_rate.to_numpy() / 100
        require(np.isclose(direct.sum(), totals.actual_fees, rtol=1e-12), 'Independent selected fee calculation failed')
        row = dict(origin=origin, arm=arm, policy=policy, risk_lambda=lam, fees=float(totals.actual_fees),
                   shortfall=float(totals.actual_shortfall), utility=float(totals.actual_utility))
        base = scored.loc[scored.arm.eq('B') & scored.policy.eq(policy) & scored.risk_lambda.eq(lam)]
        bsel = base.loc[base.selected]
        row.update(delta_fees=float(totals.actual_fees - bsel.actual_fees.sum()),
                   uplift_pct=float((totals.actual_fees / bsel.actual_fees.sum() - 1) * 100),
                   utility_gain_pct=float((totals.actual_utility - bsel.actual_utility.sum()) / bsel.actual_fees.sum() * 100),
                   top100_overlap=len(set(selected.merchant_abn) & set(bsel.merchant_abn)))
        delta = group[['merchant_abn', 'category', 'actual_fees', 'actual_shortfall', 'actual_utility', 'selected']].merge(
            base[['merchant_abn', 'selected']].rename(columns={'selected': 'baseline_selected'}), on='merchant_abn', validate='one_to_one')
        delta = delta.loc[delta.selected.ne(delta.baseline_selected)].copy()
        sign = np.where(delta.selected, 1, -1)
        delta['fee_contribution'], delta['utility_contribution'] = delta.actual_fees * sign, delta.actual_utility * sign
        delta['change'] = np.where(delta.selected, 'entered', 'removed')
        delta['origin'], delta['arm'], delta['policy'], delta['risk_lambda'] = origin, arm, policy, lam
        require(np.isclose(delta.fee_contribution.sum(), row['delta_fees'], atol=1e-6), 'Selection attribution failed')
        rows.append(row)
        changes.append(delta)
        available = ARMS.get(arm, ())
        covered = group[list(available)].notna().any(axis=1) if available else pd.Series(True, index=group.index)
        for cohort, mask in [('full_pool', np.ones(len(group), dtype=bool)), ('matched', covered.to_numpy())]:
            sub = group.loc[mask]
            if len(sub):
                target = 'return' if policy == 'return' else 'risk'
                y, p = (sub.actual_fees, sub.predicted_fees) if target == 'return' else (sub.actual_shortfall, sub.predicted_shortfall)
                accuracy.append(dict(origin=origin, arm=arm, target=target, risk_lambda=lam, cohort=cohort, **forecast_metrics(y, p)))
    return scored, pd.DataFrame(rows), pd.DataFrame(accuracy), pd.concat(changes, ignore_index=True)
