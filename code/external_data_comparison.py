"""Run experiment 03: prepare, backtest, freeze and confirm external category features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed

import external_data_sources as src
import external_data_models as mdl
from abs_momentum_comparison import PACKAGES, MODEL_PARAMS, require, write_table


def protocol(mapping):
    """Specify immutable decisions and operational interpretations before model evaluation."""
    return dict(name='External merchant features v1', sources=['CPI', 'Google Trends', 'CABEE'],
                mapping=mapping.to_dict('records'), feature_origins=src.ORIGINS.strftime('%Y-%m-%d').tolist(),
                development_origins=mdl.DEV_ORIGINS.strftime('%Y-%m-%d').tolist(),
                primary_horizon='three complete months', confirmation_origin='2022-09-30',
                confirmation_window=['2022-10-01', '2022-10-26'], holdout_status='July-September already examined; October merchant outcomes sealed',
                models=dict(tweedie=MODEL_PARAMS, boosting=mdl.TREE_PARAMS),
                arms=mdl.ARMS, internal=mdl.INTERNAL, category_controls=True,
                family_selection='minimum equal-origin mean fee Tweedie deviance (power 1.5) of B, only matured inner origins; minimum 2; default/tie Tweedie',
                target_scale=1000, sample_weight='equal merchant-origin rows', take_rate_divisor=100,
                downside_threshold=0.5, risk_lambda=1, sensitivity_lambdas=[0.5, 2], top_n=100,
                tie_break='ascending numeric ABN', informative='each enabled source varies within at least one training category and has current coverage',
                minimum_informative_origins=4, minimum_disjoint_windows=2, minimum_uplift_pct=1,
                minimum_positive_fraction=0.75, downside_minimum_deviance_improvement_pct=1,
                downside_maximum_fee_loss_pct=1, placebo_repetitions=99, placebo_seed=2026,
                placebo_shuffle='whole category trajectories independently per source within identical missingness patterns',
                placebo_gate='strictly above 95th percentile of best-arm economic gain among informative placebo arms',
                risk_rule='75% positive utility origins; >=1% utility gain / baseline fees and >=1% mean deviance reduction; fees >=99% of risk B',
                return_rule='equal-origin mean fee uplift >=1% against both B and repeat; 75% positive versus B',
                simplicity_tolerance_pp=0.25, source_priority=['C', 'E', 'G'],
                fragility='sum of selected-set changes becomes negative after removing largest positive merchant or category contribution over informative origins',
                source_removal='all included sources require positive mean marginal economic contribution on candidate informative origins',
                zero_metrics='undefined percentage denominators remain missing; zero prediction uses 1e-12 only inside deviance',
                estimability='drop overall-constant training features; report static category effects separately; informative gate requires within-category change',
                no_candidate='freeze B; October compares the internal policies and repeat/minimal only; no external feature can be promoted',
                confirmation_rule='same economic thresholds; risk also requires 1% shortfall-deviance improvement; no post-October candidate changes',
                google='three valid exports; daily observations aggregated to Sunday-Saturday means; reconstructed history, never verified historical publication',
                assumptions=['static known categories and take rates', 'recorded sales approximate hypothetical selection outcomes',
                             'gross fees exclude defaults and costs', 'synthetic data and short dependent history support pilot evidence only'])


def freeze_protocol(mapping, out):
    """Reject silent changes to the protocol in an existing experiment directory."""
    spec = json.loads(json.dumps(protocol(mapping)))
    path = out / 'protocol.json'
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == spec, 'Frozen protocol differs; use a new experiment directory')
    else:
        src.save_json(path, spec)
    src.save_json(out / 'runtime_versions.json', dict(python=sys.version, **{p: importlib.metadata.version(p) for p in [*PACKAGES, 'matplotlib']}))


def check_backtest_inputs(panel, features, out):
    """Prevent resumed placebo checkpoints from being combined with different prepared inputs."""
    def digest(frame):
        h = hashlib.sha256()
        h.update(str(list(zip(frame.columns, frame.dtypes.astype(str)))).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        return h.hexdigest()
    signature = dict(panel=digest(panel), features=digest(features))
    path = out / 'backtest_inputs.json'
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == signature,
                'Prepared inputs changed; do not reuse placebo checkpoints. Use a new output directory')
    else:
        src.save_json(path, signature)


def prepare(source_dir, out, offline=False):
    """Prepare audited data and source status while leaving October transaction payloads unread."""
    merchants, mapping = src.merchants_and_mapping(source_dir)
    freeze_protocol(mapping, out)
    panel = src.aggregate_development(source_dir, merchants, out)
    releases, values = src.acquire_abs(out=out, offline=offline)
    ext = src.external_features(mapping, releases, values, out=out)
    features = mdl.build_features(panel, merchants, ext)
    write_table(features, out, 'merchant_features')
    write_table(mapping, out, 'category_mapping')
    write_table(features.groupby(['origin', 'eligibility_reason'], as_index=False).size(), out, 'eligibility')
    coverage = []
    for origin, f in features.loc[features.eligible].groupby('origin'):
        recent = f.recent_three_month_sales * f.take_rate / 100
        for s in mdl.SOURCES:
            mask = f[s].notna()
            coverage.append(dict(origin=origin, source=s, eligible=len(f), matched=int(mask.sum()), categories=f.loc[mask, 'category'].nunique(),
                                 merchant_share_pct=float(mask.mean() * 100), recent_fee_share_pct=float(recent[mask].sum() / recent.sum() * 100)))
    write_table(pd.DataFrame(coverage), out, 'coverage')
    access = src.CACHE / 'trends' / 'access_evidence.json'
    access_evidence = json.loads(access.read_text(encoding='utf-8')) if access.exists() else {}
    status = {s: dict(status='available' if features[s].notna().any() else 'not_evaluable',
                      usable_category_origins=int(ext[s].notna().sum()), provenance='reconstructed' if s == 'G' else 'verified_dated_ABS_release')
              for s in mdl.SOURCES}
    if not features.G.notna().any():
        status['G'].update(reason='No category-origin has three valid reproducible official exports', access_evidence=access_evidence)
    src.save_json(out / 'source_status.json', status)
    return panel, features, mapping


def enabled_arms(features):
    """Retain the complete power set of usable sources, including baseline B."""
    available = {s for s in mdl.SOURCES if features[s].notna().any()}
    return [a for a, s in mdl.ARMS.items() if set(s).issubset(available)]


def input_at(panel, features, origin, horizon):
    """Construct a complete fold without reading the candidate merchants' future outcomes."""
    columns = ['merchant_abn', 'category', 'origin', 'take_rate', 'eligible',
               'recent_three_month_sales', 'reference_26'] + mdl.INTERNAL + mdl.SOURCES
    compact = features[columns]
    train = mdl.training_at(panel, compact, origin, horizon)
    test = compact.loc[compact.origin.eq(origin) & compact.eligible].sort_values('merchant_abn').reset_index(drop=True).copy()
    test['reference_sales'] = test.recent_three_month_sales if horizon == '3m' else test.reference_26
    design = mdl.design_at(train, test, sorted(features.category.unique()))
    return train, test, design


def evaluate(panel, features, origins, horizon, out, save=True, sensitivity=False, arms=None):
    """Walk forward, choosing model families only from already matured internal-baseline forecasts."""
    arms = enabled_arms(features) if arms is None else arms
    history, folds, predictions, accuracy, changes, records, prepared = [], [], [], [], [], [], []
    for origin in origins:
        print(f"{out.name}: fitting {horizon} origin {origin:%Y-%m-%d}", flush=True)
        train, test, design = input_at(panel, features, origin, horizon)
        families, family_evidence, bases = {}, {}, {}
        for target in ['return', 'risk']:
            families[target], family_evidence[target] = mdl.choose_family(history, origin, target, horizon)
            for family in ['tweedie', 'boosting']:
                bases[target, family], info = mdl.fit_model(design, train, test, 'B', family, target)
                records.append(dict(origin=str(origin.date()), selection=family_evidence[target], **info))
        forecast, details = mdl.subset_predictions(design, train, test, arms, families, bases)
        records.extend(dict(origin=str(origin.date()), **r) for r in details)
        minimal = mdl.minimal_predictions(train, test)
        pred = mdl.policy_frames(test, forecast, arms, minimal, lambdas=(0.5, 1.0, 2.0) if sensitivity else (1.0,))
        if save:
            write_table(pred, out / 'unscored_predictions', f'{origin:%Y-%m-%d}')
        # Current outcomes are touched only after all forecasts and ranks exist.
        scored, fold, acc, delta = mdl.score_policies(pred, panel, origin, horizon)
        actual_sales = mdl.target_at(panel, test.merchant_abn, origin, horizon)
        actual_short = np.maximum(0, 0.5 * test.reference_sales.to_numpy() - actual_sales)
        for target, y in [('return', actual_sales), ('risk', actual_short)]:
            for family in ['tweedie', 'boosting']:
                loss = mdl.deviance(y * test.take_rate.to_numpy() / 100, bases[target, family] * test.take_rate.to_numpy() / 100)
                history.append(dict(origin=origin, target=target, family=family, deviance=loss))
        fold['informative'] = fold.arm.map(lambda a: all(design.informative[s] for s in mdl.ARMS.get(a, ())))
        fold['return_family'], fold['risk_family'] = families['return'], families['risk']
        fold['window_schedule'] = list(origins).index(origin) % (3 if horizon == '3m' else 1)
        folds.append(fold); accuracy.append(acc); changes.append(delta); predictions.append(scored)
        prepared.append(dict(origin=origin, train=train, test=test, families=families,
                             baseline={t: bases[t, families[t]] for t in ['return', 'risk']},
                             actual_sales=actual_sales, actual_shortfall=actual_short))
    result = dict(folds=pd.concat(folds, ignore_index=True), predictions=pd.concat(predictions, ignore_index=True),
                  accuracy=pd.concat(accuracy, ignore_index=True), changes=pd.concat(changes, ignore_index=True),
                  history=pd.DataFrame(history), prepared=prepared, model_records=records, arms=arms)
    if save:
        for k, name in [('folds', 'fold_metrics'), ('predictions', 'predictions'), ('accuracy', 'forecast_metrics'),
                        ('changes', 'selection_changes'), ('history', 'baseline_validation')]:
            write_table(result[k], out, name)
        src.save_json(out / 'fitted_models.json', records)
        joblib.dump(prepared, out / 'fold_inputs.joblib', compress=0)
        composition = result['predictions'].loc[result['predictions'].selected].groupby(
            ['origin', 'arm', 'policy', 'risk_lambda', 'category'], as_index=False).agg(
                merchants=('merchant_abn', 'size'), actual_fees=('actual_fees', 'sum'), shortfall=('actual_shortfall', 'sum'))
        write_table(composition, out, 'category_composition')
        calibration = result['predictions'].loc[result['predictions'].policy.eq('risk') & result['predictions'].risk_lambda.eq(1)].copy()
        calibration['risk_decile'] = calibration.groupby(['origin', 'arm']).predicted_shortfall.transform(
            lambda v: pd.qcut(v.rank(method='first'), 10, labels=False))
        cal = calibration.groupby(['origin', 'arm', 'risk_decile'], as_index=False).agg(
            merchants=('merchant_abn', 'size'), predicted_shortfall=('predicted_shortfall', 'mean'), actual_shortfall=('actual_shortfall', 'mean'))
        write_table(cal, out, 'risk_calibration')
    return result


def strict_features(features):
    """Suppress every explicitly starred mapping without changing the eligible merchant pool."""
    strict = features.copy()
    for s, col in [('C', 'cpi_proxy'), ('E', 'cabee_proxy'), ('G', 'trends_proxy')]:
        strict.loc[strict[col], s] = np.nan
    return strict


def shuffled_external(external, seed):
    """Permute whole trajectories only between categories with identical availability histories."""
    rng = np.random.default_rng(seed)
    out, assignments = external[['category', 'origin'] + mdl.SOURCES].copy(), []
    for source in mdl.SOURCES:
        wide = external.pivot(index='category', columns='origin', values=source).sort_index()
        patterns = wide.notna().apply(lambda row: ''.join(row.astype(int).astype(str)), axis=1)
        for pattern in patterns.unique():
            cats = patterns.index[patterns.eq(pattern)].tolist()
            donors = rng.permutation(cats)
            for recipient, donor in zip(cats, donors):
                mask = out.category.eq(recipient)
                series = wide.loc[donor]
                out.loc[mask, source] = out.loc[mask, 'origin'].map(series).to_numpy()
                assignments.append(dict(seed=seed, source=source, category=recipient, donor_category=donor, pattern=pattern))
    require(np.array_equal(out[mdl.SOURCES].isna(), external[mdl.SOURCES].isna()), 'Placebo changed source availability')
    return out, assignments


def nonoverlap_count(origins, horizon='3m'):
    """Count a maximal chronological subset of disjoint outcome windows."""
    last, count = pd.Timestamp.min, 0
    for o in sorted(origins):
        if o + pd.offsets.MonthBegin(1) > last:
            count += 1
            last = mdl.horizon_end(o, horizon)
    return count


def arm_summary(folds, accuracy, changes=None):
    """Compute equal-origin decision metrics on outcome-independent informative folds."""
    rows = []
    for (arm, policy), group in folds.loc[folds.arm.isin(mdl.ARMS) & ~folds.arm.eq('B')
                                        & (folds.policy.eq('return') | folds.risk_lambda.eq(1))].groupby(['arm', 'policy']):
        f = group.loc[group.informative]
        origins = f.origin.tolist()
        b = folds.loc[folds.arm.eq('B') & folds.policy.eq(policy) & folds.risk_lambda.eq(0 if policy == 'return' else 1)].set_index('origin').reindex(origins)
        repeat = folds.loc[folds.arm.eq('repeat')].set_index('origin').reindex(origins)
        score = 'uplift_pct' if policy == 'return' else 'utility_gain_pct'
        row = dict(arm=arm, policy=policy, informative_origins=len(f), nonoverlapping_windows=nonoverlap_count(origins),
                   origins=','.join(pd.Timestamp(x).strftime('%Y-%m-%d') for x in origins),
                   mean_uplift_pct=float(f.uplift_pct.mean()), mean_delta_fees=float(f.delta_fees.mean()),
                   mean_repeat_uplift_pct=float(np.mean((f.fees.to_numpy() / repeat.fees.to_numpy() - 1) * 100)) if len(f) else np.nan,
                   mean_utility_gain_pct=float(f.utility_gain_pct.mean()), positive_origins=int(f[score].gt(0).sum()),
                   positive_fraction=float(f[score].gt(0).mean()), score=float(f[score].mean()),
                   mean_shortfall_deviance_improvement_pct=np.nan, fragile=False)
        if policy == 'risk' and len(f):
            a = accuracy.loc[accuracy.arm.eq(arm) & accuracy.target.eq('risk') & accuracy.cohort.eq('full_pool')
                             & accuracy.risk_lambda.eq(1)].set_index('origin').reindex(origins).deviance
            base = accuracy.loc[accuracy.arm.eq('B') & accuracy.target.eq('risk') & accuracy.cohort.eq('full_pool')
                                & accuracy.risk_lambda.eq(1)].set_index('origin').reindex(origins).deviance
            row['mean_shortfall_deviance_improvement_pct'] = float(((1 - a / base) * 100).mean())
        if changes is not None and len(f):
            d = changes.loc[changes.arm.eq(arm) & changes.policy.eq(policy) & changes.origin.isin(origins)
                            & changes.risk_lambda.eq(0 if policy == 'return' else 1)]
            col = 'fee_contribution' if policy == 'return' else 'utility_contribution'
            total = float(d[col].sum())
            max_merchant = max(0.0, float(d.groupby('merchant_abn')[col].sum().max())) if len(d) else 0.0
            max_category = max(0.0, float(d.groupby('category')[col].sum().max())) if len(d) else 0.0
            row.update(total_contribution=total, without_largest_merchant=total - max_merchant,
                       without_largest_category=total - max_category, fragile=bool(total > 0 and min(total - max_merchant, total - max_category) < -1e-6))
        rows.append(row)
    return pd.DataFrame(rows)


def placebo_one(rep, prepared, external, arms, directory):
    """Run one cached null experiment, refitting external arms while reusing identical baselines."""
    path = directory / f'{rep:03d}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    shuffled, assignments = shuffled_external(external, 2026 + rep)
    folds, acc = [], []
    for fold in prepared:
        def replace(frame):
            return frame.drop(columns=mdl.SOURCES).merge(shuffled, on=['category', 'origin'], how='left', validate='many_to_one')
        train, test = replace(fold['train']), replace(fold['test'])
        design = mdl.design_at(train, test, sorted(external.category.unique()))
        bases = {(target, fold['families'][target]): pred for target, pred in fold['baseline'].items()}
        forecast, _ = mdl.subset_predictions(design, train, test, arms, fold['families'], bases)
        rate = test.take_rate.to_numpy() / 100
        actual, loss = fold['actual_sales'] * rate, fold['actual_shortfall'] * rate
        repeats = mdl.ranks(test.merchant_abn, test.reference_sales.to_numpy() * rate) <= 100
        for policy in ['return', 'risk']:
            lam = float(policy == 'risk')
            bscores = (forecast['return', 'B'] - lam * forecast['risk', 'B']) * rate
            bmask = mdl.ranks(test.merchant_abn, bscores) <= 100
            bfee, butility = actual[bmask].sum(), (actual - lam * loss)[bmask].sum()
            for arm in arms:
                scores = (forecast['return', arm if policy == 'return' else 'B'] - lam * forecast['risk', arm]) * rate
                mask = mdl.ranks(test.merchant_abn, scores) <= 100
                fee, utility = actual[mask].sum(), (actual - lam * loss)[mask].sum()
                folds.append(dict(origin=fold['origin'], arm=arm, policy=policy, risk_lambda=lam,
                                  informative=all(design.informative[s] for s in mdl.ARMS[arm]), fees=fee,
                                  delta_fees=fee - bfee, uplift_pct=(fee / bfee - 1) * 100,
                                  utility_gain_pct=(utility - butility) / bfee * 100))
                acc.append(dict(origin=fold['origin'], arm=arm, target=policy, risk_lambda=lam, cohort='full_pool',
                                deviance=mdl.deviance(actual if policy == 'return' else loss,
                                                      forecast[policy, arm] * rate)))
        folds.append(dict(origin=fold['origin'], arm='repeat', policy='return', risk_lambda=0.0,
                          informative=True, fees=actual[repeats].sum(), delta_fees=0, uplift_pct=0, utility_gain_pct=0))
    summary = arm_summary(pd.DataFrame(folds), pd.DataFrame(acc))
    usable = summary.loc[summary.informative_origins.ge(4) & summary.nonoverlapping_windows.ge(2)]
    result = dict(repetition=rep, seed=2026 + rep,
                  best_return=float(usable.loc[usable.policy.eq('return'), 'score'].max()),
                  best_risk=float(usable.loc[usable.policy.eq('risk'), 'score'].max()))
    src.save_json(path, result)
    write_table(pd.DataFrame(assignments), directory, f'{rep:03d}_assignments')
    write_table(pd.DataFrame(folds), directory, f'{rep:03d}_folds')
    return result


def run_backtest(panel, features, out, jobs=4):
    """Complete main, proxy, risk-penalty and 99 placebo comparisons before candidate freezing."""
    require(not (out / 'confirmation_read.json').exists(), 'October was opened; development must not be changed')
    check_backtest_inputs(panel, features, out)
    result = evaluate(panel, features, mdl.DEV_ORIGINS, '3m', out / 'development', sensitivity=True)
    strict = evaluate(panel, strict_features(features), mdl.DEV_ORIGINS, '3m', out / 'strict_mapping', sensitivity=True)
    write_table(arm_summary(strict['folds'], strict['accuracy'], strict['changes']), out, 'strict_mapping_summary')
    external = features[['category', 'origin'] + mdl.SOURCES].drop_duplicates().reset_index(drop=True)
    require(not external.duplicated(['category', 'origin']).any(), 'Nonunique external source values')
    print('Running 99 complete-trajectory placebo comparisons', flush=True)
    generator = Parallel(n_jobs=jobs, backend='loky', return_as='generator_unordered')(
        delayed(placebo_one)(rep, result['prepared'], external, result['arms'], out / 'placebos') for rep in range(1, 100))
    placebo = []
    for r in generator:
        placebo.append(r)
        if len(placebo) % 5 == 0 or len(placebo) == 99:
            print(f"Completed {len(placebo)}/99 placebo experiments", flush=True)
    table = pd.DataFrame(placebo).sort_values('repetition')
    write_table(table, out, 'placebo_distribution')
    summary = arm_summary(result['folds'], result['accuracy'], result['changes'])
    for policy in ['return', 'risk']:
        threshold = float(table['best_' + policy].quantile(0.95))
        summary.loc[summary.policy.eq(policy), 'placebo_95_pct'] = threshold
    write_table(summary, out, 'development_summary')
    write_table(result['folds'].groupby(['arm', 'policy', 'risk_lambda', 'window_schedule'], as_index=False).agg(
        origins=('origin', 'size'), mean_delta_fees=('delta_fees', 'mean'), mean_uplift_pct=('uplift_pct', 'mean'), mean_utility_gain_pct=('utility_gain_pct', 'mean')),
        out, 'nonoverlapping_schedules')
    return result, summary


def decide(summary, folds):
    """Apply all frozen pilot gates and choose the simplest qualifying source combination."""
    summary = summary.copy()
    summary['qualifies'] = False
    summary['positive_marginal_sources'] = True
    lookup = {frozenset(v): k for k, v in mdl.ARMS.items()}
    for index, row in summary.iterrows():
        for source in mdl.ARMS[row.arm]:
            smaller = lookup[frozenset(set(mdl.ARMS[row.arm]) - {source})]
            dates = row.origins.split(',') if row.origins else []
            mask = folds.policy.eq(row.policy) & folds.risk_lambda.eq(0 if row.policy == 'return' else 1) & folds.origin.isin(pd.to_datetime(dates))
            a = folds.loc[mask & folds.arm.eq(row.arm)].set_index('origin')
            b = folds.loc[mask & folds.arm.eq(smaller)].set_index('origin')
            col = 'fees' if row.policy == 'return' else 'utility'
            gain = ((a[col] - b[col]) / b.fees * 100).mean()
            if not gain > 0:
                summary.loc[index, 'positive_marginal_sources'] = False
        common = (row.informative_origins >= 4 and row.nonoverlapping_windows >= 2 and row.positive_fraction >= 0.75
                  and row.score > row.placebo_95_pct and not row.fragile and bool(summary.loc[index, 'positive_marginal_sources']))
        if row.policy == 'return':
            success = row.mean_uplift_pct >= 1 and row.mean_repeat_uplift_pct >= 1
        else:
            success = row.mean_shortfall_deviance_improvement_pct >= 1 and row.mean_utility_gain_pct >= 1 and row.mean_uplift_pct >= -1
        summary.loc[index, 'qualifies'] = bool(common and success)
    choices = {}
    priority = {s: i for i, s in enumerate(['C', 'E', 'G'])}
    for policy in ['return', 'risk']:
        options = summary.loc[summary.policy.eq(policy) & summary.qualifies]
        if not len(options):
            choices[policy] = 'B'
            continue
        options = options.loc[options.score.ge(options.score.max() - 0.25)]
        choices[policy] = min(options.arm, key=lambda a: (len(mdl.ARMS[a]), sorted(priority[s] for s in mdl.ARMS[a])))
    return summary, choices


def freeze_candidates(out):
    """Freeze source subsets, including the no-winner fallback, without reading October outcomes."""
    require(not (out / 'confirmation_read.json').exists(), 'Cannot change candidates after October is opened')
    check_backtest_inputs(pd.read_parquet(out / 'merchant_month.parquet'), pd.read_parquet(out / 'merchant_features.parquet'), out)
    folds = pd.read_parquet(out / 'development/fold_metrics.parquet')
    summary = pd.read_parquet(out / 'development_summary.parquet')
    decision, choices = decide(summary, folds)
    write_table(decision, out, 'candidate_decisions')
    frozen = dict(choices=choices, frozen_at=pd.Timestamp.now(tz='UTC').isoformat(),
                  evidence='retrospective three-month development', source_status=json.loads((out / 'source_status.json').read_text(encoding='utf-8')))
    path = out / 'frozen_candidates.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        require(previous['choices'] == choices and previous['source_status'] == frozen['source_status'], 'Frozen candidate specification changed')
        return previous
    src.save_json(path, frozen)
    return frozen


def october_panel(source_dir, merchants, out):
    """Open October only after candidate and forecast artifacts exist; audit IDs against development."""
    require((out / 'frozen_candidates.json').exists() and (out / 'confirmation/unscored_predictions.parquet').exists(),
            'October access requires frozen candidates and saved unscored predictions')
    path = out / 'confirmation/october_panel.parquet'
    if path.exists():
        return pd.read_parquet(path)
    marker = out / 'confirmation_read.json'
    require(not marker.exists(), 'An earlier October read did not finish; inspect the interrupted confirmation before retrying')
    audit = json.loads((out / 'merchant_audit.json').read_text(encoding='utf-8'))
    require(audit['archives'] == src.archive_signature(source_dir), 'Sources changed after development audit')
    src.save_json(marker, dict(opened_at=pd.Timestamp.now(tz='UTC').isoformat(), reason='frozen October confirmation', status='started'))
    old_ids = sqlite3.connect(f"file:{(out / 'development_order_ids.sqlite').as_posix()}?mode=ro", uri=True)
    pieces, daily = [], []
    seen = set()
    try:
        for part in [2, 3, 4]:
            with zipfile.ZipFile(source_dir / f'project-2-bnpl-tables-part{part}.zip') as z:
                for name in sorted(n for n in z.namelist() if n.endswith('.parquet')):
                    m = re.search(r'order_datetime=(\d{4}-\d{2}-\d{2})/', name)
                    require(m is not None, 'Missing October partition date')
                    day = pd.Timestamp(m[1])
                    if not pd.Timestamp('2022-10-01') <= day <= pd.Timestamp('2022-10-26'):
                        continue
                    df = pq.read_table(io.BytesIO(z.read(name)), columns=['order_id', 'merchant_abn', 'dollar_value']).to_pandas()
                    require(df.notna().all().all() and df.order_id.is_unique and np.isfinite(df.dollar_value).all() and df.dollar_value.gt(0).all(), 'Invalid October transactions')
                    require(not (seen & set(df.order_id)), 'Duplicate October order IDs')
                    seen.update(df.order_id)
                    ids = df.order_id.tolist()
                    for start in range(0, len(ids), 500):
                        chunk = ids[start:start+500]
                        query = 'SELECT id FROM orders WHERE id IN (' + ','.join('?' for _ in chunk) + ') LIMIT 1'
                        require(old_ids.execute(query, chunk).fetchone() is None, 'October ID duplicates a development order')
                    g = df.groupby('merchant_abn').dollar_value.agg(sales_26='sum', orders_26='size').reset_index()
                    pieces.append(g)
                    daily.append(dict(day=day, orders=len(df), sales=float(df.dollar_value.sum())))
    finally:
        old_ids.close()
    days = pd.DataFrame(daily)
    require(set(days.day) == set(pd.date_range('2022-10-01', '2022-10-26')), 'Incomplete October coverage')
    raw = pd.concat(pieces).groupby('merchant_abn', as_index=False)[['sales_26', 'orders_26']].sum()
    require(raw.orders_26.sum() == days.orders.sum() and np.isclose(raw.sales_26.sum(), days.sales.sum(), rtol=1e-12), 'October aggregation mismatch')
    unknown = raw.loc[~raw.merchant_abn.isin(merchants.merchant_abn)]
    panel = raw.set_index('merchant_abn').reindex(merchants.merchant_abn, fill_value=0).reset_index()
    panel['month'] = pd.Timestamp('2022-10-01')
    write_table(panel, out / 'confirmation', 'october_panel')
    write_table(unknown, out / 'confirmation', 'unknown_october_merchants')
    write_table(days, out / 'confirmation', 'october_audit')
    src.save_json(marker, dict(opened_at=json.loads(marker.read_text())['opened_at'], status='complete',
                               source_orders=int(days.orders.sum()), source_sales=float(days.sales.sum()), unique_ids=len(seen)))
    return panel


def confirm(panel, features, merchants, source_dir, out):
    """Fit the frozen subsets for 26 days, save predictions, then score October exactly once."""
    if (out / 'confirmation/decision.json').exists():
        return json.loads((out / 'confirmation/decision.json').read_text(encoding='utf-8'))
    check_backtest_inputs(panel, features, out)
    frozen = json.loads((out / 'frozen_candidates.json').read_text(encoding='utf-8'))
    choices = frozen['choices']
    origin = pd.Timestamp('2022-09-30')
    history_dir = out / 'confirmation_development'
    # Short-horizon model selection uses only baseline losses; feature subsets remain frozen.
    needed = set(mdl.ARMS[choices['return']]) | set(mdl.ARMS[choices['risk']])
    confirm_arms = [a for a, items in mdl.ARMS.items() if set(items).issubset(needed)]
    history = evaluate(panel, features, pd.date_range('2021-09-30', '2022-08-31', freq='ME'), '26d', history_dir, arms=confirm_arms)
    train, test, design = input_at(panel, features, origin, '26d')
    family = {t: mdl.choose_family(history['history'].to_dict('records'), origin, t, '26d')[0] for t in ['return', 'risk']}
    forecasts, details = mdl.subset_predictions(design, train, test, confirm_arms, family)
    # The public comparison reports B and the frozen selected method only, never a new October search.
    chosen_return = forecasts['return', choices['return']]
    chosen_risk = forecasts['risk', choices['risk']]
    frames = mdl.policy_frames(test, forecasts, ['B'], mdl.minimal_predictions(train, test))
    for policy in ['return', 'risk']:
        p = frames.loc[frames.arm.eq('B') & frames.policy.eq(policy)].copy()
        p['arm'] = 'frozen_selected'
        p['predicted_fees'] = chosen_return * p.take_rate / 100
        p['predicted_shortfall'] = chosen_risk * p.take_rate / 100
        p['score'] = p.predicted_fees - (policy == 'risk') * p.predicted_shortfall
        p['rank'] = mdl.ranks(p.merchant_abn, p.score)
        p['selected'] = p['rank'] <= 100
        frames = pd.concat([frames, p], ignore_index=True)
    previous = out / 'confirmation/unscored_predictions.parquet'
    if previous.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(previous), frames, check_exact=True)
    else:
        write_table(frames, out / 'confirmation', 'unscored_predictions')
    src.save_json(out / 'confirmation/model_settings.json', dict(families=family, choices=choices, fits=details))
    october = october_panel(source_dir, merchants, out)
    scored, folds, accuracy, changes = mdl.score_policies(frames, october, origin, '26d')
    for frame, name in [(scored, 'predictions'), (folds, 'fold_metrics'), (accuracy, 'forecast_metrics'), (changes, 'selection_changes')]:
        write_table(frame, out / 'confirmation', name)
    write_table(scored.loc[scored.selected], out, 'final_selected_merchants')
    result = dict(choices=choices, checked_window='2022-10-01 through 2022-10-26', results={})
    for policy in ['return', 'risk']:
        row = folds.loc[folds.arm.eq('frozen_selected') & folds.policy.eq(policy)].iloc[0]
        if policy == 'return':
            repeat = folds.loc[folds.arm.eq('repeat')].iloc[0]
            ok = row.uplift_pct >= 1 and (row.fees / repeat.fees - 1) * 100 >= 1
        else:
            a = accuracy.loc[accuracy.arm.eq('frozen_selected') & accuracy.target.eq('risk') & accuracy.cohort.eq('full_pool')].iloc[0]
            b = accuracy.loc[accuracy.arm.eq('B') & accuracy.target.eq('risk') & accuracy.cohort.eq('full_pool')].iloc[0]
            ok = row.utility_gain_pct >= 1 and row.uplift_pct >= -1 and (1 - a.deviance / b.deviance) * 100 >= 1
        result['results'][policy] = dict(passed=bool(choices[policy] != 'B' and ok), uplift_pct=float(row.uplift_pct),
                                          utility_gain_pct=float(row.utility_gain_pct), external_candidate=choices[policy] != 'B')
    src.save_json(out / 'confirmation/decision.json', result)
    return result


def source_variation(out):
    """Separate static category differences from within-category changes using training inputs only."""
    rows = []
    for fold in joblib.load(out / 'development/fold_inputs.joblib'):
        for source in mdl.SOURCES:
            train = fold['train'].loc[fold['train'][source].notna(), ['category', 'origin', source]]
            if not len(train):
                continue
            value = train[source]
            category_mean = train.groupby('category')[source].transform('mean')
            total = float(((value - value.mean()) ** 2).mean())
            static = float(((category_mean - value.mean()) ** 2).mean())
            changing = float(((value - category_mean) ** 2).mean())
            require(np.isclose(total, static + changing, rtol=1e-10, atol=1e-14), 'Source variation decomposition failed')
            rows.append(dict(origin=fold['origin'], source=source, training_origins=train.origin.nunique(),
                             category_variance=static, within_category_variance=changing, total_variance=total,
                             within_category_share_pct=100 * changing / total if total else np.nan))
    result = pd.DataFrame(rows)
    write_table(result, out, 'source_variation')
    return result


def report(out):
    """Write an honest source-by-source finding, including blocked data and consumed outcomes."""
    candidates = pd.read_parquet(out / 'candidate_decisions.parquet')
    status = json.loads((out / 'source_status.json').read_text(encoding='utf-8'))
    confirmation = json.loads((out / 'confirmation/decision.json').read_text(encoding='utf-8'))
    variation = source_variation(out)
    choices = confirmation['choices']
    lines = ['# External category features: findings', '',
             f"Frozen return method: **{choices['return']}**. Frozen downside method: **{choices['risk']}**. B is the category-controlled internal baseline.", '',
             '| Source | Return conclusion | Downside conclusion |', '|---|---|---|']
    conclusions = {}
    for s, name in [('C', 'ABS CPI'), ('G', 'Google Trends'), ('E', 'ABS business exits')]:
        conclusions[s] = {}
        for role in ['return', 'risk']:
            if status[s]['status'] != 'available':
                text = 'Not evaluable: source access/export requirements not met'
            else:
                participating = candidates.loc[candidates.arm.map(lambda a: s in mdl.ARMS[a]) & candidates.policy.eq(role)]
                winner = choices[role]
                if s in mdl.ARMS[winner]:
                    passed = confirmation['results'][role]['passed']
                    text = 'Supported for a further pilot' if passed and 'G' not in mdl.ARMS[winner] else 'Promising but unconfirmed'
                elif participating.qualifies.any():
                    text = 'Promising but unconfirmed: omitted by simplicity rule'
                elif participating.informative_origins.lt(4).all():
                    text = 'Not evaluable: insufficient informative origins'
                else:
                    text = 'No demonstrated contribution'
            conclusions[s][role] = text
        lines.append(f"| {name} | {conclusions[s]['return']} | {conclusions[s]['risk']} |")
    lines += ['', '## Three-month development results', '',
              'These are retrospective comparisons. July–September had already been examined. Means below use each arm’s informative origins.', '',
              '| Arm | Role | Informative origins | Fee uplift vs B | Fee uplift vs repeat | Utility gain / B fees | Positive origins | Placebo 95% | Qualified |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in candidates.itertuples():
        lines.append(f'| {r.arm} | {r.policy} | {r.informative_origins} | {r.mean_uplift_pct:+.3f}% | {r.mean_repeat_uplift_pct:+.3f}% | {r.mean_utility_gain_pct:+.3f}% | {r.positive_origins}/{r.informative_origins} | {r.placebo_95_pct:+.3f}% | {r.qualifies} |')
    folds = pd.read_parquet(out / 'development/fold_metrics.parquet')
    internal = folds.loc[folds.policy.eq('return') & folds.arm.isin(['B', 'minimal', 'repeat'])].groupby('arm').agg(
        mean_fees=('fees', 'mean'), mean_uplift=('uplift_pct', 'mean'))
    arm_status = []
    for arm, sources in mdl.ARMS.items():
        missing = [s for s in sources if status[s]['status'] != 'available']
        arm_status.append(dict(arm=arm, sources=','.join(sources), status='disabled' if missing else 'evaluated',
                               reason='; '.join(s + ': ' + status[s].get('reason', 'unavailable') for s in missing),
                               development_origins=int(folds.loc[folds.arm.eq(arm), 'origin'].nunique())))
    write_table(pd.DataFrame(arm_status), out, 'arm_status')
    lines += ['', 'The internal comparators, averaged over all eight development origins:', '',
              '| Method | Mean selected gross fees | Mean uplift vs B |', '|---|---:|---:|']
    for name, r in internal.iterrows():
        lines.append(f'| {name} | A${r.mean_fees:,.2f} | {r.mean_uplift:+.3f}% |')
    lines += ['', 'Freezing B when no external candidate qualifies follows the experimental protocol; it does not establish that adding category indicators improves the previous internal baseline.', '',
              'Shortfall forecast deviance improvements on each arm’s informative origins:', '']
    for r in candidates.loc[candidates.policy.eq('risk')].itertuples():
        lines.append(f'- {r.arm}: {r.mean_shortfall_deviance_improvement_pct:+.3f}% (required: at least +1%).')
    risk_base = folds.loc[folds.arm.eq('B') & folds.policy.eq('risk') & folds.risk_lambda.eq(1)]
    if risk_base.shortfall.eq(0).all():
        lines += ['', 'The baseline’s selected 100 have zero realized shortfall under the specified 50% threshold in every development window. '
                  'Shortfalls occur elsewhere in the eligible pool, but this selection was not exposed to a realized severe-revenue event. '
                  'The lack of selection gains therefore provides limited evidence about performance under a future stress period.']
    lines += ['', 'The `source_variation` table decomposes training-feature variance into static category differences and changes within categories. '
              'This describes input variation; it does not attribute forecasting gains causally. At the final three-month fit:', '']
    for r in variation.loc[variation.origin.eq(variation.origin.max())].itertuples():
        lines.append(f'- {r.source}: {r.within_category_share_pct:.2f}% of feature variance is within-category change; the remainder is static category differences.')
    lines += ['', '## October check', '',
              'October 1–26 was opened only after subsets, model families, forecasts and selections were frozen. '
              'It is a separate 26-day task, not independent confirmation of three-month performance. Subsequent reruns are reproductions, not fresh holdouts.', '',
              'If B was frozen because no external candidate qualified, October compares internal benchmarks only; it cannot rescue an external feature.', '']
    for r, x in confirmation['results'].items():
        if not x['external_candidate']:
            lines.append(f'- {r}: B remained the selected method. No external candidate advanced to confirmation.')
        else:
            lines.append(f"- {r}: selected-method fee uplift {x['uplift_pct']:+.4f}%; utility gain / baseline fees {x['utility_gain_pct']:+.4f}%; external candidate passed: {x['passed']}.")
    october = pd.read_parquet(out / 'confirmation/fold_metrics.parquet')
    lines += ['', '| October return comparator | Selected gross fees | Uplift vs B |', '|---|---:|---:|']
    for r in october.loc[october.policy.eq('return') & october.arm.isin(['B', 'minimal', 'repeat'])].itertuples():
        lines.append(f'| {r.arm} | A${r.fees:,.2f} | {r.uplift_pct:+.3f}% |')
    lines += ['', '## Data and interpretation', '',
              '- CPI is nominal household price inflation for the weighted eight capital cities, not merchant margins, quantities or elasticity.',
              '- CABEE exits include administrative and structural changes. Annual industry rates are not merchant default probabilities; only two detailed vintages inform this study.',
              '- Google CSV queries and date windows were fixed. Historical exports are reconstructed today; identical export replicates would not establish independent sampling or original publication vintages.',
              '- Google access evidence is recorded in `source_status.json`: the public endpoint returned HTTP 429; one official CSV was obtained, but subsequent exports were not delivered.',
              '- Merchant category and percentage take rate are assumed stable and historically known. Recorded sales proxy outcomes under hypothetical selection.',
              '- Gross fees exclude defaults, funding costs and operating costs. Downside is the amount below half the recent comparable-period fee revenue.',
              '- Merchant data are synthetic. Shared category signals and overlapping periods mean merchant/transaction counts do not establish independent economic replications.',
              '- All eight subset definitions are implemented. Arms requiring unavailable sources are disabled and are never labelled tested-and-failed.',
              '- The placebo comparison searches every enabled subset in each repetition; its percentile is an overfitting diagnostic, not a causal p-value.',
              '- Strict mapping, alternate risk penalties, static-versus-changing signals, coverage, category composition and selection-change attribution are exported separately.',
              '- The original ABS Retail Trade experiment and its outputs were preserved.', '',
              'See the explanatory notebook and EXTERNAL_DATA_COMPARISON.md for exact reproduction commands and all artifact definitions.', '']
    (out / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    src.save_json(out / 'decision.json', dict(source_conclusions=conclusions, choices=choices, confirmation=confirmation['results']))


def main():
    """Expose resumable stages without a path that silently tunes against October outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'backtest', 'freeze', 'confirm', 'report', 'all'])
    p.add_argument('--source-dir', type=Path, default=src.ROOT.parent / 'project 2 materials')
    p.add_argument('--output-dir', type=Path, default=src.OUT)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    out = args.output_dir
    require(out.resolve() != (src.ROOT / 'tables/analysis/abs_momentum_comparison').resolve(), 'Cannot overwrite the original experiment')
    _, mapping = src.merchants_and_mapping(args.source_dir)
    freeze_protocol(mapping, out)
    if args.stage in ['all', 'prepare']:
        prepare(args.source_dir, out, args.offline)
    if args.stage in ['all', 'backtest', 'confirm']:
        panel = pd.read_parquet(out / 'merchant_month.parquet')
        features = pd.read_parquet(out / 'merchant_features.parquet')
    if args.stage in ['all', 'backtest']:
        run_backtest(panel, features, out, args.jobs)
    if args.stage in ['all', 'freeze']:
        frozen = freeze_candidates(out)
        print('Frozen feature sets:', frozen['choices'], flush=True)
    if args.stage in ['all', 'confirm']:
        merchants = pd.read_parquet(out / 'merchants.parquet')
        print(confirm(panel, features, merchants, args.source_dir, out), flush=True)
    if args.stage in ['all', 'report']:
        report(out)


if __name__ == '__main__':
    main()
