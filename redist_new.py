"""
Bond Charge Optimization (Direct Charge Space)
===============================================
Decision variables:
    charge[j]  - final charge in bps for each unlocked bond j

Per unlocked bond i:
    proceeds_i  =  sign_i × charge_i × kappa_i

Key config parameters:
    lambda_param:   Anchoring — penalizes charge deviations from starting
    gamma:          Curve shaping — penalizes deviations from charge-curve ideal
    trader_pull:    Soft penalty for trader proceeds deviating from target
    target_blend:   0=keep starting allocation, 1=full risk-weighted

Objective:  MAXIMIZE Σ proceeds_i - λ·anchoring - γ·curve_pull - tp·trader_pull
Constraints:
    1. Side band       (total proceeds within [floor×start, start/floor])
    2. Trader band     (trader proceeds within target ± buffer)
    3. Through-mid     (optional: prevent sign flip)
    4. Per-bond caps   (optional: max individual skew delta)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

import cvxpy as cp
import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame

from app.logs.logging import log

COLUMN_MAP: dict[str, str] = {
    "bond_id": "tnum",
    "trader": "desigName",
    "side": "side",
    "quote_type": "quoteType",
    "ref_mid_px": "_refMidPx",
    "ref_mid_spd": "_refMidSpd",
    "quote_px": "_my_quotePx",
    "quote_spd": "_my_quoteSpd",
    "size": "grossSize",
    "dv01": "grossDv01",
    "liq_score": "avgLiqScore",
    "bsr_notional": "firmBsrSize",
    "bsi_notional": "firmBsinSize",
    "locked": "isLocked",
}


def _c(k: str) -> str:
    return COLUMN_MAP.get(k, None)


_DEFAULT_CHARGE_STRENGTH = {
    ("BSR", "PX"): 0.10,
    ("BSI", "PX"): 0.10,
    ("BSR", "SPD"): 0.20,
    ("BSI", "SPD"): 0.20,
}
_DEFAULT_STRENGTH = 0.20


def add_adjusted_bsr_bsi(df: pl.DataFrame | pl.LazyFrame, clamp_liq=True,
                          charge_strength: dict | None = None,
                          default_strength: float | None = None):

    cs = charge_strength or _DEFAULT_CHARGE_STRENGTH
    ds = default_strength if default_strength is not None else _DEFAULT_STRENGTH

    liq_expr = pl.col(_c('liq_score')).cast(pl.Float64)

    if clamp_liq:
        liq_expr = liq_expr.clip(1.0, 10.0)

    bsr_expr = (
        pl.when(pl.col(_c('quote_type'))==pl.lit("PX")).then(pl.lit(cs.get(('BSR', 'PX'), ds)))
        .when(pl.col(_c('quote_type'))==pl.lit("SPD")).then(pl.lit(cs.get(('BSR', 'SPD'), ds)))
        .otherwise(pl.lit(ds))
        .cast(pl.Float64).alias('_bsr_mult')
    )

    bsi_expr = (
        pl.when(pl.col(_c('quote_type'))==pl.lit("PX")).then(pl.lit(cs.get(('BSI', 'PX'), ds)))
        .when(pl.col(_c('quote_type'))==pl.lit("SPD")).then(pl.lit(cs.get(('BSI', 'SPD'), ds)))
        .otherwise(pl.lit(ds))
        .cast(pl.Float64).alias('_bsi_mult')
    )

    factor_bsr_expr = (pl.lit(1.0) + bsr_expr).pow(pl.lit(11.0) - liq_expr)
    factor_bsi_expr = (pl.lit(1.0) - bsi_expr).pow(pl.lit(11.0) - liq_expr)

    return df.with_columns([
        factor_bsr_expr.alias('rawBsrCharge'),
        factor_bsi_expr.alias('rawBsiCharge'),
        (pl.col(_c('bsr_notional')) / pl.col(_c('size'))).alias('bsrPct'),
        (pl.col(_c('bsi_notional')) / pl.col(_c('size'))).alias('bsiPct')
    ]).with_columns([
        ((pl.col('rawBsrCharge') * pl.col('bsrPct')) + (pl.col('rawBsiCharge') * pl.col('bsiPct'))).alias('blendedCharge')
    ])

_SOLVER_KW: dict[str, dict] = {
    "SCS": {"max_iters": 100_000}, "ECOS": {}, "OSQP": {"max_iter": 100_000}, "CLARABEL": {},
}


@dataclass
class OptimizerConfig:
    # ── Side-level constraints ──────────────────────────────────────────
    side_floor_pct: float = 0.99
    """Symmetric band for total proceeds per side. Total must stay within
    [floor × starting, starting / floor]. Controls how much the WAVG skew
    can shift. 0.99 = ±1% total shift (wavg barely moves). 0.95 = ±5%.
    Min: 0.50 | Max: 1.00 | Default: 0.99"""

    # ── Trader-band constraints ─────────────────────────────────────────
    buffer_mode: str = 'pct'
    """How the trader-band buffer is computed. 'fixed' uses buffer_fixed;
    'pct' uses max(|target| × buffer_pct, buffer_min_flat).
    Controls how much redistribution ACROSS traders is allowed.
    Options: 'fixed', 'pct' | Default: 'pct'"""

    buffer_fixed: float = 500.0
    """Flat dollar buffer around each trader's target (when buffer_mode='fixed').
    Min: 0 | Max: unbounded | Default: 500.0"""

    buffer_pct: float = 0.25
    """Percentage of |target| used as buffer (when buffer_mode='pct').
    Controls how far each trader can deviate from their risk-weighted target.
    0.10 = ±10% (tight redistribution), 0.25 = ±25% (moderate), 0.50 = ±50% (loose).
    Min: 0.0 | Max: 1.0 | Default: 0.25"""

    buffer_min_flat: float = 200.0
    """Minimum dollar buffer floor when buffer_mode='pct'. Prevents
    tiny traders from having zero room to move.
    Min: 0 | Max: unbounded | Default: 200.0"""

    # ── Risk allocation weights ─────────────────────────────────────────
    risk_weights: Dict[str, float] = field(default_factory=dict)
    """Weights for blending risk metrics into each trader's target share.
    Keys: 'NET', 'GROSS', 'NET_BSI', 'GROSS_BSI'. Auto-normalized to sum=1.
    Default: {NET_BSI: 0.50, GROSS_BSI: 0.25, NET: 0.15, GROSS: 0.10}"""

    bsr_risk_share_weight: float = 0.70
    """Weight on the trader's actual BSR risk share when computing the
    BSR-component of the target allocation. Paired with bsr_preference_weight.
    Must satisfy: bsr_risk_share_weight + bsr_preference_weight = 1.0.
    Min: 0.0 | Max: 1.0 | Default: 0.70"""

    bsr_preference_weight: float = 0.30
    """Weight on the BSR preference boost (trader's own BSR%) when computing
    the BSR-component of the target allocation. Higher values give more
    target share to traders with higher BSR percentages, independent of
    their absolute DV01 size.
    Min: 0.0 | Max: 1.0 | Default: 0.30"""

    target_blend: float = 0.50
    """Anchoring blend between risk-weighted target and starting allocation.
    0.0 = keep starting (no redistribution). 1.0 = full risk-weighted target.
    0.5 = halfway. Controls how aggressively the optimizer redistributes
    proceeds across traders.
    Min: 0.0 | Max: 1.0 | Default: 0.50"""

    gamma: float = 1.0
    """Curve-pull strength. Penalizes each bond's charge for deviating from
    its charge-curve-implied ideal level. Higher gamma = stronger curve shaping.
    This is what makes charge_strength visible in the output.
    Works in tension with lambda_param (which anchors to starting).
    Min: 0.0 | Max: unbounded (practical: 0.1–10.0) | Default: 1.0"""

    trader_pull: float = 1.0
    """Trader target-pull strength. Soft penalty for each trader's total proceeds
    deviating from their risk-weighted target. Drives redistribution across traders.
    At 0: no trader redistribution. At 10: aggressive redistribution.
    Min: 0.0 | Max: unbounded (practical: 0.1–10.0) | Default: 1.0"""

    # ── Objective & penalty ─────────────────────────────────────────────
    lambda_param: float = 0.25
    """Penalty strength for per-bond charge deviations from starting.
    Higher values keep individual bond charges closer to starting; lower
    values allow more redistribution. Clamped to >= 0.
    Min: 0.0 | Max: unbounded (practical: 0.01–1.0) | Default: 0.25"""

    skew_asymmetry: float = 0.0
    """Additional penalty that discourages reducing a bond's absolute charge
    below its starting level (penalizes normalization toward zero). Only
    active when > 0.
    Min: 0.0 | Max: unbounded (practical: 0.0–1.0) | Default: 0.0"""

    # ── Charge curve calibration ────────────────────────────────────────
    charge_strength: Dict[tuple, float] = field(default_factory=dict)
    """Per-(BSR/BSI, PX/SPD) exponential base rates for the blended charge
    curve. Controls how steeply the charge increases with illiquidity.
    Default: {(BSR,PX): 0.10, (BSI,PX): 0.10, (BSR,SPD): 0.20, (BSI,SPD): 0.20}"""

    default_strength: float | None = None
    """Fallback charge strength if a (BSR/BSI, qt) key is missing from
    charge_strength. None uses _DEFAULT_STRENGTH (0.20).
    Min: 0.0 | Max: 1.0 | Default: None (→ 0.20)"""

    # ── Skew delta caps ─────────────────────────────────────────────────
    max_individual_spread_skew_delta: float | None = None
    """Maximum per-bond skew change for SPD bonds. None = disabled.
    Units: spread bps.
    Min: 0.0 | Max: unbounded | Default: None (disabled)"""

    max_individual_px_skew_delta: float | None = None
    """Maximum per-bond skew change for PX bonds. None = disabled.
    Units: price points (converted to bps internally via ×100).
    Min: 0.0 | Max: unbounded | Default: None (disabled)"""

    # ── Behavioral flags ────────────────────────────────────────────────
    allow_through_mid: bool = True
    """If False, constrains each bond's final charge to stay on the same
    side of mid as its starting charge (no sign flip). If True, the
    optimizer may push charges through mid for better PnL.
    Default: True"""

    isolate_traders: bool = False
    """If True, solves each trader independently (no cross-trader
    redistribution). Useful for debugging individual trader behavior.
    Default: False"""

    anchor_wavg_skew: bool = False
    """If True, applies a post-solver parallel shift to all unlocked bonds
    so that the weighted-average skew per (side, quoteType) bucket matches
    the original starting wavg skew. The solver runs normally; this is a
    correction pass that preserves the solver's relative bond ordering.
    Default: False"""

    auto_relax: bool = True
    """If True, automatically widens trader-band buffers (2x, 4x) on
    infeasibility before giving up. Prevents hard failures on tight configs.
    Default: True"""

    debug: bool = True
    """Enables debug logging (solver status, skew changes, risk_pct
    validation, auto-relax warnings).
    Default: True"""

    # ── Grouping ────────────────────────────────────────────────────────
    group_columns: list[str] | None = None
    """Alternative grouping columns to use instead of the default trader
    column ('desigName'). When set, a composite key is built from these
    columns. None = group by trader.
    Default: None"""

    # ── Solver ──────────────────────────────────────────────────────────
    starting_skew_override: dict | None = None
    """Override starting skew for "Run From X" mode. Dict of {("BUY","SPD"): 2.0, ...}
    keyed by (side, quoteType). If set, all bonds in that basket start from this
    skew value instead of their live skew. None = use live skew (default).
    Default: None"""

    solver: str = "SCS"
    """Primary CVXPY solver. Falls back through [SCS, ECOS, CLARABEL].
    Options: 'SCS', 'ECOS', 'OSQP', 'CLARABEL' | Default: 'SCS'"""

    solver_verbose: bool = False
    """Pass verbose=True to the solver for iteration-level output.
    Default: False"""

    def __post_init__(self):
        if not self.risk_weights:
            self.risk_weights = {
                "NET_BSI": 0.5,
                "GROSS_BSI": 0.25,
                "NET": 0.15,
                "GROSS": 0.1
            }
        tlt = sum(self.risk_weights.values())
        if tlt and (tlt != 1):
            for k, v in self.risk_weights.items():
                self.risk_weights[k] = v / tlt
        self.lambda_param = max(0.0, self.lambda_param)
        if not self.charge_strength:
            self.charge_strength = dict(_DEFAULT_CHARGE_STRENGTH)

        # Normalize BSR blend weights to sum to 1.0
        bsr_total = self.bsr_risk_share_weight + self.bsr_preference_weight
        if bsr_total > 0 and abs(bsr_total - 1.0) > 1e-6:
            self.bsr_risk_share_weight /= bsr_total
            self.bsr_preference_weight /= bsr_total


def _sort_dict(d):
    return {key: d[key] for key in sorted(d)}


@dataclass
class OptimizationResult:
    status: str
    optimal: bool
    objective_value: float
    X_values: dict[tuple[str, str], float]
    Y_values: dict[tuple[str, str, str], float]
    final_charges: np.ndarray
    final_proceeds: np.ndarray
    starting_proceeds_by_side: dict[str, float]
    final_proceeds_by_side: dict[str, float]
    risk_pct: dict[tuple[str, str, str], float]
    diagnostics: dict = None


PX_SOURCES = ['_refMidPx', 'bvalMidPx', 'macpMidPx', 'amMidPx', 'traceMidPx', 'cbbtMidPx']
SPD_SOURCES = ['_refMidSpd', 'bvalMidSpd', 'macpMidSpd', 'amMidSpd', 'traceMidSpd', 'cbbtMidSpd']
MY_PX_SOURCE_BID = ['newLevelPx', '_refBidPx','bvalBidPx', 'macpBidPx', 'amBidPx', 'traceBidPx', 'cbbtBidPx']
MY_PX_SOURCE_ASK = ['newLevelPx', '_refAskPx','bvalAskPx', 'macpAskPx', 'amAskPx', 'traceAskPx', 'cbbtAskPx']
MY_SPD_SOURCE_BID = ['newLevelSpd', '_refBidSpd','bvalBidSpd', 'macpBidSpd', 'amBidSpd', 'traceBidSpd', 'cbbtBidSpd']
MY_SPD_SOURCE_ASK = ['newLevelSpd', '_refAskSpd','bvalAskSpd', 'macpAskSpd', 'amAskSpd', 'traceAskSpd', 'cbbtAskSpd']

IS_SPREAD = pl.col(_c('quote_type'))=='SPD'
IS_PX = pl.col(_c('quote_type'))=='PX'
IS_BUY = pl.col(_c('side'))=='BUY'
IS_SELL = pl.col(_c('side'))=='SELL'

_GROUP_KEY_COL = '_group_key'
_GROUP_SEP = '|||'

def _resolve_inner_col(cfg: OptimizerConfig) -> tuple[str, list[str] | None]:
    """Return (inner_column_name, original_group_columns_or_None).
    When group_columns is set, the inner column is '_group_key' (a composite).
    Otherwise falls back to the standard 'desigName' trader column."""
    gc = getattr(cfg, 'group_columns', None)
    if gc and isinstance(gc, list) and len(gc) > 0:
        return _GROUP_KEY_COL, gc
    return _c('trader'), None


def _add_group_key_column(df, group_cols: list[str]):
    """Add a composite _group_key column by concatenating group_cols with a separator."""
    if len(group_cols) == 1:
        return df.with_columns(pl.col(group_cols[0]).cast(pl.Utf8).fill_null('MISSING').alias(_GROUP_KEY_COL))
    parts = [pl.col(c).cast(pl.Utf8).fill_null('MISSING') for c in group_cols]
    expr = parts[0]
    for p in parts[1:]:
        expr = expr + pl.lit(_GROUP_SEP) + p
    return df.with_columns(expr.alias(_GROUP_KEY_COL))


def _parse_group_key(key: str, group_cols: list[str]) -> dict[str, str]:
    """Parse a composite key back into individual column values."""
    parts = key.split(_GROUP_SEP)
    result = {}
    for i, col in enumerate(group_cols):
        result[col] = parts[i] if i < len(parts) else 'MISSING'
    return result


def _apply_anchor_wavg_skew(
    final_charge_bps: np.ndarray,
    final_proceeds_arr: np.ndarray,
    starting_charge_bps: np.ndarray,
    signs: np.ndarray,
    kappa: np.ndarray,
    sides: list,
    qts: list,
    locked: np.ndarray,
    is_px: np.ndarray,
    bps_to_skew: np.ndarray,
    debug: bool = False,
) -> np.ndarray:
    """Post-solver correction: parallel-shift unlocked bonds so that the
    weighted-average skew per (side, quoteType) bucket matches the original
    starting wavg skew.

    Weight = dv01 (=kappa) for SPD, size/10000 (=kappa) for PX — kappa
    already encodes the correct weighting convention.

    Modifies final_charge_bps and final_proceeds_arr in-place.
    Returns anchor_adj: per-bond adjustment in skew units (0 for locked bonds).
    """
    N = len(final_charge_bps)
    anchor_adj = np.zeros(N)
    bucket_keys = set()
    for i in range(N):
        bucket_keys.add((sides[i], qts[i]))

    for bk in sorted(bucket_keys):
        s_side, s_qt = bk
        # Masks for this bucket
        bk_mask = np.array([sides[i] == s_side and qts[i] == s_qt for i in range(N)])
        bk_unlocked = bk_mask & ~locked
        bk_kappa = kappa[bk_mask]
        ul_kappa = kappa[bk_unlocked]

        total_weight = float(np.sum(bk_kappa))
        unlocked_weight = float(np.sum(ul_kappa))

        if total_weight < 1e-12 or unlocked_weight < 1e-12:
            continue

        # Starting wavg skew for this bucket (all bonds)
        start_skew = starting_charge_bps[bk_mask] * bps_to_skew[bk_mask]
        start_wavg = float(np.sum(start_skew * bk_kappa) / total_weight)

        # Post-solver wavg skew for this bucket (all bonds)
        post_skew = final_charge_bps[bk_mask] * bps_to_skew[bk_mask]
        post_wavg = float(np.sum(post_skew * bk_kappa) / total_weight)

        drift = post_wavg - start_wavg
        if abs(drift) < 1e-8:
            continue

        # Shift needed on each unlocked bond (in skew units)
        shift_skew = -drift * total_weight / unlocked_weight
        # Convert to charge bps: for PX, bps = skew * 100; for SPD, bps = skew
        shift_bps = shift_skew / bps_to_skew[bk_unlocked][0] if len(bps_to_skew[bk_unlocked]) else shift_skew

        # Apply
        ul_indices = np.where(bk_unlocked)[0]
        for idx in ul_indices:
            final_charge_bps[idx] += shift_bps
            final_proceeds_arr[idx] = signs[idx] * final_charge_bps[idx] * kappa[idx]
            anchor_adj[idx] = shift_skew

        if debug:
            new_skew = final_charge_bps[bk_mask] * bps_to_skew[bk_mask]
            new_wavg = float(np.sum(new_skew * bk_kappa) / total_weight)
            print(f"[AnchorWAVG] {s_side} {s_qt}: start={start_wavg:.4f} → solver={post_wavg:.4f} → anchored={new_wavg:.4f}  (shift={shift_skew:.4f})")

    return anchor_adj


def solve(df: pl.DataFrame, cfg: OptimizerConfig | None = None, name=None):
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    cfg = cfg or OptimizerConfig()

    print(asdict(cfg))

    inner_col, orig_group_cols = _resolve_inner_col(cfg)

    if cfg.isolate_traders:
        return _solve_isolated(df, cfg)

    removed_ids = {}

    for _id in df.filter(pl.col('isReal')==0).hyper.ul(_c('bond_id')):
        removed_ids[_id] = 'Bond removed from portfolio'
    df = df.filter(pl.col('isReal')==1)

    for _id in df.filter(~pl.col(_c("quote_type")).is_in(['PX', 'SPD'])).hyper.ul(_c('bond_id')):
        removed_ids[_id] = 'Quote Type not supported'
    df = df.filter(pl.col(_c("quote_type")).is_in(['PX', 'SPD']))

    s = df.hyper.schema()
    px_sources = [col for col in PX_SOURCES if col in s]
    spd_sources = [col for col in SPD_SOURCES if col in s]

    my_px_source_bid = [col for col in MY_PX_SOURCE_BID if col in s]
    my_px_source_ask = [col for col in MY_PX_SOURCE_ASK if col in s]
    my_spd_source_bid = [col for col in MY_SPD_SOURCE_BID if col in s]
    my_spd_source_ask = [col for col in MY_SPD_SOURCE_ASK if col in s]

    df = df.with_columns(
        [
            pl.when(IS_BUY).then(pl.coalesce(my_px_source_bid))
            .otherwise(pl.coalesce(my_px_source_ask)).alias('_my_quotePx'),
            pl.when(IS_BUY).then(pl.coalesce(my_spd_source_bid))
            .otherwise(pl.coalesce(my_spd_source_ask)).alias('_my_quoteSpd')
        ]
    )

    df = df.with_columns(
        [
            (pl.col('_my_quoteSpd') - pl.col('_refMidSpd')).alias('skewSpd'),
            (pl.col('_my_quotePx') - pl.col('_refMidPx')).alias('skewPx'),
        ]
    ).with_columns(
        [
            pl.when(IS_SPREAD).then(pl.col('skewSpd')).otherwise(pl.col('skewPx')).alias('skew')
        ]
    )

    df = add_adjusted_bsr_bsi(df, clamp_liq=True,
                              charge_strength=cfg.charge_strength,
                              default_strength=cfg.default_strength)

    df = df.with_columns(
        [
            pl.when(IS_SPREAD).then(pl.col(_c('dv01'))).otherwise(pl.col(_c('size')) / 10_000).alias('kappa'),
            pl.when(IS_PX & IS_SELL).then(pl.lit(1))
            .when(IS_SPREAD & IS_SELL).then(pl.lit(-1))
            .when(IS_PX & IS_BUY).then(pl.lit(-1))
            .when(IS_SPREAD & IS_BUY).then(pl.lit(1)).alias('quoteSign'),

            pl.when(IS_BUY).then(pl.lit(1)).otherwise(pl.lit(-1)).alias('sideSign'),

            pl.when(IS_SPREAD & IS_SELL).then((pl.col('_refMidSpd') - pl.col('_my_quoteSpd')) * pl.col(_c('dv01')))
            .when(IS_SPREAD & IS_BUY).then((pl.col('_my_quoteSpd') - pl.col('_refMidSpd')) * pl.col(_c('dv01')))
            .when(IS_PX & IS_SELL).then((pl.col('_my_quotePx') - pl.col('_refMidPx')) * pl.col(_c('size')) / 100)
            .when(IS_PX & IS_BUY).then((pl.col('_refMidPx') - pl.col('_my_quotePx')) * pl.col(_c('size')) / 100)
            .alias('skewProceeds'),

            (
                    pl.when(pl.col(_c('side'))=='BUY')
                    .then(pl.lit(1))
                    .otherwise(pl.lit(-1)) *
                    pl.col(_c('dv01')) *
                    (pl.lit(11).sqrt() - pl.col(_c('liq_score')).sqrt()) *
                    pl.col(_c('ref_mid_spd')) / 100
            ).alias('trader_risk')

        ]
    )

    for _id in df.filter(pl.col('trader_risk').is_null()).hyper.ul(_c('bond_id')):
        removed_ids[_id] = "Insuffient data"
    df = df.filter(pl.col('trader_risk').is_not_null())

    if orig_group_cols is not None:
        df = _add_group_key_column(df, orig_group_cols)

    df = df.with_columns(
        [
            (pl.col('trader_risk')).alias('trader_net'),
            (pl.col('trader_risk').abs()).alias('trader_gross'),
            (pl.col('trader_risk') * pl.col('bsrPct')).abs().alias('trader_bsr_gross'),
            (pl.col('trader_risk') * pl.col('bsrPct')).alias('trader_bsr'),
            (pl.col('trader_risk') * pl.col('bsiPct')).abs().alias('trader_bsi_gross'),
            (pl.col('trader_risk') * pl.col('bsiPct')).alias('trader_bsi'),
        ]
    )

    N = df.hyper.height()
    if N==0:
        return pl.DataFrame(), {"status": 'FAILED'}, pl.DataFrame(), pl.DataFrame(), removed_ids

    agg_dimenstion = ["quoteType", "side"]
    trader_agg = df.group_by([inner_col] + agg_dimenstion).agg(
        [
            pl.col("trader_net").abs().sum().alias('trader_net_sum'),
            pl.col("trader_gross").abs().sum().alias('trader_gross_sum'),
            pl.col("trader_bsr_gross").abs().sum().alias('trader_bsr_gross_sum'),
            pl.col("trader_bsr").sum().alias('trader_bsr_sum'),
            pl.col("trader_bsi_gross").abs().sum().alias('trader_bsi_gross_sum'),
            pl.col("trader_bsi").sum().alias('trader_bsi_sum'),
            pl.col('skewProceeds').sum().alias('skewProceeds_sum'),
            (pl.col('skewProceeds') * (pl.col('isLocked'))).sum().alias('currentLocked_sum'),
            (pl.col('isLocked').sum() / pl.col('isLocked').count()).alias('pctLocked')
        ]
    ).with_columns(
        [
            pl.col("trader_net_sum").abs().sum().over(agg_dimenstion).alias('net_over_trader_qt'),
            pl.col("trader_gross_sum").abs().sum().over(agg_dimenstion).alias('gross_over_trader_qt'),
            pl.col("trader_bsr_sum").abs().sum().over(agg_dimenstion).alias('net_bsr_over_trader_qt'),
            pl.col("trader_bsr_gross_sum").abs().sum().over(agg_dimenstion).alias('gross_bsr_over_trader_qt'),
            pl.col("trader_bsi_sum").abs().sum().over(agg_dimenstion).alias('net_bsi_over_trader_qt'),
            pl.col("trader_bsi_gross_sum").abs().sum().over(agg_dimenstion).alias('gross_bsi_over_trader_qt'),
        ]
    ).with_columns(
        [
            pl.when(pl.col('net_over_trader_qt').fill_null(0) > 0).then(
                pl.lit(cfg.risk_weights.get('NET', 0), pl.Float64)
                ).otherwise(pl.lit(0, pl.Float64)).alias('net_weight'),
            pl.when(pl.col('gross_over_trader_qt').fill_null(0) > 0).then(
                pl.lit(cfg.risk_weights.get('GROSS', 0), pl.Float64)
                ).otherwise(pl.lit(0, pl.Float64)).alias('gross_weight'),
            pl.when(pl.col('net_bsi_over_trader_qt').fill_null(0) > 0).then(
                pl.lit(cfg.risk_weights.get('NET_BSI', 0), pl.Float64)
                ).otherwise(pl.lit(0, pl.Float64)).alias('net_bsi_weight'),
            pl.when(pl.col('gross_bsi_over_trader_qt').fill_null(0) > 0).then(
                pl.lit(cfg.risk_weights.get('GROSS_BSI', 0), pl.Float64)
                ).otherwise(pl.lit(0, pl.Float64)).alias('gross_bsi_weight')
        ]
    ).with_columns(
        [
            (pl.col('net_weight') + pl.col('gross_weight') + pl.col('net_bsi_weight') + pl.col('gross_bsi_weight')).alias('_total_weight')
        ]
    ).with_columns(
        [
            (pl.col('net_weight') / pl.col('_total_weight')).alias('net_weight'),
            (pl.col('gross_weight') / pl.col('_total_weight')).alias('gross_weight'),
            (pl.col('net_bsi_weight') / pl.col('_total_weight')).alias('net_bsi_weight'),
            (pl.col('gross_bsi_weight') / pl.col('_total_weight')).alias('gross_bsi_weight'),
        ]
    ).with_columns(
        [
            ((pl.col("trader_net_sum").abs()) / (pl.col('net_over_trader_qt'))).fill_nan(0).alias('net_size_pct'),
            ((pl.col("trader_gross_sum").abs()) / (pl.col('gross_over_trader_qt'))).fill_nan(0).alias('gross_size_pct'),
            ((pl.col("trader_bsr_sum").abs()) / (pl.col('net_bsr_over_trader_qt'))).fill_nan(0).alias('bsr_size_pct'),
            ((pl.col("trader_bsr_gross_sum").abs()) / (pl.col('gross_bsr_over_trader_qt'))).fill_nan(0).alias(
                'bsr_gross_size_pct'
                ),
            ((pl.col("trader_bsi_sum").abs()) / (pl.col('net_bsi_over_trader_qt'))).fill_nan(0).alias('bsi_size_pct'),
            ((pl.col("trader_bsi_gross_sum").abs()) / (pl.col('gross_bsi_over_trader_qt'))).fill_nan(0).alias(
                'bsi_gross_size_pct'
                ),
            (pl.col("skewProceeds_sum").sum().over(agg_dimenstion)).fill_nan(0).alias('skewProceeds_sum_sum'),
            (pl.col("currentLocked_sum").sum().over(agg_dimenstion)).fill_nan(0).alias('currentLocked_sum_sum'),
            pl.when(pl.col('pctLocked')==1).then(pl.lit(1, pl.Int8)).otherwise(pl.lit(0, pl.Int8)).alias(
                'isFullyLocked'
                )
        ]
    ).with_columns(
        [
            # FIX: Use the trader's OWN BSR percentage as preference, not (1 - bsi_share_of_total).
            # Old formula: (1 - bsi_size_pct) confused "small trader" with "high BSR trader".
            # A trader with 0% BSR but tiny DV01 would get MAX preference — completely wrong.
            # New: preference = trader's BSR% of their own book, from the aggregation.
            (pl.col('trader_bsr_gross_sum') / (pl.col('trader_bsr_gross_sum') + pl.col('trader_bsi_gross_sum')))
                .fill_null(0).fill_nan(0).clip(0.0, 1.0).alias('_own_bsr_pct'),
        ]
    ).with_columns(
        [
            # Normalize own BSR% to a share across traders within each (side, qt) bucket
            (pl.col('_own_bsr_pct') / pl.col('_own_bsr_pct').sum().over(agg_dimenstion)).fill_nan(0).alias('bsr_pref_size_pct'),
            (pl.col('_own_bsr_pct') / pl.col('_own_bsr_pct').sum().over(agg_dimenstion)).fill_nan(0).alias('bsr_pref_gross_size_pct'),
        ]
    ).with_columns(
        [
            (pl.lit(cfg.bsr_risk_share_weight) * pl.col('bsr_size_pct') + pl.lit(cfg.bsr_preference_weight) * pl.col('bsr_pref_size_pct')).alias('bsr_target_size_pct'),
            (pl.lit(cfg.bsr_risk_share_weight) * pl.col('bsr_gross_size_pct') + pl.lit(cfg.bsr_preference_weight) * pl.col('bsr_pref_gross_size_pct')).alias('bsr_target_gross_size_pct'),
        ]
    ).with_columns(
        [
            (pl.col('net_size_pct') * pl.col('net_weight') + pl.col('gross_size_pct') * pl.col('gross_weight') + pl.col('bsr_target_size_pct') * pl.col('net_bsi_weight') + pl.col('bsr_target_gross_size_pct') * pl.col('gross_bsi_weight')).alias('wavg_pct_by_side_qt')
        ]
    ).with_columns(
        [
            # Blend between risk-weighted target and starting allocation.
            # target_blend=0: keep starting. target_blend=1: full risk reallocation.
            (
                pl.lit(cfg.target_blend) * (pl.col('wavg_pct_by_side_qt') * pl.col('skewProceeds_sum_sum'))
                + pl.lit(1.0 - cfg.target_blend) * pl.col('skewProceeds_sum')
            ).alias('expected_proceeds')
        ]
    )

    # Extract per-trader risk_pct for output columns
    _risk_pct_map_raw = trader_agg.select(
        pl.concat_arr([pl.col(inner_col), pl.col(_c('side')), pl.col(_c('quote_type'))]).alias('_key'),
        pl.col('wavg_pct_by_side_qt')
    ).collect().hyper.to_map('_key', 'wavg_pct_by_side_qt')
    _risk_pct_for_bonds = {
        tuple(k) if isinstance(k, (list, np.ndarray)) else k: v
        for k, v in (_risk_pct_map_raw or {}).items()
    }

    sides = df.hyper.to_list(_c('side'))
    qts = df.hyper.to_list(_c('quote_type'))
    inner_vals = df.hyper.to_list(inner_col)
    locked = df.select(pl.col('isLocked').cast(pl.Boolean, strict=False)).collect().to_series().to_numpy()
    skew = df.select(pl.col('skew').cast(pl.Float64, strict=False)).collect().to_series().to_numpy().copy()
    kappa = df.select(pl.col('kappa').cast(pl.Float64, strict=False)).collect().to_series().to_numpy()
    signs = df.select(pl.col('quoteSign').cast(pl.Int8, strict=False)).collect().to_series().to_numpy()

    bucket_keys_s: set[tuple[str, str]] = set()
    tbk_keys_s: set[tuple[str, str, str]] = set()
    bond_bk: list[tuple[str, str]] = []
    bond_tbk: list[tuple[str, str, str]] = []
    for i in range(N):
        bk = (sides[i], qts[i])
        tbk = (inner_vals[i] or 'MISSING', sides[i], qts[i])
        bucket_keys_s.add(bk)
        tbk_keys_s.add(tbk)
        bond_bk.append(bk)
        bond_tbk.append(tbk)

    bucket_keys = sorted(bucket_keys_s)
    tbk_keys = sorted(tbk_keys_s)
    bk_idx = {bk: j for j, bk in enumerate(bucket_keys)}
    tbk_idx = {tbk: j for j, tbk in enumerate(tbk_keys)}
    n_bk, n_tbk = len(bucket_keys), len(tbk_keys)

    # Per-bond risk_pct: distribute each trader's bucket share across their
    # individual bonds proportionally by |trader_risk|.
    # Result sums to 1.0 per (side, qt) bucket.
    try:
        trader_risk_abs = df.select(pl.col('trader_risk').abs().cast(pl.Float64)).collect().to_series().to_numpy()
        tbk_risk_totals: dict[tuple, float] = {}
        for i in range(N):
            tbk = bond_tbk[i]
            tbk_risk_totals[tbk] = tbk_risk_totals.get(tbk, 0.0) + trader_risk_abs[i]
        bond_risk_pct = np.array([
            _risk_pct_for_bonds.get(bond_tbk[i], 0.0)
            * (trader_risk_abs[i] / tbk_risk_totals[bond_tbk[i]] if tbk_risk_totals.get(bond_tbk[i], 0) > 1e-12 else 0.0)
            for i in range(N)
        ])
    except Exception:
        bond_risk_pct = np.zeros(N)

    unlocked = ~locked
    ul_idx = np.where(unlocked)[0]
    lk_idx = np.where(locked)[0]
    n_ul = len(ul_idx)
    u_sides = df.hyper.ul(_c('side'))

    proceeds = df.select('skewProceeds').collect().to_series().to_numpy().copy()
    is_px = np.array([q == "PX" for q in qts])
    starting_charge_bps = np.where(is_px, skew * 100.0, skew)
    blended_raw = df.select('blendedCharge').cast(pl.Float64, strict=False).collect().to_series().to_numpy()
    bps_to_skew = np.where(is_px, 1.0 / 100.0, 1.0)

    # ── Apply "Run From X" override if set ────────────────────────────
    if cfg.starting_skew_override:
        print(f"[RunFromX] Override received: {cfg.starting_skew_override}")
        applied = 0
        for i in range(N):
            bk = (sides[i], qts[i])
            if bk in cfg.starting_skew_override:
                override_val = float(cfg.starting_skew_override[bk])
                starting_charge_bps[i] = override_val * 100.0 if is_px[i] else override_val
                skew[i] = override_val  # update skew for display
                proceeds[i] = signs[i] * starting_charge_bps[i] * kappa[i]
                applied += 1
        print(f"[RunFromX] Applied to {applied}/{N} bonds")
        # Update the DataFrame so downstream aggregations use overridden skew/proceeds
        df = df.collect().with_columns([
            pl.Series("skew", skew),
            pl.Series("skewProceeds", proceeds),
        ]).lazy()

    start_by_side: dict[str, float] = {}
    for s in u_sides:
        m = np.array([sides[i] == s for i in range(N)])
        start_by_side[s] = float(np.sum(proceeds[m]))

    # ── Extract trader targets from trader_agg ─────────────────────────
    expected_proceeds_raw = trader_agg.select(
        pl.concat_arr([pl.col(inner_col), pl.col(_c('side')), pl.col(_c('quote_type'))]).alias('_key'),
        pl.col('expected_proceeds')
    ).collect().hyper.to_map('_key', 'expected_proceeds')
    expected_proceeds_map = {
        tuple(k) if isinstance(k, (list, np.ndarray)) else k: v
        for k, v in (expected_proceeds_raw or {}).items()
    }

    risk_pct_raw = trader_agg.select(
        pl.concat_arr([pl.col(inner_col), pl.col(_c('side')), pl.col(_c('quote_type'))]).alias('_key'),
        pl.col('wavg_pct_by_side_qt')
    ).collect().hyper.to_map('_key', 'wavg_pct_by_side_qt')
    risk_pct = {
        tuple(k) if isinstance(k, (list, np.ndarray)) else k: v
        for k, v in (risk_pct_raw or {}).items()
    }

    # ══════════════════════════════════════════════════════════════════════
    #  OPTIMIZATION: Direct charge variables
    #
    #  charge[j] = final charge in bps for unlocked bond j
    #  proceeds_j = sign_j × charge_j × kappa_j
    #
    #  Maximize: total_proceeds
    #           - lambda × Σ (charge_j - ideal_j)²         (curve adherence)
    #           - gamma  × Σ ((trader_sum - target) / norm)²  (target pull)
    #
    #  ideal_j = blended_j × (bucket_avg_start / bucket_avg_blended)
    #  → charge_strength defines WHERE each bond should be, not just move cost
    #  → BSR charge=0 → ideal≈average → BSR bonds pulled toward flat
    #  → BSR charge high → ideal≈wide → BSR bonds pulled toward wide skews
    #
    #  Subject to: side band, optional through-mid, optional skew caps
    # ══════════════════════════════════════════════════════════════════════

    charge = cp.Variable(n_ul, name="charge")

    signs_ul = signs[ul_idx].astype(float)
    kappa_ul = kappa[ul_idx].astype(float)
    start_ul = starting_charge_bps[ul_idx].astype(float)

    # Per-bond proceeds as function of charge
    ul_proceeds = cp.multiply(signs_ul * kappa_ul, charge)

    # Locked proceeds (constant)
    locked_proceeds_total = float(np.sum(proceeds[lk_idx])) if len(lk_idx) else 0.0

    # ── Curve-derived ideal charges ───────────────────────────────────
    #  ideal[i] = blended[i] × (avg_start / avg_blended) per bucket
    #  Preserves bucket total while reshaping distribution by charge curve.
    #  Clamped to the correct side of mid: if sign=+1 (SELL PX, BUY SPD),
    #  ideal ≥ 0; if sign=-1 (BUY PX, SELL SPD), ideal ≤ 0.
    #  This prevents gamma from reinforcing through-mid positions.
    blended_ul = blended_raw[ul_idx].astype(float)
    ideal_charge = np.copy(start_ul)
    for bk in set(bond_bk):
        bk_ul_mask = np.array([bond_bk[ul_idx[j]] == bk for j in range(n_ul)])
        if not bk_ul_mask.any():
            continue
        bk_starts = start_ul[bk_ul_mask]
        bk_blended = blended_ul[bk_ul_mask]
        avg_start = np.mean(bk_starts) if len(bk_starts) else 1.0
        avg_blended = np.mean(bk_blended) if len(bk_blended) else 1.0
        if abs(avg_blended) < 0.01:
            avg_blended = 1.0
        ideal_charge[bk_ul_mask] = bk_blended * (avg_start / avg_blended)

    # Clamp ideals to correct side of mid — don't let gamma pull through mid
    ideal_charge = np.where(signs_ul > 0, np.maximum(ideal_charge, 0.0),
                                           np.minimum(ideal_charge, 0.0))

    # ── TWO COMPETING PENALTIES ─────────────────────────────────────────
    #  Both weighted by kappa to convert bps → $, matching the raw $ objective.
    #  L2 (quadratic) penalties give GRADUAL resistance: small moves are cheap,
    #  big moves are expensive. Break-even at δ = 1/λ bps per bond.
    #  λ=0.25 → ~4 bps room, λ=1 → ~1 bps, λ=3 → locked tight.
    #  (L1 was too sharp — threshold switch at λ=1 with no gradual middle ground.)
    l = cfg.lambda_param or 0.0
    g = cfg.gamma or 0.0

    # Guard: if all penalty strengths are zero, the objective degenerates to
    # pure PnL maximization — the solver bangs every bond to the side floor.
    # Inject a large anchoring penalty so bonds stay at their starting values.
    if l == 0 and g == 0 and cfg.trader_pull == 0:
        l = 1e6

    delta_from_start = charge - start_ul
    delta_from_ideal = charge - ideal_charge

    smoothness_penalty = l * cp.sum(cp.multiply(kappa_ul, cp.square(delta_from_start)))
    curve_penalty = g * cp.sum(cp.multiply(kappa_ul, cp.square(delta_from_ideal)))

    # Asymmetry: extra penalty for compressing charge toward zero
    asym_penalty = 0.0
    if cfg.skew_asymmetry > 0:
        abs_start = np.abs(start_ul)
        compression = cp.pos(abs_start - cp.abs(charge))
        asym_penalty = cfg.skew_asymmetry * l * cp.sum(
            cp.multiply(kappa_ul, cp.square(compression))
        )

    # ── Auto-relax tiers ──────────────────────────────────────────────
    tiers = [
        {"desc": "Strict", "buffer_mult": 1.0},
        {"desc": "Expand (2x)", "buffer_mult": 2.0},
        {"desc": "Expand (4x)", "buffer_mult": 4.0},
    ] if cfg.auto_relax else [{"desc": "Strict", "buffer_mult": 1.0}]

    is_ok_final = False
    prob = None

    for attempt, tier in enumerate(tiers):
        log.notify(f'Running solver tier {attempt}: {tier["desc"]}')
        constraints: list[cp.Constraint] = []
        # Constraint metadata for attribution: list of (constraint, type, scope)
        # scope is either a side string, a tbk tuple, or a bond index j
        _constraint_meta: list[tuple] = []

        # ── Side band constraints ─────────────────────────────────────
        for s in u_sides:
            start_val = start_by_side.get(s, 0)
            if start_val == 0:
                continue

            side_mask = np.array([sides[ul_idx[j]] == s for j in range(n_ul)])
            lk_side = float(np.sum(proceeds[lk_idx[np.array([sides[i] == s for i in lk_idx])]])) if len(lk_idx) else 0.0

            # ul_proceeds = sign * kappa * charge = FINAL proceeds per bond (not delta)
            side_total = (cp.sum(ul_proceeds[side_mask]) if side_mask.any() else 0.0) + lk_side

            if start_val >= 0:
                floor_val = cfg.side_floor_pct * start_val
                ceil_val = start_val / cfg.side_floor_pct
            else:
                floor_val = start_val / cfg.side_floor_pct
                ceil_val = cfg.side_floor_pct * start_val

            constraints.append(side_total >= floor_val)
            _constraint_meta.append((constraints[-1], 'side_floor', s))
            constraints.append(side_total <= ceil_val)
            _constraint_meta.append((constraints[-1], 'side_ceil', s))

        # ── Gamma penalty: soft pull toward trader targets ────────────
        gamma_terms = []
        for tbk in tbk_keys:
            t, s, q = tbk
            target = expected_proceeds_map.get(tbk, 0.0)
            target_norm = max(abs(target), 1.0)

            # Unlocked bonds in this trader-bucket
            tbk_ul_mask = np.array([bond_tbk[ul_idx[j]] == tbk for j in range(n_ul)])
            # Locked proceeds for this trader-bucket
            lk_tbk = [i for i in lk_idx if bond_tbk[i] == tbk]
            lk_tbk_val = float(np.sum(proceeds[lk_tbk])) if lk_tbk else 0.0
            # Starting proceeds for this trader-bucket (all bonds)
            all_tbk = [i for i in range(N) if bond_tbk[i] == tbk]
            start_tbk = float(np.sum(proceeds[all_tbk])) if all_tbk else 0.0

            trader_total = (cp.sum(ul_proceeds[tbk_ul_mask]) if tbk_ul_mask.any() else 0.0) + lk_tbk_val

            gamma_terms.append(cp.abs(trader_total - target) / target_norm)

            # Hard guardrails (wide SAFETY band, not the redistribution driver)
            # Width based on max(|target|, |starting|) so the band is always wide
            # enough to cover whichever direction the optimizer needs to move.
            # When soft penalties are active (trader_pull > 0), widen further —
            # let the soft penalty do redistribution, hard bands catch extremes only.
            if tbk_ul_mask.any():
                reference = max(abs(target), abs(start_tbk), abs(target - start_tbk))
                if cfg.buffer_mode == 'pct':
                    buf = max(reference * cfg.buffer_pct, cfg.buffer_min_flat)
                else:
                    buf = cfg.buffer_fixed
                # When soft penalties drive redistribution, hard bands are safety-only
                if cfg.trader_pull > 0 or g > 0:
                    buf = max(buf, abs(target - start_tbk) * 1.5)
                buf *= tier["buffer_mult"]
                constraints.append(trader_total <= target + buf)
                _constraint_meta.append((constraints[-1], 'trader_ceil', tbk))
                constraints.append(trader_total >= target - buf)
                _constraint_meta.append((constraints[-1], 'trader_floor', tbk))

        gamma_penalty = 0.0
        if gamma_terms and cfg.trader_pull > 0:
            gamma_penalty = cfg.trader_pull * cp.sum(cp.hstack(gamma_terms))

        # ── Through-mid constraints (optional) ────────────────────────
        if not cfg.allow_through_mid:
            for j, i in enumerate(ul_idx):
                if bond_bk[i] in (("SELL", "PX"), ("BUY", "SPD")):
                    constraints.append(charge[j] >= 0)
                    _constraint_meta.append((constraints[-1], 'mid', j))
                else:
                    constraints.append(charge[j] <= 0)
                    _constraint_meta.append((constraints[-1], 'mid', j))

        # ── Per-bond skew delta caps (optional) ──────────────────────
        for j, i in enumerate(ul_idx):
            q = qts[i]
            if q == "SPD" and cfg.max_individual_spread_skew_delta is not None:
                cap = cfg.max_individual_spread_skew_delta
                constraints.append(charge[j] - start_ul[j] <= cap)
                _constraint_meta.append((constraints[-1], 'cap_upper', j))
                constraints.append(charge[j] - start_ul[j] >= -cap)
                _constraint_meta.append((constraints[-1], 'cap_lower', j))
            elif q == "PX" and cfg.max_individual_px_skew_delta is not None:
                cap_bps = cfg.max_individual_px_skew_delta * 100.0
                constraints.append(charge[j] - start_ul[j] <= cap_bps)
                _constraint_meta.append((constraints[-1], 'cap_upper', j))
                constraints.append(charge[j] - start_ul[j] >= -cap_bps)
                _constraint_meta.append((constraints[-1], 'cap_lower', j))

        # ── Construct objective ───────────────────────────────────────
        total_proceeds_expr = cp.sum(ul_proceeds) + locked_proceeds_total
        objective = cp.Maximize(
            total_proceeds_expr - smoothness_penalty - curve_penalty - asym_penalty - gamma_penalty
        )

        prob = cp.Problem(objective, constraints)
        for slv in [cfg.solver, "SCS", "ECOS", "CLARABEL"]:
            kw = _SOLVER_KW.get(slv, {})
            try:
                prob.solve(solver=slv, verbose=cfg.solver_verbose, **kw)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    is_ok_final = True
                    break
            except Exception:
                pass

        if is_ok_final:
            if cfg.debug and attempt > 0:
                log.warning(f"Solved via auto-relax ({tier['desc']}) after {attempt} failures.")
            break

    # ══════════════════════════════════════════════════════════════════════
    #  POST-PROCESS
    # ══════════════════════════════════════════════════════════════════════

    if not is_ok_final:
        if name is None:
            log.error("Optimization failed — returning starting values as fallback.")
        else:
            log.error(f"Optimization for {name} failed — returning starting values.")
        # ── Infeasibility diagnosis ──────────────────────────────────────
        try:
            _diag_reasons = []
            _n_locked = int(np.sum(locked))
            _pct_locked = _n_locked / N * 100 if N else 0
            if _pct_locked > 80:
                _diag_reasons.append(f"{_pct_locked:.0f}% of bonds are locked — very little room for the solver to move.")
            for s in u_sides:
                s_mask = np.array([sides[i] == s for i in range(N)])
                s_locked = np.array([locked[i] and sides[i] == s for i in range(N)])
                locked_proceeds_s = float(np.sum(proceeds[s_locked]))
                total_proceeds_s = start_by_side.get(s, 0)
                if total_proceeds_s != 0:
                    locked_pct = abs(locked_proceeds_s / total_proceeds_s) * 100
                    floor = cfg.side_floor_pct * abs(total_proceeds_s)
                    unlocked_proceeds_s = abs(total_proceeds_s) - abs(locked_proceeds_s)
                    if unlocked_proceeds_s < abs(total_proceeds_s) * (1 - cfg.side_floor_pct):
                        _diag_reasons.append(
                            f"{s}: Locked bonds hold {locked_pct:.0f}% of proceeds, leaving too little "
                            f"room within the {cfg.side_floor_pct:.0%} side floor constraint."
                        )
            for tbk in tbk_keys:
                tbk_ul = [i for i in range(N) if bond_tbk[i] == tbk and not locked[i]]
                tbk_lk = [i for i in range(N) if bond_tbk[i] == tbk and locked[i]]
                if len(tbk_ul) == 0 and len(tbk_lk) > 0:
                    t, s, q = tbk
                    _diag_reasons.append(f"Trader {t} ({s} {q}): all {len(tbk_lk)} bonds are locked.")
            if not _diag_reasons:
                _diag_reasons.append("No specific cause identified — try relaxing constraints or unlocking bonds.")
            diagnostics = {
                "status": "FAILED",
                "bonds_total": N, "bonds_unlocked": int(N - _n_locked),
                "infeasibility_reasons": _diag_reasons,
            }
        except Exception:
            diagnostics = {"status": "FAILED", "bonds_total": N}

    # Extract optimal charges
    final_charge_bps = np.copy(starting_charge_bps)
    final_proceeds_arr = np.copy(proceeds)
    anchor_adj_arr = np.zeros(N)
    _bond_constraint = [''] * N
    diagnostics = {}
    if is_ok_final and charge.value is not None:
        optimal = charge.value
        for j, i in enumerate(ul_idx):
            final_charge_bps[i] = optimal[j]
            final_proceeds_arr[i] = signs[i] * optimal[j] * kappa[i]

        # ── Anchor WAVG skew correction (post-solver parallel shift) ─────
        if cfg.anchor_wavg_skew:
            anchor_adj_arr = _apply_anchor_wavg_skew(
                final_charge_bps, final_proceeds_arr, starting_charge_bps,
                signs, kappa, sides, qts, locked, is_px, bps_to_skew,
                debug=cfg.debug,
            )
            # Re-read optimal for diagnostics after anchoring
            for j, i in enumerate(ul_idx):
                optimal[j] = final_charge_bps[i]

        # ── Constraint attribution: identify binding constraints per bond ──
        # For each unlocked bond, check which constraint (if any) is binding.
        # A constraint is considered binding when its slack (residual) is near zero.
        _SLACK_TOL = 1e-4
        try:
            # Side-level and trader-level constraints: binding = affects all unlocked bonds in scope
            _binding_sides = set()
            _binding_tbks = set()
            for constr, ctype, scope in _constraint_meta:
                dv = constr.dual_value
                if dv is None:
                    continue
                is_binding = (np.abs(dv) > _SLACK_TOL) if np.isscalar(dv) else (np.any(np.abs(dv) > _SLACK_TOL))
                if not is_binding:
                    continue
                if ctype in ('side_floor', 'side_ceil'):
                    _binding_sides.add(scope)
                elif ctype in ('trader_floor', 'trader_ceil'):
                    _binding_tbks.add(scope)
                elif ctype in ('mid', 'cap_upper', 'cap_lower'):
                    j = scope  # unlocked index
                    i = ul_idx[j]  # global index
                    label = 'mid' if ctype == 'mid' else 'cap'
                    if not _bond_constraint[i]:
                        _bond_constraint[i] = label

            # Attribute group-level binding constraints to individual bonds
            for j, i in enumerate(ul_idx):
                if _bond_constraint[i]:
                    continue  # per-bond constraint takes priority
                bk = bond_bk[i]
                tbk = bond_tbk[i]
                if tbk in _binding_tbks:
                    _bond_constraint[i] = 'trader'
                elif bk[0] in _binding_sides:
                    _bond_constraint[i] = 'side'
        except Exception:
            pass  # attribution is best-effort, don't break the solve

        # ── Build diagnostics dict ───────────────────────────────────────
        _deltas = optimal - start_ul
        _abs_deltas = np.abs(_deltas)
        _ideal_deltas = optimal - ideal_charge
        _sm_val = smoothness_penalty.value if hasattr(smoothness_penalty, 'value') and smoothness_penalty.value is not None else 0
        _cv_val = curve_penalty.value if hasattr(curve_penalty, 'value') and curve_penalty.value is not None else 0
        _gm_val = gamma_penalty.value if hasattr(gamma_penalty, 'value') and gamma_penalty.value is not None else 0
        _raw_proceeds = float(np.sum(final_proceeds_arr))
        _start_proceeds = float(np.sum(proceeds))
        _avg_kappa = float(np.mean(kappa_ul))

        diagnostics = {
            "status": "optimal" if prob and prob.status in ("optimal", "optimal_inaccurate") else (prob.status if prob else "FAILED"),
            "bonds_total": N, "bonds_unlocked": n_ul, "bonds_locked": len(lk_idx),
            "lambda": round(l, 3), "gamma": round(g, 3), "trader_pull": round(cfg.trader_pull, 3),
            "raw_proceeds": round(_raw_proceeds, 0),
            "starting_proceeds": round(_start_proceeds, 0),
            "pnl_delta": round(_raw_proceeds - _start_proceeds, 0),
            "smoothness_penalty": round(float(_sm_val), 1),
            "curve_penalty": round(float(_cv_val), 1),
            "trader_penalty": round(float(_gm_val), 1),
            "avg_delta": round(float(np.mean(_abs_deltas)), 4),
            "max_delta": round(float(np.max(_abs_deltas)), 4),
            "avg_from_ideal": round(float(np.mean(np.abs(_ideal_deltas))), 4),
            "bonds_moved_01": int(np.sum(_abs_deltas > 0.01)),
            "bonds_moved_1": int(np.sum(_abs_deltas > 0.1)),
            "avg_kappa": round(_avg_kappa, 0),
            "proceeds_per_bps": round(_avg_kappa, 0),
            "lambda_cost_per_bps": round(l * _avg_kappa, 0),
            "gamma_cost_per_bps": round(g * _avg_kappa, 0),
            "anchoring_active": l * _avg_kappa > _avg_kappa,
        }

        # Print summary for server logs
        print(f"\n{'='*70}")
        print(f"SOLVER DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"  Status: {diagnostics['status']}  |  Bonds: {N} ({n_ul} unlocked)")
        print(f"  Config: λ={l:.3f}, γ={g:.3f}, tp={cfg.trader_pull:.3f}")
        print(f"  PnL: {diagnostics['raw_proceeds']:+,.0f}  (Δ {diagnostics['pnl_delta']:+,.0f})")
        print(f"  Penalties: λ=-{_sm_val:,.0f}  γ=-{_cv_val:,.0f}  tp=-{_gm_val:,.0f}")
        print(f"  Avg|Δ|={diagnostics['avg_delta']:.4f}  Max|Δ|={diagnostics['max_delta']:.4f}  Moved>0.1bps: {diagnostics['bonds_moved_1']}/{n_ul}")
        print(f"{'='*70}\n")
    else:
        log.warning("Using starting values as fallback (no valid solution).")
        diagnostics = {"status": "FAILED", "bonds_total": N, "bonds_unlocked": n_ul}

    # Mark locked bonds (outside solver success block — always applies)
    for i in lk_idx:
        _bond_constraint[i] = 'locked'

    # ── Decomposition (post-hoc) ──────────────────────────────────────
    #   skew_delta[i] = final_charge - starting_charge (in skew units, includes anchor)
    #   solver_delta[i] = skew_delta - anchor_adj (what the solver alone did)
    #   bucket_effect[i] = avg solver_delta across bonds in trader-bucket
    #   group_effect[i] = solver_delta[i] - bucket_effect[i]
    #   Invariant: bucket_effect + group_effect + anchor_adj = skew_delta (exact)

    charge_delta = final_charge_bps - starting_charge_bps
    skew_delta = charge_delta * bps_to_skew
    # anchor_adj_arr is set above (zeros if anchor disabled)
    solver_delta = skew_delta - anchor_adj_arr

    # Informational: ideal charge (from curve) vs starting
    ideal_full = np.copy(starting_charge_bps)
    for bk in set(bond_bk):
        bk_mask = np.array([bond_bk[i] == bk for i in range(N)])
        bk_s = starting_charge_bps[bk_mask]
        bk_b = blended_raw[bk_mask]
        avg_s = np.mean(bk_s) if len(bk_s) else 1.0
        avg_b = np.mean(bk_b) if len(bk_b) else 1.0
        if abs(avg_b) < 0.01:
            avg_b = 1.0
        ideal_full[bk_mask] = bk_b * (avg_s / avg_b)
    rebase_effect = (ideal_full - starting_charge_bps) * bps_to_skew

    bucket_effect = np.zeros(N)
    group_effect = np.zeros(N)

    # Per trader-bucket average: bucket_effect shows the average shift for this
    # trader in this (side, qt) basket. group_effect shows how much each individual
    # bond deviated from that trader's average (excluding anchor adjustment).
    for tbk in set(bond_tbk):
        tbk_mask = np.array([bond_tbk[i] == tbk for i in range(N)])
        tbk_delta = solver_delta[tbk_mask]
        bucket_effect[tbk_mask] = np.mean(tbk_delta) if tbk_mask.any() else 0.0

    group_effect = solver_delta - bucket_effect

    # Zero out locked bonds
    for i in lk_idx:
        rebase_effect[i] = 0.0
        bucket_effect[i] = 0.0
        group_effect[i] = 0.0

    implied_skew = final_charge_bps * bps_to_skew

    # Implied prices and spreads
    ref_mid_px = df.select(_c('ref_mid_px')).collect().to_series().to_numpy()
    ref_mid_spd = df.select(_c('ref_mid_spd')).collect().to_series().to_numpy()
    side_signs = np.array([1.0 if s == "BUY" else -1.0 for s in sides])
    sizes = df.select(_c('size')).collect().to_series().to_numpy().astype(float)
    implied_px = ref_mid_px - side_signs * (final_proceeds_arr * 100.0 / np.where(sizes > 0, sizes, 1.0))
    implied_spd = np.where(~is_px, ref_mid_spd + final_charge_bps, np.nan)

    # ── Build result objects ──────────────────────────────────────────
    final_by_side: dict[str, float] = {}
    for s in u_sides:
        m = np.array([sides[i] == s for i in range(N)])
        final_by_side[s] = float(np.sum(final_proceeds_arr[m]))

    status = prob.status if prob else 'FAILED'
    if status == 'optimal_inaccurate':
        status = 'optimal'
    if status == 'optimal' and len(removed_ids) > 0:
        status = "partial"

    res = {
        "status": status, "optimal": is_ok_final,
        "objective_value": prob.value if prob and prob.value is not None else float("nan"),
        "starting_proceeds_by_side": start_by_side,
        "final_proceeds_by_side": final_by_side,
        "diagnostics": diagnostics,
    }
    if cfg.debug:
        if is_ok_final:
            if name is None:
                log.success("Redistributed Success", **_sort_dict(res))
            else:
                log.success(f"Redistributed Success for {name}", **_sort_dict(res))

    result = OptimizationResult(
        status=status, optimal=is_ok_final,
        objective_value=prob.value if prob and prob.value is not None else float("nan"),
        X_values={}, Y_values={},
        final_charges=final_charge_bps, final_proceeds=final_proceeds_arr,
        starting_proceeds_by_side=start_by_side, final_proceeds_by_side=final_by_side,
        risk_pct=risk_pct, diagnostics=diagnostics
    )

    quote_spd = df.select(_c('quote_spd')).collect().to_series().to_numpy()
    _result_select = [
        pl.col(_c('bond_id')),
        pl.col(_c('side')),
        pl.col(_c('quote_type')),
        pl.col(_c('size')),
        pl.col(_c('dv01')),
        pl.col(_c('quote_spd')),
        pl.col(_c('quote_px')),
        pl.col(_c('ref_mid_spd')),
        pl.col(_c('ref_mid_px')),
        pl.col(_c('trader')),
        pl.col(_c('liq_score')),
        pl.col(_c('bsr_notional')),
        pl.col(_c('bsi_notional')),
        (pl.col(_c('bsr_notional')) / pl.col(_c('size'))).alias('bsr_pct'),
        pl.Series("skew", np.round(skew, 4)),
        pl.Series("starting_proceeds", np.round(proceeds, 0)),
        pl.Series("blended_charge", np.round(blended_raw, 4)),
        pl.Series("kappa", np.round(kappa, 2)),
        pl.Series("final_charge_bps", np.round(final_charge_bps, 4)),
        pl.Series("final_skew", np.round(implied_skew, 4)),
        pl.Series("skew_delta", np.round(skew_delta, 4)),
        pl.Series("rebase_effect", np.round(rebase_effect, 4)),
        pl.Series("bucket_effect", np.round(bucket_effect, 4)),
        pl.Series("group_effect", np.round(group_effect, 4)),
        pl.Series("anchor_adj", np.round(anchor_adj_arr, 4)),
        pl.Series("risk_pct", np.round(bond_risk_pct, 6)),
        pl.Series("final_proceeds", np.round(final_proceeds_arr, 0)),
        pl.Series("proceeds_delta", np.round(final_proceeds_arr - proceeds, 0)),
        pl.Series("implied_px", np.round(implied_px, 6)),
        pl.Series("implied_spd", np.round(np.nan_to_num(implied_spd, nan=0), 4)),
        pl.Series("spd_delta", np.round(np.where(~is_px, implied_spd - quote_spd, 0), 4)),
        pl.Series("binding_constraint", _bond_constraint),
        pl.col(_c('locked')).alias('isLocked'),
    ]
    if orig_group_cols is not None:
        for gc in orig_group_cols:
            _result_select.append(pl.col(gc))

    df_result = df.select(*_result_select)


    summary_overall = df_result.group_by([_c('side'), _c('quote_type')]).agg(
        [
            pl.col('skew_delta').abs().max().alias('max_skew_delta'),
            pl.col('proceeds_delta').sum().alias('proceeds_delta'),

            pl.col('skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_start_skew_dv01'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_final_skew_dv01'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_abs_skew_delta_dv01'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_skew_delta_dv01'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_rebase_effect_dv01'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_bucket_effect_dv01'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_trader_effect_dv01'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_anchor_adj_dv01'),

            pl.col('skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_start_skew_size'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_final_skew_size'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('size'))).alias('_wavg_abs_skew_delta_size'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('size'))).alias('_wavg_skew_delta_size'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_rebase_effect_size'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_bucket_effect_size'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_trader_effect_size'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('size'))).alias('_wavg_anchor_adj_size'),
        ]
    ).with_columns(
        [
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_start_skew_dv01')).otherwise(
                pl.col('_wavg_start_skew_size')
                ).alias('wavg_start_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_final_skew_dv01')).otherwise(
                pl.col('_wavg_final_skew_size')
                ).alias('wavg_final_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_abs_skew_delta_dv01')).otherwise(
                pl.col('_wavg_abs_skew_delta_size')
                ).alias('wavg_abs_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_skew_delta_dv01')).otherwise(
                pl.col('_wavg_skew_delta_size')
                ).alias('wavg_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_rebase_effect_dv01')).otherwise(
                pl.col('_wavg_rebase_effect_size')
            ).alias('wavg_rebase_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_bucket_effect_dv01')).otherwise(
                pl.col('_wavg_bucket_effect_size')
            ).alias('wavg_bucket_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_trader_effect_dv01')).otherwise(
                pl.col('_wavg_trader_effect_size')
            ).alias('wavg_trader_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_anchor_adj_dv01')).otherwise(
                pl.col('_wavg_anchor_adj_size')
            ).alias('wavg_anchor_adj'),
        ]
    ).drop(
        [
            '_wavg_start_skew_dv01', '_wavg_final_skew_dv01', '_wavg_abs_skew_delta_dv01', '_wavg_skew_delta_dv01',
            '_wavg_start_skew_size', '_wavg_final_skew_size', '_wavg_abs_skew_delta_size', '_wavg_skew_delta_size',
            '_wavg_rebase_effect_dv01', '_wavg_rebase_effect_size',
            '_wavg_bucket_effect_dv01', '_wavg_trader_effect_dv01','_wavg_bucket_effect_size', '_wavg_trader_effect_size',
            '_wavg_anchor_adj_dv01', '_wavg_anchor_adj_size',
        ], strict=False
    )

    _inner_group_by = orig_group_cols if orig_group_cols is not None else [_c('trader')]
    summary_trader = df_result.group_by([_c('side'), _c('quote_type')] + _inner_group_by).agg(
        [
            pl.col('skew_delta').abs().max().alias('max_skew_delta'),
            pl.col('proceeds_delta').sum().alias('proceeds_delta'),
            (pl.col(_c('bsr_notional')).sum() / pl.col(_c('size')).sum()).alias('pct_bsr'),
            (pl.col(_c('bsi_notional')).sum() / pl.col(_c('size')).sum()).alias('pct_bsi'),
            pl.col(_c('quote_type')).count().alias('count'),
            pl.col(_c('size')).sum().alias('size'),
            pl.col(_c('dv01')).sum().alias('dv01'),
            pl.col('risk_pct').sum().alias('risk_pct'),  # bond-level pcts sum to trader's bucket share

            pl.col(_c('liq_score')).hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_liq_score_dv01'),
            pl.col('skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_start_skew_dv01'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_final_skew_dv01'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_abs_skew_delta_dv01'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_skew_delta_dv01'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_rebase_effect_dv01'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_bucket_effect_dv01'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_trader_effect_dv01'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_anchor_adj_dv01'),

            pl.col(_c('liq_score')).hyper.wavg(pl.col(_c('size'))).alias('_wavg_liq_score_size'),
            pl.col('skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_start_skew_size'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_final_skew_size'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('size'))).alias('_wavg_abs_skew_delta_size'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('size'))).alias('_wavg_skew_delta_size'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_rebase_effect_size'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_bucket_effect_size'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_trader_effect_size'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('size'))).alias('_wavg_anchor_adj_size'),

        ]
    ).with_columns(
        [
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_liq_score_dv01')).otherwise(
                pl.col('_wavg_liq_score_size')
                ).alias('wavg_liq_score'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_start_skew_dv01')).otherwise(
                pl.col('_wavg_start_skew_size')
                ).alias('wavg_start_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_final_skew_dv01')).otherwise(
                pl.col('_wavg_final_skew_size')
                ).alias('wavg_final_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_abs_skew_delta_dv01')).otherwise(
                pl.col('_wavg_abs_skew_delta_size')
                ).alias('wavg_abs_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_skew_delta_dv01')).otherwise(
                pl.col('_wavg_skew_delta_size')
                ).alias('wavg_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_rebase_effect_dv01')).otherwise(
                pl.col('_wavg_rebase_effect_size')
            ).alias('wavg_rebase_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_bucket_effect_dv01')).otherwise(
                pl.col('_wavg_bucket_effect_size')
            ).alias('wavg_bucket_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_trader_effect_dv01')).otherwise(
                pl.col('_wavg_trader_effect_size')
            ).alias('wavg_trader_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_anchor_adj_dv01')).otherwise(
                pl.col('_wavg_anchor_adj_size')
            ).alias('wavg_anchor_adj'),
        ]
    ).drop(
        [
            '_wavg_liq_score_dv01', '_wavg_start_skew_dv01', '_wavg_final_skew_dv01', '_wavg_skew_delta_dv01',
            '_wavg_liq_score_size', '_wavg_start_skew_size', '_wavg_final_skew_size', '_wavg_skew_delta_size',
            '_wavg_rebase_effect_dv01', '_wavg_rebase_effect_size',
            '_wavg_bucket_effect_dv01', '_wavg_trader_effect_dv01', '_wavg_bucket_effect_size',
            '_wavg_trader_effect_size', '_wavg_anchor_adj_dv01', '_wavg_anchor_adj_size',
        ], strict=False
    )

    if cfg.debug:
        if is_ok_final:
            chgs = summary_overall.select(
                [
                    pl.concat_arr([pl.col(_c('side')), pl.col(_c('quote_type'))]).alias('k'),
                    pl.col('wavg_start_skew'), pl.col('wavg_final_skew'), pl.col('max_skew_delta')
                ]
            ).hyper.to_map('k', ['wavg_start_skew', 'wavg_final_skew', 'max_skew_delta'])
            res = {}
            for k, v in chgs.items():
                res["-".join(k)] = (
                    f'{round(v.get("wavg_start_skew"), 2)} -> {round(v.get("wavg_final_skew"), 2)}, ↑ {round(v.get("max_skew_delta"), 2)}')

            if name is None:
                log.success("Redistributed Success", **_sort_dict(res))
            else:
                log.success(f"Redistributed Success for {name}", **_sort_dict(res))
        else:
            if name is None:
                log.error("Redistribute FAILED", status=result.status)
            else:
                log.error(f"Redistribute FAILED for {name}", status=result.status)

    return df_result, result, summary_overall, summary_trader, removed_ids


def _build_fallback_result(sub_df: pl.LazyFrame, cfg: OptimizerConfig) -> pl.DataFrame:
    """Build a result DataFrame with starting values (zero optimization effects)
    for a trader whose solve() failed. Matches the output schema of solve()."""
    sub_df = sub_df.lazy() if isinstance(sub_df, pl.DataFrame) else sub_df

    # Match solve()'s preprocessing filters
    sub_df = sub_df.filter(pl.col('isReal') == 1)
    sub_df = sub_df.filter(pl.col(_c("quote_type")).is_in(['PX', 'SPD']))

    s = dict(sub_df.collect_schema())
    my_px_bid = [c for c in MY_PX_SOURCE_BID if c in s]
    my_px_ask = [c for c in MY_PX_SOURCE_ASK if c in s]
    my_spd_bid = [c for c in MY_SPD_SOURCE_BID if c in s]
    my_spd_ask = [c for c in MY_SPD_SOURCE_ASK if c in s]

    sub_df = sub_df.with_columns([
        pl.when(IS_BUY).then(pl.coalesce(my_px_bid)).otherwise(pl.coalesce(my_px_ask)).alias('_my_quotePx'),
        pl.when(IS_BUY).then(pl.coalesce(my_spd_bid)).otherwise(pl.coalesce(my_spd_ask)).alias('_my_quoteSpd'),
    ]).with_columns([
        pl.when(IS_SPREAD).then(pl.col('_my_quoteSpd') - pl.col('_refMidSpd'))
        .otherwise(pl.col('_my_quotePx') - pl.col('_refMidPx')).alias('skew'),
    ])

    sub_df = add_adjusted_bsr_bsi(sub_df, clamp_liq=True,
                                   charge_strength=cfg.charge_strength,
                                   default_strength=cfg.default_strength)

    sub_df = sub_df.with_columns([
        pl.when(IS_SPREAD).then(pl.col(_c('dv01'))).otherwise(pl.col(_c('size')) / 10_000).alias('kappa'),
        pl.when(IS_SPREAD & IS_SELL).then((pl.col('_refMidSpd') - pl.col('_my_quoteSpd')) * pl.col(_c('dv01')))
        .when(IS_SPREAD & IS_BUY).then((pl.col('_my_quoteSpd') - pl.col('_refMidSpd')) * pl.col(_c('dv01')))
        .when(IS_PX & IS_SELL).then((pl.col('_my_quotePx') - pl.col('_refMidPx')) * pl.col(_c('size')) / 100)
        .when(IS_PX & IS_BUY).then((pl.col('_refMidPx') - pl.col('_my_quotePx')) * pl.col(_c('size')) / 100)
        .alias('skewProceeds'),
    ])

    collected = sub_df.collect()
    N = len(collected)

    skew = collected['skew'].to_numpy().astype(float)
    kappa = collected['kappa'].to_numpy().astype(float)
    blended = collected['blendedCharge'].to_numpy().astype(float)
    proceeds = collected['skewProceeds'].to_numpy().astype(float)
    qts = collected[_c('quote_type')].to_list()
    is_px = np.array([q == "PX" for q in qts])
    bps_to_skew = np.where(is_px, 1.0 / 100.0, 1.0)
    starting_charge = np.where(is_px, skew * 100.0, skew)
    rebase = blended * bps_to_skew - skew

    sides = collected[_c('side')].to_list()
    side_signs = np.array([1.0 if si == "BUY" else -1.0 for si in sides])
    sizes = collected[_c('size')].to_numpy().astype(float)
    ref_mid_px = collected[_c('ref_mid_px')].to_numpy().astype(float)
    ref_mid_spd = collected[_c('ref_mid_spd')].to_numpy().astype(float)
    implied_px = ref_mid_px - side_signs * (proceeds * 100.0 / sizes)
    implied_spd = np.where(~is_px, ref_mid_spd + starting_charge, np.nan)
    quote_spd = collected[_c('quote_spd')].to_numpy().astype(float)

    return collected.select([
        pl.col(_c('bond_id')), pl.col(_c('side')), pl.col(_c('quote_type')),
        pl.col(_c('size')), pl.col(_c('dv01')),
        pl.col(_c('quote_spd')), pl.col(_c('quote_px')),
        pl.col(_c('ref_mid_spd')), pl.col(_c('ref_mid_px')),
        pl.col(_c('trader')), pl.col(_c('liq_score')),
        pl.col(_c('bsr_notional')), pl.col(_c('bsi_notional')),
        (pl.col(_c('bsr_notional')) / pl.col(_c('size'))).alias('bsr_pct'),
        pl.col('skew'),
        pl.Series("starting_proceeds", np.round(proceeds, 0)),
        pl.Series("blended_charge", np.round(blended, 4)),
        pl.Series("kappa", np.round(kappa, 2)),
        pl.Series("final_charge_bps", np.round(starting_charge, 4)),
        pl.Series("final_skew", np.round(skew, 4)),
        pl.Series("skew_delta", np.zeros(N)),
        pl.Series("rebase_effect", np.round(rebase, 4)),
        pl.Series("bucket_effect", np.zeros(N)),
        pl.Series("group_effect", np.zeros(N)),
        pl.Series("anchor_adj", np.zeros(N)),
        pl.Series("risk_pct", np.zeros(N)),
        pl.Series("final_proceeds", np.round(proceeds, 0)),
        pl.Series("proceeds_delta", np.zeros(N)),
        pl.Series("implied_px", np.round(implied_px, 6)),
        pl.Series("implied_spd", np.round(np.nan_to_num(implied_spd, nan=0), 4)),
        pl.Series("spd_delta", np.zeros(N)),
        pl.Series("binding_constraint", [''] * N),
        pl.col(_c('locked')).alias('isLocked'),
    ] + [pl.col(gc) for gc in (getattr(cfg, 'group_columns', None) or []) if gc in collected.columns])


def _solve_isolated(df: pl.DataFrame | pl.LazyFrame, cfg: OptimizerConfig):
    """Run the optimizer separately per group (trader).
    Each group is solved as if it were the full basket — bonds are independent
    within the sub-solve (no sub-grouping). Trader-pull is disabled since
    there's only 1 trader per sub-solve. Side constraint is loosened to avoid
    pinning tiny portfolios.
    """
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    from dataclasses import fields as dc_fields
    sub_cfg = OptimizerConfig(**{f.name: getattr(cfg, f.name) for f in dc_fields(cfg)})
    sub_cfg.isolate_traders = False
    sub_cfg.trader_pull = 0.0       # only 1 trader per sub-solve, tp is meaningless
    sub_cfg.side_floor_pct = max(cfg.side_floor_pct, 0.95)  # loosen for small portfolios

    inner_col, orig_group_cols = _resolve_inner_col(cfg)

    if orig_group_cols is not None:
        df = _add_group_key_column(df, orig_group_cols)
        iter_col = _GROUP_KEY_COL
    else:
        iter_col = _c('trader')
    keys_unique = df.select(pl.col(iter_col).unique()).collect().to_series().to_list()

    sub_cfg.group_columns = None  # sub-solve has only 1 group, grouping is meaningless

    dfs_out: list[pl.LazyFrame] = []
    per_group_diagnostics: dict[str, dict] = {}
    all_optimal = True
    combined_start_by_side, combined_final_by_side = {}, {}
    total_obj = 0.0

    for t in keys_unique:
        sub_df = df.filter(pl.col(iter_col) == t)
        try:
            df_r, res_r, _, _, _ = solve(sub_df, sub_cfg, name=t)
            if not isinstance(res_r, OptimizationResult):
                raise ValueError(f"solve returned non-OptimizationResult: {type(res_r)}")
        except Exception as e:
            if cfg.debug: log.warning(f"Isolated solve for trader {t} failed: {e} -- using starting values")
            try:
                df_r = _build_fallback_result(sub_df, sub_cfg)
                _fb_starts: dict[str, float] = {}
                for row in df_r.select([_c('side'), 'starting_proceeds']).iter_rows():
                    _fb_starts[row[0]] = _fb_starts.get(row[0], 0.0) + float(row[1])
                res_r = OptimizationResult(
                    status="infeasible", optimal=False, objective_value=0.0,
                    X_values={}, Y_values={},
                    final_charges=np.array([]), final_proceeds=np.array([]),
                    starting_proceeds_by_side=_fb_starts,
                    final_proceeds_by_side=dict(_fb_starts),
                    risk_pct={}, diagnostics={"status": "FAILED"}
                )
            except Exception as e2:
                if cfg.debug: log.warning(f"Fallback for trader {t} also failed: {e2}")
                continue

        # Tag each bond row with the isolation group for the frontend
        df_r_tagged = (df_r.lazy() if isinstance(df_r, pl.DataFrame) else df_r).with_columns(
            pl.lit(str(t)).alias('_isolation_group')
        )
        dfs_out.append(df_r_tagged)

        per_group_diagnostics[str(t)] = res_r.diagnostics or {"status": res_r.status}
        all_optimal = all_optimal and res_r.optimal
        total_obj += res_r.objective_value if not np.isnan(res_r.objective_value) else 0.0
        for s, v in res_r.starting_proceeds_by_side.items():
            combined_start_by_side[s] = combined_start_by_side.get(s, 0.0) + v
        for s, v in res_r.final_proceeds_by_side.items():
            combined_final_by_side[s] = combined_final_by_side.get(s, 0.0) + v

    if not dfs_out:
        raise RuntimeError("All isolated trader solves failed.")

    df_result = pl.concat(dfs_out)

    # Re-attach original group columns (sub-solves had group_columns=None so didn't include them)
    if orig_group_cols is not None:
        _gc_select = [pl.col(_c('bond_id'))] + [pl.col(gc) for gc in orig_group_cols]
        _gc_frame = df.select(_gc_select).distinct()
        df_result = df_result.join(_gc_frame, on=_c('bond_id'), how='left')

    # Recompute bond-level risk_pct from the FULL cross-trader dataset.
    # Each sub-solve had only 1 trader → its risk_pct was always 1.0.
    # We need the correct cross-trader allocation.
    try:
        _iso_result = df_result.collect() if hasattr(df_result, 'collect') else df_result
        _iso_sides = _iso_result[_c('side')].to_list()
        _iso_qts = _iso_result[_c('quote_type')].to_list()
        _iso_inner = _iso_result[_c('trader')].to_list()
        _iso_trader_risk = _iso_result['starting_proceeds'].to_numpy()  # use starting proceeds as risk proxy
        _iso_N = len(_iso_result)

        # Build (trader, side, qt) keys and bucket (side, qt) keys
        _iso_tbk = [((_iso_inner[i] or 'MISSING'), _iso_sides[i], _iso_qts[i]) for i in range(_iso_N)]
        _iso_bk = [(_iso_sides[i], _iso_qts[i]) for i in range(_iso_N)]

        # Sum |starting_proceeds| per (trader, side, qt) and per (side, qt) bucket
        _tbk_sums: dict[tuple, float] = {}
        _bk_sums: dict[tuple, float] = {}
        for i in range(_iso_N):
            v = abs(float(_iso_trader_risk[i]))
            _tbk_sums[_iso_tbk[i]] = _tbk_sums.get(_iso_tbk[i], 0.0) + v
            _bk_sums[_iso_bk[i]] = _bk_sums.get(_iso_bk[i], 0.0) + v

        # bond_risk_pct = |bond_risk| / bucket_total
        _iso_bond_risk_pct = np.array([
            abs(float(_iso_trader_risk[i])) / _bk_sums[_iso_bk[i]]
            if _bk_sums.get(_iso_bk[i], 0) > 1e-12 else 0.0
            for i in range(_iso_N)
        ])

        # Replace the per-sub-solve risk_pct with the correct cross-trader values
        if isinstance(df_result, pl.LazyFrame):
            df_result = _iso_result
        df_result = df_result.with_columns(
            pl.Series("risk_pct", np.round(_iso_bond_risk_pct, 6))
        )
        if not isinstance(df_result, pl.LazyFrame):
            df_result = df_result.lazy()
    except Exception as e:
        if cfg.debug: log.warning(f"Isolated risk_pct recompute failed: {e}")

    # ── Anchor WAVG skew correction for isolated mode ─────────────────
    if cfg.anchor_wavg_skew:
        try:
            _anch_df = df_result.collect() if hasattr(df_result, 'collect') else df_result
            _anch_sides = _anch_df[_c('side')].to_list()
            _anch_qts = _anch_df[_c('quote_type')].to_list()
            _anch_N = len(_anch_df)
            _anch_final_charge = _anch_df['final_charge_bps'].to_numpy().astype(float).copy()
            _anch_is_px = np.array([q == "PX" for q in _anch_qts])
            _anch_bps_to_skew = np.where(_anch_is_px, 1.0 / 100.0, 1.0)
            _anch_start_charge = np.where(
                _anch_is_px,
                _anch_df['skew'].to_numpy().astype(float) * 100.0,
                _anch_df['skew'].to_numpy().astype(float),
            )
            _anch_kappa = _anch_df['kappa'].to_numpy().astype(float)
            _anch_locked = _anch_df['isLocked'].cast(pl.Boolean, strict=False).to_numpy()
            _anch_signs = np.array([
                (1 if _anch_qts[i] == "SPD" and _anch_sides[i] == "BUY"
                 else -1 if _anch_qts[i] == "SPD" and _anch_sides[i] == "SELL"
                 else -1 if _anch_qts[i] == "PX" and _anch_sides[i] == "BUY"
                 else 1)
                for i in range(_anch_N)
            ])
            _anch_final_proceeds = _anch_df['final_proceeds'].to_numpy().astype(float).copy()

            _anch_anchor_adj = _apply_anchor_wavg_skew(
                _anch_final_charge, _anch_final_proceeds, _anch_start_charge,
                _anch_signs, _anch_kappa, _anch_sides, _anch_qts,
                _anch_locked, _anch_is_px, _anch_bps_to_skew,
                debug=cfg.debug,
            )

            # Recompute all derived columns from the corrected charge/proceeds
            _anch_implied_skew = _anch_final_charge * _anch_bps_to_skew
            _anch_skew_delta = (_anch_final_charge - _anch_start_charge) * _anch_bps_to_skew
            _anch_proceeds_delta = _anch_final_proceeds - _anch_df['starting_proceeds'].to_numpy().astype(float)

            _anch_ref_mid_spd = _anch_df[_c('ref_mid_spd')].to_numpy().astype(float)
            _anch_ref_mid_px = _anch_df[_c('ref_mid_px')].to_numpy().astype(float)
            _anch_sizes = _anch_df[_c('size')].to_numpy().astype(float)
            _anch_side_signs = np.array([1.0 if s == "BUY" else -1.0 for s in _anch_sides])
            _anch_implied_px = _anch_ref_mid_px - _anch_side_signs * (
                _anch_final_proceeds * 100.0 / np.where(_anch_sizes > 0, _anch_sizes, 1.0)
            )
            _anch_implied_spd = np.where(~_anch_is_px, _anch_ref_mid_spd + _anch_final_charge, np.nan)
            _anch_quote_spd = _anch_df[_c('quote_spd')].to_numpy().astype(float)

            # Recompute decomposition using solver-only delta (excluding anchor)
            _anch_solver_delta = _anch_skew_delta - _anch_anchor_adj
            _anch_bucket_effect = np.zeros(_anch_N)
            _anch_inner_col = _anch_df[_c('trader')].to_list()
            _anch_tbk_keys = set()
            _anch_bond_tbk = []
            for i in range(_anch_N):
                tbk = (_anch_inner_col[i] or 'MISSING', _anch_sides[i], _anch_qts[i])
                _anch_tbk_keys.add(tbk)
                _anch_bond_tbk.append(tbk)
            for tbk in _anch_tbk_keys:
                tbk_mask = np.array([_anch_bond_tbk[i] == tbk for i in range(_anch_N)])
                tbk_delta = _anch_solver_delta[tbk_mask]
                _anch_bucket_effect[tbk_mask] = np.mean(tbk_delta) if tbk_mask.any() else 0.0
            _anch_group_effect = _anch_solver_delta - _anch_bucket_effect
            # Zero out locked bonds
            for i in range(_anch_N):
                if _anch_locked[i]:
                    _anch_bucket_effect[i] = 0.0
                    _anch_group_effect[i] = 0.0

            df_result = _anch_df.with_columns([
                pl.Series("final_charge_bps", np.round(_anch_final_charge, 4)),
                pl.Series("final_skew", np.round(_anch_implied_skew, 4)),
                pl.Series("skew_delta", np.round(_anch_skew_delta, 4)),
                pl.Series("bucket_effect", np.round(_anch_bucket_effect, 4)),
                pl.Series("group_effect", np.round(_anch_group_effect, 4)),
                pl.Series("anchor_adj", np.round(_anch_anchor_adj, 4)),
                pl.Series("final_proceeds", np.round(_anch_final_proceeds, 0)),
                pl.Series("proceeds_delta", np.round(_anch_proceeds_delta, 0)),
                pl.Series("implied_px", np.round(_anch_implied_px, 6)),
                pl.Series("implied_spd", np.round(np.nan_to_num(_anch_implied_spd, nan=0), 4)),
                pl.Series("spd_delta", np.round(np.where(~_anch_is_px, _anch_implied_spd - _anch_quote_spd, 0), 4)),
            ]).lazy()
        except Exception as e:
            if cfg.debug: log.warning(f"Isolated anchor WAVG skew failed: {e}")
            if not isinstance(df_result, pl.LazyFrame):
                df_result = df_result.lazy()

    summary_overall = df_result.group_by([_c('side'), _c('quote_type')]).agg(
        [
            pl.col('skew_delta').abs().max().alias('max_skew_delta'),
            pl.col('proceeds_delta').sum().alias('proceeds_delta'),

            pl.col('skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_start_skew_dv01'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_final_skew_dv01'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_abs_skew_delta_dv01'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_skew_delta_dv01'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_rebase_effect_dv01'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_bucket_effect_dv01'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_trader_effect_dv01'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_anchor_adj_dv01'),

            pl.col('skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_start_skew_size'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_final_skew_size'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('size'))).alias('_wavg_abs_skew_delta_size'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('size'))).alias('_wavg_skew_delta_size'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_rebase_effect_size'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_bucket_effect_size'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_trader_effect_size'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('size'))).alias('_wavg_anchor_adj_size'),
        ]
    ).with_columns(
        [
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_start_skew_dv01')).otherwise(
                pl.col('_wavg_start_skew_size')).alias('wavg_start_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_final_skew_dv01')).otherwise(
                pl.col('_wavg_final_skew_size')).alias('wavg_final_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_abs_skew_delta_dv01')).otherwise(
                pl.col('_wavg_abs_skew_delta_size')).alias('wavg_abs_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_skew_delta_dv01')).otherwise(
                pl.col('_wavg_skew_delta_size')).alias('wavg_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_rebase_effect_dv01')).otherwise(
                pl.col('_wavg_rebase_effect_size')).alias('wavg_rebase_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_bucket_effect_dv01')).otherwise(
                pl.col('_wavg_bucket_effect_size')).alias('wavg_bucket_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_trader_effect_dv01')).otherwise(
                pl.col('_wavg_trader_effect_size')).alias('wavg_trader_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_anchor_adj_dv01')).otherwise(
                pl.col('_wavg_anchor_adj_size')).alias('wavg_anchor_adj'),
        ]
    ).drop(
        [
            '_wavg_start_skew_dv01', '_wavg_final_skew_dv01', '_wavg_abs_skew_delta_dv01', '_wavg_skew_delta_dv01',
            '_wavg_rebase_effect_dv01', '_wavg_rebase_effect_size',
            '_wavg_bucket_effect_dv01', '_wavg_trader_effect_dv01',
            '_wavg_start_skew_size', '_wavg_final_skew_size', '_wavg_abs_skew_delta_size', '_wavg_skew_delta_size',
            '_wavg_bucket_effect_size', '_wavg_trader_effect_size',
            '_wavg_anchor_adj_dv01', '_wavg_anchor_adj_size',
        ], strict=False
    )

    _iso_inner_col, _iso_group_cols = _resolve_inner_col(cfg)
    _iso_group_by = _iso_group_cols if _iso_group_cols is not None else [_c('trader')]
    summary_trader = df_result.group_by([_c('side'), _c('quote_type')] + _iso_group_by).agg(
        [
            pl.col('skew_delta').abs().max().alias('max_skew_delta'),
            pl.col('proceeds_delta').sum().alias('proceeds_delta'),
            pl.col('risk_pct').sum().alias('risk_pct'),

            (pl.col(_c('bsr_notional')).sum() / pl.col(_c('size')).sum()).alias('pct_bsr'),
            (pl.col(_c('bsi_notional')).sum() / pl.col(_c('size')).sum()).alias('pct_bsi'),
            pl.col(_c('quote_type')).count().alias('count'),
            pl.col(_c('size')).sum().alias('size'),
            pl.col(_c('dv01')).sum().alias('dv01'),

            pl.col('skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_start_skew_dv01'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_final_skew_dv01'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_abs_skew_delta_dv01'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_skew_delta_dv01'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_rebase_effect_dv01'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_bucket_effect_dv01'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_trader_effect_dv01'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_anchor_adj_dv01'),

            pl.col(_c('liq_score')).hyper.wavg(pl.col(_c('dv01'))).alias('_wavg_liq_score_dv01'),
            pl.col(_c('liq_score')).hyper.wavg(pl.col(_c('size'))).alias('_wavg_liq_score_size'),

            pl.col('skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_start_skew_size'),
            pl.col('final_skew').hyper.wavg(pl.col(_c('size'))).alias('_wavg_final_skew_size'),
            pl.col('skew_delta').abs().hyper.wavg(pl.col(_c('size'))).alias('_wavg_abs_skew_delta_size'),
            pl.col('skew_delta').hyper.wavg(pl.col(_c('size'))).alias('_wavg_skew_delta_size'),
            pl.col('rebase_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_rebase_effect_size'),
            pl.col('bucket_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_bucket_effect_size'),
            pl.col('group_effect').hyper.wavg(pl.col(_c('size'))).alias('_wavg_trader_effect_size'),
            pl.col('anchor_adj').hyper.wavg(pl.col(_c('size'))).alias('_wavg_anchor_adj_size'),
        ]
    ).with_columns(
        [
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_liq_score_dv01')).otherwise(
                pl.col('_wavg_liq_score_size')).alias('wavg_liq_score'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_start_skew_dv01')).otherwise(
                pl.col('_wavg_start_skew_size')).alias('wavg_start_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_final_skew_dv01')).otherwise(
                pl.col('_wavg_final_skew_size')).alias('wavg_final_skew'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_abs_skew_delta_dv01')).otherwise(
                pl.col('_wavg_abs_skew_delta_size')).alias('wavg_abs_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_skew_delta_dv01')).otherwise(
                pl.col('_wavg_skew_delta_size')).alias('wavg_skew_delta'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_rebase_effect_dv01')).otherwise(
                pl.col('_wavg_rebase_effect_size')).alias('wavg_rebase_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_bucket_effect_dv01')).otherwise(
                pl.col('_wavg_bucket_effect_size')).alias('wavg_bucket_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_trader_effect_dv01')).otherwise(
                pl.col('_wavg_trader_effect_size')).alias('wavg_trader_effect'),
            pl.when(pl.col(_c('quote_type'))=="SPD").then(pl.col('_wavg_anchor_adj_dv01')).otherwise(
                pl.col('_wavg_anchor_adj_size')).alias('wavg_anchor_adj'),
        ]
    ).drop(
        [
            '_wavg_liq_score_dv01', '_wavg_liq_score_size',
            '_wavg_start_skew_dv01', '_wavg_final_skew_dv01', '_wavg_abs_skew_delta_dv01', '_wavg_skew_delta_dv01',
            '_wavg_rebase_effect_dv01', '_wavg_rebase_effect_size',
            '_wavg_bucket_effect_dv01', '_wavg_trader_effect_dv01',
            '_wavg_start_skew_size', '_wavg_final_skew_size', '_wavg_abs_skew_delta_size', '_wavg_skew_delta_size',
            '_wavg_bucket_effect_size', '_wavg_trader_effect_size',
            '_wavg_anchor_adj_dv01', '_wavg_anchor_adj_size',
        ], strict=False
    )

    result = OptimizationResult(
        status="optimal" if all_optimal else "partial",
        optimal=all_optimal, objective_value=total_obj,
        X_values={}, Y_values={},
        final_charges=np.array([]), final_proceeds=np.array([]),
        starting_proceeds_by_side=combined_start_by_side,
        final_proceeds_by_side=combined_final_by_side, risk_pct={},
        diagnostics={"per_group": per_group_diagnostics, "isolated": True},
    )

    if cfg.debug:
        if all_optimal: log.success(
            "Isolated-trader redistribute complete", traders=len(traders_unique), objective=round(total_obj, 2)
            )
        else: log.warning(f"Isolated-trader redistribute -- some traders failed")

    return df_result, result, summary_overall, summary_trader, {}
