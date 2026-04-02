
from __future__ import annotations

import atexit
import asyncio
import itertools
import concurrent.futures
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import os
import threading
import warnings
from types import SimpleNamespace, NoneType
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableSet, Optional, Sequence, Tuple, Union, Set, Type
from app.helpers.string_helpers import clean_camel

import numpy as np
import polars as pl

try:
    import re2 as re  # type: ignore
except ImportError:  # pragma: no cover
    import re  # type: ignore

try:
    import pyarrow as pa  # type: ignore
except ImportError:  # pragma: no cover
    pa = None  # type: ignore

try:
    from pyod.models.mad import MAD as PyodMAD  # type: ignore
    from pyod.models.knn import KNN as PyodKNN  # type: ignore
    from pyod.models.lof import LOF as PyodLOF  # type: ignore
    from pyod.models.iforest import IForest as PyodIForest  # type: ignore
    from pyod.models.ecod import ECOD as PyodECOD  # type: ignore
    from pyod.models.copod import COPOD as PyodCOPOD  # type: ignore
    from pyod.models.hbos import HBOS as PyodHBOS  # type: ignore
    from pyod.models.ocsvm import OCSVM as PyodOCSVM  # type: ignore
    from pyod.models.abod import ABOD as PyodABOD  # type: ignore
    from pyod.models.pca import PCA as PyodPCA  # type: ignore
    from pyod.models.loda import LODA as PyodLODA  # type: ignore
    from pyod.models.suod import SUOD as PyodSUOD  # type: ignore
    _HAS_PYOD = True
except ImportError:  # pragma: no cover
    _HAS_PYOD = False
    PyodMAD = PyodKNN = PyodLOF = PyodIForest = None  # type: ignore
    PyodECOD = PyodCOPOD = PyodHBOS = PyodOCSVM = None  # type: ignore
    PyodABOD = PyodPCA = PyodLODA = PyodSUOD = None  # type: ignore

try:
    from app.helpers.type_helpers import ensure_list as ensure_list  # type: ignore
    from app.helpers.type_helpers import ensure_set as ensure_set  # type: ignore
except Exception:  # pragma: no cover
    def ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    def ensure_set(value: Any) -> Set[Any]:
        if value is None:
            return set()
        if isinstance(value, set):
            return value
        if isinstance(value, (list, tuple)):
            return set(value)
        return {value}

try:
    from app.helpers.date_helpers import now_date, now_time  # type: ignore
except Exception:  # pragma: no cover
    def now_date(*, utc: bool = True) -> str:
        dt = datetime.now(timezone.utc) if utc else datetime.now()
        return dt.strftime("%Y-%m-%d")

    def now_time(*, utc: bool = True) -> str:
        dt = datetime.now(timezone.utc) if utc else datetime.now()
        return dt.strftime("%H:%M:%S")

try:
    from app.helpers.regex_helpers import hyper_match  # type: ignore
except Exception:  # pragma: no cover
    def hyper_match(pattern: str, text: str, *, case_sensitive: bool = True, cached: bool = True, group: int = 0):
        if (not case_sensitive) and (not pattern.startswith("(?i)")):
            pattern = "(?i)" + pattern
        m = re.search(pattern, text)
        if not m: return None
        return m.group(group) if group is not None else m.group(0)


ExprLike = Union[str, pl.Expr]
FrameLike = Union[pl.DataFrame, pl.LazyFrame]
_FLOAT = pl.Float64

# -----------------------------------------------------------------------------
# Thread offload
# -----------------------------------------------------------------------------
_COLLECT_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_COLLECT_EXECUTOR_LOCK = threading.Lock()
_COLLECT_INFLIGHT: Optional[threading.Semaphore] = None


def _default_collect_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(2, cpu // 2))


def _default_collect_inflight() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(2, cpu // 2))


def _get_collect_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _COLLECT_EXECUTOR, _COLLECT_INFLIGHT
    with _COLLECT_EXECUTOR_LOCK:
        if _COLLECT_EXECUTOR is None:
            env = os.getenv("HYPER_COLLECT_WORKERS", "")
            try:
                workers = int(env) if env else _default_collect_workers()
            except Exception:
                workers = _default_collect_workers()
            _COLLECT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, workers),
                thread_name_prefix="hyper-polars-collect",
            )
            env2 = os.getenv("HYPER_COLLECT_MAX_INFLIGHT", "")
            try:
                inflight = int(env2) if env2 else _default_collect_inflight()
            except Exception:
                inflight = _default_collect_inflight()
            _COLLECT_INFLIGHT = threading.Semaphore(max(1, inflight))
            atexit.register(_shutdown_collect_executor)
        return _COLLECT_EXECUTOR


def _collect_gate() -> threading.Semaphore:
    gate = _COLLECT_INFLIGHT
    if gate is None:
        _get_collect_executor()
        gate = _COLLECT_INFLIGHT
    if gate is None:
        raise RuntimeError("Collect executor has been shut down")
    return gate


def _shutdown_collect_executor() -> None:
    global _COLLECT_EXECUTOR, _COLLECT_INFLIGHT
    with _COLLECT_EXECUTOR_LOCK:
        ex = _COLLECT_EXECUTOR
        _COLLECT_EXECUTOR = None
        _COLLECT_INFLIGHT = None
    if ex is not None:
        ex.shutdown(wait=False, cancel_futures=True)


def _submit_collect(fn, *args, **kwargs):
    ex = _get_collect_executor()
    gate = _collect_gate()

    def _wrapped():
        gate.acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            gate.release()

    return ex.submit(_wrapped)


async def _run_in_collect_executor(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ex = _get_collect_executor()
    gate = _collect_gate()

    def _wrapped():
        gate.acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            gate.release()

    return await loop.run_in_executor(ex, _wrapped)

# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------
def _as_lazy(frame: FrameLike) -> pl.LazyFrame:
    return frame if isinstance(frame, pl.LazyFrame) else frame.lazy()


def _to_expr(expr: ExprLike) -> pl.Expr:
    return expr if isinstance(expr, pl.Expr) else pl.col(expr)


def _normalize_exprs(exprs: Iterable[ExprLike]) -> List[pl.Expr]:
    return [_to_expr(e) for e in exprs]


def _normalize_null_like(null_like: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    return ensure_list(null_like) if null_like else None


def _ensure_float(expr: ExprLike, *, coerce_numeric: bool = True, cast_strict: bool = False) -> pl.Expr:
    e = _to_expr(expr)
    return e.cast(_FLOAT, strict=cast_strict) if coerce_numeric else e


def _safe_divide(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.is_null() | (den == 0)).then(pl.lit(None)).otherwise(num / den)


def _schema_mapping(frame: FrameLike) -> Dict[str, pl.DataType]:
    sch = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return dict(sch.items())


def _get_columns(frame: FrameLike) -> List[str]:
    return ensure_list(frame.collect_schema().names()) if isinstance(frame, pl.LazyFrame) else ensure_list(frame.columns)


def _unique_temp_name(base: str, existing: Iterable[str]) -> str:
    used: MutableSet[str] = set(existing)
    if base not in used:
        return base
    i = 1
    while True:
        c = f"{base}_{i}"
        if c not in used:
            return c
        i += 1


def _apply_null_like_values(expr: pl.Expr, null_values: Optional[List[Any]]) -> pl.Expr:
    if not null_values:
        return expr
    return pl.when(expr.is_in(null_values)).then(pl.lit(None)).otherwise(expr)


def frame_is_empty(frame: Optional[FrameLike]) -> bool:
    if frame is None:
        return True
    if isinstance(frame, pl.DataFrame):
        return frame.height == 0
    if isinstance(frame, pl.LazyFrame):
        return frame.limit(1).collect().height == 0
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def frame_height(frame: Optional[FrameLike]) -> int:
    if frame is None:
        return 0
    if isinstance(frame, pl.DataFrame):
        return frame.height
    if isinstance(frame, pl.LazyFrame):
        df = frame.select(pl.len().alias("__len__")).collect()
        return int(df["__len__"][0]) if df.height else 0
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def column_diff(left: FrameLike, right: FrameLike) -> Tuple[set[str], set[str]]:
    lc, rc = set(_get_columns(left)), set(_get_columns(right))
    return lc - rc, rc - lc


def column_overlap(left: FrameLike, right: FrameLike) -> set[str]:
    return set(_get_columns(left)) & set(_get_columns(right))


def schema_difference(left: FrameLike, right: FrameLike) -> Dict[str, Any]:
    lm, rm = _schema_mapping(left), _schema_mapping(right)
    lk, rk = set(lm), set(rm)
    mismatch: Dict[str, Tuple[pl.DataType, pl.DataType]] = {}
    for k in lk & rk:
        if lm[k] != rm[k]:
            mismatch[k] = (lm[k], rm[k])
    return {"only_left": lk - rk, "only_right": rk - lk, "dtype_mismatch": mismatch}


def missing_columns(frame: FrameLike, columns: Sequence[str]) -> List[str]:
    have = set(_get_columns(frame))
    return [c for c in columns if c not in have]


def require_columns(frame: FrameLike, columns: Sequence[str]) -> None:
    miss = missing_columns(frame, columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

def _infer_col_type(col, val):
    if col.startswith("is") and (
            isinstance(val, (bool, NoneType))
            or (isinstance(val, int) and val in (0, 1))
    ): return pl.Int8, (0 if val is None else val)
    if val is None: return pl.Null, None
    if isinstance(val, str): return pl.String, val
    if isinstance(val, (float, int)): return pl.Float64, val
    return pl.Object, val

def fill_missing(frame: FrameLike, columns: Sequence[str], *, defaults: Any = None, schema_override:Optional[Dict]=None) -> FrameLike:
    miss = missing_columns(frame, columns)
    if not miss: return frame
    default_map = defaults if isinstance(defaults, dict) else {}
    global_fill = None if isinstance(defaults, dict) else defaults
    schema_override = schema_override if isinstance(schema_override, dict) else {}
    def _schema(k):
        v = default_map.get(k, global_fill)
        return schema_override.get(k, _infer_col_type(k, v)), v
    miss_schema = {k:_schema(k) for k in miss}
    return frame.with_columns([
        pl.lit(default, dtype).alias(col) for col, (dtype, default) in miss_schema.items()
    ])

def _fuzzy_match_columns(
        columns: Sequence[str],
        pattern: str,
        *,
        regex: bool = False,
        case_sensitive: bool = False,
        exact: bool = False,
        invert: bool = False,
) -> List[str]:
    if not columns:
        return []
    if regex:
        if (not case_sensitive) and (not pattern.startswith("(?i)")):
            pattern = "(?i)" + pattern
        rx = re.compile(pattern)

        def _m(c: str) -> bool:
            return bool(rx.search(c))
    else:
        if case_sensitive:
            if exact:
                def _m(c: str) -> bool:
                    return c == pattern
            else:

                def _m(c: str) -> bool:
                    return pattern in c
        else:
            plow = pattern.lower()
            if exact:

                def _m(c: str) -> bool:
                    return c.lower() == plow
            else:

                def _m(c: str) -> bool:
                    return plow in c.lower()

    matched = [c for c in columns if _m(c)]
    if invert:
        mset = set(matched)
        return [c for c in columns if c not in mset]
    return matched


def _peek_value(frame: FrameLike, row: int = 0, col: int | str | list = 0, default: Any = None) -> Any:
    if isinstance(row, (str, list)) and col == 0:
        row, col = 0, row
    if row < 0:
        raise ValueError("row index must be non-negative")
    if isinstance(col, int) and col < 0:
        raise ValueError("column index must be non-negative")

    names = _get_columns(frame)
    cols = col if isinstance(col, list) else [col]
    selected: List[str] = []
    for c in cols:
        if isinstance(c, int):
            if 0 <= c < len(names):
                selected.append(names[c])
        else:
            if c in names:
                selected.append(c)
    if not selected:
        return default

    limited = _as_lazy(frame).select([pl.col(c) for c in selected]).slice(row, 1)
    df_small = _submit_collect(limited.collect).result() if isinstance(frame, pl.LazyFrame) else limited.collect()
    if df_small.height == 0:
        return default
    if df_small.width == 1:
        s = df_small.to_series()
        return default if s.is_empty() else s.item(0)
    return df_small.to_dicts()[0]


# -----------------------------------------------------------------------------
# Schema utilities
# -----------------------------------------------------------------------------
def _build_schema(schema, columns: Optional[Sequence[str]] = None, default: pl.DataType = pl.String) -> Dict[str, pl.DataType]:
    if schema is None:
        return {}
    if isinstance(schema, Mapping):
        return dict(schema)
    if isinstance(schema, pl.DataType):
        if columns is None:
            return defaultdict(lambda: schema)  # type: ignore[return-value]
        return {c: schema for c in columns}
    if isinstance(schema, (list, tuple)):
        if columns is None or not isinstance(columns, (list, tuple)):
            raise ValueError("Schema list needs column identifiers.")
        if len(schema) != len(columns):
            raise ValueError("Schema list and column list lengths do not match.")
        return {columns[i]: schema[i] for i in range(len(columns))}
    return defaultdict(lambda: default)  # type: ignore[return-value]


def _dedupe_rename_map(new_names: Dict[str, str], *, policy: str) -> Dict[str, str]:
    if policy == "overwrite":
        return new_names
    inv: Dict[str, List[str]] = {}
    for src, dst in new_names.items():
        inv.setdefault(dst, []).append(src)
    dups = {dst: srcs for dst, srcs in inv.items() if len(srcs) > 1}
    if not dups:
        return new_names
    if policy == "raise":
        raise ValueError(f"fuzzy_rename produced duplicate target names: {dups}")
    if policy == "suffix":
        out: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for src, dst in new_names.items():
            n = counts.get(dst, 0)
            counts[dst] = n + 1
            out[src] = dst if n == 0 else f"{dst}_{n}"
        return out
    raise ValueError("on_conflict must be one of: 'raise', 'suffix', 'overwrite'")


def _normalize_column_names(
        cols: Sequence[str],
        *,
        lower: bool = True,
        strip: bool = True,
        space_to_underscore: bool = True,
        keep_alnum_underscore: bool = True,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in cols:
        n = c.strip() if strip else c
        n = n.lower() if lower else n
        if space_to_underscore:
            n = re.sub(r"\s+", "_", n)
        if keep_alnum_underscore:
            n = re.sub(r"[^0-9a-zA-Z_]+", "", n)
        out[c] = n
    return out


def _select_existing_names(frame_cols: Sequence[str], requested: Sequence[str], *, strict: bool) -> List[str]:
    have = ensure_set(frame_cols)
    if strict:
        miss = [c for c in requested if c not in have]
        if miss:
            raise ValueError(f"Missing columns: {miss}")
        return ensure_list(requested)
    return [c for c in requested if c in have]


# -----------------------------------------------------------------------------
# Supertype (fast, pure)
# -----------------------------------------------------------------------------
def get_supertype(a: pl.DataType, b: pl.DataType) -> pl.DataType|Type:
    def _is_null_like(dt: pl.DataType) -> bool:
        return dt == pl.Null or dt == getattr(pl, "Unknown", object())

    if a == b:
        return a
    if _is_null_like(a):
        return b
    if _is_null_like(b):
        return a
    if a == pl.Object or b == pl.Object:
        return pl.Object

    def _is_bool(dt): return dt == pl.Boolean
    def _is_utf8(dt): return dt == pl.Utf8 or dt == getattr(pl, "String", pl.Utf8)
    def _is_binary(dt): return dt == pl.Binary
    def _is_categorical(dt): return dt == pl.Categorical
    def _is_enum(dt): return dt == getattr(pl, "Enum", object())
    def _is_signed_int(dt): return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    def _is_unsigned_int(dt): return dt in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    def _is_float(dt): return dt in (pl.Float32, pl.Float64)
    def _is_decimal(dt): return type(dt).__name__ == "Decimal" and hasattr(dt, "precision") and hasattr(dt, "scale")
    def _is_list(dt): return type(dt).__name__ == "List" and hasattr(dt, "inner")
    def _is_array(dt): return type(dt).__name__ == "Array" and hasattr(dt, "inner") and hasattr(dt, "width")
    def _is_struct(dt): return type(dt).__name__ == "Struct" and hasattr(dt, "fields")
    def _is_date(dt): return dt == pl.Date
    def _is_time(dt): return dt == pl.Time
    def _is_duration(dt): return type(dt).__name__ == "Duration"
    def _is_datetime(dt): return type(dt).__name__ == "Datetime"
    def _time_unit(dt): return getattr(dt, "time_unit", None)
    def _time_zone(dt): return getattr(dt, "time_zone", None)

    def _max_time_unit(u1: str | None, u2: str | None) -> str:
        order = {"ms": 0, "us": 1, "ns": 2}
        u1 = u1 or "ns"
        u2 = u2 or "ns"
        return u1 if order[u1] >= order[u2] else u2

    def _decimal(p: int, s: int) -> pl.DataType:
        return pl.Decimal(precision=p, scale=s)

    SIGNED_BITS = {pl.Int8: 8, pl.Int16: 16, pl.Int32: 32, pl.Int64: 64}
    UNSIGNED_BITS = {pl.UInt8: 8, pl.UInt16: 16, pl.UInt32: 32, pl.UInt64: 64}

    def _signed_for_bits(bits: int) -> pl.DataType | Type:
        if bits <= 8: return pl.Int8
        if bits <= 16: return pl.Int16
        if bits <= 32: return pl.Int32
        return pl.Int64

    def _unsigned_for_bits(bits: int) -> pl.DataType | Type:
        if bits <= 8: return pl.UInt8
        if bits <= 16: return pl.UInt16
        if bits <= 32: return pl.UInt32
        return pl.UInt64

    def _minimal_signed_bits_for_mix(signed_bits: int, unsigned_bits: int) -> int:
        need = max(signed_bits, unsigned_bits + 1)
        if need <= 8: return 8
        if need <= 16: return 16
        if need <= 32: return 32
        if need <= 64: return 64
        return 128

    def _int_digits(bits: int, signed: bool) -> int:
        if signed:
            return {8: 3, 16: 5, 32: 10, 64: 19}.get(bits, 39)
        return {8: 3, 16: 5, 32: 10, 64: 20}.get(bits, 39)

    def _decimal_merge(p1: int, s1: int, p2: int, s2: int, max_p: int = 38) -> pl.DataType|Type|None:
        s = s1 if s1 >= s2 else s2
        i = max(p1 - s1, p2 - s2)
        total = i + s
        return _decimal(total, s) if total <= max_p else None

    a_str = _is_utf8(a) or _is_categorical(a) or _is_enum(a)
    b_str = _is_utf8(b) or _is_categorical(b) or _is_enum(b)
    if a_str and b_str:
        return pl.Utf8
    if a_str or b_str:
        return pl.Binary if (_is_binary(a) or _is_binary(b)) else pl.Utf8
    if _is_binary(a) or _is_binary(b):
        return pl.Binary

    if _is_date(a) and _is_date(b): return pl.Date
    if _is_time(a) and _is_time(b): return pl.Time
    if _is_duration(a) and _is_duration(b):
        return pl.Duration(time_unit=_max_time_unit(_time_unit(a), _time_unit(b)))
    if _is_datetime(a) and _is_datetime(b):
        unit = _max_time_unit(_time_unit(a), _time_unit(b))
        tz_a, tz_b = _time_zone(a), _time_zone(b)
        tz = tz_a if tz_a == tz_b else (tz_a or tz_b or None)
        if tz_a and tz_b and tz_a != tz_b:
            tz = None
        return pl.Datetime(time_unit=unit, time_zone=tz)
    if _is_datetime(a) and (_is_date(b) or _is_time(b)):
        return pl.Datetime(time_unit=_max_time_unit(_time_unit(a), "ns"), time_zone=_time_zone(a))
    if _is_datetime(b) and (_is_date(a) or _is_time(a)):
        return pl.Datetime(time_unit=_max_time_unit(_time_unit(b), "ns"), time_zone=_time_zone(b))
    if (_is_date(a) and _is_time(b)) or (_is_date(b) and _is_time(a)):
        return pl.Datetime(time_unit="ns", time_zone=None)

    a_int, b_int = _is_signed_int(a), _is_signed_int(b)
    a_uint, b_uint = _is_unsigned_int(a), _is_unsigned_int(b)
    a_f, b_f = _is_float(a), _is_float(b)
    a_d, b_d = _is_decimal(a), _is_decimal(b)

    if a_f or b_f:
        return pl.Float64

    if a_d and b_d:
        m = _decimal_merge(a.precision, a.scale, b.precision, b.scale)
        return m if m is not None else pl.Float64

    if a_d and (b_int or b_uint or _is_bool(b)):
        p2 = _int_digits(SIGNED_BITS[b], True) if b_int else (_int_digits(UNSIGNED_BITS[b], False) if b_uint else 1)
        m = _decimal_merge(a.precision, a.scale, p2, 0)
        return m if m is not None else pl.Float64

    if b_d and (a_int or a_uint or _is_bool(a)):
        p1 = _int_digits(SIGNED_BITS[a], True) if a_int else (_int_digits(UNSIGNED_BITS[a], False) if a_uint else 1)
        m = _decimal_merge(p1, 0, b.precision, b.scale)
        return m if m is not None else pl.Float64

    if _is_bool(a) and _is_bool(b):
        return pl.Boolean

    if _is_bool(a) and (b_int or b_uint):
        req = _minimal_signed_bits_for_mix(SIGNED_BITS[b], 1) if b_int else _minimal_signed_bits_for_mix(1, UNSIGNED_BITS[b])
        return _signed_for_bits(req) if req <= 64 else _decimal(20, 0)
    if _is_bool(b) and (a_int or a_uint):
        req = _minimal_signed_bits_for_mix(SIGNED_BITS[a], 1) if a_int else _minimal_signed_bits_for_mix(1, UNSIGNED_BITS[a])
        return _signed_for_bits(req) if req <= 64 else _decimal(20, 0)

    if a_int and b_int:
        return _signed_for_bits(max(SIGNED_BITS[a], SIGNED_BITS[b]))
    if a_uint and b_uint:
        return _unsigned_for_bits(max(UNSIGNED_BITS[a], UNSIGNED_BITS[b]))
    if (a_int and b_uint) or (a_uint and b_int):
        s_bits = SIGNED_BITS[a] if a_int else SIGNED_BITS[b]
        u_bits = UNSIGNED_BITS[b] if b_uint else UNSIGNED_BITS[a]
        req = _minimal_signed_bits_for_mix(s_bits, u_bits)
        return _signed_for_bits(req) if req <= 64 else _decimal(20, 0)

    if _is_list(a) and _is_list(b):
        return pl.List(get_supertype(a.inner, b.inner))
    if _is_array(a) and _is_array(b):
        inner = get_supertype(a.inner, b.inner)
        return pl.Array(width=a.width, inner=inner) if a.width == b.width else pl.List(inner)
    if _is_array(a) and _is_list(b):
        return pl.List(get_supertype(a.inner, b.inner))
    if _is_array(b) and _is_list(a):
        return pl.List(get_supertype(a.inner, b.inner))

    if _is_struct(a) and _is_struct(b):
        fa = {f.name: f.dtype for f in a.fields}
        fb = {f.name: f.dtype for f in b.fields}
        names = fa.keys() | fb.keys()
        merged = [pl.Field(n, get_supertype(fa.get(n, pl.Null), fb.get(n, pl.Null))) for n in sorted(names)]
        return pl.Struct(merged)

    if (_is_date(a) or _is_time(a) or _is_duration(a) or _is_datetime(a)) or (_is_date(b) or _is_time(b) or _is_duration(b) or _is_datetime(b)):
        return pl.Utf8
    return pl.Utf8


# -----------------------------------------------------------------------------
# Core ops: dedupe_columns / safe_concat / safe_join / asof / grouped_last_before / with_delta
# -----------------------------------------------------------------------------
def _has_duplicate_names(names: Sequence[str]) -> bool:
    return len(names) != len(set(names))


def dedupe_columns(frame: FrameLike, *, policy: str = "suffix", suffix_sep: str = "_", keep: str = "first") -> FrameLike:
    names = _get_columns(frame)
    if not _has_duplicate_names(names):
        return frame
    if isinstance(frame, pl.LazyFrame):
        raise ValueError("dedupe_columns requires DataFrame when duplicate column names exist.")
    if policy == "raise":
        dups = [n for n in set(names) if names.count(n) > 1]
        raise ValueError(f"Duplicate column names: {dups}")
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")

    seen: Dict[str, int] = {}
    total: Dict[str, int] = {}
    for n in names:
        total[n] = total.get(n, 0) + 1

    new_names: List[str] = []
    drops: List[str] = []
    for n in names:
        i = seen.get(n, 0)
        seen[n] = i + 1
        if total[n] == 1:
            new_names.append(n)
            continue
        if keep == "first":
            if i == 0:
                new_names.append(n)
            else:
                nn = f"{n}{suffix_sep}{i}"
                new_names.append(nn)
                if policy == "drop":
                    drops.append(nn)
        else:
            last_i = total[n] - 1
            if i == last_i:
                new_names.append(n)
            else:
                nn = f"{n}{suffix_sep}{i}"
                new_names.append(nn)
                if policy == "drop":
                    drops.append(nn)

    df = frame.clone()
    df.columns = new_names
    if policy == "drop" and drops:
        df = df.drop(drops)
    if policy in ("suffix", "drop"):
        return df
    raise ValueError("policy must be one of: 'suffix','drop','raise'")


def _target_schema_for_concat(frames: Sequence[FrameLike]) -> Dict[str, pl.DataType]:
    merged: Dict[str, pl.DataType] = {}
    for f in frames:
        sm = _schema_mapping(f)
        for k, dt in sm.items():
            merged[k] = dt if k not in merged else get_supertype(merged[k], dt)
    return merged


def safe_concat(
        frames: Sequence[FrameLike],
        *,
        how: str = "vertical",
        rechunk: bool = False,
        supertype: bool = True,
        fill_missing: Any = None,
        drop_extra: bool = False,
) -> FrameLike:
    flist = [f for f in frames if f is not None]
    if not flist:
        return pl.DataFrame()

    any_lazy = any(isinstance(f, pl.LazyFrame) for f in flist)
    target = _target_schema_for_concat(flist) if supertype else _schema_mapping(flist[0])
    order = list(target.keys())

    if not any_lazy:
        aligned: List[pl.DataFrame] = []
        for f in flist:
            df: pl.DataFrame = f  # type: ignore[assignment]
            cols = set(df.columns)
            if drop_extra:
                df = df.select([c for c in order if c in cols])
                cols = set(df.columns)
            add = []
            for c in order:
                if c not in cols:
                    add.append(pl.lit(fill_missing).cast(target[c], strict=False).alias(c))
            if add:
                df = df.with_columns(add)
            if supertype:
                df = df.with_columns([pl.col(c).cast(target[c], strict=False).alias(c) for c in order])
            aligned.append(df.select(order))
        return pl.concat(aligned, how=how, rechunk=rechunk)

    aligned_lf: List[pl.LazyFrame] = []
    for f in flist:
        lf = _as_lazy(f)
        cols = set(_get_columns(f))
        if drop_extra:
            keep = [c for c in order if c in cols]
            lf = lf.select(keep)
            cols = set(keep)
        add_exprs: List[pl.Expr] = []
        for c in order:
            if c not in cols:
                add_exprs.append(pl.lit(fill_missing).cast(target[c], strict=False).alias(c))
        if add_exprs:
            lf = lf.with_columns(add_exprs)
        if supertype:
            lf = lf.with_columns([pl.col(c).cast(target[c], strict=False).alias(c) for c in order])
        aligned_lf.append(lf.select(order))
    return pl.concat(aligned_lf, how=how, rechunk=rechunk)


def safe_join(
        left: FrameLike,
        right: FrameLike,
        *,
        on: Union[str, Sequence[str], None] = None,
        left_on: Union[str, Sequence[str], None] = None,
        right_on: Union[str, Sequence[str], None] = None,
        how: str = "left",
        suffix: str = "_right",
        coalesce_dupes: bool = True,
        cast_on_supertype: bool = True,
        drop_right_on: bool = True,
        validate: Optional[str] = None,
) -> FrameLike:
    if on is not None:
        l_on = ensure_list(on)
        r_on = l_on
        on_cols = l_on
        use_on = True
    else:
        l_on = ensure_list(left_on)
        r_on = ensure_list(right_on)
        if len(l_on) != len(r_on) or not l_on:
            raise ValueError("Provide on=... or matching left_on/right_on")
        on_cols = []
        use_on = False

    def _join(df_or_lf_left, df_or_lf_right):
        kwargs: Dict[str, Any] = dict(how=how, suffix=suffix)
        if validate is not None:
            kwargs["validate"] = validate
        return df_or_lf_left.join(
            df_or_lf_right,
            on=on_cols if use_on else None,
            left_on=l_on if not use_on else None,
            right_on=r_on if not use_on else None,
            **kwargs,
        )

    if not isinstance(left, pl.LazyFrame) and not isinstance(right, pl.LazyFrame):
        ldf: pl.DataFrame = left  # type: ignore[assignment]
        rdf: pl.DataFrame = right  # type: ignore[assignment]

        if cast_on_supertype:
            lsch, rsch = ldf.schema, rdf.schema
            cast_l, cast_r = [], []
            for a, b in zip(l_on, r_on):
                if a in lsch and b in rsch:
                    tgt = get_supertype(lsch[a], rsch[b])
                    if lsch[a] != tgt:
                        cast_l.append(pl.col(a).cast(tgt, strict=False).alias(a))
                    if rsch[b] != tgt:
                        cast_r.append(pl.col(b).cast(tgt, strict=False).alias(b))
            if cast_l:
                ldf = ldf.with_columns(cast_l)
            if cast_r:
                rdf = rdf.with_columns(cast_r)

        overlap = (set(ldf.columns) & set(rdf.columns)) - set(r_on)
        joined = _join(ldf, rdf)

        if coalesce_dupes and overlap:
            exprs, drops = [], []
            jcols = set(joined.columns)
            for c in overlap:
                rc = f"{c}{suffix}"
                if rc in jcols:
                    exprs.append(pl.coalesce([pl.col(c), pl.col(rc)]).alias(c))
                    drops.append(rc)
            if exprs:
                joined = joined.with_columns(exprs).drop(drops)

        if drop_right_on and not use_on:
            jcols = set(joined.columns)
            drops = [c for c in r_on if c in jcols and c not in set(l_on)]
            if drops:
                joined = joined.drop(drops)

        return joined

    lf_l, lf_r = _as_lazy(left), _as_lazy(right)
    if cast_on_supertype:
        lsch, rsch = _schema_mapping(left), _schema_mapping(right)
        cast_l: List[pl.Expr] = []
        cast_r: List[pl.Expr] = []
        for a, b in zip(l_on, r_on):
            if a in lsch and b in rsch:
                tgt = get_supertype(lsch[a], rsch[b])
                if lsch[a] != tgt:
                    cast_l.append(pl.col(a).cast(tgt, strict=False).alias(a))
                if rsch[b] != tgt:
                    cast_r.append(pl.col(b).cast(tgt, strict=False).alias(b))
        if cast_l:
            lf_l = lf_l.with_columns(cast_l)
        if cast_r:
            lf_r = lf_r.with_columns(cast_r)

    overlap = (set(_get_columns(left)) & set(_get_columns(right))) - set(r_on)
    joined = _join(lf_l, lf_r)

    if coalesce_dupes and overlap:
        jcols = set(_get_columns(joined))
        exprs: List[pl.Expr] = []
        drops: List[str] = []
        for c in overlap:
            rc = f"{c}{suffix}"
            if rc in jcols:
                exprs.append(pl.coalesce([pl.col(c), pl.col(rc)]).alias(c))
                drops.append(rc)
        if exprs:
            joined = joined.with_columns(exprs).drop(drops)

    if drop_right_on and not use_on:
        jcols = set(_get_columns(joined))
        drops = [c for c in r_on if c in jcols and c not in set(l_on)]
        if drops:
            joined = joined.drop(drops)

    return joined


def asof_join_by_group(
        left: FrameLike,
        right: FrameLike,
        *,
        on: str,
        by: Union[str, Sequence[str], None] = None,
        strategy: str = "backward",
        tolerance: Optional[Any] = None,
        suffix: str = "_right",
        coalesce_dupes: bool = True,
        cast_on_supertype: bool = True,
) -> FrameLike:
    lf_l, lf_r = _as_lazy(left), _as_lazy(right)
    by_cols = ensure_list(by) if by is not None else None
    sort_cols = (by_cols or []) + [on]

    if cast_on_supertype:
        lsch, rsch = _schema_mapping(left), _schema_mapping(right)
        if on in lsch and on in rsch:
            tgt = get_supertype(lsch[on], rsch[on])
            if lsch[on] != tgt:
                lf_l = lf_l.with_columns(pl.col(on).cast(tgt, strict=False).alias(on))
            if rsch[on] != tgt:
                lf_r = lf_r.with_columns(pl.col(on).cast(tgt, strict=False).alias(on))

    lf_l, lf_r = lf_l.sort(sort_cols), lf_r.sort(sort_cols)
    joined = lf_l.join_asof(lf_r, on=on, by=by_cols, strategy=strategy, tolerance=tolerance, suffix=suffix)

    if coalesce_dupes:
        overlap = (set(_get_columns(left)) & set(_get_columns(right))) - (set(by_cols or []) | {on})
        jcols = set(_get_columns(joined))
        exprs: List[pl.Expr] = []
        drops: List[str] = []
        for c in overlap:
            rc = f"{c}{suffix}"
            if rc in jcols:
                exprs.append(pl.coalesce([pl.col(c), pl.col(rc)]).alias(c))
                drops.append(rc)
        if exprs:
            joined = joined.with_columns(exprs).drop(drops)

    return joined if (isinstance(left, pl.LazyFrame) or isinstance(right, pl.LazyFrame)) else joined.collect()


def grouped_last_before(frame: FrameLike, *, group_by: Sequence[str], order_by: str, cutoff: Any, inclusive: bool = True) -> FrameLike:
    lf = _as_lazy(frame)
    cond = pl.col(order_by) <= pl.lit(cutoff) if inclusive else pl.col(order_by) < pl.lit(cutoff)
    gb = ensure_list(group_by)
    out = lf.filter(cond).sort(gb + [order_by]).group_by(gb, maintain_order=True).tail(1)
    return out if isinstance(frame, pl.LazyFrame) else out.collect()


def _parse_lookback_to_duration(lookback: Any) -> pl.Expr:
    if isinstance(lookback, pl.Expr):
        return lookback
    if isinstance(lookback, timedelta):
        us = int(lookback.total_seconds() * 1_000_000)
        return pl.duration(microseconds=us)
    if isinstance(lookback, (int, float)):
        return pl.duration(days=int(lookback))
    if isinstance(lookback, str):
        s = lookback.strip().lower()
        m = re.fullmatch(r"(-?\d+)\s*([a-z]+)", s)
        if not m:
            raise ValueError("lookback string must look like '7d','12h','30m','500ms','100us','100ns'")
        n = int(m.group(1))
        u = m.group(2)
        if u in ("d", "day", "days"): return pl.duration(days=n)
        if u in ("h", "hr", "hrs", "hour", "hours"): return pl.duration(hours=n)
        if u in ("m", "min", "mins", "minute", "minutes"): return pl.duration(minutes=n)
        if u in ("s", "sec", "secs", "second", "seconds"): return pl.duration(seconds=n)
        if u in ("ms", "msec", "msecs", "millisecond", "milliseconds"): return pl.duration(milliseconds=n)
        if u in ("us", "usec", "usecs", "microsecond", "microseconds"): return pl.duration(microseconds=n)
        if u in ("ns", "nsec", "nsecs", "nanosecond", "nanoseconds"): return pl.duration(nanoseconds=n)
        raise ValueError("Unsupported lookback unit")
    raise TypeError("Unsupported lookback type")


def with_delta(
        frame: FrameLike,
        *,
        date_col: str,
        lookback: Any,
        target_col: str,
        group_by: Optional[Sequence[str]] = None,
        output_name: Optional[str] = None,
        strategy: str = "backward",
) -> FrameLike:
    keys = ensure_list(group_by) if group_by else []
    out_col = output_name or f"{target_col}_delta"
    lf = _as_lazy(frame)
    row_idx = _unique_temp_name("__hyper_row__", _get_columns(frame))
    dur = _parse_lookback_to_duration(lookback)

    base = lf.with_row_index(row_idx)
    left = base.with_columns((pl.col(date_col) - dur).alias("__ref_dt")).select([row_idx, *keys, "__ref_dt", pl.col(target_col).alias("__cur")])
    right = base.select([*keys, pl.col(date_col).alias("__dt"), pl.col(target_col).alias("__prev")])

    left = left.sort(keys + ["__ref_dt"])
    right = right.sort(keys + ["__dt"])

    joined = left.join_asof(right, left_on="__ref_dt", right_on="__dt", by=keys if keys else None, strategy=strategy)
    delta = joined.select([row_idx, (pl.col("__cur") - pl.col("__prev")).alias(out_col)])
    final = base.join(delta, on=row_idx, how="left").drop(row_idx)

    return final if isinstance(frame, pl.LazyFrame) else final.collect()


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------
def null_report(frame: FrameLike, columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
    cols = ensure_list(columns) if columns is not None else _get_columns(frame)
    if not cols:
        return pl.DataFrame({"column": [], "dtype": [], "null_count": [], "null_pct": [], "rows": []})
    lf = _as_lazy(frame)
    one = lf.select([pl.len().alias("__rows__")] + [pl.col(c).null_count().alias(c) for c in cols]).collect()
    rows = int(one["__rows__"][0]) if one.height else 0
    sch = _schema_mapping(frame)
    out = [{"column": c, "dtype": str(sch.get(c)), "null_count": int(one[c][0]), "null_pct": (float(one[c][0]) / rows) if rows else None, "rows": rows} for c in cols]
    return pl.DataFrame(out)


def distinct_report(frame: FrameLike, columns: Optional[Sequence[str]] = None, *, top_k: Optional[int] = None) -> pl.DataFrame:
    cols = ensure_list(columns) if columns is not None else _get_columns(frame)
    if not cols:
        return pl.DataFrame({"column": [], "dtype": [], "n_unique": [], "unique_pct": [], "rows": [], "top_values": []})

    tk = int(os.getenv("HYPER_REPORT_TOPK", "0")) if top_k is None else int(top_k)
    lf = _as_lazy(frame)
    one = lf.select([pl.len().alias("__rows__")] + [pl.col(c).n_unique().alias(c) for c in cols]).collect()
    rows = int(one["__rows__"][0]) if one.height else 0
    sch = _schema_mapping(frame)

    out_rows: List[Dict[str, Any]] = []
    for c in cols:
        nu = int(one[c][0]) if one.height else 0
        rec: Dict[str, Any] = {"column": c, "dtype": str(sch.get(c)), "n_unique": nu, "unique_pct": (float(nu) / rows) if rows else None, "rows": rows}
        if tk > 0:
            top = lf.select(pl.col(c)).collect().to_series(0).value_counts(sort=True).head(tk)
            rec["top_values"] = top.to_dicts()
        out_rows.append(rec)
    return pl.DataFrame(out_rows)


# -----------------------------------------------------------------------------
# Fill helpers
# -----------------------------------------------------------------------------
def _fill_expr_chain(e: pl.Expr, *, preference: str) -> pl.Expr:
    if preference == "forwards":
        return e.forward_fill().backward_fill()
    if preference == "backwards":
        return e.backward_fill().forward_fill()
    raise ValueError("preference must be 'forwards' or 'backwards'")


def _fill_frame(
        frame: FrameLike,
        *,
        mode: str,
        columns: Optional[Sequence[str]] = None,
        null_like: Optional[Sequence[Any]] = None,
        preference: str = "forwards",
        non_positive_to_null: bool = True,
) -> FrameLike:
    cols = ensure_list(columns) if columns is not None else _get_columns(frame)
    null_values = _normalize_null_like(null_like)
    exprs: List[pl.Expr] = []
    for c in cols:
        e = _apply_null_like_values(pl.col(c), null_values)
        if mode == "forward":
            exprs.append(e.forward_fill().alias(c))
        elif mode == "backward":
            exprs.append(e.backward_fill().alias(c))
        elif mode == "linear":
            exprs.append(e.interpolate().alias(c))
        elif mode == "geometric":
            if non_positive_to_null:
                e = pl.when(e > 0).then(e).otherwise(pl.lit(None))
            exprs.append(e.log().interpolate().exp().alias(c))
        elif mode == "both":
            exprs.append(_fill_expr_chain(e, preference=preference).alias(c))
        else:
            raise ValueError("Unsupported fill mode")
    out = _as_lazy(frame).with_columns(exprs)
    return out if isinstance(frame, pl.LazyFrame) else out.collect()


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def _random_seed() -> int:
    return int(np.random.rand() * 1_000_000_000) & ((1 << 64) - 1)

def _sample_frame(frame: FrameLike, *, n: int, consecutive: bool = False, seed: Optional[int] = None) -> FrameLike:
    if n <= 0:
        return frame if isinstance(frame, pl.LazyFrame) else frame.head(0)

    if isinstance(frame, pl.DataFrame):
        h = frame.height
        if h == 0:
            return frame
        if consecutive:
            if h <= n:
                return frame
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, h - n + 1))
            return frame.slice(start, n)
        if n >= h:
            return frame
        try:
            return frame.sample(n=n, with_replacement=False, seed=seed)
        except TypeError:
            return frame.sample(n=n, seed=seed)

    lf: pl.LazyFrame = frame  # type: ignore[assignment]

    if consecutive:
        h = frame_height(lf)
        if h == 0:
            return lf
        if h <= n:
            return lf
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, h - n + 1))
        return lf.slice(start, n)

    if hasattr(pl.LazyFrame, "sample"):
        try:
            return lf.sample(n=n, with_replacement=False, seed=seed)
        except TypeError:
            try:
                return lf.sample(n=n, seed=seed)
            except TypeError:
                pass

    cols0 = _get_columns(lf)
    row_idx = _unique_temp_name("__hyper_row__", cols0)
    key_col = _unique_temp_name("__hyper_key__", cols0 + [row_idx])

    seed_u = _random_seed() if seed is None else (int(seed) & ((1 << 64) - 1))
    idx_expr = pl.col(row_idx).cast(pl.UInt64)

    if hasattr(pl.Expr, "hash"):
        try:
            key_expr = idx_expr.hash(seed=seed_u).alias(key_col)
        except TypeError:
            key_expr = idx_expr.hash().alias(key_col)
    else:
        c1 = 6364136223846793005
        c2 = 1442695040888963407 ^ seed_u
        key_expr = (idx_expr * pl.lit(c1, dtype=pl.UInt64) + pl.lit(c2, dtype=pl.UInt64)).alias(key_col)

    return (
        lf.with_row_index(row_idx)
        .with_columns(key_expr)
        .sort(key_col)
        .head(n)
        .drop([row_idx, key_col])
    )


# -----------------------------------------------------------------------------
# Row-wise mode / weighted mode
# -----------------------------------------------------------------------------
def _row_mode_fast_expr(columns: Sequence[str], *, drop_nulls: bool = True) -> pl.Expr:
    lst = pl.concat_list([pl.col(c) for c in columns])
    lst = lst.list.drop_nulls() if drop_nulls else lst
    return lst.list.mode().list.get(0)


def _row_mode_lazy(
        frame: FrameLike,
        columns: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        *,
        drop_nulls: bool = True,
        output_name: str = "row_mode",
        include_score: bool = True,
) -> pl.LazyFrame:
    if not columns:
        raise ValueError("columns must be non-empty for row_mode")
    if weights is None and not include_score:
        return _as_lazy(frame).with_columns(_row_mode_fast_expr(columns, drop_nulls=drop_nulls).alias(output_name))
    if weights is not None and len(weights) != len(columns):
        raise ValueError("weights length must match columns length")

    lf = _as_lazy(frame)
    row_idx = _unique_temp_name("__hyper_row__", _get_columns(frame))
    lf_idx = lf.with_row_index(row_idx)
    columns = ensure_list(columns)
    melted = lf_idx.melt(id_vars=[row_idx], value_vars=columns, variable_name="__column__", value_name="__value__")
    if drop_nulls:
        melted = melted.filter(pl.col("__value__").is_not_null())
    if weights is None:
        weighted = melted.with_columns(pl.lit(1.0).alias("__weight__"))
    else:
        wdf = pl.DataFrame({"__column__": columns, "__weight__": [float(x) for x in weights]})
        weighted = melted.join(wdf.lazy(), on="__column__", how="left").with_columns(pl.col("__weight__").fill_null(0.0))

    scores = weighted.group_by([row_idx, "__value__"], maintain_order=True).agg(score=pl.col("__weight__").sum())
    best = (
        scores.sort(by=[row_idx, "score"], descending=[False, True])
        .group_by(row_idx, maintain_order=True)
        .head(1)
        .select(pl.col(row_idx), pl.col("__value__").alias(output_name), pl.col("score").alias(output_name + "_score"))
    )
    joined = lf_idx.join(best, on=row_idx, how="left").drop(row_idx)
    return joined if include_score else joined.drop(output_name + "_score")


def _weighted_mode_vertical_lazy(
        frame: FrameLike,
        value_col: str,
        weight_col: str,
        *,
        drop_nulls: bool = True,
        output_name: str = "weighted_mode",
        include_score: bool = True,
) -> pl.LazyFrame:
    lf = _as_lazy(frame)
    base = lf.select(pl.col(value_col).alias("__value__"), _ensure_float(weight_col).alias("__weight__"))
    if drop_nulls:
        base = base.filter(pl.col("__value__").is_not_null() & pl.col("__weight__").is_not_null())
    best = base.group_by("__value__").agg(score=pl.col("__weight__").sum()).sort("score", descending=True).head(1)
    exprs: List[pl.Expr] = [pl.col("__value__").alias(output_name)]
    if include_score:
        exprs.append(pl.col("score").alias(output_name + "_score"))
    return best.select(exprs)


# -----------------------------------------------------------------------------
# Weighted median
# -----------------------------------------------------------------------------
def _weighted_median_lazy(
        frame: FrameLike,
        value_col: str,
        weight_col: str,
        *,
        group_by: Optional[Sequence[str]] = None,
        drop_nulls: bool = True,
) -> pl.LazyFrame:
    keys = ensure_list(group_by) if group_by else []
    lf = _as_lazy(frame).select(keys + [pl.col(value_col).alias("__v"), _ensure_float(weight_col).alias("__w")])
    if drop_nulls:
        lf = lf.filter(pl.col("__v").is_not_null() & pl.col("__w").is_not_null())
    lf = lf.sort(keys + ["__v"]) if keys else lf.sort("__v")
    if keys:
        lf = lf.with_columns(
            [
                pl.col("__w").cum_sum().over(keys).alias("__cw"),
                pl.col("__w").sum().over(keys).alias("__tw"),
            ]
        ).filter(pl.col("__cw") >= pl.col("__tw") * 0.5)
        return lf.group_by(keys, maintain_order=True).agg(pl.col("__v").first().alias("weighted_median"))
    lf = lf.with_columns([pl.col("__w").cum_sum().alias("__cw"), pl.col("__w").sum().alias("__tw")]).filter(pl.col("__cw") >= pl.col("__tw") * 0.5)
    return lf.select(pl.col("__v").first().alias("weighted_median"))


# -----------------------------------------------------------------------------
# Interpolation (numpy core; lazy-safe join back)
# -----------------------------------------------------------------------------
def interpolate_column(
        fit_df: Union[pl.DataFrame, pl.LazyFrame],
        apply_df: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
        x_col: str = "x",
        y_col: str = "y",
        x_input_col: Optional[str] = None,
        out_col: str = "y_hat",
        *,
        clamp_x: bool = True,
        dedupe: str = "mean",
        weight_col: Optional[str] = None,
        finite_only: bool = True,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    if apply_df is None:
        apply_df = fit_df
    if x_input_col is None:
        x_input_col = x_col

    fit_lf = _as_lazy(fit_df).select([pl.col(x_col), pl.col(y_col)] + ([pl.col(weight_col)] if weight_col else []))
    fit = fit_lf.collect()
    x_fit = fit.get_column(x_col).to_numpy()
    y_fit = fit.get_column(y_col).to_numpy()
    w_fit = fit.get_column(weight_col).to_numpy() if weight_col else None

    if finite_only:
        m = np.isfinite(x_fit) & np.isfinite(y_fit)
        if w_fit is not None:
            m &= np.isfinite(w_fit)
        x_fit, y_fit = x_fit[m], y_fit[m]
        if w_fit is not None:
            w_fit = w_fit[m]

    x_fit = x_fit.astype(np.float64, copy=False)
    y_fit = y_fit.astype(np.float64, copy=False)
    if x_fit.size < 2:
        raise ValueError("need at least 2 fit points")

    order = np.argsort(x_fit, kind="mergesort")
    x_fit, y_fit = x_fit[order], y_fit[order]
    if w_fit is not None:
        w_fit = w_fit.astype(np.float64, copy=False)[order]
        w_fit = np.where(np.isfinite(w_fit) & (w_fit > 0.0), w_fit, 0.0)

    if dedupe not in ("mean", "last", "first"):
        raise ValueError('dedupe must be "mean", "last", or "first"')

    if np.any(np.diff(x_fit) == 0.0):
        change = np.empty_like(x_fit, dtype=bool)
        change[0] = True
        change[1:] = x_fit[1:] != x_fit[:-1]
        idx = np.nonzero(change)[0]
        xs = x_fit[idx]
        if dedupe == "last":
            ys = y_fit[np.r_[idx[1:] - 1, len(y_fit) - 1]]
        elif dedupe == "first":
            ys = y_fit[idx]
        else:
            ys = np.empty(xs.shape[0], dtype=np.float64)
            if w_fit is None:
                for i in range(xs.shape[0]):
                    a = idx[i]
                    b = idx[i + 1] if i + 1 < xs.shape[0] else len(x_fit)
                    ys[i] = float(np.mean(y_fit[a:b]))
            else:
                for i in range(xs.shape[0]):
                    a = idx[i]
                    b = idx[i + 1] if i + 1 < xs.shape[0] else len(x_fit)
                    ww = w_fit[a:b]
                    s = float(np.sum(ww))
                    ys[i] = float(np.sum(y_fit[a:b] * ww) / s) if s > 0.0 else float(np.mean(y_fit[a:b]))
        x_u, y_u = xs, ys
    else:
        x_u, y_u = x_fit, y_fit

    x_min, x_max = float(x_u[0]), float(x_u[-1])

    apply_lf = _as_lazy(apply_df)
    row_idx = _unique_temp_name("__hyper_row__", _get_columns(apply_df))
    small = apply_lf.with_row_index(row_idx).select([row_idx, pl.col(x_input_col)]).collect()
    x_app = small.get_column(x_input_col).to_numpy().astype(np.float64, copy=False)

    y_hat = np.full(x_app.shape[0], np.nan, dtype=np.float64)
    m = np.isfinite(x_app)
    xa = x_app[m]
    if clamp_x:
        xa = np.clip(xa, x_min, x_max)
    y_hat[m] = np.interp(xa, x_u, y_u)

    add = pl.DataFrame({row_idx: small.get_column(row_idx), out_col: pl.Series(out_col, y_hat)})
    out = apply_lf.with_row_index(row_idx).join(add.lazy(), on=row_idx, how="left").drop(row_idx)
    return out if isinstance(apply_df, pl.LazyFrame) else out.collect()


# -----------------------------------------------------------------------------
# Pivot (ticker + fields => wide)
# -----------------------------------------------------------------------------
def pivot_ticker_fields(
        frame: pl.DataFrame | pl.LazyFrame,
        ticker_col: str = "ticker",
        value_cols: list[str] | None = None,
        id_cols: list[str] | None = None,
        agg: str = "first",
        output_single_row: bool = True,
        constant_row_id_col: str = "__row_id",
        variable_name: str = "__field",
        value_name: str = "__value",
        wide_col_name: str = "__wide_col",
        separator: str = "",
        prefix: str = "",
        infix: str = "",
        suffix: str = "",
        on_columns: list[str] | None = None,
        infer_on_columns: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    is_lazy = isinstance(frame, pl.LazyFrame)
    lf = frame if is_lazy else frame.lazy()

    sch = lf.collect_schema()
    if ticker_col not in sch:
        raise ValueError(f"Missing ticker_col='{ticker_col}'. Available: {ensure_list(sch)}")

    id_cols = id_cols or []
    for c in id_cols:
        if c not in sch:
            raise ValueError(f"Missing id_col='{c}'. Available: {ensure_list(sch)}")

    if value_cols is None:
        excluded = set(id_cols)
        excluded.add(ticker_col)
        value_cols = [c for c in sch.keys() if c not in excluded]
        if not value_cols:
            raise ValueError("No value columns to pivot.")

    for c in value_cols:
        if c not in sch:
            raise ValueError(f"Missing value_col='{c}'. Available: {ensure_list(sch)}")

    if output_single_row:
        if constant_row_id_col in sch:
            raise ValueError(f"constant_row_id_col='{constant_row_id_col}' already exists.")
        pivot_index = [constant_row_id_col]
        base = lf.select([ticker_col, *value_cols]).with_columns(pl.lit(0, dtype=pl.Int32).alias(constant_row_id_col))
    else:
        if not id_cols:
            raise ValueError("When output_single_row=False, id_cols must be provided.")
        pivot_index = ensure_list(id_cols)
        base = lf.select([*id_cols, ticker_col, *value_cols])

    mid = infix if infix != "" else separator
    parts: List[pl.Expr] = []
    if prefix:
        parts.append(pl.lit(prefix))
    parts.append(pl.col(ticker_col))
    if mid:
        parts.append(pl.lit(mid))
    parts.append(pl.col(variable_name))
    if suffix:
        parts.append(pl.lit(suffix))

    long = (
        base.unpivot(
            index=[*pivot_index, ticker_col],
            on=value_cols,
            variable_name=variable_name,
            value_name=value_name,
        )
        .with_columns(pl.concat_str(parts, ignore_nulls=True).alias(wide_col_name))
    )

    if is_lazy:
        if on_columns is None:
            if not infer_on_columns:
                raise ValueError("Lazy pivot requires on_columns or infer_on_columns=True.")
            keys_df = long.select(pl.col(wide_col_name).unique().sort()).collect()
            on_columns = keys_df.get_column(wide_col_name).to_list()

        wide = long.pivot(
            on=wide_col_name,
            on_columns=on_columns,
            index=pivot_index,
            values=value_name,
            aggregate_function=agg,
        )
        if output_single_row:
            wide = wide.drop(constant_row_id_col)
        return wide

    df_long = long.collect()
    df_wide = df_long.pivot(on=wide_col_name, index=pivot_index, values=value_name, aggregate_function=agg)
    if output_single_row:
        df_wide = df_wide.drop(constant_row_id_col)
    return df_wide

def _unique_nonnull_values_for_from_columns(frame: FrameLike, col: str, *, max_unique: int) -> List[Any]:
    lf = _as_lazy(frame)
    s = lf.select(pl.col(col).drop_nulls().unique().head(max_unique)).collect().to_series(0)
    return s.to_list()


def _from_columns_expr(
        frame: FrameLike,
        from_col: str,
        to_col: str,
        *,
        on_missing: Any = None,
        on_unmatched: Any = None,
        max_unique: Optional[int] = None,
) -> pl.Expr:
    cols = set(_get_columns(frame))
    if from_col not in cols:
        return pl.lit(on_missing).alias(to_col)

    lim = int(os.getenv("HYPER_DYNAMIC_COL_MAX", "256")) if max_unique is None else int(max_unique)
    if lim <= 0:
        lim = 256

    vals = _unique_nonnull_values_for_from_columns(frame, from_col, max_unique=lim)
    candidates = [v for v in vals if isinstance(v, str) and v in cols]

    if not candidates:
        return pl.lit(on_missing).alias(to_col)

    base = pl.lit(on_unmatched) if on_unmatched is not None else pl.lit(None)
    f = pl.col(from_col)
    for name in candidates:
        base = pl.when(f == pl.lit(name)).then(pl.col(name)).otherwise(base)

    return pl.when(f.is_null()).then(pl.lit(on_missing)).otherwise(base).alias(to_col)


def _from_columns_apply(
        frame: FrameLike,
        from_cols: Union[Sequence[str], str],
        to_cols: Union[Sequence[str], str],
        *,
        on_missing: Any = None,
        on_unmatched: Any = None,
        max_unique: Optional[int] = None,
) -> FrameLike:
    fcols = ensure_list(from_cols)
    tcols = ensure_list(to_cols)
    if len(fcols) != len(tcols):
        raise ValueError(f"Expected from/to column lengths must match. Got {len(fcols)} vs {len(tcols)}")

    exprs = [
        _from_columns_expr(
            frame,
            fcols[i],
            tcols[i],
            on_missing=on_missing,
            on_unmatched=on_unmatched,
            max_unique=max_unique,
        )
        for i in range(len(fcols))
    ]
    out = _as_lazy(frame).with_columns(exprs)
    return out if isinstance(frame, pl.LazyFrame) else out.collect()


# -----------------------------------------------------------------------------
# kdb formatting helpers (optional)
# -----------------------------------------------------------------------------
def _kdb_sym_from_list(values: List[Any], *, on_none: Any = None) -> Any:
    if not values:
        return on_none
    return "`" + "`".join(str(x) for x in values)


def _kdb_str_from_list(values: List[Any], *, on_none: Any = None) -> Any:
    if not values:
        return on_none
    if len(values) == 1:
        return f"\"{values[0]}\";"
    return "\"" + "\";\"".join(str(x) for x in values) + "\""


def _kdb_sym_str_from_list(values: List[Any], *, on_none: Any = None) -> Any:
    if not values:
        return on_none
    return "`$\"" + "\";`$\"".join(str(x) for x in values) + "\""


# -----------------------------------------------------------------------------
# Winsorization / Outlier detection
# -----------------------------------------------------------------------------
class OutlierMethod(Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    MAD = "mad"
    PERCENTILE = "percentile"
    GRUBBS = "grubbs"
    KNN = "knn"
    LOF = "lof"
    IFOREST = "iforest"
    ECOD = "ecod"
    COPOD = "copod"
    HBOS = "hbos"
    OCSVM = "ocsvm"
    ABOD = "abod"
    PCA = "pca"
    LODA = "loda"
    SUOD = "suod"


_PYOD_METHOD_SET = frozenset({
    OutlierMethod.KNN, OutlierMethod.LOF, OutlierMethod.IFOREST,
    OutlierMethod.ECOD, OutlierMethod.COPOD, OutlierMethod.HBOS,
    OutlierMethod.OCSVM, OutlierMethod.ABOD, OutlierMethod.PCA,
    OutlierMethod.LODA, OutlierMethod.SUOD,
})

_PYOD_CLASS_MAP: Dict[OutlierMethod, Any] = {}
if _HAS_PYOD:
    _PYOD_CLASS_MAP = {
        OutlierMethod.KNN: PyodKNN,
        OutlierMethod.LOF: PyodLOF,
        OutlierMethod.IFOREST: PyodIForest,
        OutlierMethod.ECOD: PyodECOD,
        OutlierMethod.COPOD: PyodCOPOD,
        OutlierMethod.HBOS: PyodHBOS,
        OutlierMethod.OCSVM: PyodOCSVM,
        OutlierMethod.ABOD: PyodABOD,
        OutlierMethod.PCA: PyodPCA,
        OutlierMethod.LODA: PyodLODA,
    }

_PYOD_MIN_SAMPLES: Dict[OutlierMethod, int] = {
    OutlierMethod.KNN: 6,
    OutlierMethod.LOF: 6,
    OutlierMethod.IFOREST: 8,
    OutlierMethod.ECOD: 5,
    OutlierMethod.COPOD: 5,
    OutlierMethod.HBOS: 5,
    OutlierMethod.OCSVM: 5,
    OutlierMethod.ABOD: 6,
    OutlierMethod.PCA: 5,
    OutlierMethod.LODA: 5,
    OutlierMethod.SUOD: 8,
}


_OUTLIER_DEFAULT_METHOD = OutlierMethod.MAD
_OUTLIER_DEFAULT_SENSITIVITY = 2.5
_OUTLIER_DEFAULT_PERCENTILE_BOUNDS = (0.05, 0.95)
_OUTLIER_DEFAULT_SYMMETRIC = True
_OUTLIER_DEFAULT_NULLS_AS_ZERO = False
_OUTLIER_DEFAULT_ZEROS_AS_NULL = True
_OUTLIER_DEFAULT_FILTER_NULLS = True
_OUTLIER_DEFAULT_AUTO_RESCALE = True
_OUTLIER_DEFAULT_WEIGHT_COL: Optional[str] = None
_OUTLIER_DEFAULT_CONTAMINATION: Optional[float] = None


def _sensitivity_to_contamination(sensitivity: float) -> float:
    raw = 0.5 * np.exp(-0.5 * sensitivity)
    return float(np.clip(raw, 0.01, 0.49))


def _resolve_contamination(
    contamination: Optional[float],
    sensitivity: float,
) -> float:
    if contamination is not None:
        return float(np.clip(contamination, 0.01, 0.49))
    return _sensitivity_to_contamination(sensitivity)


def _build_pyod_detector(
    method: OutlierMethod,
    contamination: float,
    *,
    pyod_params: Optional[Dict[str, Any]] = None,
):
    if not _HAS_PYOD:
        raise ImportError(
            f"pyod is required for OutlierMethod.{method.name}. "
            "Install with: pip install pyod"
        )
    params = dict(pyod_params) if pyod_params else {}
    params.setdefault("contamination", contamination)

    if method == OutlierMethod.SUOD:
        base_estimators = [
            _PYOD_CLASS_MAP[OutlierMethod.LOF](contamination=contamination),
            _PYOD_CLASS_MAP[OutlierMethod.COPOD](contamination=contamination),
            _PYOD_CLASS_MAP[OutlierMethod.IFOREST](contamination=contamination),
        ]
        return PyodSUOD(base_estimators=base_estimators, contamination=contamination)

    if method == OutlierMethod.MAD:
        threshold = params.pop("threshold", params.pop("sensitivity", 3.5))
        cont = params.pop("contamination", contamination)
        return PyodMAD(threshold=threshold, contamination=cont)

    cls = _PYOD_CLASS_MAP.get(method)
    if cls is None:
        raise ValueError(f"No pyod class mapped for {method!r}")
    return cls(**params)


def _pyod_detect_1d(
    arr: np.ndarray,
    *,
    method: OutlierMethod,
    contamination: float,
    pyod_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    nan_mask = np.isnan(arr)
    valid = arr[~nan_mask]
    min_n = _PYOD_MIN_SAMPLES.get(method, 5)
    if valid.size < min_n:
        return np.zeros(arr.shape, dtype=np.bool_)
    X = valid.reshape(-1, 1)
    clf = _build_pyod_detector(method, contamination, pyod_params=pyod_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X)
    result = np.zeros(arr.shape, dtype=np.bool_)
    result[~nan_mask] = clf.labels_.astype(bool)
    return result


def _pyod_detect_2d_rowwise(
    data: np.ndarray,
    *,
    method: OutlierMethod,
    contamination: float,
    pyod_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    n_rows, n_cols = data.shape
    nan_mask = np.isnan(data)
    valid_count = np.sum(~nan_mask, axis=1)
    outlier_mask = np.zeros_like(data, dtype=np.bool_)
    min_n = _PYOD_MIN_SAMPLES.get(method, 5)

    for i in range(n_rows):
        row = data[i]
        row_nan = nan_mask[i]
        n_valid = valid_count[i]
        if n_valid < min_n:
            continue
        valid_vals = row[~row_nan].reshape(-1, 1)
        clf = _build_pyod_detector(method, contamination, pyod_params=pyod_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(valid_vals)
        row_labels = clf.labels_.astype(bool)
        row_mask = np.zeros(n_cols, dtype=np.bool_)
        row_mask[~row_nan] = row_labels
        outlier_mask[i] = row_mask

    return outlier_mask


def _winsor_preprocess_columns(
    lf: pl.LazyFrame,
    columns: List[str],
    *,
    nulls_as_zero: bool,
    zeros_as_null: bool,
    filter_nulls: bool,
) -> pl.LazyFrame:
    exprs = []
    for c in columns:
        col_expr = pl.col(c).cast(pl.Float64)
        if zeros_as_null:
            col_expr = pl.when(col_expr == 0.0).then(None).otherwise(col_expr)
        if nulls_as_zero and (not zeros_as_null):
            col_expr = col_expr.fill_null(0.0)
        exprs.append(col_expr.alias(c))
    return lf.with_columns(exprs)

def normalize_method(x):
    if isinstance(x, OutlierMethod): return x
    return OutlierMethod.__getitem__(str(x).upper())

def _detect_outliers_1d(
    arr: np.ndarray,
    *,
    method: OutlierMethod,
    sensitivity: float,
    percentile_bounds: tuple,
    symmetric: bool,
    contamination: Optional[float] = None,
    pyod_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if method in _PYOD_METHOD_SET:
        c = _resolve_contamination(contamination, sensitivity)
        return _pyod_detect_1d(arr, method=method, contamination=c, pyod_params=pyod_params)

    mask = np.isnan(arr)
    valid = arr[~mask]
    n = valid.size
    if n < 3:
        return np.zeros(arr.shape, dtype=np.bool_)

    if method == OutlierMethod.IQR:
        q1, q3 = np.nanpercentile(valid, [25, 75])
        iqr = q3 - q1
        lower = q1 - (sensitivity * iqr)
        upper = q3 + (sensitivity * iqr)
        return (~mask) & ((arr < lower) | (arr > upper))

    if method == OutlierMethod.ZSCORE:
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1) if n > 1 else 1.0
        if sigma < 1e-15:
            return np.zeros(arr.shape, dtype=np.bool_)
        z = np.abs(arr - mu) / sigma
        return (~mask) & (z > sensitivity)

    if method == OutlierMethod.MAD:
        if _HAS_PYOD:
            c = _resolve_contamination(contamination, sensitivity)
            params = dict(pyod_params) if pyod_params else {}
            params.setdefault("threshold", sensitivity)
            return _pyod_detect_1d(arr, method=OutlierMethod.MAD, contamination=c, pyod_params=params)
        med = np.nanmedian(valid)
        abs_dev = np.abs(valid - med)
        mad = np.nanmedian(abs_dev)
        if mad < 1e-15:
            mad = np.nanmean(abs_dev) * 1.2533
            if mad < 1e-15:
                return np.zeros(arr.shape, dtype=np.bool_)
        modified_z = 0.6745 * np.abs(arr - med) / mad
        return (~mask) & (modified_z > sensitivity)

    if method == OutlierMethod.PERCENTILE:
        lo, hi = percentile_bounds
        lower = np.nanpercentile(valid, lo * 100)
        upper = np.nanpercentile(valid, hi * 100)
        return (~mask) & ((arr < lower) | (arr > upper))

    if method == OutlierMethod.GRUBBS:
        if _HAS_PYOD:
            c = _resolve_contamination(contamination, sensitivity)
            return _pyod_detect_1d(arr, method=OutlierMethod.ECOD, contamination=c, pyod_params=pyod_params)
        if n < 6:
            return _detect_outliers_1d(
                arr, method=OutlierMethod.IQR, sensitivity=sensitivity,
                percentile_bounds=percentile_bounds, symmetric=symmetric,
            )
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1)
        if sigma < 1e-15:
            return np.zeros(arr.shape, dtype=np.bool_)
        g_scores = np.abs(arr - mu) / sigma
        alpha = 0.05
        t_approx = sensitivity
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_approx ** 2) / ((n - 2) + (t_approx ** 2)))
        return (~mask) & (g_scores > g_crit)

    return np.zeros(arr.shape, dtype=np.bool_)


def _detect_outliers_2d_rowwise(
    data: np.ndarray,
    *,
    method: OutlierMethod,
    sensitivity: float,
    percentile_bounds: tuple,
    symmetric: bool,
    contamination: Optional[float] = None,
    pyod_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if method in _PYOD_METHOD_SET:
        c = _resolve_contamination(contamination, sensitivity)
        return _pyod_detect_2d_rowwise(data, method=method, contamination=c, pyod_params=pyod_params)

    n_rows, n_cols = data.shape
    nan_mask = np.isnan(data)
    valid_count = np.sum(~nan_mask, axis=1)
    outlier_mask = np.zeros_like(data, dtype=np.bool_)
    enough_data = valid_count >= 3

    if not np.any(enough_data):
        return outlier_mask

    if method == OutlierMethod.MAD:
        if _HAS_PYOD:
            c = _resolve_contamination(contamination, sensitivity)
            params = dict(pyod_params) if pyod_params else {}
            params.setdefault("threshold", sensitivity)
            return _pyod_detect_2d_rowwise(data, method=OutlierMethod.MAD, contamination=c, pyod_params=params)
        med = np.nanmedian(data, axis=1, keepdims=True)
        abs_dev_all = np.where(nan_mask, np.nan, np.abs(data - med))
        mad = np.nanmedian(abs_dev_all, axis=1, keepdims=True)
        zero_mad = (mad < 1e-15).ravel()
        if np.any(zero_mad & enough_data):
            mean_dev = np.nanmean(abs_dev_all, axis=1, keepdims=True) * 1.2533
            mad = np.where(mad < 1e-15, mean_dev, mad)
        safe_mad = np.where(mad < 1e-15, 1.0, mad)
        modified_z = 0.6745 * np.abs(data - med) / safe_mad
        is_outlier = modified_z > sensitivity
        still_zero = (mad < 1e-15).ravel()
        if np.any(still_zero):
            is_outlier[still_zero, :] = False
        outlier_mask = (~nan_mask) & is_outlier & enough_data[:, np.newaxis]

    elif method == OutlierMethod.IQR:
        q1 = np.nanpercentile(data, 25, axis=1, keepdims=True)
        q3 = np.nanpercentile(data, 75, axis=1, keepdims=True)
        iqr = q3 - q1
        lower = q1 - (sensitivity * iqr)
        upper = q3 + (sensitivity * iqr)
        outlier_mask = (~nan_mask) & ((data < lower) | (data > upper)) & enough_data[:, np.newaxis]

    elif method == OutlierMethod.ZSCORE:
        mu = np.nanmean(data, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            sigma = np.nanstd(data, axis=1, ddof=1, keepdims=True)
        safe_sigma = np.where(sigma < 1e-15, 1.0, sigma)
        z = np.abs(data - mu) / safe_sigma
        zero_sigma = (sigma < 1e-15).ravel()
        is_outlier = z > sensitivity
        if np.any(zero_sigma):
            is_outlier[zero_sigma, :] = False
        outlier_mask = (~nan_mask) & is_outlier & enough_data[:, np.newaxis]

    elif method == OutlierMethod.PERCENTILE:
        lo, hi = percentile_bounds
        lower = np.nanpercentile(data, lo * 100, axis=1, keepdims=True)
        upper = np.nanpercentile(data, hi * 100, axis=1, keepdims=True)
        outlier_mask = (~nan_mask) & ((data < lower) | (data > upper)) & enough_data[:, np.newaxis]

    elif method == OutlierMethod.GRUBBS:
        if _HAS_PYOD:
            c = _resolve_contamination(contamination, sensitivity)
            return _pyod_detect_2d_rowwise(data, method=OutlierMethod.ECOD, contamination=c, pyod_params=pyod_params)
        need_fallback = enough_data & (valid_count < 6)
        can_grubbs = enough_data & (valid_count >= 6)

        if np.any(need_fallback):
            fallback = _detect_outliers_2d_rowwise(
                data[need_fallback],
                method=OutlierMethod.IQR, sensitivity=sensitivity,
                percentile_bounds=percentile_bounds, symmetric=symmetric,
            )
            outlier_mask[need_fallback] = fallback

        if np.any(can_grubbs):
            sub = data[can_grubbs]
            sub_nan = np.isnan(sub)
            mu = np.nanmean(sub, axis=1, keepdims=True)
            sigma = np.nanstd(sub, axis=1, ddof=1, keepdims=True)
            safe_sigma = np.where(sigma < 1e-15, 1.0, sigma)
            g_scores = np.abs(sub - mu) / safe_sigma
            ns = np.sum(~sub_nan, axis=1)
            t_approx = sensitivity
            g_crits = np.empty(ns.shape)
            for idx, ni in enumerate(ns):
                if ni < 3 or sigma.ravel()[idx] < 1e-15:
                    g_crits[idx] = np.inf
                else:
                    g_crits[idx] = ((ni - 1) / np.sqrt(ni)) * np.sqrt((t_approx ** 2) / ((ni - 2) + (t_approx ** 2)))
            is_outlier = g_scores > g_crits[:, np.newaxis]
            zero_sig = (sigma < 1e-15).ravel()
            if np.any(zero_sig):
                is_outlier[zero_sig, :] = False
            outlier_mask[can_grubbs] = (~sub_nan) & is_outlier

    return outlier_mask


def _try_rescale_value(
    val: float,
    reference_median: float,
    reference_mad: float,
) -> float:
    if (np.isnan(val)) or (np.isnan(reference_median)) or (reference_median == 0.0):
        return val
    ratio = val / reference_median
    if (80 < ratio < 120) or (-120 < ratio < -80):
        candidate = val / 100.0
        if reference_mad > 1e-15:
            if abs(candidate - reference_median) / reference_mad < 3.5:
                return candidate
        elif abs(candidate - reference_median) < (abs(reference_median) * 0.5):
            return candidate
    elif (0.008 < ratio < 0.012) or (-0.012 < ratio < -0.008):
        candidate = val * 100.0
        if reference_mad > 1e-15:
            if abs(candidate - reference_median) / reference_mad < 3.5:
                return candidate
        elif abs(candidate - reference_median) < (abs(reference_median) * 0.5):
            return candidate
    return val


def _auto_rescale_array(arr: np.ndarray) -> np.ndarray:
    valid = arr[~np.isnan(arr)]
    if valid.size < 3:
        return arr.copy()
    med = np.nanmedian(valid)
    abs_dev = np.abs(valid - med)
    mad = np.nanmedian(abs_dev)
    if mad < 1e-15:
        mad = np.nanmean(abs_dev) * 1.2533
    result = arr.copy()
    for i in range(result.shape[0]):
        if not np.isnan(result[i]):
            result[i] = _try_rescale_value(result[i], med, mad)
    return result


def _auto_rescale_2d_rowwise(data: np.ndarray) -> np.ndarray:
    result = data.copy()
    nan_mask = np.isnan(data)
    valid_count = np.sum(~nan_mask, axis=1)
    needs_rescale = valid_count >= 3
    if not np.any(needs_rescale):
        return result
    for i in np.where(needs_rescale)[0]:
        result[i] = _auto_rescale_array(result[i])
    return result


def _winsorize_clamp_2d(
    data: np.ndarray,
    outlier_mask: np.ndarray,
) -> np.ndarray:
    nan_mask = np.isnan(data)
    safe_data = np.where(nan_mask | outlier_mask, np.nan, data)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        row_min = np.nanmin(safe_data, axis=1, keepdims=True)
        row_max = np.nanmax(safe_data, axis=1, keepdims=True)
    all_nan_rows = np.all(np.isnan(safe_data), axis=1)
    row_min[all_nan_rows, :] = np.nan
    row_max[all_nan_rows, :] = np.nan
    clamped = np.where(
        outlier_mask & (~nan_mask),
        np.clip(data, row_min, row_max),
        data,
    )
    return clamped


def _weighted_nanmean_rows(
    data: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    nan_mask = np.isnan(data)
    if weights is None:
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(data, axis=1)
    w = np.where(nan_mask | np.isnan(weights), 0.0, weights)
    vals = np.where(nan_mask, 0.0, data)
    w_sum = np.sum(w, axis=1)
    safe_w_sum = np.where(w_sum < 1e-15, np.nan, w_sum)
    return np.sum(vals * w, axis=1) / safe_w_sum


def _winsorize_column_1d(
    arr: np.ndarray,
    *,
    method: OutlierMethod,
    sensitivity: float,
    percentile_bounds: tuple,
    symmetric: bool,
    auto_rescale: bool,
    contamination: Optional[float] = None,
    pyod_params: Optional[Dict[str, Any]] = None,
) -> tuple:
    if auto_rescale:
        arr = _auto_rescale_array(arr)
    outlier_mask = _detect_outliers_1d(
        arr, method=method, sensitivity=sensitivity,
        percentile_bounds=percentile_bounds, symmetric=symmetric,
        contamination=contamination, pyod_params=pyod_params,
    )
    valid_mask = (~np.isnan(arr)) & (~outlier_mask)
    valid_vals = arr[valid_mask]
    if valid_vals.size == 0:
        return np.nan, outlier_mask
    lower_bound = np.min(valid_vals)
    upper_bound = np.max(valid_vals)
    clipped = np.where(
        outlier_mask & (~np.isnan(arr)),
        np.clip(arr, lower_bound, upper_bound),
        arr,
    )
    final_vals = clipped[~np.isnan(clipped)]
    if final_vals.size == 0:
        return np.nan, outlier_mask
    return float(np.nanmean(final_vals)), outlier_mask


def horizontal_winsor(
    df: FrameLike,
    columns: List[str],
    *,
    method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
    sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
    percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
    symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
    auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
    weight_col: Optional[str] = _OUTLIER_DEFAULT_WEIGHT_COL,
    nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
    zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
    filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
    contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
    pyod_params: Optional[Dict[str, Any]] = None,
    result_alias: str = "h_winsor_mean",
    n_threads: int = 2,
):
    lf = _as_lazy(df)
    method = normalize_method(method)
    lf = _winsor_preprocess_columns(
        lf, columns,
        nulls_as_zero=nulls_as_zero,
        zeros_as_null=zeros_as_null,
        filter_nulls=filter_nulls,
    )
    collected = lf.select(columns).collect()
    data = collected.to_numpy(writable=True).astype(np.float64)

    weight_data = None
    if weight_col is not None:
        w_series = _as_lazy(df).select(pl.col(weight_col).cast(pl.Float64)).collect()
        w_arr = w_series.to_numpy(writable=True).astype(np.float64).ravel()
        weight_data = np.tile(w_arr[:, np.newaxis], (1, data.shape[1]))

    if auto_rescale:
        data = _auto_rescale_2d_rowwise(data)

    outlier_mask = _detect_outliers_2d_rowwise(
        data, method=method, sensitivity=sensitivity,
        percentile_bounds=percentile_bounds, symmetric=symmetric,
        contamination=contamination, pyod_params=pyod_params,
    )
    clamped = _winsorize_clamp_2d(data, outlier_mask)
    results = _weighted_nanmean_rows(clamped, weight_data)

    result_series = pl.Series(result_alias, results)
    return _as_lazy(df).with_columns(result_series.alias(result_alias))


def vertical_winsor(
    df: FrameLike,
    columns: List[str],
    *,
    method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
    sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
    percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
    symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
    auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
    nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
    zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
    filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
    contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
    pyod_params: Optional[Dict[str, Any]] = None,
    result_suffix: str = "_v_winsor",
    n_threads: int = 2
):
    lf = _as_lazy(df)
    method = normalize_method(method)
    lf = _winsor_preprocess_columns(
        lf, columns,
        nulls_as_zero=nulls_as_zero,
        zeros_as_null=zeros_as_null,
        filter_nulls=filter_nulls,
    )
    collected = lf.select(columns).collect()

    def _process_col(col_name: str) -> tuple:
        arr = collected.get_column(col_name).to_numpy(writable=True).astype(np.float64)
        mean_val, _ = _winsorize_column_1d(
            arr, method=method, sensitivity=sensitivity,
            percentile_bounds=percentile_bounds, symmetric=symmetric,
            auto_rescale=auto_rescale,
            contamination=contamination, pyod_params=pyod_params,
        )
        return col_name, mean_val

    ex = _get_collect_executor()
    if len(columns) > 2 and n_threads > 1:
        futures = {ex.submit(_process_col, c): c for c in columns}
        results_map = {}
        for f in concurrent.futures.as_completed(futures):
            col_name, mean_val = f.result()
            results_map[col_name] = mean_val
    else:
        results_map = {}
        for c in columns:
            col_name, mean_val = _process_col(c)
            results_map[col_name] = mean_val

    new_cols = [
        pl.lit(results_map[c]).cast(pl.Float64).alias(f"{c}{result_suffix}")
        for c in columns
    ]
    return _as_lazy(df).with_columns(new_cols)


def vertical_winsor_w_neighbors(
    df: FrameLike,
    target_columns: List[str],
    neighbor_columns: List[str],
    *,
    method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
    sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
    percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
    symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
    auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
    nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
    zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
    filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
    contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
    pyod_params: Optional[Dict[str, Any]] = None,
    result_suffix: str = "_vn_winsor",
    n_threads: int = 2,
):
    all_cols = list(dict.fromkeys(target_columns + neighbor_columns))
    lf = _as_lazy(df)
    method = normalize_method(method)
    lf = _winsor_preprocess_columns(
        lf, all_cols,
        nulls_as_zero=nulls_as_zero,
        zeros_as_null=zeros_as_null,
        filter_nulls=filter_nulls,
    )
    collected = lf.select(all_cols).collect()
    neighbor_arrays = {
        c: collected.get_column(c).to_numpy(writable=True).astype(np.float64)
        for c in neighbor_columns
    }

    def _process_target(col_name: str) -> tuple:
        target_arr = collected.get_column(col_name).to_numpy(writable=True).astype(np.float64)
        unique_neighbors = [c for c in neighbor_columns if c != col_name]
        if unique_neighbors:
            combined = np.column_stack([target_arr] + [neighbor_arrays[c] for c in unique_neighbors])
        else:
            combined = target_arr.copy()
        flat = combined.ravel() if combined.ndim > 1 else combined
        if auto_rescale:
            flat = _auto_rescale_array(flat)
        outlier_in_combined = _detect_outliers_1d(
            flat, method=method, sensitivity=sensitivity,
            percentile_bounds=percentile_bounds, symmetric=symmetric,
            contamination=contamination, pyod_params=pyod_params,
        )
        n_rows = target_arr.shape[0]
        if combined.ndim == 1:
            target_outlier_mask = outlier_in_combined[:n_rows]
        else:
            outlier_2d = outlier_in_combined.reshape(combined.shape)
            target_outlier_mask = outlier_2d[:, 0]
        if auto_rescale:
            target_arr = _auto_rescale_array(target_arr)
        valid_mask = (~np.isnan(target_arr)) & (~target_outlier_mask)
        valid_vals = target_arr[valid_mask]
        if valid_vals.size == 0:
            return col_name, np.nan
        lower_bound = np.min(valid_vals)
        upper_bound = np.max(valid_vals)
        clipped = np.where(
            target_outlier_mask & (~np.isnan(target_arr)),
            np.clip(target_arr, lower_bound, upper_bound),
            target_arr,
        )
        final_vals = clipped[~np.isnan(clipped)]
        if final_vals.size == 0:
            return col_name, np.nan
        return col_name, float(np.nanmean(final_vals))

    ex = _get_collect_executor()
    if len(target_columns) > 2 and n_threads > 1:
        futures = {ex.submit(_process_target, c): c for c in target_columns}
        results_map = {}
        for f in concurrent.futures.as_completed(futures):
            col_name, mean_val = f.result()
            results_map[col_name] = mean_val
    else:
        results_map = {}
        for c in target_columns:
            col_name, mean_val = _process_target(c)
            results_map[col_name] = mean_val

    new_cols = [
        pl.lit(results_map[c]).cast(pl.Float64).alias(f"{c}{result_suffix}")
        for c in target_columns
    ]
    return _as_lazy(df).with_columns(new_cols)


def horizontal_winsor_w_neighbors(
    df: FrameLike,
    target_columns: List[str],
    neighbor_columns: List[str],
    *,
    method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
    sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
    percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
    symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
    auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
    weight_col: Optional[str] = _OUTLIER_DEFAULT_WEIGHT_COL,
    nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
    zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
    filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
    contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
    pyod_params: Optional[Dict[str, Any]] = None,
    result_alias: str = "h_winsor_neighbor_mean",
    n_threads: int = 2,
):
    all_cols = list(dict.fromkeys(target_columns + neighbor_columns))
    lf = _as_lazy(df)
    method = normalize_method(method)
    lf = _winsor_preprocess_columns(
        lf, all_cols,
        nulls_as_zero=nulls_as_zero,
        zeros_as_null=zeros_as_null,
        filter_nulls=filter_nulls,
    )
    collected = lf.select(all_cols).collect()
    target_data = collected.select(target_columns).to_numpy(writable=True).astype(np.float64)
    neighbor_only = [c for c in neighbor_columns if c not in target_columns]
    if neighbor_only:
        neighbor_data = collected.select(neighbor_only).to_numpy(writable=True).astype(np.float64)
        combined_data = np.hstack([target_data, neighbor_data])
    else:
        combined_data = target_data.copy()

    weight_data = None
    if weight_col is not None:
        w_series = _as_lazy(df).select(pl.col(weight_col).cast(pl.Float64)).collect()
        w_arr = w_series.to_numpy(writable=True).astype(np.float64).ravel()
        weight_data = np.tile(w_arr[:, np.newaxis], (1, target_data.shape[1]))

    if auto_rescale:
        combined_data = _auto_rescale_2d_rowwise(combined_data)
        target_data = combined_data[:, :len(target_columns)]

    outlier_mask_combined = _detect_outliers_2d_rowwise(
        combined_data, method=method, sensitivity=sensitivity,
        percentile_bounds=percentile_bounds, symmetric=symmetric,
        contamination=contamination, pyod_params=pyod_params,
    )
    target_outlier = outlier_mask_combined[:, :len(target_columns)]

    nan_mask_combined = np.isnan(combined_data)
    safe_combined = np.where(nan_mask_combined | outlier_mask_combined, np.nan, combined_data)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        row_min = np.nanmin(safe_combined, axis=1, keepdims=True)
        row_max = np.nanmax(safe_combined, axis=1, keepdims=True)
    all_nan = np.all(np.isnan(safe_combined), axis=1)
    row_min[all_nan, :] = np.nan
    row_max[all_nan, :] = np.nan

    nan_mask_target = np.isnan(target_data)
    clamped_target = np.where(
        target_outlier & (~nan_mask_target),
        np.clip(target_data, row_min, row_max),
        target_data,
    )
    results = _weighted_nanmean_rows(clamped_target, weight_data)
    result_series = pl.Series(result_alias, results)
    return _as_lazy(df).with_columns(result_series.alias(result_alias))


def horizontal_outlier_mask(
    df: FrameLike,
    columns: List[str],
    *,
    method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
    sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
    percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
    symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
    auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
    nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
    zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
    filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
    contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
    pyod_params: Optional[Dict[str, Any]] = None,
    result_suffix: str = "_h_outlier",
    n_threads: int = 2,
):
    lf = _as_lazy(df)
    lf = _winsor_preprocess_columns(
        lf, columns,
        nulls_as_zero=nulls_as_zero,
        zeros_as_null=zeros_as_null,
        filter_nulls=filter_nulls,
    )
    collected = lf.select(columns).collect()
    data = collected.to_numpy(writable=True).astype(np.float64)

    if auto_rescale:
        data = _auto_rescale_2d_rowwise(data)

    outlier_matrix = _detect_outliers_2d_rowwise(
        data, method=method, sensitivity=sensitivity,
        percentile_bounds=percentile_bounds, symmetric=symmetric,
        contamination=contamination, pyod_params=pyod_params,
    ).astype(np.int8)

    mask_series = [
        pl.Series(f"{columns[j]}{result_suffix}", outlier_matrix[:, j]).cast(pl.Int8)
        for j in range(len(columns))
    ]
    result_lf = _as_lazy(df)
    for s in mask_series:
        result_lf = result_lf.with_columns(s.alias(s.name))
    return result_lf


def vertical_outlier_mask(
    df: FrameLike,
    columns: List[str],
    *,
    method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
    sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
    percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
    symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
    auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
    nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
    zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
    filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
    contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
    pyod_params: Optional[Dict[str, Any]] = None,
    result_suffix: str = "_v_outlier",
    n_threads: int = 2,
):
    lf = _as_lazy(df)
    lf = _winsor_preprocess_columns(
        lf, columns,
        nulls_as_zero=nulls_as_zero,
        zeros_as_null=zeros_as_null,
        filter_nulls=filter_nulls,
    )
    collected = lf.select(columns).collect()

    def _process_col(col_name: str) -> tuple:
        arr = collected.get_column(col_name).to_numpy(writable=True).astype(np.float64)
        if auto_rescale:
            arr = _auto_rescale_array(arr)
        omask = _detect_outliers_1d(
            arr, method=method, sensitivity=sensitivity,
            percentile_bounds=percentile_bounds, symmetric=symmetric,
            contamination=contamination, pyod_params=pyod_params,
        )
        return col_name, omask.astype(np.int8)

    ex = _get_collect_executor()
    if len(columns) > 2 and n_threads > 1:
        futures = {ex.submit(_process_col, c): c for c in columns}
        results_map = {}
        for f in concurrent.futures.as_completed(futures):
            col_name, omask = f.result()
            results_map[col_name] = omask
    else:
        results_map = {}
        for c in columns:
            col_name, omask = _process_col(c)
            results_map[col_name] = omask

    result_lf = _as_lazy(df)
    for c in columns:
        s = pl.Series(f"{c}{result_suffix}", results_map[c]).cast(pl.Int8)
        result_lf = result_lf.with_columns(s.alias(s.name))
    return result_lf


# -----------------------------------------------------------------------------
# Expression-level when/then helpers
# -----------------------------------------------------------------------------
_WHEN_OP_MAP = {
    "eq": "eq",
    "eq_missing": "eq_missing",
    "ne": "ne",
    "ne_missing": "ne_missing",
    "gt": "gt",
    "ge": "ge",
    "lt": "lt",
    "le": "le",
}


def _to_polars_lit(val):
    if isinstance(val, pl.Expr):
        return val
    return pl.lit(val)


def _build_when_condition(expr: pl.Expr, when_value, when_op: Optional[str]):
    op = when_op or "eq"
    op_name = _WHEN_OP_MAP.get(op)
    if op_name is None:
        raise ValueError(f"Unsupported when_op: {op!r}. Valid: {list(_WHEN_OP_MAP.keys())}")
    val = _to_polars_lit(when_value)
    return getattr(expr, op_name)(val)


def _combine_conditions(conditions: List[pl.Expr], comparison: str) -> pl.Expr:
    if len(conditions) == 1:
        return conditions[0]
    if comparison == "all":
        result = conditions[0]
        for c in conditions[1:]:
            result = result & c
        return result
    result = conditions[0]
    for c in conditions[1:]:
        result = result | c
    return result


def _resolve_fill_when_condition(
    expr: pl.Expr,
    when_value,
    *,
    when_op: Optional[str],
    condition_override_func: Optional[Callable],
    check_null: bool,
    comparison: str,
) -> pl.Expr:
    if condition_override_func is not None:
        condition = condition_override_func(expr)
    elif isinstance(when_value, (list, tuple)):
        conditions = [_build_when_condition(expr, v, when_op) for v in when_value]
        condition = _combine_conditions(conditions, comparison)
    else:
        condition = _build_when_condition(expr, when_value, when_op)

    condition = condition.fill_null(pl.lit(False))

    if check_null:
        condition = pl.when(expr.is_not_null()).then(condition).otherwise(pl.lit(False))

    return condition


def _expr_output_name(expr: pl.Expr) -> str:
    try:
        return expr.meta.output_name()
    except Exception:
        names = expr.meta.root_names()
        if names:
            return names[0]
        raise

# -----------------------------------------------------------------------------
# Expression factory
# -----------------------------------------------------------------------------
class HyperExprFactory:
    @staticmethod
    def row_sum(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_sum requires at least one column")
        return pl.sum_horizontal(e)

    @staticmethod
    def row_mean(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_mean requires at least one column")
        return pl.mean_horizontal(e)

    @staticmethod
    def row_max(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_max requires at least one column")
        return pl.max_horizontal(e)

    @staticmethod
    def row_min(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_min requires at least one column")
        return pl.min_horizontal(e)

    @staticmethod
    def row_any(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_any requires at least one column")
        return pl.any_horizontal(e)

    @staticmethod
    def row_all(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_all requires at least one column")
        return pl.all_horizontal(e)

    @staticmethod
    def row_coalesce(*cols: ExprLike) -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_coalesce requires at least one column")
        return pl.coalesce(e)

    @staticmethod
    def row_median(*cols: ExprLike, null_behavior: str = "ignore") -> pl.Expr:
        e = _normalize_exprs(cols)
        if not e:
            raise ValueError("row_median requires at least one column")
        lst = pl.concat_list(e)
        clean = lst.list.drop_nulls()
        med = clean.list.median()
        if null_behavior in ("ignore", "skip"):
            return med
        if null_behavior == "propagate":
            has_null = lst.list.len() != clean.list.len()
            return pl.when(has_null).then(pl.lit(None)).otherwise(med)
        raise ValueError(f"Unsupported null_behavior: {null_behavior!r}")

    @staticmethod
    def weighted_mean(value: ExprLike, weight: ExprLike, *, coerce_numeric: bool = True, cast_strict: bool = False) -> pl.Expr:
        v = _ensure_float(value, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        w = _ensure_float(weight, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        valid = v.is_not_null() & w.is_not_null()
        num = pl.when(valid).then(v * w).otherwise(None).sum()
        den = pl.when(valid).then(w).otherwise(None).sum()
        return _safe_divide(num, den)

    @staticmethod
    def weighted_std(value: ExprLike, weight: ExprLike, *, coerce_numeric: bool = True, cast_strict: bool = False) -> pl.Expr:
        v = _ensure_float(value, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        w = _ensure_float(weight, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        valid = v.is_not_null() & w.is_not_null()
        den = pl.when(valid).then(w).otherwise(None).sum()
        num1 = pl.when(valid).then(v * w).otherwise(None).sum()
        num2 = pl.when(valid).then(v * v * w).otherwise(None).sum()
        mean = _safe_divide(num1, den)
        var = _safe_divide(num2, den) - mean * mean
        var = pl.when(var < 0).then(0.0).otherwise(var)
        return var.sqrt()

    @staticmethod
    def camel_case(value:ExprLike) -> pl.Expr:
        return value.map_elements(lambda v: clean_camel(v))

    @staticmethod
    def zscore(column: ExprLike, *, coerce_numeric: bool = True, cast_strict: bool = False) -> pl.Expr:
        c = _ensure_float(column, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        m, s = c.mean(), c.std()
        return pl.when(m.is_null() | s.is_null() | (s == 0)).then(0.0).otherwise((c - m) / s)

    @staticmethod
    def zscore_by_group(column: ExprLike, group_by: Sequence[ExprLike], *, coerce_numeric: bool = True, cast_strict: bool = False) -> pl.Expr:
        c = _ensure_float(column, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        keys = [_to_expr(k) for k in group_by]
        m, s = c.mean().over(keys), c.std().over(keys)
        return pl.when(m.is_null() | s.is_null() | (s == 0)).then(0.0).otherwise((c - m) / s)

    @staticmethod
    def rolling_weighted_mean(
            value: ExprLike,
            weight: ExprLike,
            *,
            window_size: int,
            min_periods: Optional[int] = None,
            center: bool = False,
            group_by: Optional[Sequence[ExprLike]] = None,
            coerce_numeric: bool = True,
            cast_strict: bool = False,
    ) -> pl.Expr:
        v = _ensure_float(value, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        w = _ensure_float(weight, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
        valid = v.is_not_null() & w.is_not_null()
        vw = pl.when(valid).then(v * w).otherwise(0.0)
        ww = pl.when(valid).then(w).otherwise(0.0)
        num = vw.rolling_sum(window_size=window_size, min_periods=min_periods, center=center)
        den = ww.rolling_sum(window_size=window_size, min_periods=min_periods, center=center)
        if group_by:
            keys = [_to_expr(k) for k in group_by]
            num, den = num.over(keys), den.over(keys)
        return _safe_divide(num, den)

    @staticmethod
    def utc_datetime_from_columns(
            date_col: Optional[ExprLike] = None,
            time_col: Optional[ExprLike] = None,
            *,
            date_override: Optional[datetime.date] = None,
            date_format: Optional[str] = None,
            time_format: Optional[str] = None,
            default_to_now: bool = True,
            time_zone: str = "UTC",
            time_unit: str = "us",
            strict: bool = False,
    ) -> pl.Expr:
        if date_col is None and time_col is None:
            if not default_to_now:
                raise ValueError("Either date_col/time_col must be provided or default_to_now=True")
            now = datetime.now(tz=timezone.utc)
            return pl.lit(now).dt.replace_time_zone(time_zone)

        from datetime import date
        if date_override:
            if isinstance(date_override, date):
                d = pl.lit(date_override, pl.Date).cast(pl.Utf8)
            else:
                d = pl.lit(date_override, pl.Utf8)
        else:
            d = _to_expr(date_col).cast(pl.Utf8) if date_col is not None else pl.lit(now_date(utc=True)).cast(pl.Utf8)
        t = _to_expr(time_col).cast(pl.Utf8) if time_col is not None else pl.lit(now_time(utc=True)).cast(pl.Utf8)
        combined = d + pl.lit(" ") + t
        fmt = f"{date_format} {time_format}" if (date_format and time_format) else None
        return combined.str.to_datetime(format=fmt, time_unit=time_unit, time_zone=time_zone, strict=strict, exact=False)

    @staticmethod
    def safe_list_item(expr: ExprLike, index: int = 0) -> pl.Expr:
        return _to_expr(expr).list.get(index)

    @staticmethod
    def safe_array_item(expr: ExprLike, index: int = 0) -> pl.Expr:
        return _to_expr(expr).arr.get(index)

    @staticmethod
    def safe_item(expr: ExprLike, index: int = 0, *, prefer: str = "list") -> pl.Expr:
        return HyperExprFactory.safe_array_item(expr, index=index) if prefer == "array" else HyperExprFactory.safe_list_item(expr, index=index)

    @staticmethod
    def safe_first(expr: ExprLike, *, prefer: str = "list") -> pl.Expr:
        return HyperExprFactory.safe_item(expr, index=0, prefer=prefer)

    @staticmethod
    def quick_not_null_map(test_col: ExprLike, value_if_null: ExprLike, value_if_not_null: Optional[ExprLike] = None) -> pl.Expr:
        t = _to_expr(test_col)
        v0 = _to_expr(value_if_null)
        v1 = _to_expr(value_if_not_null) if value_if_not_null is not None else _to_expr(test_col)
        return pl.when(t.is_null()).then(v0).otherwise(v1)

@pl.api.register_expr_namespace("hyper")
class HyperExprNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def wavg(self, weight):
        return HyperExprFactory.weighted_mean(self._expr, weight)

    def zscore(self, **kwargs: Any):
        return HyperExprFactory.zscore(self._expr, **kwargs)

    def zscore_by_group(self, group_by: Sequence[ExprLike], **kwargs: Any):
        return HyperExprFactory.zscore_by_group(self._expr, group_by, **kwargs)

    def safe_first(self, *, prefer: str = "list"):
        return HyperExprFactory.safe_first(self._expr, prefer=prefer)

    def safe_item(self, index: int = 0, *, prefer: str = "list"):
        return HyperExprFactory.safe_item(self._expr, index=index, prefer=prefer)

    def quick_not_null_map(self, value_if_null: ExprLike, value_if_not_null: Optional[ExprLike] = None):
        return HyperExprFactory.quick_not_null_map(self._expr, value_if_null, value_if_not_null)

    def camel_case(self):
        return HyperExprFactory.camel_case(self._expr)

    def fill_when(
        self,
        when_value,
        new_value,
        *,
        when_op: Optional[str] = None,
        condition_override_func: Optional[Callable] = None,
        original_value_override=None,
        keep_alias: bool = False,
        check_null: bool = True,
        comparison: str = "all",
    ):
        expr = self._expr
        original = _to_polars_lit(original_value_override) if original_value_override is not None else expr
        new_val = _to_polars_lit(new_value)
        condition = _resolve_fill_when_condition(
            expr, when_value,
            when_op=when_op,
            condition_override_func=condition_override_func,
            check_null=check_null,
            comparison=comparison,
        )
        result = pl.when(condition).then(new_val).otherwise(original)
        if keep_alias:
            result = result.alias(_expr_output_name(expr))
        return result

    def fill_null(self, value=None, *, include_zero_as_null: bool = False):
        expr = self._expr
        if value is None and not include_zero_as_null:
            return expr
        fill_val = _to_polars_lit(value) if value is not None else pl.lit(None)
        if include_zero_as_null:
            condition = expr.is_null() | (expr == 0)
        else:
            condition = expr.is_null()
        return pl.when(condition).then(fill_val).otherwise(expr)

    def fill_zero(self, value=None, *, include_null_as_zero: bool = False):
        expr = self._expr
        if value is None:
            return expr
        fill_val = _to_polars_lit(value)
        if include_null_as_zero:
            condition = (expr == 0) | expr.is_null()
        else:
            condition = expr == 0
        return pl.when(condition).then(fill_val).otherwise(expr)

    def filter_when(
        self,
        when_value,
        *,
        when_op: Optional[str] = None,
        condition_override_func: Optional[Callable] = None,
        check_null: bool = True,
        comparison: str = "all",
    ):
        condition = _resolve_fill_when_condition(
            self._expr, when_value,
            when_op=when_op,
            condition_override_func=condition_override_func,
            check_null=check_null,
            comparison=comparison,
        )
        return ~condition

    def filter_null(self, *, include_zero_as_null: bool = False):
        expr = self._expr
        if include_zero_as_null:
            return expr.is_not_null() & (expr != 0)
        return expr.is_not_null()

    def filter_zero(self, *, include_null_as_zero: bool = False):
        expr = self._expr
        if include_null_as_zero:
            return expr.is_not_null() & (expr != 0)
        return expr != 0

    def to_capitalcase(self):
        e = self._expr
        first = e.str.slice(0, 1).str.to_uppercase()
        rest = e.str.slice(1)
        return first + rest

    def case(self, mappings: Sequence[Tuple[Any, Any]], *, default=None):
        original = self._expr
        default_expr = _to_polars_lit(default) if default is not None else original
        if not mappings:
            return default_expr
        when_val, then_val = mappings[0]
        chain = pl.when(original.eq(_to_polars_lit(when_val))).then(_to_polars_lit(then_val))
        for when_val, then_val in mappings[1:]:
            chain = chain.when(original.eq(_to_polars_lit(when_val))).then(_to_polars_lit(then_val))
        return chain.otherwise(default_expr)


# -----------------------------------------------------------------------------
# _HyperCore: single implementation shared by DF/LF namespaces
# -----------------------------------------------------------------------------
class _HyperCore:
    __slots__ = ("_frame", "_mode")

    def __init__(self, frame: FrameLike, mode: str) -> None:
        self._frame = frame
        self._mode = mode  # "df" or "lf"

    # ---------- internal ----------
    def _is_df_mode(self) -> bool:
        return self._mode == "df"

    def _lf(self) -> pl.LazyFrame:
        return _as_lazy(self._frame)

    def _df(self, obj=None) -> pl.DataFrame:
        obj = self._frame if obj is None else obj
        return obj if isinstance(obj, pl.DataFrame) else obj.collect()

    def _out_frame(self, obj: FrameLike) -> FrameLike:
        if self._is_df_mode():
            return obj.collect() if isinstance(obj, pl.LazyFrame) else obj
        return obj if isinstance(obj, pl.LazyFrame) else obj.lazy()

    def _cols(self) -> List[str]:
        return _get_columns(self._frame)

    def _schema(self) -> Dict[str, pl.DataType]:
        return _schema_mapping(self._frame)

    def flatten(self, sep=","):
        s = self._schema()
        exprs = []
        for k,d in s.items():
            if d == pl.List:
                exprs.append(pl.col(k).list.eval(pl.element().cast(pl.String)).list.join(sep).alias(k))
        return self._frame.with_columns(exprs)

    # ---------- properties (DF/LF parity) ----------
    @property
    def fields(self): return self._cols()

    @property
    def columns(self): return self._cols()

    @property
    def dtypes(self):
        sch = self._schema()
        cols = self._cols()
        return [sch.get(c) for c in cols]

    @property
    def schema_map(self): return self._schema()

    @property
    def width(self): return len(self._cols())

    @property
    def ncols(self): return len(self._cols())

    @property
    def nrows(self): return frame_height(self._frame)

    @property
    def shape(self): return frame_height(self._frame), len(self._cols())

    @property
    def is_lazy(self): return isinstance(self._frame, pl.LazyFrame)

    @property
    def is_eager(self): return isinstance(self._frame, pl.DataFrame)

    @property
    def estimated_size_bytes(self):
        if isinstance(self._frame, pl.DataFrame):
            try:
                return int(self._frame.estimated_size())
            except Exception:
                return None
        return None

    # ---------- basic ----------
    def is_empty(self) -> bool:
        return frame_is_empty(self._frame)

    def height(self) -> int:
        return frame_height(self._frame)

    def peek(self, row: int = 0, col: int | str = 0, default: Any = None) -> Any:
        return _peek_value(self._frame, row=row, col=col, default=default)

    def get_item_0_0(self) -> Any:
        return _peek_value(self._frame)

    def collect(self, **collect_kwargs: Any):
        if isinstance(self._frame, pl.DataFrame):
            return self._frame
        return self._frame.collect(**collect_kwargs)

    def collect_threaded(self, **collect_kwargs: Any):
        if isinstance(self._frame, pl.DataFrame):
            return self._frame
        return _submit_collect(self._frame.collect, **collect_kwargs).result()

    async def collect_async(self, **collect_kwargs: Any):
        if isinstance(self._frame, pl.DataFrame):
            return self._frame
        return await _run_in_collect_executor(self._frame.collect, **collect_kwargs)

    def compress_plan(self):
        if isinstance(self._frame, pl.LazyFrame):
            return self._frame.collect().lazy()
        return self._frame

    async def compress_plan_async(self):
        if isinstance(self._frame, pl.LazyFrame):
            return (await self.collect_async()).lazy()
        return self._frame

    # ---------- schema / safety ----------
    def missing_columns(self, columns: Sequence[str]) -> List[str]:
        return missing_columns(self._frame, columns)

    def require_columns(self, columns: Sequence[str]) -> None:
        require_columns(self._frame, columns)

    def fill_missing(self, columns, *, defaults:Any=None, schema_override:Optional[Dict]=None) -> FrameLike:
        columns = ensure_list(columns)
        return fill_missing(self._frame, columns, defaults=defaults, schema_override=schema_override)

    def has_columns(self, columns: Sequence[str]) -> bool:
        return not missing_columns(self._frame, columns)

    def select_existing(self, columns: Sequence[str], *, strict: bool = False):
        cols = _select_existing_names(self._cols(), columns, strict=strict)
        return self._out_frame(self._lf().select(cols))

    def select(self, columns: Sequence[str], schema=None, *, strict: bool = False, cast: bool = True):
        cols = ensure_list(columns)
        have = set(self._cols())
        miss = [c for c in cols if c not in have]
        if strict and miss:
            raise ValueError(f"Missing columns: {miss}")
        present = [c for c in cols if c in have]
        s = _build_schema(schema, cols)
        lf = self._lf().select(present) if present else self._lf().select([])
        if miss:
            lf = lf.with_columns([pl.lit(None).cast(s.get(m, pl.String), strict=False).alias(m) for m in miss])
        if cast and schema is not None:
            lf = lf.with_columns([pl.col(c).cast(s.get(c, pl.String), strict=False).alias(c) for c in cols])
        return self._out_frame(lf.select(cols))

    def drop_if_exists(self, columns: Sequence[str]):
        have = set(self._cols())
        drop_cols = [c for c in columns if c in have]
        lf = self._lf().drop(drop_cols) if drop_cols else self._lf()
        return self._out_frame(lf)

    def rename_if_exists(self, rename_map: Mapping[str, str], *, strict: bool = False, on_conflict: str = "raise"):
        have = set(self._cols())
        if strict:
            miss = [c for c in rename_map.keys() if c not in have]
            if miss:
                raise ValueError(f"Missing columns for rename: {miss}")
        mapping = {k: v for k, v in rename_map.items() if k in have}
        mapping = _dedupe_rename_map(mapping, policy=on_conflict)
        lf = self._lf().rename(mapping) if mapping else self._lf()
        return self._out_frame(lf)

    def cast_if_exists(self, dtypes: Mapping[str, pl.DataType], *, strict: bool = False, cast_strict: bool = False):
        have = set(self._cols())
        if strict:
            miss = [c for c in dtypes.keys() if c not in have]
            if miss:
                raise ValueError(f"Missing columns for cast: {miss}")
        exprs = [pl.col(c).cast(dt, strict=cast_strict).alias(c) for c, dt in dtypes.items() if c in have]
        lf = self._lf().with_columns(exprs) if exprs else self._lf()
        return self._out_frame(lf)

    def ensure_columns(self, columns: Sequence[str], *, default: Any = None, dtypes: Optional[Mapping[str, pl.DataType]] = None, reorder_like: Optional[Sequence[str]] = None, selective: bool=False):
        have = self._schema()
        add: List[pl.Expr] = []
        columns = ensure_list(columns)
        for c in columns:
            if (c in have) and ((dtypes is None) or (have.get(c) == dtypes.get(c))):
                continue
            e = pl.lit(default).alias(c)
            if dtypes and c in dtypes:
                e = e.cast(dtypes[c], strict=False)
            add.append(e)
        lf = self._lf().with_columns(add) if add else self._lf()
        if selective:
            lf = lf.select(columns)
        if reorder_like is not None:
            cols_now = _get_columns(lf)
            final = ensure_list(reorder_like) + [c for c in cols_now if c not in set(reorder_like)]
            lf = lf.select(final)
        return self._out_frame(lf)

    def reorder(self, *, columns_first: Optional[Sequence[str]] = None, columns_last: Optional[Sequence[str]] = None, strict: bool = False):
        cols = self._cols()
        first = ensure_list(columns_first) if columns_first else []
        last = ensure_list(columns_last) if columns_last else []
        if strict:
            require_columns(self._frame, first + last)
        first = [c for c in first if c in cols]
        last = [c for c in last if c in cols]
        middle = [c for c in cols if c not in set(first) and c not in set(last)]
        return self._out_frame(self._lf().select(first + middle + last))

    def align_to(self, other: FrameLike, *, cast: bool = False, fill_missing: Any = None, drop_extra: bool = False, reorder_like_other: bool = True, cast_strict: bool = False):
        target_cols = _get_columns(other)
        lf = self._lf()
        if drop_extra:
            lf = lf.select([c for c in _get_columns(lf) if c in set(target_cols)])
        lf = _HyperCore(lf, "lf").ensure_columns(
            target_cols,
            default=fill_missing,
            dtypes=_schema_mapping(other) if cast else None,
            reorder_like=target_cols if reorder_like_other else None,
        )
        lf2 = lf if isinstance(lf, pl.LazyFrame) else lf.lazy()
        if cast:
            dtypes = _schema_mapping(other)
            exprs = [pl.col(c).cast(dt, strict=cast_strict).alias(c) for c, dt in dtypes.items() if c in set(_get_columns(lf2))]
            lf2 = lf2.with_columns(exprs) if exprs else lf2
        return self._out_frame(lf2)

    def prefix_columns(self, prefix: str, *, columns: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None):
        cols = ensure_list(columns) if columns is not None else self._cols()
        excl = set(exclude or [])
        have = set(self._cols())
        mapping = {c: f"{prefix}{c}" for c in cols if c in have and c not in excl}
        lf = self._lf().rename(mapping) if mapping else self._lf()
        return self._out_frame(lf)

    def suffix_columns(self, suffix: str, *, columns: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None):
        cols = ensure_list(columns) if columns is not None else self._cols()
        excl = ensure_set(exclude or [])
        have = set(self._cols())
        mapping = {c: f"{c}{suffix}" for c in cols if c in have and c not in excl}
        lf = self._lf().rename(mapping) if mapping else self._lf()
        return self._out_frame(lf)

    def normalize_column_names(self, *, lower: bool = True, strip: bool = True, space_to_underscore: bool = True, keep_alnum_underscore: bool = True, on_conflict: str = "raise"):
        mapping = _normalize_column_names(self._cols(), lower=lower, strip=strip, space_to_underscore=space_to_underscore, keep_alnum_underscore=keep_alnum_underscore)
        mapping = _dedupe_rename_map(mapping, policy=on_conflict)
        lf = self._lf().rename(mapping) if mapping else self._lf()
        return self._out_frame(lf)

    # ---------- column utilities ----------
    def fuzzy_columns(self, pattern: str, *, regex: bool = False, case_sensitive: bool = False, exact: bool = False, invert: bool = False) -> List[str]:
        return _fuzzy_match_columns(self._cols(), pattern, regex=regex, case_sensitive=case_sensitive, exact=exact, invert=invert)

    def select_fuzzy(self, pattern: str, **kwargs: Any):
        cols = self.fuzzy_columns(pattern, **kwargs)
        return self._out_frame(self._lf().select(cols))

    def fuzzy_select(self, pattern: str, **kwargs: Any):
        return self.select_fuzzy(pattern, **kwargs)

    def fz(self, pattern: str, **kwargs: Any):
        return self.select_fuzzy(pattern, **kwargs)

    def drop_fuzzy(self, pattern: str, **kwargs: Any):
        drop = set(self.fuzzy_columns(pattern, **kwargs))
        keep = [c for c in self._cols() if c not in drop]
        return self._out_frame(self._lf().select(keep))

    def cols_like(self, pattern, *, case_sensitive=True, just_columns=True, cached=True, group=0):
        result = {}
        for f in self._cols():
            s = hyper_match(pattern, f, case_sensitive=case_sensitive, cached=cached, group=group)
            if s:
                result[f] = s
        return ensure_list(result.keys()) if just_columns else result

    def select_like(self, pattern, *, case_sensitive=True, cached=True):
        cols = self.cols_like(pattern, case_sensitive=case_sensitive, cached=cached, just_columns=True)
        return self._out_frame(self._lf().select(cols)) if cols else None

    def from_columns(
            self,
            from_cols: Union[Sequence[str], str],
            to_cols: Union[Sequence[str], str],
            on_missing: Any = None,
            *,
            on_unmatched: Any = None,
            max_unique: Optional[int] = None,
    ):
        out = _from_columns_apply(
            self._frame,
            from_cols,
            to_cols,
            on_missing=on_missing,
            on_unmatched=on_unmatched,
            max_unique=max_unique,
        )
        return self._out_frame(out)

    # ---------- conversions ----------
    def to_list(self, column: Optional[Union[ExprLike, Sequence[ExprLike]]] = None, *, unique: bool = False, drop_nulls: bool = False):

        if column is None:
            cols = self._cols()
            if len(cols) == 1:
                column = cols[0]
            else:
                raise ValueError("Missing column")

        cols = ensure_list(column)
        lf = self._lf()
        if len(cols) == 1:
            s = lf.select(_to_expr(cols[0]).alias("__tmp__")).collect().to_series(0)
            if drop_nulls:
                s = s.drop_nulls()
            if unique:
                s = s.unique()
            return s.to_list()
        out = []
        for c in cols:
            s = lf.select(_to_expr(c).alias("__tmp__")).collect().to_series(0)
            if drop_nulls:
                s = s.drop_nulls()
            if unique:
                s = s.unique()
            out.append(s.to_list())
        return out

    def to_series(self, column: Optional[ExprLike] = None, *, unique: bool = False, drop_nulls: bool = False):
        v = self.to_list(column, unique=unique, drop_nulls=drop_nulls)
        if not v: return pl.Series()
        dtype = self._schema().get(column) if isinstance(column, str) else None
        return pl.Series(v, dtype=dtype)

    def to_set(self, column: ExprLike, *, drop_nulls: bool = False) -> Set[Any]:
        return set(self.to_list(column, unique=True, drop_nulls=drop_nulls))

    def ul(self, column: ExprLike, *, drop_nulls: bool = True) -> List[Any]:
        return self.to_list(column, unique=True, drop_nulls=drop_nulls)

    def to_dict(self, column: Union[ExprLike, Sequence[ExprLike]], *, unique: bool = False, drop_nulls: bool = False) -> Dict[Any, Any]:
        cols = ensure_list(column)
        out: Dict[Any, Any] = {}
        for i, c in enumerate(cols):
            k = c if isinstance(c, str) else f"expr_{i}"
            out[k] = self.to_list(c, unique=unique, drop_nulls=drop_nulls)
        return out

    def to_map(self, key: ExprLike, value: Union[ExprLike, Sequence[ExprLike]], *, drop_nulls: bool = False, drop_null_keys:bool = False) -> Dict[Any, Any]:
        key_expr = _to_expr(key).alias("__k__")
        lf = self._lf()

        if isinstance(value, (list, tuple)):
            vals = list(value)
            if not vals:
                keys = lf.select(key_expr).collect().to_series(0).to_list()
                keys = [tuple(k) if isinstance(k ,list) else k for k in keys]
                return {k: {} for k in keys}

            out_names: List[str] = []
            for i, v in enumerate(vals):
                out_names.append(v if isinstance(v, str) else f"expr_{i}")

            sel = [key_expr] + [_to_expr(v).alias(out_names[i]) for i, v in enumerate(vals)]
            df2 = lf.select(sel).collect()
            if drop_nulls:
                df2 = df2.filter(pl.all_horizontal([pl.col(n).is_not_null() for n in out_names]))
            if drop_null_keys:
                if not isinstance(drop_null_keys, bool):
                    df2 = df2.with_columns(pl.col('__k__').fill_null(drop_null_keys))
                else:
                    df2 = df2.filter(pl.col("__k__").is_not_null())
            out: Dict[Any, Dict[str, Any]] = {}
            for row in df2.iter_rows(named=False):
                k = tuple(row[0]) if isinstance(row[0], list) else row[0]
                out[k] = {out_names[i]: row[i + 1] for i in range(len(out_names))}
            return out

        df2 = lf.select([key_expr, _to_expr(value).alias("__v__")]).collect()
        if drop_nulls:
            df2 = df2.filter(pl.col("__v__").is_not_null())
        if not isinstance(drop_null_keys, bool):
            df2 = df2.with_columns(pl.col('__k__').fill_null(drop_null_keys))
        else:
            df2 = df2.filter(pl.col("__k__").is_not_null())

        k_list = df2["__k__"].to_list()
        k_list = [tuple(k) if isinstance(k ,list) else k for k in k_list]
        return dict(zip(k_list, df2["__v__"].to_list()))

    def to_map_threaded(self, key: ExprLike, value: Union[ExprLike, Sequence[ExprLike]], *, drop_nulls: bool = False) -> Dict[Any, Any]:
        return _submit_collect(self.to_map, key, value, drop_nulls=drop_nulls).result()

    async def to_map_async(self, key: ExprLike, value: Union[ExprLike, Sequence[ExprLike]], *, drop_nulls: bool = False) -> Dict[Any, Any]:
        return await _run_in_collect_executor(self.to_map, key, value, drop_nulls=drop_nulls)

    def to_arrow(self):
        if pa is None:
            raise ImportError("pyarrow is not installed")
        return self._df().to_arrow()

    def to_arrow_batches(self, *, batch_size: int = 65536):
        if pa is None:
            raise ImportError("pyarrow is not installed")
        return self.to_arrow().to_batches(max_chunksize=int(batch_size))

    def to_arrow_reader(self, *, batch_size: int = 65536):
        if pa is None:
            raise ImportError("pyarrow is not installed")
        batches = self.to_arrow_batches(batch_size=batch_size)
        return pa.RecordBatchReader.from_batches(self.to_arrow().schema, batches)  # type: ignore[union-attr]

    def to_pandas(self, *, use_pyarrow: bool = True):
        df = self._df()
        try:
            return df.to_pandas(use_pyarrow_extension_array=use_pyarrow)  # type: ignore[attr-defined]
        except TypeError:
            return df.to_pandas()

    def to_numpy(self, columns: Optional[Sequence[str]] = None):
        cols = ensure_list(columns) if columns is not None else self._cols()
        if not cols:
            return np.empty((0, 0))
        df = self._lf().select(cols).collect()
        return df.to_numpy()

    def to_records(self, columns: Optional[Sequence[str]] = None):
        cols = ensure_list(columns) if columns is not None else self._cols()
        return self._lf().select(cols).collect().to_dicts()

    def to_kdb_sym(self, column: Any, *, unique: bool = True, drop_nulls: bool = True, on_none: Any = None):
        cols = ensure_list(column)
        if len(cols) > 1:
            return {c: self.to_kdb_sym(c, unique=unique, drop_nulls=drop_nulls, on_none=on_none) for c in cols}
        vals = self.to_list(cols[0], unique=unique, drop_nulls=drop_nulls)
        return _kdb_sym_from_list(vals, on_none=on_none)

    def to_kdb_str(self, column: Any, *, unique: bool = True, drop_nulls: bool = True, on_none: Any = None):
        cols = ensure_list(column)
        if len(cols) > 1:
            return {c: self.to_kdb_str(c, unique=unique, drop_nulls=drop_nulls, on_none=on_none) for c in cols}
        vals = self.to_list(cols[0], unique=unique, drop_nulls=drop_nulls)
        return _kdb_str_from_list(vals, on_none=on_none)

    def to_kdb_sym_str(self, column: Any, *, unique: bool = True, drop_nulls: bool = True, on_none: Any = None):
        cols = ensure_list(column)
        if len(cols) > 1:
            return {c: self.to_kdb_sym_str(c, unique=unique, drop_nulls=drop_nulls, on_none=on_none) for c in cols}
        vals = self.to_list(cols[0], unique=unique, drop_nulls=drop_nulls)
        return _kdb_sym_str_from_list(vals, on_none=on_none)

    # ---------- stats / transforms ----------
    def weighted_mean(self, value_col: ExprLike, weight_col: ExprLike, *, coerce_numeric: bool = True, cast_strict: bool = False):
        df = self._lf().select(HyperExprFactory.weighted_mean(value_col, weight_col, coerce_numeric=coerce_numeric, cast_strict=cast_strict).alias("__wmean__")).collect()
        return df["__wmean__"][0] if df.height else None

    def weighted_std(self, value_col: ExprLike, weight_col: ExprLike, *, coerce_numeric: bool = True, cast_strict: bool = False):
        df = self._lf().select(HyperExprFactory.weighted_std(value_col, weight_col, coerce_numeric=coerce_numeric, cast_strict=cast_strict).alias("__wstd__")).collect()
        return df["__wstd__"][0] if df.height else None

    def weighted_median(self, value_col: str, weight_col: str, *, group_by: Optional[Sequence[str]] = None, drop_nulls: bool = True):
        out = _weighted_median_lazy(self._frame, value_col, weight_col, group_by=group_by, drop_nulls=drop_nulls).collect()
        if group_by:
            return out
        return out["weighted_median"][0] if out.height else None

    def weighted_mode_vertical(self, value_col: str, weight_col: str, *, drop_nulls: bool = True, include_score: bool = True, as_scalar: bool = True, output_name: str = "weighted_mode"):
        df = _weighted_mode_vertical_lazy(self._frame, value_col, weight_col, drop_nulls=drop_nulls, output_name=output_name, include_score=include_score).collect()
        if not as_scalar:
            return df if self._is_df_mode() else df.lazy()
        if df.height == 0:
            return None if not include_score else (None, None)
        v = df[output_name][0]
        return (v, df[output_name + "_score"][0]) if include_score else v

    def winsorize_by_group(self, columns: Sequence[str], *, group_by: Sequence[str], lower_quantile: float = 0.01, upper_quantile: float = 0.99, coerce_numeric: bool = True, cast_strict: bool = False, mode="fill"):
        keys = ensure_list(group_by)
        exprs: List[pl.Expr] = []
        for c in columns:
            x = _ensure_float(c, coerce_numeric=coerce_numeric, cast_strict=cast_strict)
            ql = x.quantile(lower_quantile).over(keys)
            qh = x.quantile(upper_quantile).over(keys)
            if mode == 'fill':
                exprs.append(pl.when(x < ql).then(ql).when(x > qh).then(qh).otherwise(x).alias(c))
            elif mode == 'drop':
                exprs.append(~((x < ql) | (x > qh)))
            else:
                raise ValueError("Unknown winsorize mode: valid options are fill/drop")
        if mode == 'fill':
            return self._out_frame(self._lf().with_columns(exprs))
        elif mode == 'drop':
            return self._out_frame(self._lf().filter(exprs))
        else:
            raise ValueError

    def zscore_by_group(self, columns: Sequence[str], *, group_by: Sequence[str], coerce_numeric: bool = True, cast_strict: bool = False, suffix: str = "_z"):
        keys = [_to_expr(k) for k in group_by]
        exprs = [HyperExprFactory.zscore_by_group(c, keys, coerce_numeric=coerce_numeric, cast_strict=cast_strict).alias(f"{c}{suffix}") for c in columns]
        return self._out_frame(self._lf().with_columns(exprs))

    def rolling_weighted_mean(self, value_col: ExprLike, weight_col: ExprLike, *, window_size: int, min_periods: Optional[int] = None, center: bool = False, group_by: Optional[Sequence[str]] = None, output_name: str = "rolling_wmean"):
        expr = HyperExprFactory.rolling_weighted_mean(value_col, weight_col, window_size=window_size, min_periods=min_periods, center=center, group_by=group_by).alias(output_name)
        return self._out_frame(self._lf().with_columns(expr))

    # ---------- row-wise ----------
    def row_sum(self, columns: Sequence[ExprLike], *, output_name: str = "row_sum"):
        return self._out_frame(self._lf().with_columns(HyperExprFactory.row_sum(*columns).alias(output_name)))

    def row_mean(self, columns: Sequence[ExprLike], *, output_name: str = "row_mean"):
        return self._out_frame(self._lf().with_columns(HyperExprFactory.row_mean(*columns).alias(output_name)))

    def row_median(self, columns: Sequence[ExprLike], *, output_name: str = "row_median", null_behavior: str = "ignore"):
        return self._out_frame(self._lf().with_columns(HyperExprFactory.row_median(*columns, null_behavior=null_behavior).alias(output_name)))

    def row_mode(self, columns: Sequence[str], *, output_name: str = "row_mode", include_score: bool = False, drop_nulls: bool = True):
        lf = _row_mode_lazy(self._frame, columns, weights=None, drop_nulls=drop_nulls, output_name=output_name, include_score=include_score)
        return self._out_frame(lf)

    def row_weighted_mode(self, columns: Sequence[str], weights: Sequence[float], *, output_name: str = "row_wmode", include_score: bool = False, drop_nulls: bool = True):
        lf = _row_mode_lazy(self._frame, columns, weights=weights, drop_nulls=drop_nulls, output_name=output_name, include_score=include_score)
        return self._out_frame(lf)

    # ---------- fills ----------
    def fill_forward(self, columns: Optional[Sequence[str]] = None, *, null_like: Optional[Sequence[Any]] = None):
        return self._out_frame(_fill_frame(self._frame, mode="forward", columns=columns, null_like=null_like))  # type: ignore[arg-type]

    def fill_backward(self, columns: Optional[Sequence[str]] = None, *, null_like: Optional[Sequence[Any]] = None):
        return self._out_frame(_fill_frame(self._frame, mode="backward", columns=columns, null_like=null_like))  # type: ignore[arg-type]

    def fill_linear_interpolation(self, columns: Optional[Sequence[str]] = None, *, null_like: Optional[Sequence[Any]] = None):
        return self._out_frame(_fill_frame(self._frame, mode="linear", columns=columns, null_like=null_like))  # type: ignore[arg-type]

    def fill_geometric_interpolation(self, columns: Optional[Sequence[str]] = None, *, null_like: Optional[Sequence[Any]] = None, non_positive_to_null: bool = True):
        return self._out_frame(_fill_frame(self._frame, mode="geometric", columns=columns, null_like=null_like, non_positive_to_null=non_positive_to_null))  # type: ignore[arg-type]

    def fill_both(self, columns: Optional[Sequence[str]] = None, *, null_like: Optional[Sequence[Any]] = None, preference: str = "forwards"):
        return self._out_frame(_fill_frame(self._frame, mode="both", columns=columns, null_like=null_like, preference=preference))  # type: ignore[arg-type]

    # ---------- sampling ----------
    def sample(self, n: int, *, consecutive: bool = False, seed: Optional[int] = None):
        return self._out_frame(_sample_frame(self._frame, n=n, consecutive=consecutive, seed=seed))  # type: ignore[arg-type]

    # ---------- joins / concat / dedupe ----------
    def dedupe_columns(self, *, policy: str = "suffix", suffix_sep: str = "_", keep: str = "first"):
        df = self._df()
        return dedupe_columns(df, policy=policy, suffix_sep=suffix_sep, keep=keep)  # type: ignore[misc]

    def safe_join(self, right: FrameLike, *, on=None, left_on=None, right_on=None, how: str = "left", suffix: str = "_right", coalesce_dupes: bool = True, cast_on_supertype: bool = True, drop_right_on: bool = True, validate: Optional[str] = None):
        out = safe_join(self._frame, right, on=on, left_on=left_on, right_on=right_on, how=how, suffix=suffix, coalesce_dupes=coalesce_dupes, cast_on_supertype=cast_on_supertype, drop_right_on=drop_right_on, validate=validate)
        return self._out_frame(out)

    def safe_concat(self, others: Sequence[FrameLike], *, how: str = "vertical", rechunk: bool = False, supertype: bool = True, fill_missing: Any = None, drop_extra: bool = False):
        out = safe_concat([self._frame, *others], how=how, rechunk=rechunk, supertype=supertype, fill_missing=fill_missing, drop_extra=drop_extra)
        return self._out_frame(out)

    def asof_join_by_group(self, right: FrameLike, *, on: str, by: Union[str, Sequence[str], None] = None, strategy: str = "backward", tolerance: Optional[Any] = None, suffix: str = "_right", coalesce_dupes: bool = True, cast_on_supertype: bool = True):
        out = asof_join_by_group(self._frame, right, on=on, by=by, strategy=strategy, tolerance=tolerance, suffix=suffix, coalesce_dupes=coalesce_dupes, cast_on_supertype=cast_on_supertype)
        return self._out_frame(out)

    def grouped_last_before(self, *, group_by: Sequence[str], order_by: str, cutoff: Any, inclusive: bool = True):
        out = grouped_last_before(self._frame, group_by=group_by, order_by=order_by, cutoff=cutoff, inclusive=inclusive)
        return self._out_frame(out)

    def with_delta(self, *, date_col: str, lookback: Any, target_col: str, group_by: Optional[Sequence[str]] = None, output_name: Optional[str] = None, strategy: str = "backward"):
        out = with_delta(self._frame, date_col=date_col, lookback=lookback, target_col=target_col, group_by=group_by, output_name=output_name, strategy=strategy)
        return self._out_frame(out)

    # ---------- interpolate / pivot ----------
    def interpolate(self, x_col: str, y_col: str, x_input_col: Optional[str] = None, fit_df: Optional[FrameLike] = None, out_col: Optional[str] = None, **kwargs: Any):
        fit = self._frame if fit_df is None else fit_df
        nm = out_col or f"{y_col}_est"
        out = interpolate_column(fit_df=fit, apply_df=self._frame, x_col=x_col, y_col=y_col, x_input_col=x_input_col, out_col=nm, **kwargs)
        return self._out_frame(out)

    def pivot_ticker_fields(self, **kwargs: Any):
        out = pivot_ticker_fields(self._frame, **kwargs)  # type: ignore[arg-type]
        return self._out_frame(out)  # type: ignore[arg-type]

    # ---------- reports ----------
    def null_report(self, columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
        return null_report(self._frame, columns=columns)

    def distinct_report(self, columns: Optional[Sequence[str]] = None, *, top_k: Optional[int] = None) -> pl.DataFrame:
        return distinct_report(self._frame, columns=columns, top_k=top_k)

    # ---------- set ops ----------
    def column_difference(self, other: FrameLike) -> Tuple[set[str], set[str]]:
        return column_diff(self._frame, other)

    def column_overlap(self, other: FrameLike) -> set[str]:
        return column_overlap(self._frame, other)

    def schema_difference(self, other: FrameLike) -> Dict[str, Any]:
        return schema_difference(self._frame, other)

    def schema(self) -> Dict[str, pl.DataType]:
        return self._schema()

    # ---------- winsorization / outlier detection ----------
    def horizontal_winsor_mean(
        self,
        columns: Sequence[str],
        *,
        method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
        sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
        percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
        symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
        auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
        weight_col: Optional[str] = _OUTLIER_DEFAULT_WEIGHT_COL,
        nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
        zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
        filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
        contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
        pyod_params: Optional[Dict[str, Any]] = None,
        result_alias: str = "h_winsor_mean",
        n_threads: int = 2,
    ):
        method = normalize_method(method)
        out = horizontal_winsor(
            self._frame, ensure_list(columns),
            method=method, sensitivity=sensitivity, percentile_bounds=percentile_bounds,
            symmetric=symmetric, auto_rescale=auto_rescale, weight_col=weight_col,
            nulls_as_zero=nulls_as_zero, zeros_as_null=zeros_as_null,
            filter_nulls=filter_nulls, contamination=contamination, pyod_params=pyod_params,
            result_alias=result_alias, n_threads=n_threads,
        )
        return self._out_frame(out)

    def vertical_winsor_mean(
        self,
        columns: Sequence[str],
        *,
        method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
        sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
        percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
        symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
        auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
        nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
        zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
        filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
        contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
        pyod_params: Optional[Dict[str, Any]] = None,
        result_suffix: str = "_v_winsor",
        n_threads: int = 2,
        as_item: bool = False
    ):
        method = normalize_method(method)
        out = vertical_winsor(
            self._frame, ensure_list(columns),
            method=method, sensitivity=sensitivity, percentile_bounds=percentile_bounds,
            symmetric=symmetric, auto_rescale=auto_rescale,
            nulls_as_zero=nulls_as_zero, zeros_as_null=zeros_as_null,
            filter_nulls=filter_nulls, contamination=contamination, pyod_params=pyod_params,
            result_suffix=result_suffix, n_threads=n_threads,
        )
        return self._out_frame(out) if not as_item else out.hyper.to_list(columns)

    def vertical_winsor_w_neighbors(
        self,
        target_columns: Sequence[str],
        neighbor_columns: Sequence[str],
        *,
        method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
        sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
        percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
        symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
        auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
        nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
        zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
        filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
        contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
        pyod_params: Optional[Dict[str, Any]] = None,
        result_suffix: str = "_vn_winsor",
        n_threads: int = 2,
    ):
        method = normalize_method(method)
        out = vertical_winsor_w_neighbors(
            self._frame, ensure_list(target_columns), ensure_list(neighbor_columns),
            method=method, sensitivity=sensitivity, percentile_bounds=percentile_bounds,
            symmetric=symmetric, auto_rescale=auto_rescale,
            nulls_as_zero=nulls_as_zero, zeros_as_null=zeros_as_null,
            filter_nulls=filter_nulls, contamination=contamination, pyod_params=pyod_params,
            result_suffix=result_suffix, n_threads=n_threads,
        )
        return self._out_frame(out)

    def horizontal_winsor_mean_w_neighbors(
        self,
        target_columns: Sequence[str],
        neighbor_columns: Sequence[str],
        *,
        method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
        sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
        percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
        symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
        auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
        weight_col: Optional[str] = _OUTLIER_DEFAULT_WEIGHT_COL,
        nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
        zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
        filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
        contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
        pyod_params: Optional[Dict[str, Any]] = None,
        result_alias: str = "h_winsor_neighbor_mean",
        n_threads: int = 2,
    ):
        out = horizontal_winsor_w_neighbors(
            self._frame, ensure_list(target_columns), ensure_list(neighbor_columns),
            method=method, sensitivity=sensitivity, percentile_bounds=percentile_bounds,
            symmetric=symmetric, auto_rescale=auto_rescale, weight_col=weight_col,
            nulls_as_zero=nulls_as_zero, zeros_as_null=zeros_as_null,
            filter_nulls=filter_nulls, contamination=contamination, pyod_params=pyod_params,
            result_alias=result_alias, n_threads=n_threads,
        )
        return self._out_frame(out)

    def horizontal_outlier_mask(
        self,
        columns: Sequence[str],
        *,
        method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
        sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
        percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
        symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
        auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
        nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
        zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
        filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
        contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
        pyod_params: Optional[Dict[str, Any]] = None,
        result_suffix: str = "_h_outlier",
        n_threads: int = 2,
    ):
        method = normalize_method(method)
        out = horizontal_outlier_mask(
            self._frame, ensure_list(columns),
            method=method, sensitivity=sensitivity, percentile_bounds=percentile_bounds,
            symmetric=symmetric, auto_rescale=auto_rescale,
            nulls_as_zero=nulls_as_zero, zeros_as_null=zeros_as_null,
            filter_nulls=filter_nulls, contamination=contamination, pyod_params=pyod_params,
            result_suffix=result_suffix, n_threads=n_threads,
        )
        return self._out_frame(out)

    def vertical_outlier_mask(
        self,
        columns: Sequence[str],
        *,
        method: OutlierMethod = _OUTLIER_DEFAULT_METHOD,
        sensitivity: float = _OUTLIER_DEFAULT_SENSITIVITY,
        percentile_bounds: tuple = _OUTLIER_DEFAULT_PERCENTILE_BOUNDS,
        symmetric: bool = _OUTLIER_DEFAULT_SYMMETRIC,
        auto_rescale: bool = _OUTLIER_DEFAULT_AUTO_RESCALE,
        nulls_as_zero: bool = _OUTLIER_DEFAULT_NULLS_AS_ZERO,
        zeros_as_null: bool = _OUTLIER_DEFAULT_ZEROS_AS_NULL,
        filter_nulls: bool = _OUTLIER_DEFAULT_FILTER_NULLS,
        contamination: Optional[float] = _OUTLIER_DEFAULT_CONTAMINATION,
        pyod_params: Optional[Dict[str, Any]] = None,
        result_suffix: str = "_v_outlier",
        n_threads: int = 2,
    ):
        method = normalize_method(method)
        out = vertical_outlier_mask(
            self._frame, ensure_list(columns),
            method=method, sensitivity=sensitivity, percentile_bounds=percentile_bounds,
            symmetric=symmetric, auto_rescale=auto_rescale,
            nulls_as_zero=nulls_as_zero, zeros_as_null=zeros_as_null,
            filter_nulls=filter_nulls, contamination=contamination, pyod_params=pyod_params,
            result_suffix=result_suffix, n_threads=n_threads,
        )
        return self._out_frame(out)

    # ---------- clipboard ----------
    def to_clipboard(self) -> None:
        self._df(self.flatten()).write_clipboard()

    def write_clipboard(self) -> None:
        self.to_clipboard()

    def clip(self) -> None:
        self.to_clipboard()

    def utc_datetime(
            self,
            *,
            date_col: Optional[Union[ExprLike, Sequence[ExprLike]]] = None,
            time_col: Optional[Union[ExprLike, Sequence[ExprLike]]] = None,
            date_format: Optional[str] = None,
            time_format: Optional[str] = None,
            date_override: Optional[str, datetime.date]=None,
            default_to_now: bool = True,
            time_zone: str = "UTC",
            time_unit: str = "us",
            strict: bool = False,
            output_name: str = "utc_datetime",
    ) -> pl.DataFrame:
        date_cols = ensure_list(date_col) if date_col is not None else [None]
        time_cols = ensure_list(time_col) if time_col is not None else [None]

        n = max(len(date_cols), len(time_cols))
        date_cols = list(itertools.islice(itertools.cycle(date_cols), n))
        time_cols = list(itertools.islice(itertools.cycle(time_cols), n))

        exprs: List[pl.Expr] = []
        for i, (d, t) in enumerate(zip(date_cols, time_cols)):
            name = output_name if n == 1 else f"{output_name}_{i}"
            exprs.append(
                HyperExprFactory.utc_datetime_from_columns(
                    date_col=d,
                    time_col=t,
                    date_format=date_format,
                    time_format=time_format,
                    date_override=date_override,
                    default_to_now=default_to_now,
                    time_zone=time_zone,
                    time_unit=time_unit,
                    strict=strict,
                ).alias(name)
            )
        return self._frame.with_columns(exprs)


# -----------------------------------------------------------------------------
# Namespaces exposed to Polars (tiny adapters; full parity by construction)
# -----------------------------------------------------------------------------
@pl.api.register_dataframe_namespace("hyper")
class HyperDataFrameNamespace:
    def __init__(self, df: pl.DataFrame) -> None:
        self._core = _HyperCore(df, "df")

    def __getattr__(self, name: str):
        return getattr(self._core, name)


@pl.api.register_lazyframe_namespace("hyper")
class HyperLazyFrameNamespace:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._core = _HyperCore(lf, "lf")

    def __getattr__(self, name: str):
        return getattr(self._core, name)


pl.hyper = SimpleNamespace(
    get_supertype=get_supertype,
    safe_concat=safe_concat,
    safe_join=safe_join,
    asof_join_by_group=asof_join_by_group,
    grouped_last_before=grouped_last_before,
    with_delta=with_delta,
    interpolate_column=interpolate_column,
    pivot_ticker_fields=pivot_ticker_fields,
    null_report=null_report,
    distinct_report=distinct_report,
    horizontal_winsor=horizontal_winsor,
    vertical_winsor=vertical_winsor,
    vertical_winsor_w_neighbors=vertical_winsor_w_neighbors,
    horizontal_winsor_w_neighbors=horizontal_winsor_w_neighbors,
    horizontal_outlier_mask=horizontal_outlier_mask,
    vertical_outlier_mask=vertical_outlier_mask,
    OutlierMethod=OutlierMethod,
)  # type: ignore[attr-defined]


