

import base64, uuid, time
from typing import Any, List, Optional, Union, Iterable, Sequence, Tuple

import msgspec
import functools
from app.helpers.type_helpers import *
import polars as pl
from bidict import bidict
from pyroaring import BitMap as RoaringBitmap
from app.helpers.hash import hash_as_int, hash_any, encode_dict, decode_dict
import numpy as np

# ----------------------------- Tunables --------------------------------------

DEFAULT_CONFLICT_PRIORITY: int = 0      # Lower number == loses to higher numbers on same cell
DEFAULT_BROADCAST: bool = True          # Whether deltas are broadcast to clients by default
DEFAULT_BROADCAST_ALL: bool = False     # Whether deltas are broadcast to ALL clients by default
DEFAULT_PERSIST: bool = True            # Whether deltas are persisted by the background worker
DEFAULT_RELAY: bool = True              # Should we relay this log via passthru?
DEFAULT_LOG: bool = True                # Should we log this execution

INDEX_COL_NAME = "_row_idx"
INDEX_DTYPE = pl.UInt32

DELTA_MODES = ('add', 'update', 'remove')
TOAST_TYPES = {'info', 'success', 'error', 'loading', 'warning', 'dismiss'}

# ----------------------------- Casting -------------------------------------
def _ensure_bitmap(value):
    if value is None: return RoaringBitmap()
    if isinstance(value, RoaringBitmap): return RoaringBitmap(value)
    if isinstance(value, np.ndarray):
        arr = ensure_uint32_numpy(value)
        return RoaringBitmap(arr)
    if isinstance(value, pl.Series):
        return RoaringBitmap(_series_to_uint32_numpy(value))
    if isinstance(value, (list, tuple, set, range)):
        _UINT32_MAX = np.iinfo(np.uint32).max
        filtered = []
        for x in value:
            if x is None:
                continue
            ix = int(x)
            if 0 <= ix <= _UINT32_MAX:
                filtered.append(ix)
        arr = np.array(filtered, dtype=np.uint32)
        return RoaringBitmap(arr)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        bm = RoaringBitmap()
        for x in value:
            if x is None: continue
            ix = int(x)
            if ix >= 0:
                bm.add(ix)
        return bm
    return RoaringBitmap()

def _sort_dict(d):
    return {key: d[key] for key in sorted(d)}

def _normalize_columns(value, *, default=None, allow_none=True):
    return ensure_tuple(value, default=default, allow_none=allow_none)

def _series_to_uint32_numpy(series: pl.Series) -> np.ndarray:
    s = series.drop_nulls()
    try:
        s = s.filter((s >= 0) & (s <= (2**32 - 1)))
    except Exception:
        s = pl.Series([int(x) for x in s if x is not None and 0 <= int(x) <= (2 ** 32 - 1)])
    s = s.cast(INDEX_DTYPE, strict=False)
    if s.len() == 0: return np.empty(0, dtype=np.uint32)
    arr = s.to_numpy()
    return arr if arr.dtype == np.uint32 else arr.astype(np.uint32, copy=False)

def _rows_hint_to_bitmap(rows_hint):
    return _ensure_bitmap(rows_hint)

# ----------------------------- Utils -------------------------------------

def gather_rows_by_bitmap(df: pl.DataFrame, bm: Optional[RoaringBitmap]) -> pl.DataFrame:
    if not isinstance(df, pl.DataFrame) or df.is_empty() or not bm:
        return df if isinstance(df, pl.DataFrame) else pl.DataFrame()
    if INDEX_COL_NAME not in df.columns:
        return df  # caller should attach if they truly need index filtering
    if len(bm) == 0:
        return pl.DataFrame().with_columns([pl.Series(INDEX_COL_NAME, [], dtype=INDEX_DTYPE)])
    arr = pl.Series(INDEX_COL_NAME, np.array(bm, dtype=np.uint32), dtype=INDEX_DTYPE)
    return df.filter(pl.col(INDEX_COL_NAME).is_in(arr))

def _dedupe_preserve_order(seq):
    if not seq: return ()
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return tuple(out)

def _pk_first_columns(df, pk_cols):
    if not pk_cols: return tuple(df.columns)
    cols = df.columns
    if tuple(cols[:len(pk_cols)]) == tuple(pk_cols): return tuple(cols)
    lead = [c for c in pk_cols if c in cols]
    rest = [c for c in cols if c not in lead]
    return tuple(lead + rest)

# ----------------------------- Checks -------------------------------------

def _assert_columns_present(df, cols, where):
    if not cols: return
    cs = set(df.columns)
    missing = [c for c in cols if c not in cs]
    if missing: raise KeyError(f"{where}: missing columns {missing}")

def _assert_delta_mode(mode):
    m = (mode or "update").lower()
    if m not in DELTA_MODES:
        raise ValueError(f"Delta.mode must be 'update'|'add'|'remove', got {mode!r}")
    return m

def make_polars_enc_hook(*, orient: str = "columns", collect_lazy: bool = False):
    def enc_hook(obj):
        if isinstance(obj, pl.DataFrame):
            if orient == "columns": return obj.to_dict(as_series=False)
            if orient == "records": return obj.to_dicts()
            if orient == "rows":    return obj.rows()
            raise NotImplementedError
        if isinstance(obj, pl.Series):
            return obj.to_list()
        if isinstance(obj, pl.LazyFrame):
            if collect_lazy:
                return enc_hook(obj.collect())
            raise NotImplementedError
        raise NotImplementedError
    return enc_hook


def make_roaring_enc_hook(*, orient: str = "ints"):
    types = []
    try:
        from pyroaring import BitMap as _PB, FrozenBitMap as _PFB  # type: ignore
        types += [_PB, _PFB]
    except Exception:
        pass
    roaring_types = tuple(types) if types else ()

    def _to_bytes(o):
        if hasattr(o, "serialize"): return o.serialize()
        if hasattr(o, "to_bytes"):  return o.to_bytes()
        if hasattr(o, "tobytes"):   return o.tobytes()
        if hasattr(o, "__bytes__"): return bytes(o)
        return None

    def enc_hook(obj):
        if roaring_types and isinstance(obj, roaring_types):
            if orient == "bytes":
                b = _to_bytes(obj)
                if b is not None: return b
            # default / fallback: list of ints (JSON-friendly, deterministic)
            return list(obj)
        raise NotImplementedError
    return enc_hook


def make_combined_enc_hook(
        *,
        polars_orient: str = "columns",
        roaring_orient: str = "ints",
        collect_lazy: bool = True,
):
    polars_hook = None
    try:
        polars_hook = make_polars_enc_hook(orient=polars_orient, collect_lazy=collect_lazy)
    except Exception:
        pass

    roaring_hook = None
    try:
        roaring_hook = make_roaring_enc_hook(orient=roaring_orient)
    except Exception:
        pass

    def enc_hook(obj):
        if polars_hook is not None:
            try: return polars_hook(obj)
            except NotImplementedError: pass
        if roaring_hook is not None:
            try: return roaring_hook(obj)
            except NotImplementedError: pass
        raise NotImplementedError
    return enc_hook

def make_polars_dec_hook(*, lazy_to_df: bool = False):
    DF, S = pl.DataFrame, pl.Series
    LF = getattr(pl, "LazyFrame", None)

    def dec_hook(typ, value):
        if typ is DF: return DF(value)
        if typ is S: return S(value)
        if LF is not None and typ is LF:
            return DF(value) if lazy_to_df else DF(value).lazy()
        raise NotImplementedError
    return dec_hook


def make_roaring_dec_hook(*, orient: str = "ints"):
    try:
        from pyroaring import BitMap as PB, FrozenBitMap as PFB  # type: ignore
        roaring_types = (PB, PFB)
    except Exception:
        from pyroaring import BitMap as PB  # type: ignore
        roaring_types = (PB,)

    def _as_bytes(x):
        if isinstance(x, (bytes, bytearray, memoryview)): return bytes(x)
        if isinstance(x, str):
            try: return base64.b64decode(x, validate=True)
            except Exception: return None
        return None

    def _from_ints(typ, it):
        try: return typ(it)
        except Exception: pass
        try:
            obj = typ(); obj.update(it); return obj
        except Exception: pass
        raise TypeError("Cannot construct Roaring from ints")

    def _from_bytes(typ, b):
        for name in ("deserialize", "frombytes", "from_bytes"):
            m = getattr(typ, name, None)
            if callable(m):
                try: return m(b)
                except Exception: pass
        try: return typ(b)
        except Exception: pass
        raise TypeError("Cannot construct Roaring from bytes")

    def dec_hook(typ, value):
        if roaring_types and isinstance(typ, type) and issubclass(typ, roaring_types):
            if orient == "ints":  return _from_ints(typ, value)
            if orient == "bytes":
                b = _as_bytes(value)
                if b is None: raise TypeError("Expected base64/bytes for Roaring(bytes) orient")
                return _from_bytes(typ, b)
            raise NotImplementedError("Unknown orient for Roaring")
        raise NotImplementedError
    return dec_hook


def make_combined_dec_hook(
        *,
        polars_lazy_to_df: bool = False,
        roaring_orient: str = "ints",
):
    polars_hook = None
    try: polars_hook = make_polars_dec_hook(lazy_to_df=polars_lazy_to_df)
    except Exception: pass

    roaring_hook = None
    try: roaring_hook = make_roaring_dec_hook(orient=roaring_orient)
    except Exception: pass

    def dec_hook(typ, value):
        if polars_hook is not None:
            try: return polars_hook(typ, value)
            except NotImplementedError: pass
        if roaring_hook is not None:
            try: return roaring_hook(typ, value)
            except NotImplementedError: pass
        raise NotImplementedError
    return dec_hook


# ----------------------------- Shared -------------------------------------

ENC_HOOK = make_combined_enc_hook()
DEC_HOOK = make_combined_dec_hook()

_json_enc = msgspec.json.Encoder(enc_hook=ENC_HOOK)

def convert(obj, target):
    if isinstance(obj, target): return obj
    if hasattr(obj, 'to_dict'): obj = obj.to_dict()
    return msgspec.convert(obj, target, dec_hook=DEC_HOOK)

def tobuiltins(obj):
    if hasattr(obj, 'items'):
        it = obj.items(_all=False)
    else:
        it = msgspec.to_builtins(obj, enc_hook=ENC_HOOK)
    return msgspec.to_builtins(dict(it), enc_hook=ENC_HOOK)

def _to_public_builtins(obj):
    if isinstance(obj, msgspec.Struct):
        if hasattr(obj, "items"):
            return {k: _to_public_builtins(v) for k, v in obj.items(_all=False)}
        return _to_public_builtins(msgspec.to_builtins(obj, enc_hook=ENC_HOOK))
    if isinstance(obj, dict):
        return {k: _to_public_builtins(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [_to_public_builtins(v) for v in obj]
    return msgspec.to_builtins(obj, enc_hook=ENC_HOOK)

# ----------------------------- Registry ---------------------------------


# ----------------------------- Main -------------------------------------

class Struct(msgspec.Struct):
    def items(self, *, _all=False):
        fields = self.__struct_fields__
        get = self.__getattribute__
        if _all:
            return ((k, get(k)) for k in fields)
        return ((k, get(k)) for k in fields if not k.startswith('_'))

    def to_dict(self):
        return tobuiltins(self)

    def copy(self):
        return type(self)(**self.to_dict())

    def shallow_copy(self):
        return type(self)(**{k: getattr(self, k) for k in self.__struct_fields__})

    def replace(self, k, v, ascopy=True):
        p = self.copy() if ascopy else self
        if hasattr(p, k):
            setattr(p, k, v)
        return p

    def replace_many(self, d: dict, ascopy=True):
        p = self.copy() if ascopy else self
        for k, v in d.items():
            if hasattr(p, k): setattr(p, k, v)
        return p

    def len(self):
        t = 0
        fields = self.__struct_fields__
        defaults = self.__struct_defaults__
        get = self.__getattribute__
        # defaults tuple is right-aligned: it covers the last len(defaults) fields
        offset = len(fields) - len(defaults)
        for i in range(len(fields)):
            b = get(fields[i])
            di = i - offset
            if di < 0:
                # Field has no default — count it if set to non-None
                if b is not None:
                    t += 1
                continue
            a = defaults[di]
            if (a is None) and (b is None): continue
            elif (a is None) != (b is None): t += 1
            elif a != b: t += 1
        return t

    def __len__(self):
        return self.len()


class PayloadOptions(Struct, dict=True):
    log: bool = DEFAULT_LOG
    broadcast: bool = DEFAULT_BROADCAST
    persist: bool = DEFAULT_PERSIST
    broadcast_to_all: bool = DEFAULT_BROADCAST_ALL
    priority: int = DEFAULT_CONFLICT_PRIORITY
    relay: bool = DEFAULT_RELAY
    trigger_rules: bool = True
    silent: bool = False
    full_reset: bool = False

@functools.lru_cache(maxsize=128)
def _from_key_cached(key: Tuple[str, str]) -> 'RoomContext':
    grid_id, b64_string = key
    grid_filters = decode_dict(b64_string)
    return RoomContext(grid_id=grid_id, grid_filters=grid_filters)

def _from_key(key: Tuple[str, str]) -> 'RoomContext':
    """Return a shallow copy to prevent callers from mutating the cached value."""
    return _from_key_cached(key).shallow_copy()

def _encode_filters(grid_filters) -> str:
    return encode_dict(grid_filters)

class RoomContext(Struct, omit_defaults=True, dict=True):
    room: Optional[str] = None
    grid_id: Optional[str] = None
    grid_filters: Optional[dict] = None
    primary_keys: Optional[Sequence] = None
    columns: Optional[List] = None
    matched_pattern: Optional[str] = None
    full_reset: Optional[bool] = False

    def __post_init__(self):
        if self.room is not None:
            self.room = ensure_str(self.room).upper()
        self.grid_filters = _sort_dict(self.grid_filters or {})

    @property
    def key(self) -> Tuple[str, str]:
        return self.grid_id, _encode_filters(self.grid_filters)

    @staticmethod
    def from_key(key: Tuple[str, str]) -> 'RoomContext':
        return _from_key(key=key)

    @property
    def json_filters(self):
        return msgspec.json.encode(_sort_dict(self.grid_filters))

    @staticmethod
    def split_grid_context(grid_id, grid_filters=None):
        if hasattr(grid_id, 'grid_filters'):
            grid_filters = grid_id.grid_filters
            grid_id = grid_id.grid_id
        grid_filters = grid_filters or {}
        return grid_id, grid_filters

class User(Struct, omit_defaults=True, dict=True):
    fingerprint: Optional[str] = None
    sessionFingerprint: Optional[str] = None
    username: Optional[str] = None
    displayName: Optional[str] = None
    email: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    nickname: Optional[str] = None
    region: Optional[str] = None
    role: Optional[str] = None
    client_ip: Optional[str] = None
    client_port: Optional[int|str] = None
    impersonateMode: Optional[bool] = None

class ServerUser(User):
    fingerprint: str = "SERVER"
    sessionFingerprint: str = "SERVER"
    username: str = "SERVER"
    displayName:str = "SERVER"
    region:str="SERVER"

class Status(Struct, omit_defaults=True, dict=True):
    code: Optional[int] = None
    action: Optional[str] = None
    reason: Optional[str] = None
    ts: Optional[int] = None

    def __post_init__(self):
        self.ts = self.ts or int(time.time())
        self.code = 200 if self.code is None else self.code

class FeedbackData(Struct, omit_defaults=True, dict=True):
    feedbackType: Optional[str] = None
    feedbackText: Optional[str] = None

class ToastData(Struct, omit_defaults=True):
    toastType: Optional[str] = None
    title: Optional[str] = None
    message: Optional[str] = None
    persist: Optional[bool] = False
    permanent: Optional[bool] = False
    updateOnExist: Optional[bool] = True
    toastId: Optional[str] = None
    toastIcon: Optional[str] = None
    link: Optional[str] = None
    options: Optional[dict] = msgspec.field(default_factory=dict)

    def __post_init__(self):
        self.toastType = self.toastType or "info"
        self.toastId = self.toastId or Message._generate_trace_id()
        if self.toastType and self.toastType not in TOAST_TYPES:
            raise ValueError(f"Invalid toast type: {self.toastType!r}")
        if (self.message is None) and (not self.title is None):
            self.message = self.title
            self.title = None

@functools.lru_cache(maxsize=1)
def get_codec():
    from app.services.payload.columnar_codec import OptimizedColumnarCodec
    return OptimizedColumnarCodec()

def _convert_frame(x, allow_columnar=False):
    if x is None: return x
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)): return x
    if isinstance(x, dict):
        if '_format' in x:
            if allow_columnar: return x
            return get_codec().decode_to_polars(x)
        return pl.from_dict(x)
    if isinstance(x, list):
        if not x: return pl.DataFrame()
        if isinstance(x[0], dict):
            return pl.from_dicts(x)
        return pl.DataFrame(x)
    raise NotImplementedError

def _normalize_frame(df, pk):
    df = _convert_frame(df)
    if df is None:return
    if not pk:return df
    if isinstance(df, dict):
        return df
    s = df.schema if isinstance(df, pl.DataFrame) else df.collect_schema()
    m = [x for x in pk if x not in s]
    if m: raise KeyError(f"Missing primary keys: {m}, {s}")
    return df

def _compute_changed(df, cols):
    if cols: return ensure_tuple(cols)
    if df is None: return ensure_tuple(cols, allow_none=True)
    if isinstance(df, dict):
        if '_format' in df:
            fmt = df.get('_format')
            if fmt == 'columnar':
                return ensure_tuple(list((df.get('_columns') or {}).keys()))
            if fmt == 'single':
                return ensure_tuple((df.get('_data') or {}).keys())
            return ensure_tuple(())
        elif 'data' in df:
            raise NotImplementedError
            # data = df.get('data')
            # if 'payloads' in data:
            #     payloads = data.get('payloads')
            #     _adds = payloads.get('add', [])
            #     _upds = payloads.get('update', [])
            #     _rems = payloads.get('remove', [])

        return ensure_tuple(())
    if isinstance(df, pl.DataFrame):
        return ensure_tuple(df.columns)
    raise NotImplementedError

def _normalize_changed(df, cols, pk_columns):
    cols = _compute_changed(df, cols)
    if not cols: return cols
    pk = ensure_set(pk_columns, default=set)
    return ensure_tuple([x for x in cols if x not in pk])

def _get_row_index(df):
    if df is None: return None
    s = RoaringBitmap()
    s.add_range(0, df.height)
    return s

def _attach_index(df):
    if df is None: return
    if INDEX_COL_NAME in df.columns:
        return df.with_columns(pl.col(INDEX_COL_NAME).cast(INDEX_DTYPE))
    return df.with_row_index(INDEX_COL_NAME).with_columns(pl.col(INDEX_COL_NAME).cast(INDEX_DTYPE))

def _normalize_row_hints(s):
    if s is None: return
    return RoaringBitmap(s)

def _hint_and_attach(df):
    if df is None: raise ValueError
    rh = _get_row_index(df)
    idf = _attach_index(df)
    return rh, idf

def _get_schema(obj):
    if isinstance(obj, pl.DataFrame): return obj.schema
    if isinstance(obj, pl.LazyFrame): return obj.collect_schema()
    if isinstance(obj, dict): return obj.keys()
    raise NotImplementedError

class RowIndex(Struct, dict=True, tag=True):
    source: Optional[pl.DataFrame] = None
    primary_keys: Optional[Iterable] = None
    row_index: Optional[RoaringBitmap] = None
    isource: Optional[pl.DataFrame] = None

    def __post_init__(self):
        if self.source is None: return
        src = _convert_frame(self.source)
        if isinstance(src, pl.LazyFrame):
            src = src.collect()
        self.source = src
        self.primary_keys = ensure_tuple(self.primary_keys)
        self.row_index, self.isource = _hint_and_attach(self.source)

    def attach_row_idx(self, other_frame: pl.DataFrame):
        if self.isource is None: return
        if not self.primary_keys: return

        keys = list(self.primary_keys)

        def _align_left(left):
            lsch = _get_schema(left)
            rsch = _get_schema(self.isource)
            casts = []
            for k in keys:
                ldt = lsch.get(k)
                rdt = rsch.get(k)
                if (ldt is not None) and (rdt is not None) and (ldt != rdt):
                    casts.append(pl.col(k).cast(rdt))
            if not casts: return left
            if isinstance(left, pl.LazyFrame):
                return left.with_columns(*casts)
            return left.with_columns(casts)

        if isinstance(other_frame, pl.LazyFrame):
            lf = other_frame
            if INDEX_COL_NAME in _get_schema(lf):
                return lf.with_columns(pl.col(INDEX_COL_NAME).cast(INDEX_DTYPE))
            aligned = _align_left(lf)
            return aligned.join(self.isource.lazy(), on=keys, how="left")

        if INDEX_COL_NAME in other_frame.columns:
            if other_frame.schema.get(INDEX_COL_NAME) != INDEX_DTYPE:
                other_frame = other_frame.with_columns(pl.col(INDEX_COL_NAME).cast(INDEX_DTYPE))
            return other_frame

        aligned = _align_left(other_frame)
        return aligned.join(self.isource, on=keys, how="left")

    def extract_row_bitmap(self, other_frame: pl.DataFrame):
        if isinstance(other_frame, pl.LazyFrame):
            lf = other_frame
            if INDEX_COL_NAME not in _get_schema(lf):
                if self.isource is None:
                    raise KeyError("RowIndex requires primary_keys and source; index attach failed")
                lf = lf.join(self.isource.lazy(), on=list(self.primary_keys or ()), how="left")
            idx_df = lf.select(pl.col(INDEX_COL_NAME)).collect()
            return _ensure_bitmap(idx_df[INDEX_COL_NAME])

        if INDEX_COL_NAME not in other_frame.columns:
            other_frame = self.attach_row_idx(other_frame)
            if other_frame is None:
                raise KeyError("RowIndex requires primary_keys and source; index attach failed")
        return _ensure_bitmap(other_frame[INDEX_COL_NAME])


class Delta(Struct, dict=True):
    frame: Optional[Any] = None
    pk_columns: Optional[Any] = None
    changed_columns: Optional[Any] = None
    row_hint: Optional[RoaringBitmap] = None
    _source_index: Optional['RowIndex'] = None
    _iframe: Optional[pl.DataFrame] = None
    mode: Optional[str] = "update"

    def __post_init__(self):
        self.mode = _assert_delta_mode(self.mode)
        self.pk_columns = ensure_tuple(self.pk_columns, allow_none=True)
        self.frame = _normalize_frame(self.frame, self.pk_columns)
        self.changed_columns = _normalize_changed(self.frame, self.changed_columns, self.pk_columns)
        si = self._source_index
        if isinstance(si, RowIndex) and (self.row_hint is None) and (not isinstance(self.frame, dict)):
            f = self.frame
            try:
                iframe = si.attach_row_idx(f)
                if iframe is not None:  # Add null check
                    self._iframe = iframe
                    self.row_hint = si.extract_row_bitmap(iframe)
            finally:
                # Never retain heavy frames on the object
                self._iframe = None

    def __repr__(self):
        return str({k: getattr(self, k) for k in self.__struct_fields__ if not k.startswith("_")})

    def __str__(self):
        return self.__repr__()

class Payloads(Struct, dict=True, omit_defaults=True):
    add: Optional[Any] = None
    update: Optional[Any] = None
    remove: Optional[Any] = None
    _pk_columns: Optional[Iterable] = None
    _source_index: Optional['RowIndex'] = None
    based_on: Optional[int] = 0
    action_seq: Optional[int] = 0

    @property
    def add_size(self): return len(self.add) if self.add else 0
    @property
    def update_size(self): return len(self.update) if self.update else 0
    @property
    def remove_size(self): return len(self.remove) if self.remove else 0
    @property
    def size(self):
        return self.add_size + self.update_size + self.remove_size

    def __post_init__(self):

        self.based_on = ensure_int(self.based_on)
        self.action_seq = ensure_int(self.action_seq)

        add_in = ensure_list(self.add, allow_none=False, default=list)
        upd_in = ensure_list(self.update, allow_none=False, default=list)
        rem_in = ensure_list(self.remove, allow_none=False, default=list)

        self.add = []
        self.update = []
        self.remove = []

        append_add = self.append_add
        append_update = self.append_update
        append_remove = self.append_remove

        for a in add_in:
            append_add(a)
        for u in upd_in:
            append_update(u)
        for r in rem_in:
            append_remove(r)

    def _append_to(self, v, where, what):
        if v is None: return self

        if isinstance(v, Delta):
            v.mode = what
            try:
                if self._source_index is not None: v._source_index = self._source_index
                if self._pk_columns is not None: v.pk_columns = self._pk_columns
            except Exception: pass
            where.append(v); return self

        if isinstance(v, dict):
            if self._source_index is not None: v.setdefault('_source_index', self._source_index)
            if self._pk_columns is not None: v.setdefault('pk_columns', self._pk_columns)
            v['mode'] = what
            where.append(convert(v, Delta)); return self

        if isinstance(v, (pl.DataFrame, pl.LazyFrame)):
            d = Delta(frame=v, _source_index=self._source_index, pk_columns=self._pk_columns, mode=what)
            where.append(d); return self

        if isinstance(v, (list, tuple)):
            for item in v:
                self._append_to(item, where, what=what)
            return self

        raise TypeError(f"Unsupported payload item type: {type(v)!r}")

    def append_add(self, v):
        return self._append_to(v, where=self.add, what="add")

    def append_update(self, v):
        return self._append_to(v, where=self.update, what="update")

    def append_remove(self, v):
        return self._append_to(v, where=self.remove, what="remove")

    def append_deltas(self, d: dict):
        for k,v in d.items():
            k = k.lower()
            if k == 'add': self.append_add(v)
            elif k == 'update': self.append_update(v)
            elif k == 'remove': self.append_remove(v)
            else: raise ValueError(f'Unsupported delta key: {k}')

    def rebind_master_index(self, master_index, pk_columns=None):
        if master_index is self._source_index:
            if pk_columns is None or ensure_tuple(pk_columns, allow_none=True) == self._pk_columns:
                return self

        self._source_index = master_index
        if pk_columns is not None:
            self._pk_columns = ensure_tuple(pk_columns, allow_none=True)

        mi = self._source_index
        default_pk = self._pk_columns

        Delta_cls = Delta
        _append_to = self._append_to

        def _rebuild(src_list, what="update"):
            if not src_list:
                return []
            out = []
            ap = out.append
            for item in src_list:
                if isinstance(item, Delta_cls):
                    pk = item.pk_columns or default_pk
                    d = Delta_cls(
                        frame=item.frame,
                        pk_columns=pk,
                        changed_columns=item.changed_columns,
                        _source_index=mi,
                        mode = item.mode,
                        row_hint=item.row_hint,
                    )
                    ap(d)
                else:
                    _append_to(item, out, what=what)
            return out

        self.add = _rebuild(self.add, what="add")
        self.update = _rebuild(self.update, what="update")
        self.remove = _rebuild(self.remove, what="remove")

        return self

class Data(Struct, omit_defaults=True, repr_omit_defaults=True):
    payloads: Optional[Any] = None
    toast: Optional[Any] = None
    feedback: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.toast, dict): self.toast = ToastData(**self.toast)
        if isinstance(self.payloads, dict): self.payloads = Payloads(**self.payloads)
        if isinstance(self.feedback, dict): self.feedback = FeedbackData(**self.feedback)

    def append_add(self, v: Union[list, dict, pl.DataFrame, Delta]):
        if v is None: return
        if self.payloads is None:
            self.payloads = Payloads(add=v)
            return self.payloads
        else:
            return self.payloads.append_add(v)

    def append_update(self, v: Union[list, dict, pl.DataFrame, Delta]):
        if v is None: return
        if self.payloads is None:
            self.payloads = Payloads(update=v)
            return self.payloads
        else:
            return self.payloads.append_update(v)

    def append_remove(self, v: Union[list, dict, pl.DataFrame, Delta]):
        if v is None: return
        if self.payloads is None:
            self.payloads = Payloads(remove=v)
            return self.payloads
        else:
            return self.payloads.append_remove(v)

class Message:

    CLASS_REGISTRY = bidict({
        0: "identify",
        1: "control",
        2: "sync",
        3: "publish",
        4: "subscribe",
        5: "unsubscribe",
        6: "update_filter",
        7: "error",
        8: "trash",
        9: "ack",
        10: "duplicate",
        11: "ping",
        12: "sync_setting",
        13: "feedback",
        14: "toast",
        15: 'refresh',
        16: 'push',
        17: 'upload',
        18: 'execute',
        19: 'disconnect',
        20: 'fetch_columns',
        21: 'fetch_schema',
        22: 'micro_publish',
        23: 'micro_subscribe',
        24: 'micro_unsubscribe',
        25: 'redistribute',
    })

    def __init__(self,
                 action_str:str="message",
                 action_int:int=-1,
                 status:Optional[dict]=None,
                 data:Optional[dict]=None,
                 context:Optional[dict]=None,
                 user:Optional[dict]=None,
                 options:Optional[dict]=None,
                 trace:Optional[str]=None,
                 **kwargs):

        self.action = action_str
        self.status = convert(status or {}, Status)
        self.data = convert(data or {}, Data)
        self.context = convert(context or {}, RoomContext)
        self.user = convert(user or {}, User)
        self.options = convert(options or {}, PayloadOptions)
        self.trace = ensure_str(trace, allow_none=False, default=Message._generate_trace_id)
        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self): return str(self.to_dict())

    @staticmethod
    def _generate_trace_id():
        return str(uuid.uuid4().int >> 64)

    @classmethod
    def get_str(cls, key, default=None, *, strict=True, error_on_none=True):
        if key is None:
            if error_on_none: raise KeyError
            return default
        if (not strict) and isinstance(key, str): return key
        if isinstance(key, str) and (key.lower() in Message.CLASS_REGISTRY.values()): return key.lower()
        action = Message.CLASS_REGISTRY.get(key, default)
        if (action is None) and error_on_none: raise KeyError
        return action

    @classmethod
    def get_int(cls, key, default=None, *, strict=True, error_on_none=True):
        if key is None:
            if error_on_none: raise KeyError
            return default
        if isinstance(key, float):
            if not key.is_integer():
                raise ValueError
            key = int(key)
        if (not strict) and isinstance(key, int): return key
        if isinstance(key, int) and (key in Message.CLASS_REGISTRY.keys()): return key
        if isinstance(key, str): key = key.lower()
        action = Message.CLASS_REGISTRY.inverse.get(key, default)
        if (action is None) and error_on_none: raise KeyError
        return action

    @classmethod
    def name(cls, key): return cls.get_str(key)

    @property
    def str(self): return self.get_str(self.action)

    @property
    def int(self): return self.get_int(self.action)

    @property
    def builder(self): return self.__class__

    @classmethod
    def from_dict(cls, d):
        try: return cls(**d)
        except Exception: return convert(d, cls)

    @staticmethod
    def build(key):
        action = Message.get_str(key)
        if action is None: raise ValueError(f"Unknown class name: {key}")
        cls = CLASS_BUILDERS.get(action)
        if cls is None: raise ValueError(f"Unknown builder: {action}")
        return cls

    @staticmethod
    def construct(key_or_dict):
        key = key_or_dict.get('action', None) if isinstance(key_or_dict, dict) else key_or_dict
        cls = Message.build(key)
        return cls() if isinstance(key_or_dict, str) else cls(**key_or_dict)

    def to_dict(self):
        public = {k: v for k, v in self.__dict__.items() if not str(k).startswith('_')}
        return _to_public_builtins(public)
    def to_json(self):
        return _json_enc.encode(self.to_dict())
    def copy(self): return self.from_dict(self.to_dict())
    def replace(self, **kwargs):
        state = self.to_dict()
        state.update(kwargs)
        return self.from_dict(state)
    def with_override(self, **kwargs): return self.replace(**kwargs)
    def is_class(self, cls): return isinstance(self, cls)

    @property
    def fields(self): return self.__dict__.keys()
    def keys(self): return self.fields

    def items(self):
        return list(self.__dict__.items())

    @property
    def type(self): return type(self)

    def append_add(self, v: Union[list, dict, pl.DataFrame, Delta]):
        if v is None: return
        if self.data is None:
            self.data = Data(payloads=Payloads(add=v))
            return self.data.payloads
        else:
            return self.data.append_add(v)

    def append_update(self, v: Union[list, dict, pl.DataFrame, Delta]):
        if v is None: return
        if self.data is None:
            self.data = Data(payloads=Payloads(update=v))
            return self.data.payloads
        else:
            return self.data.append_update(v)

    def append_remove(self, v: Union[list, dict, pl.DataFrame, Delta]):
        if v is None: return
        if self.data is None:
            self.data = Data(payloads=Payloads(remove=v))
            return self.data.payloads
        else:
            return self.data.append_remove(v)


def _any(x): return any(y is not None for y in x)

# toastType: str = None
# title: str = None
# message: str = None
# persist: bool = False
# permanent: bool = False
# updateOnExist: bool = True
# toastId: str = msgspec.field(default_factory=lambda: uuid.uuid4().hex)
# toastIcon: Optional[str] = None
# link: Optional[str] = None
# options: Optional[dict] = msgspec.field(default_factory=dict)

class BaseMessage(Message):
    def __init__(self, action_str, action_int, *,
                 status=None, code=None, status_reason=None, status_action=None,
                 user=None, fingerprint=None, session=None, username=None, displayName=None,
                 role=None, firstName=None, lastName=None, client_ip=None, client_port=None,
                 impersonateMode=None,
                 data=None, based_on=None, action_seq=None,
                 payloads=None, add=None, update=None, remove=None, pk_columns=None,
                 toast=None, toastType=None, toastTitle=None, toastMessage=None, toastId=None, toastOptions=None,
                 feedback=None, feedbackText=None,feedbackType=None,
                 context=None, grid_id=None, room=None, grid_filters=None,
                 trace=None,
                 **kwargs):
        if (status is None) and _any([code, status_reason, status_action]):
            status = Status(code=code, reason=status_reason, action=status_action)
        if (user is None) and _any([fingerprint, session, username, displayName, role, firstName, lastName]):
            user = User(fingerprint=fingerprint, sessionFingerprint=session, username=username, displayName=displayName, role=role, firstName=firstName, lastName=lastName, client_ip=client_ip, client_port=client_port, impersonateMode=impersonateMode)
        if (payloads is None) and _any([add, update, remove]):
            payloads = Payloads(add=add, update=update, remove=remove, _pk_columns=pk_columns, based_on=based_on, action_seq=action_seq)
        if (toast is None) and _any([toastType, toastTitle, toastMessage]):
            toastOptions = toastOptions or {}
            toast = ToastData(toastType=toastType, title=toastTitle, message=toastMessage, toastId=toastId, options=toastOptions)
            p = toastOptions.get("persist")
            q = toastOptions.get("permanent")
            if p is not None and toast.persist is False: toast.persist = bool(p)
            if q is not None and toast.permanent is False: toast.permanent = bool(q)
        if (feedback is None) and _any([feedbackText]):
            feedback = FeedbackData(feedbackType=feedbackType, feedbackText=feedbackText)
        if (data is None) and _any([toast, payloads, feedback]):
            data = Data(payloads=payloads, toast=toast, feedback=feedback)
        if (context is None) and _any([grid_id, room, grid_filters]):
            context = RoomContext(grid_id=grid_id, room=room, grid_filters=grid_filters, primary_keys=pk_columns)
        super().__init__(action_str, action_int, status=status, user=user, data=data, context=context, trace=trace, **kwargs)

def _check_required(x):
    return not((x is None) or (hasattr(x, '__len__') and len(x) == 0))

class Ack(BaseMessage):
    def __init__(self, *, suppress_context=False, **kwargs):
        super().__init__("ack", 9, **kwargs)
        if (not suppress_context) and (not _check_required(self.context)): raise ValueError("Missing room context")
        if not _check_required(self.status): raise ValueError("Missing acknowledge")

class BroadcastMessage(BaseMessage):
    def __init__(self, action_str, action_int, *, suppress_context=False, **kwargs):
        super().__init__(action_str, action_int, **kwargs)
        self.suppress_context = suppress_context
        if (not suppress_context) and (not _check_required(self.context)): raise ValueError("Missing room context")

    def success(self, reason=None, code=200, *, as_ack=True, keep_data=False, keep_user=False, **kwargs):
        ack_data = self.data if keep_data else None
        ack_user= self.user if keep_user else None
        if as_ack: return Ack(code=code, status_reason=reason, status_action=self.str, context=self.context, data=ack_data, user=ack_user, suppress_context=self.suppress_context, trace=self.trace, **kwargs)
        self.status = Status(code=code, reason=reason, action=self.str)
        return self

    def error(self, reason=None, code=400, *, as_ack=True, keep_data=False, keep_user=False, **kwargs):
        ack_data = self.data if keep_data else None
        ack_user = self.user if keep_user else None
        reason = kwargs.pop('toastMessage', reason)
        title = kwargs.pop('toastTitle', 'Error')
        if as_ack: return Ack(code=code, status_reason=reason, data=ack_data, user=ack_user, status_action=self.str, context=self.context, suppress_context=self.suppress_context, trace=self.trace, toastType='error', toastMessage=reason, toastTitle=title, **kwargs)
        self.status = Status(code=code, reason=reason, action=self.str)
        return self

    def package(self):
        msg = self.success(as_ack=False, keep_data=True)
        return msg.to_dict()

    def as_emit(self, code, reason=None, *, as_ack=True, **kwargs):
        if as_ack: return Ack(code=code, status_reason=reason, status_action=self.str, context=self.context, suppress_context=self.suppress_context, trace=self.trace,**kwargs)
        self.status = Status(code=code, reason=reason, action=self.str)
        return self

    def toast(self, message, *, title=None, toastType='info', persist=False, permanent=False):
        tid = self.data.toast.toastId if self.data and self.data.toast else self.trace
        to = {"persist": persist, "permanent": permanent}
        return Toast(toastMessage=message, toastTitle=title, toastType=toastType, context=self.context, toastId=tid, toastOptions=to, suppress_context=self.suppress_context, persist=persist, permanent=permanent, trace=self.trace)

class Identify(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("identify", 0, **kwargs)
        if not _check_required(self.user): raise ValueError("Missing user information")

class Control(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("control", 1, **kwargs)

class Sync(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("sync", 2, **kwargs)

class Publish(BroadcastMessage):
    def __init__(self, delta_frame=None, delta_mode=None, **kwargs):

        # Enables simple construction from string
        _add = delta_frame if delta_mode == 'add' else None
        _update = delta_frame if delta_mode == 'update' else None
        _remove = delta_frame if delta_mode == 'remove' else None

        kwargs.setdefault('add', _add)
        kwargs.setdefault('update', _update)
        kwargs.setdefault('remove', _remove)

        super().__init__("publish", 3, **kwargs)
        if not _check_required(self.data): raise ValueError("Missing data field")
        if not _check_required(self.data.payloads): raise ValueError("Missing payloads")
        if not self.data.payloads.size: raise ValueError("Payloads are empty")

    def _replace_data(
            self,
            *,
            data=None,
            payloads=None,
            add=None,
            update=None,
            remove=None,
            pk_columns=None,
            based_on=None,
            action_seq=None,
            toast=None,
            feedback=None,
    ):
        # 1) If a complete Data() is provided (typed or dict), install it.
        if data is not None:
            if isinstance(data, Data):
                self.data = data
            else:
                # Accept dict-like; Data handles nested normalization.
                self.data = Data(**data)
            return self

        # 2) Normalize/construct Payloads.
        ploads = payloads
        if ploads is None:
            # Build from pieces if a full payloads object wasn't provided.
            ploads = Payloads(
                add=add,
                update=update,
                remove=remove,
                _pk_columns=pk_columns,
                based_on=based_on,
                action_seq=action_seq,
            )
        elif isinstance(ploads, dict):
            ploads = Payloads(**ploads)

        # 3) Preserve or override toast/feedback.
        existing_toast = getattr(getattr(self, "data", None), "toast", None)
        existing_feedback = getattr(getattr(self, "data", None), "feedback", None)

        if toast is not None:
            if isinstance(toast, dict):
                existing_toast = ToastData(**toast)
            else:
                existing_toast = toast  # assume already normalized

        if feedback is not None:
            if isinstance(feedback, dict):
                existing_feedback = FeedbackData(**feedback)
            else:
                existing_feedback = feedback  # assume already normalized

        # 4) Write back a normalized Data container.
        self.data = Data(payloads=ploads, toast=existing_toast, feedback=existing_feedback)
        return self

class Subscribe(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("subscribe", 4, **kwargs)

class Unsubscribe(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("unsubscribe", 5, **kwargs)

class UpdateFilter(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("update_filter", 6, **kwargs)
        if not self.context.grid_filters: raise ValueError("Missing grid_filters")

class Error(BroadcastMessage):
    def __init__(self, suppress_context=True, **kwargs):
        super().__init__("error", 7, suppress_context=suppress_context, **kwargs)

class Trash(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("trash", 8, **kwargs)

class Duplicate(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("duplicate", 10, **kwargs)

class Ping(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("ping", 11, **kwargs)

class SyncSettings(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("sync_setting", 12, **kwargs)

class Feedback(BroadcastMessage):
    def __init__(self, suppress_context=True, **kwargs):
        super().__init__("feedback", 13, suppress_context=suppress_context, **kwargs)

class Toast(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("toast", 14, **kwargs)
        if not _check_required(self.data): raise ValueError("Missing data field")
        if not _check_required(self.data.toast): raise ValueError("Missing toast field")

class Execute(BroadcastMessage):
    def __init__(self, **kwargs):
        super().__init__("execute", 18, **kwargs)
        if not _check_required(self.context): raise ValueError("Missing room context")

# =============================================================================
# Micro-Grid Messages
# =============================================================================

class MicroPublish:
    """
    Lightweight broadcast payload for micro-grid deltas.
    Not a full BroadcastMessage — micro-grids use a simpler JSON-based protocol
    instead of Arrow IPC columnar encoding.
    """
    __slots__ = (
        "action", "micro_name", "payloads",
        "based_on", "action_seq", "pk_columns",
        "trace", "snapshot",
    )

    def __init__(
            self,
            micro_name: str,
            *,
            payloads: Optional[dict] = None,
            based_on: int = 0,
            action_seq: int = 0,
            pk_columns: Optional[Tuple[str, ...]] = None,
            trace: Optional[str] = None,
            snapshot: Optional[list] = None,
    ):
        self.action = "micro_publish"
        self.micro_name = micro_name
        self.payloads = payloads or {}
        self.based_on = based_on
        self.action_seq = action_seq
        self.pk_columns = pk_columns or ()
        self.trace = trace
        self.snapshot = snapshot

    def to_dict(self) -> dict:
        d = {
            "action": self.action,
            "micro_name": self.micro_name,
            "data": {
                "payloads": self.payloads,
                "based_on": self.based_on,
                "action_seq": self.action_seq,
                "pk_columns": list(self.pk_columns) if self.pk_columns else [],
            },
            "context": {
                "room": f"MICRO.{self.micro_name.upper()}",
                "grid_id": f"micro_{self.micro_name}",
            },
        }
        if self.trace:
            d["trace"] = self.trace
        if self.snapshot is not None:
            d["data"]["snapshot"] = self.snapshot
        return d

    def package(self):
        return self.to_dict()


CLASS_BUILDERS = {
    "identify": Identify,
    "control": Control,
    "sync": Sync,
    "publish": Publish,
    "subscribe":Subscribe,
    "unsubscribe":Unsubscribe,
    "update_filter":UpdateFilter,
    "error":Error,
    "trash":Trash,
    "ack":Ack,
    "duplicate":Duplicate,
    "ping":Ping,
    "sync_setting":SyncSettings,
    "feedback":Feedback,
    "toast":Toast,
    "execute":Execute
}



