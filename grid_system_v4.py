



from __future__ import annotations

import asyncio
import bisect
import fnmatch
import heapq
import inspect
import threading
import time
import aiologic
import traceback
import weakref
import contextlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from async_lru import alru_cache

import polars as pl
from app.helpers.polars_hyper_plugin import *
from app.services.payload.payloadV4 import (
    Publish,
    PayloadOptions,
    Payloads,
    Delta,
    User,
    RoomContext,
    DEFAULT_CONFLICT_PRIORITY,
    MicroPublish,
)
from app.services.server.router import PubSubRouter
from app.helpers.hash import hash_any, hash_as_int
from app.logs.logging import log

try:
    from pyroaring import BitMap as RoaringBitmap  # type: ignore
except ImportError:  # pragma: no cover
    RoaringBitmap = None  # type: ignore

# =============================================================================
# Tunables
# =============================================================================

# Reaper
_GRID_REAPER_INTERVAL_S = 2.0   # How often do we check?
_GRID_ACTOR_IDLE_S = 300.0      # Hibernate if idle & still subscribed
_GRID_REAPER_MAX_ACTIONS = 8    #
_GRID_REMOVE_GRACE_S = 0.25     # debounce remove after last unsubscribe

# Persistence
_PERSIST_DEBOUNCE_S = 0.050            # batch window after first dirty signal
_PERSIST_MAX_ACTORS_PER_BATCH = 32     # max actors flushed per dequeue batch
_PERSIST_MAX_ROWS_PER_OP = 25_000      # chunk size per db op
_PERSIST_ERROR_BACKOFF_BASE_S = 0.75   # exponential backoff base
_PERSIST_ERROR_BACKOFF_MAX_S = 20.0    # max backoff cap

_JOURNAL_CAP = 50_000

# =============================================================================
# Globals
# =============================================================================

_now_ns = time.monotonic_ns

# =============================================================================
# Helpers
# =============================================================================

def _sort_dict(d):
    return {key: d[key] for key in sorted(d)}

async def _to_thread(fn, *args):
    from app.server import get_threads
    try:
        fut = get_threads().submit(fn, *args)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)
    except Exception as exc:
        await log.warning(f"Thread pool submit failed, falling back to sync execution: {exc}")
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args)
        return fn(*args)

def _subscriber_ref(x: Any):
    try:
        return weakref.ref(x)
    except TypeError:
        return x


def _subscriber_deref(x: Any):
    if isinstance(x, weakref.ReferenceType):
        return x()
    return x

def _purge_mvcc_store_heavy(store: "GridMVCCStore") -> None:
    try:
        store.base_cols.clear()
    except Exception:
        pass
    try:
        store.chains.clear()
    except Exception:
        pass
    try:
        store.col_last_write.clear()
    except Exception:
        pass
    try:
        store.alive_cache.clear()
    except Exception:
        pass
    try:
        store.alive_current = None
    except Exception:
        pass
    try:
        store.row_index.pk_to_row.clear()
    except Exception:
        pass
    try:
        store.row_index.row_to_pk.clear()
    except Exception:
        pass

def _chunk_ranges(n: int, chunk: int):
    n = int(n)
    chunk = int(chunk)
    if n <= 0 or chunk <= 0:
        return
    for i in range(0, n, chunk):
        j = i + chunk
        yield i, (j if j < n else n)

def _build_pk_filter_from_row_ids(store: "GridMVCCStore", row_ids: List[int]) -> Dict[str, List[Any]]:
    pk_cols = store.pk_cols
    n_pk = len(pk_cols)
    out: Dict[str, List[Any]] = {k: [] for k in pk_cols}
    pk_lists = [out[k] for k in pk_cols]
    row_to_pk = store.row_index.row_to_pk

    if n_pk == 1:
        pk0 = pk_lists[0]
        for r in row_ids:
            pk0.append(row_to_pk[r][0])
    elif n_pk == 2:
        pk0, pk1 = pk_lists[0], pk_lists[1]
        for r in row_ids:
            pk = row_to_pk[r]
            pk0.append(pk[0])
            pk1.append(pk[1])
    else:
        for r in row_ids:
            pk = row_to_pk[r]
            for j in range(n_pk):
                pk_lists[j].append(pk[j])
    return out


async def _db_upsert_sparse(db, grid_id: str, df: pl.DataFrame) -> None:
    if (df is None) or (not isinstance(df, (pl.DataFrame, pl.LazyFrame))) or df.hyper.is_empty():
        return
    _pf = await df.hyper.collect_async()
    # log.persist(f"Persisted {_pf.hyper.height()} rows", columns=_pf.hyper.fields, color='#4764f5')
    await db.upsert(grid_id, _pf, enable_new_columns=False)


async def _db_delete_by_pk_filter(db, grid_id: str, pk_filter: Dict[str, List[Any]]) -> None:
    if not pk_filter: return
    any_vals = False
    for v in pk_filter.values():
        if v:
            any_vals = True
            break
    if not any_vals:
        return
    await db.remove(grid_id, pk_filter)


# =============================================================================
# Async clocks
# =============================================================================

class AsyncGlobalClock:
    __slots__ = ("_v", "_lock")

    def __init__(self, start: int = 0) -> None:
        self._v = int(start)
        self._lock = asyncio.Lock()

    async def next(self) -> int:
        async with self._lock:
            self._v += 1
            return self._v

    def get(self) -> int:
        return self._v

_STAMP_PRIORITY_MAX = (1 << 47) - 1

def pack_stamp_key(based_on: int, priority: int, commit_seq: int) -> int:
    """
    Stamp key ordering (higher wins):
      1) based_on (higher wins)
      2) priority (higher wins, clamped to [0, 2^47))
      3) commit_seq (higher wins, last write wins)
    """
    priority = max(0, min(int(priority), _STAMP_PRIORITY_MAX))
    x = (int(based_on) << 96) | (priority << 48) | int(commit_seq)
    return x


# =============================================================================
# Catalogs
# =============================================================================

PKTuple = Tuple[Any, ...]


@dataclass(slots=True)
class ColumnCatalog:
    name_to_id: Dict[str, int]
    id_to_name: List[str]

    @classmethod
    def from_columns(cls, cols: Iterable[str]) -> "ColumnCatalog":
        name_to_id: Dict[str, int] = {}
        id_to_name: List[str] = []
        for c in cols:
            if c not in name_to_id:
                name_to_id[c] = len(id_to_name)
                id_to_name.append(c)
        return cls(name_to_id=name_to_id, id_to_name=id_to_name)

    def ensure(self, col_name: str) -> int:
        cid = self.name_to_id.get(col_name)
        if cid is not None:
            return cid
        cid = len(self.id_to_name)
        self.name_to_id[col_name] = cid
        self.id_to_name.append(col_name)
        return cid

    def get_id(self, col_name: str) -> Optional[int]:
        return self.name_to_id.get(col_name)

    def get_name(self, col_id: int) -> str:
        return self.id_to_name[col_id]


@dataclass(slots=True)
class RowIndex:
    pk_cols: Tuple[str, ...]
    pk_to_row: Dict[PKTuple, int]
    row_to_pk: List[PKTuple]

    @classmethod
    def build(cls, df: pl.DataFrame, pk_cols: Sequence[str]) -> "RowIndex":
        pk_cols_t = tuple(pk_cols)
        # .rows() returns all rows at once via a single C-level call
        all_pks = df.select(pk_cols_t).rows()
        n = len(all_pks)
        pk_to_row: Dict[PKTuple, int] = {}

        # Deduplicate: last row wins for duplicate PKs (H-07)
        # Also validate no NULL PK values (H-08)
        seen_dupe = False
        for i, pk in enumerate(all_pks):
            if any(v is None for v in pk):
                log.warning(f"Skipping row {i} with NULL PK value: {pk}")
                continue
            if pk in pk_to_row:
                seen_dupe = True
            pk_to_row[pk] = i

        if seen_dupe:
            log.warning(f"Duplicate PKs detected in initial DataFrame for {pk_cols_t}; last row wins")
            # Rebuild row_to_pk as dense list of only the winning entries
            inverse: Dict[int, PKTuple] = {row_id: pk for pk, row_id in pk_to_row.items()}
            winning_row_ids = sorted(inverse.keys())
            # Re-index rows to 0..len-1
            row_to_pk_deduped: List[PKTuple] = []
            pk_to_row_new: Dict[PKTuple, int] = {}
            for new_id, old_id in enumerate(winning_row_ids):
                pk = inverse[old_id]
                row_to_pk_deduped.append(pk)
                pk_to_row_new[pk] = new_id
            return cls(pk_cols=pk_cols_t, pk_to_row=pk_to_row_new, row_to_pk=row_to_pk_deduped)

        return cls(pk_cols=pk_cols_t, pk_to_row=pk_to_row, row_to_pk=all_pks)

    def resolve_or_insert(self, pk: PKTuple) -> Tuple[int, bool]:
        rid = self.pk_to_row.get(pk)
        if rid is not None:
            return rid, False
        rid = len(self.row_to_pk)
        self.pk_to_row[pk] = rid
        self.row_to_pk.append(pk)
        return rid, True

    def resolve_existing(self, pk: PKTuple) -> Optional[int]:
        return self.pk_to_row.get(pk)

    def pk_for_row(self, row_id: int) -> PKTuple:
        return self.row_to_pk[row_id]

    def row_count(self) -> int:
        return len(self.row_to_pk)


# =============================================================================
# Patches (dedup per cell inside one commit)
# =============================================================================

@dataclass(slots=True)
class CellPatch:
    # col_id -> (row_ids, values) aligned
    updates_by_col: Dict[int, Tuple[List[int], List[Any]]]
    # row IDs that were newly inserted (ADD) rather than updated
    inserted_rows: FrozenSet[int] = field(default_factory=frozenset)

    def is_empty(self) -> bool:
        return not self.updates_by_col

    def touched_cols(self) -> Set[int]:
        return set(self.updates_by_col.keys())

    def cell_count(self) -> int:
        return sum(len(rv[0]) for rv in self.updates_by_col.values())


class PatchBuilder:
    """
    Dedupes within a commit (last assignment wins per (row,col)).
    """
    __slots__ = ("_cols", "_inserted_rows")

    def __init__(self) -> None:
        self._cols: Dict[int, Dict[int, Any]] = {}
        self._inserted_rows: Set[int] = set()

    def mark_inserted(self, row_ids: List[int]) -> None:
        """Record row IDs that are newly inserted (ADD)."""
        self._inserted_rows.update(row_ids)

    def set_cell(self, col_id: int, row_id: int, value: Any) -> None:
        self._cols.setdefault(col_id, {})[row_id] = value

    def set_column(self, col_id: int, row_ids: List[int], values: List[Any]) -> None:
        """Bulk-set an entire column's worth of (row_id, value) pairs at once."""
        bucket = self._cols.get(col_id)
        if bucket is None:
            # Fast path: no prior entries for this column — build dict directly
            self._cols[col_id] = dict(zip(row_ids, values))
        else:
            # Merge: last write wins per row
            _update = bucket.update
            _update(zip(row_ids, values))

    def set_rows_single_value(self, col_id: int, row_ids: List[int], value: Any) -> None:
        """Bulk-set many rows in a single column to the same value."""
        bucket = self._cols.get(col_id)
        if bucket is None:
            self._cols[col_id] = dict.fromkeys(row_ids, value)
        else:
            for r in row_ids:
                bucket[r] = value

    def finalize(self) -> CellPatch:
        out: Dict[int, Tuple[List[int], List[Any]]] = {}
        for cid, m in self._cols.items():
            if not m:
                continue
            rids = sorted(m.keys())
            vals = [m[r] for r in rids]
            out[cid] = (rids, vals)
        return CellPatch(out, frozenset(self._inserted_rows))


# =============================================================================
# MVCC Version Chains
# =============================================================================

class VersionChain:
    __slots__ = ("commit_seqs", "stamp_keys", "values")

    def __init__(self) -> None:
        self.commit_seqs: List[int] = []
        self.stamp_keys: List[int] = []
        self.values: List[Any] = []

    def latest_stamp_key(self) -> int:
        return self.stamp_keys[-1] if self.stamp_keys else 0

    def latest_commit_seq(self) -> int:
        return self.commit_seqs[-1] if self.commit_seqs else 0

    def append(self, commit_seq: int, stamp_key: int, value: Any) -> None:
        self.commit_seqs.append(commit_seq)
        self.stamp_keys.append(stamp_key)
        self.values.append(value)

    def get_at_snapshot(self, snapshot_seq: int) -> Tuple[Any, bool]:
        cs = self.commit_seqs
        if not cs:
            return None, False
        i = bisect.bisect_right(cs, snapshot_seq) - 1
        if i < 0:
            return None, False
        return self.values[i], True

    def gc_floor(self, min_live_seq: int) -> "VersionChain":
        """
        Keep floor record (newest <= min_live_seq) plus everything newer.
        """
        cs = self.commit_seqs
        n = len(cs)
        if n <= 2:
            return self

        i = bisect.bisect_right(cs, min_live_seq) - 1
        if i <= 0:
            return self

        new = VersionChain()
        new.commit_seqs = cs[i:]
        new.stamp_keys = self.stamp_keys[i:]
        new.values = self.values[i:]
        return new

    def __len__(self) -> int:
        return len(self.commit_seqs)


# =============================================================================
# Snapshot tracker
# =============================================================================

_SNAPSHOT_TTL_NS = 120_000_000_000  # 120 seconds max snapshot hold time

class SnapshotTracker:
    """
    Tracks active baseline snapshots (commit_seq integers) with reference counts.
    Includes TTL-based force-expire to prevent leaked snapshots from pinning GC forever.
    """
    __slots__ = ("_lock", "_counts", "_acquired_at")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[int, int] = {}
        self._acquired_at: Dict[int, int] = {}  # snapshot_seq -> first acquire time (ns)

    def acquire(self, snapshot_seq: int) -> None:
        with self._lock:
            self._counts[snapshot_seq] = self._counts.get(snapshot_seq, 0) + 1
            if snapshot_seq not in self._acquired_at:
                self._acquired_at[snapshot_seq] = _now_ns()

    def release(self, snapshot_seq: int) -> None:
        with self._lock:
            cur = self._counts.get(snapshot_seq, 0)
            if cur <= 1:
                self._counts.pop(snapshot_seq, None)
                self._acquired_at.pop(snapshot_seq, None)
            else:
                self._counts[snapshot_seq] = cur - 1

    def min_active(self, default: int) -> int:
        with self._lock:
            self._expire_stale_locked()
            if not self._counts:
                return default
            return min(self._counts.keys())

    def any_active(self) -> bool:
        with self._lock:
            return bool(self._counts)

    def _expire_stale_locked(self) -> None:
        """Force-release snapshots held longer than TTL to prevent GC starvation."""
        if not self._acquired_at:
            return
        now = _now_ns()
        expired = [
            seq for seq, acquired in self._acquired_at.items()
            if (now - acquired) > _SNAPSHOT_TTL_NS
        ]
        for seq in expired:
            if seq in self._counts:
                log.warning(f"Force-expiring leaked snapshot seq={seq} held for >{_SNAPSHOT_TTL_NS // 1_000_000_000}s")
            self._counts.pop(seq, None)
            self._acquired_at.pop(seq, None)


# =============================================================================
# Dependencies
# =============================================================================

@dataclass()
class ColumnDep:
    grid_id: str
    snapshot_seq: int
    col_ids: Tuple[int, ...]
    room: str = field(init=False)

    def __post_init__(self):
        self.room = f"{self.grid_id.upper()}.*"

@dataclass()
class RowLocalDep:
    grid_id: str
    snapshot_seq: int
    col_ids: Tuple[int, ...]
    row_ids: Tuple[int, ...]
    room: str = field(init=False)

    def __post_init__(self):
        self.room = f"{self.grid_id.upper()}.*"

@dataclass()
class ReadDeps:
    col_deps: Tuple[ColumnDep, ...] = ()
    row_deps: Tuple[RowLocalDep, ...] = ()

# =============================================================================
# MVCC Store
# =============================================================================

_INTERNAL_ROW_ALIVE = "__row_alive"

def _make_alive_index(n: int):
    n = int(n)
    if n <= 0:
        return RoaringBitmap() if RoaringBitmap is not None else set()

    if RoaringBitmap is not None:
        b = RoaringBitmap()
        if hasattr(b, "add_range"):
            b.add_range(0, n)
        else:
            for r in range(n):
                b.add(r)
        return b

    return set(range(n))

def _alive_contains(alive_idx, r: int) -> bool:
    r = int(r)
    return r in alive_idx

def _alive_iter(alive_idx) -> Iterable[int]:
    if RoaringBitmap is not None and isinstance(alive_idx, RoaringBitmap):
        return alive_idx
    return iter(sorted(alive_idx))

def _alive_copy(alive_idx):
    if RoaringBitmap is not None and isinstance(alive_idx, RoaringBitmap):
        try:
            return RoaringBitmap(alive_idx)
        except Exception:
            b = RoaringBitmap()
            for r in alive_idx:
                b.add(int(r))
            return b
    return set(alive_idx)

def _alive_add(alive_idx, r: int) -> None:
    r = int(r)
    if hasattr(alive_idx, "add"):
        alive_idx.add(r)  # type: ignore[attr-defined]
        return
    try:
        alive_idx |= {r}
        return
    except Exception:
        pass
    log.error(f"Failed to add row {r} to alive index (type={type(alive_idx).__name__})")


def _alive_remove(alive_idx, r: int) -> None:
    r = int(r)
    if hasattr(alive_idx, "discard"):
        try:
            alive_idx.discard(r)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    if hasattr(alive_idx, "remove"):
        try:
            alive_idx.remove(r)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    try:
        alive_idx -= {r}
        return
    except Exception:
        pass
    log.error(f"Failed to remove row {r} from alive index (type={type(alive_idx).__name__})")

_ALIVE_CACHE_MAX = 64  # LRU cap for alive_cache

@dataclass(slots=True)
class GridMVCCStore:
    row_index: RowIndex
    cols: ColumnCatalog
    pk_cols: Tuple[str, ...]
    pk_col_ids: Tuple[int, ...]
    base_cols: Dict[int, List[Any]]
    chains: Dict[int, Dict[int, VersionChain]]              # col_id -> {row_id -> chain}
    col_last_write: Dict[int, int]                          # col_id -> last commit_seq written (any row)
    _commit_seq_committed: int = 0
    row_alive_col_id: int = -1
    schema: Optional[Any] = None
    _col_lock: threading.Lock = field(default_factory=threading.Lock)

    # Alive indices:
    # - alive_current: current snapshot only (commit_seq_committed)
    # - alive_cache: snapshot_seq -> alive index computed lazily (cleared on snapshot release)
    alive_current: Any = None
    alive_cache: Dict[int, Any] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df: pl.DataFrame, pk_cols: Sequence[str]) -> "GridMVCCStore":
        row_index = RowIndex.build(df, pk_cols)
        cols = ColumnCatalog.from_columns(df.columns)

        pk_cols_t = tuple(pk_cols)
        pk_col_ids: List[int] = []
        for c in pk_cols_t:
            cid = cols.get_id(c)
            if cid is None:
                raise ValueError(f"PK column missing from initial frame: {c}")
            pk_col_ids.append(cid)

        # Batch extract all columns at once
        df_columns = df.columns
        n_cols = len(df_columns)
        base_cols: Dict[int, List[Any]] = {}
        chains: Dict[int, Dict[int, VersionChain]] = {}
        col_last_write: Dict[int, int] = {}

        # Pre-allocate dicts for all columns + alive
        for i, c in enumerate(df_columns):
            base_cols[i] = df.get_column(c).to_list()
            chains[i] = {}
            col_last_write[i] = 0

        # internal row alive flag
        alive_cid = cols.ensure(_INTERNAL_ROW_ALIVE)
        n = row_index.row_count()
        base_cols[alive_cid] = [True] * n
        chains[alive_cid] = {}
        col_last_write[alive_cid] = 0

        store = cls(
            row_index=row_index,
            cols=cols,
            pk_cols=pk_cols_t,
            pk_col_ids=tuple(pk_col_ids),
            base_cols=base_cols,
            chains=chains,
            col_last_write=col_last_write,
            _commit_seq_committed=0,
            row_alive_col_id=alive_cid,
            schema = df.hyper.schema()
        )
        store.alive_current = _make_alive_index(n)
        return store

    def committed_seq(self) -> int:
        return self._commit_seq_committed

    def ensure_column(self, col_name: str) -> int:
        with self._col_lock:
            cid = self.cols.ensure(col_name)
            if cid not in self.base_cols:
                n = self.row_index.row_count()
                self.base_cols[cid] = [None] * n
                self.chains[cid] = {}
                self.col_last_write[cid] = 0
            return cid

    def _ensure_row_capacity(self, new_n: int) -> None:
        """
        Ensure all base columns have length >= new_n.

        IMPORTANT MVCC semantics:
        - Newly-created rows must NOT appear in snapshots prior to the commit that created them.
        - Therefore, we extend __row_alive base values with False (not True).
        - We also DO NOT mutate alive_current here; alive_current is updated only when
          __row_alive wins in apply_patch() at a commit.
        """
        new_n = int(new_n)
        if new_n <= 0:
            return

        # Canonical current capacity = length of __row_alive base column if present,
        # else max length across base columns.
        alive_base = self.base_cols.get(self.row_alive_col_id)
        cur_n = len(alive_base) if alive_base is not None else 0
        if cur_n == 0 and self.base_cols:
            cur_n = max(len(b) for b in self.base_cols.values())

        # If requested capacity is not larger, still pad any short columns up to cur_n.
        target = max(cur_n, new_n)

        # Extend each base column to 'target' length (idempotent if already long enough).
        for cid, base in self.base_cols.items():
            need = target - len(base)
            if need <= 0:
                continue
            if cid == self.row_alive_col_id:
                # New rows are historically NOT alive until their creating commit writes True.
                base.extend([False] * need)
            else:
                base.extend([None] * need)

    def _set_pk_base_for_row(self, row_id: int, pk: PKTuple) -> None:
        """
        Ensure capacity and populate PK base columns for the given row.
        """
        row_id = int(row_id)

        # Ensure the base arrays can hold row_id.
        self._ensure_row_capacity(row_id + 1)

        if len(pk) != len(self.pk_col_ids):
            raise ValueError(f"PK tuple size mismatch: got {len(pk)} expected {len(self.pk_col_ids)}")

        for j, cid in enumerate(self.pk_col_ids):
            base = self.base_cols[cid]
            if row_id >= len(base):
                # Extremely defensive: should not happen after _ensure_row_capacity
                base.extend([None] * (row_id + 1 - len(base)))
            base[row_id] = pk[j]

    def get_value(self, row_id: int, col_id: int, snapshot_seq: int) -> Any:
        colmap = self.chains.get(col_id)
        if colmap:
            chain = colmap.get(row_id)
            if chain:
                v, found = chain.get_at_snapshot(snapshot_seq)
                if found:
                    return v
        base = self.base_cols.get(col_id)
        if base is None or row_id >= len(base):
            return None
        return base[row_id]

    def cell_latest_commit_seq(self, row_id: int, col_id: int) -> int:
        colmap = self.chains.get(col_id)
        if not colmap:
            return 0
        chain = colmap.get(row_id)
        return 0 if chain is None else chain.latest_commit_seq()

    def _compute_alive_for_snapshot(self, snapshot_seq: int) -> Any:
        n = self.row_index.row_count()
        alive = RoaringBitmap() if RoaringBitmap is not None else set()
        if n == 0:
            return alive
        cid = self.row_alive_col_id
        base = self.base_cols.get(cid)
        colmap = self.chains.get(cid, {})

        if not colmap:
            # M-05: Fast path — batch alive row collection using list comprehension
            if base is not None:
                base_len = len(base)
                alive_rows = [r for r in range(min(n, base_len)) if base[r]]
                if RoaringBitmap is not None:
                    try:
                        return RoaringBitmap(alive_rows)
                    except Exception:
                        pass
                    for r in alive_rows:
                        alive.add(r)
                else:
                    alive = set(alive_rows)
            return alive

        # Slow path: check chains first, then fall back to base
        _colmap_get = colmap.get
        _alive_add_fn = alive.add
        base_len = len(base) if base is not None else 0
        for r in range(n):
            chain = _colmap_get(r)
            if chain:
                v, found = chain.get_at_snapshot(snapshot_seq)
                if found:
                    if v:
                        _alive_add_fn(r)
                    continue
            if base is not None and r < base_len and base[r]:
                _alive_add_fn(r)
        return alive

    def drop_alive_cache(self, snapshot_seq: int) -> None:
        self.alive_cache.pop(int(snapshot_seq), None)

    def _alive_index_for_snapshot(self, snapshot_seq: int) -> Any:
        """
        Fast path for running snapshot, cached path for older snapshots.
        LRU eviction when cache exceeds _ALIVE_CACHE_MAX entries.
        """
        snapshot_seq = int(snapshot_seq)
        if snapshot_seq == self._commit_seq_committed and self.alive_current is not None:
            return self.alive_current

        cached = self.alive_cache.get(snapshot_seq)
        if cached is not None:
            return cached

        alive = self._compute_alive_for_snapshot(snapshot_seq)
        # Evict oldest entries if cache exceeds LRU cap
        if len(self.alive_cache) >= _ALIVE_CACHE_MAX:
            to_evict = sorted(self.alive_cache.keys())[:len(self.alive_cache) - _ALIVE_CACHE_MAX + 1]
            for k in to_evict:
                self.alive_cache.pop(k, None)
        self.alive_cache[snapshot_seq] = alive
        return alive

    def apply_patch(self, patch: CellPatch, based_on: int, priority: int, commit_seq: int) -> Tuple[CellPatch, int]:
        """
        Apply sparse patch with per-cell conflict resolution.
        Returns (survivors_patch, dropped_cells).
        M-20: stages chain entries before applying, rolls back on failure.
        """
        if patch.is_empty():
            return CellPatch({}), 0

        stamp_key = pack_stamp_key(based_on, priority, commit_seq)
        survivors: Dict[int, Tuple[List[int], List[Any]]] = {}
        dropped = 0

        _base_cols = self.base_cols
        _chains = self.chains
        _col_last_write = self.col_last_write
        _row_count = self.row_index.row_count
        alive_cid = self.row_alive_col_id
        alive_current = self.alive_current

        # M-20: Stage all chain modifications, then apply atomically
        staged_appends: List[tuple] = []  # (chain, commit_seq, stamp_key, value)
        staged_new_chains: List[tuple] = []  # (colmap, row_id, chain)
        staged_alive_current = None

        try:
            for col_id, (row_ids, values) in patch.updates_by_col.items():
                if col_id not in _base_cols:
                    n = _row_count()
                    _base_cols[col_id] = [None] * n
                    _chains[col_id] = {}
                    _col_last_write[col_id] = 0

                colmap = _chains[col_id]
                _colmap_get = colmap.get
                n_entries = len(row_ids)
                win_rows: List[int] = []
                win_vals: List[Any] = []
                _wr_append = win_rows.append
                _wv_append = win_vals.append
                col_dropped = 0

                for i in range(n_entries):
                    r = row_ids[i]
                    chain = _colmap_get(r)
                    if chain is None:
                        chain = VersionChain()
                        staged_new_chains.append((colmap, r, chain))
                        chain.commit_seqs.append(commit_seq)
                        chain.stamp_keys.append(stamp_key)
                        chain.values.append(values[i])
                        _wr_append(r)
                        _wv_append(values[i])
                    elif stamp_key >= chain.stamp_keys[-1]:
                        staged_appends.append((chain, commit_seq, stamp_key, values[i]))
                        chain.commit_seqs.append(commit_seq)
                        chain.stamp_keys.append(stamp_key)
                        chain.values.append(values[i])
                        _wr_append(r)
                        _wv_append(values[i])
                    else:
                        col_dropped += 1

                dropped += col_dropped

                if win_rows:
                    survivors[col_id] = (win_rows, win_vals)
                    _col_last_write[col_id] = commit_seq

                    # Maintain alive_current for the running snapshot (copy-on-write)
                    if col_id == alive_cid and alive_current is not None:
                        new_alive = _alive_copy(alive_current)
                        for i in range(len(win_rows)):
                            r = win_rows[i]
                            if win_vals[i]:
                                _alive_add(new_alive, int(r))
                            else:
                                _alive_remove(new_alive, int(r))
                        staged_alive_current = new_alive
                        alive_current = new_alive  # update local ref

            # Commit staged new chains to their colmaps
            for colmap, row_id, chain in staged_new_chains:
                colmap[row_id] = chain

            # Commit alive_current
            if staged_alive_current is not None:
                self.alive_current = staged_alive_current

        except Exception:
            # Rollback: remove appended entries from existing chains
            for chain, cs, sk, val in staged_appends:
                if chain.commit_seqs and chain.commit_seqs[-1] == cs:
                    chain.commit_seqs.pop()
                    chain.stamp_keys.pop()
                    chain.values.pop()
            # Don't commit staged_new_chains or staged_alive_current
            raise

        # Propagate inserted_rows — intersect with rows that survived
        if patch.inserted_rows:
            surviving_rows: Set[int] = set()
            for rids, _ in survivors.values():
                surviving_rows.update(rids)
            kept_inserted = patch.inserted_rows & surviving_rows
        else:
            kept_inserted = frozenset()
        return CellPatch(survivors, kept_inserted), dropped

    def gc_versions(self, min_live_seq: int) -> int:
        """
        Compact version chains based on min_live_seq.
        Also folds single-entry chains back into base_cols and removes dead row storage
        when alive/total ratio drops below 50% (H-04).
        Returns number of chains replaced.
        """
        min_live_seq = int(min_live_seq)
        if min_live_seq <= 0:
            return 0

        replaced = 0
        folded = 0
        for cid, colmap in self.chains.items():
            if not colmap:
                continue
            base = self.base_cols.get(cid)
            to_del: List[int] = []
            for row_id, chain in list(colmap.items()):
                if len(chain) < 3:
                    # Fold single-entry chains back into base (H-03/H-04)
                    if len(chain) == 1 and chain.commit_seqs[0] <= min_live_seq:
                        if base is not None and row_id < len(base):
                            base[row_id] = chain.values[0]
                            to_del.append(row_id)
                            folded += 1
                    continue
                new_chain = chain.gc_floor(min_live_seq)
                if new_chain is not chain:
                    colmap[row_id] = new_chain
                    replaced += 1
                    # Fold if reduced to single entry below floor
                    if len(new_chain) == 1 and new_chain.commit_seqs[0] <= min_live_seq:
                        if base is not None and row_id < len(base):
                            base[row_id] = new_chain.values[0]
                            to_del.append(row_id)
                            folded += 1
            for row_id in to_del:
                colmap.pop(row_id, None)

        return replaced + folded

    def materialize(
            self,
            row_ids: Optional[List[int]],
            col_ids: List[int],
            snapshot_seq: int,
            *,
            include_removed: bool = False,
            schema_override: Optional[Dict] = None
    ) -> pl.DataFrame:
        snapshot_seq = int(snapshot_seq)

        if row_ids is None:
            if include_removed:
                row_ids = list(range(self.row_index.row_count()))
            else:
                alive = self._alive_index_for_snapshot(snapshot_seq)
                row_ids = list(_alive_iter(alive))
        else:
            if not include_removed:
                alive = self._alive_index_for_snapshot(snapshot_seq)
                row_ids = [r for r in row_ids if _alive_contains(alive, int(r))]

        n_rows = len(row_ids)
        if n_rows == 0:
            base_schema = schema_override or self.schema
            pk_cols_t = self.pk_cols
            req_non_pk = [cid for cid in col_ids if cid not in set(self.pk_col_ids) and cid != self.row_alive_col_id]
            empty_data = {pk: [] for pk in pk_cols_t}
            for cid in req_non_pk:
                empty_data[self.cols.get_name(cid)] = []
            my_schema = {col: base_schema.get(col, pl.String) for col in empty_data.keys()} if base_schema else None
            return pl.DataFrame(empty_data, schema=my_schema, strict=False) if my_schema else pl.DataFrame(empty_data)

        pk_cols_t = self.pk_cols
        pk_col_ids_set = set(self.pk_col_ids)
        n_pk = len(pk_cols_t)

        # Pre-allocate all output lists
        data: Dict[str, List[Any]] = {}
        pk_lists: List[List[Any]] = []
        for pk_name in pk_cols_t:
            lst = [None] * n_rows
            data[pk_name] = lst
            pk_lists.append(lst)

        req_non_pk = [cid for cid in col_ids if cid not in pk_col_ids_set and cid != self.row_alive_col_id]
        non_pk_out: List[List[Any]] = []
        for cid in req_non_pk:
            lst = [None] * n_rows
            data[self.cols.get_name(cid)] = lst
            non_pk_out.append(lst)

        # ---- Populate PKs ----
        # Optimized: avoid per-element tuple unpacking for single-column PKs
        row_to_pk = self.row_index.row_to_pk
        if n_pk == 1:
            pk0 = pk_lists[0]
            for idx in range(n_rows):
                pk0[idx] = row_to_pk[row_ids[idx]][0]
        elif n_pk == 2:
            pk0, pk1 = pk_lists[0], pk_lists[1]
            for idx in range(n_rows):
                pk = row_to_pk[row_ids[idx]]
                pk0[idx] = pk[0]
                pk1[idx] = pk[1]
        else:
            for idx in range(n_rows):
                pk = row_to_pk[row_ids[idx]]
                for j in range(n_pk):
                    pk_lists[j][idx] = pk[j]

        # ---- Populate non-PK columns ----
        # Shallow-copy dicts for thread safety: materialize runs in thread pool
        # while apply_patch may concurrently mutate self.chains / self.base_cols.
        # The inner lists are append-only so a shallow copy of the outer dicts
        # gives us a stable view of which columns/rows exist.
        chains_snap = dict(self.chains)
        base_cols_snap = dict(self.base_cols)

        for col_idx, cid in enumerate(req_non_pk):
            out = non_pk_out[col_idx]
            colmap = chains_snap.get(cid)
            base = base_cols_snap.get(cid)

            if not colmap:
                # Fast path: no version chains — read directly from base column
                if base is not None:
                    base_len = len(base)
                    for idx in range(n_rows):
                        rr = row_ids[idx]
                        if rr < base_len:
                            out[idx] = base[rr]
            else:
                # Slow path: check chains first, then fall back to base
                _colmap_get = colmap.get
                base_len = len(base) if base is not None else 0
                for idx in range(n_rows):
                    rr = row_ids[idx]
                    chain = _colmap_get(rr)
                    if chain:
                        v, found = chain.get_at_snapshot(snapshot_seq)
                        if found:
                            out[idx] = v
                            continue
                    if base is not None and rr < base_len:
                        out[idx] = base[rr]

        base_schema = schema_override or self.schema
        my_schema = {col: base_schema.get(col, pl.String) for col in data.keys()}
        return pl.DataFrame(data, schema=my_schema, strict=False)


# =============================================================================
# Registry
# =============================================================================

class ActorRegistry:
    def __init__(self) -> None:
        self._actors: Dict[str, "GridActor"] = {}
        self.mailbox = ActorMailbox()
        self._closed = False
        self.persistence: Optional["GridPersistenceManager"] = None
        self._reg_lock = aiologic.Lock()

    async def register(self, actor: "GridActor") -> None:
        async with self._reg_lock:
            key = actor.key
            if key in self._actors:
                raise ValueError(f"Actor already registered for room={actor.room}, key={key}")
            self._actors[key] = actor
            await self.mailbox.enroll(actor)

    def get(self, context:RoomContext) -> Optional["GridActor"]:
        key = GridActor.get_key(context)
        return self._actors.get(key)

    def get_by_key(self, key: str) -> Optional["GridActor"]:
        return self._actors.get(key)

    def require(self, context:RoomContext) -> "GridActor":
        a = self.get(context)
        if a is None:
            raise KeyError(f"Unknown room: {context.room}, {context.grid_filters}")
        return a

    def rooms(self) -> List[str]:
        return sorted(self._actors.keys())

    def iter_actors(self) -> List["GridActor"]:
        return list(self._actors.values())

    def find_actors(self, room: str, payload: Publish) -> List[Tuple["GridActor", str]]:
        return self.mailbox.mail(room, payload)

    async def remove(self, context: RoomContext, *, purge_store: bool = True) -> bool:
        async with self._reg_lock:
            actor = self.get(context)
            if actor is None: return False
            self._actors.pop(actor.key, None)
        try:
            await self.mailbox.disenroll(actor)
        except Exception as exc:
            await log.warning(f"Failed to disenroll actor {actor.key}: {exc}")
        try:
            await actor.shutdown(purge_store=purge_store)
        except Exception as exc:
            await log.warning(f"Failed to shutdown actor {actor.key}: {exc}")
        return True

    async def shutdown(self, *, purge_store: bool = True) -> None:
        if self._closed:
            return
        self._closed = True

        actors = list(self._actors.values())
        flush_ok = 0
        flush_fail = 0

        for a in actors:
            try:
                await self.mailbox.disenroll(a)
            except Exception as exc:
                await log.warning(f"Failed to disenroll actor {a.key} during shutdown: {exc}")

        for a in actors:
            try:
                await a.shutdown(purge_store=purge_store)
                flush_ok += 1
            except Exception as exc:
                flush_fail += 1
                await log.warning(f"Failed to shutdown actor {a.key}: {exc}")

        if flush_fail:
            await log.warning(f"Registry shutdown: {flush_ok} actors flushed OK, {flush_fail} failed")

        self._actors.clear()

        with contextlib.suppress(Exception):
            await self.mailbox.shutdown()

# =============================================================================
# Commit requests / results
# =============================================================================

@dataclass(slots=True)
class CommitRequest:
    ingress_id: int
    based_on: int
    priority: int
    source_task: str
    patch: CellPatch
    read_deps: Optional[ReadDeps]
    persist: bool
    done_fut: asyncio.Future["AppliedPatch"]


@dataclass(slots=True)
class AppliedPatch:
    ingress_id: int
    based_on: int
    priority: int
    source_task: str
    commit_seq: int
    survivors: CellPatch
    dropped_cells: int
    dropped_by_deps: int
    did_commit: bool


# =============================================================================
# GridActor
# =============================================================================

async def query_primary_keys(grid_id: str):
    return await GridActor._query_primary_keys(grid_id=grid_id)

@dataclass(slots=False)
class GridActor:
    context: RoomContext
    store: Optional[GridMVCCStore]
    registry: ActorRegistry

    room: str = field(init=False)
    grid_id: str = field(init=False)
    _key: Optional[str] = None

    snapshots: SnapshotTracker = field(default_factory=SnapshotTracker)
    _apply_q: asyncio.Queue[CommitRequest] = field(default_factory=lambda: asyncio.Queue(maxsize=1024))
    _applier_task: Optional[asyncio.Task] = None

    _gc_every_commits: int = 256
    _commits_since_gc: int = 0

    _last_touched_ns: int = time.monotonic_ns()

    _shutdown_started: bool = field(init=False)
    _closed: bool = field(init=False)

    _sub_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _subscriber_ids: Set[int] = field(default_factory=set, repr=False)
    _subscribers: Dict[int, Any] = field(default_factory=dict, repr=False)
    _lifecycle_gen: int = 0

    _hib_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _hibernating: bool = False
    _hibernated_at_ns: int = 0
    _hibernated_commit_seq: int = 0
    _hibernated_pk_cols: Tuple[str, ...] = field(default_factory=tuple)

    # persistence journal
    _persist_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _persist_dirty_cols_by_row: Dict[int, Set[int]] = field(default_factory=dict)
    _persist_deleted_rows: Set[int] = field(default_factory=set)

    # Track which ingress last wrote each column (for same-ingress dep filtering)
    _col_last_write_ingress: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.room = self.context.room
        self.grid_id = self.context.grid_id
        self.grid_filters = self.context.grid_filters
        self._shutdown_started = False
        self._closed = False

        if self.store is not None:
            self._hibernated_pk_cols = tuple(self.store.pk_cols)
        elif self.context.primary_keys:
            self._hibernated_pk_cols = tuple(self.context.primary_keys)

    def __hash__(self):
        return hash_as_int(self.key)

    @property
    def key(self):
        if self._key is None:
            self._key = self.get_key(self.context)
        return self._key

    @staticmethod
    def get_key(context: RoomContext):
        return hash_any(context.grid_id.upper(), _sort_dict(context.grid_filters))

    def touch(self) -> None:
        self._last_touched_ns = _now_ns()

    def last_touched_ns(self) -> int:
        return int(self._last_touched_ns)

    def subscriber_count(self) -> int:
        # Opportunistically prune dead weakrefs to avoid inflated counts
        dead = [
            tid for tid, ref in self._subscribers.items()
            if isinstance(ref, weakref.ReferenceType) and ref() is None
        ]
        if dead:
            for tid in dead:
                self._subscriber_ids.discard(tid)
                self._subscribers.pop(tid, None)
            self._lifecycle_gen += 1
        return len(self._subscriber_ids)

    def lifecycle_gen(self) -> int:
        return int(self._lifecycle_gen)

    def is_hibernating(self) -> bool:
        return bool(self._hibernating or self.store is None)

    def committed_seq_hint(self) -> int:
        s = self.store
        if s is not None:
            return int(getattr(s, "_commit_seq_committed", 0) or 0)
        return int(self._hibernated_commit_seq or 0)

    def can_reap_remove(self) -> bool:
        if self._closed or self._shutdown_started:
            return True
        if self.subscriber_count()!=0:
            return False
        if self.snapshots.any_active():
            return False
        try:
            if self._apply_q.qsize() > 0:
                return False
        except Exception:
            pass
        return True

    @staticmethod
    def get_subscriber_token(subscriber):
        from app.services.redux.connectionManager import _subscriber_token
        return _subscriber_token(subscriber)

    async def add_subscriber(self, subscriber: Any) -> bool:
        tid = self.get_subscriber_token(subscriber)
        async with self._sub_lock:
            if tid in self._subscriber_ids:
                self.touch()
                return False
            self._subscriber_ids.add(tid)
            self._subscribers[tid] = _subscriber_ref(subscriber)
            self._lifecycle_gen += 1
        self.touch()
        return True

    async def remove_subscriber(self, subscriber: Any) -> bool:
        tid = self.get_subscriber_token(subscriber)
        removed = False
        async with self._sub_lock:
            if tid in self._subscriber_ids:
                self._subscriber_ids.remove(tid)
                self._subscribers.pop(tid, None)
                self._lifecycle_gen += 1
                removed = True
        self.touch()
        return removed

    async def remove_all_subscribers(self) -> List[Any]:
        async with self._sub_lock:
            subs = []
            for ref in self._subscribers.values():
                o = _subscriber_deref(ref)
                if o is not None:
                    subs.append(o)
            self._subscriber_ids.clear()
            self._subscribers.clear()
            self._lifecycle_gen += 1
        self.touch()
        return subs

    async def list_subscribers(self) -> List[Any]:
        async with self._sub_lock:
            out: List[Any] = []
            dead: List[int] = []
            for tid, ref in self._subscribers.items():
                o = _subscriber_deref(ref)
                if o is None:
                    dead.append(tid)
                    continue
                out.append(o)
            if dead:
                for tid in dead:
                    self._subscriber_ids.discard(tid)
                    self._subscribers.pop(tid, None)
                self._lifecycle_gen += 1
            return out

    async def ensure_awake(self) -> None:

        if (self.store is not None) and (not self._hibernating):
            return

        async with self._hib_lock:
            if (self.store is not None) and (not self._hibernating):
                return

            # Never wake a closed actor
            if self._closed or self._shutdown_started:
                raise RuntimeError("Actor is closed")

            await log.reaper(f"Waking grid {self.room}", color="#333436")
            df, pk_cols = await asyncio.wait_for(
                self.build_from_db(self.context, return_pks=True),
                timeout=20.0
            )
            store = GridMVCCStore.from_frame(df, pk_cols=pk_cols)

            self.store = store
            self.context.primary_keys = list(pk_cols)
            self._hibernated_pk_cols = tuple(pk_cols)
            self._hibernating = False
            self._hibernated_at_ns = 0
            self._hibernated_commit_seq = 0

            self.touch()

    async def hibernate(self) -> bool:
        # Hibernation drops store memory but preserves subscribers + mailbox enrollment.
        if self._closed or self._shutdown_started:
            return False
        async with self._hib_lock:
            if self._hibernating or self.store is None:
                self._hibernating = True
                self.store = None
                # clear any persisted journal (nothing flushable without store)
                async with self._persist_lock:
                    self._persist_dirty_cols_by_row.clear()
                    self._persist_deleted_rows.clear()
                return True

            if self.snapshots.any_active():
                return False
            try:
                if self._apply_q.qsize() > 0:
                    return False
            except Exception:
                pass

            # Must persist before dropping store
            try:
                ok, still = await self.persist_flush()
                if not ok or still:
                    return False
            except Exception:
                return False

            store = self.store
            if store is None:
                self._hibernating = True
                return True

            self._hibernated_commit_seq = int(getattr(store, "_commit_seq_committed", 0) or 0)
            self._hibernated_pk_cols = tuple(store.pk_cols)
            self.context.primary_keys = list(self._hibernated_pk_cols)

            t = self._applier_task
            if (t is not None) and (not t.done()):
                t.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.gather(t, return_exceptions=True)
            self._applier_task = None

            # Re-check for new dirty data that may have arrived during persist_flush
            async with self._persist_lock:
                if self._persist_dirty_cols_by_row or self._persist_deleted_rows:
                    return False  # new data arrived, unsafe to hibernate

            self._hibernating = True
            self._hibernated_at_ns = _now_ns()

            self.store = None
            await _to_thread(_purge_mvcc_store_heavy, store)
            self.touch()
            return True

    async def force_rebuild_from_db(self, *, allow_when_busy: bool = False) -> bool:
        if self._closed or self._shutdown_started:
            return False

        async with self._hib_lock:
            if not allow_when_busy:
                if self.snapshots.any_active():
                    return False
                try:
                    if self._apply_q.qsize() > 0:
                        return False
                except Exception:
                    pass

            # Must persist before discarding store state
            try:
                ok, still = await self.persist_flush()
                if not ok or still:
                    return False
            except Exception:
                return False

            t = self._applier_task
            if (t is not None) and (not t.done()):
                t.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.gather(t, return_exceptions=True)
            self._applier_task = None

            df, pk_cols = await self.build_from_db(self.context, return_pks=True)
            self.store = GridMVCCStore.from_frame(df, pk_cols=pk_cols)
            self.context.primary_keys = list(pk_cols)
            self._hibernated_pk_cols = tuple(pk_cols)
            self._hibernating = False
            self._hibernated_at_ns = 0
            self._hibernated_commit_seq = 0
            self.touch()
            return True

    async def rebuild_from_db(self) -> bool:
        return await self.force_rebuild_from_db(allow_when_busy=False)

    @classmethod
    async def build_from_db(cls, context: RoomContext, return_pks: bool = True):
        from app.server import get_db
        grid_id = context.grid_id
        grid_filters = context.grid_filters or {}
        primary_keys = context.primary_keys or (await get_db().list_primary_keys(grid_id))
        frame = await get_db().select(grid_id, filters=grid_filters)
        return (frame, primary_keys) if return_pks else frame

    @staticmethod
    async def _query_primary_keys(grid_id:str):
        from app.server import get_db
        return tuple(await get_db().list_primary_keys(grid_id.lower()))

    def start(self) -> None:
        if (self._applier_task is None) or self._applier_task.done():
            self._applier_task = asyncio.create_task(self._applier_loop(), name=f"applier:{self.room}")

    def acquire_snapshot(self, snapshot_seq: int) -> None:
        self.snapshots.acquire(int(snapshot_seq))

    def release_snapshot(self, snapshot_seq: int) -> None:
        seq = int(snapshot_seq)
        self.snapshots.release(seq)
        s = self.store
        if s is not None:
            s.drop_alive_cache(seq)

    async def commit_patch(
            self,
            *,
            ingress_id: int,
            based_on: int,
            priority: int,
            source_task: str,
            patch: CellPatch,
            read_deps: Optional[ReadDeps],
            await_commit: bool = True,
            persist: bool = True) -> Optional[AppliedPatch]:

        if patch.is_empty():
            return None
        await self.ensure_awake()
        self.touch()
        self.start()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[AppliedPatch] = loop.create_future()
        await self._apply_q.put(
            CommitRequest(
                ingress_id=ingress_id,
                based_on=based_on,
                priority=priority,
                source_task=source_task,
                patch=patch,
                read_deps=read_deps,
                persist=bool(persist),
                done_fut=fut,
            )
        )
        return await fut if await_commit else None

    async def materialize_running_frame(
            self,
            cols: Optional[List[str]] = None,
            *,
            include_removed: bool = False,
    ) -> pl.DataFrame:
        await self.ensure_awake()
        self.touch()
        if self.store is None:
            raise RuntimeError("Store is None after ensure_awake")
        snapshot = self.store.committed_seq()
        if cols is None:
            names = list(self.store.cols.id_to_name)
            cols = [c for c in names if c not in self.store.pk_cols and c != _INTERNAL_ROW_ALIVE]
        col_ids = [self.store.ensure_column(c) for c in cols]
        from app.server import get_threads
        fut = get_threads().submit(self.store.materialize, None, col_ids, snapshot, include_removed=include_removed)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    async def _applier_loop(self) -> None:
        while True:
            req: Optional[CommitRequest] = None
            try:
                req = await self._apply_q.get()
                await self.ensure_awake()
                if self.store is None:
                    raise RuntimeError("Store is None after ensure_awake")

                filtered_patch, dropped_by_deps = await self._apply_dependency_filter(req, req.patch, req.read_deps)

                did_commit = False
                survivors = CellPatch({})
                dropped_cells = 0
                commit_seq = self.store._commit_seq_committed

                if not filtered_patch.is_empty():
                    candidate_seq = self.store._commit_seq_committed + 1
                    survivors, dropped_cells = self.store.apply_patch(filtered_patch, req.based_on, req.priority, candidate_seq)

                    if not survivors.is_empty():
                        # only advance commit_seq if state actually changed
                        self.store._commit_seq_committed = candidate_seq
                        commit_seq = candidate_seq
                        did_commit = True
                        # Track which ingress wrote each surviving column
                        for _sc_id in survivors.updates_by_col:
                            self._col_last_write_ingress[_sc_id] = req.ingress_id
                    else:
                        # no survivors => no state change => do not tick commit seq
                        commit_seq = self.store._commit_seq_committed
                        did_commit = False

                applied = AppliedPatch(
                    ingress_id=req.ingress_id,
                    based_on=req.based_on,
                    priority=req.priority,
                    source_task=req.source_task,
                    commit_seq=commit_seq,
                    survivors=survivors,
                    dropped_cells=dropped_cells,
                    dropped_by_deps=dropped_by_deps,
                    did_commit=did_commit,
                )

                if not req.done_fut.done():
                    req.done_fut.set_result(applied)

                # Persistence journal
                if did_commit and (not survivors.is_empty()) and bool(getattr(req, "persist", True)):
                    try:
                        await self._persist_record_patch(survivors)
                    except Exception as persist_exc:
                        await log.error(f"_persist_record_patch failed for {self.grid_id}: {persist_exc}")

                # MVCC GC only on real commits
                if did_commit:
                    self._commits_since_gc += 1
                    if self._commits_since_gc >= self._gc_every_commits:
                        self._commits_since_gc = 0
                        min_live = self.snapshots.min_active(self.store._commit_seq_committed)
                        self.store.gc_versions(min_live)

            except asyncio.CancelledError:
                # Resolve in-flight future before re-raising so callers don't hang
                if req is not None and not req.done_fut.done():
                    req.done_fut.set_result(AppliedPatch(
                        ingress_id=req.ingress_id, based_on=req.based_on,
                        priority=req.priority, source_task=req.source_task,
                        commit_seq=0, survivors=CellPatch({}),
                        dropped_cells=req.patch.cell_count(), dropped_by_deps=0,
                        did_commit=False,
                    ))
                raise
            except Exception as exc:
                await log.error(f"_applier_loop error for {self.grid_id}: {exc}\n{traceback.format_exc()}")
                # Resolve the future with an error so callers don't hang forever
                if req is not None and not req.done_fut.done():
                    try:
                        req.done_fut.set_exception(exc)
                    except Exception:
                        pass
                # Continue the loop — do not let one bad request kill the applier

    async def _apply_dependency_filter(self, req: CommitRequest, patch: CellPatch, deps: Optional[ReadDeps]) -> Tuple[CellPatch, int]:
        if deps is None or patch.is_empty():
            return patch, 0

        _pub = self.patch_to_payload_publish(
            based_on=req.based_on,
            action_seq=req.ingress_id,
            patch=patch
        )

        # 1) Column-global deps can reject entire patch
        for d in deps.col_deps:
            for src, _ in self.registry.find_actors(d.room, _pub):
                if src is None:
                    return CellPatch({}), patch.cell_count()
                if src.store is None:
                    continue  # source actor is hibernating, skip dep check
                for cid in d.col_ids:
                    last = src.store.col_last_write.get(cid, 0)
                    if last > d.snapshot_seq:
                        # Skip if the column was written by a sibling rule
                        # in the same ingress — not a true upstream conflict.
                        if src._col_last_write_ingress.get(cid) == req.ingress_id:
                            continue
                        return CellPatch({}), patch.cell_count()

        # 2) Row-local deps: invalidate only stale rows (only if dep.grid_id matches this grid)
        await self.ensure_awake()
        if self.store is None:
            raise RuntimeError("Store is None after ensure_awake in _apply_dependency_filter")

        invalid_rows: Set[int] = set()
        for d in deps.row_deps:
            if d.grid_id != self.context.grid_id:
                continue
            for r in d.row_ids:
                rr = int(r)
                for cid in d.col_ids:
                    if self.store.cell_latest_commit_seq(rr, cid) > d.snapshot_seq:
                        # Skip if the column was written by the same ingress
                        if self._col_last_write_ingress.get(cid) == req.ingress_id:
                            continue
                        invalid_rows.add(rr)
                        break

        if not invalid_rows:
            return patch, 0

        dropped = 0
        out: Dict[int, Tuple[List[int], List[Any]]] = {}
        for cid, (rids, vals) in patch.updates_by_col.items():
            nr: List[int] = []
            nv: List[Any] = []
            for r, v in zip(rids, vals):
                if int(r) in invalid_rows:
                    dropped += 1
                    continue
                nr.append(int(r))
                nv.append(v)
            if nr:
                out[cid] = (nr, nv)

        kept_inserted = patch.inserted_rows - invalid_rows if patch.inserted_rows else frozenset()
        return CellPatch(out, kept_inserted), dropped

    # -------------------------------------------------------------------------
    # Payload ingestion / emission helpers
    # -------------------------------------------------------------------------

    def _row_ids_from_df(self, df: pl.DataFrame, *, allow_insert: bool) -> Tuple[List[int], List[bool]]:
        if self.store is None:
            raise RuntimeError(f"(1) Store is None for actor {getattr(self, 'grid_id', '?')}")
        pk_cols = self.store.pk_cols
        if df.is_empty():
            return [], []

        # Extract all PK tuples at once via .rows() — one C-level call instead of iter_rows()
        pk_rows = df.select(pk_cols).rows()
        n = len(pk_rows)

        out_rids: List[int] = [0] * n  # pre-allocate
        out_ins: List[bool] = [False] * n

        if allow_insert:
            _resolve = self.store.row_index.resolve_or_insert
            _row_count = self.store.row_index.row_count
            _ensure = self.store._ensure_row_capacity
            _set_pk = self.store._set_pk_base_for_row
            any_inserted = False
            write_idx = 0
            # M-06: resolve all PKs first, then batch ensure_row_capacity once
            inserted_pks: List[tuple] = []
            for pk in pk_rows:
                rid, inserted = _resolve(pk)
                if inserted:
                    any_inserted = True
                    inserted_pks.append((rid, pk))
                out_rids[write_idx] = int(rid)
                out_ins[write_idx] = inserted
                write_idx += 1
            # Batch capacity expansion and PK setting after all inserts
            if any_inserted:
                _ensure(_row_count())
                for rid, pk in inserted_pks:
                    _set_pk(rid, pk)
            # write_idx == n always in insert mode (every row gets a rid)
        else:
            _resolve_existing = self.store.row_index.resolve_existing
            write_idx = 0
            for pk in pk_rows:
                rid = _resolve_existing(pk)
                if rid is not None:
                    out_rids[write_idx] = int(rid)
                    # out_ins[write_idx] already False
                    write_idx += 1
            # Trim if some rows were skipped
            if write_idx < n:
                out_rids = out_rids[:write_idx]
                out_ins = out_ins[:write_idx]

        return out_rids, out_ins

    def payloads_to_patch(self, payloads: Payloads) -> CellPatch:
        """
        Convert payloadV4.Payloads(add/update/remove) to a single deduped patch.
        Deterministic order: add -> update -> remove.
        Row deletion represented via internal __row_alive=False.

        Optimized: uses columnar bulk writes instead of per-cell set_cell.
        """
        if self.store is None:
            raise RuntimeError(f"(2) Store is None for actor {getattr(self, 'grid_id', '?')}")
        pb = PatchBuilder()
        alive_cid = self.store.row_alive_col_id
        pk_set = set(self.store.pk_cols)
        alive_idx = self.store.alive_current
        _alive_contains_fn = _alive_contains
        _set_column = pb.set_column
        _set_rows_single = pb.set_rows_single_value
        _ensure_column = self.store.ensure_column
        _has_alive = alive_idx is not None

        _mark_inserted = pb.mark_inserted

        def handle_add_update(d: Delta, is_add: bool) -> None:
            raw = d.frame
            if raw is None:
                return
            df = raw.collect() if isinstance(raw, pl.LazyFrame) else raw if isinstance(raw, pl.DataFrame) else None
            if df is None or df.is_empty():
                return

            row_ids, inserted_flags = self._row_ids_from_df(df, allow_insert=True)
            if not row_ids:
                return

            # Track genuinely new rows from ADD payloads
            if is_add:
                new_rows = [r for r, ins in zip(row_ids, inserted_flags) if ins]
                if new_rows:
                    _mark_inserted(new_rows)

            # Vectorized alive-True: only rows that need the transition
            if _has_alive:
                need_alive = [
                    r for r, ins in zip(row_ids, inserted_flags)
                    if ins or not _alive_contains_fn(alive_idx, r)
                ]
            else:
                # No alive index => all need it
                need_alive = row_ids
            if need_alive:
                _set_rows_single(alive_cid, need_alive, True)

            # Columnar bulk writes for data columns
            cols = d.changed_columns or tuple(c for c in df.columns if c not in pk_set)
            df_columns = df.columns
            for c in cols:
                if c in pk_set:
                    continue
                cid = _ensure_column(c)
                if c in df_columns:
                    values = df.get_column(c).to_list()
                else:
                    values = [None] * len(row_ids)
                _set_column(cid, row_ids, values)

        def handle_remove(d: Delta) -> None:
            raw = d.frame
            if raw is None:
                return
            df = raw.collect() if isinstance(raw, pl.LazyFrame) else raw if isinstance(raw, pl.DataFrame) else None
            if df is None or df.is_empty():
                return
            row_ids, _ = self._row_ids_from_df(df, allow_insert=False)
            if not row_ids:
                return

            # Vectorized alive-False: only rows currently alive
            if _has_alive:
                need_dead = [r for r in row_ids if _alive_contains_fn(alive_idx, r)]
            else:
                need_dead = row_ids
            if need_dead:
                _set_rows_single(alive_cid, need_dead, False)

        for d in (payloads.add or []):
            handle_add_update(d, True)
        for d in (payloads.update or []):
            handle_add_update(d, False)
        for d in (payloads.remove or []):
            handle_remove(d)

        return pb.finalize()

    def patch_to_df(self, patch: CellPatch) -> pl.DataFrame:
        if patch.is_empty():
            return pl.DataFrame()
        if self.store is None:
            raise RuntimeError(f"(3) Store is None for actor {getattr(self, 'grid_id', '?')}")

        alive_cid = self.store.row_alive_col_id

        # Union all touched rows (excluding alive column)
        rows_set: Set[int] = set()
        _rows_update = rows_set.update
        for cid, (rids, _) in patch.updates_by_col.items():
            if cid != alive_cid:
                _rows_update(rids)
        rows = sorted(rows_set)
        n_rows = len(rows)
        if n_rows == 0:
            return pl.DataFrame()

        pk_cols = self.store.pk_cols
        n_pk = len(pk_cols)

        # Pre-allocate all output
        data: Dict[str, List[Any]] = {}
        pk_lists: List[List[Any]] = []
        for pk_name in pk_cols:
            lst = [None] * n_rows
            data[pk_name] = lst
            pk_lists.append(lst)

        col_ids = sorted(cid for cid in patch.updates_by_col if cid != alive_cid)
        col_outs: List[List[Any]] = []
        for cid in col_ids:
            lst = [None] * n_rows
            data[self.store.cols.get_name(cid)] = lst
            col_outs.append(lst)

        # Build row_pos index
        row_pos = dict(zip(rows, range(n_rows)))

        # Populate PKs
        row_to_pk = self.store.row_index.row_to_pk
        if n_pk == 1:
            pk0 = pk_lists[0]
            for idx, r in enumerate(rows):
                pk0[idx] = row_to_pk[r][0]
        elif n_pk == 2:
            pk0, pk1 = pk_lists[0], pk_lists[1]
            for idx, r in enumerate(rows):
                pk = row_to_pk[r]
                pk0[idx] = pk[0]
                pk1[idx] = pk[1]
        else:
            for idx, r in enumerate(rows):
                pk = row_to_pk[r]
                for j in range(n_pk):
                    pk_lists[j][idx] = pk[j]

        # Populate data columns — vectorized per-column
        _row_pos_get = row_pos.__getitem__
        for col_idx, cid in enumerate(col_ids):
            rids, vals = patch.updates_by_col[cid]
            out = col_outs[col_idx]
            for i in range(len(rids)):
                out[_row_pos_get(rids[i])] = vals[i]

        return pl.DataFrame(data)

    def patch_to_payload_publish(
            self,
            *,
            based_on: int,
            action_seq: int,
            patch: CellPatch,
            options: Optional[PayloadOptions] = None,
            trace: Optional[str] = None,
    ) -> Optional[Publish]:
        """
        Build payloadV4.Publish from survivors patch.
        - remove rows => remove delta
        - other cells => update delta
        """
        if patch.is_empty():
            return None

        if self.store is None:
            raise RuntimeError(f"(4) Store is None for actor {getattr(self, 'grid_id', '?')}")

        pk_cols = list(self.store.pk_cols)
        options = options or PayloadOptions()

        alive_cid = self.store.row_alive_col_id

        # Removed rows — vectorized detection
        removed_rows: List[int] = []
        alive_update = patch.updates_by_col.get(alive_cid)
        if alive_update is not None:
            rids, vals = alive_update
            removed_rows = [rids[i] for i in range(len(rids)) if vals[i] is False]

        add_list: List[Delta] = []
        upd_list: List[Delta] = []
        rem_list: List[Delta] = []

        # remove delta frame: pk-only — vectorized PK extraction
        if removed_rows:
            removed_rows.sort()
            n_pk = len(pk_cols)
            row_to_pk = self.store.row_index.row_to_pk
            pk_data: Dict[str, List[Any]] = {}
            pk_lists: List[List[Any]] = []
            for pk_name in pk_cols:
                lst: List[Any] = []
                pk_data[pk_name] = lst
                pk_lists.append(lst)

            if n_pk == 1:
                pk0 = pk_lists[0]
                for r in removed_rows:
                    pk0.append(row_to_pk[r][0])
            elif n_pk == 2:
                pk0, pk1 = pk_lists[0], pk_lists[1]
                for r in removed_rows:
                    pk = row_to_pk[r]
                    pk0.append(pk[0])
                    pk1.append(pk[1])
            else:
                for r in removed_rows:
                    pk = row_to_pk[r]
                    for j in range(n_pk):
                        pk_lists[j].append(pk[j])

            rem_df = pl.DataFrame(pk_data) if pk_data[pk_cols[0]] else pl.DataFrame()
            if not rem_df.is_empty():
                rem_list.append(Delta(frame=rem_df, pk_columns=pk_cols, mode="remove"))

        data_df = self.patch_to_df(patch)
        if not data_df.is_empty():
            # M-10: filter out removed rows from data delta to avoid emitting both
            if removed_rows:
                remove_pk_tuples = {self.store.row_index.row_to_pk[r] for r in removed_rows if r < len(self.store.row_index.row_to_pk)}
                if remove_pk_tuples:
                    pk_col_vals = data_df.select(pk_cols).rows()
                    keep_mask = [pk not in remove_pk_tuples for pk in pk_col_vals]
                    if not all(keep_mask):
                        data_df = data_df.filter(pl.Series(keep_mask))

            if not data_df.is_empty():
                changed = tuple(c for c in data_df.columns if c not in pk_cols)
                if changed:
                    # Split into ADD vs UPDATE based on inserted_rows tracking
                    _inserted = patch.inserted_rows
                    if _inserted:
                        pk_col_vals = data_df.select(pk_cols).rows()
                        row_to_pk = self.store.row_index.row_to_pk
                        # Build set of PK tuples for inserted rows
                        inserted_pk_set = frozenset(
                            row_to_pk[r] for r in _inserted
                            if r < len(row_to_pk)
                        )
                        is_add_mask = [pk in inserted_pk_set for pk in pk_col_vals]

                        if any(is_add_mask):
                            add_df = data_df.filter(pl.Series(is_add_mask))
                            if not add_df.is_empty():
                                add_changed = tuple(c for c in add_df.columns if c not in pk_cols)
                                if add_changed:
                                    add_list.append(Delta(frame=add_df, pk_columns=pk_cols, changed_columns=add_changed, mode="add"))

                        if not all(is_add_mask):
                            upd_mask = [not x for x in is_add_mask]
                            upd_df = data_df.filter(pl.Series(upd_mask))
                            if not upd_df.is_empty():
                                upd_changed = tuple(c for c in upd_df.columns if c not in pk_cols)
                                if upd_changed:
                                    upd_list.append(Delta(frame=upd_df, pk_columns=pk_cols, changed_columns=upd_changed, mode="update"))
                    else:
                        upd_list.append(Delta(frame=data_df, pk_columns=pk_cols, changed_columns=changed, mode="update"))

        if not (add_list or upd_list or rem_list):
            return None

        ctx = RoomContext(room=self.room, grid_id=self.grid_id, primary_keys=pk_cols)
        msg = Publish(
            room=self.room,
            grid_id=self.grid_id,
            pk_columns=pk_cols,
            based_on=based_on,
            action_seq=action_seq,
            add=add_list or None,
            update=upd_list or None,
            remove=rem_list or None,
            options=options,
            trace=trace,
            context=ctx,
        )
        return msg

    async def _persist_record_patch(self, patch: CellPatch) -> None:
        if patch is None or patch.is_empty():
            return
        s = self.store
        if s is None:
            return

        alive_cid = s.row_alive_col_id

        async with self._persist_lock:
            deleted_rows = self._persist_deleted_rows
            dirty_map = self._persist_dirty_cols_by_row

            # 1) Row alive/dead transitions — vectorized
            alive_update = patch.updates_by_col.get(alive_cid)
            if alive_update is not None:
                rids, vals = alive_update
                # Separate dead and revived rows
                dead_rids = []
                alive_rids = []
                for i in range(len(rids)):
                    if vals[i] is False:
                        dead_rids.append(rids[i])
                    elif vals[i] is True:
                        alive_rids.append(rids[i])

                if dead_rids:
                    deleted_rows.update(dead_rids)
                    _pop = dirty_map.pop
                    for rr in dead_rids:
                        _pop(rr, None)

                if alive_rids:
                    deleted_rows.difference_update(alive_rids)
                    _setdefault = dirty_map.setdefault
                    for rr in alive_rids:
                        _setdefault(rr, set())

            # 2) Cell updates — bulk per column (exclude alive column)
            for cid, (rids, _vals) in patch.updates_by_col.items():
                if cid == alive_cid:
                    continue
                cid_int = int(cid)
                _setdefault = dirty_map.setdefault
                _deleted_contains = deleted_rows.__contains__
                for r in rids:
                    if not _deleted_contains(r):
                        _setdefault(r, set()).add(cid_int)

            # H-06: cap journal size — if too large, mark all columns dirty (full re-persist)
            if len(dirty_map) > _JOURNAL_CAP:
                await log.warning(f"Persistence journal for {self.grid_id} exceeded {_JOURNAL_CAP} rows, switching to full-dirty")
                all_cids = set()
                for cols in dirty_map.values():
                    all_cids.update(cols)
                # Compact: keep only alive rows with the union of all dirty columns
                self._persist_dirty_cols_by_row = {r: set(all_cids) for r in list(dirty_map.keys())[:_JOURNAL_CAP]}

        p = getattr(self.registry, "persistence", None)
        if p is not None:
            try:
                p.notify_dirty(self)
            except Exception as e:
                await log.error(f"[persist] notify_dirty failed for {self.grid_id}: {e}")

    def _persist_merge_back_unlocked(self, dirty_cols_by_row: Dict[int, Set[int]], deleted_rows: Set[int]) -> None:
        if deleted_rows:
            # M-03: only re-add deleted rows that haven't been revived (i.e., not in dirty_map)
            cur_dirty = self._persist_dirty_cols_by_row
            for r in deleted_rows:
                rr = int(r)
                if rr not in cur_dirty:
                    self._persist_deleted_rows.add(rr)

        if dirty_cols_by_row:
            cur = self._persist_dirty_cols_by_row
            for r, cols in dirty_cols_by_row.items():
                rr = int(r)
                # M-03: skip merging dirty entries for rows now in the deleted set
                if rr in self._persist_deleted_rows:
                    continue
                cur.setdefault(rr, set()).update(int(c) for c in cols)

    async def _persist_is_dirty(self) -> bool:
        async with self._persist_lock:
            return bool(self._persist_dirty_cols_by_row) or bool(self._persist_deleted_rows)

    async def persist_flush(self, *, db=None) -> Tuple[bool, bool]|None:
        """
        Flush persistence journal to DB.
        Returns (ok, still_dirty).
        - ok=False means DB ops failed; journal is merged back and will retry.
        - still_dirty=True means new changes arrived during flush window.
        """
        # Fast-path: nothing to do
        async with self._persist_lock:
            if not self._persist_dirty_cols_by_row and not self._persist_deleted_rows:
                return True, False
            snap_dirty = self._persist_dirty_cols_by_row
            snap_deleted = self._persist_deleted_rows
            self._persist_dirty_cols_by_row = {}
            self._persist_deleted_rows = set()

        if not snap_dirty and not snap_deleted:
            return True, False

        s = self.store
        if s is None:
            async with self._persist_lock:
                self._persist_merge_back_unlocked(snap_dirty, snap_deleted)
                still = bool(self._persist_dirty_cols_by_row) or bool(self._persist_deleted_rows)
            return False, still

        try:
            if db is None:
                from app.server import get_db
                db = get_db()
        except Exception:
            db = None

        if db is None:
            async with self._persist_lock:
                self._persist_merge_back_unlocked(snap_dirty, snap_deleted)
                still = bool(self._persist_dirty_cols_by_row) or bool(self._persist_deleted_rows)
            return False, still

        snapshot_seq = int(s.committed_seq())
        self.acquire_snapshot(snapshot_seq)
        alive_idx = s._alive_index_for_snapshot(snapshot_seq)

        ok = False  # C-04: init False so CancelledError always triggers merge-back
        try:
            # 1) Deletes (only if row is currently NOT alive)
            if snap_deleted:
                del_rows = [int(r) for r in snap_deleted if not _alive_contains(alive_idx, int(r))]
                if del_rows:
                    # chunk deletes to keep filter payload reasonable
                    del_rows.sort()
                    for i, j in _chunk_ranges(len(del_rows), _PERSIST_MAX_ROWS_PER_OP):
                        chunk = del_rows[i:j]
                        pk_filter = _build_pk_filter_from_row_ids(s, chunk)
                        await _db_delete_by_pk_filter(db, self.grid_id, pk_filter)

            # 2) Upserts (only for rows currently alive)
            if snap_dirty:
                groups: Dict[Tuple[int, ...], List[int]] = {}
                for r, cols in snap_dirty.items():
                    rr = int(r)
                    if not _alive_contains(alive_idx, rr):
                        continue
                    if cols:
                        # ensure deterministic grouping key — skip empty col sets (M-09)
                        k = tuple(sorted(int(c) for c in cols if int(c)!=s.row_alive_col_id))
                    else:
                        k = ()
                    if not k:
                        continue  # M-09: skip rows with no non-alive dirty columns
                    groups.setdefault(k, []).append(rr)

                if groups:
                    from app.server import get_threads
                    for col_key, row_ids in groups.items():
                        if not row_ids:
                            continue
                        row_ids.sort()
                        col_ids = list(col_key)

                        for i, j in _chunk_ranges(len(row_ids), _PERSIST_MAX_ROWS_PER_OP):
                            chunk_rows = row_ids[i:j]
                            fut = get_threads().submit(s.materialize, chunk_rows, col_ids, snapshot_seq, include_removed=False)
                            df = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)
                            await _db_upsert_sparse(db, self.grid_id, df)

            ok = True  # C-04: only set True after all DB ops succeed
        except (Exception, asyncio.CancelledError) as exc:
            ok = False
            await log.error(
                f"persist_flush failed for grid_id={self.grid_id} dirty_rows={len(snap_dirty)} deleted_rows={len(snap_deleted)}: {exc}\n{traceback.format_exc()}")
        finally:
            self.release_snapshot(snapshot_seq)
            if not ok:
                async with self._persist_lock:
                    self._persist_merge_back_unlocked(snap_dirty, snap_deleted)

        still_dirty = await self._persist_is_dirty()
        return ok, still_dirty

    async def shutdown(self, *, purge_store: bool = True) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._closed = True

        # Best-effort persistence flush before teardown
        with contextlib.suppress(Exception):
            await self.persist_flush()

        t = self._applier_task
        if (t is not None) and (not t.done()):
            t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(t, return_exceptions=True)
        self._applier_task = None

        commit_seq = self.committed_seq_hint()

        q = self._apply_q
        while True:
            try:
                r = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not r.done_fut.done():
                dropped = 0
                with contextlib.suppress(Exception):
                    dropped = int(r.patch.cell_count())
                r.done_fut.set_result(
                    AppliedPatch(
                        ingress_id=r.ingress_id,
                        based_on=r.based_on,
                        priority=r.priority,
                        source_task=r.source_task,
                        commit_seq=commit_seq,
                        survivors=CellPatch({}),
                        dropped_cells=dropped,
                        dropped_by_deps=dropped,
                        did_commit=False,
                    )
                )

        self.snapshots = SnapshotTracker()

        if purge_store:
            store = self.store
            self.store = None
            if store is not None:
                await _to_thread(_purge_mvcc_store_heavy, store)

        # clear journal always
        with contextlib.suppress(Exception):
            async with self._persist_lock:
                self._persist_dirty_cols_by_row.clear()
                self._persist_deleted_rows.clear()
        self._col_last_write_ingress.clear()

# =============================================================================
# Rule defs
# =============================================================================

class EmitMode(Enum):
    IMMEDIATE = auto()
    END = auto()
    NONE = auto()


class DepMode(Enum):
    SUCCEEDED = auto()
    FINISHED = auto()
    ERRORED = auto()


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    ERRORED = auto()
    SKIPPED = auto()

class Priority(IntEnum):
    CRITICAL = 1000
    HIGH = 750
    MEDIUM = 500
    LOW = 250
    AUDIT = 0

@dataclass(frozen=True, slots=True)
class RuleDependency:
    task: str
    mode: DepMode = DepMode.SUCCEEDED

    def satisfied(self, status: TaskStatus) -> bool:
        if self.mode == DepMode.SUCCEEDED:
            return status == TaskStatus.SUCCEEDED
        if self.mode == DepMode.ERRORED:
            return status == TaskStatus.ERRORED
        # DepMode.FINISHED
        return status in (TaskStatus.SUCCEEDED, TaskStatus.ERRORED, TaskStatus.SKIPPED)


RuleFunc = Callable[["RuleContext"], Any]


@dataclass(frozen=False, slots=True)
class RuleDef:
    name: str

    room_pattern: str = "*"
    target_room: Optional[str] = None
    target_grid_id: Optional[str] = None
    target_primary_keys: Optional[Tuple[str, ...]] = None

    task_triggers_any: Tuple[str, ...] = ()
    task_triggers_all: Tuple[str, ...] = ()

    column_triggers_any: Tuple[str, ...] = ()
    column_triggers_all: Tuple[str, ...] = ()

    depends_on_all: Tuple[RuleDependency, ...] = ()
    depends_on_any: Tuple[RuleDependency, ...] = ()

    priority: int = DEFAULT_CONFLICT_PRIORITY
    emit_mode: EmitMode = EmitMode.IMMEDIATE
    suppress_cascade: bool = False

    declared_column_outputs: Optional[Tuple[str, ...]] = None  # scheduler hint only
    func: RuleFunc = None  # type: ignore

    def __post_init__(self):
        if (self.target_grid_id is None) and (self.target_room is not None):
            self.target_grid_id = room_to_grid_id(self.target_room)
        elif (self.target_grid_id is not None) and (self.target_room is None):
            self.target_room = f"*.{self.target_grid_id.upper()}"

    def applies_to_room(self, room: str) -> bool:
        return fnmatch.fnmatchcase(room.upper(), self.room_pattern.upper())

def rule(name=None, **kwargs):
    def decorator(func):
        rname = name or func.__name__
        return RuleDef(name=rname, func=func, **kwargs)
    return decorator

# =============================================================================
# Snapshot lease manager
# =============================================================================

class SnapshotLeases:
    __slots__ = ("system", "baseline_by_room", "_leased")

    def __init__(self, system: "GridSystem", baseline_by_room: Dict[str, int]) -> None:
        self.system = system
        self.baseline_by_room = baseline_by_room
        self._leased: List[Tuple[RoomContext, int]] = []

    async def ensure(self, context: RoomContext) -> int:
        lease_key = GridActor.get_key(context)
        existing = self.baseline_by_room.get(lease_key)
        if existing is not None:
            return int(existing)
        actor = self.system.registry.require(context)
        await actor.ensure_awake()
        if actor.store is None:
            raise RuntimeError(f"(5) Store is None for actor {getattr(actor, 'grid_id', '?')}")
        seq = int(actor.store.committed_seq())
        self.baseline_by_room[lease_key] = seq
        actor.acquire_snapshot(seq)
        self._leased.append((context, seq))
        return seq

    def acquire_existing(self, context:RoomContext, seq: int) -> None:
        seq = int(seq)
        actor = self.system.registry.require(context)
        actor.acquire_snapshot(seq)
        self._leased.append((context, seq))

    def release_all(self) -> None:
        for context, seq in self._leased:
            try:
                self.system.registry.require(context).release_snapshot(seq)
            except Exception:
                pass  # actor may have been destroyed; safe to skip
        self._leased.clear()


# =============================================================================
# Rule context
# =============================================================================

@dataclass(slots=True)
class RuleContext:
    system: "GridSystem"
    leases: SnapshotLeases

    source_context: RoomContext
    source_room: str
    target_room: str
    target_pks: Tuple[str, ...]
    based_on: int
    ingress_id: int
    triggering_delta: pl.DataFrame
    ingress_user: Optional[User] = None

    source_grid_id: str = field(init=False)
    target_grid_id: str = field(init=False)

    # deps stored as deduped maps
    _col_deps: Dict[Tuple[str, int], Set[int]] = field(default_factory=dict, repr=False)
    _row_deps: Dict[Tuple[str, int], Tuple[Set[int], Set[int]]] = field(default_factory=dict, repr=False)
    # (room,snapshot) -> (col_ids_set, row_ids_set)

    def __post_init__(self):
        self.source_grid_id = room_to_grid_id(self.source_room)
        self.target_grid_id = room_to_grid_id(self.target_room)

    async def baseline_seq(self, context: RoomContext) -> int:
        return await self.leases.ensure(context)

    def _record_col_dep(self, context: RoomContext, snapshot_seq: int, col_ids: List[int]) -> None:
        if not col_ids:
            return
        col_key = (context.grid_id, int(snapshot_seq))
        s = self._col_deps.setdefault(col_key, set())
        s.update(int(c) for c in col_ids)

    def _record_row_dep(self, context: RoomContext, snapshot_seq: int, col_ids: List[int], row_ids: List[int]) -> None:
        if not col_ids or not row_ids:
            return
        row_key = (context.grid_id, int(snapshot_seq))
        cols_set, rows_set = self._row_deps.get(row_key, (set(), set()))
        cols_set.update(int(c) for c in col_ids)
        rows_set.update(int(r) for r in row_ids)
        self._row_deps[row_key] = (cols_set, rows_set)

    def current_read_deps(self) -> ReadDeps:
        col_deps: List[ColumnDep] = []
        row_deps: List[RowLocalDep] = []

        for (grid_id, snap), cols_set in self._col_deps.items():
            cols = tuple(sorted(cols_set))
            if cols:
                col_deps.append(ColumnDep(grid_id=grid_id, snapshot_seq=snap, col_ids=cols))

        for (grid_id, snap), (cols_set, rows_set) in self._row_deps.items():
            cols = tuple(sorted(cols_set))
            rows = tuple(sorted(rows_set))
            if cols and rows:
                row_deps.append(RowLocalDep(
                    grid_id=grid_id,
                    snapshot_seq=snap,
                    col_ids=cols,
                    row_ids=rows)
                )
        return ReadDeps(col_deps=tuple(col_deps), row_deps=tuple(row_deps))

    def get_micro_grid_data(self, micro_name: str) -> Optional[pl.DataFrame]:
        """Access micro-grid snapshot from any rule (main grid or micro-grid).
        Returns a cloned DataFrame or None if the micro-grid doesn't exist."""
        try:
            from app.services.redux.micro_grid import get_micro_actor
            actor = get_micro_actor(micro_name)
            return actor.snapshot()
        except (KeyError, Exception):
            return None

    async def snapshot_running(self, context: Optional[RoomContext] = None) -> int:
        a = self.system.registry.require((context or self.source_context))
        await a.ensure_awake()
        if a.store is None:
            raise RuntimeError(f"(6) Store is None for actor {getattr(a, 'grid_id', '?')}")
        return int(a.store.committed_seq())

    @staticmethod
    def _resolve_pk_rows(df: pl.DataFrame, store: "GridMVCCStore") -> List[int]:
        """Batch-resolve PK tuples from a DataFrame to row IDs via .rows()."""
        pk_rows = df.select(store.pk_cols).rows()
        _resolve = store.row_index.resolve_existing
        row_ids: List[int] = []
        _append = row_ids.append
        for pk in pk_rows:
            rid = _resolve(pk)
            if rid is not None:
                _append(int(rid))
        return row_ids

    async def running_delta_slice(self, columns: Optional[List[str]] = None, *, context: Optional[RoomContext] = None) -> pl.DataFrame:
        context = (context or self.source_context)
        actor = self.system.registry.require(context)
        await actor.ensure_awake()
        actor.touch()
        if actor.store is None:
            raise RuntimeError(f"(7) Store is None for actor {getattr(actor, 'grid_id', '?')}")
        snapshot = int(actor.store.committed_seq())

        df = self.triggering_delta
        if df.is_empty():
            return pl.DataFrame()

        pk_cols = actor.store.pk_cols
        for pk in pk_cols:
            if pk not in df.columns:
                raise KeyError(f"running_delta_slice: missing PK column {pk!r} in delta")

        cols = [c for c in df.columns if c not in pk_cols]
        if columns:
            for c in columns:
                if c not in cols and c not in pk_cols:
                    cols.append(c)

        row_ids = self._resolve_pk_rows(df, actor.store)

        col_ids = [actor.store.ensure_column(c) for c in cols]
        self._record_row_dep(context, snapshot, col_ids, row_ids)
        from app.server import get_threads
        fut = get_threads().submit(actor.store.materialize, row_ids, col_ids, snapshot, include_removed=True)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    async def running_frame_slice(self, columns: Optional[List[str]] = None, *, context: Optional[RoomContext] = None) -> pl.DataFrame:
        context = (context or self.source_context)
        actor = self.system.registry.require(context)
        await actor.ensure_awake()
        actor.touch()
        if actor.store is None:
            raise RuntimeError(f"(8) Store is None for actor {getattr(actor, 'grid_id', '?')}")
        snapshot = int(actor.store.committed_seq())

        if columns is None:
            names = list(actor.store.cols.id_to_name)
            columns = [c for c in names if c not in actor.store.pk_cols and c != _INTERNAL_ROW_ALIVE]

        col_ids = [actor.store.ensure_column(c) for c in columns]
        self._record_col_dep(context, snapshot, col_ids)
        from app.server import get_threads
        fut = get_threads().submit(actor.store.materialize, None, col_ids, snapshot, include_removed=False)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    async def ingress_delta_slice(self, columns: Optional[List[str]] = None, *, context: Optional[RoomContext] = None) -> pl.DataFrame:
        context = (context or self.source_context)
        actor = self.system.registry.require(context)
        snapshot = int(await self.baseline_seq(context))
        await actor.ensure_awake()
        actor.touch()
        if actor.store is None:
            raise RuntimeError(f"(9) Store is None for actor {getattr(actor, 'grid_id', '?')}")

        df = self.triggering_delta
        if df.is_empty():
            return pl.DataFrame()

        pk_cols = actor.store.pk_cols
        cols = [c for c in df.columns if c not in pk_cols]
        if columns:
            for c in columns:
                if c not in cols and c not in pk_cols:
                    cols.append(c)

        row_ids = self._resolve_pk_rows(df, actor.store)

        col_ids = [actor.store.ensure_column(c) for c in cols]
        from app.server import get_threads
        fut = get_threads().submit(actor.store.materialize, row_ids, col_ids, snapshot, include_removed=True)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    async def ingress_frame_slice(self, columns: Optional[List[str]] = None, *, context: Optional[RoomContext] = None) -> pl.DataFrame:
        context = (context or self.source_context)
        actor = self.system.registry.require(context)
        snapshot = int(await self.baseline_seq(context))
        await actor.ensure_awake()
        actor.touch()
        if actor.store is None:
            raise RuntimeError(f"(10) Store is None for actor {getattr(actor, 'grid_id', '?')}")

        if columns is None:
            names = list(actor.store.cols.id_to_name)
            columns = [c for c in names if c not in actor.store.pk_cols and c != _INTERNAL_ROW_ALIVE]

        col_ids = [actor.store.ensure_column(c) for c in columns]
        from app.server import get_threads
        fut = get_threads().submit(actor.store.materialize, None, col_ids, snapshot, include_removed=False)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    async def prior_delta_slice(self, columns: Optional[List[str]] = None, *, context: Optional[RoomContext] = None) -> pl.DataFrame:
        context = (context or self.source_context)
        actor = self.system.registry.require(context)
        snapshot = int(await self.baseline_seq(context))-1
        await actor.ensure_awake()
        actor.touch()
        if actor.store is None:
            raise RuntimeError(f"(11) Store is None for actor {getattr(actor, 'grid_id', '?')}")

        df = self.triggering_delta
        if df.is_empty():
            return pl.DataFrame()

        pk_cols = actor.store.pk_cols
        cols = [c for c in df.columns if c not in pk_cols]
        if columns:
            for c in columns:
                if c not in cols and c not in pk_cols:
                    cols.append(c)

        row_ids = self._resolve_pk_rows(df, actor.store)

        col_ids = [actor.store.ensure_column(c) for c in cols]
        from app.server import get_threads
        fut = get_threads().submit(actor.store.materialize, row_ids, col_ids, snapshot, include_removed=True)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    async def prior_frame_slice(self, columns: Optional[List[str]] = None, *, context: Optional[RoomContext] = None) -> pl.DataFrame:
        context = (context or self.source_context)
        actor = self.system.registry.require(context)
        snapshot = int(await self.baseline_seq(context))-1
        await actor.ensure_awake()
        actor.touch()
        if actor.store is None:
            raise RuntimeError(f"(12) Store is None for actor {getattr(actor, 'grid_id', '?')}")

        if columns is None:
            names = list(actor.store.cols.id_to_name)
            columns = [c for c in names if c not in actor.store.pk_cols and c != _INTERNAL_ROW_ALIVE]

        col_ids = [actor.store.ensure_column(c) for c in columns]
        from app.server import get_threads
        fut = get_threads().submit(actor.store.materialize, None, col_ids, snapshot, include_removed=False)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)


# =============================================================================
# Rules engine
# =============================================================================

@dataclass(order=False, slots=True)
class ScheduledRule:
    priority: int
    seq: int
    rule: RuleDef
    source_context: RoomContext
    source_room: str
    target_room: str
    target_grid_id: str
    target_pks: Tuple[str, ...]
    triggering_delta: pl.DataFrame

    def __lt__(self, other: "ScheduledRule") -> bool:
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.seq < other.seq


class OutputLockManager:
    """
    Optional conservative mode: if declared outputs are None, treat as "unknown"
    and lock the entire room.
    """
    def __init__(self, conservative_undeclared: bool = False) -> None:
        self._lock = asyncio.Lock()
        self._held: Dict[str, Set[str]] = {}
        self._conservative = bool(conservative_undeclared)

    def _cols_key(self, cols: Optional[Tuple[str, ...]]) -> Set[str]:
        if cols is None:
            return {"*"} if self._conservative else set()
        s = {c for c in cols if c and c != _INTERNAL_ROW_ALIVE}
        return s or ({"*"} if self._conservative else set())

    async def try_acquire(self, room:str, cols: Optional[Tuple[str, ...]]) -> bool:
        want = self._cols_key(cols)
        if not want:
            return True
        async with self._lock:
            held = self._held.setdefault(room, set())
            if "*" in held or "*" in want:
                if held:
                    return False
                held |= want
                return True
            if held & want:
                return False
            held |= want
            return True

    async def release(self, room: str, cols: Optional[Tuple[str, ...]]) -> None:
        want = self._cols_key(cols)
        if not want:
            return
        async with self._lock:
            held = self._held.get(room)
            if not held:
                return
            held -= want
            if not held:
                self._held.pop(room, None)

def room_to_grid_id(room:str):
    return room.split(".")[-1].lower()

class RulesEngine:
    def __init__(self, *, max_concurrent: int = 16, max_rules_per_ingress: int = 10_000, conservative_undeclared_outputs: bool = False) -> None:
        self._rules: Dict[str, RuleDef] = {}
        self._max_concurrent = int(max_concurrent)
        self._max_rules_per_ingress = int(max_rules_per_ingress)
        self._conservative_undeclared = bool(conservative_undeclared_outputs)
        self._shutdown_event = threading.Event()

    def register(self, rule: RuleDef|Any) -> None:
        try:
            if rule.name in self._rules:
                raise ValueError(f"Rule already registered: {rule.name}")
            self._rules[rule.name] = rule
        except Exception as e:
            log.error(f"Failed to register rule ({rule.name}): {e}")

    async def run_ingress(
            self,
            *,
            system: "GridSystem",
            actor_context: RoomContext,
            ingress_id: int,
            based_on: int,
            publish_delta_by_room: Dict[str, pl.DataFrame],
            primary_keys: Tuple[str, ...],
            leases: SnapshotLeases,
            trace: Optional[str],
            emit_options: Optional[PayloadOptions],
            trigger_rules: bool,
            ingress_user: Optional[User]=None
    ) -> AsyncIterator[Publish]:

        if not trigger_rules: return

        persist_enabled = bool(getattr(emit_options, "persist", True)) if emit_options is not None else True

        lock_mgr = OutputLockManager(conservative_undeclared=self._conservative_undeclared)

        cols_hit: Dict[str, Dict[str, Set[str]]] = {}  # room -> task -> columns
        task_status: Dict[str, Dict[str, TaskStatus]] = {}  # room -> task -> status
        tasks_observed: Dict[str, Set[str]] = {}  # room -> tasks that emitted a delta

        def record_delta(room: str, task: str, delta_df: pl.DataFrame) -> Set[str]:
            pk = set(primary_keys)
            changed = set(delta_df.columns) - pk
            cols_hit.setdefault(room, {}).setdefault(task, set()).update(changed)
            return changed

        def record_task(room: str, task: str, result: TaskStatus, details:str=None):
            tasks_observed.setdefault(room, set()).add(task)
            task_status.setdefault(room, {})[task] = result
            if result is TaskStatus.ERRORED:
                log.rules(f"Task ({task}) has {result.name}", details=details, color="red")
            else:
                log.rules(f"Task ({task}) has {result.name}")

        # seed
        for room, df in publish_delta_by_room.items():
            record_delta(room, "publish", df)
            record_task(room, "publish", TaskStatus.SUCCEEDED)

        scheduled_names: Set[Tuple[str, str]] = set()  # (key, rule_name)
        pending: List[Tuple[int, ScheduledRule]] = []
        running: Dict[Tuple[str, str], asyncio.Task] = {}
        end_buffer: List[Publish] = []
        seq = 0
        _schedule_lock = asyncio.Lock()  # H-14: serialize schedule_if_eligible

        yield_queue: asyncio.Queue[Publish] = asyncio.Queue(maxsize=4096)  # M-17: bounded

        async def schedule_if_eligible(room: str, triggering_delta: pl.DataFrame) -> None:
            nonlocal seq
            async with _schedule_lock:  # H-14: serialize to prevent TOCTOU on shared state
                room_tasks = task_status.setdefault(room, {})
                room_obs = tasks_observed.setdefault(room, set())
                room_hits = cols_hit.setdefault(room, {})

                def union_cols_for(tasks: Iterable[str]) -> Set[str]:
                    u: Set[str] = set()
                    for t in tasks:
                        u.update(room_hits.get(t, set()))
                    return u

                for r in self._rules.values():

                    if not r.applies_to_room(room):
                        continue

                    rule_key = (room, r.name)
                    if rule_key in scheduled_names:
                        continue

                    # deps (completion)
                    ok = True
                    if r.depends_on_all:
                        for dep in r.depends_on_all:
                            st = room_tasks.get(dep.task, TaskStatus.PENDING)
                            if not dep.satisfied(st):
                                ok = False
                                break
                        if not ok:
                            continue

                    if r.depends_on_any:
                        any_ok = False
                        for dep in r.depends_on_any:
                            st = room_tasks.get(dep.task, TaskStatus.PENDING)
                            if dep.satisfied(st):
                                any_ok = True
                                break
                        if not any_ok:
                            continue

                    # task triggers (presence/observation)
                    if r.task_triggers_any:
                        if not any(t in room_obs for t in r.task_triggers_any):
                            continue
                    if r.task_triggers_all:
                        if not all(t in room_obs for t in r.task_triggers_all):
                            continue

                    # column triggers (based on union across relevant tasks)
                    rel_tasks: Set[str] = set(r.task_triggers_any) | set(r.task_triggers_all)
                    if not rel_tasks:
                        rel_tasks = set(room_hits.keys())

                    union = union_cols_for(rel_tasks)

                    if r.column_triggers_any:
                        if not (set(r.column_triggers_any) & union):
                            continue
                    if r.column_triggers_all:
                        if not set(r.column_triggers_all).issubset(union):
                            continue

                    # Add to scheduled_names immediately to prevent double-schedule
                    scheduled_names.add(rule_key)
                    seq += 1
                    tgt = r.target_room or room
                    tgt_pks = r.target_primary_keys or primary_keys or actor_context.primary_keys
                    tgt_grid = r.target_grid_id or room_to_grid_id(room)

                    if not tgt_pks:
                        tgt_pks = await query_primary_keys(tgt_grid)

                    stamp_key = -1 * pack_stamp_key(based_on, int(r.priority), ingress_id)
                    heapq.heappush(
                        pending,
                        (stamp_key, ScheduledRule(
                            priority=int(r.priority),
                            seq=seq,
                            rule=r,
                            source_context=actor_context,
                            source_room=room,
                            target_room=tgt,
                            target_pks=tgt_pks,
                            target_grid_id=tgt_grid,
                            triggering_delta=triggering_delta,
                        ),
                         ))

        # initial schedule
        for room, df in publish_delta_by_room.items():
            await schedule_if_eligible(room, df)

        async def run_one(task: ScheduledRule) -> None:
            rdef = task.rule
            tgt = task.target_room
            tgt_pks = task.target_pks
            room = task.source_room

            task_status.setdefault(room, {})[rdef.name] = TaskStatus.RUNNING

            ctx = RuleContext(
                system=system,
                leases=leases,
                source_context=actor_context,
                source_room=room,
                target_room=tgt,
                target_pks=tgt_pks,
                based_on=based_on,
                ingress_id=ingress_id,
                triggering_delta=task.triggering_delta,
                ingress_user=ingress_user
            )

            def normalize_output(obj: Any) -> List[Delta]:
                if obj is None:
                    return []
                if isinstance(obj, Delta):
                    return [obj]
                if isinstance(obj, pl.LazyFrame):
                    obj = obj.hyper.collect_threaded()
                    return [Delta(frame=obj, pk_columns=tgt_pks, mode="update")]
                if isinstance(obj, pl.DataFrame):
                    # Default to update delta
                    return [Delta(frame=obj, pk_columns=tgt_pks, mode="update")]
                if isinstance(obj, (list, tuple)):
                    out: List[Delta] = []
                    for it in obj:
                        out.extend(normalize_output(it))
                    return out
                raise TypeError(f"Unsupported rule output type: {type(obj)!r}")

            async def commit_delta(d: Delta) -> None:
                df = d.frame if isinstance(d.frame, (pl.LazyFrame, pl.DataFrame)) else None
                if (df is None) or df.hyper.is_empty():
                    return

                primary_keys = d.pk_columns or tgt_pks

                # Cross-room PK validation
                if primary_keys:
                    for pk in primary_keys:
                        if pk not in df.hyper.schema():
                            await log.error("Missing pks!")
                            raise KeyError(f"Rule {rdef.name} output to room={tgt} missing PK column {pk!r}")

                # Convert Delta -> Payloads -> Patch
                ploads = Payloads(
                    add=[d] if d.mode=="add" else None,
                    update=[d] if d.mode=="update" else None,
                    remove=[d] if d.mode=="remove" else None,
                    _pk_columns=primary_keys,
                    based_on=based_on,
                    action_seq=ingress_id,
                )

                _pub = Publish(
                    room=task.target_room,
                    grid_id=task.target_grid_id,
                    payloads=ploads
                )

                deps = ctx.current_read_deps()
                actors = system.registry.find_actors(task.target_room, _pub)
                await log.notify(f"Found {len(actors)} interested actors using for {task.target_room}")
                for actor, matched_room in actors:

                    try:
                        await actor.ensure_awake()
                        actor.touch()
                        if actor.store is None:
                            raise RuntimeError(f"(13) Store is None for actor {getattr(actor, 'grid_id', '?')}")
                    except Exception as e:
                        await log.error(f"Error committing delta for rule {rdef.name}: {e}\n{traceback.format_exc()}")
                        continue

                    patch = actor.payloads_to_patch(ploads)
                    applied = await actor.commit_patch(
                        ingress_id=ingress_id,
                        based_on=based_on,
                        priority=rdef.priority,
                        source_task=rdef.name,
                        patch=patch,
                        read_deps=deps,
                        await_commit=True,
                        persist=persist_enabled,
                    )
                    if applied is None or applied.survivors.is_empty() or not applied.did_commit:
                        continue

                    survivors_df = actor.patch_to_df(applied.survivors)

                    pub = actor.patch_to_payload_publish(
                        based_on=based_on,
                        action_seq=ingress_id,
                        patch=applied.survivors,
                        options=emit_options,
                        trace=trace,
                    )



                    if pub is not None and rdef.emit_mode!=EmitMode.NONE:
                        if rdef.emit_mode==EmitMode.IMMEDIATE:
                            yield_queue.put_nowait(pub)
                        else:
                            end_buffer.append(pub)

                    if not rdef.suppress_cascade and not survivors_df.is_empty():
                        record_delta(matched_room, rdef.name, survivors_df)
                        await schedule_if_eligible(matched_room, survivors_df)

            try:
                if not ((rdef is None) or (getattr(rdef, "func", None) is None)):
                    res = rdef.func(ctx)
                    if inspect.isasyncgen(res):
                        async for out in res:  # type: ignore[misc]
                            for d in normalize_output(out):
                                await commit_delta(d)
                    else:
                        out = await res if inspect.isawaitable(res) else res
                        for d in normalize_output(out):
                            await commit_delta(d)

                record_task(room, rdef.name, TaskStatus.SUCCEEDED)
            except Exception as e:
                record_task(room, rdef.name, TaskStatus.ERRORED, details=traceback.format_exc())
            finally:
                await lock_mgr.release(tgt, rdef.declared_column_outputs)
                await schedule_if_eligible(room, task.triggering_delta)

        def attach_done(room: str, rule_name: str, task_obj: asyncio.Task) -> None:
            task_key = (room, rule_name)

            def _cb(_t: asyncio.Task) -> None:
                running.pop(task_key, None)
                log.rules(f"Task is done: {task_key}")

            task_obj.add_done_callback(_cb)

        async def _cancel_running() -> None:
            if not running:
                return
            tasks = list(running.values())
            running.clear()
            for t in tasks:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)

        async def try_start() -> None:
            blocked: List[Tuple[int, ScheduledRule]] = []
            while pending and len(running) < self._max_concurrent:
                p, s = heapq.heappop(pending)
                if len(scheduled_names) > self._max_rules_per_ingress:
                    break
                ok = await lock_mgr.try_acquire(s.target_room, s.rule.declared_column_outputs)
                if not ok:
                    blocked.append((p, s))
                    continue

                task_room = s.source_room
                task_key = (task_room, s.rule.name)
                t = asyncio.create_task(
                    run_one(s),
                    name=f"rule:{ingress_id}:{task_room}:{s.rule.name}"
                )
                await log.debug(f"Running task: {s.rule.name}")
                running[task_key] = t
                attach_done(task_room, s.rule.name, t)

            for b in blocked:
                heapq.heappush(pending, b)

        try:
            while not self._shutdown_event.is_set():
                await try_start()

                while not yield_queue.empty():
                    yield yield_queue.get_nowait()

                if not pending and not running:
                    break

                if running:
                    # Wait for at least one running task to finish instead of busy-spinning
                    done, _ = await asyncio.wait(
                        list(running.values()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await asyncio.sleep(0)

            for pub in end_buffer:
                yield pub
        finally:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await _cancel_running()
            pending.clear()

    async def shutdown(self):
        self._shutdown_event.set()

# =============================================================================
# Developer API types
# =============================================================================

@dataclass(slots=True)
class ActorStatus:
    key: str
    room: str
    grid_id: str
    grid_filters: dict
    subscriber_count: int
    hibernating: bool
    last_touched_ns: int
    idle_s: float
    commit_seq: int
    apply_q_size: int
    snapshots_active: bool
    lifecycle_gen: int
    closed: bool

# =============================================================================
# GridSystem
# =============================================================================
class ActorMailbox:
    def __init__(self):
        self.router = PubSubRouter()

    async def enroll(self, actor: GridActor):
        return await self.router.subscribe(actor, actor.context.room, actor.context.grid_filters)

    async def disenroll(self, actor: GridActor):
        return await self.router.unsubscribe(actor, actor.context.room)

    def mail(self, grid_id:str, payload:Publish):
        return self.router.publish(grid_id, payload, strict=False, dedupe_subscribers=True)

    def list_subscriptions(self, actor: GridActor):
        return self.router.subscriptions_for(actor)

    async def unsubscribe_all(self, actor):
        return await self.router.unsubscribe_all(actor)

    async def shutdown(self) -> None:
        await self.router.shutdown()

class GridSystem:
    def __init__(self, *, max_rule_concurrency:int=16) -> None:
        self.registry = ActorRegistry()
        self.clock = AsyncGlobalClock()
        self.action_clock = AsyncGlobalClock()
        self.rules = RulesEngine(max_concurrent=int(max_rule_concurrency))
        self._closed = False

        self._reaper_task: Optional[asyncio.Task] = None
        self._reaper_stop: Optional[asyncio.Event] = None
        self._remove_tasks: Dict[str, asyncio.Task] = {}

        self._reaper_interval_ns = max(1, int(_GRID_REAPER_INTERVAL_S * 1_000_000_000))
        self._actor_idle_ns = max(1, int(_GRID_ACTOR_IDLE_S * 1_000_000_000))
        self._remove_grace_ns = max(0, int(_GRID_REMOVE_GRACE_S * 1_000_000_000))
        self._reaper_max_actions = int(_GRID_REAPER_MAX_ACTIONS)

        self.persistence = GridPersistenceManager(registry=self.registry)
        self.registry.persistence = self.persistence
        self._persistence_started = False

    def _ensure_persistence_started(self) -> None:
        if self._closed or self._persistence_started: return
        self._persistence_started = True
        if self.persistence is not None:
            with contextlib.suppress(Exception):
                self.persistence.start()
            log.persist("Persistence Loop Started", color='#4764f5')

    def _ensure_reaper_started(self) -> None:
        t = self._reaper_task
        if t is not None and not t.done(): return
        self._reaper_stop = asyncio.Event()
        self._reaper_task = asyncio.create_task(self._reaper_loop(), name="grid-reaper")
        log.reaper(f"Grid Reaper started", color="#333436")


    def _ensure_loops_started(self):
        self._ensure_persistence_started()
        self._ensure_reaper_started()

    async def start_reaper(self) -> bool:
        if self._closed: return False
        self._ensure_reaper_started()
        return True

    async def stop_reaper(self) -> bool:
        stop = self._reaper_stop
        if stop is None: return False
        stop.set()
        t = self._reaper_task
        if t is not None:
            t.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await t
        self._reaper_task = None
        self._reaper_stop = None
        return True

    def check_reaper_status(self) -> dict:
        t = self._reaper_task
        running = (t is not None) and (not t.done())
        return {
            "running": running,
            "task_name": getattr(t, "get_name", lambda: None)() if t else None,
            "interval_s": self._reaper_interval_ns / 1_000_000_000,
            "actor_idle_s": self._actor_idle_ns / 1_000_000_000,
            "remove_grace_s": self._remove_grace_ns / 1_000_000_000,
            "max_actions": self._reaper_max_actions,
            "closed": self._closed,
        }

    def adjust_idle_timeout(self, seconds: float) -> None:
        ns = max(1, int(float(seconds) * 1_000_000_000))
        self._actor_idle_ns = ns

    def adjust_reaper_interval(self, seconds: float) -> None:
        ns = max(1, int(float(seconds) * 1_000_000_000))
        self._reaper_interval_ns = ns

    def adjust_remove_grace(self, seconds: float) -> None:
        ns = max(0, int(float(seconds) * 1_000_000_000))
        self._remove_grace_ns = ns

    def adjust_reaper_max_actions(self, n: int) -> None:
        self._reaper_max_actions = max(1, int(n))

    async def _reaper_loop(self) -> None:
        stop = self._reaper_stop
        if stop is None:
            return
        try:
            while not stop.is_set() and not self._closed:
                start = _now_ns()
                actions = 0

                actors = self.registry.iter_actors()
                now = start

                for a in actors:
                    if actions >= self._reaper_max_actions:
                        break
                    if a is None or a._closed or a._shutdown_started:
                        continue

                    subs = a.subscriber_count()
                    if subs == 0:
                        if a.can_reap_remove():
                            actions += 1
                            with contextlib.suppress(Exception):
                                await self.shutdown_actor(a.context, purge_store=True, unsubscribe_subscribers=False)
                        continue

                    if not a.is_hibernating():
                        idle = now - a.last_touched_ns()
                        if idle >= self._actor_idle_ns and not a.snapshots.any_active():
                            try:
                                if a._apply_q.qsize() == 0:
                                    actions += 1
                                    with contextlib.suppress(Exception):
                                        res = await a.hibernate()
                                        if res:
                                            await log.reaper(f"Hibernating {a.context.room}...", color="#333436")
                            except Exception:
                                pass

                elapsed = _now_ns() - start
                sleep_ns = self._reaper_interval_ns - elapsed
                if sleep_ns <= 0:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(sleep_ns / 1_000_000_000)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await log.error(f"_reaper_loop crashed: {exc}\n{traceback.format_exc()}")
            # H-16: auto-restart after backoff
            if not self._closed and (stop is None or not stop.is_set()):
                await asyncio.sleep(5.0)
                self._ensure_reaper_started()

    async def _remove_if_still_empty(self, context: RoomContext, gen: int) -> bool | None:
        key = GridActor.get_key(context)
        try:
            grace_ns = self._remove_grace_ns
            if grace_ns > 0:
                await asyncio.sleep(grace_ns / 1_000_000_000)
            actor = self.registry.get(context)
            if actor is None:
                return False
            if actor.lifecycle_gen() != gen:
                return False
            if actor.subscriber_count() != 0:
                return False
            if not actor.can_reap_remove():
                return False
            await self.shutdown_actor(context, purge_store=True)
            return True
        finally:
            self._remove_tasks.pop(key, None)


    async def register_actor(self, *, context: RoomContext, df: Optional[pl.DataFrame] = None, pk_cols: Optional[Sequence[str]] = None) -> GridActor:
        if self._closed:
            raise RuntimeError("GridSystem is shut down")
        self._ensure_loops_started()
        if df is None:
            df, pk_cols = await GridActor.build_from_db(context, return_pks=True)
        store = GridMVCCStore.from_frame(df, pk_cols=pk_cols)
        actor = GridActor(context=context, store=store, registry=self.registry)
        actor.touch()
        await self.registry.register(actor)
        return actor

    async def get_actor(self, context: RoomContext, create_on_missing: bool = True) -> GridActor:
        if self._closed:
            raise RuntimeError("GridSystem is shut down")
        self._ensure_loops_started()
        # H-02: use registry lock to prevent TOCTOU race on actor creation
        async with self.registry._reg_lock:
            actor = self.registry.get(context)
            if actor is None:
                if not create_on_missing:
                    raise KeyError(f"Actor does not exist: {context.room}")
                # Release lock during DB fetch, then re-check
        if actor is None:
            actor = await self.register_actor(context=context)
        await actor.ensure_awake()
        actor.touch()
        return actor

    async def shutdown_actor(self, context: RoomContext, *, purge_store: bool = True, unsubscribe_subscribers: bool = True) -> List[Any]:
        if self._closed: return []
        actor = self.registry.get(context)
        if actor is None: return []
        removed: List[Any] = []
        if unsubscribe_subscribers:
            with contextlib.suppress(Exception):
                removed = await actor.remove_all_subscribers()
        key = actor.key
        t = self._remove_tasks.pop(key, None)
        if t is not None:
            t.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await t
        await self.registry.remove(context, purge_store=purge_store)
        return removed

    async def trigger_hibernate(self, context: RoomContext) -> bool:
        actor = self.registry.get(context)
        if actor is None:
            return False
        return await actor.hibernate()

    async def trigger_awake(self, context: RoomContext) -> bool:
        actor = self.registry.get(context)
        if actor is None:
            return False
        await actor.ensure_awake()
        return True

    async def trigger_rebuild(self, context: RoomContext, *, allow_when_busy: bool = False) -> bool:
        actor = self.registry.get(context)
        if actor is None:
            return False
        return await actor.force_rebuild_from_db(allow_when_busy=allow_when_busy)

    async def user_subscribe(self, context: RoomContext, subscriber: Any, *, create_on_missing: bool = True) -> None:
        if self._closed: return
        actor = await self.get_actor(context, create_on_missing=create_on_missing)
        await actor.add_subscriber(subscriber)

    async def user_unsubscribe(self, context: RoomContext, subscriber: Any) -> None:
        if self._closed: return
        actor = self.registry.get(context)
        if actor is None: return
        from app.server import get_ctx
        await actor.remove_subscriber(subscriber)
        if actor.subscriber_count()==0:
            key = actor.key
            gen = actor.lifecycle_gen()
            if key not in self._remove_tasks:
                self._remove_tasks[key] = get_ctx().spawn(
                    self._remove_if_still_empty(context, gen),
                    name=f"actor-remove:{actor.room}"
                )

    async def list_actor_subscribers(self, context: RoomContext) -> List[Any]:
        actor = self.registry.get(context)
        if actor is None:
            return []
        return await actor.list_subscribers()

    async def get_frame(self, context: RoomContext, cols: Optional[List[str]] = None, *, include_removed: bool = False) -> pl.DataFrame:
        if self._closed:
            raise RuntimeError("GridSystem is shut down")
        actor = await self.get_actor(context, create_on_missing=False)
        if actor.store is None:
            raise RuntimeError(f"(14) Store is None for actor {getattr(actor, 'grid_id', '?')}")

        from app.server import get_threads
        if cols is None:
            return await actor.materialize_running_frame(include_removed=include_removed)
        snapshot = actor.store.committed_seq()
        col_ids = [actor.store.ensure_column(c) for c in cols]
        fut = get_threads().submit(actor.store.materialize, None, col_ids, snapshot, include_removed=include_removed)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60.0)

    def build_publish_from_df(
            self,
            *,
            context: RoomContext,
            df: pl.DataFrame,
            mode: str = "update",
            options: Optional[PayloadOptions] = None,
            trace: Optional[str] = None,
    ) -> Publish:
        actor = self.registry.require(context)
        if actor.store is None:
            raise RuntimeError(f"(15) Store is None for actor {getattr(actor, 'grid_id', '?')}")
        pk_cols = list(actor.store.pk_cols)
        return Publish(
            context=context,
            pk_columns=pk_cols,
            delta_frame=df,
            delta_mode=mode,
            options=options or PayloadOptions(),
            trace=trace,
        )

    async def apply_df(self, *, context: RoomContext, df: pl.DataFrame, mode: str = "update", options: Optional[PayloadOptions] = None) -> List[Publish]:
        pub = self.build_publish_from_df(context=context, df=df, mode=mode, options=options)
        out: List[Publish] = []
        async for m in self.ingest_publish(pub):
            out.append(m)
        return out

    async def ingest_publish(self, msg: Publish) -> AsyncIterator[Publish]:
        """
        Full pipeline:
          - assign global based_on/action_seq
          - commit publish to actor
          - yield committed survivors publish payload
          - run rules (possibly cross-room) and yield their survivors as publish payloads
        """
        actor = await self.get_actor(msg.context, create_on_missing=True)
        context: RoomContext = actor.context
        if context is None:
            raise ValueError("Invalid Publish Payload, no Room Context")

        actor.touch()
        based_on = await self.clock.next()
        action_seq = await self.action_clock.next()
        ingress_id = action_seq

        options = getattr(msg, "options", None) or PayloadOptions()
        priority = int(getattr(options, "priority", DEFAULT_CONFLICT_PRIORITY) or DEFAULT_CONFLICT_PRIORITY)
        trigger_rules = bool(getattr(options, "trigger_rules", True))
        silent = bool(getattr(options, "silent", False))
        trace = getattr(msg, "trace", None)
        persist_enabled = bool(getattr(options, "persist", True))
        user = getattr(msg, "user", User())

        # baseline snapshots for this ingress (multi-room)
        room = context.room
        if actor.store is None:
            raise RuntimeError(f"(16) Store is None for actor {getattr(actor, 'grid_id', '?')}")
        baseline_by_room: Dict[str, int] = {room: int(actor.store.committed_seq())}
        leases = SnapshotLeases(self, baseline_by_room)
        actor.acquire_snapshot(baseline_by_room[room])
        leases._leased.append((context, baseline_by_room[room]))

        try:
            ploads = msg.data.payloads
            if ploads is None:
                await log.error("No payloads found.")
                return

            patch = actor.payloads_to_patch(ploads)
            applied = await actor.commit_patch(
                ingress_id=ingress_id,
                based_on=based_on,
                priority=priority,
                source_task="publish",
                patch=patch,
                read_deps=None,
                await_commit=True,
                persist=persist_enabled,
            )
            if applied is None or applied.survivors.is_empty() or not applied.did_commit:
                reason = "empty patch" if applied is None else "conflict resolution dropped all cells"
                await log.error(f"No survivors ({reason}) for {actor.grid_id}, patch_cells={patch.cell_count()}, add={len(ploads.add or [])}, update={len(ploads.update or [])}, remove={len(ploads.remove or [])}")
                return

            survivors_pub = actor.patch_to_payload_publish(
                based_on=based_on,
                action_seq=action_seq,
                patch=applied.survivors,
                options=options,
                trace=trace,
            )

            # Passthrough
            if survivors_pub is not None and not silent:
                yield survivors_pub

            patched_df = actor.patch_to_df(applied.survivors)
            publish_delta_by_room = {room: patched_df}

            async for out_msg in self.rules.run_ingress(
                    system=self,
                    actor_context=context,
                    ingress_id=ingress_id,
                    based_on=based_on,
                    publish_delta_by_room=publish_delta_by_room,
                    primary_keys=tuple(context.primary_keys or actor.store.pk_cols),
                    leases=leases,
                    trace=trace,
                    emit_options=options,
                    trigger_rules=trigger_rules,
                    ingress_user=user
            ):
                if silent: continue
                yield out_msg

        finally:
            leases.release_all()
            await log.notify("DONE INGESTING")

    # =========================================================================
    # Micro-Grid ingestion
    # =========================================================================

    async def ingest_micro_publish(
            self,
            micro_name: str,
            payloads: Dict[str, Any],
            user: str = "unknown",
            trace: Optional[str] = None,
    ) -> AsyncIterator[MicroPublish]:
        """
        Full pipeline for micro-grid edits:
          1. Apply edit to MicroGridActor
          2. Yield MicroPublish delta for broadcast
          3. Run eligible rules inline (no MVCC leasing — micro-grids are simple)
          4. Persist to DB
        """
        from app.services.redux.micro_grid import get_micro_actor

        actor = get_micro_actor(micro_name)
        config = actor.config

        based_on = await self.clock.next()
        action_seq = await self.action_clock.next()

        remove_payloads = payloads.get("remove") or []

        broadcast_payloads = {}
        if remove_payloads:
            pk_col = config.primary_keys[0]
            # remove_payloads is List[pl.DataFrame] — extract all PK values
            removed_ids = []
            for r in ensure_list(remove_payloads):
                if r is None:
                    continue
                if isinstance(r, pl.DataFrame) and pk_col in r.columns:
                    removed_ids.extend(r[pk_col].cast(pl.String).to_list())
                elif isinstance(r, dict):
                    rid = r.get(pk_col)
                    if rid is not None:
                        removed_ids.append(str(rid))
            if removed_ids:
                broadcast_payloads["remove"] = [{pk_col: rid} for rid in removed_ids]

        # 1. Apply edit
        delta_df = await actor.apply_edit(payloads, user=user)
        if (delta_df is not None) and (not delta_df.hyper.is_empty()):

            # 2. Build delta payload for broadcast
            delta_rows = delta_df.to_dicts()
            add_count = getattr(delta_df, '_micro_add_count', 0)

            if add_count > 0:
                broadcast_payloads["add"] = delta_rows[:add_count]
            if add_count < len(delta_rows):
                broadcast_payloads["update"] = delta_rows[add_count:]

        if not broadcast_payloads:
            await log.error('no payloads to broadcast?')
            return

        await log.micro("building micro payload")
        micro_pub = MicroPublish(
            micro_name=micro_name,
            payloads=broadcast_payloads,
            based_on=based_on,
            action_seq=action_seq,
            pk_columns=config.primary_keys,
            trace=trace
        )
        yield micro_pub


        if config.rules_enabled:
            room = config.room
            changed_cols = set(delta_df.columns) - set(config.primary_keys)

            for rdef in self.rules._rules.values():
                try:
                    if not rdef.applies_to_room(room):
                        continue

                    # Check column triggers
                    if rdef.column_triggers_any:
                        if not changed_cols.intersection(rdef.column_triggers_any):
                            continue
                    if rdef.column_triggers_all:
                        if not changed_cols.issuperset(rdef.column_triggers_all):
                            continue

                    # Build a lightweight RuleContext
                    micro_context = RoomContext(
                        room=room,
                        grid_id=config.grid_id,
                        primary_keys=list(config.primary_keys),
                    )
                    target_room = rdef.target_room or room
                    target_pks = rdef.target_primary_keys or config.primary_keys

                    log.micro("building context")
                    ctx = RuleContext(
                        system=self,
                        leases=SnapshotLeases(self, {}),
                        source_context=micro_context,
                        source_room=room,
                        target_room=target_room,
                        target_pks=target_pks,
                        based_on=based_on,
                        ingress_id=action_seq,
                        triggering_delta=delta_df,
                    )

                    # Execute rule
                    log.micro("executing rule")
                    result = await rdef.func(ctx)

                    if result is None:
                        continue

                    # Normalize result to DataFrame
                    if isinstance(result, pl.LazyFrame):
                        result = result.collect()
                    if not isinstance(result, pl.DataFrame) or result.hyper.is_empty():
                        continue

                    # Apply rule output
                    if target_room == room:
                        # Rule outputs to the same micro-grid — apply directly
                        rule_payloads = {"update": [result]}
                        rule_delta = await actor.apply_edit(rule_payloads, user="rule:" + rdef.name)

                        if rule_delta is not None and not rule_delta.hyper.is_empty():
                            rule_pub = MicroPublish(
                                micro_name=micro_name,
                                payloads={"update": rule_delta.to_dicts()},
                                based_on=based_on,
                                action_seq=action_seq,
                                pk_columns=config.primary_keys,
                                trace=trace,
                            )
                            yield rule_pub
                    else:

                        log.micro('output to main grid.')
                        # Rule outputs to a main grid — use normal ingest_publish
                        target_context = RoomContext(
                            room=target_room,
                            grid_id=rdef.target_grid_id or target_room.split(".")[-1].lower(),
                            primary_keys=list(target_pks),
                        )
                        pub = self.build_publish_from_df(
                            context=target_context,
                            df=result,
                            mode="update",
                            trace=trace,
                        )
                        async for out_msg in self.ingest_publish(pub):
                            yield out_msg

                except Exception as e:
                    await log.error(f"[MicroGrid] Rule '{rdef.name}' error for {micro_name}: {e}")

        # 4. Persist
        if config.persist:
            try:
                await actor.persist()
            except Exception as e:
                await log.error(f"[MicroGrid] Persist error for {micro_name}: {e}")

    def list_all_actors(self) -> List[ActorStatus]:
        now = _now_ns()
        out: List[ActorStatus] = []
        for a in self.registry.iter_actors():
            if a is None: continue
            out.append(
                ActorStatus(
                    key=a.key,
                    room=a.room,
                    grid_id=a.grid_id,
                    grid_filters=dict(a.grid_filters or {}),
                    subscriber_count=a.subscriber_count(),
                    hibernating=a.is_hibernating(),
                    last_touched_ns=a.last_touched_ns(),
                    idle_s=max(0.0, (now - a.last_touched_ns()) / 1_000_000_000),
                    commit_seq=a.committed_seq_hint(),
                    apply_q_size=(a._apply_q.qsize() if hasattr(a._apply_q, "qsize") else 0),
                    snapshots_active=a.snapshots.any_active(),
                    lifecycle_gen=a.lifecycle_gen(),
                    closed=bool(a._closed or a._shutdown_started),
                )
            )
        return out

    def list_subscriber_count_per_actor(self) -> Dict[str, int]:
        return {a.key: a.subscriber_count() for a in self.registry.iter_actors() if a is not None}

    def get_actor_status(self, context: RoomContext) -> Optional[ActorStatus]:
        a = self.registry.get(context)
        if a is None:
            return None
        now = _now_ns()
        return ActorStatus(
            key=a.key,
            room=a.room,
            grid_id=a.grid_id,
            grid_filters=dict(a.grid_filters or {}),
            subscriber_count=a.subscriber_count(),
            hibernating=a.is_hibernating(),
            last_touched_ns=a.last_touched_ns(),
            idle_s=max(0.0, (now - a.last_touched_ns()) / 1_000_000_000),
            commit_seq=a.committed_seq_hint(),
            apply_q_size=(a._apply_q.qsize() if hasattr(a._apply_q, "qsize") else 0),
            snapshots_active=a.snapshots.any_active(),
            lifecycle_gen=a.lifecycle_gen(),
            closed=bool(a._closed or a._shutdown_started),
        )

    def list_registry_mailbox_subscriptions(self, context: RoomContext):
        a = self.registry.get(context)
        if a is None: return []
        return self.registry.mailbox.list_subscriptions(a)

    def find_actor_by_key(self, key: str) -> Optional[GridActor]:
        return self.registry.get_by_key(key)

    def find_actors_by_room_pattern(self, room_pattern: str) -> List[GridActor]:
        rp = (room_pattern or "*").upper()
        out = []
        for a in self.registry.iter_actors():
            if a is None: continue
            if fnmatch.fnmatchcase(a.room.upper(), rp):
                out.append(a)
        return out

    async def collect_subscribers_for_room_pattern(self, room_pattern: str, *, dedupe: bool = True) -> List[Any]:
        actors = self.find_actors_by_room_pattern(room_pattern)
        if not actors:
            return []
        subs: List[Any] = []
        if not dedupe:
            for a in actors:
                subs.extend(await a.list_subscribers())
            return subs
        seen: Set[int] = set()
        for a in actors:
            for s in await a.list_subscribers():
                tid = a.get_subscriber_token(s)
                if tid in seen:
                    continue
                seen.add(tid)
                subs.append(s)
        return subs

    async def shutdown(self, *, purge_store: bool = True):
        if self._closed:
            return
        self._closed = True

        await self.stop_reaper()

        if self._remove_tasks:
            tasks = list(self._remove_tasks.values())
            self._remove_tasks.clear()
            for t in tasks:
                t.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)

        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self.registry.shutdown(purge_store=purge_store)

        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self.rules.shutdown()

        # stop persistence last
        if self.persistence is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self.persistence.shutdown()

class GridPersistenceManager:

    def __init__(self, *, registry: "ActorRegistry") -> None:
        self._registry_ref = weakref.ref(registry)
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self._pending: Set[str] = set()
        self._pending_lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self._stop = threading.Event()
        self._fail_count: Dict[str, int] = {}

    def start(self) -> None:
        if self._closed: return
        t = self._task
        if (t is not None) and (not t.done()): return
        self._task = asyncio.create_task(self._loop(), name="grid-persist")

    def notify_dirty(self, actor: "GridActor") -> None:
        if self._closed or (actor is None): return
        self.start()
        key = actor.key
        with self._pending_lock:
            if key in self._pending: return
            self._pending.add(key)
        try:
            self._q.put_nowait(key)
        except Exception:
            with self._pending_lock:
                self._pending.discard(key)

    def _pending_remove(self, key: str) -> None:
        with self._pending_lock:
            self._pending.discard(key)
        self._fail_count.pop(key, None)

    async def shutdown(self) -> None:
        if self._closed: return
        self._closed = True
        self._stop.set()
        t = self._task
        self._task = None
        if t is not None and not t.done():
            t.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(t, return_exceptions=True)

        # C-05/H-13: best-effort flush of remaining queued items before discarding
        reg = self._registry_ref()
        flushed = 0
        failed = 0
        remaining_keys: List[str] = []
        while True:
            try:
                key = self._q.get_nowait()
                remaining_keys.append(key)
                self._q.task_done()
            except asyncio.QueueEmpty:
                break
            except (Exception, asyncio.CancelledError):
                break

        if reg is not None and remaining_keys:
            for key in remaining_keys:
                actor = reg.get_by_key(key)
                if actor is None:
                    continue
                try:
                    ok, _ = await asyncio.wait_for(actor.persist_flush(), timeout=10.0)
                    if ok:
                        flushed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        if remaining_keys:
            await log.info(f"Persistence shutdown: {flushed} queued actors flushed, {failed} failed, {len(remaining_keys)} total")

        with self._pending_lock:
            self._pending.clear()
        self._fail_count.clear()

    async def _loop(self) -> None:
        from app.server import get_db
        db = get_db()

        try:
            while not self._stop.is_set() and not self._closed:
                try:
                    first = await self._q.get()
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(0)
                    continue

                keys = [first]
                # drain more
                for _ in range(_PERSIST_MAX_ACTORS_PER_BATCH - 1):
                    try:
                        keys.append(self._q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                    except Exception:
                        break

                # coalesce per-actor writes into a single flush window
                if _PERSIST_DEBOUNCE_S > 0 and not self._stop.is_set():
                    await asyncio.sleep(_PERSIST_DEBOUNCE_S)

                reg = self._registry_ref()
                if reg is None:
                    # no registry => shut down
                    break

                for key in keys:
                    # mark queue items done even on error
                    with contextlib.suppress(Exception):
                        self._q.task_done()

                    actor = reg.get_by_key(key)
                    if actor is None:
                        self._pending_remove(key)
                        continue

                    ok = True
                    still_dirty = False
                    try:
                        ok, still_dirty = await actor.persist_flush(db=db)
                    except Exception as exc:
                        ok = False
                        still_dirty = True
                        await log.error("persist_flush raised for {key}: {exc}\n{traceback.format_exc()}")

                    if ok and not still_dirty:
                        self._pending_remove(key)
                        continue

                    fc = self._fail_count.get(key, 0)
                    if not ok:
                        fc = min(20, fc + 1)
                        self._fail_count[key] = fc

                        # H-06: max retry cap — stop retrying after 100 consecutive failures
                        if fc >= 100:
                            await log.critical(f"Persistence permanently failed for {key} after {fc} retries, ejecting")
                            self._pending_remove(key)
                            continue

                        delay = _PERSIST_ERROR_BACKOFF_BASE_S * (2.0 ** min(8, fc))
                        if delay > _PERSIST_ERROR_BACKOFF_MAX_S:
                            delay = _PERSIST_ERROR_BACKOFF_MAX_S
                        await asyncio.sleep(delay)
                    else:
                        # clean flush but new dirt arrived during flush; small debounce
                        await asyncio.sleep(_PERSIST_DEBOUNCE_S)

                    # M-02: remove from pending BEFORE re-queuing to prevent blocking
                    if still_dirty and (not self._stop.is_set()):
                        self._pending_remove(key)
                        try:
                            self.notify_dirty(actor)
                        except Exception:
                            await log.warning(f"Failed to requeue dirty actor {key}")
                    else:
                        # not still_dirty: remove from pending so future notify_dirty can enqueue
                        self._pending_remove(key)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await log.critical(f"Persistence loop crashed: {exc}\n{traceback.format_exc()}")
            # C-05: auto-restart with backoff
            if not self._closed and not self._stop.is_set():
                await asyncio.sleep(5.0)
                self.start()


# =============================================================================
# Arrow IPC Query Engine
# =============================================================================
#
# Integrated from arrow_grid_router.py — provides Arrow IPC streaming,
# AG Grid SSRM support, reactive subscriptions, and room-based deltas.
#
# Public API surface:
#   Models:   ColumnVO, SortModel, FilterCondition, RowsRequest, AggregateRequest,
#             ValuesRequest, PivotRequest, CountRequest, DistinctRequest,
#             DescribeRequest, HistogramRequest, PercentileRequest, TopNRequest,
#             GroupByRequest, CrossTabRequest, SearchRequest, ExportRequest
#   Classes:  ArrowQueryEngine, ArrowSubscription, ArrowRoomSubscription,
#             ArrowSubscriptionRegistry, ArrowRoomRegistry
#   Funcs:    ag_filter_to_polars, apply_filter_model, apply_sort_model,
#             normalize_filters, apply_room_filters, row_matches_room_filters,
#             serialize_arrow_ipc, ipc_response, streaming_ipc_response,
#             compute_room_initial, recompute_subscription, content_hash,
#             extract_touched_columns, project_row, load_grid_table
# =============================================================================

import io as _io
import json as _json
import math as _math
import uuid as _uuid_mod
import zlib as _zlib

import pyarrow as pa
import pyarrow.ipc as _ipc

try:
    from pydantic import BaseModel as _BaseModel
except ImportError:
    class _BaseModel:  # type: ignore[no-redef]
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

try:
    import re2 as _re
except ImportError:
    import re as _re

from collections import defaultdict as _defaultdict
from typing import FrozenSet

# ---------------------------------------------------------------------------
# Pydantic models – mirror AG Grid's SSRM request contract
# ---------------------------------------------------------------------------

class ColumnVO(_BaseModel):
    id: str
    displayName: Optional[str] = None
    field: Optional[str] = None
    aggFunc: Optional[str] = None

class SortModel(_BaseModel):
    colId: str
    sort: str = "asc"

class FilterCondition(_BaseModel):
    filterType: Optional[str] = None
    type: Optional[str] = None
    filter: Optional[Any] = None
    filterTo: Optional[Any] = None
    values: Optional[List[Any]] = None
    operator: Optional[str] = None
    conditions: Optional[List["FilterCondition"]] = None

try:
    FilterCondition.model_rebuild()
except Exception:
    pass

class RowsRequest(_BaseModel):
    startRow: int = 0
    endRow: int = 100
    columns: Optional[List[str]] = None
    rowGroupCols: Optional[List[ColumnVO]] = None
    valueCols: Optional[List[ColumnVO]] = None
    pivotCols: Optional[List[ColumnVO]] = None
    pivotMode: bool = False
    groupKeys: Optional[List[Any]] = None
    filterModel: Optional[Dict[str, Any]] = None
    sortModel: Optional[List[SortModel]] = None

class AggregateRequest(_BaseModel):
    columns: List[str]
    functions: List[str]
    filterModel: Optional[Dict[str, Any]] = None
    groupBy: Optional[List[str]] = None

class ValuesRequest(_BaseModel):
    column: str
    filterModel: Optional[Dict[str, Any]] = None
    searchText: Optional[str] = None
    limit: int = 1000

class PivotRequest(_BaseModel):
    pivotColumn: str
    valueColumns: List[str]
    aggFunc: str = "sum"
    rowGroupColumns: Optional[List[str]] = None
    filterModel: Optional[Dict[str, Any]] = None

class CountRequest(_BaseModel):
    filterModel: Optional[Dict[str, Any]] = None
    groupBy: Optional[List[str]] = None

class DistinctRequest(_BaseModel):
    columns: List[str]
    filterModel: Optional[Dict[str, Any]] = None
    limit: int = 10000

class DescribeRequest(_BaseModel):
    columns: Optional[List[str]] = None
    filterModel: Optional[Dict[str, Any]] = None
    percentiles: Optional[List[float]] = None

class HistogramRequest(_BaseModel):
    column: str
    bins: int = 20
    filterModel: Optional[Dict[str, Any]] = None
    range: Optional[List[float]] = None

class PercentileRequest(_BaseModel):
    column: str
    percentiles: List[float] = [0.25, 0.5, 0.75]
    filterModel: Optional[Dict[str, Any]] = None
    groupBy: Optional[List[str]] = None

class TopNRequest(_BaseModel):
    column: str
    n: int = 10
    direction: str = "desc"
    columns: Optional[List[str]] = None
    filterModel: Optional[Dict[str, Any]] = None

class GroupByRequest(_BaseModel):
    groupBy: List[str]
    columns: List[str]
    functions: List[str]
    filterModel: Optional[Dict[str, Any]] = None
    sortBy: Optional[str] = None
    sortDirection: str = "desc"
    limit: Optional[int] = None

class CrossTabRequest(_BaseModel):
    rowColumn: str
    colColumn: str
    valueColumn: Optional[str] = None
    aggFunc: str = "count"
    filterModel: Optional[Dict[str, Any]] = None

class SearchRequest(_BaseModel):
    text: str
    columns: Optional[List[str]] = None
    limit: int = 100
    filterModel: Optional[Dict[str, Any]] = None
    caseSensitive: bool = False

class ExportRequest(_BaseModel):
    format: str = "csv"
    columns: Optional[List[str]] = None
    filterModel: Optional[Dict[str, Any]] = None
    sortModel: Optional[List[SortModel]] = None
    limit: Optional[int] = None

# ---------------------------------------------------------------------------
# Arrow IPC serialisation (LZ4 compressed)
# ---------------------------------------------------------------------------

_IPC_OPTS = _ipc.IpcWriteOptions(compression="lz4_frame")
_ARROW_MEDIA = "application/vnd.apache.arrow.stream"
_CHUNK = 256 * 1024


def serialize_arrow_ipc(table: pa.Table) -> bytes:
    """Serialize a PyArrow table to LZ4-compressed Arrow IPC streaming format."""
    sink = pa.BufferOutputStream()
    with _ipc.new_stream(sink, table.schema, options=_IPC_OPTS) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def ipc_response(table: pa.Table, extra_headers: Optional[Dict[str, str]] = None):
    """Build a Starlette Response with Arrow IPC body."""
    from starlette.responses import Response
    body = serialize_arrow_ipc(table)
    headers = {
        "Content-Length": str(len(body)),
        "X-Row-Count": str(table.num_rows),
        "X-Col-Count": str(table.num_columns),
        "X-Arrow-Compression": "lz4_frame",
    }
    if extra_headers:
        headers.update(extra_headers)
    return Response(content=body, media_type=_ARROW_MEDIA, headers=headers)


def streaming_ipc_response(table: pa.Table):
    """Build a Starlette StreamingResponse with chunked Arrow IPC body."""
    from starlette.responses import StreamingResponse
    buf = serialize_arrow_ipc(table)
    async def _generate():
        mv = memoryview(buf)
        for i in range(0, len(mv), _CHUNK):
            yield bytes(mv[i : i + _CHUNK])
    return StreamingResponse(
        _generate(),
        media_type=_ARROW_MEDIA,
        headers={
            "Content-Length": str(len(buf)),
            "X-Row-Count": str(table.num_rows),
            "X-Col-Count": str(table.num_columns),
            "X-Arrow-Compression": "lz4_frame",
        },
    )


def content_hash(data: bytes) -> int:
    """Fast content hash for equality checks (CRC32 is ~10x faster than MD5)."""
    return _zlib.crc32(data)


# ---------------------------------------------------------------------------
# Filter translation: AG Grid filterModel → Polars expressions
# ---------------------------------------------------------------------------

def ag_filter_to_polars(col: str, filt: dict) -> Optional[pl.Expr]:
    """Convert a single AG Grid filter entry to a Polars expression."""
    # Compound filter (AND/OR of conditions)
    if "operator" in filt and "conditions" in filt:
        sub_exprs = [ag_filter_to_polars(col, c) for c in filt["conditions"]]
        sub_exprs = [e for e in sub_exprs if e is not None]
        if not sub_exprs:
            return None
        combined = sub_exprs[0]
        op = filt["operator"].upper()
        for e in sub_exprs[1:]:
            combined = combined & e if op == "AND" else combined | e
        return combined

    # Nested condition1/condition2 format
    if "condition1" in filt and "condition2" in filt:
        e1 = ag_filter_to_polars(col, filt["condition1"])
        e2 = ag_filter_to_polars(col, filt["condition2"])
        if e1 is None and e2 is None:
            return None
        if e1 is None:
            return e2
        if e2 is None:
            return e1
        op = filt.get("operator", "AND").upper()
        return (e1 & e2) if op == "AND" else (e1 | e2)

    filter_type = filt.get("filterType", "text")
    op = filt.get("type", "equals")
    val = filt.get("filter")
    val_to = filt.get("filterTo")
    values = filt.get("values")
    c = pl.col(col)

    if filter_type == "set" and values is not None:
        return c.is_in(values)

    if filter_type == "multi":
        sub_filters = filt.get("filterModels", [])
        sub_exprs = []
        for sf in sub_filters:
            if sf is not None:
                expr = ag_filter_to_polars(col, sf)
                if expr is not None:
                    sub_exprs.append(expr)
        if not sub_exprs:
            return None
        combined = sub_exprs[0]
        for e in sub_exprs[1:]:
            combined = combined & e
        return combined

    if filter_type == "number":
        if op == "equals":                return c == val
        if op == "notEqual":              return c != val
        if op == "greaterThan":           return c > val
        if op == "greaterThanOrEqual":    return c >= val
        if op == "lessThan":              return c < val
        if op == "lessThanOrEqual":       return c <= val
        if op == "inRange" and val is not None and val_to is not None:
            return (c >= val) & (c <= val_to)
        if op == "blank":                 return c.is_null()
        if op == "notBlank":              return c.is_not_null()

    if filter_type == "text":
        if val is None:
            if op == "blank":    return c.is_null() | (c.cast(pl.Utf8) == "")
            if op == "notBlank": return c.is_not_null() & (c.cast(pl.Utf8) != "")
            return None
        sval = str(val)
        if op == "equals":      return c.cast(pl.Utf8) == sval
        if op == "notEqual":    return c.cast(pl.Utf8) != sval
        if op == "contains":    return c.cast(pl.Utf8).str.contains(sval, literal=True)
        if op == "notContains": return ~c.cast(pl.Utf8).str.contains(sval, literal=True)
        if op == "startsWith":  return c.cast(pl.Utf8).str.starts_with(sval)
        if op == "endsWith":    return c.cast(pl.Utf8).str.ends_with(sval)
        if op == "blank":       return c.is_null() | (c.cast(pl.Utf8) == "")
        if op == "notBlank":    return c.is_not_null() & (c.cast(pl.Utf8) != "")

    if filter_type == "date":
        date_from = filt.get("dateFrom") or val
        date_to = filt.get("dateTo") or val_to
        if op == "equals":                return c == date_from
        if op == "notEqual":              return c != date_from
        if op == "greaterThan":           return c > date_from
        if op == "greaterThanOrEqual":    return c >= date_from
        if op == "lessThan":              return c < date_from
        if op == "lessThanOrEqual":       return c <= date_from
        if op == "inRange" and date_from is not None and date_to is not None:
            return (c >= date_from) & (c <= date_to)
        if op == "blank":                 return c.is_null()
        if op == "notBlank":              return c.is_not_null()

    if filter_type == "boolean":
        if val is True or val == "true" or val == 1:
            return c == True  # noqa: E712
        if val is False or val == "false" or val == 0:
            return c == False  # noqa: E712
        if op == "blank":    return c.is_null()
        if op == "notBlank": return c.is_not_null()

    return None


def apply_filter_model(df: pl.DataFrame, filter_model: Optional[Dict[str, Any]]) -> pl.DataFrame:
    """Apply an AG Grid filterModel dict to a Polars DataFrame."""
    if not filter_model:
        return df
    combined = None
    for col, filt in filter_model.items():
        if col not in df.columns:
            continue
        expr = ag_filter_to_polars(col, filt)
        if expr is not None:
            combined = expr if combined is None else (combined & expr)
    if combined is not None:
        df = df.filter(combined)
    return df


def apply_sort_model(df: pl.DataFrame, sort_model) -> pl.DataFrame:
    """Apply an AG Grid sortModel list to a Polars DataFrame."""
    if not sort_model:
        return df
    models = sort_model
    if isinstance(sort_model, list) and sort_model and isinstance(sort_model[0], dict):
        models = [SortModel(**s) for s in sort_model]
    by = []
    descending = []
    for s in models:
        cid = s.colId if hasattr(s, "colId") else s.get("colId")
        srt = s.sort if hasattr(s, "sort") else s.get("sort", "asc")
        if cid in df.columns:
            by.append(cid)
            descending.append(str(srt).lower() == "desc")
    if by:
        df = df.sort(by, descending=descending, nulls_last=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_AGG_MAP = {
    "sum": "sum", "avg": "mean", "mean": "mean",
    "min": "min", "max": "max", "count": "count",
    "first": "first", "last": "last",
    "std": "std", "var": "var", "median": "median",
    "n_unique": "n_unique",
}


def polars_agg_expr(col: str, func: str) -> pl.Expr:
    """Build a Polars aggregation expression for *col* with *func*."""
    fn = _AGG_MAP.get(func.lower())
    if fn is None:
        raise ValueError(f"Unsupported aggregation function: {func}")
    return getattr(pl.col(col), fn)().alias(f"{col}_{func}")


# ---------------------------------------------------------------------------
# Row grouping for SSRM
# ---------------------------------------------------------------------------

def apply_row_grouping(
        df: pl.DataFrame,
        row_group_cols,
        group_keys: List[Any],
        value_cols,
        start: int,
        end: int,
) -> Tuple[pl.DataFrame, int]:
    depth = len(group_keys)
    for i, key in enumerate(group_keys):
        rc = row_group_cols[i]
        col = (rc.field or rc.id) if hasattr(rc, "field") else (rc.get("field") or rc.get("id"))
        df = df.filter(pl.col(col) == key)

    if depth < len(row_group_cols):
        rc = row_group_cols[depth]
        group_col = (rc.field or rc.id) if hasattr(rc, "field") else (rc.get("field") or rc.get("id"))
        agg_exprs: list[pl.Expr] = [pl.len().alias("__group_count")]
        if value_cols:
            for vc in value_cols:
                vcol = (vc.field or vc.id) if hasattr(vc, "field") else (vc.get("field") or vc.get("id"))
                afn = vc.aggFunc if hasattr(vc, "aggFunc") else vc.get("aggFunc")
                if vcol in df.columns and afn:
                    try:
                        agg_exprs.append(polars_agg_expr(vcol, afn))
                    except ValueError:
                        pass
        grouped = df.group_by(group_col, maintain_order=True).agg(agg_exprs)
        total = grouped.height
        return grouped.slice(start, end - start), total

    total = df.height
    return df.slice(start, end - start), total


def build_pivot(df, pivot_col, value_cols, agg_func, row_group_cols):
    index_cols = row_group_cols or []
    missing = [c for c in [pivot_col] + value_cols + index_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    return df.pivot(
        on=pivot_col,
        index=index_cols if index_cols else None,
        values=value_cols,
        aggregate_function=agg_func,
    )


# ---------------------------------------------------------------------------
# Room filter normalization + operators
# ---------------------------------------------------------------------------

_NEGATE_OP = {
    "eq": "neq", "neq": "eq",
    "gt": "lte", "gte": "lt", "lt": "gte", "lte": "gt",
    "in": "not_in", "not_in": "in",
    "contains": "not_contains", "not_contains": "contains",
    "startswith": "not_startswith", "not_startswith": "startswith",
    "endswith": "not_endswith", "not_endswith": "endswith",
    "is_null": "is_not_null", "is_not_null": "is_null",
    "regex": "not_regex", "like": "not_like", "ilike": "not_ilike",
    "between": "not_between",
}

_SUFFIX_OP_MAP = {
    ">": "gt", ">=": "gte", "<": "lt", "<=": "lte",
    "!=": "neq", "=": "eq", "==": "eq",
    "~": "regex", "^": "startswith", "$": "endswith",
    "*": "contains",
}


def _normalize_filter(entry: dict) -> List[dict]:
    """Normalize a single filter entry into canonical ``{field, op, value}`` dicts.

    Accepted formats:
      Canonical:   ``{"field": "x", "op": "eq", "value": 5}``
      Shorthand:   ``{"price": 5}`` → eq,  ``{"price": None}`` → is_null,
                   ``{"price": [1,2]}`` → in,  ``{"!price": 5}`` → neq
      Explicit op: ``{"price": {"gt": 5}}``,  ``{"price": {"between": [1,10]}}``
      Suffix:      ``{"price >": 5}``,  ``{"name ~": "pat"}``
    """
    if "field" in entry and ("op" in entry or "value" in entry):
        return [entry]

    results = []
    for raw_key, value in entry.items():
        if raw_key in ("field", "op", "value"):
            continue

        negate = False
        key = raw_key
        if key.startswith("!"):
            negate = True
            key = key[1:]
        elif key.startswith("not "):
            negate = True
            key = key[4:]

        suffix_op = None
        for suffix, mapped_op in _SUFFIX_OP_MAP.items():
            if key.endswith(" " + suffix):
                key = key[: -(len(suffix) + 1)]
                suffix_op = mapped_op
                break

        key = key.strip()
        if not key:
            continue

        if suffix_op:
            op, val = suffix_op, value
        elif isinstance(value, dict):
            if len(value) == 1:
                op, val = next(iter(value.items()))
            else:
                for inner_op, inner_val in value.items():
                    effective_op = _NEGATE_OP.get(inner_op, inner_op) if negate else inner_op
                    results.append({"field": key, "op": effective_op, "value": inner_val})
                continue
        elif value is None:
            op, val = "is_null", None
        elif isinstance(value, (list, tuple)):
            op, val = "in", list(value)
        else:
            op, val = "eq", value

        if negate:
            op = _NEGATE_OP.get(op, op)

        results.append({"field": key, "op": op, "value": val})

    return results


def normalize_filters(filters: Optional[List[dict]]) -> Optional[List[dict]]:
    """Normalize a mixed list of filters into canonical ``{field, op, value}`` dicts."""
    if not filters:
        return filters
    out: list[dict] = []
    for entry in filters:
        out.extend(_normalize_filter(entry))
    return out


ROOM_FILTER_OPS = {
    "eq":             lambda c, v: c == v,
    "neq":            lambda c, v: c != v,
    "gt":             lambda c, v: c > v,
    "gte":            lambda c, v: c >= v,
    "lt":             lambda c, v: c < v,
    "lte":            lambda c, v: c <= v,
    "in":             lambda c, v: c.is_in(v) if isinstance(v, list) else c == v,
    "not_in":         lambda c, v: ~c.is_in(v) if isinstance(v, list) else c != v,
    "between":        lambda c, v: (c >= v[0]) & (c <= v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else c == v,
    "contains":       lambda c, v: c.cast(pl.Utf8).str.contains(str(v), literal=True),
    "not_contains":   lambda c, v: ~c.cast(pl.Utf8).str.contains(str(v), literal=True),
    "startswith":     lambda c, v: c.cast(pl.Utf8).str.starts_with(str(v)),
    "not_startswith": lambda c, v: ~c.cast(pl.Utf8).str.starts_with(str(v)),
    "endswith":       lambda c, v: c.cast(pl.Utf8).str.ends_with(str(v)),
    "not_endswith":   lambda c, v: ~c.cast(pl.Utf8).str.ends_with(str(v)),
    "is_null":        lambda c, v: c.is_null(),
    "is_not_null":    lambda c, v: c.is_not_null(),
    "regex":          lambda c, v: c.cast(pl.Utf8).str.contains(str(v), literal=False),
    "not_regex":      lambda c, v: ~c.cast(pl.Utf8).str.contains(str(v), literal=False),
    "like":           lambda c, v: c.cast(pl.Utf8).str.contains(
        str(v).replace("%", ".*").replace("_", "."), literal=False),
    "not_like":       lambda c, v: ~c.cast(pl.Utf8).str.contains(
        str(v).replace("%", ".*").replace("_", "."), literal=False),
    "ilike":          lambda c, v: c.cast(pl.Utf8).str.to_lowercase().str.contains(
        str(v).lower().replace("%", ".*").replace("_", "."), literal=False),
    "not_ilike":      lambda c, v: ~c.cast(pl.Utf8).str.to_lowercase().str.contains(
        str(v).lower().replace("%", ".*").replace("_", "."), literal=False),
    "not_between":    lambda c, v: ~((c >= v[0]) & (c <= v[1])) if isinstance(v, (list, tuple)) and len(v) >= 2 else c != v,
}


def apply_room_filters(df: pl.DataFrame, filters: Optional[List[dict]]) -> pl.DataFrame:
    """Apply room-style filters to a Polars DataFrame."""
    filters = normalize_filters(filters)
    if not filters:
        return df
    col_set = set(df.columns)
    combined = None
    for f in filters:
        fld = f.get("field")
        op = f.get("op", "eq")
        value = f.get("value")
        if fld and fld in col_set:
            fn = ROOM_FILTER_OPS.get(op)
            if fn:
                expr = fn(pl.col(fld), value)
                combined = expr if combined is None else (combined & expr)
    if combined is not None:
        df = df.filter(combined)
    return df


def row_matches_room_filters(row: dict, filters: Optional[List[dict]]) -> bool:
    """Evaluate room filters against a single row dict (fast-path for deltas)."""
    filters = normalize_filters(filters)
    if not filters:
        return True
    for f in filters:
        field_name = f.get("field")
        op = f.get("op", "eq")
        value = f.get("value")
        if op == "is_null":
            if field_name is None:
                continue
            if row.get(field_name) is not None:
                return False
            continue
        if op == "is_not_null":
            if field_name is None:
                continue
            if row.get(field_name) is None:
                return False
            continue
        if field_name is None or field_name not in row:
            continue
        rv = row[field_name]
        if op == "eq"  and not (rv == value):       return False
        if op == "neq" and not (rv != value):       return False
        if op == "gt"  and not (rv > value):        return False
        if op == "gte" and not (rv >= value):       return False
        if op == "lt"  and not (rv < value):        return False
        if op == "lte" and not (rv <= value):       return False
        if op == "in":
            if rv not in (value if isinstance(value, list) else [value]):
                return False
        if op == "not_in":
            if rv in (value if isinstance(value, list) else [value]):
                return False
        if op == "contains" and str(value) not in str(rv):              return False
        if op == "not_contains" and str(value) in str(rv):              return False
        if op == "startswith" and not str(rv).startswith(str(value)):    return False
        if op == "not_startswith" and str(rv).startswith(str(value)):    return False
        if op == "endswith"   and not str(rv).endswith(str(value)):      return False
        if op == "not_endswith" and str(rv).endswith(str(value)):        return False
        if op == "between":
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                if not (value[0] <= rv <= value[1]):
                    return False
            elif rv != value:
                return False
        if op == "not_between":
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                if value[0] <= rv <= value[1]:
                    return False
            elif rv == value:
                return False
        if op == "regex":
            if not _re.search(str(value), str(rv)):
                return False
        if op == "not_regex":
            if _re.search(str(value), str(rv)):
                return False
        if op == "like":
            pattern = str(value).replace("%", ".*").replace("_", ".")
            if not _re.search(pattern, str(rv)):
                return False
        if op == "not_like":
            pattern = str(value).replace("%", ".*").replace("_", ".")
            if _re.search(pattern, str(rv)):
                return False
        if op == "ilike":
            pattern = str(value).lower().replace("%", ".*").replace("_", ".")
            if not _re.search(pattern, str(rv).lower()):
                return False
        if op == "not_ilike":
            pattern = str(value).lower().replace("%", ".*").replace("_", ".")
            if _re.search(pattern, str(rv).lower()):
                return False
    return True


def project_row(row: dict, columns: Optional[List[str]]) -> dict:
    """Return only the requested columns from a row dict."""
    if columns is None:
        return row
    col_set = frozenset(columns)
    return {k: v for k, v in row.items() if k in col_set}


# ---------------------------------------------------------------------------
# Core execution engine — shared by HTTP and WebSocket paths
# ---------------------------------------------------------------------------

def execute_rows(df: pl.DataFrame, params: dict) -> Tuple[bytes, int]:
    """Execute a rows query. Returns (ipc_bytes, total_rows)."""
    columns = params.get("columns")
    filter_model = params.get("filterModel")
    sort_model = params.get("sortModel")
    row_group_cols = params.get("rowGroupCols")
    value_cols = params.get("valueCols")
    group_keys = params.get("groupKeys")
    start_row = params.get("startRow", 0)
    end_row = params.get("endRow", 100)

    col_set = set(df.columns)
    needed: set[str] = set()
    if columns:
        needed.update(c for c in columns if c in col_set)
    if row_group_cols:
        for rc in row_group_cols:
            f = rc.get("field") or rc.get("id") if isinstance(rc, dict) else (rc.field or rc.id)
            if f in col_set: needed.add(f)
    if value_cols:
        for vc in value_cols:
            f = vc.get("field") or vc.get("id") if isinstance(vc, dict) else (vc.field or vc.id)
            if f in col_set: needed.add(f)
    if sort_model:
        for s in sort_model:
            cid = s.get("colId") if isinstance(s, dict) else s.colId
            if cid in col_set: needed.add(cid)
    if filter_model:
        needed.update(c for c in filter_model if c in col_set)
    if needed:
        df = df.select(sorted(needed))

    df = apply_filter_model(df, filter_model)

    if row_group_cols and group_keys is not None:
        result_df, total = apply_row_grouping(df, row_group_cols, group_keys, value_cols, start_row, end_row)
    else:
        df = apply_sort_model(df, sort_model)
        total = df.height
        result_df = df.slice(start_row, end_row - start_row)

    if columns:
        keep = [c for c in columns if c in result_df.columns]
        for c in result_df.columns:
            if c not in keep: keep.append(c)
        result_df = result_df.select(keep)

    return serialize_arrow_ipc(result_df.to_arrow()), total


def execute_values(df: pl.DataFrame, params: dict) -> bytes:
    column = params["column"]
    filter_model = params.get("filterModel")
    search_text = params.get("searchText")
    limit = params.get("limit", 1000)
    if column not in df.columns:
        return serialize_arrow_ipc(pa.table({"values": pa.array([], type=pa.utf8())}))
    df = apply_filter_model(df, filter_model)
    series = df[column].drop_nulls().unique().sort()
    if search_text:
        st = search_text.lower()
        series = series.filter(series.cast(pl.Utf8).str.to_lowercase().str.contains(st, literal=True))
    if limit and limit > 0:
        series = series.head(limit)
    return serialize_arrow_ipc(pa.table({"values": series.to_arrow()}))


def execute_aggregate(df: pl.DataFrame, params: dict) -> bytes:
    columns = params["columns"]
    functions = params["functions"]
    filter_model = params.get("filterModel")
    group_by = params.get("groupBy")
    df = apply_filter_model(df, filter_model)
    agg_exprs: list[pl.Expr] = []
    for col in columns:
        if col not in df.columns: continue
        for func in functions:
            try: agg_exprs.append(polars_agg_expr(col, func))
            except ValueError: pass
    if not agg_exprs:
        return serialize_arrow_ipc(pa.table({}))
    if group_by:
        valid = [g for g in group_by if g in df.columns]
        if valid:
            result = df.group_by(valid, maintain_order=True).agg(agg_exprs)
        else:
            result = df.select(agg_exprs)
    else:
        result = df.select(agg_exprs)
    return serialize_arrow_ipc(result.to_arrow())


def execute_pivot(df: pl.DataFrame, params: dict) -> bytes:
    filter_model = params.get("filterModel")
    df = apply_filter_model(df, filter_model)
    result = build_pivot(
        df, params["pivotColumn"], params["valueColumns"],
        params.get("aggFunc", "sum"), params.get("rowGroupColumns"),
    )
    return serialize_arrow_ipc(result.to_arrow())


def execute_count(df: pl.DataFrame, params: dict) -> bytes:
    filter_model = params.get("filterModel")
    group_by = params.get("groupBy")
    df = apply_filter_model(df, filter_model)
    if group_by:
        valid = [g for g in group_by if g in df.columns]
        if valid:
            result = df.group_by(valid, maintain_order=True).agg(pl.len().alias("count"))
            return serialize_arrow_ipc(result.to_arrow())
    return serialize_arrow_ipc(pa.table({"count": [df.height]}))


def execute_distinct(df: pl.DataFrame, params: dict) -> bytes:
    columns = params.get("columns", [])
    filter_model = params.get("filterModel")
    limit = params.get("limit", 10000)
    df = apply_filter_model(df, filter_model)
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return serialize_arrow_ipc(pa.table({}))
    result = df.select(valid).unique()
    if limit and limit > 0:
        result = result.head(limit)
    return serialize_arrow_ipc(result.to_arrow())


def execute_describe(df: pl.DataFrame, params: dict) -> bytes:
    columns = params.get("columns")
    filter_model = params.get("filterModel")
    percentiles_list = params.get("percentiles") or [0.25, 0.5, 0.75]
    df = apply_filter_model(df, filter_model)
    if columns:
        valid = [c for c in columns if c in df.columns]
    else:
        valid = [c for c in df.columns if df[c].dtype.is_numeric()]
    if not valid:
        return serialize_arrow_ipc(pa.table({}))
    stats_rows = []
    for col_name in valid:
        s = df[col_name]
        row: dict = {"column": col_name, "count": s.len(), "null_count": s.null_count()}
        if s.dtype.is_numeric():
            row.update({"mean": s.mean(), "std": s.std(), "min": s.min(),
                        "max": s.max(), "median": s.median()})
            for p in percentiles_list:
                row[f"p{int(p * 100)}"] = s.quantile(p)
        elif s.dtype == pl.Utf8 or s.dtype == pl.String:
            row.update({"mean": None, "std": None, "median": None, "n_unique": s.n_unique()})
            non_null = s.drop_nulls()
            if non_null.len() > 0:
                sorted_s = non_null.sort()
                row["min"], row["max"] = sorted_s[0], sorted_s[-1]
            else:
                row["min"], row["max"] = None, None
        else:
            row.update({"mean": None, "std": None, "min": None, "max": None, "median": None})
        stats_rows.append(row)
    return serialize_arrow_ipc(pl.DataFrame(stats_rows).to_arrow())


def execute_histogram(df: pl.DataFrame, params: dict) -> bytes:
    column = params.get("column", "")
    bins = params.get("bins", 20)
    filter_model = params.get("filterModel")
    range_override = params.get("range")
    df = apply_filter_model(df, filter_model)
    if column not in df.columns:
        return serialize_arrow_ipc(pa.table({}))
    series = df[column].drop_nulls()
    if series.len() == 0:
        return serialize_arrow_ipc(pa.table({"bin_start": [], "bin_end": [], "count": []}))
    if range_override and len(range_override) >= 2:
        lo, hi = float(range_override[0]), float(range_override[1])
    else:
        lo, hi = float(series.min()), float(series.max())
    if lo == hi:
        return serialize_arrow_ipc(pa.table({"bin_start": [lo], "bin_end": [lo], "count": [series.len()]}))
    bin_width = (hi - lo) / bins
    bin_indices = ((series - lo) / bin_width).cast(pl.Int64).clip(0, bins - 1)
    counts_df = bin_indices.alias("__bin").to_frame().group_by("__bin").agg(pl.len().alias("count"))
    counts_map = dict(zip(counts_df["__bin"].to_list(), counts_df["count"].to_list()))
    bin_starts, bin_ends, counts = [], [], []
    for i in range(bins):
        bin_starts.append(round(lo + i * bin_width, 10))
        bin_ends.append(round(lo + (i + 1) * bin_width, 10))
        counts.append(counts_map.get(i, 0))
    return serialize_arrow_ipc(pa.table({"bin_start": bin_starts, "bin_end": bin_ends, "count": counts}))


def execute_percentile(df: pl.DataFrame, params: dict) -> bytes:
    column = params.get("column", "")
    percentiles_list = params.get("percentiles", [0.25, 0.5, 0.75])
    filter_model = params.get("filterModel")
    group_by = params.get("groupBy")
    df = apply_filter_model(df, filter_model)
    if column not in df.columns:
        return serialize_arrow_ipc(pa.table({}))
    if group_by:
        valid_groups = [g for g in group_by if g in df.columns]
        if valid_groups:
            agg_exprs = [pl.col(column).quantile(p).alias(f"p{int(p * 100)}") for p in percentiles_list]
            result = df.group_by(valid_groups, maintain_order=True).agg(agg_exprs)
            return serialize_arrow_ipc(result.to_arrow())
    s = df[column]
    row = {f"p{int(p * 100)}": s.quantile(p) for p in percentiles_list}
    return serialize_arrow_ipc(pl.DataFrame([row]).to_arrow())


def execute_top_n(df: pl.DataFrame, params: dict) -> bytes:
    column = params.get("column", "")
    n = params.get("n", 10)
    direction = params.get("direction", "desc")
    proj_columns = params.get("columns")
    filter_model = params.get("filterModel")
    df = apply_filter_model(df, filter_model)
    if column not in df.columns:
        return serialize_arrow_ipc(pa.table({}))
    df = df.sort(column, descending=direction.lower() == "desc", nulls_last=True).head(n)
    if proj_columns:
        valid = [c for c in proj_columns if c in df.columns]
        if column not in valid: valid.insert(0, column)
        df = df.select(valid)
    return serialize_arrow_ipc(df.to_arrow())


def execute_group_by(df: pl.DataFrame, params: dict) -> bytes:
    group_by = params.get("groupBy", [])
    columns = params.get("columns", [])
    functions = params.get("functions", [])
    filter_model = params.get("filterModel")
    sort_by = params.get("sortBy")
    sort_direction = params.get("sortDirection", "desc")
    limit = params.get("limit")
    df = apply_filter_model(df, filter_model)
    valid_groups = [g for g in group_by if g in df.columns]
    if not valid_groups:
        return serialize_arrow_ipc(pa.table({}))
    agg_exprs: list[pl.Expr] = []
    for col_name in columns:
        if col_name not in df.columns: continue
        for func in functions:
            try: agg_exprs.append(polars_agg_expr(col_name, func))
            except ValueError: pass
    if not agg_exprs:
        agg_exprs = [pl.len().alias("count")]
    result = df.group_by(valid_groups, maintain_order=True).agg(agg_exprs)
    if sort_by and sort_by in result.columns:
        result = result.sort(sort_by, descending=sort_direction.lower() == "desc", nulls_last=True)
    if limit and limit > 0:
        result = result.head(limit)
    return serialize_arrow_ipc(result.to_arrow())


def execute_cross_tab(df: pl.DataFrame, params: dict) -> bytes:
    row_column = params.get("rowColumn", "")
    col_column = params.get("colColumn", "")
    value_column = params.get("valueColumn")
    agg_func = params.get("aggFunc", "count")
    filter_model = params.get("filterModel")
    df = apply_filter_model(df, filter_model)
    if row_column not in df.columns or col_column not in df.columns:
        return serialize_arrow_ipc(pa.table({}))
    if value_column and value_column in df.columns:
        result = df.pivot(on=col_column, index=[row_column], values=[value_column], aggregate_function=agg_func)
    else:
        df = df.with_columns(pl.lit(1).alias("__ct_val"))
        result = df.pivot(on=col_column, index=[row_column], values=["__ct_val"], aggregate_function="sum")
    return serialize_arrow_ipc(result.to_arrow())


def execute_search(df: pl.DataFrame, params: dict) -> bytes:
    text = params.get("text", "")
    columns = params.get("columns")
    limit = params.get("limit", 100)
    filter_model = params.get("filterModel")
    case_sensitive = params.get("caseSensitive", False)
    df = apply_filter_model(df, filter_model)
    if not text:
        return serialize_arrow_ipc(df.head(limit).to_arrow())
    search_cols = columns if columns else df.columns
    valid = [c for c in search_cols if c in df.columns]
    if not valid:
        return serialize_arrow_ipc(pa.table({}))
    search_text = text if case_sensitive else text.lower()
    combined = None
    for col_name in valid:
        col_str = pl.col(col_name).cast(pl.Utf8)
        if not case_sensitive:
            col_str = col_str.str.to_lowercase()
        expr = col_str.str.contains(search_text, literal=True)
        combined = expr if combined is None else (combined | expr)
    if combined is not None:
        df = df.filter(combined)
    if limit and limit > 0:
        df = df.head(limit)
    return serialize_arrow_ipc(df.to_arrow())


def execute_export(df: pl.DataFrame, params: dict) -> Tuple[bytes, str]:
    """Export data as CSV, JSON, Parquet, or Arrow IPC.

    Returns (content_bytes, media_type).
    """
    fmt = params.get("format", "csv").lower()
    filter_model = params.get("filterModel")
    sort_model = params.get("sortModel")
    columns = params.get("columns")
    limit = params.get("limit")

    df = apply_filter_model(df, filter_model)
    if sort_model:
        df = apply_sort_model(df, sort_model)
    if columns:
        valid = [c for c in columns if c in df.columns]
        if valid:
            df = df.select(valid)
    if limit and limit > 0:
        df = df.head(limit)

    if fmt == "csv":
        buf = _io.BytesIO()
        df.write_csv(buf)
        return buf.getvalue(), "text/csv"
    elif fmt == "json":
        rows = df.to_dicts()
        return _json.dumps(rows, default=str).encode(), "application/json"
    elif fmt == "parquet":
        buf = _io.BytesIO()
        df.write_parquet(buf)
        return buf.getvalue(), "application/octet-stream"
    else:
        return serialize_arrow_ipc(df.to_arrow()), "application/vnd.apache.arrow.stream"


# ---------------------------------------------------------------------------
# Column dependency tracking for reactive subscriptions
# ---------------------------------------------------------------------------

def extract_touched_columns(sub_type: str, params: dict) -> FrozenSet[str]:
    """Determine which data columns a subscription depends on."""
    cols: set[str] = set()
    fm_key = "filterModel"
    if sub_type == "rows":
        if params.get("columns"):       cols.update(params["columns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
        if params.get("sortModel"):
            for s in params["sortModel"]:
                cols.add(s.get("colId", "") if isinstance(s, dict) else s.colId)
    elif sub_type == "values":
        if params.get("column"):         cols.add(params["column"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "aggregate":
        if params.get("columns"):        cols.update(params["columns"])
        if params.get("groupBy"):        cols.update(params["groupBy"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "pivot":
        if params.get("pivotColumn"):    cols.add(params["pivotColumn"])
        if params.get("valueColumns"):   cols.update(params["valueColumns"])
        if params.get("rowGroupColumns"):cols.update(params["rowGroupColumns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "count":
        if params.get("groupBy"):        cols.update(params["groupBy"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "distinct":
        if params.get("columns"):        cols.update(params["columns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "describe":
        if params.get("columns"):        cols.update(params["columns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "histogram":
        if params.get("column"):         cols.add(params["column"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "percentile":
        if params.get("column"):         cols.add(params["column"])
        if params.get("groupBy"):        cols.update(params["groupBy"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "top_n":
        if params.get("column"):         cols.add(params["column"])
        if params.get("columns"):        cols.update(params["columns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "group_by":
        if params.get("groupBy"):        cols.update(params["groupBy"])
        if params.get("columns"):        cols.update(params["columns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "cross_tab":
        if params.get("rowColumn"):      cols.add(params["rowColumn"])
        if params.get("colColumn"):      cols.add(params["colColumn"])
        if params.get("valueColumn"):    cols.add(params["valueColumn"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    elif sub_type == "search":
        if params.get("columns"):        cols.update(params["columns"])
        if params.get(fm_key):           cols.update(params[fm_key].keys())
    return frozenset(cols) if cols else frozenset()


# ---------------------------------------------------------------------------
# Subscription + Room subscription dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArrowSubscription:
    """A single reactive Arrow IPC subscription owned by a WebSocket client."""
    sub_id: str
    sub_type: str                      # "rows" | "values" | "aggregate" | etc.
    params: Dict[str, Any]
    reactive: bool = True
    touched_columns: FrozenSet[str] = field(default_factory=frozenset)
    last_hash: int = 0
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class ArrowRoomSubscription:
    """A room-based subscription with filter+column projection."""
    sub_id: str
    room: str
    grid_id: str
    columns: Optional[List[str]] = None
    filters: Optional[List[dict]] = field(default_factory=list)
    reactive: bool = True
    single_row: bool = False
    row_key_field: Optional[str] = None
    last_hash: int = 0
    created_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# ArrowSubscriptionRegistry — tracks reactive Arrow IPC query subscriptions
# ---------------------------------------------------------------------------

class ArrowSubscriptionRegistry:
    """Per-connection-manager registry for reactive Arrow IPC subscriptions.

    Provides O(1) lookup of which subscriptions are affected when a set
    of columns changes, via a column→subscription index.
    """

    def __init__(self):
        self._clients: Dict[str, Any] = {}                         # client_id → ws
        self._subs: Dict[str, Dict[str, ArrowSubscription]] = _defaultdict(dict)
        self._col_index: Dict[str, Set[Tuple[str, str]]] = _defaultdict(set)
        self._wildcard_subs: Set[Tuple[str, str]] = set()
        self._lock = asyncio.Lock()

    async def register_client(self, client_id: str, ws):
        async with self._lock:
            self._clients[client_id] = ws
            self._subs[client_id] = {}

    async def unregister_client(self, client_id: str):
        async with self._lock:
            subs = self._subs.pop(client_id, {})
            self._clients.pop(client_id, None)
            for sub in subs.values():
                self._remove_from_index(client_id, sub)

    async def add_subscription(self, client_id: str, sub: ArrowSubscription):
        async with self._lock:
            old = self._subs[client_id].pop(sub.sub_id, None)
            if old:
                self._remove_from_index(client_id, old)
            self._subs[client_id][sub.sub_id] = sub
            self._add_to_index(client_id, sub)

    async def remove_subscription(self, client_id: str, sub_id: str) -> bool:
        async with self._lock:
            sub = self._subs.get(client_id, {}).pop(sub_id, None)
            if sub:
                self._remove_from_index(client_id, sub)
                return True
            return False

    async def update_subscription(self, client_id: str, sub_id: str, new_params: dict) -> Optional[ArrowSubscription]:
        async with self._lock:
            subs = self._subs.get(client_id, {})
            old = subs.get(sub_id)
            if not old:
                return None
            self._remove_from_index(client_id, old)
            merged = {**old.params, **new_params}
            touched = extract_touched_columns(old.sub_type, merged)
            updated = ArrowSubscription(
                sub_id=sub_id, sub_type=old.sub_type, params=merged,
                reactive=old.reactive, touched_columns=touched,
                last_hash=old.last_hash, created_at=old.created_at,
            )
            subs[sub_id] = updated
            self._add_to_index(client_id, updated)
            return updated

    async def get_affected(self, changed_columns: Optional[Set[str]] = None) -> List[Tuple[str, ArrowSubscription, Any]]:
        """Return (client_id, subscription, ws) tuples affected by column changes."""
        async with self._lock:
            affected: List[Tuple[str, ArrowSubscription, Any]] = []
            seen: Set[Tuple[str, str]] = set()
            for cid, sid in self._wildcard_subs:
                ws = self._clients.get(cid)
                sub = self._subs.get(cid, {}).get(sid)
                if ws and sub and sub.reactive:
                    affected.append((cid, sub, ws))
                    seen.add((cid, sid))
            if changed_columns:
                for col in changed_columns:
                    for cid, sid in self._col_index.get(col, set()):
                        if (cid, sid) in seen: continue
                        ws = self._clients.get(cid)
                        sub = self._subs.get(cid, {}).get(sid)
                        if ws and sub and sub.reactive:
                            affected.append((cid, sub, ws))
                            seen.add((cid, sid))
            else:
                for cid, subs_dict in self._subs.items():
                    ws = self._clients.get(cid)
                    if not ws: continue
                    for sid, sub in subs_dict.items():
                        if (cid, sid) not in seen and sub.reactive:
                            affected.append((cid, sub, ws))
                            seen.add((cid, sid))
            return affected

    async def remove_all_subscriptions(self, client_id: str) -> int:
        async with self._lock:
            subs = self._subs.get(client_id, {})
            count = len(subs)
            for sub in subs.values():
                self._remove_from_index(client_id, sub)
            subs.clear()
            return count

    async def get_client_subs(self, client_id: str) -> Dict[str, ArrowSubscription]:
        async with self._lock:
            return dict(self._subs.get(client_id, {}))

    async def get_client_ws(self, client_id: str):
        async with self._lock:
            return self._clients.get(client_id)

    def _add_to_index(self, client_id: str, sub: ArrowSubscription):
        if sub.touched_columns:
            for col in sub.touched_columns:
                self._col_index[col].add((client_id, sub.sub_id))
        elif sub.reactive:
            self._wildcard_subs.add((client_id, sub.sub_id))

    def _remove_from_index(self, client_id: str, sub: ArrowSubscription):
        key = (client_id, sub.sub_id)
        self._wildcard_subs.discard(key)
        if sub.touched_columns:
            for col in sub.touched_columns:
                s = self._col_index.get(col)
                if s:
                    s.discard(key)
                    if not s: del self._col_index[col]


# ---------------------------------------------------------------------------
# ArrowRoomRegistry — tracks room-based subscriptions with sparse deltas
# ---------------------------------------------------------------------------

class ArrowRoomRegistry:
    """Per-connection-manager registry for room-based subscriptions.

    Indices: room → set of (client_id, sub_id),
             grid_id → set of (client_id, sub_id).
    """

    def __init__(self):
        self._clients: Dict[str, Any] = {}
        self._subs: Dict[str, Dict[str, ArrowRoomSubscription]] = _defaultdict(dict)
        self._room_index: Dict[str, Set[Tuple[str, str]]] = _defaultdict(set)
        self._grid_index: Dict[str, Set[Tuple[str, str]]] = _defaultdict(set)
        self._lock = asyncio.Lock()

    async def register_client(self, client_id: str, ws):
        async with self._lock:
            self._clients[client_id] = ws
            self._subs[client_id] = {}

    async def unregister_client(self, client_id: str):
        async with self._lock:
            subs = self._subs.pop(client_id, {})
            self._clients.pop(client_id, None)
            for sub in subs.values():
                self._remove_from_index(client_id, sub)

    async def join_room(self, client_id: str, sub: ArrowRoomSubscription):
        async with self._lock:
            old = self._subs[client_id].pop(sub.sub_id, None)
            if old:
                self._remove_from_index(client_id, old)
            self._subs[client_id][sub.sub_id] = sub
            self._add_to_index(client_id, sub)

    async def leave_room(self, client_id: str, sub_id: str) -> bool:
        async with self._lock:
            sub = self._subs.get(client_id, {}).pop(sub_id, None)
            if sub:
                self._remove_from_index(client_id, sub)
                return True
            return False

    async def update_room(self, client_id: str, sub_id: str,
                          filters=None, columns=None) -> Optional[ArrowRoomSubscription]:
        async with self._lock:
            subs = self._subs.get(client_id, {})
            old = subs.get(sub_id)
            if not old:
                return None
            self._remove_from_index(client_id, old)
            updated = ArrowRoomSubscription(
                sub_id=sub_id, room=old.room, grid_id=old.grid_id,
                columns=columns if columns is not None else old.columns,
                filters=filters if filters is not None else old.filters,
                reactive=old.reactive, single_row=old.single_row,
                row_key_field=old.row_key_field, last_hash=0, created_at=old.created_at,
            )
            subs[sub_id] = updated
            self._add_to_index(client_id, updated)
            return updated

    async def get_room_subs(self, room: str) -> List[Tuple[str, ArrowRoomSubscription, Any]]:
        async with self._lock:
            out = []
            for cid, sid in self._room_index.get(room, set()):
                ws = self._clients.get(cid)
                sub = self._subs.get(cid, {}).get(sid)
                if ws and sub and sub.reactive:
                    out.append((cid, sub, ws))
            return out

    async def get_client_room_subs(self, client_id: str) -> Dict[str, ArrowRoomSubscription]:
        async with self._lock:
            return dict(self._subs.get(client_id, {}))

    def _add_to_index(self, cid: str, sub: ArrowRoomSubscription):
        key = (cid, sub.sub_id)
        self._room_index[sub.room].add(key)
        self._grid_index[sub.grid_id].add(key)

    def _remove_from_index(self, cid: str, sub: ArrowRoomSubscription):
        key = (cid, sub.sub_id)
        s = self._room_index.get(sub.room)
        if s:
            s.discard(key)
            if not s: del self._room_index[sub.room]
        s = self._grid_index.get(sub.grid_id)
        if s:
            s.discard(key)
            if not s: del self._grid_index[sub.grid_id]


# ---------------------------------------------------------------------------
# ArrowQueryEngine — unified query dispatch
# ---------------------------------------------------------------------------

class ArrowQueryEngine:
    """Stateless query execution engine for Arrow IPC data access.
    Maps subscription types to execute functions and provides
    helper methods for recomputation and initial room data.
    """

    EXECUTORS = {
        "rows": execute_rows,
        "values": execute_values,
        "aggregate": execute_aggregate,
        "pivot": execute_pivot,
        "count": execute_count,
        "distinct": execute_distinct,
        "describe": execute_describe,
        "histogram": execute_histogram,
        "percentile": execute_percentile,
        "top_n": execute_top_n,
        "group_by": execute_group_by,
        "cross_tab": execute_cross_tab,
        "search": execute_search,
        "export": execute_export,
    }

    @classmethod
    def execute(cls, df: pl.DataFrame, sub_type: str, params: dict):
        """Execute a query. Returns (ipc_bytes,) or (ipc_bytes, total) for rows."""
        executor = cls.EXECUTORS.get(sub_type)
        if executor is None:
            return serialize_arrow_ipc(pa.table({}))
        return executor(df, params)

    @classmethod
    def recompute_subscription(cls, df: pl.DataFrame, sub: ArrowSubscription) -> Tuple[bytes, dict]:
        """Recompute a subscription's result against *df*."""
        meta: dict = {}
        result = cls.execute(df, sub.sub_type, sub.params)
        if isinstance(result, tuple):
            ipc_bytes, total = result
            meta["totalRows"] = total
        else:
            ipc_bytes = result
        return ipc_bytes, meta

    @classmethod
    def compute_room_initial(cls, df: pl.DataFrame, sub: ArrowRoomSubscription) -> Tuple[bytes, int]:
        """Compute initial data for a room subscription."""
        df = apply_room_filters(df, sub.filters)
        total = df.height
        if sub.columns:
            valid = [c for c in sub.columns if c in df.columns]
            if sub.row_key_field and sub.row_key_field in df.columns and sub.row_key_field not in valid:
                valid.insert(0, sub.row_key_field)
            if valid:
                df = df.select(valid)
        if sub.single_row:
            df = df.head(1)
        return serialize_arrow_ipc(df.to_arrow()), total
