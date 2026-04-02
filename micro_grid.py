

from __future__ import annotations

import asyncio
import uuid
import weakref
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from app.helpers.type_helpers import ensure_list

import polars as pl
from app.logs.logging import log

# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class MicroGridConfig:
    """Immutable definition of a single micro-grid."""
    name: str                                       # e.g. "hot_tickers"
    table_name: str                                 # DB table, e.g. "micro_hot_tickers"
    primary_keys: Tuple[str, ...]                   # e.g. ("id",)
    columns: Dict[str, Any]                         # {col_name: default_value} — schema
    column_types: Dict[str, pl.DataType | Type]            # polars types per column
    rules_enabled: bool = True
    persist: bool = True

    @property
    def room(self) -> str:
        return f"MICRO.{self.name.upper()}"

    @property
    def grid_id(self) -> str:
        return f"micro_{self.name}"

    @property
    def schema(self) -> Dict[str, pl.DataType]:
        return dict(self.column_types)


@dataclass(frozen=True)
class MicroGridGroup:
    """Groups multiple micro-grids as tabs in a single modal."""
    name: str                                       # e.g. "pt_tools"
    display_name: str                               # e.g. "Portfolio Tools"
    grids: Tuple[str, ...]                          # ordered micro-grid names

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "grids": list(self.grids),
        }


# =============================================================================
# Registry — singleton
# =============================================================================

class MicroGridRegistry:
    """Central registry for micro-grid configs and groups."""

    def __init__(self) -> None:
        self._configs: Dict[str, MicroGridConfig] = {}
        self._groups: Dict[str, MicroGridGroup] = {}
        self._actors: Dict[str, MicroGridActor] = {}

    # -- configs --

    def register(self, config: MicroGridConfig) -> None:
        if config.name in self._configs:
            raise ValueError(f"Micro-grid already registered: {config.name}")
        self._configs[config.name] = config

    def get_config(self, name: str) -> MicroGridConfig:
        cfg = self._configs.get(name)
        if cfg is None:
            raise KeyError(f"Unknown micro-grid: {name}")
        return cfg

    def list_configs(self) -> List[MicroGridConfig]:
        return list(self._configs.values())

    # -- groups --

    def register_group(self, group: MicroGridGroup) -> None:
        if group.name in self._groups:
            raise ValueError(f"Micro-grid group already registered: {group.name}")
        for g in group.grids:
            if g not in self._configs:
                raise ValueError(f"Micro-grid group '{group.name}' references unknown grid: {g}")
        self._groups[group.name] = group

    def get_group(self, name: str) -> MicroGridGroup:
        grp = self._groups.get(name)
        if grp is None:
            raise KeyError(f"Unknown micro-grid group: {name}")
        return grp

    def list_groups(self) -> List[MicroGridGroup]:
        return list(self._groups.values())

    # -- actors --

    def get_actor(self, name: str) -> MicroGridActor:
        actor = self._actors.get(name)
        if actor is None:
            raise KeyError(f"Micro-grid actor not initialized: {name}")
        return actor

    def set_actor(self, name: str, actor: MicroGridActor) -> None:
        self._actors[name] = actor

    async def shutdown(self) -> None:
        for actor in self._actors.values():
            try:
                if actor._dirty:
                    await actor.persist()
            except Exception as e:
                await log.error(f"[MicroGrid] Error persisting {actor.config.name} on shutdown: {e}")
        self._actors.clear()


# Module-level singleton
_registry = MicroGridRegistry()


def get_micro_registry() -> MicroGridRegistry:
    return _registry


def register_micro_grid(config: MicroGridConfig) -> None:
    _registry.register(config)


def register_micro_grid_group(group: MicroGridGroup) -> None:
    _registry.register_group(group)


def get_micro_actor(name: str) -> MicroGridActor:
    return _registry.get_actor(name)


def get_micro_group(name: str) -> MicroGridGroup:
    return _registry.get_group(name)


# =============================================================================
# MicroGridActor — lightweight in-memory actor
# =============================================================================

class MicroGridActor:
    """
    Simplified actor for a single global micro-grid.
    No MVCC, no slices, no filters, no hibernation.
    Uses an asyncio.Lock to serialize mutations.
    """

    __slots__ = (
        "config", "_data", "_subscribers", "_dirty",
        "_lock", "_loaded", "_persist_dirty_ids",
        "_persist_deleted_ids",
    )

    def __init__(self, config: MicroGridConfig) -> None:
        self.config = config
        self._data: pl.DataFrame = pl.DataFrame(
            schema={col: config.column_types[col] for col in config.columns}
        )
        self._subscribers: Dict[str, weakref.ref] = {}
        self._dirty: bool = False
        self._lock = asyncio.Lock()
        self._loaded: bool = False
        self._persist_dirty_ids: Set[str] = set()
        self._persist_deleted_ids: Set[str] = set()

    # -- DB lifecycle --

    async def load_from_db(self) -> None:
        from app.server import get_db
        db = get_db()
        table_name = self.config.table_name

        exists = await db._table_exists(table_name)
        if not exists:
            if self.config.persist:
                await log.warning(f"Creating a new micro-table: {self.config.table_name}")
                await self._create_table(db)
            self._loaded = True
            return

        try:
            df = await db.select(table_name)
            if df is not None and not df.hyper.is_empty():
                # Align schema — only keep columns we know about
                known = set(self.config.columns.keys())
                keep = [c for c in df.columns if c in known]
                if keep:
                    df = df.select(keep)
                    # Cast to expected types
                    casts = {}
                    for col in df.columns:
                        expected = self.config.column_types.get(col)
                        if expected and df[col].dtype != expected:
                            casts[col] = expected
                    if casts:
                        df = df.cast(casts, strict=False)
                    self._data = df
                await log.info(f"[MicroGrid] Loaded {self._data.hyper.height()} rows from {table_name}")
            else:
                await log.info(f"[MicroGrid] Table {table_name} exists but is empty")
        except Exception as e:
            await log.error(f"[MicroGrid] Failed to load from {table_name}: {e}")

        self._loaded = True

    async def _create_table(self, db) -> None:
        """Create the backing table with the configured schema."""
        schema_df = pl.DataFrame(
            schema={col: self.config.column_types[col] for col in self.config.columns}
        )
        try:
            await db.create(
                self.config.table_name,
                schema_df,
                on_exists="replace",
                primary_keys=list(self.config.primary_keys),
            )
            await log.info(f"[MicroGrid] Created table: {self.config.table_name}")
        except Exception as e:
            await log.error(f"[MicroGrid] Failed to create table {self.config.table_name}: {e}")

    async def persist(self) -> None:
        """Flush dirty rows to the database."""
        if not self.config.persist: return
        from app.server import get_db
        db = get_db()

        dirty_ids = set()
        deleted_ids = set()
        data_snapshot = None

        try:
            async with self._lock:
                if not self._dirty: return
                dirty_ids = set(self._persist_dirty_ids)
                deleted_ids = set(self._persist_deleted_ids)
                self._persist_dirty_ids.clear()
                self._persist_deleted_ids.clear()
                # _dirty stays True until DB ops succeed
                data_snapshot = self._data.clone()

            # Delete removed rows
            if deleted_ids:
                pk_col = self.config.primary_keys[0]
                for rid in deleted_ids:
                    try:
                        await db.remove(
                            self.config.table_name,
                            filters={pk_col: rid},
                        )
                    except Exception as e:
                        await log.error(f"[MicroGrid] Delete failed for {rid}: {e}")

            # Upsert dirty rows
            if dirty_ids:
                pk_col = self.config.primary_keys[0]
                dirty_df = data_snapshot.filter(pl.col(pk_col).is_in(list(dirty_ids)))
                if not dirty_df.hyper.is_empty():
                    await db.upsert(
                        table_name=self.config.table_name,
                        dataframe=dirty_df,
                        primary_keys=ensure_list(self.config.primary_keys)
                    )

            # Success — mark clean only if no new dirty state accumulated
            async with self._lock:
                self._dirty = bool(self._persist_dirty_ids or self._persist_deleted_ids)

            await log.info(f"[MicroGrid] Persisted {self.config.name}: "
                           f"{len(dirty_ids)} upserted, {len(deleted_ids)} deleted")
        except Exception as e:
            await log.error(f"[MicroGrid] Persist failed for {self.config.name}: {e}")
            # Merge back for retry
            async with self._lock:
                self._persist_dirty_ids.update(dirty_ids)
                self._persist_deleted_ids.update(deleted_ids)
                self._dirty = True

    # -- Subscriber management --

    def add_subscriber(self, ws) -> str:
        token = str(uuid.uuid4())
        self._subscribers[token] = weakref.ref(ws)
        # Store token on ws for removal lookup
        if not hasattr(ws, '_micro_sub_tokens'):
            ws._micro_sub_tokens = {}
        ws._micro_sub_tokens[self.config.name] = token
        return token

    def remove_subscriber(self, ws) -> None:
        token = getattr(ws, '_micro_sub_tokens', {}).get(self.config.name)
        if token:
            self._subscribers.pop(token, None)
            ws._micro_sub_tokens.pop(self.config.name, None)
        else:
            # Fallback: search for matching ws ref
            to_remove = [t for t, ref in self._subscribers.items() if ref() is ws]
            for t in to_remove:
                del self._subscribers[t]

    def subscriber_count(self) -> int:
        dead = [t for t, ref in self._subscribers.items() if ref() is None]
        for t in dead:
            del self._subscribers[t]
        return len(self._subscribers)

    def get_live_subscribers(self) -> list:
        """Return list of live websocket references, pruning dead ones."""
        live = []
        dead = []
        for token, ref in self._subscribers.items():
            ws = ref()
            if ws is None:
                dead.append(token)
            else:
                live.append(ws)
        for t in dead:
            del self._subscribers[t]
        return live

    # -- Data operations --

    def snapshot(self) -> pl.DataFrame:
        """Return a copy of current data."""
        return self._data.clone()

    def snapshot_as_rows(self) -> List[Dict[str, Any]]:
        """Return data as list of dicts (for JSON serialization)."""
        return self._data.to_dicts()

    async def apply_edit(
            self,
            payloads: Dict[str, Any],
            user: str = "unknown",
    ) -> Optional[pl.DataFrame]:

        pk_col = self.config.primary_keys[0]
        target_schema = {c: self.config.column_types[c] for c in self.config.columns}

        # Normalize raw payload values into List[pl.DataFrame]
        add_frames = _normalize_payload_frames(payloads.get("add"), target_schema)
        update_frames = _normalize_payload_frames(payloads.get("update"), target_schema)
        remove_frames = _normalize_payload_frames(payloads.get("remove"), target_schema)

        add_deltas: List[pl.DataFrame] = []
        update_dirty_ids: Set[str] = set()
        has_removes = False

        async with self._lock:
            current = self._data

            # --- Add: process each frame independently ---
            for frame in add_frames:
                if pk_col not in frame.columns:
                    continue
                # Auto-generate IDs for rows missing the PK
                pk_series = frame[pk_col].cast(pl.String)
                needs_id_mask = pk_series.is_null() | (pk_series == "")
                if needs_id_mask.any():
                    new_ids = [
                        str(uuid.uuid4()) if needs else existing
                        for needs, existing in zip(
                            needs_id_mask.to_list(), pk_series.to_list()
                        )
                    ]
                    frame = frame.with_columns(
                        pl.Series(pk_col, new_ids, dtype=pl.String)
                    )

                # Fill missing columns with config defaults (only those absent)
                fill_exprs = []
                for col, default in self.config.columns.items():
                    if col not in frame.columns:
                        expected_type = self.config.column_types.get(col, pl.String)
                        fill_exprs.append(
                            pl.lit(default, dtype=expected_type).alias(col)
                        )
                if fill_exprs:
                    frame = frame.with_columns(fill_exprs)

                # Select and cast to target schema column order + types
                frame = frame.select(
                    [pl.col(c).cast(target_schema[c], strict=False)
                     for c in target_schema]
                    )

                # Deduplicate against current data before appending
                new_ids_list = frame[pk_col].cast(pl.String).to_list()
                dedup_mask = current.select(pl.col(pk_col).cast(pl.String).is_in(new_ids_list)).to_series()
                if dedup_mask.any():
                        current = current.filter(~dedup_mask)

                try:
                    frame = frame.hyper.ensure_columns(current.hyper.fields, selective=True)
                    current = pl.concat([current, frame], how="diagonal_relaxed")
                except Exception as e:
                    await log.error(e)
                    print(current.schema)
                    print(frame.schema)
                    continue

                for rid in new_ids_list:
                    self._persist_dirty_ids.add(str(rid))
                add_deltas.append(frame)

            # --- Update: vectorized join-based approach ---
            if update_frames:
                # Group frames by their updatable columns (as a frozenset key)
                grouped: Dict[frozenset, List[pl.DataFrame]] = {}
                for frame in update_frames:
                    if pk_col not in frame.hyper.schema():
                        await log.warning(f'Micro update missing pk_col: {pk_col}')
                        continue
                    ucols = frozenset(
                        c for c in frame.hyper.fields
                        if c != pk_col and c in current.hyper.schema()
                    )
                    if not ucols:
                        await log.warning('No updatable micro columns found')
                        continue
                    grouped.setdefault(ucols, []).append(frame)

                    for ucols, frames in grouped.items():
                        update_cols = sorted(ucols)
                        # Concat all frames in this group into one batch
                        select_cols = [pk_col] + update_cols
                        batch = pl.concat(
                            [f.select(select_cols) for f in frames],
                            how="vertical_relaxed",
                        )
                        # Cast PK and value columns to target types
                        batch = batch.with_columns(
                            pl.col(pk_col).cast(pl.String)
                        )
                        # Deduplicate: keep last occurrence per PK (latest wins)
                        batch = batch.unique(subset=[pk_col], keep="last")

                        # Track dirty IDs
                        batch_ids = batch[pk_col].to_list()
                        for rid_str in batch_ids:
                            self._persist_dirty_ids.add(str(rid_str))
                            update_dirty_ids.add(str(rid_str))

                        # Left-join: bring update values onto current, then coalesce
                        # Suffix the update columns so they don't collide
                        current = current.with_columns(
                            pl.col(pk_col).cast(pl.String).alias(pk_col)
                        )
                        joined = current.join(
                            batch.rename({c: f"_upd_{c}" for c in update_cols}),
                            on=pk_col,
                            how="left",
                        )
                        # For each update column, prefer the joined value when present
                        coalesce_exprs = []
                        for col in update_cols:
                            upd_col = f"_upd_{col}"
                            expected_type = self.config.column_types.get(col, pl.String)
                            coalesce_exprs.append(
                                pl.when(pl.col(upd_col).is_not_null())
                                .then(pl.col(upd_col).cast(expected_type, strict=False))
                                .otherwise(pl.col(col))
                                .alias(col)
                            )
                        current = joined.with_columns(coalesce_exprs).drop(
                            [f"_upd_{c}" for c in update_cols]
                        )

            # --- Remove: process each frame independently ---
            for frame in remove_frames:
                if pk_col not in frame.columns:
                    continue
                remove_ids = frame[pk_col].cast(pl.String).to_list()
                remove_id_set = set(remove_ids)
                if not remove_id_set:
                    continue
                remove_mask = current[pk_col].cast(pl.String).is_in(
                    list(remove_id_set)
                )
                if remove_mask.any():
                    has_removes = True
                    current = current.filter(~remove_mask)
                    for rid in remove_id_set:
                        self._persist_deleted_ids.add(rid)
                        self._persist_dirty_ids.discard(rid)

            self._data = current

            # --- Build delta output ---
            # Add deltas: each frame was already shaped to target_schema
            add_count = 0
            if add_deltas:

                try:
                    add_delta = (
                        pl.concat(add_deltas, how="diagonal_relaxed")
                        if len(add_deltas) > 1 else add_deltas[0]
                    )
                except Exception as e:
                    await log.error(e, v=2)
                    print(add_deltas)
                    raise

                add_count = add_delta.height
            else:
                add_delta = None

            # Update delta: materialize from current data by dirty IDs
            if update_dirty_ids:
                update_delta = current.filter(
                    current[pk_col].cast(pl.String).is_in(
                        list(update_dirty_ids)
                    )
                )
                if update_delta.hyper.is_empty():
                    update_delta = None
            else:
                update_delta = None

            # Combine
            delta_parts = [p for p in (add_delta, update_delta) if p is not None]
            if delta_parts or has_removes:
                self._dirty = True

            if delta_parts:
                if len(delta_parts) == 1:
                    delta_df = delta_parts[0]
                else:
                    # Both add and update deltas share full target_schema
                    try:
                        delta_df = pl.concat(delta_parts, how="diagonal_relaxed")
                    except Exception as e:
                        await log.error(e, v=3)
                        print(delta_parts)
                        raise

                delta_df._micro_add_count = add_count
            elif has_removes:
                delta_df = pl.DataFrame(schema=target_schema)
                delta_df._micro_add_count = 0
            else:
                return None

        return delta_df


# =============================================================================
# Payload normalization
# =============================================================================

def _normalize_payload_frames(
    raw: Any,
    target_schema: Dict[str, pl.DataType],
) -> List[pl.DataFrame]:
    """
    Normalize a raw payload value into List[pl.DataFrame].

    Handles all shapes that arrive here:
      - None / empty list → []
      - List[pl.DataFrame] → returned as-is (empty frames filtered out)
      - single pl.DataFrame → wrapped in a list
      - List[dict] → each dict becomes a single-row DataFrame (legacy/rules path)
    """
    if raw is None:
        return []
    if isinstance(raw, pl.DataFrame):
        return [raw] if not raw.hyper.is_empty() else []

    frames = ensure_list(raw)
    if not frames:
        return []

    first = frames[0]
    if isinstance(first, pl.DataFrame):
        return [f for f in frames if not f.hyper.is_empty()]
    elif isinstance(first, dict):
        # Batch dicts that share the same column set into a single
        # DataFrame for efficient vectorized processing downstream.
        # Group by frozenset of columns present in the target schema.
        grouped: Dict[frozenset, List[dict]] = {}
        for d in frames:
            if not isinstance(d, dict) or not d:
                continue
            cols = frozenset(k for k in d if k in target_schema)
            if not cols:
                continue
            grouped.setdefault(cols, []).append(d)
        out: List[pl.DataFrame] = []
        for cols, dicts in grouped.items():
            row_schema = {k: target_schema[k] for k in cols}
            out.append(
                pl.DataFrame(dicts, schema=row_schema, strict=False)
            )
        return out
    return []


# =============================================================================
# Initialization
# =============================================================================

async def init_micro_grids() -> None:
    """Load all registered micro-grids from DB. Call at startup."""
    registry = _registry
    for config in registry.list_configs():
        actor = MicroGridActor(config)
        await actor.load_from_db()
        registry.set_actor(config.name, actor)
        await log.info(f"[MicroGrid] Initialized: {config.name} ({actor._data.hyper.height()} rows)")


async def shutdown_micro_grids() -> None:
    """Persist and clean up all micro-grid actors."""
    await _registry.shutdown()
