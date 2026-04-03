

import asyncio
import contextlib
import time
import uuid as _uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, NewType, Union

import msgspec.json
import pyarrow as pa

try:
    import re2 as re
except ImportError:
    import re
from async_lru import alru_cache

from app.helpers.codecHelpers import prep_outgoing_payload, encode, compress
from app.helpers.common import PRICE_TYPES, SPREAD_TYPES, MMY_TYPES, BUY_TYPES, SELL_TYPES
from app.helpers.date_helpers import isonow
from app.helpers.hash import md5_string
from app.helpers.type_helpers import ensure_list
from app.logs.logging import log
from app.services.payload.columnar_codec import OptimizedColumnarCodec
from app.services.payload.payloadBatcher import PayloadBatcher
from app.services.payload.payloadV4 import *
from app.services.redux.grid_system_v4 import (
    GridSystem, GridActor, RuleDef, RuleContext, Priority, query_primary_keys,
    _INTERNAL_ROW_ALIVE,
    ArrowQueryEngine, ArrowSubscription, ArrowRoomSubscription,
    ArrowSubscriptionRegistry, ArrowRoomRegistry,
    serialize_arrow_ipc, content_hash, execute_export,
    extract_touched_columns, normalize_filters,
    apply_room_filters, row_matches_room_filters,
    project_row,
)
from app.services.server.router import PubSubRouter
from app.services.storage.sqlManagerV2 import _package_arrow_metadata, _arrow_ipc_from_arrow

from app.helpers.polars_hyper_plugin import *  # noqa: F401,F403

try:
    from starlette.websockets import WebSocket, WebSocketState
    from starlette.websockets import WebSocketDisconnect
except Exception:
    class WebSocket: ...
    class WebSocketDisconnect(Exception): ...
    class WebSocketState:
        CONNECTED = 1


# =============================================================================
# Config
# =============================================================================

INDEX_CACHE_MAX = 64

# Outbound Payload batcher
OUT_MAX_BATCH = 10_000
OUT_FLUSH_MS = 260

# Rules Engine -
SINK_QUEUE_MAX = 10_000
SINK_SHARDS = 1
DROP_OLDEST_WHEN_FULL = True

CLIENT_WRITE_TIMEOUT_S = 2.5
CLIENT_WRITE_TIMEOUTS_BEFORE_CLOSE = 3
SINK_DRAIN_WARN_INTERVAL_S = 2.0

# Rooms
UNKNOWN_ROOM = "UNKNOWN"

# Client
ENABLE_PER_SOCKET_SENDER = False
PER_SOCKET_QUEUE_MAX = 64

# Identifier validation for execute handler
_IDENT = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


# =============================================================================
# Utilities
# =============================================================================

_json_dec = msgspec.json.Decoder(type=dict)


def _safe_att(obj, k):
    return getattr(obj, k, None)


def _upper(s: Optional[str]) -> str:
    return (s or "").upper()


def _is_connected(ws: WebSocket) -> bool:
    if ws is None: return False
    try:
        return (
                getattr(ws, "client_state", None)==WebSocketState.CONNECTED and
                getattr(ws, "application_state", None)==WebSocketState.CONNECTED
        )
    except Exception:
        return False


def _build_generic_room_context(portfolio_key):
    pku, pkl = portfolio_key.upper(), portfolio_key.lower()
    pt = RoomContext(
        room=f"{pku}.PORTFOLIO",
        grid_id="portfolio",
        grid_filters={"field": "portfolioKey", "op": "eq", "value": pkl},
    )
    meta = RoomContext(
        room=f"{pku}.META",
        grid_id="meta",
        grid_filters={"field": "portfolioKey", "op": "eq", "value": pkl},
    )
    return pt, meta


def generate_portfolio_key(rfq_list_id: Optional[str] = None):
    r = rfq_list_id or str(_uuid.uuid4())
    return md5_string(r)


def _normalize_user(d):
    if d is None:
        return {}
    if isinstance(d, User):
        return d.to_dict()
    if not isinstance(d, dict):
        return {}
    u = d.get("user", {})
    if hasattr(u, "to_dict"):
        d["user"] = u.to_dict()
    else:
        d["user"] = dict(u)
    return d


SubscriberId = NewType("SubscriberId", int)

_TOKEN_COUNTER = 0
_TOKEN_MAP: dict[int, SubscriberId] = {}  # id(obj) -> stable token


def _subscriber_token(x: Any) -> SubscriberId:
    global _TOKEN_COUNTER
    if x is None:
        return SubscriberId(0)
    if isinstance(x, int):
        return SubscriberId(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        return SubscriberId(hash(bytes(x)))
    if isinstance(x, str):
        return SubscriberId(hash(x))
    obj_id = id(x)
    existing = _TOKEN_MAP.get(obj_id)
    if existing is not None:
        return existing
    _TOKEN_COUNTER += 1
    tok = SubscriberId(_TOKEN_COUNTER)
    _TOKEN_MAP[obj_id] = tok
    return tok


def _release_token(x: Any):
    """Remove the token mapping for *x* so its id can be reused safely."""
    _TOKEN_MAP.pop(id(x), None)


@dataclass(frozen=True)
class _ClientKey:
    room: str
    grid_id: str
    filters_json: Any  # bytes/dict depending on your RoomContext.json_filters


import polars as pl


def _camel_case_token(token: str) -> str:
    # Fast, deterministic, good enough for market identifiers
    # "ICE_SWAP" -> "iceSwap", "ICE Swap" -> "iceSwap", "ice-swap" -> "iceSwap"
    if not token:
        return ""
    # normalize to alnum words
    out_words = []
    w = []
    for ch in token:
        o = ord(ch)
        if 48 <= o <= 57 or 65 <= o <= 90 or 97 <= o <= 122:
            w.append(ch)
        else:
            if w:
                out_words.append("".join(w))
                w.clear()
    if w:
        out_words.append("".join(w))

    if not out_words:
        return ""

    first = out_words[0].lower()
    # titlecase remaining words (ASCII-only, faster than .title())
    rest = []
    for s in out_words[1:]:
        if not s:
            continue
        rest.append(s[0].upper() + s[1:].lower())
    return first + "".join(rest)


def _build_prefix_map(skews: pl.LazyFrame, ref_mkt_disp_col: str = "refMktDisp"):
    """
    Builds a python dict mapping the first token of refMktDisp -> camelCase token.
    Done once; avoids per-row string casing UDFs in the query plan.
    """
    tokens = (
        skews.select(
            pl.col(ref_mkt_disp_col)
            .cast(pl.Utf8, strict=False)
            .str.split(" ")
            .list.get(0)
            .alias("_tok")
        )
        .unique()
        .collect(streaming=True)["_tok"]
        .to_list()
    )
    mapping = {}
    for t in tokens:
        if t is None:
            continue
        cc = _camel_case_token(t)
        if cc:
            mapping[t] = cc
    return mapping


def enrich_skews_with_ref_levels(
        skews: pl.LazyFrame,
        mkts: pl.LazyFrame,
        *,
        bid_px_mkts: list[str],
        mid_px_mkts: list[str],
        ask_px_mkts: list[str],
        bid_spd_mkts: list[str],
        mid_spd_mkts: list[str],
        ask_spd_mkts: list[str],
        all_mkts: list[str],
        IS_PX: pl.Expr,
        IS_SPD: pl.Expr,
):
    """
    Produces:
      _refBidPx, _refMidPx, _refAskPx, _refBidSpd, _refMidSpd, _refAskSpd

    Requirements / assumptions (matching your original intent):
      - quoteType and QT exist on skews
      - refBid/refMid/refAsk, refMktDisp, refMktMkt, tnum exist on skews
      - mkts contains tnum + all_mkts columns (wide), values numeric
      - Px market columns are named like "<prefix><Side>Px"
      - Spd market columns are named like "<prefix><Side>"  (no "Spd" suffix)
      - refMktMkt already matches the mkts prefix naming; refMktDisp needs camelCasing of its first token
    """
    # Precompute mapping once (cheap vs the original per-row string logic)
    prefix_map = _build_prefix_map(skews)

    # mkts: wide -> long only for the columns we need
    mkts_long = (
        mkts.select(["tnum"] + list(all_mkts))
        .unpivot(
            index="tnum",
            on=list(all_mkts),
            variable_name="_mkt_col",
            value_name="_mkt_val",
        )
        .with_columns(pl.col("_mkt_val").cast(pl.Float64, strict=False))
    )

    # Build all keys + direct values on skews
    base = (
        skews.with_row_index("_rid")
        .with_columns(
            (pl.col("quoteType") == pl.col("QT")).alias("_qt_match"),
            pl.col("refMktDisp")
            .cast(pl.Utf8, strict=False)
            .str.split(" ")
            .list.get(0)
            .alias("_ref_tok"),
            )
        .with_columns(
            # token -> camelCase prefix via vectorized replace
            pl.col("_ref_tok").replace(prefix_map, default=pl.col("_ref_tok")).alias("_ref_prefix"),
        )
        .with_columns(
            # Base market names (no underscores; matches your camelCase intent)
            pl.concat_str([pl.col("_ref_prefix"), pl.lit("Bid")], separator="").alias("_refBidMkt"),
            pl.concat_str([pl.col("_ref_prefix"), pl.lit("Mid")], separator="").alias("_refMidMkt"),
            pl.concat_str([pl.col("_ref_prefix"), pl.lit("Ask")], separator="").alias("_refAskMkt"),
            pl.concat_str([pl.col("refMktMkt").cast(pl.Utf8, strict=False), pl.lit("Bid")], separator="").alias("_fb_refBidMkt"),
            pl.concat_str([pl.col("refMktMkt").cast(pl.Utf8, strict=False), pl.lit("Mid")], separator="").alias("_fb_refMidMkt"),
            pl.concat_str([pl.col("refMktMkt").cast(pl.Utf8, strict=False), pl.lit("Ask")], separator="").alias("_fb_refAskMkt"),
        )
        .with_columns(
            # Candidate Px column names
            pl.concat_str([pl.col("_refBidMkt"), pl.lit("Px")], separator="").alias("_refBidPxCol"),
            pl.concat_str([pl.col("_refMidMkt"), pl.lit("Px")], separator="").alias("_refMidPxCol"),
            pl.concat_str([pl.col("_refAskMkt"), pl.lit("Px")], separator="").alias("_refAskPxCol"),
            pl.concat_str([pl.col("_fb_refBidMkt"), pl.lit("Px")], separator="").alias("_fb_refBidPxCol"),
            pl.concat_str([pl.col("_fb_refMidMkt"), pl.lit("Px")], separator="").alias("_fb_refMidPxCol"),
            pl.concat_str([pl.col("_fb_refAskMkt"), pl.lit("Px")], separator="").alias("_fb_refAskPxCol"),
        )
        .with_columns(
            # Direct values (only when quoteType matches QT)
            pl.when(pl.col("_qt_match") & IS_PX).then(pl.col("refBid").cast(pl.Float64, strict=False)).otherwise(None).alias("_direct_refBidPx"),
            pl.when(pl.col("_qt_match") & IS_PX).then(pl.col("refMid").cast(pl.Float64, strict=False)).otherwise(None).alias("_direct_refMidPx"),
            pl.when(pl.col("_qt_match") & IS_PX).then(pl.col("refAsk").cast(pl.Float64, strict=False)).otherwise(None).alias("_direct_refAskPx"),

            pl.when(pl.col("_qt_match") & IS_SPD).then(pl.col("refBid").cast(pl.Float64, strict=False)).otherwise(None).alias("_direct_refBidSpd"),
            pl.when(pl.col("_qt_match") & IS_SPD).then(pl.col("refMid").cast(pl.Float64, strict=False)).otherwise(None).alias("_direct_refMidSpd"),
            pl.when(pl.col("_qt_match") & IS_SPD).then(pl.col("refAsk").cast(pl.Float64, strict=False)).otherwise(None).alias("_direct_refAskSpd"),
        )
        .with_columns(
            # Choose the single best lookup key per slot using membership tests.
            # This replicates your "primary then fallback" behavior without scanning all columns.
            pl.when(IS_PX)
            .then(
                pl.when(pl.col("_refBidPxCol").is_in(bid_px_mkts))
                .then(pl.col("_refBidPxCol"))
                .when(pl.col("_fb_refBidPxCol").is_in(bid_px_mkts))
                .then(pl.col("_fb_refBidPxCol"))
                .otherwise(None)
            )
            .otherwise(None)
            .alias("_k_refBidPx"),

            pl.when(IS_PX)
            .then(pl.when(pl.col("_refMidPxCol").is_in(mid_px_mkts)).then(pl.col("_refMidPxCol")).otherwise(None))
            .otherwise(None)
            .alias("_k_refMidPx"),

            pl.when(IS_PX)
            .then(pl.when(pl.col("_refAskPxCol").is_in(ask_px_mkts)).then(pl.col("_refAskPxCol")).otherwise(None))
            .otherwise(None)
            .alias("_k_refAskPx"),

            pl.when(IS_SPD)
            .then(pl.when(pl.col("_refBidMkt").is_in(bid_spd_mkts)).then(pl.col("_refBidMkt")).otherwise(None))
            .otherwise(None)
            .alias("_k_refBidSpd"),

            pl.when(IS_SPD)
            .then(pl.when(pl.col("_refMidMkt").is_in(mid_spd_mkts)).then(pl.col("_refMidMkt")).otherwise(None))
            .otherwise(None)
            .alias("_k_refMidSpd"),

            pl.when(IS_SPD)
            .then(pl.when(pl.col("_refAskMkt").is_in(ask_spd_mkts)).then(pl.col("_refAskMkt")).otherwise(None))
            .otherwise(None)
            .alias("_k_refAskSpd"),
            )
    )

    # Convert skews to "requests" long form: 6 lookups per row, joined once
    requests = (
        base.select(
            "_rid",
            "tnum",
            pl.concat_list(
                [
                    pl.struct(
                        pl.lit("_refBidPx").alias("_slot"),
                        pl.col("_k_refBidPx").alias("_mkt_col"),
                        pl.col("_direct_refBidPx").alias("_direct"),
                    ),
                    pl.struct(
                        pl.lit("_refMidPx").alias("_slot"),
                        pl.col("_k_refMidPx").alias("_mkt_col"),
                        pl.col("_direct_refMidPx").alias("_direct"),
                    ),
                    pl.struct(
                        pl.lit("_refAskPx").alias("_slot"),
                        pl.col("_k_refAskPx").alias("_mkt_col"),
                        pl.col("_direct_refAskPx").alias("_direct"),
                    ),
                    pl.struct(
                        pl.lit("_refBidSpd").alias("_slot"),
                        pl.col("_k_refBidSpd").alias("_mkt_col"),
                        pl.col("_direct_refBidSpd").alias("_direct"),
                    ),
                    pl.struct(
                        pl.lit("_refMidSpd").alias("_slot"),
                        pl.col("_k_refMidSpd").alias("_mkt_col"),
                        pl.col("_direct_refMidSpd").alias("_direct"),
                    ),
                    pl.struct(
                        pl.lit("_refAskSpd").alias("_slot"),
                        pl.col("_k_refAskSpd").alias("_mkt_col"),
                        pl.col("_direct_refAskSpd").alias("_direct"),
                    ),
                ]
            ).alias("_reqs"),
        )
        .explode("_reqs")
        .unnest("_reqs")
    )

    # Join once to fetch lookup value; coalesce(direct, lookup)
    resolved_long = (
        requests.join(mkts_long, on=["tnum", "_mkt_col"], how="left")
        .with_columns(pl.coalesce([pl.col("_direct"), pl.col("_mkt_val")]).alias("_val"))
        .select("_rid", "_slot", "_val")
    )

    # Pivot back to wide 6 columns and join to original skews
    resolved_wide = (
        resolved_long.pivot(
            index="_rid",
            columns="_slot",
            values="_val",
            aggregate_function="first",
        )
    )

    # Final: attach and clean up helper columns; no need to join mkts wide, so no huge drop()
    out = (
        base.drop(
            [
                "_qt_match",
                "_ref_tok",
                "_ref_prefix",
                "_refBidMkt",
                "_refMidMkt",
                "_refAskMkt",
                "_fb_refBidMkt",
                "_fb_refMidMkt",
                "_fb_refAskMkt",
                "_refBidPxCol",
                "_refMidPxCol",
                "_refAskPxCol",
                "_fb_refBidPxCol",
                "_fb_refMidPxCol",
                "_fb_refAskPxCol",
                "_direct_refBidPx",
                "_direct_refMidPx",
                "_direct_refAskPx",
                "_direct_refBidSpd",
                "_direct_refMidSpd",
                "_direct_refAskSpd",
                "_k_refBidPx",
                "_k_refMidPx",
                "_k_refAskPx",
                "_k_refBidSpd",
                "_k_refMidSpd",
                "_k_refAskSpd",
            ],
            strict=False,
        )
        .join(resolved_wide, on="_rid", how="left")
        .drop("_rid", strict=False)
    )

    return out

# =============================================================================
# ConnectionManager
# =============================================================================


class ConnectionManager:

    def __init__(self):
        self.db_handler = None
        self.ctx = None
        self.ksm = None
        self.executor = None

        self.router = PubSubRouter(delimiter=".", debug=True)

        # token -> User / ws
        self.user_by_socket: Dict[SubscriberId, User] = {}
        self.socket_by_token: Dict[SubscriberId, WebSocket] = {}

        self.user_by_ip: Dict[str, User] = {}
        self.user_count = defaultdict(set)
        self.sockets_by_username: Dict[str, Set[SubscriberId]] = defaultdict(set)

        self._socket_queues: dict[SubscriberId, asyncio.Queue] = {}
        self._socket_tasks: dict[SubscriberId, asyncio.Task] = {}

        # token -> set[(room, grid_id, filters_json)]
        self._active_contexts: dict[SubscriberId, set[tuple[str, str, Any]]] = defaultdict(set)

        # token -> consecutive timeouts
        self._ws_timeouts: dict[SubscriberId, int] = defaultdict(int)

        # Per-subscription lazy column tracking.
        # Key: (SubscriberId, room_upper, grid_id)
        # Value: set of column names currently loaded by this subscriber.
        #        Absent key means the subscriber receives all columns (non-lazy).
        self._lazy_columns: Dict[tuple, Set[str]] = {}
        # Reverse index: token -> set of lazy_columns keys for O(1) cleanup
        self._lazy_tokens: Dict[SubscriberId, Set[tuple]] = defaultdict(set)

        # Guard against concurrent _cleanup_socket for same token
        self._cleaning_up: set[SubscriberId] = set()

        self.grid_system = GridSystem()
        self.market_columns = None

        # Arrow IPC reactive subscription registries
        self.arrow_sub_registry = ArrowSubscriptionRegistry()
        self.arrow_room_registry = ArrowRoomRegistry()
        self._arrow_token_to_client: Dict[int, str] = {}

        self._queue = asyncio.Queue(SINK_QUEUE_MAX)
        self._pump_task: Optional[asyncio.Task] = None

        self.outboundBatcher: Optional[PayloadBatcher] = None
        self._emit_task: Optional[asyncio.Task] = None
        self._outbound_running = False

        self.initialized = False
        self._init_lock = asyncio.Lock()

        # Cached dispatch table (populated lazily)
        self._DISPATCH: Optional[dict] = None

    # -------------------------------------------------------------------------
    # Boot / Shutdown
    # -------------------------------------------------------------------------

    async def init(self):
        async with self._init_lock:
            if self.initialized:
                return
            self.initialized = True

        from app.server import get_ctx, get_db, get_ksm, get_threads
        self.db_handler = get_db()
        self.ksm = get_ksm()
        self.ctx = get_ctx()
        self.executor = get_threads()

        await self.install_default_rules()
        await self._init_micro_grids()

        self.outboundBatcher = PayloadBatcher(
            max_batch_size=OUT_MAX_BATCH,
            flush_interval_ms=OUT_FLUSH_MS,
            flatten=False,
        )
        await self.outboundBatcher.init()

        if not self._outbound_running:
            self._outbound_running = True
            self._emit_task = asyncio.create_task(self._emit_loop(), name="emit-loop")

        self.market_columns = await self.db_handler.market_columns()

    async def install_default_rules(self):

        def b(n, s):
            async def add_wow(ctx: RuleContext) -> pl.DataFrame:
                snap = await ctx.ingress_delta_slice(columns=["comment"])
                await asyncio.sleep(s)
                res = snap.with_columns(
                    pl.when(pl.col("comment").is_not_null())
                    .then(pl.concat_str(pl.col("comment"), pl.lit(n)))
                    .otherwise(pl.col("comment"))
                    .alias("comment")
                )
                return res

            return add_wow

        test1 = RuleDef(
            "test1",
            column_triggers_all=("comment",),
            priority=Priority.HIGH,
            declared_column_outputs=("comment",),
            func=b(1, 0),
        )
        test2 = RuleDef(
            "test2",
            column_triggers_all=("comment",),
            priority=Priority.LOW,
            declared_column_outputs=("comment",),
            func=b(2, 5),
        )
        test3 = RuleDef(
            "test3",
            column_triggers_all=("comment",),
            priority=Priority.HIGH,
            declared_column_outputs=("comment",),
            func=b(3, 1),
        )
        self.grid_system.rules.register(test1)
        self.grid_system.rules.register(test2)
        self.grid_system.rules.register(test3)

        from app.services.rules.portfolio.portfolio_rules_v4 import register_portfolio_rules
        register_portfolio_rules(self.grid_system.rules)

    async def _init_micro_grids(self):
        """Register and initialize all micro-grids at startup."""
        try:
            import polars as pl
            from app.services.redux.micro_grid import (
                MicroGridConfig, MicroGridGroup,
                register_micro_grid, register_micro_grid_group,
                init_micro_grids,
            )

            # --- Register micro-grid configs ---
            register_micro_grid(MicroGridConfig(
                name="hot_tickers",
                table_name="micro_hot_tickers",
                primary_keys=("id",),
                columns={
                    "id": "",
                    "column": "ticker",
                    "pattern": "",
                    "match_mode": "literal",
                    "severity": "low",
                    "display": "column",
                    "color": "#FFFF00",
                    "tags": "",
                    "notes": "",
                    "updated_at": "",
                    "updated_by": "",
                },
                column_types={
                    "id": pl.String,
                    "column": pl.String,
                    "pattern": pl.String,
                    "match_mode": pl.String,
                    "severity": pl.String,
                    "display": pl.String,
                    "color": pl.String,
                    "tags": pl.String,
                    "notes": pl.String,
                    "updated_at": pl.String,
                    "updated_by": pl.String,
                },
                rules_enabled=True,
                persist=True,
            ))

            # --- Redist solver parameters ---
            register_micro_grid(MicroGridConfig(
                name="redist_params",
                table_name="micro_redist_params",
                primary_keys=("param_key",),
                columns={
                    "category": "",
                    "param_key": "",
                    "param_value": 0.0,
                    "description": "",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "updated_at": "",
                    "updated_by": "",
                },
                column_types={
                    "category": pl.String,
                    "param_key": pl.String,
                    "param_value": pl.Float64,
                    "description": pl.String,
                    "min_value": pl.Float64,
                    "max_value": pl.Float64,
                    "updated_at": pl.String,
                    "updated_by": pl.String,
                },
                rules_enabled=False,
                persist=True,
            ))

            # --- Register micro-grid groups ---
            register_micro_grid_group(MicroGridGroup(
                name="pt_tools",
                display_name="Portfolio Tools",
                grids=("hot_tickers",),
            ))

            # --- Load from DB ---
            await init_micro_grids()

            # --- Seed redist_params with defaults if empty ---
            await self._seed_redist_params()

            await log.info("[MicroGrid] All micro-grids initialized")

        except Exception as e:
            await log.error(f"[MicroGrid] Initialization failed: {e}")

    async def _seed_redist_params(self):
        """Populate the redist_params micro-grid with factory defaults when empty."""
        from app.services.redux.micro_grid import get_micro_actor
        try:
            actor = get_micro_actor("redist_params")
            if actor._data.hyper.height() > 0:
                return  # already seeded

            import uuid
            _SEED_ROWS = [
                # -- Charge strength (BSR/BSI multipliers by quote type) --
                {
                    "category"   : "charge_strength", "param_key": "BSR_PX",
                    "param_value": 0.10, "description": "BSR charge multiplier for PX-quoted bonds",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "charge_strength", "param_key": "BSI_PX",
                    "param_value": 0.10, "description": "BSI charge multiplier for PX-quoted bonds",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "charge_strength", "param_key": "BSR_SPD",
                    "param_value": 0.20, "description": "BSR charge multiplier for SPD-quoted bonds",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "charge_strength", "param_key": "BSI_SPD",
                    "param_value": 0.20, "description": "BSI charge multiplier for SPD-quoted bonds",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                # -- Default strength (fallback when quote type is unknown) --
                {
                    "category"   : "charge_strength", "param_key": "DEFAULT",
                    "param_value": 0.20, "description": "Fallback charge strength for unknown quote types",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                # -- Risk weights --
                {
                    "category"   : "risk_weight", "param_key": "NET_BSI",
                    "param_value": 0.50, "description": "Risk weight for net BSI exposure",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "risk_weight", "param_key": "GROSS_BSI",
                    "param_value": 0.25, "description": "Risk weight for gross BSI exposure",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "risk_weight", "param_key": "NET",
                    "param_value": 0.15, "description": "Risk weight for net position risk",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "risk_weight", "param_key": "GROSS",
                    "param_value": 0.10, "description": "Risk weight for gross position risk",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                # -- Solver defaults --
                {
                    "category"   : "solver_default", "param_key": "side_floor_pct",
                    "param_value": 0.99, "description": "Symmetric band for total proceeds per side (0.99 = ±1% wavg shift)",
                    "min_value"  : 0.50, "max_value": 1.0
                },
                {
                    "category"   : "solver_default", "param_key": "buffer_mode",
                    "param_value": 1, "description": "0 = fixed$ buffer, 1 = percentage buffer"
                },
                {
                    "category"   : "solver_default", "param_key": "buffer_fixed",
                    "param_value": 500.0, "description": "Flat $ buffer per trader (when buffer_mode=fixed)",
                    "min_value"  : 0.0, "max_value": 50000.0
                },
                {
                    "category"   : "solver_default", "param_key": "buffer_pct",
                    "param_value": 0.25, "description": "Pct buffer per trader as wide safety guardrail (0.25 = ±25%)",
                    "min_value"  : 0.0, "max_value": 1.0
                },
                {
                    "category"   : "solver_default", "param_key": "buffer_min_flat",
                    "param_value": 200.0, "description": "Minimum $ buffer floor when buffer_mode=pct",
                    "min_value"  : 0.0, "max_value": 10000.0
                },
                {
                    "category"   : "solver_default", "param_key": "lambda_param",
                    "param_value": 0.25, "description": "Anchoring strength — resists change from starting charges",
                    "min_value"  : 0.0, "max_value": 20.0
                },
                {
                    "category"   : "solver_default", "param_key": "gamma",
                    "param_value": 1.0, "description": "Curve-pull strength — pulls bonds toward charge-curve ideal",
                    "min_value"  : 0.0, "max_value": 20.0
                },
                {
                    "category"   : "solver_default", "param_key": "trader_pull",
                    "param_value": 1.0, "description": "Trader target-pull — drives redistribution across traders",
                    "min_value"  : 0.0, "max_value": 20.0
                },
                {
                    "category"   : "solver_default", "param_key": "target_blend",
                    "param_value": 0.50, "description": "Target anchoring: 0=keep starting, 1=full risk-weighted",
                    "min_value"  : 0.0, "max_value": 1.0
                },
            ]

            import polars as pl
            seed_df = pl.DataFrame(_SEED_ROWS)
            await actor.apply_edit({"add": seed_df}, user="system")
            await actor.persist()
            await log.info(f"[MicroGrid] Seeded redist_params with {len(_SEED_ROWS)} default rows")

        except Exception as e:
            await log.error(f"[MicroGrid] Failed to seed redist_params: {e}")

    async def shutdown(self):
        self._outbound_running = False

        try:
            if self._emit_task:
                self._emit_task.cancel()
                with contextlib.suppress(Exception):
                    await self._emit_task
                self._emit_task = None
        except (Exception, asyncio.CancelledError) as e:
            await log.error(f"Errored while shutting down emit task: {e}")

        try:
            if self.outboundBatcher:
                await self.outboundBatcher.shutdown()
                self.outboundBatcher = None
        except (Exception, asyncio.CancelledError) as e:
            await log.error(f"Errored while shutting down outbound batcher: {e}")

        try:
            if ENABLE_PER_SOCKET_SENDER:
                for t in self._socket_tasks.values():
                    t.cancel()
                await asyncio.gather(*list(self._socket_tasks.values()), return_exceptions=True)
        except (Exception, asyncio.CancelledError) as e:
            await log.error(f"Errored while shutting down socket per sender: {e}")

        try:
            await self.router.shutdown()
        except (Exception, asyncio.CancelledError) as e:
            await log.error(f"Errored while shutting down router: {e}")

        self.user_by_socket.clear()
        self.socket_by_token.clear()
        self.sockets_by_username.clear()
        self.user_by_ip.clear()
        self.user_count.clear()
        self._active_contexts.clear()
        self._ws_timeouts.clear()
        self._lazy_columns.clear()
        self._lazy_tokens.clear()
        self._cleaning_up.clear()

        try:
            from app.services.redux.micro_grid import shutdown_micro_grids
            await shutdown_micro_grids()
        except Exception as e:
            await log.error(f"Errored while shutting down micro-grids: {e}")

        try:
            await self.grid_system.shutdown()
        except (Exception, asyncio.CancelledError) as e:
            await log.error(f"Errored while shutting down grid_system: {e}")

        self.initialized = False

    # -------------------------------------------------------------------------
    # User Data / Identity
    # -------------------------------------------------------------------------

    @classmethod
    def _create_user_data(cls, identity_data: Dict[str, Any]):
        fp = identity_data.get("fp", "UNKNOWN")
        sfp = identity_data.get("sfp", "UNKNOWN")
        username = identity_data.get("un", "UNKNOWN")
        client_ip = identity_data.get("client_ip", None)
        client_port = identity_data.get("client_port", None)
        return User(
            fingerprint=fp,
            sessionFingerprint=sfp,
            username=username,
            client_ip=client_ip,
            client_port=client_port,
        )

    def set_user(self, token: SubscriberId, websocket: WebSocket, user: User):
        # Clean up old username's tracking if re-identifying under a different name
        old = self.user_by_socket.get(token)
        if old is not None and hasattr(old, 'username') and old.username != user.username:
            self.sockets_by_username[old.username].discard(token)
            self.user_count[old.username].discard(token)

        self.user_by_socket[token] = user
        self.socket_by_token[token] = websocket

        self.sockets_by_username[user.username].add(token)
        self.user_count[user.username].add(token)
        self.user_by_ip[self._get_user_ip(user)] = user

    def get_user_by_socket(self, websocket: WebSocket):
        token = _subscriber_token(websocket)
        d = self.user_by_socket.get(token, {}) or {}
        return _normalize_user(d)

    def get_user_by_token(self, token: SubscriberId):
        d = self.user_by_socket.get(token, {}) or {}
        return _normalize_user(d)

    def get_user_by_username(self, username: str, n=1):
        # NOTE: original code here looked incorrect; leaving conservative behavior
        d = list(self.sockets_by_username.get(username, set()) or set())
        d = d[:n]
        return _normalize_user({"tokens": d})

    # -------------------------------------------------------------------------
    # WebSocket lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, identity_data: Dict[str, Any]):
        """
        IMPORTANT: Accept should happen exactly once.
        Recommended: endpoint calls `await websocket.accept()` before calling connect().
        """
        try:
            user_data = self._create_user_data(identity_data)
        except Exception as e:
            return await log.websocket_error(f"WebSocket connect failed, malformed identity: {e}")

        try:
            await self.on_connect(websocket, user_data)
        except Exception as e:
            return await log.websocket_error(
                f"WebSocket connect failed for FP={user_data.fingerprint}, "
                f"SFP={user_data.sessionFingerprint}, U={user_data.username}: {e}"
            )

    async def on_connect(self, websocket: WebSocket, user: User):
        token = _subscriber_token(websocket)
        self.set_user(token, websocket, user)

        if ENABLE_PER_SOCKET_SENDER:
            q = asyncio.Queue(PER_SOCKET_QUEUE_MAX)
            self._socket_queues[token] = q
            self._socket_tasks[token] = asyncio.create_task(self._socket_sender_loop(websocket, q))

    async def _handle_disconnect(self, websocket: WebSocket, message: Any | BroadcastMessage):
        maybe_identity = message.get('user', None) if hasattr(message, 'get') else None
        maybe_reason = message.get('reason', None) if hasattr(message, 'get') else None
        await self.disconnect(websocket, identity_data=maybe_identity, reason=maybe_reason)

    async def disconnect(self, websocket: WebSocket, identity_data=None, reason=None):
        try:
            user_data = self.get_user_by_socket(websocket)
            if user_data is None:
                identity_data = {} if identity_data is None else identity_data
                user_data = self._create_user_data(identity_data)

            if user_data:
                un = user_data.get("username") if isinstance(user_data, dict) else getattr(user_data, "username", None)
                prior = len(self.user_count.get(un, [])) if un else 0
                if prior != 0:
                    user_str = f"{user_data.get('displayName', 'A user')} disconnected."
                    reason_str = f" Reason: {reason} " if reason is not None else ''
                    prior_str = f"(#{prior} -> {('#' + str(prior - 1)) if prior > 1 else 'null'})"
                    await log.websocket(f"{user_str}{reason_str}{prior_str}")

            await self.on_disconnect(websocket)
        except (Exception, asyncio.CancelledError) as e:
            await log.websocket_error(f"Error during websocket cleanup: {e}")

    @classmethod
    def _get_user_ip(cls, user: User):
        return f"{user.client_ip}:{user.client_port}"

    async def _cleanup_socket(self, websocket: WebSocket):
        token = _subscriber_token(websocket)

        # Guard against concurrent cleanup for the same token
        if token in self._cleaning_up:
            return
        self._cleaning_up.add(token)

        try:
            user = self.user_by_socket.pop(token, None)
            self.socket_by_token.pop(token, None)

            if user:
                # Always remove token from both tracking sets
                self.sockets_by_username[user.username].discard(token)
                self.user_count[user.username].discard(token)
                self.user_by_ip.pop(self._get_user_ip(user), None)
                # Clean up empty entries to prevent monotonic growth
                if not self.user_count[user.username]:
                    del self.user_count[user.username]
                if not self.sockets_by_username.get(user.username):
                    self.sockets_by_username.pop(user.username, None)

            # Ensure grid actor subscriber counts are accurate even on abrupt disconnects.
            active = self._active_contexts.pop(token, set())
            if active:
                for room, grid_id, filters_json in active:
                    try:
                        gf = (
                            msgspec.json.decode(filters_json)
                            if isinstance(filters_json, (bytes, bytearray, memoryview))
                            else (filters_json or {})
                        )
                        ctx = RoomContext(room=room, grid_id=grid_id, grid_filters=gf)
                        await self.grid_system.user_unsubscribe(ctx, websocket)
                    except Exception:
                        pass

            try:
                await self.router.unsubscribe_all(websocket)
            except Exception:
                pass

            # Clean up Arrow IPC reactive subscriptions
            arrow_client_id = self._arrow_client_id_for(token)
            if arrow_client_id:
                try:
                    await self.arrow_sub_registry.unregister_client(arrow_client_id)
                except Exception:
                    pass
                try:
                    await self.arrow_room_registry.unregister_client(arrow_client_id)
                except Exception:
                    pass
                self._arrow_token_to_client.pop(token, None)

            if ENABLE_PER_SOCKET_SENDER:
                t = self._socket_tasks.pop(token, None)
                if t:
                    t.cancel()
                    with contextlib.suppress(Exception):
                        await t
                self._socket_queues.pop(token, None)

            self._ws_timeouts.pop(token, None)
            self._clear_lazy_columns_for_token(token)
            _release_token(websocket)
        finally:
            self._cleaning_up.discard(token)

    async def on_disconnect(self, websocket: WebSocket):
        await self._cleanup_socket(websocket)

    async def _socket_sender_loop(self, ws: WebSocket, q: asyncio.Queue):
        token = _subscriber_token(ws)
        try:
            while _is_connected(ws):
                bs = await q.get()
                try:
                    await asyncio.wait_for(ws.send_bytes(bs), timeout=self._get_write_timeout(ws))
                    self._ws_timeouts.pop(token, None)
                except asyncio.TimeoutError:
                    c = self._ws_timeouts[token] + 1
                    self._ws_timeouts[token] = c
                    await log.websocket_error(f"ws send timeout ({c}/{CLIENT_WRITE_TIMEOUTS_BEFORE_CLOSE})")
                    if c >= CLIENT_WRITE_TIMEOUTS_BEFORE_CLOSE:
                        await self._cleanup_socket(ws)
                        break
                except WebSocketDisconnect:
                    await self._cleanup_socket(ws)
                    break
                except RuntimeError as e:
                    await log.websocket_error(f"ws send runtime error: {e}")
                    await self._cleanup_socket(ws)
                    break
                except Exception as e:
                    await log.websocket_error(f"ws send loop error: {e}")
                    await self._cleanup_socket(ws)
                    break
                finally:
                    q.task_done()
        except (Exception, asyncio.CancelledError):
            pass

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    async def is_subscribed(self, websocket: WebSocket, context: RoomContext):
        try:
            actor = await self.grid_system.get_actor(context, create_on_missing=False)
        except KeyError:
            return False
        key = (actor.context.room, actor.context.grid_id, actor.context.json_filters)
        token = _subscriber_token(websocket)
        return key in self._active_contexts.get(token, set())

    async def subscribe(self, websocket: WebSocket, context: RoomContext, trace: str = None, action_key: str = "subscribe", *, columns: Optional[List[str]] = None, include_full_schema: bool = False):
        actor = await self.grid_system.get_actor(context, create_on_missing=True)
        key = (actor.context.room, actor.context.grid_id, actor.context.json_filters)

        token = _subscriber_token(websocket)
        if key not in self._active_contexts[token]:
            self._active_contexts[token].add(key)
            await self.grid_system.user_subscribe(actor.context, websocket, create_on_missing=True)

        await self.router.subscribe(websocket, actor.room, context.json_filters)

        # Track per-subscriber lazy column set for delta filtering
        if include_full_schema and columns is not None:
            pk_cols = actor.store.pk_cols if actor.store else None
            self._set_lazy_columns(websocket, context.room, context.grid_id, columns, pk_cols)
        else:
            # Full subscriber — clear any prior lazy tracking for this subscription
            self._set_lazy_columns(websocket, context.room, context.grid_id, None)

        return await self._send_initial_snapshot(
            websocket, actor, action_key=action_key, trace=trace,
            columns=columns, include_full_schema=include_full_schema,
        )

    async def unsubscribe(self, websocket: WebSocket, context: RoomContext):
        try:
            await self.router.unsubscribe(websocket, context.room)
        finally:
            token = _subscriber_token(websocket)
            key = (context.room, context.grid_id, context.json_filters)
            if key in self._active_contexts.get(token, set()):
                self._active_contexts[token].discard(key)
                with contextlib.suppress(Exception):
                    await self.grid_system.user_unsubscribe(context, websocket)
            # Clean up lazy column tracking for this subscription
            self._lazy_columns.pop((token, _upper(context.room), context.grid_id), None)
        return True

    # -------------------------------------------------------------------------
    # Per-subscriber lazy column tracking
    # -------------------------------------------------------------------------

    def _set_lazy_columns(self, websocket: WebSocket, room: str, grid_id: str,
                          columns: Optional[List[str]], pk_cols=None):
        """Register a lazy subscription's initially loaded columns.
        Pass columns=None to mark as a full (non-lazy) subscription.
        """
        key = (_subscriber_token(websocket), _upper(room), grid_id)
        if columns is None:
            self._lazy_columns.pop(key, None)
            self._lazy_tokens.get(key[0], set()).discard(key)
        else:
            col_set = set(columns)
            if pk_cols:
                col_set.update(pk_cols)
            self._lazy_columns[key] = col_set
            self._lazy_tokens[key[0]].add(key)

    def _expand_lazy_columns(self, websocket: WebSocket, room: str, grid_id: str,
                             new_columns: List[str]):
        """Widen a lazy subscription's loaded column set after a fetch_columns."""
        key = (_subscriber_token(websocket), _upper(room), grid_id)
        existing = self._lazy_columns.get(key)
        if existing is not None:
            existing.update(new_columns)

    def _get_lazy_columns(self, token: SubscriberId, room: str, grid_id: str) -> Optional[Set[str]]:
        """Return the loaded column set for a subscription, or None if full."""
        return self._lazy_columns.get((token, _upper(room), grid_id))

    def _clear_lazy_columns_for_token(self, token: SubscriberId):
        """Remove all lazy column tracking for a subscriber (disconnect cleanup)."""
        for k in self._lazy_tokens.pop(token, set()):
            self._lazy_columns.pop(k, None)

    def list_subscriptions(self, websocket: WebSocket):
        return self.router.subscriptions_for(websocket)

    def get_stats(self):
        return self.router.get_stats()

    # -------------------------------------------------------------------------
    # Message dispatch
    # -------------------------------------------------------------------------

    async def handle_message(self, websocket: WebSocket, message: Any | BroadcastMessage):
        trace = self._extract_trace(message)
        try:
            action = Message.get_str(message.get("action")) if isinstance(message, dict) else message.action
        except Exception as e:
            await log.websocket(f"Message decode error: {e}")
            return await self._send_direct_message(websocket, Error(trace=trace).error("Malformed message, unknown action"))

        await log.notify(f"Dispatching: {action}")
        if self._DISPATCH is None:
            self._DISPATCH = {
                "identify": self._handle_identify,
                "subscribe": self._handle_subscribe,
                "unsubscribe": self._handle_unsubscribe,
                "update_filter": self._handle_update_filter,
                "publish": self._handle_publish,
                "duplicate": self._handle_duplicate,
                "trash": self._handle_trash,
                "ping": self._handle_ping,
                "sync": self._handle_sync,
                "refresh": self._handle_refresh,
                "push": self._handle_push,
                "feedback": self._handle_feedback,
                "upload": self._handle_upload,
                "execute": self._handle_execute,
                "disconnect": self._handle_disconnect,
                "fetch_columns": self._handle_fetch_columns,
                "fetch_schema": self._handle_fetch_schema,

                # Arrow IPC reactive subscriptions
                "arrow_subscribe": self._handle_arrow_subscribe,
                "arrow_unsubscribe": self._handle_arrow_unsubscribe,
                "arrow_update": self._handle_arrow_update,
                "arrow_unsubscribe_all": self._handle_arrow_unsubscribe_all,
                "arrow_list": self._handle_arrow_list,
                "arrow_fetch": self._handle_arrow_fetch,
                "arrow_batch": self._handle_arrow_batch,
                # Arrow IPC room subscriptions
                "arrow_join_room": self._handle_arrow_join_room,
                "arrow_leave_room": self._handle_arrow_leave_room,
                "arrow_update_room": self._handle_arrow_update_room,
                "arrow_list_rooms": self._handle_arrow_list_rooms,
                # Arrow IPC export
                "arrow_export": self._handle_arrow_export,

                # Micro-grid
                "micro_publish": self._handle_micro_publish,
                "micro_subscribe": self._handle_micro_subscribe,
                "micro_unsubscribe": self._handle_micro_unsubscribe,
                # Redistribute proceeds
                "redistribute": self._handle_redistribute,
            }
        dispatch = self._DISPATCH.get(action)

        if not dispatch:
            await log.websocket_error(f"Unknown message action: {action}")
            return await self._send_direct_message(websocket, Error(trace=trace).error(f"Unknown action: '{action}'"))

        try:
            t = dispatch(websocket, message)
            return t, action
        except Exception as e:
            await log.websocket_error(f"Dispatch error: {e}")

    # -------------------------------------------------------------------------
    # Handlers
    # -------------------------------------------------------------------------

    async def _build_payload(self, message: Any, payload_type: Any | BroadcastMessage, *, websocket=None):
        if isinstance(message, payload_type): return message
        trace = message.get("trace", None)

        if websocket is not None:
            user_obj = message.get("user", None)
            if user_obj is None:
                try:
                    user_obj = self.get_user_by_socket(websocket)
                    user_obj = user_obj.to_dict() if hasattr(user_obj, "to_dict") else dict(user_obj)
                except Exception as e:
                    await log.error(e)
                    user_obj = {"username": "error"}
            message["user"] = user_obj
        else:
            if message.get("user", None) is None:
                message["user"] = ServerUser().to_dict()
            else:
                u = ServerUser().to_dict()
                u.update(message.get("user", {}))
                message["user"] = u

        try:
            return payload_type(**message)
        except Exception as e:
            await log.error(f"Failed to build payload ({_safe_att(payload_type, '__name__')}): {e}")
            if websocket is not None:
                await self._send_direct_message(websocket, Error(trace=trace).error("Failed to build payload"))
            return None

    @classmethod
    def _extract_trace(cls, message):
        try:
            trace = _safe_att(message, "trace")
            if trace is not None:
                return trace
            if isinstance(message, dict):
                return message.get("trace", None)
            return None
        except Exception:
            return None

    @classmethod
    def _extract_context(cls, message: Union[dict, Message]):
        return RoomContext(**message.get("context")) if isinstance(message, dict) else message.context

    async def _handle_identify(self, websocket: WebSocket, d: Dict[str, Any]):
        token = _subscriber_token(websocket)
        info = self.get_user_by_token(token)
        trace = self._extract_trace(d)

        if info is None:
            return await self._send_direct_message(websocket, Error(trace=trace).error("Failed to identify user."))

        ipub = d.get("user", {})
        if ipub is None:
            return await self._send_direct_message(websocket, Error(trace=trace).error("Failed to identify user."))

        i = {**info, **ipub}
        p = Identify(**i, trace=trace, suppress_context=True)
        self.set_user(token, websocket, p.user)

        connections = len(self.user_count[p.user.username])
        await log.websocket(
            f"Welcome {p.user.displayName} (#{connections})",
            username=p.user.username,
            client=f"{p.user.client_ip}:{p.user.client_port}",
        )
        return await self._send_direct_message(websocket, p.success(keep_user=True, as_ack=False))

    async def _handle_subscribe(self, websocket: WebSocket, d: Dict[str, Any]):
        p = await self._build_payload(d, Subscribe, websocket=websocket)
        if p is None:
            return

        context = p.context
        room = context.room
        grid_id = context.grid_id
        trace = self._extract_trace(d)

        if not room or not grid_id:
            return await self._send_direct_message(websocket, p.error("Missing room or grid_id"))

        if not await self._authorize(websocket, "subscribe", context):
            return await self._send_direct_message(websocket, p.error("You are not authorized for this action."))


        columns = list(context.columns) if context.columns else None
        include_full_schema = True

        key_pt_id_cols = {'tnum', 'portfolioKey', 'isin', 'side', 'rfqListId', 'quoteType', 'isReal', 'description', 'desigName', 'assignedTrader'}
        key_pt_static_cols = {'emProductType', 'yieldCurvePosition', 'ratingCombined'}
        key_pt_size_cols = {'grossSize', 'grossDv01', 'netSize', 'netDv01'}
        key_pt_risk_cols = {'liqScoreCombined', 'bvalSnapshot', "firmAggBsrSize", "firmAggBsiSize", "isMarketAxe", 'isBidAxe', 'isAskAxe', "isInAlgoUniverse", 'isDnt', "firmAggPosition"}
        key_pt_nl_cols = {'newLevel', 'newLevelPx', 'newLevelSpd', 'newLevelMmy', 'newLevelYld', 'newLevelDm'}
        key_pt_skew_cols = {'relativeSkewValue', 'skewType', 'relativeSkewTargetMkt', 'relativeSkewTargetQuoteType', 'relativeSkewTargetSide'}
        ket_pt_cols = key_pt_id_cols | key_pt_static_cols | key_pt_size_cols | key_pt_nl_cols | key_pt_skew_cols | key_pt_risk_cols;

        if (grid_id == 'portfolio') and (columns is not None):
            columns = list(set(columns) | ket_pt_cols | self.market_columns)

        ok = await self.subscribe(
            websocket, context, trace,
            columns=columns,
            include_full_schema=include_full_schema,
        )

        if not ok:
            return await self._send_direct_message(websocket, p.error(f"Failed to subscribe to: {room}"))

    async def _handle_unsubscribe(self, websocket: WebSocket, d: Dict[str, Any]):
        p = await self._build_payload(d, Unsubscribe, websocket=websocket)
        if p is None: return
        ok = await self.unsubscribe(websocket, p.context)
        if not ok:
            return await self._send_direct_message(websocket, p.error(f"Failed to subscribe to: {p.context.room}"))
        return await self._send_direct_message(websocket, p.success())

    async def _handle_update_filter(self, websocket, d: Dict[str, Any]):
        trace = self._extract_trace(d)
        try:
            p = await self._build_payload(d, UpdateFilter, websocket=websocket)
            trace = self._extract_trace(d)
            room = p.context.room
            grid_id = p.context.grid_id

            if not room or not grid_id:
                return await self._send_direct_message(websocket, Error(trace=trace).error("Missing room or grid_id"))
            if not await self._authorize(websocket, "update_filter", p.context):
                return await self._send_direct_message(websocket, p.error("Not authorized"))

            if await self.is_subscribed(websocket, p.context):
                try:
                    await asyncio.wait_for(self.router.update_filter(websocket, room, p.context.json_filters), 5)
                    actor = await self.grid_system.get_actor(p.context, create_on_missing=True)
                    # Preserve lazy column state: if the subscriber has a tracked
                    # lazy column set, send only those columns + full schema.
                    token = _subscriber_token(websocket)
                    existing_cols = self._get_lazy_columns(token, room, grid_id)
                    return await self._send_initial_snapshot(
                        websocket, actor, action_key="update_filter", trace=trace,
                        columns=list(existing_cols) if existing_cols is not None else None,
                        include_full_schema=existing_cols is not None,
                    )
                except (Exception, asyncio.TimeoutError) as e:
                    await log.error(f"FAILED to update filters: {e}")
                    await log.error(f"Re-subscribing to {room} with fresh state")

            # Fallback: re-subscribe using context.columns for lazy loading
            columns = list(p.context.columns) if p.context.columns else None
            ok = await self.subscribe(
                websocket, p.context, trace, action_key="update_filter",
                columns=columns,
                include_full_schema=columns is not None,
            )
            if not ok:
                return await self._send_direct_message(websocket, p.error(f"Failed to update filters for: {room}"))

        except Exception as e:
            await log.websocket_error(f"update_filter failed: {e}")
            return await self._send_direct_message(websocket, Error(trace=trace).error("Failed to update filter"))

    async def _handle_publish(self, websocket: Optional[WebSocket], d: Dict[str, Any]):
        trace = self._extract_trace(d)
        p: Publish = await self._build_payload(d, Publish, websocket=websocket)
        if p is None:
            await log.error(f"Failed to publish payload: {d}")
            await self._send_direct_message(websocket, Error(suppress_context=True).error("Failed to publish payload"))
            return

        if (websocket is not None) and (not await self._authorize(websocket, "publish", p.context)):
            await log.error("Improper access attempted: publish")
            return await self._send_direct_message(
                websocket,
                p.error("Not authorized",
                        toastMessage="You are not authorized for this action. No changes have been persisted.",
                        ),
            )

        if p.context.primary_keys is None:
            p.context.primary_keys = await query_primary_keys(p.context.grid_id)

        actor = await self.grid_system.get_actor(p.context, create_on_missing=True)

        outbounds = []
        async for outbound_payload in self.grid_system.ingest_publish(p):
            outbounds.append(self.ctx.spawn(self.outboundBatcher.add_message(outbound_payload)))
        await asyncio.gather(*outbounds, return_exceptions=True)

    # -------------------------------------------------------------------------
    # Micro-grid handlers
    # -------------------------------------------------------------------------

    async def _handle_micro_publish(self, websocket: Optional[WebSocket], d: Dict[str, Any]):
        trace = self._extract_trace(d)
        micro_name = d.get("micro_name")
        if not micro_name:
            return await self._send_direct_message(
                websocket, Error(trace=trace, suppress_context=True).error("Missing micro_name")
            )

        payloads = (d.get("data") or {}).get("payloads") or d.get("payloads") or {}
        user_info = self.get_user_by_socket(websocket) if websocket else {}
        user = user_info.get("displayName", None)

        # Auto-stamp updated_at / updated_by on add/update payloads
        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cnt = 0
        for key in ("add", "update"):
            tbls = payloads.get(key)
            if tbls is not None:
                new_tbls = []
                for tbl in ensure_list(tbls):
                    exprs = [pl.lit(now_str, pl.String).alias('updated_at')]
                    if user is not None:
                        exprs.append(pl.lit(user, pl.String).alias('updated_by'))

                    if isinstance(tbl, pl.DataFrame):
                        new_tbls.append(tbl.with_columns(exprs))
                    elif isinstance(tbl, dict):
                        tbl['updated_at'] = now_str
                        if user is not None:
                            tbl['updated_by'] = user
                        new_tbls.append(tbl)
                    else:
                        new_tbls.append(tbl)
                payloads[key] = new_tbls
                cnt+=1
        try:
            outbounds = []
            await log.notify(f"Ingesting {cnt} micro update categories")
            async for outbound in self.grid_system.ingest_micro_publish(
                    micro_name, payloads, user=user, trace=trace
            ):
                if isinstance(outbound, MicroPublish):
                    await log.micro('broadcasting micro message')
                    outbounds.append(self.ctx.spawn(
                        self._broadcast_micro(outbound)
                    ))
                else:
                    # Regular Publish from rules engine cross-grid writes
                    await log.micro('broadcasting normal message')
                    outbounds.append(self.ctx.spawn(
                        self.outboundBatcher.add_message(outbound)
                    ))
            await asyncio.gather(*outbounds, return_exceptions=True)
        except KeyError as e:
            await log.error(f"[MicroGrid] Unknown micro-grid: {micro_name}")
            return await self._send_direct_message(
                websocket, Error(trace=trace, suppress_context=True).error(f"Unknown micro-grid: {micro_name}")
            )
        except Exception as e:
            await log.error(f"[MicroGrid] Publish error: {e}")
            return await self._send_direct_message(
                websocket, Error(trace=trace, suppress_context=True).error(f"Micro-grid publish failed: {e}")
            )

    async def _handle_micro_subscribe(self, websocket: WebSocket, d: Dict[str, Any]):
        trace = self._extract_trace(d)
        micro_name = d.get("micro_name")
        if not micro_name:
            return await self._send_direct_message(
                websocket, Error(trace=trace, suppress_context=True).error("Missing micro_name")
            )

        try:
            from app.services.redux.micro_grid import get_micro_actor
            actor = get_micro_actor(micro_name)
        except KeyError:
            return await self._send_direct_message(
                websocket, Error(trace=trace, suppress_context=True).error(f"Unknown micro-grid: {micro_name}")
            )

        # Register subscriber on the actor
        actor.add_subscriber(websocket)

        # Subscribe to the router topic for real-time updates
        topic = f"MICRO.{micro_name.upper()}"
        await self.router.subscribe(websocket, topic)

        # Send initial snapshot
        snapshot = actor.snapshot_as_rows()

        response = {
            "action": "micro_subscribe",
            "micro_name": micro_name,
            "data": {
                "snapshot": snapshot,
                "pk_columns": list(actor.config.primary_keys),
                "columns": list(actor.config.columns.keys()),
            },
            "context": {
                "room": topic,
                "grid_id": actor.config.grid_id,
            },
            "status": {"code": 200, "success": True},
            "trace": trace,
        }
        await self._send_direct_message(websocket, response)

    async def _handle_micro_unsubscribe(self, websocket: WebSocket, d: Dict[str, Any]):
        trace = self._extract_trace(d)
        micro_name = d.get("micro_name")
        if not micro_name:
            return

        try:
            from app.services.redux.micro_grid import get_micro_actor
            actor = get_micro_actor(micro_name)
            actor.remove_subscriber(websocket)
        except KeyError:
            pass

        topic = f"MICRO.{micro_name.upper()}"
        try:
            await self.router.unsubscribe(websocket, topic)
        except Exception:
            pass

    async def _broadcast_micro(self, micro_pub: MicroPublish) -> int:
        """Broadcast a MicroPublish delta to all subscribers of the micro-grid topic."""
        await log.micro("BROADCASTING MICRO CHANGE")
        topic = f"MICRO.{micro_pub.micro_name.upper()}"
        payload_dict = micro_pub.to_dict()
        return await self.broadcast_to_room(topic, payload_dict)

    async def _handle_trash(self, websocket: WebSocket, d: Dict[str, Any]):
        trace = self._extract_trace(d)

        # Build a typed Trash payload — normalises context, user, data, payloads
        p = await self._build_payload(d, Trash, websocket=websocket)
        if p is None:
            return await self._send_direct_message(websocket, Error(trace=trace).error("Failed to build trash payload"))

        context = p.context
        data = d.get('data', {})

        portfolio_key = data.get('portfolioKey')
        date = data.get('date')

        if (portfolio_key is None) or (date is None):
            return await self._send_direct_message(websocket, Error(trace=trace).error("Missing either portfolioKey or Date"))

        try:
            # 1) Warn any subscribers currently viewing this portfolio
            pt_ctx, _ = _build_generic_room_context(portfolio_key)
            pt_actor = self.grid_system.registry.get(pt_ctx)
            if pt_actor is not None and pt_actor.subscriber_count() > 0:
                subs = await pt_actor.list_subscribers()
                if subs:
                    warn_toast = Ack(
                        trace=trace,
                        context=pt_ctx,
                        status_reason="trash",
                        toastType="warning",
                        toastTitle="Portfolio Deleted",
                        toastMessage=f"Portfolio {portfolio_key} has been deleted by another user.",
                    )
                    await self.send_to_multiple(subs, warn_toast)

            # 2) Delete from database (both meta and portfolio tables)
            await self.db_handler.remove_pt_by_keys(portfolio_key)

            # 3) Shutdown and purge the portfolio grid actor cache
            try:
                await self.grid_system.shutdown_actor(pt_ctx, purge_store=True, unsubscribe_subscribers=True)
            except Exception:
                pass

            # 4) Ingest remove through the grid system so the *.META actor cache is updated
            #    AND the outbound publish is broadcast to all subscribers.
            remove_df = pl.DataFrame({"portfolioKey": [portfolio_key], "date": [date]}) if date else pl.DataFrame({"portfolioKey": [portfolio_key]})
            pub = Publish(
                context=context,
                remove=remove_df,
                pk_columns=context.primary_keys or ["portfolioKey", "date"],
                options=PayloadOptions(broadcast=True, persist=False, trigger_rules=True),
                trace=trace,
            )
            outbounds = []
            outbounds.append(self.ctx.spawn(self.outboundBatcher.add_message(pub)))
            async for outbound in self.grid_system.ingest_publish(pub):
                outbounds.append(self.ctx.spawn(self.outboundBatcher.add_message(outbound)))
            await asyncio.gather(*outbounds, return_exceptions=True)

            # 5) Send success ack to the requesting client
            return await self._send_direct_message(
                websocket,
                Ack(
                    trace=trace,
                    context=context,
                    status_reason="trash",
                    toastType="success",
                    toastTitle="Trash",
                    toastMessage=f"Portfolio {portfolio_key} has been deleted.",
                ),
            )

        except Exception as e:
            await log.error(f"[trash] error: {e}")
            return await self._send_direct_message(websocket, Error(trace=trace).error(f"Failed to trash portfolio: {e}"))

    async def _handle_duplicate(self, websocket: WebSocket, d: Dict[str, Any]):
        trace = self._extract_trace(d)

        # Build a typed Duplicate payload — normalises context, user, data, payloads
        p = await self._build_payload(d, Duplicate, websocket=websocket)
        if p is None:
            return await self._send_direct_message(websocket, Error(trace=trace).error("Failed to build duplicate payload"))

        context = p.context
        data = d.get('data', {})
        source_key = data.get('portfolioKey')

        if not source_key:
            return await self._send_direct_message(websocket, p.error("Missing portfolioKey"))

        try:
            # 1) Read the source meta and portfolio from database
            source_meta = await self.db_handler.select_meta(source_key)
            source_pt = await self.db_handler.select_pt(source_key)

            if source_meta.is_empty():
                return await self._send_direct_message(websocket, p.error(f"Portfolio {source_key} not found"))

            # 2) Generate a new portfolio key
            new_key = generate_portfolio_key()

            # 3) Clone meta under the new key, mark as copy
            exprs = [
                pl.lit(new_key, pl.String).alias("portfolioKey"),
                pl.lit(0, pl.Int8).alias('isReal')
            ]
            client_col = "client"
            if client_col in source_meta.columns:
                orig_client = source_meta.item(0, client_col)
                copy_label = f"(COPY) {orig_client}" if orig_client else "(COPY)"
                exprs.append(pl.lit(copy_label).alias(client_col))

            new_meta = source_meta.with_columns(exprs)

            # 4) Clone portfolio rows under the new key
            if source_pt.is_empty():
                return await self._send_direct_message(websocket, p.error(f"Portfolio {source_key} is empty"))

            # Persist portfolio
            new_pt = source_pt.with_columns(pl.lit(new_key, pl.String).alias("portfolioKey"))
            await self.db_handler.upsert("portfolio", new_pt, )

            # 5) Persist the new meta
            await self.db_handler.upsert("meta", new_meta)

            # 6) Ingest add through the grid system so the *.META actor cache is updated
            #    AND the outbound publish is broadcast to all subscribers.
            pub = Publish(
                context=context,
                add=new_meta,
                pk_columns=context.primary_keys or ["portfolioKey", "date"],
                options=PayloadOptions(broadcast=True, persist=False, trigger_rules=False),
                trace=trace,
            )

            outbounds = []
            outbounds.append(self.ctx.spawn(self.outboundBatcher.add_message(pub)))
            async for outbound in self.grid_system.ingest_publish(pub):
                outbounds.append(self.ctx.spawn(self.outboundBatcher.add_message(outbound)))
            await asyncio.gather(*outbounds, return_exceptions=True)

            # 7) Send success ack to the requesting client
            return await self._send_direct_message(
                websocket,
                Ack(
                    trace=trace,
                    context=context,
                    status_reason="duplicate",
                    toastType="success",
                    toastTitle="Duplicate",
                    toastMessage=f"Portfolio duplicated as {new_key}",
                ),
            )

        except Exception as e:
            await log.error(f"[duplicate] error: {e}")
            return await self._send_direct_message(websocket, Error(trace=trace).error(f"Failed to duplicate portfolio: {e}"))

    async def _handle_ping(self, websocket: WebSocket, d: Dict[str, Any]):
        p = await self._build_payload(d, Ping, websocket=websocket)
        if p is None: return
        return await self._send_direct_message(websocket, p.success())

    async def _handle_sync(self, websocket: WebSocket, d: Dict[str, Any]):
        p = await self._build_payload(d, Sync, websocket=websocket)
        if p is None: return
        data = d.get("data") or {}

        n = int(data.get("n", 1))
        suppressLoad = data.get("suppressLoad", False)
        suppressCompare = data.get("suppressCompare", False)
        dates = data.get("dates", None)

        await log.sync('Checking for updates', n=n, dates=dates, suppressLoad=suppressLoad, suppressCompare=suppressCompare)

        try:
            self.ctx.spawn(self.ksm.ksm.sync_n_with_kdb(n=n, suppressLoad=suppressLoad, suppressCompare=suppressCompare, dates=dates))
        except Exception as e:
            await log.websocket_error(f"Failed to schedule manual sync: {e}")
            return await self._send_direct_message(websocket, p.error("Failed to schedule manual sync"))
        return await self._send_direct_message(websocket, p.success())

    async def _handle_feedback(self, websocket: WebSocket, d: Dict[str, Any]):
        p = await self._build_payload(d, Feedback, websocket=websocket)
        if (p is None) or (_safe_att(p, 'data') is None) or (_safe_att(p.data, 'feedback') is None):
            return await self._send_direct_message(websocket, Error().error("Malformed feedback"))

        fb = p.data.feedback
        user = getattr(p.user, "displayName", "<unknown>")
        trace = getattr(p.trace, "trace", str(_uuid.uuid4()))
        await log.feedback(f"Feedback received ({user})", type=fb.feedbackType, feedback=fb.feedbackText)

        await self.db_handler.upsert('feedback', pl.DataFrame([{
            "timestamp": isonow(),
            "user": user,
            "feedbackType": fb.feedbackType,
            "feedbackText": fb.feedbackText,
            "trace": trace
        }]))
        return await self._send_direct_message(websocket, p.success())

    async def _handle_push(self, websocket: WebSocket, d: Dict[str, Any]):
        quick_context = self._extract_context(d)
        if quick_context is None:
            await log.error("[push] missing context")
            return

        if quick_context.grid_id!='portfolio': return

        # Always target the portfolio grid of the supplied room's key
        room = quick_context.room
        key = room.split(".")[0] if room else None
        if not key:
            await log.error("[push] invalid room/key", room=room)
            return

        from app.services.kdb.publish import publish_to_ptinternaldata
        try:
            event = d.get("data", {}).get('event', '')
            user_d = d.get("user", {})
            user = User(username=user_d.get('username', None), displayName=user_d.get('displayName', None))
            await publish_to_ptinternaldata(portfolioKey=key.lower(), event=event, user=user)
            opts = d.get('options', {})
            if not (opts.get('silent', False)):
                await self.toast_success(quick_context, "Successfully pushed basket to KDB", toastTitle="Push to KDB")
        except Exception as e:
            await self.toast_error(quick_context, "Failed to push basket to KDB", toastTitle="Push to KDB")
            await log.error("[push] unable to push basket to KDB", error=e)

    # -------------------------------------------------------------------------
    # Lazy Column Loading
    # -------------------------------------------------------------------------

    _MAX_COLUMNS_PER_FETCH = 200

    async def _handle_fetch_columns(self, websocket: WebSocket, d: Dict[str, Any]):
        """
        Handle a lazy column fetch request from the frontend.

        Expected message format:
        {
            "action": "fetch_columns",
            "context": { "room": "...", "grid_id": "..." },
            "columns": ["col1", "col2", "col3"],
            "trace": "..."
        }

        Responds with a partial Arrow IPC snapshot containing only the requested columns.
        """
        trace = self._extract_trace(d)

        try:
            quick_context = self._extract_context(d)
        except (TypeError, KeyError, Exception) as e:
            await log.error(e)
            quick_context = None
        if quick_context is None:
            await log.error("[fetch_columns] missing context")
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("Missing context for fetch_columns")
            )

        if not await self._authorize(websocket, "subscribe", quick_context):
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("You are not authorized for this action.")
            )

        columns = d.get("columns", [])
        if not columns or not isinstance(columns, list):
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("fetch_columns requires a non-empty 'columns' list")
            )

        if len(columns) > self._MAX_COLUMNS_PER_FETCH:
            return await self._send_direct_message(
                websocket, Error(trace=trace).error(
                    f"fetch_columns limited to {self._MAX_COLUMNS_PER_FETCH} columns per request"
                )
            )

        try:
            actor = await self.grid_system.get_actor(quick_context, create_on_missing=True)
        except KeyError:
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("No active grid for this context")
            )
        if actor.store is None:
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("No active grid for this context")
            )

        try:
            # Filter to only user-visible columns that exist in the store
            pk_set = set(actor.store.pk_cols) if actor.store.pk_cols else set()
            available_cols = (
                {c for c in actor.store.cols.id_to_name
                 if c not in pk_set and c != _INTERNAL_ROW_ALIVE}
                if hasattr(actor.store, 'cols') else set()
            )
            valid_cols = [c for c in columns if c in available_cols]

            if not valid_cols:
                return await self._send_direct_message(
                    websocket, Error(trace=trace).error("None of the requested columns are available")
                )

            # Expand the lazy column set BEFORE materializing so that any concurrent
            # broadcast deltas arriving on the event loop include these columns.
            self._expand_lazy_columns(websocket, quick_context.room, quick_context.grid_id, valid_cols)

            df = await actor.materialize_running_frame(cols=valid_cols, include_removed=False)
            if not isinstance(df, pl.DataFrame):
                return await self._send_direct_message(
                    websocket, Error(trace=trace).error("Failed to materialize requested columns")
                )

            meta = {
                "action": "fetch_columns",
                "grid_id": quick_context.grid_id,
                "room": quick_context.room,
                "columns": valid_cols,
            }
            raw = self._build_arrow_snapshot_bytes(df, meta)
            dict_payload = {
                "action": "fetch_columns",
                "context": meta,
                "columns": valid_cols,
                "data": raw,
                "trace": trace,
            }
            return await self._send_direct_message(websocket, dict_payload)

        except Exception as e:
            await log.error(f"[fetch_columns] error: {e}")
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("Internal error fetching columns")
            )

    async def _handle_fetch_schema(self, websocket: WebSocket, d: Dict[str, Any]):
        """
        Handle a schema-only fetch request.

        Returns the full schema (all column names and types) without any row data.
        Used by the frontend to know all available columns for lazy loading.

        Expected message format:
        {
            "action": "fetch_schema",
            "context": { "room": "...", "grid_id": "..." },
            "trace": "..."
        }
        """
        trace = self._extract_trace(d)

        try:
            quick_context = self._extract_context(d)
        except (TypeError, KeyError, Exception):
            quick_context = None
        if quick_context is None:
            await log.error("[fetch_schema] missing context")
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("Missing context for fetch_schema")
            )

        if not await self._authorize(websocket, "subscribe", quick_context):
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("You are not authorized for this action.")
            )

        try:
            actor = await self.grid_system.get_actor(quick_context, create_on_missing=False)
        except KeyError:
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("No active grid for this context")
            )
        if actor.store is None:
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("No active grid for this context")
            )

        try:
            # Build an empty DataFrame with the full schema (all columns, zero rows)
            pk_set = set(actor.store.pk_cols) if actor.store.pk_cols else set()
            all_col_names = [
                c for c in actor.store.cols.id_to_name
                if c not in pk_set and c != _INTERNAL_ROW_ALIVE
            ]

            probe_cols = [all_col_names[0]] if all_col_names else []
            if probe_cols:
                probe_df = await actor.materialize_running_frame(cols=probe_cols, include_removed=False)
                if not isinstance(probe_df, pl.DataFrame):
                    return await self._send_direct_message(
                        websocket, Error(trace=trace).error("Failed to materialize schema")
                    )
                # Build a zero-row frame with all column names using the probe's schema for types
                schema_df = probe_df.head(0)
            else:
                schema_df = pl.DataFrame()

            meta = {
                "action": "fetch_schema",
                "grid_id": quick_context.grid_id,
                "room": quick_context.room,
                "all_columns": all_col_names,
            }

            if actor.store.pk_cols:
                meta["primary_keys"] = actor.store.pk_cols

            raw = self._build_arrow_snapshot_bytes(schema_df, meta)
            dict_payload = {
                "action": "fetch_schema",
                "context": meta,
                "all_columns": all_col_names,
                "data": raw,
                "trace": trace,
            }
            return await self._send_direct_message(websocket, dict_payload)

        except Exception as e:
            await log.error(f"[fetch_schema] error: {e}")
            return await self._send_direct_message(
                websocket, Error(trace=trace).error("Internal error fetching schema")
            )

    async def _handle_refresh(self, websocket: WebSocket, d: Dict[str, Any]):
        from app.services.kdb.publish import publish_to_ptinternaldata
        trace = self._extract_trace(d)
        quick_context = self._extract_context(d)
        if quick_context is None:
            await log.error("[refresh] missing context")
            return

        if quick_context.grid_id!='portfolio': return

        actor = await self.grid_system.get_actor(quick_context)
        frame = await actor.materialize_running_frame(cols=ensure_list(actor.store.pk_cols), include_removed=True)
        stamp = isonow(True)  # ISO string in UTC
        update_df = frame.with_columns(pl.lit(stamp, pl.String).alias("refSyncTime"))

        # Emit in chunks as UPDATE-only; do not persist/broadcast this sentinel
        opts = PayloadOptions(broadcast=True, persist=True, trigger_rules=True, relay=True)
        n = update_df.hyper.height()
        pub = self.grid_system.build_publish_from_df(context=actor.context, df=update_df, mode="update", trace=trace, options=opts)
        tid = await self.toast_loading(quick_context, "Refreshing reference markets", toastTitle="Refresh", toastOptions={"persist": True})
        try:
            await self._handle_publish(websocket, pub)
            await self.toast_success(quick_context, "Refreshing reference complete", toastTitle="Refresh", toastId=tid, toastOptions={"persist": False}, trace=trace)
            await self.broadcast_to_room(quick_context.room, Ack(context=quick_context, trace=trace, status_reason="refresh"))
            await publish_to_ptinternaldata(quick_context.room.split(".")[0].lower(), event="Refresh Markets")
        except Exception as e:
            return await self._send_direct_message(websocket, pub.error(f"Error refreshing: {e}"))

    async def _handle_upload(self, websocket: WebSocket, d: Dict[str, Any]):

        from app.services.payload.payloadV4 import Ack
        from app.services.portfolio.meta import create_meta_for_kdb_portfolio

        trace = self._extract_trace(d)

        TOAST_PERSIST_OPTS = {"persist": True, "timeOut": 0, "extendedTimeOut": 0}
        TOAST_DONE_OPTS = {"timeOut": 6000}
        MAX_ROWS = 50_000

        def _parse_asof(asof):
            if asof is None:
                return None, None
            s = str(asof).strip()
            if not s:
                return None, None
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                date_str = dt.date().isoformat()
                time_str = dt.time().replace(microsecond=0).isoformat()
                return date_str, time_str
            except Exception:
                m = re.match(r"(\d{4}-\d{2}-\d{2})(?:[T\s](\d{2}:\d{2})(?::(\d{2}))?)?", s)
                if not m:
                    return (s[:10] if len(s) >= 10 else None), None
                date_part = m.group(1)
                hhmm = m.group(2)
                ss = m.group(3) or "00"
                return date_part, (f"{hhmm}:{ss}" if hhmm else None)

        def _normalize_side(v):
            if v is None: return "BUY"
            s = str(v).strip().upper()
            if not s: return "BUY"
            if s in BUY_TYPES: return "BUY"
            if s in SELL_TYPES: return "SELL"
            return "BUY"

        def _normalize_quote_type(v):
            if v is None: return "PX"
            s = str(v).strip().upper()
            if not s: return "PX"
            if s in PRICE_TYPES: return "PX"
            if s in MMY_TYPES: return "MMY"
            if s in SPREAD_TYPES: return "SPD"
            return "PX"

        data = (d or {}).get("data") or {}
        meta = data.get("meta") or {}
        rows = data.get("rows") or []
        toast_id = data.get("toastId") or f"upload-{_uuid.uuid4().hex}"

        await self._send_direct_message(
            websocket,
            Ack(
                trace=trace,
                context=d['context'],
                status_reason="upload_portfolio",
                toastType="info",
                toastTitle="Upload",
                toastMessage="Validating upload…",
                toastId=toast_id,
                toastOptions=TOAST_PERSIST_OPTS,
            ),
        )

        if not isinstance(rows, list) or not rows:
            return await self._send_direct_message(
                websocket,
                Ack(
                    trace=trace,
                    context=d['context'],
                    status_reason="upload_portfolio",
                    toastType="error",
                    toastTitle="Upload",
                    toastMessage="No rows to upload.",
                    toastId=toast_id,
                    toastOptions=TOAST_DONE_OPTS,
                ),
            )

        if len(rows) > MAX_ROWS:
            return await self._send_direct_message(
                websocket,
                Ack(
                    trace=trace,
                    context=d['context'],
                    status_reason="upload_portfolio",
                    toastType="error",
                    toastTitle="Upload",
                    toastMessage=f"Too many rows ({len(rows)}). Limit is {MAX_ROWS}.",
                    toastId=toast_id,
                    toastOptions=TOAST_DONE_OPTS,
                ),
            )

        portfolio_key = generate_portfolio_key()
        client = str(meta.get("client") or meta.get("name") or "Manual Portfolio").strip() or "Manual Portfolio"
        region = str(meta.get("region") or "US").strip() or "US"
        venue = str(meta.get("venue") or "MANUAL").strip() or "MANUAL"
        state = str(meta.get("state") or "NEW").strip() or "NEW"
        rfq_state = str(meta.get("rfqState") or "MANUAL").strip() or "MANUAL"
        ttp = meta.get("timeToPrice") or 0

        as_of = meta.get("asOf") or meta.get("date")
        date_str, time_str = _parse_asof(as_of)

        from app.helpers.date_helpers import now_date, now_time
        if not date_str:
            date_str = now_date(utc=True)
        if not time_str:
            time_str = now_time(utc=True)

        rfqIds, ids, isins, cusips, descs, sizes, sides, qts, tnums = [], [], [], [], [], [], [], [], []
        tnum = 1
        for r in rows:
            if not isinstance(r, dict):
                continue
            sec_id = str(r.get("id") or r.get("isin") or r.get("cusip") or "").strip()
            if not sec_id:
                continue

            ids.append(sec_id)
            # keep meta unique counts sane even if user pasted cusips only
            isins.append(str(r.get("isin") or sec_id or "").strip())
            cusips.append(str(r.get("cusip") or "").strip())
            descs.append(str(r.get("description") or "").strip())

            try:
                sizes.append(float(r.get("size") or 0))
            except Exception:
                sizes.append(0.0)

            sides.append(_normalize_side(r.get("direction") or r.get("side")))
            qts.append(_normalize_quote_type(r.get("quoteType")))
            tnums.append(str(tnum))
            rfqIds.append("manual-" + str(tnum))
            tnum += 1

        if not ids:
            return await self._send_direct_message(
                websocket,
                Ack(
                    trace=trace,
                    context=d['context'],
                    status_reason="upload_portfolio",
                    toastType="error",
                    toastTitle="Upload",
                    toastMessage="No valid rows (missing id).",
                    toastId=toast_id,
                    toastOptions=TOAST_DONE_OPTS,
                ),
            )

        await self._send_direct_message(
            websocket,
            Ack(
                trace=trace,
                context=d['context'],
                status_reason="upload_portfolio",
                toastType="info",
                toastTitle="Upload",
                toastMessage=f"Building portfolio ({len(ids)} rows)…",
                toastId=toast_id,
                toastOptions=TOAST_PERSIST_OPTS,
            ),
        )

        from app.helpers.date_helpers import next_biz_date_from_today, now_date
        sd = next_biz_date_from_today().strftime("%Y-%m-%d")
        cd = now_date()

        pt_df = pl.DataFrame(
            {
                "portfolioKey": [portfolio_key] * len(ids),
                "rfqId": rfqIds,
                "tnum": tnums,
                "id": ids,
                "isin": isins,
                "cusip": cusips,
                "size": sizes,
                "side": sides,
                "settleDate": [sd] * len(ids),
                "rfqLeg": [0] * len(ids),
                "rfqListId": [portfolio_key] * len(ids),
                "rfqBenchmark": [None] * len(ids),
                "rfqDescription": [None] * len(ids),
                "rfqTicker": [None] * len(ids),
                "rfqCreateDate": [cd] * len(ids),
                "quoteType": qts,
            }
        )
        return
        # from app.services.loaders.load_sequence import load_portfolio
        # start = time.time()
        # my_pt = await load_portfolio(pt_df.lazy(), dates=None, portfolio_key=portfolio_key, return_loader=None, broadcaster=self.ksm.ksm)
        # load_time = time.time() - start
        #
        # await self._send_direct_message(
        #     websocket,
        #     Ack(
        #         trace=trace,
        #         context=d['context'],
        #         status_reason="upload_portfolio",
        #         toastType="info",
        #         toastTitle="Upload",
        #         toastMessage="Building meta…",
        #         toastId=toast_id,
        #         toastOptions=TOAST_PERSIST_OPTS,
        #     ),
        # )
        #
        # client_data = {
        #     "client": client,
        #     "state": state,
        #     "rfqState": rfq_state,
        #     "venue": venue,
        #     "venueShort": venue,
        #     "timeToPrice": ttp,
        #     "rfqListId": None,
        # }
        #
        # my_meta = await create_meta_for_kdb_portfolio(
        #     my_pt,
        #     constituents=None,
        #     portfolio_key=portfolio_key,
        #     region=region,
        #     client_data=client_data,
        #     manualFlag=True,
        #     loadTime=load_time
        # )
        #
        # meta_cols = set(my_meta.columns)
        # override_exprs = []
        # if "date" in meta_cols:
        #     override_exprs.append(pl.lit(date_str).cast(pl.Utf8).alias("date"))
        # if "time" in meta_cols:
        #     # ensure HH:MM:SS for downstream formatting
        #     if len(time_str)==5:
        #         time_str = f"{time_str}:00"
        #     override_exprs.append(pl.lit(time_str).cast(pl.Utf8).alias("time"))
        #
        # if override_exprs:
        #     my_meta = my_meta.with_columns(override_exprs)
        #
        #     if {"date", "time", "datetime"}.issubset(meta_cols):
        #         my_meta = my_meta.with_columns(
        #             pl.concat_str([pl.col("date"), pl.lit(" "), pl.col("time")]).alias("_dt_str")
        #         ).with_columns(
        #             pl.col("_dt_str")
        #             .str.to_datetime(format="%Y-%m-%d %H:%M:%S", time_zone="UTC")
        #             .alias("datetime")
        #         ).drop("_dt_str")
        #
        #     if {"datetime", "datetimeEt"}.issubset(meta_cols):
        #         my_meta = my_meta.with_columns(
        #             pl.col("datetime").dt.convert_time_zone("America/New_York").alias("datetimeEt")
        #         )
        #
        #     if {"datetimeEt", "dateEt"}.issubset(meta_cols):
        #         my_meta = my_meta.with_columns(pl.col("datetimeEt").dt.date().cast(pl.Utf8).alias("dateEt"))
        #
        #     if {"datetimeEt", "timeEt"}.issubset(meta_cols):
        #         my_meta = my_meta.with_columns(pl.col("datetimeEt").dt.time().cast(pl.Utf8).alias("timeEt"))
        #
        #     if {"datetime", "timeToPrice", "dueTime"}.issubset(meta_cols):
        #         my_meta = my_meta.with_columns(
        #             (pl.col("datetime") + (pl.col("timeToPrice") * 1000).cast(pl.Duration("ms"))).alias("dueTime")
        #         )
        #
        #     if {"datetimeEt", "timeToPrice", "dueTimeEt"}.issubset(meta_cols):
        #         my_meta = my_meta.with_columns(
        #             (pl.col("datetimeEt") + (pl.col("timeToPrice") * 1000).cast(pl.Duration("ms"))).alias("dueTimeEt")
        #         )

        # await self.ksm.ksm.publish_portfolio(portfolio_key, my_pt, my_meta)
        #
        # return await self._send_direct_message(
        #     websocket,
        #     Ack(
        #         trace=trace,
        #         context=d['context'],
        #         status_reason="upload_portfolio",
        #         toastType="success",
        #         toastTitle="Upload",
        #         toastMessage=f"Uploaded {client} ({portfolio_key})",
        #         toastId=toast_id,
        #         toastOptions=TOAST_DONE_OPTS,
        #         result={"portfolioKey": portfolio_key, "date": date_str},
        #     ),
        # )

    async def _handle_execute(self, websocket: WebSocket, d: Dict[str, Any]):
        trace = self._extract_trace(d)
        p = await self._build_payload(d, Execute, websocket=websocket)
        if p is None:
            return await self._send_direct_message(websocket, Error(trace=trace).error("Failed to build execute payload"))

        context = p.context
        if not context.room or not context.grid_id:
            return await self._send_direct_message(websocket, p.error("Missing room or grid_id"))

        if not await self._authorize(websocket, "execute", context):
            return await self._send_direct_message(websocket, p.error("Not authorized", toastType='error', toastTitle="Not Authorized", toastMessage="You are not authorized for this action."))

        data = d.get("data") or {}
        portfolio_key = data.get("portfolioKey")
        func_name_raw = data.get("funcName")
        bundle_payload = bool(data.get("bundlePayload", False))

        if not portfolio_key:
            return await self._send_direct_message(websocket, p.error("Missing portfolioKey"))
        if not func_name_raw:
            return await self._send_direct_message(websocket, p.error("Missing funcName"))

        # Normalise to list — accept "myFunc" or ["myFunc", "otherFunc"]
        func_names = func_name_raw if isinstance(func_name_raw, list) else [func_name_raw]
        func_names = [str(f).strip() for f in func_names if f]

        for fn_name in func_names:
            if not _IDENT.match(fn_name):
                return await self._send_direct_message(websocket, p.error(f"Invalid function name: {fn_name}"))

        label = ", ".join(func_names) if len(func_names) <= 3 else f"{len(func_names)} functions"
        tid = await self.toast_loading(context, f"Executing {label}…", toastTitle="Execute", toastOptions={"persist": True})

        # ── Helper: send trace-based progress event to the requesting client ──
        async def _send_trace_event(status: str, detail: str = ""):
            """Sends an ack-style event keyed by the original trace so the
            frontend ExecutePanel can correlate spinners / status indicators.
            status: "running" | "success" | "error"
            """
            await self._send_direct_message(websocket, {
                "action": "execute_progress",
                "trace": trace,
                "data": {
                    "executeStatus": status,
                    "funcNames": func_names,
                    "detail": detail,
                },
            })

        try:
            import importlib
            mod = importlib.import_module("app.services.loaders.kdb_queries_dev_v2")

            # -- Resolve every function up-front before running any --------
            resolved = []
            for fn_name in func_names:
                fn = getattr(mod, fn_name, None)
                if fn is None or not asyncio.iscoroutinefunction(fn):
                    await _send_trace_event("error", f"Function '{fn_name}' not found or not async")
                    return await self._send_direct_message(websocket, p.error(
                        f"Function '{fn_name}' not found or not async",
                        toastType='error', toastTitle="Execute", toastId=tid, toastOptions={"persist": False}
                    ))
                resolved.append((fn_name, fn))

            if context.primary_keys is None:
                context.primary_keys = await query_primary_keys(context.grid_id)
            actor = await self.grid_system.get_actor(context, create_on_missing=True)
            opts = PayloadOptions(broadcast=True, persist=True, trigger_rules=True, relay=True)

            # -- Execute each function and collect frames -------------------
            bundled_frames = []
            total_rows = 0

            for fn_name, fn in resolved:
                result = await fn(portfolio_key)

                if isinstance(result, pl.LazyFrame):
                    result_df = result.collect()
                elif isinstance(result, pl.DataFrame):
                    result_df = result
                else:
                    await _send_trace_event("error", f"Function '{fn_name}' did not return a DataFrame/LazyFrame")
                    return await self._send_direct_message(websocket, p.error(
                        f"Function '{fn_name}' did not return a DataFrame/LazyFrame",
                        toastType='error', toastTitle="Execute", toastId=tid, toastOptions={"persist": False}
                    ))

                if result_df.is_empty():
                    continue

                total_rows += result_df.shape[0]

                if bundle_payload:
                    bundled_frames.append(result_df)
                else:
                    # Fire immediately — each function becomes its own Publish
                    pub = Publish(context=context, update=result_df, pk_columns=context.primary_keys, options=opts, trace=trace)
                    async for outbound_payload in self.grid_system.ingest_publish(pub):
                        self.ctx.spawn(self.outboundBatcher.add_message(outbound_payload))

            # -- If bundling, concat and fire a single Publish --------------
            if bundle_payload and bundled_frames:
                merged = pl.concat(bundled_frames, how="diagonal_relaxed")
                pub = Publish(context=context, update=merged, pk_columns=context.primary_keys, options=opts, trace=trace)
                outbounds = []
                async for outbound_payload in self.grid_system.ingest_publish(pub):
                    outbounds.append(self.ctx.spawn(self.outboundBatcher.add_message(outbound_payload)))
                await asyncio.gather(*outbounds, return_exceptions=True)

            # ── Terminal trace event: success ──
            if total_rows==0:
                await _send_trace_event("success", f"{label}: no rows returned")
                await self.toast_success(context, f"{label}: no rows returned", toastTitle="Execute", toastId=tid, toastOptions={"persist": False}, trace=trace)
            else:
                await _send_trace_event("success", f"Executed {label} ({total_rows} rows)")
                await self.toast_success(context, f"Executed {label} ({total_rows} rows)", toastTitle="Execute", toastId=tid, toastOptions={"persist": False}, trace=trace)

        except Exception as e:
            await log.error(f"[execute] error: {e}")
            await _send_trace_event("error", str(e))
            return await self._send_direct_message(websocket, p.error(
                f"Execute failed: {e}",
                toastType='error', toastTitle="Execute", toastId=tid, toastOptions={"persist": False}
            ))

    # -------------------------------------------------------------------------
    # Redistribute Proceeds
    # -------------------------------------------------------------------------

    def _read_redist_params(self):
        """
        Read live solver parameters from the redist_params micro-grid actor.
        Returns (charge_strength, default_strength, risk_weights, solver_defaults).
        Falls back to empty dicts (letting OptimizerConfig.__post_init__ fill defaults)
        if the microgrid is unavailable.
        """
        from app.services.redux.micro_grid import get_micro_actor

        charge_strength: dict[tuple, float] = {}
        default_strength: float | None = None
        risk_weights: dict[str, float] = {}
        solver_defaults: dict[str, float] = {}

        try:
            actor = get_micro_actor("redist_params")
            rows = actor.snapshot_as_rows()
        except Exception:
            return charge_strength, default_strength, risk_weights, solver_defaults

        # Key mapping for charge_strength: "BSR_PX" -> ("BSR", "PX")
        _CS_KEY_MAP = {
            "BSR_PX": ("BSR", "PX"),
            "BSI_PX": ("BSI", "PX"),
            "BSR_SPD": ("BSR", "SPD"),
            "BSI_SPD": ("BSI", "SPD"),
        }

        for row in rows:
            cat = row.get("category", "")
            key = row.get("param_key", "")
            val = row.get("param_value")
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue

            if cat == "charge_strength":
                if key == "DEFAULT":
                    default_strength = val
                elif key in _CS_KEY_MAP:
                    charge_strength[_CS_KEY_MAP[key]] = val

            elif cat == "risk_weight":
                risk_weights[key] = val

            elif cat == "solver_default":
                solver_defaults[key] = val

            else:
                solver_defaults[key] = val

        return charge_strength, default_strength, risk_weights, solver_defaults

    async def _handle_redistribute(self, websocket: WebSocket, d: Dict[str, Any]):

        trace = self._extract_trace(d)
        quick_context = self._extract_context(d)
        options = d.get('solverOptions', {})
        # --- Resolve portfolio grid actor ---
        try:
            actor = await self.grid_system.get_actor(quick_context, create_on_missing=True)
        except Exception as e:
            await log.error(f"[redistribute] actor lookup failed: {e}")
            return await self._send_direct_message(websocket, {
                "action": "redistribute",
                "trace": trace,
                "data": {"error": f"No active grid: {e}"},
            })


        try:
            base_frame = await actor.materialize_running_frame(include_removed=False)
            base_frame = base_frame.lazy() if isinstance(base_frame, pl.DataFrame) else base_frame
            grid_frame = base_frame

            skews = pl.DataFrame(d.get('data', {}).get('params', {}).get('skews', []))

            if (skews is not None) and (not skews.hyper.is_empty()):

                # _refMidPx, _refMidSpd
                IS_PX = pl.col('quoteType') == 'PX'
                IS_SPD = pl.col('quoteType') == 'SPD'

                s = grid_frame.hyper.schema()
                MKT_SOURCES = ['_ref', 'bval', 'macp', 'am', 'trace', 'cbbt']
                def mg(side, qt, s):
                    t = [("_" if m.startswith("_") else "") + clean_camel(m, side, qt) for m in MKT_SOURCES]
                    return [col for col in t if col in s]

                mkts = grid_frame.select([
                    pl.col('tnum'),
                    pl.coalesce(mg('bid','spd', s)).alias('_fallback_bid_spd'),
                    pl.coalesce(mg('mid', 'spd', s)).alias('_fallback_mid_spd'),
                    pl.coalesce(mg('ask', 'spd', s)).alias('_fallback_ask_spd'),
                    pl.coalesce(mg('bid', 'px', s)).alias('_fallback_bid_px'),
                    pl.coalesce(mg('mid', 'px', s)).alias('_fallback_mid_px'),
                    pl.coalesce(mg('ask', 'px', s)).alias('_fallback_ask_px'),
                ])

                my_skews = skews.lazy().with_columns([
                    pl.when(IS_PX).then(pl.col('refBid').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, pl.Float64)).alias('_refBidPx'),
                    pl.when(IS_PX).then(pl.col('refMid').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, pl.Float64)).alias('_refMidPx'),
                    pl.when(IS_PX).then(pl.col('refAsk').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, pl.Float64)).alias('_refAskPx'),
                    pl.when(IS_SPD).then(pl.col('refBid').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, pl.Float64)).alias('_refBidSpd'),
                    pl.when(IS_SPD).then(pl.col('refMid').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, pl.Float64)).alias('_refMidSpd'),
                    pl.when(IS_SPD).then(pl.col('refAsk').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, pl.Float64)).alias('_refAskSpd'),
                ]).join(
                    other=mkts.lazy(), on='tnum', how='left', maintain_order="none"
                ).with_columns([
                    pl.when(pl.col('_refBidPx').is_not_null()).then(pl.col('_refBidPx')).otherwise(pl.col('_fallback_bid_px')).alias('_refBidPx'),
                    pl.when(pl.col('_refMidPx').is_not_null()).then(pl.col('_refMidPx')).otherwise(pl.col('_fallback_mid_px')).alias('_refMidPx'),
                    pl.when(pl.col('_refAskPx').is_not_null()).then(pl.col('_refAskPx')).otherwise(pl.col('_fallback_ask_px')).alias('_refAskPx'),
                    pl.when(pl.col('_refBidSpd').is_not_null()).then(pl.col('_refBidSpd')).otherwise(pl.col('_fallback_bid_spd')).alias('_refBidSpd'),
                    pl.when(pl.col('_refMidSpd').is_not_null()).then(pl.col('_refMidSpd')).otherwise(pl.col('_fallback_mid_spd')).alias('_refMidSpd'),
                    pl.when(pl.col('_refAskSpd').is_not_null()).then(pl.col('_refAskSpd')).otherwise(pl.col('_fallback_ask_spd')).alias('_refAskSpd'),
                ]).select([
                    pl.col('tnum'),
                    pl.col('_refBidPx'), pl.col('_refMidPx'), pl.col('_refAskPx'),
                    pl.col('_refBidSpd'), pl.col('_refMidSpd'), pl.col('_refAskSpd'),
                ])

                grid_frame = ensure_lazy(grid_frame)
                grid_frame = grid_frame.join(my_skews, on='tnum', how='left')

            _DEFAULT_LIQ_SCORE=1
            s = grid_frame.hyper.schema()
            if 'avgLiqScore' not in s:
                grid_frame = grid_frame.with_columns(pl.col('liqScoreCombined').alias('avgLiqScore'))
            if 'firmBsrSize' not in s:
                grid_frame = grid_frame.with_columns([
                    pl.col('firmAggBsrSize').alias('firmBsrSize'),
                    pl.col('firmAggBsiSize').alias('firmBsinSize')
                ])
            if 'isLocked' not in s:
                grid_frame = grid_frame.with_columns([
                    pl.lit(0, pl.Int8).alias('isLocked')
                ])

            grid_frame = grid_frame.with_columns([
                pl.col('avgLiqScore').hyper.fill_null(_DEFAULT_LIQ_SCORE, include_zero_as_null=True).alias('avgLiqScore'),
                pl.col('isLocked').hyper.fill_null(0).alias('isLocked'),
            ])

            from app.services.rules.portfolio.redist_new import solve, OptimizerConfig

            isolate_traders = options.get('isolate_traders', False)
            debug_mode = options.get('debug', False)

            # Skew cap options (None = uncapped)
            max_individual_spread_skew_delta = options.get('max_individual_spread_skew_delta', None)
            max_individual_px_skew_delta = options.get('max_individual_px_skew_delta', None)

            # Dynamic group columns for "Match Pivot Groups"
            group_columns = options.get('group_columns', None)
            if group_columns is not None and not isinstance(group_columns, list):
                group_columns = None

            # Sanitize: ensure float or None
            def _opt_float(v):
                if v is None:
                    return None
                try:
                    f = float(v)
                    return f if f >= 0 else None
                except (TypeError, ValueError):
                    return None

            # --- Read live solver params from redist_params microgrid ---
            charge_strength, default_strength, risk_weights, solver_defaults = \
                self._read_redist_params()

            # Parse "Run From X" overrides: {"BUY|PX": 2.0, "SELL|SPD": -1.5} → {("BUY","PX"): 2.0, ...}
            def _parse_skew_override(raw):
                if not raw or not isinstance(raw, dict):
                    return None
                result = {}
                for k, v in raw.items():
                    if v is None or v == '':
                        continue
                    try:
                        parts = k.split('|')
                        if len(parts) == 2:
                            result[(parts[0], parts[1])] = float(v)
                    except (ValueError, TypeError):
                        pass
                return result if result else None

            buffer_mode = 'pct' if solver_defaults.get('buffer_mode', 1) == 1 else 'fixed'

            config = OptimizerConfig(
                # Side constraint (wavg control)
                side_floor_pct=options.get('side_floor_pct', solver_defaults.get('side_floor_pct', 0.99)),

                # Trader band guardrails
                buffer_mode=buffer_mode,
                buffer_fixed=options.get('buffer_fixed', solver_defaults.get('buffer_fixed', 500.0)),
                buffer_pct=options.get('buffer_pct', solver_defaults.get('buffer_pct', 0.25)),
                buffer_min_flat=options.get('buffer_min_flat', solver_defaults.get('buffer_min_flat', 200.0)),

                # Three competing penalties
                lambda_param=options.get('lambda_param', solver_defaults.get('lambda_param', 0.25)),
                gamma=options.get('gamma', solver_defaults.get('gamma', 1.0)),
                trader_pull=options.get('trader_pull', solver_defaults.get('trader_pull', 1.0)),

                # Target allocation
                target_blend=options.get('target_blend', solver_defaults.get('target_blend', 0.5)),
                bsr_risk_share_weight=options.get('bsr_risk_share_weight', solver_defaults.get('bsr_risk_share_weight', 0.7)),
                bsr_preference_weight=options.get('bsr_preference_weight', solver_defaults.get('bsr_preference_weight', 0.3)),

                # Charge curve
                charge_strength=charge_strength,
                default_strength=default_strength,
                risk_weights=risk_weights,

                # Skew caps
                max_individual_spread_skew_delta=_opt_float(max_individual_spread_skew_delta),
                max_individual_px_skew_delta=_opt_float(max_individual_px_skew_delta),

                # Behavioral
                allow_through_mid=bool(options.get('allow_through_mid', solver_defaults.get('allow_through_mid', 1))),
                isolate_traders=isolate_traders,
                anchor_wavg_skew=bool(options.get('anchor_wavg_skew', False)),
                skew_asymmetry=options.get('skew_asymmetry', solver_defaults.get('skew_asymmetry', 0)),

                # Solver
                debug=debug_mode,
                group_columns=group_columns,

                # "Run From X" starting skew override
                starting_skew_override=_parse_skew_override(options.get('starting_skew_override')),
            )
            await log.debug("Entering solver...", color="gray")
            df_result, result, summary_overall, summary_trader, removed_ids = solve(grid_frame, config)
            await log.debug('Solver complete.', color="gray")

            if (df_result is None) or df_result.hyper.is_empty():
                return await self._send_direct_message(websocket, {
                    "action": "redistribute",
                    "trace": trace,
                    "data": {"error": f"No bonds available to solve: {removed_ids}"},
                })

            # Build update payload mapped back to original grid column names
            df_result = df_result.join(base_frame.select('tnum', 'description'), on='tnum', how='left')
            j_frame = await base_frame.select('tnum', 'portfolioKey', 'skewType').hyper.collect_async()
            i_frame = await df_result.select('tnum', 'final_skew', 'implied_px', 'implied_spd','quoteType').hyper.collect_async()
            by_skew = i_frame.join(j_frame, on='tnum', how='inner').partition_by(['skewType', 'quoteType'], include_key=True, as_dict=True)

            updates = (
                by_skew.get((0, "PX"), pl.DataFrame({'tnum':[], 'implied_px':[]})).select(
                    pl.col('tnum'),
                    pl.col('implied_px').alias('newLevelPx')
                ).to_dicts() +
                by_skew.get((1, "PX"), pl.DataFrame({'tnum':[], 'implied_px':[], 'final_skew':[]})).select(
                    pl.col('tnum'),
                    pl.col('final_skew').alias('refSkew'),
                ).to_dicts() +
                by_skew.get((0, "SPD"), pl.DataFrame({'tnum': [], 'implied_spd': []})).select(
                    pl.col('tnum'),
                    pl.col('implied_spd').alias('newLevelSpd'),
                ).to_dicts() +
                by_skew.get((1, "SPD"), pl.DataFrame({'tnum': [], 'implied_spd': [], 'final_skew':[]})).select(
                    pl.col('tnum'),
                    pl.col('final_skew').alias('refSkew'),
                ).to_dicts()
            )

            response_data = {
                "result": [result.status],
                "diagnostics": getattr(result, 'diagnostics', {}),
                "detail": (await df_result.hyper.collect_async()).sort([
                    pl.col('description'),
                    pl.col('skew_delta'),
                    pl.col('avgLiqScore')
                ]).to_dicts(),
                "summary": (await summary_overall.hyper.collect_async()).to_dicts(),
                "trader": (await summary_trader.sort([pl.col('quoteType'), pl.col('side'), pl.col('wavg_skew_delta'), pl.col('wavg_liq_score')]).hyper.collect_async()).to_dicts(),
                "updates": updates,
                "removed": removed_ids,
                "group_columns": group_columns,
            }

        except Exception as e:
            await log.error(f"[redistribute] Server Error: {e}")
            return await self._send_direct_message(websocket, {
                "action": "redistribute",
                "trace": trace,
                "data": {"error": f"Server Error: {e}"},
            })

        # --- Send result back to the requesting client only ---
        import orjson
        return await self._send_direct_message(websocket, orjson.dumps({
            "action": "redistribute",
            "trace": trace,
            "data": response_data,
        }), suppress_encode=False)

    # -------------------------------------------------------------------------
    # Emit Loop (batch → sockets)
    # -------------------------------------------------------------------------

    async def _emit_loop(self):
        while self._outbound_running:
            try:
                async for message in self.outboundBatcher:
                    try:
                        if isinstance(message, BroadcastMessage):
                            room = message.context.room
                        elif isinstance(message, dict):
                            room = (message.get("context") or {}).get("room", None) or UNKNOWN_ROOM
                        else:
                            room = UNKNOWN_ROOM

                        await self.broadcast_to_room(room, message)

                    except Exception as e:
                        await log.error(f"[Emit] loop error: {e}")
            except asyncio.CancelledError:
                return
            except Exception as e:
                await log.error(f"[Emit] loop crash, restarting: {e}")
                await asyncio.sleep(1)

    def _retarget_room_in_dict(self, payload_dict: dict, room: str):
        out = payload_dict.copy()
        ctx = dict(out.get("context") or {})
        ctx["room"] = room
        out["context"] = ctx
        return out

    def _decode_payloads_to_rows(self, pdict: dict):
        try:
            if pdict.get("action") != "publish":
                return pdict
            ctx = pdict.get("context") or {}
            if (ctx.get("grid_id") or "").lower() != "meta":
                return pdict

            codec = get_codec()
            payloads = ((pdict.get("data") or {}).get("payloads") or {})
            meta_keep = {k: payloads.get(k) for k in ("based_on", "action_seq") if k in payloads}

            out = {}
            for act in ("add", "update", "remove"):
                act_rows = []
                for item in (payloads.get(act) or []):
                    fr = item.get("frame")
                    if isinstance(fr, dict) and fr.get("_format"):
                        for df in codec.decode_to_polars_partitions(fr):
                            act_rows.extend(df.to_dicts())
                    elif isinstance(fr, pl.DataFrame):
                        act_rows.extend(fr.to_dicts())
                if act_rows:
                    out[act] = act_rows

            if out:
                out.update(meta_keep)
                new_data = dict(pdict.get("data") or {})
                new_data["payloads"] = out
                return dict(pdict, data=new_data)
        except Exception:
            pass
        return pdict

    def _payload_to_dict(self, payload, *, topic: str | None = None) -> Optional[dict]:
        """Convert a payload to its dict form (pre-serialization).
        Returns None for raw bytes or payloads that can't be dict-ified.
        """
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return None

        if isinstance(payload, dict):
            pkg = self._retarget_room_in_dict(payload, topic) if topic else payload
            return self._decode_payloads_to_rows(pkg)

        if hasattr(payload, "package"):
            pkg = payload.package()
            if hasattr(pkg, "to_dict"):
                pkg = pkg.to_dict()
            elif not isinstance(pkg, dict):
                to_dict = getattr(payload, "to_dict", None)
                pkg = to_dict() if callable(to_dict) else {"action": getattr(payload, "action", "message")}
            if topic:
                pkg = self._retarget_room_in_dict(pkg, topic)
            return self._decode_payloads_to_rows(pkg)

        return None

    @staticmethod
    def _filter_payload_dict_columns(pkg: dict, columns: frozenset) -> Optional[dict]:
        """Filter a dict-form Publish payload to only include columns in *columns*.

        Returns a shallow-modified copy of *pkg* with each delta's frame dict
        trimmed to the intersection of its keys and *columns*.  For 'update'
        deltas whose changed_columns have no overlap with *columns* the delta
        is dropped entirely.  Returns None if the payload is not a publish or
        no deltas survive filtering.
        """
        if pkg.get("action") != "publish":
            return pkg  # non-publish payloads pass through unmodified

        data = pkg.get("data")
        if not data:
            return pkg
        payloads = data.get("payloads")
        if not payloads:
            return pkg

        new_payloads = {}
        any_deltas = False
        for act in ("add", "update", "remove"):
            items = payloads.get(act)
            if not items:
                continue
            filtered = []
            for delta in items:
                frame = delta.get("frame")
                changed = delta.get("changed_columns")

                # For updates: skip the entire delta if none of the changed
                # columns overlap with what this subscriber has loaded.
                if act == "update" and changed:
                    if not columns.intersection(changed):
                        continue

                # Filter the frame dict to only subscribed columns.
                if isinstance(frame, dict):
                    # pk_columns are always in the subscriber's column set
                    # so they survive filtering automatically.
                    narrowed = {k: v for k, v in frame.items() if k in columns}
                    if not narrowed:
                        continue
                    delta = dict(delta, frame=narrowed)
                    # Also narrow changed_columns to avoid confusing the client.
                    if changed:
                        delta["changed_columns"] = [c for c in changed if c in columns]
                filtered.append(delta)

            if filtered:
                new_payloads[act] = filtered
                any_deltas = True

        if not any_deltas:
            return None

        # Preserve metadata keys like based_on / action_seq
        for k in ("based_on", "action_seq"):
            if k in payloads:
                new_payloads[k] = payloads[k]

        new_data = dict(data, payloads=new_payloads)
        return dict(pkg, data=new_data)

    def _encode_for_wire(self, payload, *, topic: str | None = None) -> bytes:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        pkg = self._payload_to_dict(payload, topic=topic)
        if pkg is not None:
            return prep_outgoing_payload(pkg)
        return prep_outgoing_payload(payload)

    def _get_write_timeout(self, websocket: WebSocket) -> float:
        user = self.get_user_by_socket(websocket)
        try:
            t = getattr(user, "write_timeout", None) or user.get("write_timeout")
            if t:
                return float(t)
        except Exception:
            pass
        return float(CLIENT_WRITE_TIMEOUT_S)

    async def _send_direct_message(self, websocket: WebSocket, payload: Union[Message, Dict, bytes, bytearray, memoryview], **kwargs):
        if websocket is None: return
        token = _subscriber_token(websocket)
        if not _is_connected(websocket):
            return False

        suppress_encode = kwargs.pop('suppress_encode', False)
        suppress_compress = kwargs.pop('suppress_compress', False)

        bs = payload if isinstance(payload, (bytes, bytearray, memoryview)) else prep_outgoing_payload(
            payload.package() if hasattr(payload, "package") else payload,
            suppress_encode=suppress_encode,
            suppress_compress=suppress_compress
        )

        # bs = prep_outgoing_payload(payload.package() if hasattr(payload, "package") else payload)


        if not _is_connected(websocket):
            return False

        try:
            await asyncio.wait_for(websocket.send_bytes(bs), timeout=self._get_write_timeout(websocket))
            self._ws_timeouts.pop(token, None)
            return True

        except asyncio.TimeoutError:
            c = self._ws_timeouts[token] + 1
            self._ws_timeouts[token] = c
            await log.websocket_error(f"Send timed out ({c}/{CLIENT_WRITE_TIMEOUTS_BEFORE_CLOSE}); dropping message.")
            if c >= CLIENT_WRITE_TIMEOUTS_BEFORE_CLOSE:
                await self._cleanup_socket(websocket)
            return False

        except WebSocketDisconnect:
            await self._cleanup_socket(websocket)
            return False

        except RuntimeError as e:
            await log.websocket_error(f"Runtime error during send: {e}")
            await self._cleanup_socket(websocket)
            return False

        except Exception as e:
            await log.websocket_error(f"Failed to send message: {e}")
            await self._cleanup_socket(websocket)
            return False

    async def send_to_multiple(self, websockets, payload, *, topic: str | None = None) -> int:
        unique = {ws for ws in (websockets or []) if _is_connected(ws)}
        if not unique:
            return 0
        bs = self._encode_for_wire(payload, topic=topic)
        results = await asyncio.gather(*[self._send_direct_message(ws, bs) for ws in unique], return_exceptions=True)
        return sum(1 for r in results if r is True)

    async def send_to_all_users(self, payload, *, topic: str | None = None) -> int:
        sockets = list(self.socket_by_token.values())
        return await self.send_to_multiple(sockets, payload, topic=topic)

    def _group_by_topic(self, pairs, default_topic: str):
        groups = {}
        for ws, topic in pairs:
            groups.setdefault(topic or default_topic, []).append(ws)
        return groups

    async def broadcast_to_room(self, room: str, payload):
        try:
            recipients = self.router.publish(_upper(room), payload, dedupe_subscribers=True)
        except Exception as e:
            await log.error(e)
            recipients = self.router.get_all_subscribers_for_topic(_upper(room))

        if not recipients:
            return 0

        # Extract grid_id from the payload context for lazy-column lookup.
        grid_id = None
        try:
            ctx = getattr(payload, "context", None)
            if ctx is not None:
                grid_id = getattr(ctx, "grid_id", None)
        except Exception:
            pass

        room_upper = _upper(room)

        try:
            groups = self._group_by_topic(recipients, room_upper)
            send_count = 0
            # Cache: we only compute the dict form once per topic, and only if
            # there are lazy subscribers in that topic group.
            _dict_cache: dict[str, Optional[dict]] = {}

            for topic, conns in groups.items():
                live = [ws for ws in conns if _is_connected(ws)]
                if not live:
                    continue

                # --- Partition into full vs lazy subscribers ------------------
                full_subs = []
                lazy_by_colset: dict[frozenset, list] = {}

                if grid_id and self._lazy_columns:
                    for ws in live:
                        token = _subscriber_token(ws)
                        col_set = self._get_lazy_columns(token, room_upper, grid_id)
                        if col_set is None:
                            # Full subscriber — receives everything
                            full_subs.append(ws)
                        else:
                            key = frozenset(col_set)
                            lazy_by_colset.setdefault(key, []).append(ws)
                else:
                    # No grid_id or no lazy subscribers at all — fast path
                    full_subs = live

                # --- Send to full subscribers (original fast path) -----------
                if full_subs:
                    bs = self._encode_for_wire(payload, topic=topic)
                    results = await asyncio.gather(
                        *[self._send_direct_message(ws, bs) for ws in full_subs],
                        return_exceptions=True,
                    )
                    send_count += sum(1 for r in results if r is True)

                # --- Send column-filtered payloads to lazy groups ------------
                if lazy_by_colset:
                    # Build the dict form once per topic (shared across groups)
                    if topic not in _dict_cache:
                        _dict_cache[topic] = self._payload_to_dict(payload, topic=topic)
                    base_dict = _dict_cache[topic]

                    if base_dict is None:
                        # Raw bytes or non-dictifiable — send unfiltered
                        bs = self._encode_for_wire(payload, topic=topic)
                        all_lazy = [ws for subs in lazy_by_colset.values() for ws in subs]
                        results = await asyncio.gather(
                            *[self._send_direct_message(ws, bs) for ws in all_lazy],
                            return_exceptions=True,
                        )
                        send_count += sum(1 for r in results if r is True)
                    else:
                        for col_fs, subs in lazy_by_colset.items():
                            filtered = self._filter_payload_dict_columns(base_dict, col_fs)
                            if filtered is None:
                                # All deltas filtered out — nothing to send
                                continue
                            bs = prep_outgoing_payload(filtered)
                            results = await asyncio.gather(
                                *[self._send_direct_message(ws, bs) for ws in subs],
                                return_exceptions=True,
                            )
                            send_count += sum(1 for r in results if r is True)

            return send_count
        except Exception as e:
            await log.error(e)
            return 0

    async def broadcast_to_room_exact(self, room: str, payload) -> int:
        recipients = self.router.get_exact_subscribers_for_topic(_upper(room))
        if not recipients:
            return 0
        conns = [ws for (ws, _topic) in recipients if _is_connected(ws)]
        if not conns:
            return 0
        bs = self._encode_for_wire(payload, topic=_upper(room))
        results = await asyncio.gather(*[self._send_direct_message(ws, bs) for ws in conns], return_exceptions=True)
        return sum(1 for r in results if r is True)

    async def broadcast_to_grid_id(self, room: str, payload) -> int:
        recipients = self.router.get_all_subscribers_for_topic(_upper(room))
        if not recipients:
            return 0
        groups = self._group_by_topic(recipients, _upper(room))
        send_count = 0
        for topic, conns in groups.items():
            live = [ws for ws in conns if _is_connected(ws)]
            if not live:
                continue
            bs = self._encode_for_wire(payload, topic=topic)
            results = await asyncio.gather(*[self._send_direct_message(ws, bs) for ws in live], return_exceptions=True)
            send_count += sum(1 for r in results if r is True)
        return send_count

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_arrow_snapshot_bytes(df: pl.DataFrame, meta: dict) -> bytes:
        tbl: pa.Table = df.to_arrow()
        return compress(_arrow_ipc_from_arrow(_package_arrow_metadata(tbl, meta)), 7)

    async def _send_initial_snapshot(self, websocket: WebSocket, actor: GridActor, action_key: str, trace: str = None, *, columns: Optional[List[str]] = None, include_full_schema: bool = False):

        await actor.ensure_awake()
        df = await actor.materialize_running_frame(cols=columns, include_removed=False)
        if df is None:
            await log.warning(f"Grid not found in cache, pulling from DB directly: {actor.context.grid_id}")
            await actor.rebuild_from_db()
            df = await actor.materialize_running_frame(cols=columns, include_removed=False)
        if not isinstance(df, pl.DataFrame):
            raise ValueError(f"Missing frame for: {actor.context.room}")

        try:
            meta = actor.context.to_dict()
            if (actor.store is not None) and (actor.store.pk_cols is not None):
                meta["primary_keys"] = actor.store.pk_cols

            if include_full_schema and (actor.store is not None):
                pk_set = set(actor.store.pk_cols) if actor.store.pk_cols else set()
                all_col_names = [
                    c for c in actor.store.cols.id_to_name
                    if c not in pk_set and c != _INTERNAL_ROW_ALIVE
                ]
                meta["all_columns"] = all_col_names
                meta["lazy_columns_enabled"] = True

            raw = self._build_arrow_snapshot_bytes(df, meta)
            dict_payload = {"action": action_key, "context": meta, "data": raw, "trace": trace}
            await self._send_direct_message(websocket, dict_payload)
            return True
        except Exception as e:
            await log.error(f"[Snapshot] send error for {actor.context.grid_id}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Toasts
    # -------------------------------------------------------------------------

    async def send_toast(self, context: RoomContext, toastMessage, *, toastTitle=None, toastType=None, toastId=None, toastOptions=None, **kwargs):
        toast = Toast(context=context, toastTitle=toastTitle, toastType=toastType, toastId=toastId, toastMessage=toastMessage, toastOptions=toastOptions, **kwargs)
        await self.broadcast_to_room(context.room, toast)
        return toast.data.toast.toastId

    async def toast_info(self, context: RoomContext, toastMessage, *, toastTitle=None, toastId=None, toastOptions=None, **kwargs):
        return await self.send_toast(context=context, toastMessage=toastMessage, toastTitle=toastTitle, toastId=toastId, toastOptions=toastOptions, toastType="info", **kwargs)

    async def toast_warning(self, context: RoomContext, toastMessage, *, toastTitle=None, toastId=None, toastOptions=None, **kwargs):
        return await self.send_toast(context=context, toastMessage=toastMessage, toastTitle=toastTitle, toastId=toastId, toastOptions=toastOptions, toastType="warning", **kwargs)

    async def toast_error(self, context: RoomContext, toastMessage, *, toastTitle=None, toastId=None, toastOptions=None, **kwargs):
        return await self.send_toast(context=context, toastMessage=toastMessage, toastTitle=toastTitle, toastId=toastId, toastOptions=toastOptions, toastType="error", **kwargs)

    async def toast_success(self, context: RoomContext, toastMessage, *, toastTitle=None, toastId=None, toastOptions=None, **kwargs):
        return await self.send_toast(context=context, toastMessage=toastMessage, toastTitle=toastTitle, toastId=toastId, toastOptions=toastOptions, toastType="success", **kwargs)

    async def toast_loading(self, context: RoomContext, toastMessage, *, toastTitle=None, toastId=None, toastOptions=None, **kwargs):
        return await self.send_toast(context=context, toastMessage=toastMessage, toastTitle=toastTitle, toastId=toastId, toastOptions=toastOptions, toastType="loading", persist=True, **kwargs)

    # -------------------------------------------------------------------------
    # Authorization
    # -------------------------------------------------------------------------

    @alru_cache(maxsize=128)
    async def query_auth(self, action: str, username: str):
        await log.auth(f"Querying for... {action}:{username}")
        from app.server import get_db
        try:
            res = await get_db().select("auth", filters={"id": username.lower(), "action": [action, "all"]}, columns=["allowed"])
            if (res is None) or (res.hyper.is_empty()):
                return True
            if res.hyper.height() > 1:
                res = res.filter(pl.col("action") == action)
            return res.hyper.peek("allowed") != 0
        except Exception as e:
            await log.error(f"Auth check failed: {e}", action=action, username=username)
            return False

    async def _authorize(self, websocket, action: str, context: RoomContext, force: bool = False) -> bool:
        token = _subscriber_token(websocket)
        user = self.user_by_socket.get(token)
        if user is None: return True
        if force: self.clear_auth_cache_for_username(action, user.username)
        return await self.query_auth(action, user.username)

    def clear_auth_all_caches(self):
        self.query_auth.cache_clear()

    def auth_cache_info(self):
        return self.query_auth.cache_info()

    def clear_auth_cache_for_username(self, action: str, username: str):
        self.query_auth.cache_invalidate(action, username)

    def _arrow_client_id_for(self, token) -> Optional[str]:
        return self._arrow_token_to_client.get(token)

    def _arrow_ensure_client(self, websocket: WebSocket) -> str:
        """Ensure the websocket is registered in the arrow registries. Returns client_id."""
        token = _subscriber_token(websocket)
        cid = self._arrow_token_to_client.get(token)
        if cid is None:
            cid = str(_uuid.uuid4())
            self._arrow_token_to_client[token] = cid
            # Registration is async but we can't await in a sync context.
            # Callers should call _arrow_register_client() if this returns a new cid.
        return cid

    async def _arrow_register_client(self, websocket: WebSocket) -> str:
        """Register a websocket in arrow registries if not already registered."""
        token = _subscriber_token(websocket)
        cid = self._arrow_token_to_client.get(token)
        if cid is None:
            cid = str(_uuid.uuid4())
            self._arrow_token_to_client[token] = cid
            await self.arrow_sub_registry.register_client(cid, websocket)
            await self.arrow_room_registry.register_client(cid, websocket)
        return cid

    # ── Data loader for Arrow queries ────────────────────────────────────────

    async def _arrow_load_table(self, grid_id: str = "default") -> 'pl.DataFrame':
        """Load a Polars DataFrame for Arrow IPC queries, using the existing db_handler."""
        import polars as pl_local
        db = self.db_handler
        if db is None:
            from app.server import get_db
            db = get_db()
        if hasattr(db, "select_random_pt"):
            result = await db.select_random_pt(lazy=False)
            if isinstance(result, pl_local.DataFrame):
                return result
            return pl_local.DataFrame(result)
        return await db.select(grid_id, as_format="polars")

    # ── Subscription handlers ────────────────────────────────────────────────

    async def _handle_arrow_subscribe(self, websocket: WebSocket, message, preloaded_df=None):
        """Create a reactive Arrow IPC subscription."""
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)

        sub_id = d.get("subId") or str(_uuid.uuid4())
        sub_type = d.get("subType", "rows")
        params = d.get("params", {})
        reactive = d.get("reactive", True)
        touched = extract_touched_columns(sub_type, params)

        sub = ArrowSubscription(
            sub_id=sub_id, sub_type=sub_type, params=params,
            reactive=reactive, touched_columns=touched,
        )
        await self.arrow_sub_registry.add_subscription(cid, sub)

        await self._send_direct_message(websocket, {
            "action": "arrow_subscribed",
            "subId": sub_id, "subType": sub_type, "reactive": reactive, "trace": trace,
        })

        # Push initial data
        df = preloaded_df if preloaded_df is not None else await self._arrow_load_table()
        ipc_bytes, meta = ArrowQueryEngine.recompute_subscription(df, sub)
        sub.last_hash = content_hash(ipc_bytes)

        resp = {"action": "arrow_data", "subId": sub_id, "subType": sub_type,
                "bytes": len(ipc_bytes), "trace": trace, "data": ipc_bytes}
        resp.update(meta)
        await self._send_direct_message(websocket, resp)

        if not reactive:
            await self.arrow_sub_registry.remove_subscription(cid, sub_id)

    async def _handle_arrow_unsubscribe(self, websocket: WebSocket, message):
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        sub_id = d.get("subId")
        if not sub_id:
            return await self._send_direct_message(websocket, {
                "action": "error", "message": "subId required", "trace": trace,
            })
        removed = await self.arrow_sub_registry.remove_subscription(cid, sub_id)
        await self._send_direct_message(websocket, {
            "action": "arrow_unsubscribed", "subId": sub_id, "found": removed, "trace": trace,
        })

    async def _handle_arrow_update(self, websocket: WebSocket, message, preloaded_df=None):
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        sub_id = d.get("subId")
        new_params = d.get("params", {})
        if not sub_id:
            return await self._send_direct_message(websocket, {
                "action": "error", "message": "subId required", "trace": trace,
            })
        updated = await self.arrow_sub_registry.update_subscription(cid, sub_id, new_params)
        if not updated:
            return await self._send_direct_message(websocket, {
                "action": "error", "message": f"Subscription {sub_id} not found", "trace": trace,
            })
        await self._send_direct_message(websocket, {
            "action": "arrow_updated", "subId": sub_id, "params": updated.params, "trace": trace,
        })

        # Push refreshed data
        df = preloaded_df if preloaded_df is not None else await self._arrow_load_table()
        ipc_bytes, meta = ArrowQueryEngine.recompute_subscription(df, updated)
        updated.last_hash = content_hash(ipc_bytes)

        resp = {"action": "arrow_data", "subId": sub_id, "subType": updated.sub_type,
                "bytes": len(ipc_bytes), "trace": trace, "data": ipc_bytes}
        resp.update(meta)
        await self._send_direct_message(websocket, resp)

    async def _handle_arrow_unsubscribe_all(self, websocket: WebSocket, message):
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        count = await self.arrow_sub_registry.remove_all_subscriptions(cid)
        await self._send_direct_message(websocket, {
            "action": "arrow_unsubscribed_all", "count": count, "trace": trace,
        })

    async def _handle_arrow_list(self, websocket: WebSocket, message):
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        subs = await self.arrow_sub_registry.get_client_subs(cid)
        items = [
            {"subId": s.sub_id, "subType": s.sub_type, "reactive": s.reactive, "params": s.params}
            for s in subs.values()
        ]
        await self._send_direct_message(websocket, {
            "action": "arrow_list", "subscriptions": items, "trace": trace,
        })

    async def _handle_arrow_fetch(self, websocket: WebSocket, message, preloaded_df=None):
        """One-shot Arrow IPC fetch without creating a subscription."""
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        sub_type = d.get("subType", "rows")
        params = d.get("params", {})
        req_id = d.get("reqId") or str(_uuid.uuid4())

        df = preloaded_df if preloaded_df is not None else await self._arrow_load_table()
        tmp_sub = ArrowSubscription(sub_id=req_id, sub_type=sub_type, params=params, reactive=False)
        ipc_bytes, meta = ArrowQueryEngine.recompute_subscription(df, tmp_sub)

        resp = {"action": "arrow_fetch_result", "reqId": req_id, "subType": sub_type,
                "bytes": len(ipc_bytes), "trace": trace, "data": ipc_bytes}
        resp.update(meta)
        await self._send_direct_message(websocket, resp)

    async def _handle_arrow_batch(self, websocket: WebSocket, message):
        """Execute multiple arrow actions in a single message."""
        d = message if isinstance(message, dict) else message.to_dict()
        items = d.get("items", [])
        df = await self._arrow_load_table()

        for item in items:
            action = item.get("action")
            if action == "arrow_subscribe":
                await self._handle_arrow_subscribe(websocket, item, preloaded_df=df)
            elif action == "arrow_unsubscribe":
                await self._handle_arrow_unsubscribe(websocket, item)
            elif action == "arrow_update":
                await self._handle_arrow_update(websocket, item, preloaded_df=df)
            elif action == "arrow_fetch":
                await self._handle_arrow_fetch(websocket, item, preloaded_df=df)

    # ── Room subscription handlers ───────────────────────────────────────────

    async def _handle_arrow_join_room(self, websocket: WebSocket, message):
        """Join an Arrow IPC room subscription with filter+column projection."""
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)

        sub_id = d.get("subId") or str(_uuid.uuid4())
        room = d.get("room", "")
        grid_id = d.get("gridId", "default")
        columns = d.get("columns")
        filters = normalize_filters(d.get("filters", [])) or []
        reactive = d.get("reactive", True)
        single_row = d.get("singleRow", False)
        row_key_field = d.get("rowKeyField")

        sub = ArrowRoomSubscription(
            sub_id=sub_id, room=room, grid_id=grid_id,
            columns=columns, filters=filters, reactive=reactive,
            single_row=single_row, row_key_field=row_key_field,
        )
        await self.arrow_room_registry.join_room(cid, sub)

        await self._send_direct_message(websocket, {
            "action": "arrow_room_joined", "subId": sub_id, "room": room,
            "gridId": grid_id, "reactive": reactive, "singleRow": single_row, "trace": trace,
        })

        # Push initial data as Arrow IPC
        df = await self._arrow_load_table(grid_id)
        ipc_bytes, total = ArrowQueryEngine.compute_room_initial(df, sub)
        sub.last_hash = content_hash(ipc_bytes)

        header = {
            "action": "arrow_room_data", "subId": sub_id, "room": room,
            "gridId": grid_id, "totalRows": total, "bytes": len(ipc_bytes),
            "trace": trace, "data": ipc_bytes,
        }
        try:
            await self._send_direct_message(websocket, header)
        except Exception:
            await log.warning(f"Failed to push initial arrow room data to sub {sub_id}")
            await self.arrow_room_registry.leave_room(cid, sub_id)
            raise

        if not reactive:
            await self.arrow_room_registry.leave_room(cid, sub_id)

    async def _handle_arrow_leave_room(self, websocket: WebSocket, message):
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        sub_id = d.get("subId")
        if not sub_id:
            return await self._send_direct_message(websocket, {
                "action": "error", "message": "subId required", "trace": trace,
            })
        found = await self.arrow_room_registry.leave_room(cid, sub_id)
        await self._send_direct_message(websocket, {
            "action": "arrow_room_left", "subId": sub_id, "found": found, "trace": trace,
        })

    async def _handle_arrow_update_room(self, websocket: WebSocket, message):
        """Update filters/columns for a room subscription and push refreshed data."""
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        sub_id = d.get("subId")
        if not sub_id:
            return await self._send_direct_message(websocket, {
                "action": "error", "message": "subId required", "trace": trace,
            })
        new_filters = normalize_filters(d.get("filters")) if d.get("filters") is not None else None
        new_columns = d.get("columns")

        updated = await self.arrow_room_registry.update_room(
            cid, sub_id, filters=new_filters, columns=new_columns)
        if not updated:
            return await self._send_direct_message(websocket, {
                "action": "error", "message": f"Room sub {sub_id} not found", "trace": trace,
            })

        await self._send_direct_message(websocket, {
            "action": "arrow_room_updated", "subId": sub_id, "room": updated.room,
            "filters": updated.filters, "columns": updated.columns, "trace": trace,
        })

        # Push refreshed data
        df = await self._arrow_load_table(updated.grid_id)
        ipc_bytes, total = ArrowQueryEngine.compute_room_initial(df, updated)
        updated.last_hash = content_hash(ipc_bytes)

        header = {
            "action": "arrow_room_data", "subId": sub_id, "room": updated.room,
            "gridId": updated.grid_id, "totalRows": total, "bytes": len(ipc_bytes),
            "trace": trace, "data": ipc_bytes,
        }
        await self._send_direct_message(websocket, header)

    async def _handle_arrow_list_rooms(self, websocket: WebSocket, message):
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        cid = await self._arrow_register_client(websocket)
        subs = await self.arrow_room_registry.get_client_room_subs(cid)
        items = [
            {
                "subId": s.sub_id, "room": s.room, "gridId": s.grid_id,
                "columns": s.columns, "filters": s.filters,
                "reactive": s.reactive, "singleRow": s.single_row,
            }
            for s in subs.values()
        ]
        await self._send_direct_message(websocket, {
            "action": "arrow_room_list", "subscriptions": items, "trace": trace,
        })

    async def _handle_arrow_export(self, websocket: WebSocket, message):
        """Export data as CSV, JSON, Parquet or Arrow IPC via WebSocket."""
        d = message if isinstance(message, dict) else message.to_dict()
        trace = d.get("trace")
        req_id = d.get("reqId") or str(_uuid.uuid4())
        params = d.get("params", {})

        df = await self._arrow_load_table()
        content_bytes, media_type = execute_export(df, params)

        await self._send_direct_message(websocket, {
            "action": "arrow_export_result", "reqId": req_id,
            "format": params.get("format", "csv"), "mediaType": media_type,
            "bytes": len(content_bytes), "trace": trace, "data": content_bytes,
        })

    # ── Reactive push: notify when data changes ─────────────────────────────

    async def notify_arrow_data_changed(self, changed_columns: Optional[Set[str]] = None):
        """Push updated data to all affected Arrow IPC subscriptions.

        Call this when backend data changes. *changed_columns* is the set of
        column names that were modified (None = all columns may have changed).
        """
        import json as _json_notify
        affected = await self.arrow_sub_registry.get_affected(changed_columns)
        if not affected:
            return

        df = await self._arrow_load_table()

        # Deduplicate computation: group by (sub_type, params) key
        compute_groups: Dict[str, tuple] = {}
        sub_to_key: Dict[tuple, str] = {}
        for client_id, sub, ws in affected:
            cache_key = f"{sub.sub_type}:{_json_notify.dumps(sub.params, sort_keys=True)}"
            sub_to_key[(client_id, sub.sub_id)] = cache_key
            if cache_key not in compute_groups:
                ipc_bytes, meta = ArrowQueryEngine.recompute_subscription(df, sub)
                h = content_hash(ipc_bytes)
                compute_groups[cache_key] = (sub.sub_type, sub.params, ipc_bytes, h)

        _push_sem = asyncio.Semaphore(50)
        failed_clients: set = set()

        async def _push(client_id, sub, ws, ipc_bytes, h):
            if h == sub.last_hash:
                return
            msg = {"action": "arrow_data", "subId": sub.sub_id,
                   "subType": sub.sub_type, "bytes": len(ipc_bytes),
                   "data": ipc_bytes}
            async with _push_sem:
                try:
                    await self._send_direct_message(ws, msg)
                    sub.last_hash = h
                except Exception:
                    failed_clients.add(client_id)
                if not sub.reactive:
                    await self.arrow_sub_registry.remove_subscription(client_id, sub.sub_id)

        tasks = []
        for client_id, sub, ws in affected:
            ck = sub_to_key[(client_id, sub.sub_id)]
            _, _, ipc_bytes, h = compute_groups[ck]
            tasks.append(_push(client_id, sub, ws, ipc_bytes, h))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for cid in failed_clients:
            await self.arrow_sub_registry.unregister_client(cid)

    async def notify_arrow_room_changed(self, room: str, changes: List[dict]):
        """Push sparse delta updates to all Arrow IPC room subscribers.

        *changes* is a list of dicts: {op, data, fullRow, rowIndex, rowKey}.
        """
        subs = await self.arrow_room_registry.get_room_subs(room)
        if not subs:
            return

        failed_clients: set = set()

        async def _push_delta(client_id, sub, ws):
            col_set = frozenset(sub.columns) if sub.columns else None
            ops = []
            for change in changes:
                op = change.get("op", "update")
                data = change.get("data", {})
                full_row = change.get("fullRow")
                row_index = change.get("rowIndex")
                row_key = change.get("rowKey")

                if op == "remove":
                    effective_key = row_key
                    if effective_key is None and sub.row_key_field and data:
                        effective_key = data.get(sub.row_key_field)
                    if row_index is None and effective_key is None:
                        continue
                    ops.append({"op": "remove", "rowIndex": row_index, "rowKey": effective_key})
                    continue

                if sub.filters and data:
                    eval_row = {**(full_row or {}), **data} if full_row else data
                    if not row_matches_room_filters(eval_row, sub.filters):
                        if op == "update":
                            effective_key = row_key
                            if effective_key is None and sub.row_key_field and data:
                                effective_key = data.get(sub.row_key_field)
                            ops.append({"op": "remove", "rowIndex": row_index, "rowKey": effective_key})
                        continue

                if col_set is not None:
                    projected = {k: v for k, v in data.items() if k in col_set}
                else:
                    projected = data
                if not projected:
                    continue

                ops.append({"op": op, "rowIndex": row_index, "rowKey": row_key, "data": projected})

            if not ops:
                return

            msg = {
                "action": "arrow_delta", "room": room, "subId": sub.sub_id,
                "gridId": sub.grid_id, "ops": ops,
            }
            try:
                await self._send_direct_message(ws, msg)
            except Exception:
                failed_clients.add(client_id)

        await asyncio.gather(
            *[_push_delta(cid, sub, ws) for cid, sub, ws in subs],
            return_exceptions=True,
        )
        for cid in failed_clients:
            await self.arrow_room_registry.unregister_client(cid)
