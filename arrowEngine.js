



import { license } from '@/config/licenses.js';
import { LicenseManager, createGrid} from 'ag-grid-enterprise';
LicenseManager.setLicenseKey(license);
import * as arrow from 'apache-arrow';
import * as utils from '@stdlib/utils';
import {asArray, hash64Any, capitalize, measureText, asyncZipArray, debounce, moveInPlace, generateRandomLetterString, debouncePerArgs} from '@/utils/helpers.js';
import {CustomCellEditor, processDataFromClipboard, nullishValueSetter, didCellValueActuallyChange} from "@/grids/js/genericCellEditor.js";
import {TreeColumnChooser} from "@/grids/js/newTree.js";
import ArrowSsrAggregationStatusBar from "@/grids/js/arrow/arrowSsrAggregationStatusbar.js";
import {AgColumnRegistry, ArrowAgGridFilter, GlobalSearchManager, TypedArrayPool} from '@/grids/js/arrow/arrowExtensions.js';
import {CLEAR_SENTINEL, coerceToNumber, formatNumber, coerceToDate, coerceToBool} from '@/utils/typeHelpers.js';
import memoize from 'fast-memoize';
import {LRUCache} from 'mnemonist';
import {installKeyCoalescer, isArrowKey, isTextInputTarget} from "@/grids/js/arrow/keyCoalescer.js";
import {asyncForEach} from 'modern-async';
import {writeObjectToClipboard, writeStringToClipboard} from "@/utils/clipboardHelpers.js";
import {std, mean, median, quantileSeq} from 'mathjs';
import {HyperTable} from '@/utils/hyperTable.js';
import {PasteOverrideManager} from '@/grids/js/arrow/overrideManager.js';
import {CadesEmitter} from "@/utils/cades-emitter.js";


/* ---------------------------------- Tunables ---------------------------------- */

const MASK_NONE = 0;
const MASK_NULL = 1;
const MASK_VALUE = 2;

/* ----------------------------------- Clipboard ------------------------------ */
const FLASH_MODE = 'overlay';           // 'cells' | 'overlay' | 'viewport'
const THROTTLE_MS = 100;

const FLASH_DURATION_MS = 180;
const FLASH_RGB = '199, 225, 223';       // "cells" and overlays use same color
const FLASH_OPACITY_START = 0.75;     // 0..1

const OVERLAY_MAX_SCAN = 100;            // cap number of DOM rects sampled for union
const OVERLAY_PADDING_PX = 0;         // expand overlay slightly to cover borders
const Z_INDEX = 2147483647;           // sits above any grid UI
let lastCopyAt = 0;

const GUESS_MIN_W = 80;
const GUESS_MAX_W = 240;

// ---------- CSS setup (per-cell pulse) ----------
const STYLE_ID = 'ag-copy-flash-style-v3';
const FLASH_CLASS = 'copy-flash-active';

function getElement(elementOrSelector) {
    const el = (typeof elementOrSelector === 'string')
        ? (typeof document !== 'undefined' ? document.querySelector(elementOrSelector) : null)
        : elementOrSelector;
    if (!el) throw new Error('ArrowAgGridAdapter.mount: target element not found');
    return el
}

function isAsync(fn) {
    return fn.constructor.name === "AsyncFunction";
}

function getRowNode(api, rowIndex, rowPinned) {
    if (rowPinned === 'top') return api.getPinnedTopRow(rowIndex);
    if (rowPinned === 'bottom') return api.getPinnedBottomRow(rowIndex);
    return api.getDisplayedRowAtIndex(rowIndex);
}

function isEditableCell(colDef, ctx) {
    if (colDef && colDef.readOnly === true) return false;
    if (colDef && typeof colDef.editable === 'function') {
        try { return !!colDef.editable(ctx); } catch { return false; }
    }
    return !(colDef && colDef.editable === false);

}

function makeCellEventPayload(baseEvent, rowNode, column, colDef, newValue) {
    const field = colDef.field || column.getColId();
    const currentData = rowNode.data || {};
    const updatedData = { ...currentData, [field]: newValue };

    return {
        ...baseEvent,
        rowIndex: rowNode.rowIndex,
        node: rowNode,
        data: updatedData,
        colDef,
        column,
        value: newValue,
        newValue,
        oldValue: currentData[field],
        __isDelete: newValue === null,
    };
}

function safeInvoke(fn) {
    try {
        const out = fn();
        if (out && typeof out.then === 'function') {
            return out.catch(err => console.error('[grid] bulk clear cell failed', err));
        }
        return Promise.resolve(out);
    } catch (err) {
        console.error('[grid] bulk clear cell threw', err);
        return Promise.resolve();
    }
}

function nextTick() {
    return new Promise(resolve => setTimeout(resolve, 0));
}


async function clearSelectedCellsOnDelete(e, adapter) {
    const api = e.api;
    const colApi = e.columnApi;
    const ranges = api.getCellRanges();

    const pendingCalls = [];
    const seen = new Set();

    for (const r of ranges) {
        const startIdx = Math.min(r.startRow.rowIndex, r.endRow.rowIndex);
        const endIdx = Math.max(r.startRow.rowIndex, r.endRow.rowIndex);
        const rowPinned = r.startRow.rowPinned || r.endRow.rowPinned || null;
        const columns = r.columns || [];

        for (let rowIndex = startIdx; rowIndex <= endIdx; rowIndex++) {
            const rowNode = getRowNode(api, rowIndex, rowPinned);
            if (!rowNode) continue;

            for (const col of columns) {
                const colId = col.getColId();
                const key = (rowNode.id ?? String(rowNode.rowIndex)) + '|' + colId;
                if (seen.has(key)) continue;
                seen.add(key);

                const colDef = col.getColDef();
                if (!isEditableCell(colDef, {
                    api, columnApi: colApi, column: col, node: rowNode, data: rowNode.data,
                })) {
                    continue;
                }

                const cellEventPayload = makeCellEventPayload(e, rowNode, col, colDef, null);
                pendingCalls.push(() => adapter._processValueChange(cellEventPayload));
            }
        }
    }

    // Micro-batches to keep UI responsive for large selections.
    const CHUNK_SIZE = 250;
    for (let i = 0; i < pendingCalls.length; i += CHUNK_SIZE) {
        const batch = pendingCalls.slice(i, i + CHUNK_SIZE).map(fn => safeInvoke(fn));
        await Promise.all(batch);
        await nextTick();
    }
}

function _coerceOverlayNumber(value){
    if (value === undefined) return undefined;
    if (value === null) return null;

    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value === 'bigint'){
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    }

    let s0 = (typeof value === 'string') ? value : String(value);
    s0 = s0.trim();
    if (!s0) return null;

    let mult = 1;
    if (s0.charCodeAt(s0.length - 1) === 37){ // %
        s0 = s0.slice(0, -1).trim();
        if (!s0) return null;
        mult = 0.01;
    }

    if (s0.charCodeAt(0) === 43) s0 = s0.slice(1); // leading '+'

    // Remove common thousand separators only if present
    if (
        s0.indexOf(',') !== -1 ||
        s0.indexOf(' ') !== -1 ||
        s0.indexOf('\u00A0') !== -1 ||
        s0.indexOf('_') !== -1
    ){
        s0 = s0.replace(/[\u00A0\s,](?=\d{3}\b)/g, '').replace(/_/g, '');
    }

    const n = Number(s0);
    if (!Number.isFinite(n)) return null;

    const out = n * mult;
    return Number.isFinite(out) ? out : null;
}

function _coerceArrowScalar(v) {
    if (v === undefined) return null;
    if (v === CLEAR_SENTINEL) return null;
    if (typeof v === 'number' && !Number.isFinite(v)) return null;
    if (typeof v === 'bigint') {
        if (v > 9007199254740991n || v < -9007199254740991n) {
            console.warn('_coerceArrowScalar: bigint exceeds Number.MAX_SAFE_INTEGER, precision loss:', v);
        }
        return Number(v);
    }
    return v;
}

function _buildArrowTableFromRows(schema, rows) {
    const fields = schema.fields;
    const outCols = Object.create(null);

    for (let i = 0; i < fields.length; i++) {
        const f = fields[i];
        // Downgrade Int64/BigInt types to Float64 so the builder never expects BigInt values
        const isBigIntType = (f.type instanceof arrow.Int64) ||
            ((f.type instanceof arrow.Int) && f.type.bitWidth === 64);
        const builderType = isBigIntType ? new arrow.Float64() : f.type;
        const b = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });
        for (let r = 0; r < rows.length; r++) {
            const row = rows[r];
            const v = row && Object.prototype.hasOwnProperty.call(row, f.name) ? _coerceArrowScalar(row[f.name]) : null;
            b.append(v);
        }
        outCols[f.name] = b.finish().toVector();
    }

    return arrow.makeTable(outCols);
}

function _tryConcatArrowTables(left, right) {
    if (!left) return right;
    if (!right) return left;

    try {
        if (typeof left.concat === 'function') return left.concat(right);
    } catch {}

    try {
        if (arrow.Table && typeof arrow.Table.concat === 'function') return arrow.Table.concat([left, right]);
    } catch {}


    // Fallback: materialize to JS arrays, then build vectors in one shot.
    const schema = left.schema;
    const fields = schema.fields;
    const outCols = Object.create(null);
    const ln = left.numRows >>> 0;
    const rn = right.numRows >>> 0;
    const total = ln + rn;

    for (let i = 0; i < fields.length; i++) {
        const f = fields[i];
        const lv = left.getChildAt(i);
        const rv = right.getChildAt(i);

        // Collect into a flat JS array (one allocation, no reallocs)
        const arr = new Array(total);
        for (let r = 0; r < ln; r++) arr[r] = _coerceArrowScalar(lv.get(r));
        for (let r = 0; r < rn; r++) arr[ln + r] = _coerceArrowScalar(rv.get(r));

        // Downgrade Int64/BigInt types to Float64 so the builder never expects BigInt values
        const isBigIntType = (f.type instanceof arrow.Int64) ||
            ((f.type instanceof arrow.Int) && f.type.bitWidth === 64);
        const builderType = isBigIntType ? new arrow.Float64() : f.type;
        const b = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });
        for (let r = 0; r < total; r++) b.append(arr[r]);
        outCols[f.name] = b.finish().toVector();
    }

    return arrow.makeTable(outCols);
}

function ensureStyles(replace=false) {
    const style_box = document.getElementById(STYLE_ID);
    if (style_box) {
        if (!replace) return;
        style_box.remove();
    }
    const css = `
  .ag-root.${FLASH_CLASS} .ag-cell-range-selected { will-change: opacity; }
  .ag-root.${FLASH_CLASS} .ag-cell-range-selected::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: rgba(var(--agCopyFlashRgb, ${FLASH_RGB}), var(--agCopyFlashAlpha, ${FLASH_OPACITY_START}));
    z-index: 1;
    animation: agCopyPulse var(--agCopyFlashDur, ${FLASH_DURATION_MS}ms) ease-out forwards;
    transform: translateZ(0);
  }
  .${FLASH_CLASS} .ag-cell-range-selected::after {
    content: "";
    position: absolute; inset: 0; pointer-events: none;
    background: rgba(var(--agCopyFlashRgb, ${FLASH_RGB}), var(--agCopyFlashAlpha, ${FLASH_OPACITY_START}));
    z-index: 1;
    animation: agCopyPulse var(--agCopyFlashDur, ${FLASH_DURATION_MS}ms) ease-out forwards;
    transform: translateZ(0);
  }

  /* Overlay elements (single rect or viewport blink) */
  .ag-copy-overlay {
    position: fixed;
    pointer-events: none;
    will-change: opacity;
    animation: agCopyPulse ${FLASH_DURATION_MS}ms ease-out forwards;
    background: rgba(${FLASH_RGB}, ${FLASH_OPACITY_START});
    z-index: ${Z_INDEX};
    transform: translateZ(0);
  }
  .ag-copy-overlay.hidden { display: none; }

  @keyframes agCopyPulse {
    0%   { opacity: ${FLASH_OPACITY_START}; }
    70%  { opacity: 0.18; }
    100% { opacity: 0; }
  }`;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = css;
    document.head.appendChild(style);
}


// ---------- Flash helpers ----------
function findGridRootForCurrentSelection(elementOrSelector=null) {
    const base = elementOrSelector ? getElement(elementOrSelector) : document;
    const firstSel = base.querySelector('.ag-cell-range-selected:has(.ag-cell-focus)');
    if (firstSel) {
        const root = firstSel.closest('.ag-root') || firstSel.closest('.ag-root-wrapper') || null;
        if (root) return root;
    }
    return base.querySelector('.ag-root') || document.body;
}

function flashCellsCSS(elementOrSelector=null) {
    const root = findGridRootForCurrentSelection(elementOrSelector);
    if (!root) return;
    root.style.setProperty('--agCopyFlashDur', `${FLASH_DURATION_MS}ms`);
    root.style.setProperty('--agCopyFlashRgb', FLASH_RGB);
    root.style.setProperty('--agCopyFlashAlpha', String(FLASH_OPACITY_START));
    root.classList.remove(FLASH_CLASS);
    // Restart keyframe reliably across engines
    void root.offsetWidth; // reflow
    root.classList.add(FLASH_CLASS);
    setTimeout(() => root.classList.remove(FLASH_CLASS), FLASH_DURATION_MS + 48);
}

let overlayMap = new Map();
let overlayTimer = 0;

function _evictStaleOverlays() {
    for (const [k, el] of overlayMap) {
        if (!document.body.contains(el)) {
            try { el.remove(); } catch {}
            overlayMap.delete(k);
        }
    }
}

function ensureOverlay(elementOrSelector) {
    // Evict stale entries when map grows beyond expected working set
    if (overlayMap.size > 8) {
        _evictStaleOverlays();
    }
    const existing = overlayMap.get(elementOrSelector);
    if (existing) {
        if (document.body.contains(existing)) return existing;
        // Old element no longer in DOM — remove stale reference
        try { existing.remove(); } catch {}
        overlayMap.delete(elementOrSelector);
    }
    const overlayEl = document.createElement('div');
    overlayEl.className = 'ag-copy-overlay hidden';
    overlayEl.style.cssText = `
    top:0; left:0; width:0; height:0; opacity:${FLASH_OPACITY_START};
  `;
    document.body.appendChild(overlayEl);
    overlayMap.set(elementOrSelector, overlayEl);
    return overlayEl;
}

function restartOverlayAnimation(el) {
    el.classList.remove('hidden');
    el.style.animation = 'none';
    void el.offsetWidth;
    el.style.animation = `agCopyPulse ${FLASH_DURATION_MS}ms ease-out forwards`;
    if (overlayTimer) clearTimeout(overlayTimer);
    overlayTimer = setTimeout(() => { el.classList.add('hidden'); }, FLASH_DURATION_MS + 48);
}

function flashViewportBlink(elementOrSelector=null, api=null) {
    const root = findGridRootForCurrentSelection(elementOrSelector);
    if (!root) {
        console.error('no root.')
        return;
    }
    const r = root.getBoundingClientRect();
    const el = ensureOverlay(elementOrSelector);
    const headerBuffer = api?.getGridOption('headerHeight') ?? 0;
    const horizontalBuffer = root.querySelector('.ag-body-vertical-scroll-viewport')?.getBoundingClientRect()?.width ?? 0;
    const verticalBuffer = root.querySelector('.ag-body-horizontal-scroll-viewport')?.getBoundingClientRect()?.height ?? 0;

    el.style.top = `${Math.max(0, r.top+headerBuffer)}px`;
    el.style.left = `${Math.max(0, r.left)}px`;
    el.style.width = `${Math.max(0, r.width-horizontalBuffer)}px`;
    el.style.height = `${Math.max(0, r.height-headerBuffer-verticalBuffer)}px`;
    el.style.background = `rgba(${FLASH_RGB}, ${FLASH_OPACITY_START})`;
    restartOverlayAnimation(el);
}

function flashSingleOverlayRect(elementOrSelector=null, api=null) {
    const root = findGridRootForCurrentSelection(elementOrSelector);
    if (!root) { flashViewportBlink(elementOrSelector, api); return; }
    const cells = root.querySelectorAll('.ag-cell-range-selected');
    const len = cells.length;
    if (!len) { flashViewportBlink(elementOrSelector, api); return; }

    const bounds = root.getBoundingClientRect();
    const headerBuffer = api?.getGridOption('headerHeight') ?? 0;

    let top, right, bottom, left, width, height;
    let top_left, bottom_right;

    const single_check = root.querySelector('.ag-cell-range-single-cell')?.getBoundingClientRect();

    if (single_check) {
        top_left = single_check;
        bottom_right = single_check;
    } else {
        top_left = root.querySelector('.ag-cell-range-top.ag-cell-range-left')?.getBoundingClientRect();
        bottom_right = root.querySelector('.ag-cell-range-bottom.ag-cell-range-right')?.getBoundingClientRect();
    }


    if (!bottom_right || !top_left) {
        return flashViewportBlink(elementOrSelector, api);
    }

    top = Math.max(top_left.top, bounds.top+headerBuffer);
    left = Math.max(top_left.left, bounds.left);
    bottom = Math.min(bottom_right.bottom, bounds.bottom);
    right = Math.min(bottom_right.right, bounds.right);
    width = Math.max(0, right - left);
    height = Math.max(0, bottom - top);

    const el = ensureOverlay(elementOrSelector);
    el.style.top = `${top}px`;
    el.style.left = `${left}px`;
    el.style.width = `${width}px`;
    el.style.height = `${height}px`;
    el.style.background = `rgba(${FLASH_RGB}, ${FLASH_OPACITY_START})`;
    restartOverlayAnimation(el);
}

export function flashSelection(elementOrSelector=null, api=null) {
    switch (FLASH_MODE) {
        case 'overlay':  return flashSingleOverlayRect(elementOrSelector, api);
        case 'viewport': return flashViewportBlink(elementOrSelector);
        case 'cells':
        default: return flashCellsCSS(elementOrSelector);
    }
}

/* ---------------------------------- utils ---------------------------------- */
export const isCopyHotKey = (e) => { return e.ctrlKey /*&& e.shiftKey*/ && e.key.toLowerCase() === 'c' }
const fullIndex = (n) => { const a = new Int32Array(n); for (let i=0;i<n;i++) a[i]=i; return a; };
const isArrowTable = (x) => !!(x && typeof x.schema === 'object' && Number.isFinite(x.numRows));
const _toInt32 = (x) => { return (x|0); }
const _toUint32 = (x) => { return (x>>>0); }

export const minimalValueSetter = function (p) {
    const colId = (p.column && typeof p.column.getColId === 'function' && p.column.getColId()) || p.colId || (p.colDef && p.colDef.field);
    if (!colId || !p || !p.data) return false;
    const prev = p.data[colId];
    const next = p.newValue;
    if (Object.is(prev, next)) return false;
    p.data[colId] = next;
    return true;
};

/* ---------------------------------- Edits ---------------------------------- */

class EditSignalCoalescer {
    constructor(emitter, engine, options = {}) {
        if (!emitter) throw new Error('EditSignalCoalescer: emitter required');

        this.emitter = emitter;
        this.engine = engine;
        this._emit = (typeof emitter.emitAsync === 'function')
            ? emitter.emitAsync.bind(emitter)
            : (typeof emitter.emitSync === 'function' ? emitter.emitSync.bind(emitter) : null);
        if (!this._emit) throw new Error('EditSignalCoalescer: emitter must have emitSync or emitAsync');

        this.eventNameBatch = options.eventNameBatch || 'cell-edit';
        this.eventNamePerRow = options.eventNamePerRow || 'row-edit';
        this.eventNamePerColumn = options.eventNamePerColumn || 'column-edit';
        this.alsoEmitPerRow = !!options.alsoEmitPerRow;
        this.alsoEmitPerColumn = !!options.alsoEmitPerColumn;
        this.alsoEmitPerCell = !!options.alsoEmitPerCell;

        this.emitEpochEvent = !!options.emitEpochEvent;
        this.epochEventName = options.epochEventName || 'epoch-change';

        this.useRaf = options.useRaf !== false;
        this.maxWaitMs = typeof options.maxWaitMs === 'number' ? options.maxWaitMs : 16;
        this.maxBatchCells = typeof options.maxBatchCells === 'number' ? options.maxBatchCells : 2048;

        this.timesliceMs = typeof options.timesliceMs === 'number' ? options.timesliceMs : 6;
        this.maxRowsPerSlice = typeof options.maxRowsPerSlice === 'number' ? options.maxRowsPerSlice : 512;

        this.shouldMaterialize = typeof options.shouldMaterialize === 'function' ? options.shouldMaterialize : null;

        this._front = new Map();
        this._back = new Map();
        this._pendingCellCount = 0;

        this._rafId = null;
        this._timeoutId = null;
        this._scheduled = false;

        this._flushing = false;
        this._sliceEntries = null;
        this._sliceIdx = 0;
        this._sliceOut = null;
        this._sliceTStart = 0;

        this._inFlight = new Set();
        this._disposed = false;

        this._epoch = 0;

        this._mc = null; // Disabled: MessageChannel fires before rendering, creating tight loops

        this._stats = {
            cellsEnqueued: 0,
            rowsEnqueued: 0,
            batchesEmitted: 0,
            rowsEmitted: 0,
            lastFlushMs: 0
        };
    }

    silent(rowIndex, columnName, value, engine) {
        if (this._disposed) return;
        let row = this._front.get(rowIndex);
        if (!row) {
            row = { cols: [], vals: [], idx: Object.create(null), engine };
            this._front.set(rowIndex, row);
            this._stats.rowsEnqueued++;
        }
        const idx = row.idx[columnName];
        if (idx === undefined) {
            row.idx[columnName] = row.cols.length;
            row.cols.push(columnName);
            row.vals.push(value);
            this._pendingCellCount++;
            this._stats.cellsEnqueued++;
        } else {
            row.vals[idx] = value;
        }
        row.engine = engine;
    }

    enqueue(rowIndex, columnName, value, engine) {
        if (this._disposed) return;

        let row = this._front.get(rowIndex);
        if (!row) {
            row = { cols: [], vals: [], idx: Object.create(null), engine };
            this._front.set(rowIndex, row);
            this._stats.rowsEnqueued++;
        }

        const idx = row.idx[columnName];
        if (idx === undefined) {
            row.idx[columnName] = row.cols.length;
            row.cols.push(columnName);
            row.vals.push(value);
            this._pendingCellCount++;
            this._stats.cellsEnqueued++;
        } else {
            row.vals[idx] = value;
        }
        row.engine = engine;

        if (!this._scheduled && !this._flushing) this._scheduleFlush();
        if (this._pendingCellCount >= this.maxBatchCells && !this._flushing) this._flushNow();
    }

    async drain() {
        if (this._disposed) return;
        this._flushNow();
        if (this._inFlight.size === 0) return;
        await Promise.allSettled([...this._inFlight]);
    }

    dispose() {
        if (this._disposed) return;
        this._disposed = true;
        if (this._rafId !== null) cancelAnimationFrame(this._rafId);
        if (this._timeoutId !== null) clearTimeout(this._timeoutId);
        if (this._mc) {
            try { this._mc.port1.onmessage = null; this._mc.port1.close(); this._mc.port2.close(); } catch (_) {}
        }
        this._rafId = null;
        this._timeoutId = null;
        this._scheduled = false;
        this._front.clear();
        this._back.clear();
        this._inFlight.clear();
        this._sliceEntries = null;
        this._sliceOut = null;
    }

    getStats() {
        return {
            ...this._stats,
            pendingRows: this._front.size,
            pendingCells: this._pendingCellCount,
            epoch: this._epoch
        };
    }

    _scheduleFlush() {
        this._scheduled = true;
        if (this.useRaf && typeof requestAnimationFrame === 'function') {
            this._rafId = requestAnimationFrame(() => {
                this._rafId = null;
                if (this._timeoutId !== null) { clearTimeout(this._timeoutId); this._timeoutId = null; }
                this._beginFlush();
            });
            // Safety net: if RAF is starved (e.g. tab backgrounded), setTimeout fires
            this._timeoutId = setTimeout(() => {
                this._timeoutId = null;
                if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
                this._beginFlush();
            }, this.maxWaitMs);
        } else {
            this._timeoutId = setTimeout(() => {
                this._timeoutId = null;
                this._beginFlush();
            }, this.maxWaitMs);
        }
    }

    _flushNow() {
        if (!this._scheduled && !this._flushing) {
            this._scheduled = true;
        }
        if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
        if (this._timeoutId !== null) { clearTimeout(this._timeoutId); this._timeoutId = null; }
        this._beginFlush();
    }

    _beginFlush() {
        if (this._flushing || this._disposed) return;
        this._scheduled = false;
        if (this._front.size === 0) return;

        const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

        const tmp = this._back;
        this._back = this._front;
        this._front = tmp;
        this._front.clear();

        this._pendingCellCount = 0;

        this._sliceEntries = Array.from(this._back.entries());
        this._sliceIdx = 0;
        this._sliceOut = [];
        this._sliceTStart = t0;

        this._flushing = true;
        this._runSlice();
    }

    _runSlice() {
        if (!this._flushing || this._disposed) return;

        const start = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        let deadline = start + (this.timesliceMs > 0 ? this.timesliceMs : 0);
        let rowsProcessed = 0;

        const entries = this._sliceEntries;
        const out = this._sliceOut;
        const limit = this.maxRowsPerSlice > 0 ? this.maxRowsPerSlice : Number.POSITIVE_INFINITY;

        for (; this._sliceIdx < entries.length; this._sliceIdx++) {
            const pair = entries[this._sliceIdx];
            const rowIndex = pair[0];
            const buf = pair[1];

            if (!this.shouldMaterialize || this.shouldMaterialize(rowIndex)) {
                const obj = Object.create(null);
                const cols = buf.cols;
                const vals = buf.vals;
                for (let i = 0, n = cols.length; i < n; i++) obj[cols[i]] = vals[i];
                out.push({ rowIndex, changes: obj, engine: this.engine });
            }

            rowsProcessed++;
            if (rowsProcessed >= limit) break;
            if (this.timesliceMs > 0) {
                const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
                if (now >= deadline) {
                    this._sliceIdx++;
                    break;
                }
            }
        }

        if (this._sliceIdx < entries.length) {
            // Use setTimeout instead of MessageChannel to yield to the event loop
            // between slices, allowing rendering and input processing.
            setTimeout(() => this._runSlice(), 0);
            return;
        }

        const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        this._stats.lastFlushMs = t1 - (this._sliceTStart || t1);

        const batch = out;
        this._sliceEntries = null;
        this._sliceOut = null;
        this._back.clear();
        this._flushing = false;

        if (batch.length === 0) {
            return;
        }

        // Yield to the event loop BEFORE emitting, so the browser can render
        // and process input.  emitAsync uses queueMicrotask which chains
        // microtasks synchronously — without this yield, a cascade of
        // microtasks (emit → handler → epoch → refreshServerSide → _getRows)
        // can block the main thread indefinitely.
        setTimeout(() => this._emitBatch(batch), 0);
    }

    _emitBatch(batch) {
        if (this._disposed) return;

        const promises = [];
        try {
            const p = this._emit ? this._emit(this.eventNameBatch, batch) : null;
            if (p && typeof p.then === 'function') promises.push(p);
        } catch (_) {}

        if (this.alsoEmitPerRow) {
            for (let i = 0; i < batch.length; i++) {
                const r = batch[i];
                try {
                    const p = this._emit ? this._emit(this.eventNamePerRow, r) : null;
                    if (p && typeof p.then === 'function') promises.push(p);
                } catch (_) {}
            }
        }

        if (this.alsoEmitPerColumn) {
            const seenCols = new Set();
            for (let i = 0; i < batch.length; i++) {
                const r = batch[i];
                const cols = r.changes;
                for (const col in cols) {
                    if (seenCols.has(col)) continue;
                    seenCols.add(col);
                    try {
                        const p = this._emit ? this._emit(this.eventNamePerColumn, col) : null;
                        if (p && typeof p.then === 'function') promises.push(p);
                    } catch (_) {
                    }
                }
            }
        }

        if (this.alsoEmitPerCell) {
            for (let i = 0; i < batch.length; i++) {
                const r = batch[i];
                const cols = r.changes;
                for (const col in cols) {
                    try {
                        const p = this._emit ? this._emit('cell-edit', {
                            rowIndex: r.rowIndex,
                            columnName: col,
                            value: cols[col],
                        }) : null;
                        if (p && typeof p.then === 'function') promises.push(p);
                    } catch (_) {}
                }
            }
        }

        if (this.emitEpochEvent) {
            this._epoch++;
            try {
                const p = this._emit ? this._emit(this.epochEventName, this._epoch) : null;
                if (p && typeof p.then === 'function') promises.push(p);
            } catch (_) {}
        }

        for (let i = 0; i < promises.length; i++) {
            const pr = promises[i];
            this._inFlight.add(pr);
            pr.finally(() => {
                if (!this._disposed) {
                    this._inFlight.delete(pr)
                }
            });
        }

        this._stats.batchesEmitted++;
        this._stats.rowsEmitted += batch.length;
    }

}

/* ---------------------------------- Arrow Helpers ---------------------------------- */

function projectTableToSchema(sourceTable, targetSchema) {
    if (!sourceTable || !targetSchema) {
        throw new Error('projectTableToSchema: both sourceTable and targetSchema are required');
    }

    const sourceSchema = sourceTable.schema;
    const srcFields = sourceSchema.fields;
    const srcVectorsByName = new Map();

    for (let i = 0; i < srcFields.length; i++) {
        const field = srcFields[i];
        const vec = sourceTable.getChildAt(i);
        if (vec) srcVectorsByName.set(field.name, vec);
    }

    const outCols = {};
    const numRows = sourceTable.numRows >>> 0;
    const targetFields = targetSchema.fields;

    for (let i = 0; i < targetFields.length; i++) {
        const targetField = targetFields[i];
        const name = targetField.name;
        const sourceVector = srcVectorsByName.get(name);
        let builderType = targetField.type;

        const isBigIntLike =
            builderType instanceof arrow.Int64 ||
            (builderType instanceof arrow.Int && (builderType.bitWidth === 64));

        if (isBigIntLike) {
            builderType = new arrow.Float64();
        }

        const builder = arrow.makeBuilder({
            type: builderType,
            nullValues: [null, undefined]
        });

        if (sourceVector) {
            for (let rowIndex = 0; rowIndex < numRows; rowIndex++) {
                let value = sourceVector.get(rowIndex);
                if (isBigIntLike && typeof value === 'bigint') {
                    value = Number(value);
                }
                builder.append(value);
            }
        } else {
            for (let rowIndex = 0; rowIndex < numRows; rowIndex++) {
                builder.append(null);
            }
        }

        outCols[name] = builder.finish().toVector();
    }

    return arrow.makeTable(outCols);
}

function buildEmptyTableFromSchema(schema) {
    const outCols = {};
    for (let i = 0; i < schema.fields.length; i++) {
        const f = schema.fields[i];
        const isBigIntType = (f.type instanceof arrow.Int64) ||
            ((f.type instanceof arrow.Int) && f.type.bitWidth === 64);
        const builderType = isBigIntType ? new arrow.Float64() : f.type;
        const b = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });
        outCols[f.name] = b.finish().toVector(); // zero‑length
    }
    return new arrow.Table(outCols);
}

/* ---------------------------------- Engine ---------------------------------- */

export class ArrowEngine {
    constructor(context, table, opts={}){
        if (!table || !table.schema) throw new TypeError('ArrowEngine: provide a valid Apache Arrow Table');

        const {
            idProperty='__rowId',
            idxProperty='__srcIndex',
            rowIdGetter=null,
            autoCommitThreshold=0.25,
            useEpochCache=false,
            yieldEvery=8192,
            decodeDictionaries=false,
            scaleTimestampsToMs=true,
            precomputeScalers=false,
            precomputeDecoders=false,
            classCacheSize=5000,
            master_schema=null
        } = opts;

        this.opts = opts;
        this.context = context;
        this.table = table;
        this.master_schema = master_schema || this.table.schema;
        this._fields = table.schema.fields;
        this._nameToIndex = new Map();
        this._nameToVector = new Map();
        for (let i=0;i<this._fields.length;i++){
            const nm = this._fields[i].name;
            this._nameToIndex.set(nm, i);
            this._nameToVector.set(nm, table.getChildAt(i));
        }
        this._fieldNamesCache = null;
        this._fieldSetCache = null;
        Object.defineProperty(this, '_fieldNames', { get() { return this._fieldNamesCache || (this._fieldNamesCache = Array.from(this._nameToIndex.keys())); } });

        this.columnDefs = new Map();
        this._idProperty = idProperty;
        this._idxProperty = idxProperty;
        this._rowIdGetter = typeof rowIdGetter==='function' ? rowIdGetter : ((ri, engine) => String(ri));
        this._rowIdToIndex = null;
        this._useMinMax = true;

        this._decodeDictionaries = !!decodeDictionaries;
        this._scaleTimestampsToMs = !!scaleTimestampsToMs;
        this._yieldEvery = _toInt32(yieldEvery);
        this._autoCommitThreshold = +autoCommitThreshold;
        this._autoCommitEditInterval = 20; // materialize every N physical edits; reset after each trigger

        this.timeUnitScale = Object.create(null);
        this._decoder = Object.create(null);
        if (precomputeScalers) this._initScalers();
        if (precomputeDecoders) this._initDecoders();

        this._overlays = new Map();

        this._derived = new Map();
        this._columnDependencies = new Map();
        this._columnDependents = new Map();
        this._depClosureCache = new Map();
        this._dependentsClosureCache = new Map();
        this._derivedSettingsSubs = new Map();
        this._settingsGetter = new Map();
        this._valueSetters = new Map();
        this._editDrainScheduled = false;

        this.cellCacheSize = classCacheSize;
        const cacheSz = Math.max(512, table.numCols * 128);
        this._dictCacheMax = 512;
        this._classCache = new LRUCache(cacheSz);
        this._fmtCache = new LRUCache(cacheSz);

        this._frozenOverrides = new Map();  // Map<colName, Set<rowIndex>> — fast freeze/unfreeze without string concat
        this._idxMinMax = new Map();
        this._indexBlockSize = 2048;

        this._dictWindowCache = new Map();
        this._dictEpochByVec = new WeakMap();
        this._dictEpochCounter = 1;

        this._materializeTimer = null
        this._isMaterializing = false;

        this._useEpochCache = !!useEpochCache;
        this._epochQueue = [];
        this._epochScheduled = false;
        this._suppressEpochEmit = false;
        this._dtypes = null;

        this._pool = new TypedArrayPool();
        this._sortedRowCache = new Map();
        this._tableReplaceSeq = 0;

        this.emitter = new CadesEmitter();
        this._editCoalescer = new EditSignalCoalescer(this.emitter, this, {
            eventNameBatch: 'cell-edit',
            eventNamePerRow: 'row-edit',
            alsoEmitPerRow: false,
            alsoEmitPerCell: false,
            alsoEmitPerColumn: false,
            useRaf: true,
            maxWaitMs: 275,
            maxBatchCells: 5000
        });
        this._disposed = false;
    }

    /* ---------------------------- registration DSL --------------------------- */
    registerColumnsFromDefs(columnDefs, settingsDict){
        if (!Array.isArray(columnDefs)) return this;
        for (let i=0;i<columnDefs.length;i++){
            const def = columnDefs[i] || {};
            const name = def.field ?? def.colId ?? def.id;
            if (!name) continue;

            if (!def.context) def.context = {};
            const ctx = def.context;
            const compute = ctx.compute;
            const deps = ctx.deps;
            const inverse = ctx.inverse || (ctx.edit && ctx.edit.inverse);
            const kind = ctx.dataType === 'number' ? 'number' : (ctx.kind || undefined);
            const engine = this;

            if (typeof compute === 'function') {
                let settingsSpec;
                if (typeof ctx.settingsDict === 'string') {
                    if (this.context.page[ctx.settingsDict]) {
                        settingsSpec = { dict: this.context.page[ctx.settingsDict], keys: ctx.settingsKeys ?? ["*"] }
                    }
                } else {
                    settingsSpec = ctx.settings || (settingsDict && (ctx.settingKeys || ctx.settingsKeys || ctx.keys)
                        ? { dict: settingsDict, keys: asArray(ctx.settingKeys || ctx.settingsKeys || ctx.keys) }
                        : undefined);
                }

                this.registerDerivedColumn(name, compute, { deps, inverse, kind, settings: settingsSpec });
            }

            const vs = def.valueSetter || (ctx.edit && ctx.edit.setter);
            if (typeof vs === 'function') {
                this._valueSetters.set(name, async ({rowIndex, value, engine}) => {
                    try { return await vs({rowIndex, value, engine}); }
                    catch(e){ console.error('valueSetter error:', e); return false; }
                });
            }

            const tv = def?.tooltipValueGetter || (ctx?.tooltipValueGetter);
            if (typeof tv === 'function') {
                def.tooltipValueGetter = (params) => tv(params, engine);
            }
            this.columnDefs.set(name, def);
        }
        return this;
    }

    /* --------------------------------- ids ---------------------------------- */
    getRowIdByIndex(rowIndex){ return this._rowIdGetter(_toInt32(rowIndex), this); }
    // getRowIndexById(rowId){
    //     if (rowId == null) return -1;
    //     const n = (this.table?.numRows | 0);
    //     if (!this._rowIdToIndex || this._rowIdToIndex.size !== n) {
    //         this._rowIdToIndex = new Map();
    //         for (let i = 0; i < n; i++) this._rowIdToIndex.set(this._rowIdGetter(i, this), i);
    //     }
    //     return this._rowIdToIndex.get(rowId) ?? -1;
    // }

    getRowIndexById(rowId) {
        if (rowId == null) return -1;

        const n = (this.table?.numRows | 0);
        if (n <= 0) return -1;

        // Fast path: numeric ids that map directly to rowIndex.
        if (typeof rowId === 'number') {
            const ri = rowId | 0;
            if (ri >= 0 && ri < n && ri === rowId) {
                const candidate = this._rowIdGetter(ri, this);
                if (candidate === String(ri)) return ri;
            }
        } else if (typeof rowId === 'string' && rowId.length > 0) {
            // Only attempt fast path if the string looks purely numeric
            const parsed = +rowId;
            if (Number.isFinite(parsed)) {
                const ri = parsed | 0;
                if (ri >= 0 && ri < n && String(ri) === rowId) {
                    const candidate = this._rowIdGetter(ri, this);
                    if (candidate === rowId) return ri;
                }
            }
        }

        if (!this._rowIdToIndex || this._rowIdToIndex.size !== n) {
            this._rowIdToIndex = new Map();
            for (let i = 0; i < n; i++) this._rowIdToIndex.set(this._rowIdGetter(i, this), i);
        }
        return this._rowIdToIndex.get(rowId) ?? -1;
    }

    /* -------------------------------- events -------------------------------- */

    // Returns a callable unsubscribe function so callers can do: const off = engine.onX(fn); off();
    _wrapOff(token) { return () => { try { this.emitter.offToken(token); } catch (_) {} }; }

    onEpochChange(fn){ return this._wrapOff(this.emitter.on('epoch-change', fn)); }
    onMaterialize(fn){ return this._wrapOff(this.emitter.on('table-materialized', fn)); }
    onDerivedDirty(fn){ return this._wrapOff(this.emitter.on('derived-dirty', fn)); }
    _emitMaterialize(rows, cols){ this.emitter.emit('table-materialized', rows, cols); }
    _notifyDerivedDirty(name){ this.emitter.emit('derived-dirty', new Set(asArray(name))); }

    _notifyColumnEvent(event) {
        this.emitter.emit(`grid:ColumnEvent-${event}`)
    }
    onColumnEvent(event, fn) { return this._wrapOff(this.emitter.on(`grid:ColumnEvent-${event}`, fn)); }
    onTableWillReplace(fn){ return this._wrapOff(this.emitter.on('table-will-replace', fn)); }
    onTableDidReplace(fn){ return this._wrapOff(this.emitter.on('table-did-replace', fn)); }
    _emitEpochChange(payload, event='epoch-change') {
        this._epochQueue.push(payload);
        if (!this._epochScheduled) {
            this._epochScheduled = true;
            setTimeout(() => {
                const batch = this._epochQueue;
                this._epochQueue = [];
                this._epochScheduled = false;
                if (this._disposed || !this.emitter) return;
                const coalesced = this._coalesceEpochs(batch);

                let allCols = coalesced.colsChanged;
                if (allCols && allCols !== true) {
                    const arr = Array.isArray(allCols) ? allCols : [allCols];
                    const closure = this.getDependentsClosure(arr);
                    // getDependentsClosure returns seeds + dependents; check for new columns
                    if (closure.length > arr.length) {
                        const dirtyDerived = closure.filter(c => this._isDerived(c));
                        if (dirtyDerived.length) {
                            this.emitter.emit('derived-dirty', new Set(dirtyDerived));
                        }
                        allCols = closure;
                        coalesced.colsChanged = allCols;
                    } else {
                        // No dependents beyond seeds — still check if seeds themselves are derived
                        const dirtyDerived = arr.filter(c => this._isDerived(c));
                        if (dirtyDerived.length) {
                            this.emitter.emit('derived-dirty', new Set(dirtyDerived));
                        }
                    }
                }

                // Clear caches for all affected columns so downstream reads are fresh.
                this._clearCellCaches(allCols === true ? true : (allCols || []));

                // Synchronous broadcast — subscribers own their own scheduling
                // (debounce, rAF, etc). No extra setTimeout hop.
                this.emitter.emit(event, coalesced);
            }, 0);
        }
    }

    onColumnEpochChange(columns, fn, opts) {
        let my_columns = Array.isArray(columns) ? columns : [columns];
        my_columns = new Set(this._normalizeColumnSelector(my_columns));
        return this.onEpochChange((epoch)=>{
            let all_epochs = Array.isArray(epoch) ? epoch : [epoch];
            for (const my_epoch of all_epochs) {
                if (!my_epoch.colsChanged) return;
                if (my_epoch.colsChanged.some(item => my_columns.has(item))) {
                    return fn(my_epoch);
                }
            }
        }, opts)
    }

    _emitTableWillReplace(prevTable, nextTable){
        try {
            const prevNumRows = prevTable ? (prevTable.numRows | 0) : 0;
            const nextNumRows = nextTable ? (nextTable.numRows | 0) : 0;
            const prevNumCols = prevTable ? (prevTable.numCols | 0) : 0;
            const nextNumCols = nextTable ? (nextTable.numCols | 0) : 0;
            this.emitter.emit('table-will-replace', { prevNumRows, nextNumRows, prevNumCols, nextNumCols });
        } catch {}
    }

    _emitTableDidReplace(prevTable, nextTable){
        try {
            const prevNumRows = prevTable ? (prevTable.numRows | 0) : 0;
            const nextNumRows = nextTable ? (nextTable.numRows | 0) : 0;
            const prevNumCols = prevTable ? (prevTable.numCols | 0) : 0;
            const nextNumCols = nextTable ? (nextTable.numCols | 0) : 0;
            this.emitter.emit('table-did-replace', { prevNumRows, nextNumRows, prevNumCols, nextNumCols });
        } catch {}
    }

    _coalesceEpochs(batch) {
        if (!Array.isArray(batch) || batch.length === 0) return { colsChanged: null, rowsChanged: null };

        // Fast path: single-element batch (the overwhelmingly common case)
        if (batch.length === 1) {
            const p = batch[0] || {};
            if (p.global) return { global: true };
            let c = p.colsChanged ?? null;
            if (typeof c === 'string') c = [c];
            return { colsChanged: c, rowsChanged: p.rowsChanged ?? null };
        }

        let global = false;
        const cols = new Set();
        const rows = new Set();

        for (let i = 0; i < batch.length; i++) {
            const p = batch[i] || {};
            if (p.global) { global = true; break; }

            const c = p.colsChanged;
            if (c) {
                if (c === true) { cols.clear(); cols.add(true); }
                else if (!cols.has(true)) {
                    if (Array.isArray(c)) { for (let j = 0; j < c.length; j++) cols.add(c[j]); }
                    else if (c instanceof Set) { for (const v of c) cols.add(v); }
                    else cols.add(c);
                }
            }

            const r = p.rowsChanged;
            if (r === true) { rows.clear(); rows.add(true); }
            else if (r) {
                if (r instanceof Int32Array) { for (let j = 0; j < r.length; j++) rows.add(r[j] | 0); }
                else if (Array.isArray(r)) { for (let j = 0; j < r.length; j++) rows.add(r[j] | 0); }
                else if (r instanceof Set) { for (const v of r) rows.add(v | 0); }
            }
        }

        if (global) return { global: true };
        const colsChanged = (cols.size === 1 && cols.has(true)) ? true : (cols.size ? Array.from(cols) : null);
        const rowsChanged = rows.size ? (rows.has(true) ? true : Int32Array.from(rows)) : null;
        return { colsChanged, rowsChanged };
    }


    /* ---------------------------- schema utilities --------------------------- */
    _normalizeColumnSelector(selector){
        const total = this._fieldNames.length;
        const idxToName = (idx)=>{ if (idx<0||idx>=total) throw new RangeError('Column index out of bounds:'+idx); return this._fieldNames[idx]; };
        if (selector == null) return this._fieldNames.slice();
        if (typeof selector === 'string') return [selector];
        if (typeof selector === 'number') return [idxToName(selector)];
        if (Array.isArray(selector)) return selector.map(x => typeof x==='string' ? x : idxToName(x));
        throw new TypeError('Unsupported column selector');
    }

    _getVector(name) {
        if (!this._nameToVector.has(name)) {
            const idx = this._nameToIndex.get(name);
            if (idx == null) {
                // Log once per column per table lifetime, not once per row.
                if (!this._missingColWarned) this._missingColWarned = new Set();
                if (!this._missingColWarned.has(name)) {
                    this._missingColWarned.add(name);
                    console.warn(`[ArrowEngine] Column not found in table schema: "${name}"`);
                }
                return null;
            }
            this._nameToVector.set(name, this.table.getChildAt(idx));
        }
        return this._nameToVector.get(name);
    }

    _detectTimeUnitScale(vec){
        const t = vec && (vec.type || vec);
        const unit = t && t.unit ? String(t.unit) : '';
        if (unit === 'SECOND') return 1000;
        if (unit === 'MILLISECOND') return 1;
        if (unit === 'MICROSECOND') return 0.001;
        if (unit === 'NANOSECOND') return 0.000001;
        return 1;
    }

    _initScalers(){
        for (const [name, vec] of this._nameToVector.entries()){
            this.timeUnitScale[name] = this._scaleTimestampsToMs ? this._detectTimeUnitScale(vec) : 1;
        }
    }

    _buildDictionaryDecoder(vec){
        const hasDict = vec && vec.type && vec.type.dictionary;
        if (!hasDict || typeof vec.getDictionary !== 'function') return null;
        const dictVec = vec.getDictionary();
        return (rowIndex) => {
            const k = vec.get(rowIndex);
            return (k == null) ? null : dictVec.get(k);
        };
    }

    _initDecoders(){
        for (const [name, vec] of this._nameToVector.entries()){
            this._decoder[name] = this._decodeDictionaries ? this._buildDictionaryDecoder(vec) : null;
        }
    }

    _getScale(name){
        if (this.timeUnitScale[name] != null) return this.timeUnitScale[name];
        const s = this._scaleTimestampsToMs ? this._detectTimeUnitScale(this._getVector(name)) : 1;
        this.timeUnitScale[name] = s; return s;
    }

    _getDecoder(name){
        if (this._decoder[name] != null) return this._decoder[name];
        const vec = this._getVector(name);
        const d = this._decodeDictionaries ? this._buildDictionaryDecoder(vec) : null;
        this._decoder[name] = d;
        return d;
    }

    _isDerived(name){ return this._derived.has(name); }
    _getDerived(name){ return this._derived.get(name); }

    _isNumericArrowType(t){
        const type = (x)=>{
            if (!x) return null;
            if (x.typeId != null) return x;
            if (x.type && x.type.typeId != null) return x.type;
            return null;
        };
        let dt = type(t);
        if (!dt) return false;
        if (dt.typeId === arrow.Type.Dictionary){
            let inner = dt.valueType || null;
            if (!inner && t && typeof t.getDictionary==='function'){
                const dictVec = t.getDictionary && t.getDictionary();
                inner = dictVec && dictVec.type;
            }
            dt = type(inner);
            if (!dt) return false;
        }
        switch (dt.typeId){
            case arrow.Type.Int: case arrow.Type.Int8: case arrow.Type.Int16: case arrow.Type.Int32: case arrow.Type.Int64:
            case arrow.Type.Uint8: case arrow.Type.Uint16: case arrow.Type.Uint32: case arrow.Type.Uint64:
            case arrow.Type.Float: case arrow.Type.Float16: case arrow.Type.Float32: case arrow.Type.Float64:
            case arrow.Type.Decimal: case arrow.Type.Decimal128: case arrow.Type.Decimal256:
                return true;
            default: return false;
        }
    }

    _getValueGetter(name, params) {
        if (!this._getterCache) this._getterCache = new Map();

        let cached = this._getterCache.get(name);
        if (cached) return cached;

        const baseGetter = this._getValueGetterDirect(name, params);
        const engine = this;

        const getter = function (rowIndex) {
            const ri = _toInt32(rowIndex);
            const ov = engine._overlays.get(name);
            if (ov && engine._overlayHas(ov, ri)) {
                const v = engine._overlayGet(ov, ri);
                return v === undefined ? baseGetter(ri) : v;
            }
            return baseGetter(ri);
        };


        const exists =
            this._nameToVector.has(name) ||
            this._derived.has(name);

        if (exists) this._getterCache.set(name, getter);

        return getter;
    }

    _getValueGetterDirect(name, params) {

        if (this._isDerived(name)) {
            const d = this._getDerived(name);
            const settingsFn = this._settingsGetter.get(name);
            return (rowIndex) => d.getter(_toInt32(rowIndex), this, settingsFn ? settingsFn() : undefined, params);
        }

        const vec = this._getVector(name);
        if (!vec) {
            return (_rowIndex) => null;
        }

        const dec = this._getDecoder(name);
        if (dec) return (rowIndex) => dec(_toInt32(rowIndex));

        const scale = this._getScale(name);
        if (scale !== 1) return (rowIndex) => {
            let raw = vec.get(_toInt32(rowIndex));
            if (typeof raw === 'bigint') raw = Number(raw);
            return raw != null ? raw * scale : raw;
        };

        return (rowIndex) => {
            let raw = vec.get(_toInt32(rowIndex));
            if (typeof raw === 'bigint') raw = Number(raw);
            return raw;
        };
    }

    /* ------------------------------ public reads ----------------------------- */
    numRows(){ return this.table.numRows | 0; }
    numCols(){ return this.table.numCols | 0; }
    schema(){ return this.table.schema; }
    getRowCount() {return this.numRows()}

    fields() { return this._fieldNames; }
    _get_dtypes() { return this.schema().fields.reduce((m, f) => {
        m.set(f.name, f.type);
        return m
    }, new Map([]))};
    fieldSet() { if (!this._fieldSetCache) this._fieldSetCache = new Set(this._fieldNames); return this._fieldSetCache; }

    dtypes() {
        if (!this._dtypes) this._dtypes = this._get_dtypes();
        return this._dtypes;
    }

    getDtype(column) {
        if (!this._dtypes || !this.dtypes().has(column)) this._dtypes = this._get_dtypes();
        return this._dtypes.get(column)
    }

    getCell(rowIndex, column, params) {
        const name = typeof column === 'string' ? column : this._normalizeColumnSelector(column)[0];

        if (!this._nameToVector.has(name) &&
            !this._derived.has(name) &&
            !this._overlays.has(name)) {
            return null;
        }

        const get = this._getValueGetter(name, params);
        return get(_toInt32(rowIndex));
    }

    getCellById(id, column, params){
        const idx = this.getRowIndexById(id);
        return this.getCell(idx, column, params)
    }

    getRowObject(rowIndex, columns=null){
        const names = this._normalizeColumnSelector(columns);
        const obj = { [this._idProperty]: this.getRowIdByIndex(rowIndex) };
        for (let i=0;i<names.length;i++) {
            try {
                obj[names[i]] = this.getCell(rowIndex, names[i]);
            } catch (e) {
                console.error(`Failed to get cell: ${names[i]}`);
                obj[names[i]] = undefined;
            }

        }
        return obj;
    }

    getArrowTable() { return this.table || null; }
    asTable() { return this.table; }

    toMap(k='portfolioKey') {
        if (!k) return
        const r = new Map();
        this.getAllRows().forEach(row => {
            const key = row[k];
            if (key) r.set(key, row)
        });
        return r
    }

    /* ------------------------------ overlays core ---------------------------- */
    _ensureOverlay(name){
        try {
            let ov = this._overlays.get(name);
            const n = this.numRows();
            if (ov) {
                if (ov.kind === 'number') {
                    if (!(ov.mask instanceof Uint8Array) || ov.mask.length !== n || !(ov.values instanceof Float64Array) || ov.values.length !== n) {
                        const next = { kind: 'number', mask: new Uint8Array(n), values: new Float64Array(n), size: 0 };
                        if (ov.mask && ov.values) {
                            const lim = Math.min(n, ov.mask.length | 0);
                            next.mask.set(ov.mask.subarray(0, lim));
                            next.values.set(ov.values.subarray(0, lim));
                            let s = 0;
                            for (let i = 0; i < lim; i++) if (next.mask[i] !== 0) s++;
                            next.size = s;
                        }
                        this._overlays.set(name, (ov = next));
                    }
                } else {
                    if (!ov.map || typeof ov.map.set !== 'function') ov.map = new Map(ov.map);
                    if (typeof ov.size !== 'number') ov.size = ov.map.size;
                }
                return ov;
            }

            // Determine overlay kind quickly and correctly
            let overlayKind = 'other';

            if (this.columnDefs.has(name)) {
                const dt = this.columnDefs.get(name)?.context?.dataType; // string like 'number'|'text'|...
                if (dt === 'number' || dt === 'integer' || dt === 'float' || dt === 'percentage' || dt === 'currency') overlayKind = 'number';
            } else if (this._nameToVector.has(name)) {
                const vec = this._getVector(name);
                overlayKind = this._isNumericArrowType(vec) ? 'number' : 'other';
            } else if (this._derived.has(name)) {
                const meta = this._derived.get(name);
                const kind = meta && meta.kind;
                if (kind === 'number' || kind === 'integer' || kind === 'float' || kind === 'percentage' || kind === 'currency') {
                    overlayKind = 'number';
                } else {
                    try {
                        const g = this._getValueGetterDirect(name);
                        const nrows = this.numRows();
                        const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
                        if (isNum(g(0)) || (nrows && isNum(g(nrows >>> 1))) || (nrows && isNum(g(nrows - 1)))) overlayKind = 'number';
                    } catch {}
                }
            }

            ov = { kind: overlayKind === 'number' ? 'sparse_number' : 'other', map: new Map(), size: 0 };
            this._overlays.set(name, ov);
            return ov;
        } catch (e) {
            console.error(e);
            // Return a safe fallback to prevent null-deref in callers.
            const fallback = { kind: 'other', map: new Map(), size: 0 };
            this._overlays.set(name, fallback);
            return fallback;
        }
    }

    _promoteToDense(name, ov) {
        if (!ov || ov.kind !== 'sparse_number') return ov;
        const n = this.numRows() | 0;
        const DENSITY_THRESHOLD = 0.05; // 5% of rows edited → promote
        if (ov.size < n * DENSITY_THRESHOLD && ov.size < 1024) return ov;

        const mask = new Uint8Array(n);
        const values = new Float64Array(n);
        let size = 0;
        const mp = ov.map;

        for (const [ri, v] of mp) {
            if (ri < 0 || ri >= n) continue;
            if (v === null) {
                mask[ri] = MASK_NULL;
                size++;
            } else {
                const num = (typeof v === 'number' && Number.isFinite(v)) ? v : _coerceOverlayNumber(v);
                if (num === null || num === undefined) {
                    mask[ri] = MASK_NULL;
                } else {
                    mask[ri] = MASK_VALUE;
                    values[ri] = num;
                }
                size++;
            }
        }

        const dense = { kind: 'number', mask, values, size };
        this._overlays.set(name, dense);
        return dense;
    }

    _overlayHas(ov, ri) {
        if (!ov) return false;

        if (ov.kind === 'number') {
            const mask = ov.mask;
            if (!mask || ri < 0 || ri >= mask.length) return false;
            return mask[ri] !== MASK_NONE;
        }

        // sparse_number and 'other' both use map
        const mp = ov.map;
        return !!(mp && mp.has(ri));
    }

    _overlayGet(ov, ri){
        if (!ov) return undefined;
        if (ov.kind === 'number'){
            const mask = ov.mask;
            if (!mask || ri < 0 || ri >= mask.length) return undefined;
            const m = mask[ri];
            if (m === MASK_NULL) return null;
            return (m === MASK_VALUE && ov.values) ? ov.values[ri] : undefined;
        }
        return ov.map ? ov.map.get(ri) : undefined;
    }

    _overlaySet(ov, ri, value) {
        if (!ov) return;

        if (ov.kind === 'number') {
            const mask = ov.mask;
            const vals = ov.values;
            if (!mask || !vals || ri < 0 || ri >= mask.length) return;

            const prev = mask[ri];

            if (value === undefined) {
                if (prev !== MASK_NONE) {
                    mask[ri] = MASK_NONE;
                    if (prev === MASK_VALUE) vals[ri] = 0;
                    ov.size = Math.max(0, ov.size - 1);
                }
                return;
            }

            if (value === null) {
                if (prev === MASK_NONE) ov.size++;
                mask[ri] = MASK_NULL;
                return;
            }

            const num = _coerceOverlayNumber(value);
            if (num === null || num === undefined) {
                if (prev === MASK_NONE) ov.size++;
                mask[ri] = MASK_NULL;
                return;
            }

            if (prev === MASK_NONE) ov.size++;
            mask[ri] = MASK_VALUE;
            vals[ri] = num;
            return;
        }

        // sparse_number and 'other' — both use Map
        const mp = ov.map;
        if (!mp) return;

        if (value === undefined) {
            if (mp.delete(ri)) {
                ov.size = Math.max(0, ov.size - 1);
            }
            return;
        }

        if (!mp.has(ri)) ov.size++;
        mp.set(ri, value);
    }

    setOverlayValue(rowIndex, column, value, {bump=true}={}){
        const name = this._normalizeColumnSelector(column)[0];
        const r = _toInt32(rowIndex);
        const ov = this._ensureOverlay(name);
        this._overlaySet(ov, r, value);
        if (bump) this._bumpCellEpochByName(r, name);
        return this;
    }

    setOverlayValuesByColumnBatch(column, rowIndices, values, { freeze = false, bump = true } = {}) {
        let name;
        try {
            name = this._normalizeColumnSelector(column)[0];
        } catch {
            return { applied: 0, rowsChanged: new Int32Array(0) };
        }

        const known =
            this._nameToVector.has(name) ||
            this._derived.has(name) ||
            this.columnDefs.has(name);

        if (!known) return { applied: 0, rowsChanged: new Int32Array(0) };

        let ov = this._ensureOverlay(name);
        if (!ov) return { applied: 0, rowsChanged: new Int32Array(0) };

        const vals = values || [];
        const valsLen = vals.length | 0;

        if (!rowIndices || valsLen === 0) return { applied: 0, rowsChanged: new Int32Array(0) };

        const rowCount = this.numRows() | 0;

        const rowsIsI32 = rowIndices instanceof Int32Array;
        const rowsSrc = rowsIsI32 ? rowIndices : (Array.isArray(rowIndices) ? rowIndices : Array.from(rowIndices || []));
        const rowsLen = rowsSrc.length | 0;

        const n0 = Math.min(rowsLen, valsLen) | 0;
        if (n0 === 0) return { applied: 0, rowsChanged: new Int32Array(0) };

        // ---- Promotion decision ----
        // If the overlay is sparse_number and the incoming batch is large
        // enough to justify dense storage, promote now.  This avoids
        // Map overhead for bulk writes and lets us use the typed-array
        // fast path below.
        const PROMOTE_ABS = 512;
        const PROMOTE_REL = 0.05; // 5% of total rows
        const shouldPromote =
            ov.kind === 'sparse_number' &&
            ((n0 >= PROMOTE_ABS) || (n0 >= rowCount * PROMOTE_REL));

        if (shouldPromote) {
            ov = this._promoteToDense(name, ov);
            // _promoteToDense may return the same object if promotion
            // conditions inside it disagree — re-fetch to be safe.
            ov = this._overlays.get(name) || ov;
        }

        const isNumber = ov.kind === 'number';

        let rowsOut = null;
        let outCount = 0;
        let copying = !rowsIsI32;

        if (copying) rowsOut = new Int32Array(n0);

        // ---- Reusable coercion result (zero-alloc hot path) ----
        const _cnfResult = { kind: 0, num: 0 };

        const coerceNumberFast = (v) => {
            if (v === undefined) { _cnfResult.kind = 0; _cnfResult.num = 0; return _cnfResult; }
            if (v === null)      { _cnfResult.kind = 1; _cnfResult.num = 0; return _cnfResult; }

            if (typeof v === 'number') {
                if (Number.isFinite(v)) { _cnfResult.kind = 2; _cnfResult.num = v; }
                else                    { _cnfResult.kind = 1; _cnfResult.num = 0; }
                return _cnfResult;
            }
            if (typeof v === 'bigint') {
                const n = Number(v);
                if (Number.isFinite(n)) { _cnfResult.kind = 2; _cnfResult.num = n; }
                else                    { _cnfResult.kind = 1; _cnfResult.num = 0; }
                return _cnfResult;
            }

            let s0 = String(v).trim();
            if (s0 === '') { _cnfResult.kind = 1; _cnfResult.num = 0; return _cnfResult; }

            let isPct = false;
            if (s0.charCodeAt(s0.length - 1) === 37) { // '%'
                isPct = true;
                s0 = s0.slice(0, -1).trim();
                if (s0 === '') { _cnfResult.kind = 1; _cnfResult.num = 0; return _cnfResult; }
            }

            if (s0.charCodeAt(0) === 43) s0 = s0.slice(1); // '+'

            if (s0.indexOf(',') >= 0 || s0.indexOf(' ') >= 0 || s0.indexOf('\u00A0') >= 0) {
                s0 = s0.replace(/[\u00A0\s,](?=\d{3}\b)/g, '');
            }

            const n = Number(s0);
            if (!Number.isFinite(n)) { _cnfResult.kind = 1; _cnfResult.num = 0; return _cnfResult; }
            _cnfResult.kind = 2;
            _cnfResult.num = isPct ? (n * 0.01) : n;
            return _cnfResult;
        };

        // ---- Shared row-index validation ----
        // Returns validated ri or -1 for skip.  Inlined below for the
        // hot paths to avoid function-call overhead, but extracted here
        // for the sparse path where the overhead doesn't matter.
        const validateRi = (i) => {
            let ri;
            if (rowsIsI32) {
                ri = rowsSrc[i] | 0;
            } else {
                const raw = rowsSrc[i];
                if (typeof raw === 'number') {
                    if (!Number.isFinite(raw)) return -1;
                    ri = raw | 0;
                    if (ri !== raw) return -1;
                } else if (typeof raw === 'string') {
                    const t = raw.trim();
                    if (!t) return -1;
                    ri = t | 0;
                    if (t !== String(ri)) return -1;
                } else {
                    return -1;
                }
            }
            if (ri < 0 || ri >= rowCount) return -1;
            return ri;
        };

        // ================================================================
        //  DENSE NUMBER FAST PATH  (kind === 'number')
        // ================================================================
        if (isNumber) {
            const mask = ov.mask;
            const valsArr = ov.values;

            for (let i = 0; i < n0; i++) {
                let ri;

                // ---- inline row validation (avoids function call) ----
                if (rowsIsI32) {
                    ri = rowsSrc[i] | 0;
                    if (ri < 0 || ri >= rowCount) {
                        if (!copying) {
                            copying = true;
                            rowsOut = new Int32Array(n0);
                            for (let j = 0; j < i; j++) rowsOut[outCount++] = rowsSrc[j] | 0;
                        }
                        continue;
                    }
                } else {
                    const raw = rowsSrc[i];
                    if (typeof raw === 'number') {
                        if (!Number.isFinite(raw)) { continue; }
                        ri = raw | 0;
                        if (ri !== raw) { continue; }
                    } else if (typeof raw === 'string') {
                        const t = raw.trim();
                        if (!t) { continue; }
                        ri = t | 0;
                        if (t !== String(ri)) { continue; }
                    } else {
                        continue;
                    }
                    if (ri < 0 || ri >= rowCount) continue;
                }

                const cv = coerceNumberFast(vals[i]);
                const prev = mask[ri] | 0;

                if (cv.kind === 0) { // unset
                    if (prev !== MASK_NONE) {
                        mask[ri] = MASK_NONE;
                        ov.size--;
                        if (ov.size < 0) ov.size = 0;
                    }
                } else if (cv.kind === 1) { // null
                    if (prev === MASK_NONE) ov.size++;
                    mask[ri] = MASK_NULL;
                } else { // value
                    valsArr[ri] = cv.num;
                    if (prev === MASK_NONE) ov.size++;
                    mask[ri] = MASK_VALUE;
                }

                if (copying) rowsOut[outCount++] = ri;
            }

            // ================================================================
            //  SPARSE NUMBER PATH  (kind === 'sparse_number')
            //  Small batch that didn't meet the promotion threshold.
            //  Uses Map storage but coerces values to numbers so a
            //  future promotion to dense is lossless.
            // ================================================================
        } else if (ov.kind === 'sparse_number') {
            const mp = ov.map || (ov.map = new Map());

            for (let i = 0; i < n0; i++) {
                const ri = validateRi(i);
                if (ri < 0) {
                    // For I32 source we need the deferred-copy logic
                    if (rowsIsI32 && !copying) {
                        copying = true;
                        rowsOut = new Int32Array(n0);
                        for (let j = 0; j < i; j++) rowsOut[outCount++] = rowsSrc[j] | 0;
                    }
                    continue;
                }

                const v = vals[i];

                if (v === undefined) {
                    if (mp.delete(ri)) { ov.size--; if (ov.size < 0) ov.size = 0; }
                } else if (v === null) {
                    if (!mp.has(ri)) ov.size++;
                    mp.set(ri, null);
                } else {
                    // Coerce to number for consistency with dense path.
                    const cv = coerceNumberFast(v);
                    if (cv.kind === 0) {
                        if (mp.delete(ri)) { ov.size--; if (ov.size < 0) ov.size = 0; }
                    } else if (cv.kind === 1) {
                        if (!mp.has(ri)) ov.size++;
                        mp.set(ri, null);
                    } else {
                        if (!mp.has(ri)) ov.size++;
                        mp.set(ri, cv.num);
                    }
                }

                if (copying) rowsOut[outCount++] = ri;
            }

            // ================================================================
            //  GENERIC MAP PATH  (kind === 'other')
            //  Non-numeric columns: text, date, boolean, etc.
            // ================================================================
        } else {
            const mp = ov.map || (ov.map = new Map());

            for (let i = 0; i < n0; i++) {
                let ri;

                // ---- inline row validation ----
                if (rowsIsI32) {
                    ri = rowsSrc[i] | 0;
                    if (ri < 0 || ri >= rowCount) {
                        if (!copying) {
                            copying = true;
                            rowsOut = new Int32Array(n0);
                            for (let j = 0; j < i; j++) rowsOut[outCount++] = rowsSrc[j] | 0;
                        }
                        continue;
                    }
                } else {
                    const raw = rowsSrc[i];
                    if (typeof raw === 'number') {
                        if (!Number.isFinite(raw)) continue;
                        ri = raw | 0;
                        if (ri !== raw) continue;
                    } else if (typeof raw === 'string') {
                        const t = raw.trim();
                        if (!t) continue;
                        ri = t | 0;
                        if (t !== String(ri)) continue;
                    } else {
                        continue;
                    }
                    if (ri < 0 || ri >= rowCount) continue;
                }

                const v = vals[i];
                if (v === undefined) {
                    if (mp.delete(ri)) { ov.size--; if (ov.size < 0) ov.size = 0; }
                } else {
                    if (!mp.has(ri)) ov.size++;
                    mp.set(ri, v);
                }

                if (copying) rowsOut[outCount++] = ri;
            }
        }

        // ================================================================
        //  Post-write bookkeeping
        // ================================================================
        const rowsChanged = copying
            ? ((outCount && rowsOut) ? rowsOut.subarray(0, outCount) : new Int32Array(0))
            : rowsSrc.subarray(0, n0);

        if ((rowsChanged.length | 0) === 0) return { applied: 0, rowsChanged: new Int32Array(0) };

        // Freeze derived cells when asked
        if (freeze && this._isDerived(name)) {
            for (let i = 0; i < rowsChanged.length; i++) this._markFrozen(name, rowsChanged[i] | 0);
        }

        // Unfreeze dependents once per row if base column is physical
        if (this._nameToVector.has(name)) {
            for (let i = 0; i < rowsChanged.length; i++) this._unfreezeDependentsOnChange(name, rowsChanged[i] | 0);
        }

        if (bump) this._emitEpochChange({ rowsChanged, colsChanged: name });
        return { applied: rowsChanged.length | 0, rowsChanged };
    }

    setOverlayValuesByRowBatch(rowIndex, changes, { freezeDerived = true, bump = true } = {}) {
        const ri = rowIndex | 0;
        const n = this.numRows() | 0;
        if (ri < 0 || ri >= n) return { applied: 0, colsChanged: [] };
        if (!changes) return { applied: 0, colsChanged: [] };

        const changedCols = new Set();

        const applyOne = (name, v) => {
            if (!name) return;

            const known =
                this._nameToVector.has(name) ||
                this._derived.has(name) ||
                this.columnDefs.has(name);

            if (!known) return;

            const ov = this._ensureOverlay(name);
            if (!ov) return;

            this._overlaySet(ov, ri, v);
            changedCols.add(name);

            if (freezeDerived && this._isDerived(name)) this._markFrozen(name, ri);
            if (this._nameToVector.has(name)) this._unfreezeDependentsOnChange(name, ri);
        };

        if (changes instanceof Map) {
            for (const [k, v] of changes.entries()) applyOne(String(k), v);
        } else if (Array.isArray(changes)) {
            for (let i = 0; i < changes.length; i++) {
                const pair = changes[i];
                if (!pair || pair.length < 2) continue;
                applyOne(String(pair[0]), pair[1]);
            }
        } else if (typeof changes === 'object') {
            const names = Object.keys(changes);
            for (let i = 0; i < names.length; i++) {
                const name = names[i];
                applyOne(name, changes[name]);
            }
        } else {
            return { applied: 0, colsChanged: [] };
        }

        const colsChanged = Array.from(changedCols);
        if (colsChanged.length === 0) return { applied: 0, colsChanged: [] };

        if (bump) this._emitEpochChange({ rowsChanged: [ri], colsChanged });
        return { applied: colsChanged.length, colsChanged };
    }

    clearOverlayValue(rowIndex, column, {bump=true}={}){
        const name = this._normalizeColumnSelector(column)[0];
        const r = _toInt32(rowIndex);
        const ov = this._overlays.get(name);
        if (!ov) return this;
        if (this._overlayHas(ov, r)){
            this._overlaySet(ov, r, undefined);
            if (bump) this._bumpCellEpochByName(r, name);
        }
        return this;
    }

    clearOverlayColumn(column, {bump=true}={}){
        const name = this._normalizeColumnSelector(column)[0];
        const ov = this._overlays.get(name);
        if (!ov) return this;
        if (ov.kind === 'number'){ ov.mask.fill(MASK_NONE); ov.size = 0; }
        else { ov.map.clear(); ov.size = 0; }
        if (bump) this._bumpColEpochByName(name);
        return this;
    }

    hasEdits(column=null){
        if (column == null){ for (const ov of this._overlays.values()) if (ov.size>0) return true; return false; }
        const name = this._normalizeColumnSelector(column)[0];
        const ov = this._overlays.get(name);
        return !!ov && ov.size>0;
    }

    /* ------------------------------- epoch bumps ----------------------------- */
    _colIndex(name){
        if (this._nameToIndex.has(name)) return this._nameToIndex.get(name);
        const key = `__derived__${name}`;
        if (!this._derivedNameToIndex) this._derivedNameToIndex = new Map();
        if (!this._derivedNameToIndex.has(name)) this._derivedNameToIndex.set(name, (this._nameToIndex.size + this._derivedNameToIndex.size));
        return this._derivedNameToIndex.get(name);
    }

    _clearCellCaches(names) {
        if (names === true) {
            this._classCache.clear();
            this._fmtCache.clear();
        } else {
            asArray(names).forEach(n => {
                if (this._classCache.has(n)) this._classCache.get(n).clear();
                if (this._fmtCache.has(n)) this._fmtCache.get(n).clear();
            });
        }
    }

    _bumpColEpochByName(name){
        this._emitEpochChange({ colsChanged: name });
    }
    _bumpCellEpochByName(rowIndex, name){
        this._emitEpochChange({ rowsChanged: [_toInt32(rowIndex)], colsChanged: name });
    }
    _bumpRowEpoch(rowIndex){
        this._emitEpochChange({ rowsChanged: [_toInt32(rowIndex)] });
    }
    _setupDerivedClassCache() {}

    async _broadcastEdit(rowIndex, columnName, value) {
        const engine = this;
        this._editCoalescer.enqueue(rowIndex, columnName, value, engine);
    }
    async flushPendingEdits() {
        await this._editCoalescer.drain();
    }

    /* ------------------------------- persistence ----------------------------- */
    async persistEdit(rowIndex, column, value){
        const name = this._normalizeColumnSelector(column)[0];
        if (!this._nameToVector.has(name)){
            throw new Error(`persistEdit: "${name}" is not a physical Arrow column`);
        }
        const ri = _toInt32(rowIndex);
        await this._broadcastEdit(ri, name, value);

        const ov = this._ensureOverlay(name);
        this._overlaySet(ov, ri, value);
        this._bumpCellEpochByName(ri, name);
        this._unfreezeDependentsOnChange(name, ri);
        return this;
    }

    _ensureEditDrainScheduled() {
        if (this._editDrainScheduled) return;
        this._editDrainScheduled = true;

        const run = () => {
            this._editDrainScheduled = false;
            try {
                const p = this._editCoalescer?.drain?.();
                if (p && typeof p.then === 'function') p.catch(() => {});
            } catch {}
        };

        // Use setTimeout to yield to the event loop — queueMicrotask would
        // chain synchronously with the epoch/emit microtask cascade.
        setTimeout(run, 0);
    }

    getEditHandler(name) {
        if (this._valueSetters.has(name)) return this._valueSetters.get(name);

        if (this._nameToVector.has(name)) {
            return async ({ rowIndex, value }) => { await this.persistEdit(rowIndex, name, value); return true; };
        }

        if (this._derived.has(name)) {
            const def = this._derived.get(name);
            const inverse = typeof def.inverse === 'function' ? def.inverse : null;
            const settingsFn = this._settingsGetter.get(name);

            if (inverse) {
                return async ({ rowIndex, value }) => {
                    const planMaybe = inverse((rowIndex|0), value, this, settingsFn ? settingsFn() : undefined);
                    const plan = (planMaybe && typeof planMaybe.then === 'function') ? await planMaybe : planMaybe;
                    return this._applyEditPlan((rowIndex|0), plan, { bumpSource: true });
                };
            }

            return async ({ rowIndex, value }) => {
                this.setOverlayValue((rowIndex|0), name, value);
                this._markFrozen(name, (rowIndex|0));
                return true;
            };
        }

        return async () => false;
    }

    _numericDt(x){
        return (x === 'number') || (x === 'integer') || (x === 'float') || (x === 'percentage') || (x === 'currency')
    }

    async applyValue(rowIndex, column, raw, colDef){
        const name = this._normalizeColumnSelector(column)[0];
        let typeHint = colDef?.context?.dataType

        let v = raw;
        if (!colDef?.context?.rawEdit) {
            typeHint = typeHint == null ? 'number' : typeHint;
            if (this._numericDt(typeHint)) {
                v = coerceToNumber(raw, {onNaN: null});
            }
        }

        if (!this._isDerived(name)) {
            this._editCount = (this._editCount || 0) + 1;
            if (this._editCount % 20 === 0) {
                await this.materializeEdits({commit:'auto'});
            }
        }

        const handler = this.getEditHandler(name);
        return handler({ rowIndex:_toInt32(rowIndex), value:v, engine:this });
    }

    async _applyEditPlan(rowIndex, plan, { bumpSource=true }={}) {
        const ops = Array.isArray(plan) ?
            plan : (plan ? [plan] : []);
        if (ops.length === 0) return false;
        for (let i = 0; i < ops.length; i++) {
            const op = ops[i] ||
                {};
            const col = this._normalizeColumnSelector(op.column)[0];
            const isPhysical = this._nameToVector.has(col);
            const isDerived  = !isPhysical && this._derived.has(col);
            let mode = op.mode;
            if (!mode) {
                mode = isPhysical ? 'persist' : 'overlay';
            }

            let value = op.value;
            const isClear = (value === CLEAR_SENTINEL || value === null);
            if (mode === 'overlay' || (!isPhysical && mode !== 'persist')) {
                if (isClear) {
                    // remove overlay -> reveal base
                    this.clearOverlayValue(rowIndex, col, { bump: false });
                    if (this._isFrozen(col, rowIndex)) this._unfreeze(col, rowIndex);
                } else {
                    // set overlay (defer bump to end for consistency)
                    this.setOverlayValue(rowIndex, col, value, { bump: false });
                    if (op.freeze || isDerived) this._markFrozen(col, rowIndex);
                }
                if (bumpSource) this._bumpCellEpochByName(rowIndex, col);
                continue;
            }

            // Persist path (physical columns)
            if (!isPhysical && mode === 'persist') {
                // Treat non-physical persist as overlay+freeze fallback
                if (isClear) {
                    this.clearOverlayValue(rowIndex, col, { bump: false
                    });
                    if (this._isFrozen(col, rowIndex)) this._unfreeze(col, rowIndex);
                } else {
                    this.setOverlayValue(rowIndex, col, value, { bump: false });
                    this._markFrozen(col, rowIndex);
                }
                if (bumpSource) this._bumpCellEpochByName(rowIndex, col);
                continue;
            }

            // Physical persist (default path)
            // persistEdit already calls _bumpCellEpochByName + _unfreezeDependentsOnChange
            await this.persistEdit(rowIndex, col, isClear ? null : value);
        }
        return true;
    }

    async _applyEditPlanBatch(rowIndex, plan, tracker = {}, opts={}) {
        const ops = Array.isArray(plan) ? plan : (plan ? [plan] : []);
        if (ops.length === 0) return tracker;

        const suppressEditSignals = opts?.suppressEditSignals === true;
        const deferUnfreeze = opts?.deferUnfreeze === true;

        const changedCols = tracker.changedCols || (tracker.changedCols = new Set());
        const changedRows = tracker.changedRows || (tracker.changedRows = new Set());
        const physicalCols = deferUnfreeze ? (tracker.physicalCols || (tracker.physicalCols = new Set())) : null;

        const ri = (rowIndex | 0);
        let broadcasted = false;

        for (let i = 0; i < ops.length; i++) {
            const op = ops[i] || {};
            const col = this._normalizeColumnSelector(op.column)[0];

            const isPhysical = this._nameToVector.has(col);
            const isDerived  = !isPhysical && this._derived.has(col);
            let mode = op.mode;
            if (!mode) {
                mode = isPhysical ? 'persist' : 'overlay';
            }

            const isClear = (op.value === CLEAR_SENTINEL || op.value === null);
            const value   = isClear ? null : op.value;

            /* Overlay path or non-physical without explicit persist -> same as base (silent overlay + optional freeze) */
            if (mode === 'overlay' || (!isPhysical && mode !== 'persist')) {
                if (isClear) {
                    this.clearOverlayValue(ri, col, { bump: false });
                    if (this._isFrozen(col, ri)) this._unfreeze(col, ri);
                } else {
                    this.setOverlayValue(ri, col, value, { bump: false });
                    if (op.freeze || isDerived) this._markFrozen(col, ri);
                }
                changedCols.add(col);
                changedRows.add(ri);
                if (isPhysical) {
                    if (deferUnfreeze) physicalCols.add(col);
                    else this._unfreezeDependentsOnChange(col, ri);
                }
                continue;
            }

            /* Non-physical with 'persist' requested -> keep base fallback (overlay+freeze) */
            if (!isPhysical && mode === 'persist') {
                if (isClear) {
                    this.clearOverlayValue(ri, col, { bump: false });
                    if (this._isFrozen(col, ri)) this._unfreeze(col, ri);
                } else {
                    this.setOverlayValue(ri, col, value, { bump: false });
                    this._markFrozen(col, ri);
                }
                changedCols.add(col);
                changedRows.add(ri);
                continue;
            }

            const ov = this._ensureOverlay(col);
            this._overlaySet(ov, ri, value);
            changedCols.add(col);
            changedRows.add(ri);

            if (isPhysical) {
                if (deferUnfreeze) physicalCols.add(col);
                else this._unfreezeDependentsOnChange(col, ri);
            }

            try {
                if (!suppressEditSignals) {
                    this._editCoalescer?.silent?.(ri, col, value, this);
                    broadcasted = true;
                }
            } catch {}
        }

        if (broadcasted) this._ensureEditDrainScheduled();
        return tracker;
    }

    /* --------------------------- freeze / unfreeze --------------------------- */
    _markFrozen(name, rowIndex){
        let set = this._frozenOverrides.get(name);
        if (!set) { set = new Set(); this._frozenOverrides.set(name, set); }
        set.add(rowIndex | 0);
    }
    _isFrozen(name, rowIndex){
        const set = this._frozenOverrides.get(name);
        return set ? set.has(rowIndex | 0) : false;
    }
    _unfreeze(name, rowIndex){
        const set = this._frozenOverrides.get(name);
        if (set && set.delete(rowIndex | 0)) {
            if (set.size === 0) this._frozenOverrides.delete(name);
            this.clearOverlayValue(rowIndex | 0, name, { bump: false });
            this._bumpCellEpochByName(rowIndex | 0, name);
        }
    }
    _unfreezeAllFrozenFor(name){
        const set = this._frozenOverrides.get(name);
        if (!set || set.size === 0) return;
        const rows = Array.from(set);
        this._frozenOverrides.delete(name);
        for (let i = 0; i < rows.length; i++) {
            this.clearOverlayValue(rows[i], name, { bump: false });
            this._bumpCellEpochByName(rows[i], name);
        }
    }
    _freeAllFrozen({ trimToRowCount = null, trimUnknownColumns = true, clearOverlays = false, bump = false, clearAll = false } = {}){
        const frozen = this._frozenOverrides;
        if (!frozen || frozen.size === 0) return 0;

        const n = (trimToRowCount == null) ? (this.table ? (this.table.numRows | 0) : -1) : (trimToRowCount | 0);

        // Optional "nuke" mode (off by default).
        if (clearAll === true) {
            let removed = 0;
            if (clearOverlays) {
                for (const [col, rowSet] of frozen) {
                    const ov = this._overlays.get(col);
                    if (!ov || !rowSet || rowSet.size === 0) { removed += rowSet ? rowSet.size : 0; continue; }
                    removed += rowSet.size;
                    if (ov.kind === 'number') {
                        const mask = ov.mask;
                        if (!mask) continue;
                        for (const ri of rowSet) {
                            const prev = mask[ri] | 0;
                            if (prev !== MASK_NONE) { mask[ri] = MASK_NONE; ov.size--; }
                        }
                        if (ov.size < 0) ov.size = 0;
                    } else {
                        const mp = ov.map;
                        if (!mp) continue;
                        for (const ri of rowSet) {
                            if (mp.delete(ri)) ov.size--;
                        }
                        if (ov.size < 0) ov.size = 0;
                    }
                }
            } else {
                for (const [, rowSet] of frozen) removed += rowSet ? rowSet.size : 0;
            }
            frozen.clear();
            if (bump) this._emitEpochChange({ rowsChanged: true, colsChanged: true });
            return removed;
        }

        let removed = 0;
        const hasVec = this._nameToVector;
        const hasDer = this._derived;
        const hasDef = this.columnDefs;
        const colsToDelete = [];

        for (const [col, rowSet] of frozen) {
            // Check if entire column should be dropped
            if (trimUnknownColumns === true && !(hasVec.has(col) || hasDer.has(col) || hasDef.has(col))) {
                if (clearOverlays) {
                    const ov = this._overlays.get(col);
                    if (ov) {
                        for (const ri of rowSet) this._overlaySet(ov, ri, undefined);
                    }
                }
                removed += rowSet.size;
                colsToDelete.push(col);
                continue;
            }

            // Trim out-of-bounds rows within the column
            if (n >= 0) {
                const badRows = [];
                for (const ri of rowSet) {
                    if (ri < 0 || ri >= n) badRows.push(ri);
                }
                if (badRows.length > 0) {
                    for (let i = 0; i < badRows.length; i++) {
                        rowSet.delete(badRows[i]);
                        if (clearOverlays) {
                            const ov = this._overlays.get(col);
                            if (ov) this._overlaySet(ov, badRows[i], undefined);
                        }
                    }
                    removed += badRows.length;
                    if (rowSet.size === 0) colsToDelete.push(col);
                }
            }
        }

        for (let i = 0; i < colsToDelete.length; i++) frozen.delete(colsToDelete[i]);
        if (removed && bump) this._emitEpochChange({ rowsChanged: true, colsChanged: true });
        return removed;
    }
    _unfreezeDependentsOnChange(baseColName, rowIndex){
        if (this._frozenOverrides.size === 0) return;
        const deps = this.getDependentsClosure(baseColName) || [];
        for (let i=0;i<deps.length;i++){
            const d = deps[i];
            if (!this._derived.has(d)) continue;
            if (rowIndex == null){
                this._unfreezeAllFrozenFor(d);
            } else {
                if (this._isFrozen(d, rowIndex)) this._unfreeze(d, rowIndex);
            }
        }
    }

    /**
     * Batch-unfreeze all derived dependents for a set of physical columns and rows.
     * Unlike _unfreezeDependentsOnChange (which is called per-column per-row and emits
     * per-cell epoch bumps), this method:
     *   1. Computes the union of all derived dependents ONCE
     *   2. Clears frozen overlays in bulk WITHOUT individual epoch bumps
     *   3. Caller is responsible for emitting ONE coalesced epoch afterward
     *
     * This eliminates O(cols × rows × deps) string-concat + epoch-bump overhead.
     */
    _batchUnfreezeDependents(physicalCols, rowIndices) {
        if (this._frozenOverrides.size === 0) return;

        // Collect the union of all derived dependents across all changed physical cols
        const derivedDeps = new Set();
        for (const col of physicalCols) {
            const deps = this.getDependentsClosure(col) || [];
            for (let i = 0; i < deps.length; i++) {
                if (this._derived.has(deps[i])) derivedDeps.add(deps[i]);
            }
        }
        if (derivedDeps.size === 0) return;

        // For each derived dependent, unfreeze affected rows without epoch bumps
        for (const dep of derivedDeps) {
            const frozenSet = this._frozenOverrides.get(dep);
            if (!frozenSet || frozenSet.size === 0) continue;

            const ov = this._overlays.get(dep);
            for (const ri of rowIndices) {
                if (frozenSet.delete(ri)) {
                    // Clear overlay inline (skip clearOverlayValue overhead)
                    if (ov) this._overlaySet(ov, ri, undefined);
                }
            }
            if (frozenSet.size === 0) this._frozenOverrides.delete(dep);
        }
        // NO epoch bumps — caller emits one coalesced epoch
    }

    /* ----------------------------- derived columns --------------------------- */
    registerDerivedColumn(name, getter, meta={}){
        if (!name || typeof getter!=='function') throw new Error('registerDerivedColumn: invalid args');
        // if (this._nameToVector.has(name)) throw new Error('registerDerivedColumn: name collides with physical column: '+name);

        let baseDeps = [];
        if (Array.isArray(meta.deps)) baseDeps = meta.deps.slice();
        else if (typeof meta.deps === 'function') baseDeps = asArray(meta.deps(this));
        baseDeps = baseDeps.filter(d => d && d !== name);

        const depSet = new Set(baseDeps);
        this._columnDependencies.set(name, depSet);
        for (let i=0;i<baseDeps.length;i++){
            const depName = baseDeps[i];
            if (!this._columnDependents.has(depName)) this._columnDependents.set(depName, new Set());
            this._columnDependents.get(depName).add(name);
        }
        this._depClosureCache.clear(); this._dependentsClosureCache.clear();

        const def = { getter, async: !!meta.async, inverse: meta.inverse, kind: meta.kind, settings: meta.settings };
        this._derived.set(name, def);

        this._detachSettingsSubscriptions(name);
        const settingsSpec = meta.settings;
        if (settingsSpec){

            const dict = typeof settingsSpec.dict === 'function' ? settingsSpec.dict(this.table) : (settingsSpec.dict || null);
            const keys = asArray(settingsSpec.keys || settingsSpec.settingKeys || settingsSpec.settings);
            if (dict && keys.length){
                const subs = new Set();
                const add = (fn) => { try{ const off = fn || (()=>{}); subs.add(off); }catch{} };
                for (let k=0;k<keys.length;k++){
                    const key = keys[k];
                    if (key === "*") {
                        add(dict.onChanges(() => {
                            if (this._getterCache) this._getterCache.delete(name);
                            if (this._fmtCache && this._fmtCache.has(name)) {
                                this._fmtCache.get(name).clear();
                            }
                            if (this._classCache && this._classCache.has(name)) {
                                this._classCache.get(name).clear();
                            }
                            this._unfreezeAllFrozenFor(name);
                            this._bumpColEpochByName(name);
                        }));
                    } else if (typeof dict.onValueAddedOrChanged === 'function'){
                        add(dict.onValueAddedOrChanged(key, () => {
                            if (this._getterCache) this._getterCache.delete(name);
                            if (this._fmtCache && this._fmtCache.has(name)) {
                                this._fmtCache.get(name).clear();
                            }
                            if (this._classCache && this._classCache.has(name)) {
                                this._classCache.get(name).clear();
                            }

                            this._unfreezeAllFrozenFor(name);
                            this._bumpColEpochByName(name);
                        }));
                    } else if (typeof dict.onChange === 'function'){
                        add(dict.onChange(key, () => {
                            if (this._getterCache) this._getterCache.delete(name);
                            if (this._fmtCache && this._fmtCache.has(name)) {
                                this._fmtCache.get(name).clear();
                            }
                            if (this._classCache && this._classCache.has(name)) {
                                this._classCache.get(name).clear();
                            }
                            this._unfreezeAllFrozenFor(name);
                            this._bumpColEpochByName(name);
                        }));
                    }
                }
                this._derivedSettingsSubs.set(name, subs);
                if (typeof dict.asObject === 'function' && keys.length){
                    this._settingsGetter.set(name, () => {
                        const obj = dict.asObject();
                        const out = {}; for (let i=0;i<keys.length;i++){ const k = keys[i]; out[k] = obj[k]; }
                        return out;
                    });
                } else if (typeof dict.get === 'function' && keys.length){
                    this._settingsGetter.set(name, () => {
                        const out = {}; for (let i=0;i<keys.length;i++){ const k = keys[i]; out[k] = dict.get(k); }
                        return out;
                    });
                }
            }
        }

        if (this._getterCache) this._getterCache.delete(name);
        this._bumpColEpochByName(name);
        return this;
    }

    addDependencies(derivedColName, newDepsArray) {
        if (!this._derived.has(derivedColName) || !Array.isArray(newDepsArray) || newDepsArray.length === 0) {
            return this;
        }

        const depSet = this._columnDependencies.get(derivedColName);
        if (!depSet) {
            // This shouldn't happen, but to be safe
            this._columnDependencies.set(derivedColName, new Set(newDepsArray));
        }

        for (const depName of newDepsArray) {
            if (depName && depName !== derivedColName) {
                if (depSet) depSet.add(depName); // Add to existing set

                if (!this._columnDependents.has(depName)) {
                    this._columnDependents.set(depName, new Set());
                }
                this._columnDependents.get(depName).add(derivedColName);
            }
        }

        this._depClosureCache.clear();
        this._dependentsClosureCache.clear();
        this._bumpColEpochByName(derivedColName); // Mark as dirty
        return this;
    }

    _detachSettingsSubscriptions(name){
        const subs = this._derivedSettingsSubs.get(name);
        if (!subs) return;
        for (const s of subs){
            try{
                if (s && typeof s.unsubscribe === 'function') s.unsubscribe();
                else if (typeof s === 'function') s();
            }catch{}
        }
        this._derivedSettingsSubs.delete(name);
    }

    listDerivedColumns(){ return Array.from(this._derived.keys()); }

    getDependenciesClosure(columns) {
        const seeds = Array.isArray(columns) ? columns : [columns];
        const out = new Set();
        const stack = [];

        for (let i = 0; i < seeds.length; i++) {
            const name = seeds[i];
            if (name == null) continue;
            if (!out.has(name)) {
                out.add(name);
                stack.push(name);
            }
        }

        while (stack.length) {
            const current = stack.pop();
            let cached = this._depClosureCache.get(current);

            if (!cached) {
                cached = new Set();
                const direct = this._columnDependencies.get(current);

                if (direct && direct.size) {
                    const localVisited = new Set();
                    const dfsStack = Array.from(direct);

                    while (dfsStack.length) {
                        const next = dfsStack.pop();
                        if (localVisited.has(next)) continue;
                        localVisited.add(next);
                        cached.add(next);

                        const further = this._columnDependencies.get(next);
                        if (further && further.size) {
                            for (const f of further) dfsStack.push(f);
                        }
                    }
                }

                this._depClosureCache.set(current, cached);
            }

            for (const dep of cached) {
                if (!out.has(dep)) {
                    out.add(dep);
                    stack.push(dep);
                }
            }
        }

        return Array.from(out);
    }

    getDependentsClosure(columns) {
        const seeds = Array.isArray(columns) ? columns : [columns];
        const out = new Set();
        const stack = [];

        for (let i = 0; i < seeds.length; i++) {
            const name = seeds[i];
            if (name == null) continue;
            if (!out.has(name)) {
                out.add(name);
                stack.push(name);
            }
        }

        while (stack.length) {
            const current = stack.pop();
            let cached = this._dependentsClosureCache.get(current);

            if (!cached) {
                cached = new Set();
                const direct = this._columnDependents.get(current);

                if (direct && direct.size) {
                    const localVisited = new Set();
                    const dfsStack = Array.from(direct);

                    while (dfsStack.length) {
                        const next = dfsStack.pop();
                        if (localVisited.has(next)) continue;
                        localVisited.add(next);
                        cached.add(next);

                        const further = this._columnDependents.get(next);
                        if (further && further.size) {
                            for (const f of further) dfsStack.push(f);
                        }
                    }
                }

                this._dependentsClosureCache.set(current, cached);
            }

            for (const dep of cached) {
                if (!out.has(dep)) {
                    out.add(dep);
                    stack.push(dep);
                }
            }
        }

        return Array.from(out);
    }

    /* ---------------------------- materialize edits -------------------------- */
    async materializeEdits({ commit = 'auto' } = {}) {
        if (this._disposed) return this.table;

        // Cancel any pending debounce timer so we don't double-fire.
        if (this._materializeTimer) {
            clearTimeout(this._materializeTimer);
            this._materializeTimer = null;
        }

        if (!this._materializeResolvers) this._materializeResolvers = [];

        return new Promise((resolve) => {
            this._materializeResolvers.push(resolve);

            // Debounce: coalesce rapid successive calls into one execution.
            this._materializeTimer = setTimeout(async () => {
                this._materializeTimer = null;

                // Snapshot and clear the resolver list atomically so any
                // calls that arrive *during* execution start a fresh batch.
                const resolvers = this._materializeResolvers;
                this._materializeResolvers = [];

                let result;
                try {
                    result = await this._materializeEdits({ commit });
                } catch {
                    result = this.table;
                }
                for (let i = 0; i < resolvers.length; i++) resolvers[i](result);
            }, 200);
        });
    }

    async _materializeEdits({ commit = 'auto', columns: scopeCols = null } = {}) {
        // Reentrancy guard: if already materializing, enqueue and wait.
        if (this._isMaterializing) {
            return new Promise((resolve) => {
                if (!this._materializeQueue) this._materializeQueue = [];
                this._materializeQueue.push({ commit, scopeCols, resolve });
            });
        }

        this._isMaterializing = true;

        try {
            let result = this._materializeEditsInner(commit, scopeCols);

            // Drain loop: keep processing queued requests until the queue is
            // empty.  Each iteration merges all pending entries into a single
            // pass so we converge quickly.  The loop structure guarantees
            // _isMaterializing is held for the entire drain, preventing new
            // concurrent executions from slipping through.
            while (this._materializeQueue && this._materializeQueue.length) {
                const queued = this._materializeQueue.splice(0);

                // Merge: if ANY queued call wants commit:true, honour it.
                // Merge scope columns: null means "all columns".
                let mergedCommit = commit;
                let mergedScope = scopeCols ? new Set(scopeCols) : null;

                for (let i = 0; i < queued.length; i++) {
                    if (queued[i].commit === true) mergedCommit = true;

                    if (!queued[i].scopeCols) {
                        mergedScope = null; // null = all columns
                    } else if (mergedScope) {
                        const sc = queued[i].scopeCols;
                        if (sc instanceof Set) {
                            for (const c of sc) mergedScope.add(c);
                        } else if (Array.isArray(sc)) {
                            for (let j = 0; j < sc.length; j++) mergedScope.add(sc[j]);
                        }
                    }
                }

                let drainResult;
                try {
                    drainResult = this._materializeEditsInner(mergedCommit, mergedScope);
                } catch {
                    drainResult = this.table;
                }

                // Resolve all waiters from this batch with the drain result.
                for (let i = 0; i < queued.length; i++) queued[i].resolve(drainResult);

                // Update result for the original caller in case it cares.
                result = drainResult;
            }

            return result;
        } finally {
            // ALWAYS reset the flag, no matter what.  This is the single
            // point of responsibility — no other code path touches it.
            this._isMaterializing = false;
        }
    }

    /**
     * Inner synchronous materialize — no awaits, no yielding, no reentrancy risk.
     * @param {string|boolean} commit - 'auto'|true|false
     * @param {Set|null} scopeCols - if provided, only rebuild these columns (others left as-is)
     */
    _materializeEditsInner(commit, scopeCols) {
        const n = this.numRows() | 0;
        if (this._overlays.size === 0 || n === 0) return this.table;

        const physNames = this._fieldNames.filter(nm => this._nameToVector.has(nm));
        if (physNames.length === 0) return this.table;

        const tableColumns = {};
        let any = false;
        const rebuilt = [];

        for (let i = 0; i < physNames.length; i++) {
            const name = physNames[i];
            const vec = this._getVector(name);
            const ov = this._overlays.get(name);

            // Skip columns not in scope (if scope is specified)
            if (scopeCols && !scopeCols.has(name)) { tableColumns[name] = vec; continue; }

            if (!ov || ov.size === 0) { tableColumns[name] = vec; continue; }

            const density = ov.size / Math.max(1, n);
            const should = (commit === true) || (commit !== false && density >= this._autoCommitThreshold);
            if (!should) { tableColumns[name] = vec; continue; }

            const isBigIntType = (vec.type instanceof arrow.Int64) ||
                ((vec.type instanceof arrow.Int) && vec.type.bitWidth === 64);
            const builderType = isBigIntType ? new arrow.Float64() : vec.type;
            const builder = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });

            if (ov.kind === 'number') {
                const mask = ov.mask, vals = ov.values;
                for (let r = 0; r < n; r++) {
                    const m = mask[r] | 0;
                    if (m === MASK_NONE) { let val = vec.get(r); if (typeof val === 'bigint') val = Number(val); builder.append(val); continue; }
                    if (m === MASK_NULL) { builder.append(null); continue; }
                    const v = vals[r];
                    builder.append(Number.isFinite(v) ? v : null);
                }
            } else {
                const mp = ov.map || new Map();
                for (let r = 0; r < n; r++) {
                    if (mp.has(r)) { let val = mp.get(r); if (typeof val === 'bigint') val = Number(val); builder.append(val); }
                    else { let val = vec.get(r); if (typeof val === 'bigint') val = Number(val); builder.append(val); }
                }
            }

            tableColumns[name] = builder.finish().toVector();
            rebuilt.push(name);
            any = true;
        }

        const newTable = any ? arrow.makeTable(tableColumns) : this.table;

        if (any && (commit === true || commit !== false)) {
            this.replaceTable(newTable);
            for (let i = 0; i < rebuilt.length; i++) {
                const nm = rebuilt[i];
                const ov = this._overlays.get(nm);
                if (!ov) continue;
                if (ov.kind === 'number') {
                    ov.mask.fill(0);
                    ov.size = 0;
                } else {
                    ov.map.clear();
                    ov.size = 0;
                }
            }
            for (let i = 0; i < rebuilt.length; i++) {
                this._bumpColEpochByName(rebuilt[i]);
            }
            this._editCount = 0;
            return this.table;
        }

        this._editCount = 0;
        return newTable;
    }

    replaceTable(newTable){
        if (!newTable || !newTable.schema) throw new Error('replaceTable: invalid table');

        const prevTable = this.table;
        const prevNumRows = prevTable ? (prevTable.numRows | 0) : 0;
        const prevNumCols = prevTable ? (prevTable.numCols | 0) : 0;

        const nextNumRows = newTable.numRows | 0;
        const nextNumCols = newTable.numCols | 0;

        // Ensure required members always exist (even if older instances slip through).
        if (!this._sortedRowCache) this._sortedRowCache = new Map();
        if (this._tableReplaceSeq == null) this._tableReplaceSeq = 0;
        if (this._getterCache) this._getterCache.clear();

        this._emitTableWillReplace(prevTable, newTable);

        // [MEMORY] Release old references explicitly
        this.table = null;
        this._nameToVector.clear();
        this._nameToIndex.clear();
        this._fieldNamesCache = null;
        this._fieldSetCache = null;
        this._missingColWarned = null;

        this.table = newTable;
        this._fields = newTable.schema.fields;

        for (let i=0;i<this._fields.length;i++){
            const n = this._fields[i].name;
            this._nameToVector.set(n, newTable.getChildAt(i));
            this._nameToIndex.set(n, i);
        }
        this._fieldNamesCache = null;
        this._fieldSetCache = null;

        this.timeUnitScale = Object.create(null);
        this._decoder = Object.create(null);

        const n = nextNumRows;
        const didShrink = nextNumRows < prevNumRows;
        const schemaChanged = nextNumCols !== prevNumCols;

        // Compact overlays to new size + drop overlays for columns that no longer exist anywhere.
        for (const [name, ov] of this._overlays.entries()){
            if (!ov) { this._overlays.delete(name); continue; }

            const keep =
                this._nameToVector.has(name) ||
                this._derived.has(name) ||
                this.columnDefs.has(name);

            if (!keep) {
                // Free memory aggressively for dead columns.
                if (ov.kind === 'number') {
                    ov.mask = null;
                    ov.values = null;
                    ov.size = 0;
                } else {
                    try { ov.map?.clear?.(); } catch {}
                    ov.size = 0;
                }
                this._overlays.delete(name);
                continue;
            }

            if (ov.kind === 'number'){
                const needsResize =
                    !(ov.mask instanceof Uint8Array) ||
                    !(ov.values instanceof Float64Array) ||
                    (ov.mask.length !== n) ||
                    (ov.values.length !== n);

                if (needsResize){
                    const prevMask = (ov.mask instanceof Uint8Array) ? ov.mask : null;
                    const prevVals = (ov.values instanceof Float64Array) ? ov.values : null;

                    const m2 = new Uint8Array(n);
                    const v2 = new Float64Array(n);

                    if (prevMask && prevVals){
                        const copyLen = Math.min(prevMask.length | 0, n);
                        if (copyLen > 0) {
                            m2.set(prevMask.subarray(0, copyLen));
                            v2.set(prevVals.subarray(0, copyLen));
                        }

                        // PERF: only scan copyLen; the rest are known zeros.
                        let s = 0;
                        for (let i = 0; i < copyLen; i++) if (m2[i] !== MASK_NONE) s++;
                        ov.size = s;
                    } else {
                        ov.size = 0;
                    }

                    ov.mask = null;
                    ov.values = null;
                    ov.mask = m2;
                    ov.values = v2;
                }
            } else {
                if (!ov.map || typeof ov.map.set !== 'function') ov.map = new Map(ov.map);
                const mp = ov.map;

                if (didShrink && mp && mp.size) {
                    // Only delete out-of-range keys when the table shrank.
                    let s = 0;
                    for (const k of mp.keys()){
                        const ri = (k | 0);
                        if (ri < 0 || ri >= n) mp.delete(k);
                        else s++;
                    }
                    ov.size = s;
                } else {
                    ov.size = mp ? mp.size : 0;
                }
            }
        }

        // Trim frozen entries only when it matters (shrink and/or schema changes).
        if (this._frozenOverrides && this._frozenOverrides.size && (didShrink || schemaChanged)) {
            this._freeAllFrozen({
                trimToRowCount: nextNumRows,
                trimUnknownColumns: schemaChanged,
                clearOverlays: false,
                bump: false,
                clearAll: false
            });
        }

        this._rowIdToIndex = null;

        this._dictWindowCache.clear();
        this._dictEpochByVec = new WeakMap();
        this._dictEpochCounter = 1;

        this._idxMinMax.clear();
        this._pool.clear();
        this._classCache.clear();
        this._fmtCache.clear();

        // Clear any sort cache keyed to the old table/data.
        if (this._sortedRowCache && typeof this._sortedRowCache.clear === 'function') this._sortedRowCache.clear();

        this._tableReplaceSeq = (this._tableReplaceSeq | 0) + 1;
        this._emitTableDidReplace(prevTable, newTable);
    }

    resetOverlays(){
        for (const [, ov] of this._overlays.entries()) {
            if (!ov) continue;
            if (ov.kind === 'number') {
                ov.mask = new Uint8Array(0);
                ov.values = new Float64Array(0);
                ov.size = 0;
            } else {
                ov.map.clear();
                ov.size = 0;
            }
        }
    }

    removeAllRows({resetOverlays = true, resetFrozen = true, rebuildIndexes = true, emitGlobalEpoch = true} = {}) {
        const newTable = buildEmptyTableFromSchema(this.table.schema);
        this.replaceTable(newTable);

        if (resetOverlays) {
            this.resetOverlays()
        }
        if (resetFrozen && this._frozenOverrides) this._frozenOverrides.clear();
        if (rebuildIndexes && this._useMinMax) this._idxMinMax.clear();
        if (emitGlobalEpoch) this._emitEpochChange({ global: true });
        return this;
    }

    replaceAllRows(rowsOrTable, {preserveSchema=true, resetOverlays=false, rebuildIndexes=true, emitGlobalEpoch=false, useMasterSchema=true} = {}) {
        const before = this.numRows() | 0;

        const preserveById =
            (typeof(resetOverlays) === 'string') && (
                resetOverlays === 'byId' ||
                resetOverlays === 'preserveById' ||
                resetOverlays === 'remapById'
            )

        // Snapshot overlays keyed by rowId (not rowIndex) so they can be restored after row reorder/filtering.
        let overlaysById = null;   // Map<colName, Map<rowId, value>>
        let frozenById = null;     // Map<colName, Set<rowId>>

        if (preserveById && before > 0) {
            // ---- overlays snapshot ----
            if (this._overlays && this._overlays.size) {
                const snap = new Map();
                for (const [name, ov] of this._overlays.entries()) {
                    if (!ov || (ov.size | 0) <= 0) continue;

                    const perCol = new Map();

                    if (ov.kind === 'number') {
                        const mask = ov.mask;
                        const vals = ov.values;
                        if (mask && mask.length) {
                            let remaining = ov.size | 0;
                            const n = mask.length | 0;
                            for (let ri = 0; ri < n && remaining > 0; ri++) {
                                const m = mask[ri] | 0;
                                if (m === MASK_NONE) continue;
                                remaining--;

                                const rowId = String(this.getRowIdByIndex(ri));
                                if (m === MASK_NULL) {
                                    perCol.set(rowId, null);
                                } else {
                                    const v = vals ? vals[ri] : NaN;
                                    perCol.set(rowId, Number.isFinite(v) ? v : null);
                                }
                            }
                        }
                    } else {
                        const mp = ov.map;
                        if (mp && mp.size) {
                            for (const [ri, v] of mp.entries()) {
                                const rowId = String(this.getRowIdByIndex(ri | 0));
                                perCol.set(rowId, v);
                            }
                        }
                    }

                    if (perCol.size) snap.set(name, perCol);
                }
                if (snap.size) overlaysById = snap;
            }

            // ---- frozen snapshot ----
            if (this._frozenOverrides && this._frozenOverrides.size) {
                const snap = new Map();
                for (const [col, rowSet] of this._frozenOverrides) {
                    if (!rowSet || rowSet.size === 0) continue;
                    let idSet = snap.get(col);
                    if (!idSet) { idSet = new Set(); snap.set(col, idSet); }
                    for (const ri of rowSet) {
                        if (ri < 0 || ri >= before) continue;
                        idSet.add(String(this.getRowIdByIndex(ri)));
                    }
                    if (idSet.size === 0) snap.delete(col);
                }
                if (snap.size) frozenById = snap;
            }
        }

        // Build next table as before
        let nextTable;
        if (isArrowTable(rowsOrTable)) {
            if (preserveSchema) {
                const targetSchema = useMasterSchema ? this.master_schema : this.table.schema;
                nextTable = projectTableToSchema(rowsOrTable, targetSchema);
            } else {
                nextTable = rowsOrTable;
            }
        } else {
            nextTable = this._rowsToTable(rowsOrTable, this.table.schema);
        }

        // Replace base table (clears caches, compacts overlays arrays/maps to new size)
        this.replaceTable(nextTable);

        if (preserveById) {
            // Clear all overlays by index (they no longer correspond after replace)
            const n = this.numRows() | 0;
            for (const [, ov] of this._overlays.entries()) {
                if (!ov) continue;
                if (ov.kind === 'number') {
                    ov.mask = new Uint8Array(n);
                    ov.values = new Float64Array(n);
                    ov.size = 0;
                } else {
                    if (ov.map && typeof ov.map.clear === 'function') ov.map.clear();
                    else ov.map = new Map();
                    ov.size = 0;
                }
            }
            if (this._frozenOverrides) this._frozenOverrides.clear();

            // Restore overlays by rowId
            if (overlaysById && overlaysById.size) {
                for (const [col, mp] of overlaysById.entries()) {
                    if (!mp || mp.size === 0) continue;
                    const ov = this._ensureOverlay(col);
                    for (const [rowId, value] of mp.entries()) {
                        const ri = this.getRowIndexById(rowId);
                        if (ri < 0) continue;
                        this._overlaySet(ov, ri | 0, value);
                    }
                }
            }

            // Restore frozen flags by rowId
            if (frozenById && frozenById.size && this._frozenOverrides) {
                for (const [col, idSet] of frozenById.entries()) {
                    if (!idSet || idSet.size === 0) continue;
                    let rowSet = this._frozenOverrides.get(col);
                    if (!rowSet) { rowSet = new Set(); this._frozenOverrides.set(col, rowSet); }
                    for (const rowId of idSet.values()) {
                        const ri = this.getRowIndexById(rowId);
                        if (ri < 0) continue;
                        rowSet.add(ri | 0);
                    }
                    if (rowSet.size === 0) this._frozenOverrides.delete(col);
                }
            }
        } else {
            const shouldResetOverlays =
                resetOverlays === true ||
                (resetOverlays === 'auto' && (before === 0 || !isArrowTable(rowsOrTable)));

            if (shouldResetOverlays) {
                const n = this.numRows() | 0;
                for (const [, ov] of this._overlays.entries()) {
                    if (ov.kind === 'number') {
                        ov.mask = new Uint8Array(n);
                        ov.values = new Float64Array(n);
                        ov.size = 0;
                    } else {
                        ov.map.clear();
                        ov.size = 0;
                    }
                }
                if (this._frozenOverrides) this._frozenOverrides.clear();
            }
        }

        this._rowIdToIndex = null;
        if (this._useMinMax && rebuildIndexes) this.refreshIndexes();

        // intentionally keep your original behavior (no global epoch emit here)
        return { previousRowCount: before, nextRowCount: this.numRows() | 0 };
    }


    resetWithRows(rows, {rebuildIndexes=true,emitGlobalEpoch=true} = {}) {
        const tbl = this._rowsToTable(rows, this.table.schema);
        this.replaceTable(tbl);

        for (const [, ov] of this._overlays.entries()) {
            if (ov.kind === 'number') {
                ov.mask = new Uint8Array(this.numRows() | 0);
                ov.values = new Float64Array(this.numRows() | 0);
                ov.size = 0;
            } else {
                ov.map.clear();
                ov.size = 0;
            }
        }
        if (this._frozenOverrides) this._frozenOverrides.clear();
        this._rowIdToIndex = null;
        if (this._useMinMax && rebuildIndexes) this.refreshIndexes();
        if (emitGlobalEpoch) this._emitEpochChange({ global: true });
        return this;
    }

    rebuildAllIndexes(columns = null) {
        if (!this._useMinMax) return this;
        this.refreshIndexes(columns);
        return this;
    }

    getAllRowIds() {
        const n = this.numRows() | 0;
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = String(this.getRowIdByIndex(i));
        return out;
    }

    /* ---------------------------- min/max pushdown ---------------------------- */
    enableIndexes({ numericMinMax=true } = {}){ this._useMinMax = !!numericMinMax; return this; }

    setActiveColumnsFromProjection(projection) {
        const names = this._normalizeColumnSelector(projection);

        if (!names.length) {
            this._activeColumns = null;
            if (this._useMinMax) this.refreshIndexes(null);
            return this;
        }

        const active = new Set();

        for (let i = 0; i < names.length; i++) {
            const col = names[i];
            if (!col) continue;
            active.add(col);

            const deps = this.getDependenciesClosure(col);
            if (Array.isArray(deps)) {
                for (let j = 0; j < deps.length; j++) {
                    active.add(deps[j]);
                }
            }
        }

        this._activeColumns = active;

        if (this._useMinMax) {
            this.refreshIndexes(Array.from(active));
        }

        return this;
    }

    getActiveColumns() {
        if (this._activeColumns && this._activeColumns.size) {
            return Array.from(this._activeColumns);
        }
        // Fallback: all known fields
        return this._normalizeColumnSelector(null);
    }

    isColumnActive(column) {
        const names = this._normalizeColumnSelector(column);
        if (!names.length) return false;
        if (!this._activeColumns || !this._activeColumns.size) return true;
        return this._activeColumns.has(names[0]);
    }

    refreshIndexes(columns=null){
        if (!this._useMinMax) return this;
        if (!columns) return this;

        const names = this._normalizeColumnSelector(columns);
        const n = this.numRows()|0;
        const bs = this._indexBlockSize|0;
        const nb = ((n + bs - 1) / bs) | 0;

        for (let i=0;i<names.length;i++){
            const name = names[i];
            if (!this._isDerived(name)) {
                const vec = this._getVector(name);
                if (!vec || !this._isNumericArrowType(vec)) continue;
                const mins = new Float64Array(nb);
                mins.fill(Number.POSITIVE_INFINITY);
                const maxs = new Float64Array(nb);
                maxs.fill(Number.NEGATIVE_INFINITY);
                for (let b = 0; b < nb; b++) {
                    const s = b * bs, e = Math.min(s + bs, n);
                    for (let r = s; r < e; r++) {
                        const v = this.getCell(r, name);
                        if (v == null || !Number.isFinite(v)) continue;
                        if (v < mins[b]) mins[b] = v;
                        if (v > maxs[b]) maxs[b] = v;
                    }
                }
                this._idxMinMax.set(name, {blockSize: bs, mins, maxs});
            }
        }
        return this;
    }

    _queryBlocksByRange(name, op, a, b){
        const idx = this._idxMinMax.get(name);
        if (!idx) return null;
        const { blockSize, mins, maxs } = idx;
        const picks = [];
        for (let bi=0; bi<mins.length; bi++){
            const lo = mins[bi], hi = maxs[bi];
            if (lo === Infinity && hi === -Infinity) continue;
            let hit=false;
            switch(op){
                case '<': hit = lo < a; break;
                case '<=': hit = lo <= a; break;
                case '>': hit = hi > a; break;
                case '>=': hit = hi >= a; break;
                case '==': hit = (a >= lo && a <= hi); break;
                case 'between': hit = !(b < lo || a > hi); break;
                default: hit = true;
            }
            if (hit) picks.push(bi);
        }
        return { picks, blockSize };
    }

    queryRange({ column, op, value, lo, hi }){
        const name = this._normalizeColumnSelector(column)[0];
        const n = this.numRows()|0;
        const mm = this._queryBlocksByRange(name, op, value ?? lo, hi);
        const out = new Int32Array(n); let count=0;

        const test = (v)=>{
            if (v == null || !Number.isFinite(v)) return false;
            switch(op){
                case '<': return v < value;
                case '<=': return v <= value;
                case '>': return v > value;
                case '>=': return v >= value;
                case '==': return v === value;
                case 'between': return v >= lo && v <= hi;
                default: return false;
            }
        };

        if (mm){
            const { picks, blockSize } = mm;
            for (let k=0;k<picks.length;k++){
                const bi = picks[k], s = bi*blockSize, e = Math.min(s+blockSize, n);
                for (let r=s;r<e;r++){ if (test(this.getCell(r, name))) out[count++] = r; }
            }
        } else {
            for (let r=0;r<n;r++){ if (test(this.getCell(r, name))) out[count++] = r; }
        }
        return out.subarray(0, count);
    }

    /* -------------------------------- sorting -------------------------------- */
    sortIndex(indexArray, comparator){
        const out = new Int32Array(indexArray);
        out.sort(comparator);
        return out;
    }

    buildComparatorByColumns(sortModel, opts = {}) {
        const specs = this._extractSortSpecs(sortModel); // Uses helper from
        if (specs.length === 0) {
            return () => 0;
        }

        const comparators = specs.map(({ colId, dir }) => {
            const get = this._getValueGetter(colId);
            const colDef = this.columnDefs.get(colId);
            const ctx = colDef?.context;
            let mode = 'string'; // default

            // 1. Check context.dataType
            const dt = ctx?.dataType;
            if (dt === 'number' || dt === 'integer' || dt === 'float' || dt === 'currency' || dt === 'percentage') {
                mode = 'number';
            } else if (dt === 'date' || dt === 'datetime') {
                mode = 'date';
            } else if (dt === 'boolean') {
                mode = 'boolean';
            } else if (!dt) {
                // 2. Check vector type if no dataType
                try {
                    if (!this._isDerived(colId)) {
                        const vec = this._getVector(colId);
                        if (this._isNumericArrowType(vec)) {
                            mode = 'number';
                        } else if (this._detectTimeUnitScale(vec) !== 1) {
                            mode = 'date'; // Timestamps are numeric-like dates
                        }
                    }
                } catch {}
            }

            // 3. Custom comparator from colDef (e.g., benchmarkComparator )
            if (colDef?.comparator) {
                return (aIdx, bIdx) => {
                    const av = get(aIdx);
                    const bv = get(bIdx);
                    // Handle nulls consistently: nulls are always "greater than" (last)
                    const aNullish = (av == null);
                    const bNullish = (bv == null);
                    if (aNullish && bNullish) return 0;
                    if (aNullish) return 1;
                    if (bNullish) return -1;
                    return colDef.comparator(av, bv) * dir;
                };
            }

            // Return a specialized comparator function
            if (mode === 'number') {
                return (aIdx, bIdx) => {
                    const av = get(aIdx);
                    const bv = get(bIdx);
                    const aNullish = (av == null) || (typeof av === 'number' && !Number.isFinite(av));
                    const bNullish = (bv == null) || (typeof bv === 'number' && !Number.isFinite(bv));
                    if (aNullish && bNullish) return 0;
                    if (aNullish) return 1;
                    if (bNullish) return -1;
                    const an = (typeof av === 'number') ? av : +av;
                    const bn = (typeof bv === 'number') ? bv : +bv;
                    if (an < bn) return -dir;
                    if (an > bn) return dir;
                    return 0;
                };
            }

            if (mode === 'date') {
                return (aIdx, bIdx) => {
                    const av = get(aIdx);
                    const bv = get(bIdx);
                    const aNullish = (av == null);
                    const bNullish = (bv == null);
                    if (aNullish && bNullish) return 0;
                    if (aNullish) return 1;
                    if (bNullish) return -1;
                    const at = (av instanceof Date) ? av.getTime() : +av;
                    const bt = (bv instanceof Date) ? bv.getTime() : +bv;
                    if (at < bt) return -dir;
                    if (at > bt) return dir;
                    return 0;
                };
            }

            if (mode === 'boolean') {
                return (aIdx, bIdx) => {
                    const av = get(aIdx);
                    const bv = get(bIdx);
                    const aNullish = (av == null);
                    const bNullish = (bv == null);
                    if (aNullish && bNullish) return 0;
                    if (aNullish) return 1;
                    if (bNullish) return -1;
                    const ab = av ? 1 : 0;
                    const bb = bv ? 1 : 0;
                    if (ab < bb) return -dir;
                    if (ab > bb) return dir;
                    return 0;
                };
            }

            // Default: string comparator
            return (aIdx, bIdx) => {
                const av = get(aIdx);
                const bv = get(bIdx);
                const aNullish = (av == null);
                const bNullish = (bv == null);
                if (aNullish && bNullish) return 0;
                if (aNullish) return 1;
                if (bNullish) return -1;
                // Locale-aware compare for strings
                const as = String(av);
                const bs = String(bv);
                return as.localeCompare(bs) * dir;
            };
        });

        // Final comparator that iterates through the specialized ones
        return (aIdx, bIdx) => {
            for (let i = 0; i < comparators.length; i++) {
                const result = comparators[i](aIdx, bIdx);
                if (result !== 0) return result;
            }
            return 0;
        };
    }

    _packSortKeys(specs, idx, columnMap) {
        const n = idx.length;
        const packedKeys = new Array(specs.length);

        for (let k = 0; k < specs.length; k++) {
            const { col, dir } = specs[k];
            const get = this._getValueGetter(col);
            const def = columnMap.get(col);
            const dt = def?.context?.dataType;

            let mode = 'string';
            if (dt) {
                if (dt === 'number' || dt === 'integer' || dt === 'float' || dt === 'percentage' || dt === 'currency') mode = 'number';
                else if (dt === 'date' || dt === 'datetime') mode = 'date';
            } else {
                const sampleIndices = [idx[0], idx[Math.floor(n / 2)], idx[n - 1]].filter(i => i !== undefined);
                const samples = sampleIndices.map(ri => get(ri));
                const isNum = v => typeof v === 'number' && Number.isFinite(v);
                const isDateLike = v => v instanceof Date || (isNum(v) && v > 0 && v < 8.64e15);
                if (samples.some(isNum)) mode = 'number';
                else if (samples.some(isDateLike)) mode = 'date';
            }

            const values = (mode === 'string') ? new Array(n) : this._pool.acquire('float64', n);
            const nulls = this._pool.acquire('uint8', n);

            for (let i = 0; i < n; i++) {
                const ri = idx[i];
                let v = get(ri);

                if (mode === 'number') v = coerceToNumber(v, {onNaN: null});
                else if (mode === 'date') v = coerceToDate(v)
                const isNullish = (v == null) || (mode === 'number' && !Number.isFinite(v));

                if (isNullish) {
                    nulls[i] = 1;
                    if (mode !== 'string') values[i] = NaN;
                    continue;
                }

                if (mode === 'date') values[i] = v
                else if (mode === 'number') values[i] = v;
                else values[i] = String(v);
            }
            packedKeys[k] = { dir, mode, values, nulls, def};
        }
        return packedKeys;
    }

    sortIndexByModel(sortModel, idx, columnMap) {
        const specs = this._extractSortSpecs(sortModel);
        if (!specs.length || !idx || !idx.length) return idx;

        const packedKeys = this._packSortKeys(specs, idx, columnMap);
        try {
            const n = idx.length;
            const positions = new Uint32Array(n);
            for (let i = 0; i < n; i++) positions[i] = i;

            positions.sort((posA, posB) => {
                for (let k = 0; k < packedKeys.length; k++) {
                    const { dir, values, nulls, def } = packedKeys[k];
                    const aIsNull = nulls[posA];
                    const bIsNull = nulls[posB];

                    if (aIsNull && bIsNull) continue;
                    if (aIsNull) return 1;
                    if (bIsNull) return -1;

                    const valA = values[posA];
                    const valB = values[posB];
                    if (def?.comparator) return def.comparator(valA, valB) * dir;

                    if (valA < valB) return -dir;
                    if (valA > valB) return dir;
                }
                return 0;
            });

            const sortedIdx = new Int32Array(n);
            for (let i = 0; i < n; i++) {
                sortedIdx[i] = idx[positions[i]];
            }
            return sortedIdx;
        } finally {
            for (const keys of packedKeys) {
                if (keys.mode !== 'string') {
                    this._pool.release('float64', keys.values);
                }
                this._pool.release('uint8', keys.nulls);
            }
        }
    }

    _extractSortSpecs(sortModel){
        const model = Array.isArray(sortModel) ? sortModel : [];
        const norm = (m)=>{
            if (!m) return null;
            const col = (m.colId ?? m.field ?? m.column ?? m.col ?? null);
            const sort = (m.sort ?? m.dir ?? m.order ?? 'asc');
            if (!col) return null;
            const dir = String(sort).toLowerCase()==='desc' || String(sort)==='-1' ? -1 : 1;
            return { col, dir };
        };
        const out=[]; for (let i=0;i<model.length;i++){ const n = norm(model[i]); if (n) out.push(n); }
        return out;
    }

    /* --------------------------- row window helpers -------------------------- */
    getAllRows({ format = 'objects', columns = null, useActiveColumns=true} = {}) {
        const s = 0;
        const e = this.numRows();
        const [rs, re] = [Math.max(0, s), Math.max(Math.min(this.numRows(), e), s)];
        if (re <= rs) return [];

        let cols = columns;
        if (cols == null && useActiveColumns && this._activeColumns && this._activeColumns.size > 0) {
            cols = Array.from(this._activeColumns);
        }

        return format === 'array'
            ? this.getRowWindowArrays(rs, re, cols)
            : this.getRowWindowObjects(rs, re, cols);
    }

    getRows(startRow, endRow, {format='objects', columns=null}={}){
        const s = startRow|0; const e = endRow==null ? (s+1) : (endRow|0);
        const [rs, re] = [ Math.max(0, s), Math.max(Math.min(this.numRows(), e), s) ];
        if (re<=rs) return [];
        return format==='array' ? this.getRowWindowArrays(rs, re, columns) : this.getRowWindowObjects(rs, re, columns);
    }

    getRowWindowObjects(startRow, endRow, columns=null, includeId=true){
        const names = this._normalizeColumnSelector(columns);
        const rows = new Array(endRow - startRow);
        for (let i=0;i<rows.length;i++){
            const ri = startRow + i;
            const obj = includeId ? { [this._idProperty]: this.getRowIdByIndex(ri) } : {};
            for (let c=0;c<names.length;c++) obj[names[c]] = this.getCell(ri, names[c]);
            rows[i] = obj;
        }
        return rows;
    }

    getDisjointRowObjectsFast(row_idxs, columns = null, { includeId = true, useFormats = false, gridApi = null, useHeaders = false } = {}) {
        const names = this._normalizeColumnSelector(columns);
        const rows = new Array(row_idxs.length);

        // Hoist getter creation outside the row loop — one closure per column, not per cell.
        const getters = new Array(names.length);
        const keys = new Array(names.length);

        for (let c = 0; c < names.length; c++) {
            getters[c] = this._getValueGetter(names[c]);
            keys[c] = names[c];

            if ((useFormats || useHeaders) && gridApi) {
                const cd = gridApi.getColumnDef(names[c]);
                if (useHeaders && cd?.headerName) {
                    keys[c] = cd.headerName;
                }
                if (useFormats && cd) {
                    const vf = cd.valueFormatter;
                    if (vf) {
                        const baseGetter = getters[c];
                        let fmt = cd?.context?.formatting || {};
                        if (typeof fmt === 'function') fmt = fmt({}, this);
                        if (!fmt?.sigFigs) {
                            const sf = cd?.context?.market?.sigFigs;
                            fmt['sigFigs'] = (typeof sf === 'function') ? sf({}, this) : (sf ?? 3);
                        }
                        const capturedFmt = fmt;
                        getters[c] = (ri) => {
                            const v = baseGetter(ri);
                            return vf({ value: v }, this, capturedFmt);
                        };
                    }
                }
            }
        }

        for (let i = 0; i < row_idxs.length; i++) {
            const ri = row_idxs[i];
            const obj = includeId ? { [this._idProperty]: this.getRowIdByIndex(ri) } : {};
            for (let c = 0; c < names.length; c++) {
                obj[keys[c]] = getters[c](ri);
            }
            rows[i] = obj;
        }
        return rows;
    }

    getDisjointRowObjects(row_idxs, columns=null, {includeId=true, useFormats=false, gridApi=null, useHeaders=false}={}){
        const names = this._normalizeColumnSelector(columns);
        const rows = new Array(row_idxs.length);
        for (let i=0;i<rows.length;i++){
            const ri = row_idxs[i];
            const obj = includeId ? { [this._idProperty]: this.getRowIdByIndex(ri) } : {};
            for (let c=0;c<names.length;c++) {
                let v = this.getCell(ri, names[c]);
                let k = names[c];
                if ((useFormats || useHeaders) && gridApi) {
                    const params = {value: v}
                    const cd = gridApi.getColumnDef(names[c]);

                    if (useFormats) {
                        const vf = cd?.valueFormatter;
                        if (vf) {
                            let fmt = cd?.context?.formatting || {};
                            if (typeof fmt === 'function') {fmt=fmt(params, this)}
                            if (!fmt?.sigFigs) {
                                const sf = cd?.context?.market?.sigFigs;
                                let sfv = sf ?? 3;
                                if (typeof sf === 'function') {
                                    sfv = sf(params, this);
                                }
                                fmt['sigFigs'] = sfv
                            }
                            v = vf(params, this, fmt);
                        }
                    }

                    if (useHeaders) {
                        k = cd?.headerName ?? names[c];
                    }

                } else if (useFormats || useHeaders) {
                    console.error('Copying with formats requires gridApi as well')
                }
                obj[k] = v
            }
            rows[i] = obj;
        }
        return rows;
    }

    getDisjointRowArrays(row_idxs, columns=null){
        const names = this._normalizeColumnSelector(columns);
        const rows = new Array(row_idxs.length);
        for (let i=0;i<rows.length;i++){
            const ri = row_idxs[i], arr = new Array(names.length);
            for (let c=0;c<names.length;c++) arr[c] = this.getCell(ri, names[c]);
            rows[i] = arr;
        }
        return rows;
    }

    getRowWindowArrays(startRow, endRow, columns=null){
        const names = this._normalizeColumnSelector(columns);
        const rows = new Array(endRow - startRow);
        for (let i=0;i<rows.length;i++){
            const ri = startRow + i, arr = new Array(names.length);
            for (let c=0;c<names.length;c++) arr[c] = this.getCell(ri, names[c]);
            rows[i] = arr;
        }
        return rows;
    }

    getColumns(columnOrColumns) {
        return this.getRowWindowObjects(0, this.numRows(), columnOrColumns)
    }
    getColumnValues(column){
        const columns = asArray(column)
        if (columns.length === 1) return this._getColumnValues(columns[0]);
        const result = {}
        for (const col of columns) {
            result[col] = this._getColumnValues(col);
        }
        return result
    }
    _getColumnValues(name){
        const n = this.numRows();
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = this.getCell(i, name);
        return out;
    }
    getFormattedColumnValues(column){
        const columns = asArray(column)
        if (columns.length === 1) return this._getFormattedColumnValues(columns[0]);
        const result = {}
        for (const col of columns) {
            result[col] = this._getFormattedColumnValues(col);
        }
        return result
    }
    _getFormattedColumnValues(name){
        const n = this.numRows();
        const out = new Array(n);
        const d = this.columnDefs.get(name);
        const fmt = d?.context?.valueFormatter || d?.valueFormatter;
        const f = d?.context?.formatting || {};
        for (let i = 0; i < n; i++) {
            const v = this.getCell(i, name);
            out[i] = fmt ? fmt({value: v}, this, f) : v;
        }
        return out;
    }

    getDistinctColumnValues(columns, asArr=false) {
        const cols = asArray(columns);
        const s = new Set();
        const n = this.numRows();
        if (cols.length === 1) {
            const col = cols[0];
            for (let i = 0; i < n; i++) s.add(this.getCell(i, col));
        } else {
            for (let c = 0; c < cols.length; c++) {
                for (let i = 0; i < n; i++) s.add(this.getCell(i, cols[c]));
            }
        }
        return asArr ? Array.from(s) : s;
    }

    getColumnType(columns) {
        const names = this._normalizeColumnSelector(columns);
        if (names.length === 1) return this.getDtype(names[0]) || null;
        const out = {};
        for (let i = 0; i < names.length; i++) out[names[i]] = this.getDtype(names[i]) || null;
        return out;
    }

    /* ----------------------------- dictionary windows ------------------------ */
    _getDictionaryEpoch(vector){
        if ((!vector) || (typeof vector.getDictionary !== 'function')) return 0;
        const dictVec = vector.getDictionary(); if (!dictVec) return 0;
        let e = this._dictEpochByVec.get(dictVec);
        if (!e){ e = this._dictEpochCounter++; this._dictEpochByVec.set(dictVec, e); }
        return e;
    }

    _bulkDecodeDictionary(name, start, end) {
        const vec = this._getVector(name);
        if ((!vec) || (typeof vec.getDictionary !== 'function')) return null;

        const dictVec = vec.getDictionary();
        const epoch = this._getDictionaryEpoch(vec);
        const key = `${name}|${start}|${end}|${epoch}`;

        if (this._dictWindowCache.has(key)) return this._dictWindowCache.get(key);

        const out = new Array(end - start);
        for (let i = start, j = 0; i < end; i++, j++) {
            const k = vec.get(i);
            out[j] = (k == null) ? null : dictVec.get(k);
        }

        const LIMIT = this._dictCacheMax | 0 || 512;
        if (this._dictWindowCache.size >= LIMIT) {
            // Evict oldest 25% to amortize eviction cost and reduce thrashing.
            const evictCount = Math.max(1, (LIMIT >>> 2));
            const iter = this._dictWindowCache.keys();
            for (let i = 0; i < evictCount; i++) {
                const k = iter.next();
                if (k.done) break;
                this._dictWindowCache.delete(k.value);
            }
        }
        this._dictWindowCache.set(key, out);
        return out;
    }

    /* --------------------------- high-level conveniences --------------------- */
    mapColumn(column, mapFn, startRow=0, endRow=this.numRows()){
        const name = this._normalizeColumnSelector(column)[0];
        const out = new Array(endRow - startRow);
        for (let r=startRow, i=0; r<endRow; r++, i++) out[i] = mapFn(this.getCell(r, name), r);
        return out;
    }

    computeStats(column, startRow = 0, endRow = this.numRows()){
        const name = this._normalizeColumnSelector(column)[0];
        const s = startRow|0, e = endRow|0;
        const n = Math.max(0, e - s);
        if (n <= 0) return { count:0, nulls:0, min:NaN, max:NaN, sum:0, mean:NaN, std:NaN, variance:NaN };

        let count = 0, nulls = 0, min = Infinity, max = -Infinity;
        let mean = 0, M2 = 0, sum = 0;

        for (let r = s; r < e; r++) {
            const v = this.getCell(r, name);
            if (typeof v !== 'number' || !Number.isFinite(v)) { nulls++; continue; }

            sum += v;
            count++;
            if (v < min) min = v;
            if (v > max) max = v;

            const delta = v - mean;
            mean += delta / count;
            M2 += delta * (v - mean);
        }

        if (count === 0) return { count, nulls, min:NaN, max:NaN, sum:0, mean:NaN, std:NaN, variance:NaN };

        const variance = count > 1 ? (M2 / (count - 1)) : 0;
        const std = Math.sqrt(variance);

        return { count, nulls, min, max, sum, mean, std, variance };
    }


    _isBigInt(field) {
        return (
            (field.type instanceof arrow.Int64) ||
            ((field.type instanceof arrow.Int) && (field.type.bitWidth === 64))
        )
    }


    _rowsToTable(rows, my_schema=null) {
        const schema = my_schema || this.table.schema;
        const fields = schema.fields;
        const builders = {};
        fields.forEach((field) => {
            // Downgrade Int64/BigInt types to Float64 so the builder never expects BigInt values
            const isBigIntType = (field.type instanceof arrow.Int64) ||
                ((field.type instanceof arrow.Int) && field.type.bitWidth === 64);
            const builderType = isBigIntType ? new arrow.Float64() : field.type;
            builders[field.name] = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });
        });

        for (const row of rows) {
            if (!row || typeof row !== 'object') continue;
            for (let i = 0; i < fields.length; i++) {
                const field = fields[i];
                const fieldName = field.name;
                const b = builders[fieldName];
                let value = typeof row.get === 'function' ? row.get(fieldName) : row[fieldName];
                if (value == null) { b.append(null); continue; }
                try {
                    if (typeof value === 'bigint') value = Number(value);
                    b.append(value);
                } catch (e) { console.error(`Error appending to builder for ${fieldName}:`, e); b.append(null); }
            }
        }

        const newVectors = {};
        Object.entries(builders).forEach(([fieldName, b]) => {
            newVectors[fieldName] = b.finish().toVector();
        });
        return new arrow.Table(newVectors);
    }

    addRows(rows, updateIfExists = true) {
        const list = Array.isArray(rows) ? rows : (rows ? [rows] : []);
        if (list.length === 0) return { addedIndices: new Int32Array(0), addedIds: [] };

        const idKey = this._idProperty || '__rowId';
        const n0 = this.numRows() | 0;

        const addRows = [];
        const updateRows = [];

        if (updateIfExists) {
            const seenNewIds = new Map(); // id -> index into addRows
            for (let i = 0; i < list.length; i++) {
                const row = list[i];
                if (!row || typeof row !== 'object') continue;
                const rawId = row[idKey];
                if (rawId == null || (typeof rawId === 'number' && Number.isNaN(rawId))) continue;
                const id = String(rawId);
                const existingIdx = this.getRowIndexById(id);
                if (existingIdx >= 0) {
                    updateRows.push({ rowIndex: existingIdx, row });
                } else {
                    // Deduplicate within the batch: last writer wins
                    const prevIdx = seenNewIds.get(id);
                    if (prevIdx !== undefined) {
                        addRows[prevIdx] = row; // overwrite earlier entry
                    } else {
                        seenNewIds.set(id, addRows.length);
                        addRows.push(row);
                    }
                }
            }
        } else {
            for (let i = 0; i < list.length; i++) {
                const row = list[i];
                if (row && typeof row === 'object') addRows.push(row);
            }
        }

        // Apply updates via overlays (consistent with the rest of the realtime engine).
        if (updateRows.length) {
            for (let i = 0; i < updateRows.length; i++) {
                const { rowIndex, row } = updateRows[i];
                const changes = Object.create(null);
                const keys = Object.keys(row);
                for (let k = 0; k < keys.length; k++) {
                    const c = keys[k];
                    if (c === idKey || c === this._idxProperty) continue;
                    changes[c] = _coerceArrowScalar(row[c]);
                }
                this.setOverlayValuesByRowBatch(rowIndex, changes, { freezeDerived: true, bump: true });
            }
        }

        if (!addRows.length) return { addedIndices: new Int32Array(0), addedIds: [] };

        const newTable = _buildArrowTableFromRows(this.table.schema, addRows);
        const merged = _tryConcatArrowTables(this.table, newTable);

        this.replaceTable(merged);
        this._rowIdToIndex = null;

        const nAdd = addRows.length | 0;
        const addedIndices = new Int32Array(nAdd);
        const addedIds = new Array(nAdd);

        for (let i = 0; i < nAdd; i++) {
            const globalIdx = n0 + i;
            addedIndices[i] = globalIdx;
            const supplied = addRows[i] && addRows[i][idKey] != null ? String(addRows[i][idKey]) : null;
            addedIds[i] = supplied || this.getRowIdByIndex(globalIdx);
        }

        return { addedIndices, addedIds };
    }

    /**
     * Remove rows by rowId (or by row payloads containing id).
     * Compacts base vectors and reindexes overlays before swapping the table.
     */

    removeRowsById(idsOrRows) {
        const idKey = this._idProperty || '__rowId';
        const list = Array.isArray(idsOrRows) ? idsOrRows : (idsOrRows ? [idsOrRows] : []);
        if (list.length === 0) return { removedIndices: new Int32Array(0), removedIds: [] };

        const ids = [];
        for (let i = 0; i < list.length; i++) {
            const x = list[i];
            ids.push(typeof x === 'object' ? String(x[idKey]) : String(x));
        }

        const seen = new Set();
        const remIdx = [];
        const remIds = [];
        for (let i = 0; i < ids.length; i++) {
            const id = ids[i];
            if (!id || seen.has(id)) continue;
            seen.add(id);
            const idx = this.getRowIndexById(id);
            if (idx >= 0) { remIdx.push(idx); remIds.push(id); }
        }
        if (remIdx.length === 0) return { removedIndices: new Int32Array(0), removedIds: [] };

        remIdx.sort((a, b) => a - b);
        const n0 = this.numRows();
        const n1 = n0 - remIdx.length;
        const removedIndices = Int32Array.from(remIdx);
        const removedSet = new Set(remIdx);

        const cols = {};
        const names = this._fieldNames;
        for (let i = 0; i < names.length; i++) {
            const name = names[i];
            const vec = this._getVector(name);
            const dec = this._getDecoder(name);
            // Use a typed Arrow builder so makeTable doesn't fail on
            // all-null columns where the type can't be inferred from values.
            const isBigIntType = (vec.type instanceof arrow.Int64) ||
                ((vec.type instanceof arrow.Int) && vec.type.bitWidth === 64);
            const builderType = isBigIntType ? new arrow.Float64() : vec.type;
            const builder = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });
            for (let src = 0; src < n0; src++) {
                if (removedSet.has(src)) continue;
                let v;
                if (dec) v = dec(src);
                else {
                    v = vec.get(src);
                    if (typeof v === 'bigint') v = Number(v);
                }
                builder.append(v);
            }
            cols[name] = builder.finish().toVector();
        }

        for (const [col, ov] of this._overlays.entries()) {
            if (!ov) continue;
            if (ov.kind === 'number') {
                const newMask = new Uint8Array(n1);
                const newVals = new Float64Array(n1);
                let size = 0;
                let dst = 0;
                for (let src = 0; src < n0; src++) {
                    if (removedSet.has(src)) continue;
                    const m = ov.mask[src] | 0;
                    newMask[dst] = m;
                    if (m === MASK_VALUE) {
                        newVals[dst] = ov.values[src];
                        size++;
                    } else if (m === MASK_NULL) {
                        size++;
                    }
                    dst++;
                }
                ov.mask = newMask; ov.values = newVals; ov.size = size;
            } else {
                const mp = ov.map || new Map();
                // Collect surviving entries and sort by key to enable correct shift computation.
                const entries = [];
                for (const [ri, val] of mp.entries()) {
                    if (!removedSet.has(ri)) entries.push([ri, val]);
                }
                entries.sort((a, b) => a[0] - b[0]);

                const next = new Map();
                let size = 0;
                let rPtr = 0;
                const rList = remIdx;
                for (let k = 0; k < entries.length; k++) {
                    const ri = entries[k][0];
                    while (rPtr < rList.length && rList[rPtr] <= ri) rPtr++;
                    const dst = ri - rPtr;
                    next.set(dst, entries[k][1]);
                    size++;
                }
                ov.map = next; ov.size = size;
            }
        }

        const newTable = arrow.makeTable(cols);
        this.replaceTable(newTable);
        this._rowIdToIndex = null;

        this._emitEpochChange({ rowsChanged: true, colsChanged: true });
        return { removedIndices, removedIds: remIds };
    }

    removeRowsByIndices(rowIndices) {
        const list = (rowIndices instanceof Int32Array) ? rowIndices : Int32Array.from(rowIndices || []);
        const n0 = this.numRows() | 0;
        if (!list.length || n0 === 0) return { removedIndices: new Int32Array(0), removedIds: [] };

        const remIdx = Array.from(list).filter(i => i >= 0 && i < n0).sort((a, b) => a - b);
        if (remIdx.length === 0) return { removedIndices: new Int32Array(0), removedIds: [] };

        const removedIndices = Int32Array.from(remIdx);
        const removedSet = new Set(remIdx);

        const cols = {};
        const names = this._fieldNames;
        for (let i = 0; i < names.length; i++) {
            const name = names[i];
            const vec = this._getVector(name);
            // Use a typed Arrow builder so makeTable doesn't fail on
            // all-null columns where the type can't be inferred from values.
            const isBigIntType = (vec.type instanceof arrow.Int64) ||
                ((vec.type instanceof arrow.Int) && vec.type.bitWidth === 64);
            const builderType = isBigIntType ? new arrow.Float64() : vec.type;
            const builder = arrow.makeBuilder({ type: builderType, nullValues: [null, undefined] });
            for (let src = 0; src < n0; src++) {
                if (removedSet.has(src)) continue;
                let v = vec.get(src);
                if (typeof v === 'bigint') v = Number(v);
                builder.append(v);
            }
            cols[name] = builder.finish().toVector();
        }

        // Reindex overlays
        for (const [col, ov] of this._overlays.entries()) {
            if (!ov) continue;
            if (ov.kind === 'number') {
                const newMask = new Uint8Array(n0 - remIdx.length);
                const newVals = new Float64Array(n0 - remIdx.length);
                let size = 0;
                let dst = 0;
                for (let src = 0; src < n0; src++) {
                    if (removedSet.has(src)) continue;
                    const m = ov.mask[src] | 0;
                    newMask[dst] = m;
                    if (m === 2) { newVals[dst] = ov.values[src]; size++; }
                    else if (m === 1) { size++; }
                    dst++;
                }
                ov.mask = newMask; ov.values = newVals; ov.size = size;
            } else {
                const mp = ov.map || new Map();
                // Collect surviving entries and sort by key to enable correct shift computation.
                const entries = [];
                for (const [ri, val] of mp.entries()) {
                    if (!removedSet.has(ri)) entries.push([ri, val]);
                }
                entries.sort((a, b) => a[0] - b[0]);

                const next = new Map();
                let size = 0;
                let rPtr = 0;
                const rList = remIdx;
                for (let k = 0; k < entries.length; k++) {
                    const ri = entries[k][0];
                    while (rPtr < rList.length && rList[rPtr] <= ri) rPtr++;
                    const dst = ri - rPtr;
                    next.set(dst, entries[k][1]); size++;
                }
                ov.map = next; ov.size = size;
            }
        }

        // Gather ids BEFORE the table swap (post-swap the indices are invalid)
        const removedIds = new Array(removedIndices.length);
        for (let i = 0; i < removedIndices.length; i++) removedIds[i] = this.getRowIdByIndex(removedIndices[i]);

        this.replaceTable(arrow.makeTable(cols));
        this._rowIdToIndex = null;
        this._emitEpochChange({ rowsChanged: true, colsChanged: true });

        return { removedIndices, removedIds };
    }

    dispose() {
        if (this._disposed) return;
        this._disposed = true;

        // Cancel any pending materialize timer
        if (this._materializeTimer) { clearTimeout(this._materializeTimer); this._materializeTimer = null; }
        if (this._materializeResolvers) { this._materializeResolvers.length = 0; }

        this._editCoalescer.dispose();

        // Release heavy Arrow tables
        this.table = null;
        this._nameToVector.clear();
        this._nameToIndex.clear();

        // Clear all caches
        this._classCache.clear();
        this._fmtCache.clear();
        this._dictWindowCache.clear();
        this._idxMinMax.clear();
        this._overlays.forEach(ov => {
            if (ov.mask) ov.mask = null;
            if (ov.values) ov.values = null;
            if (ov.map) { ov.map.clear(); ov.map = null; }
        });
        this._overlays.clear();

        // Clear frozen overrides (unbounded string Set)
        if (this._frozenOverrides) { this._frozenOverrides.clear(); }

        // Clear Derived subs
        this._derivedSettingsSubs.forEach(subs => {
            for (const s of subs) {
                if (typeof s === 'function') s();
                else if (s?.unsubscribe) s.unsubscribe();
            }
        });
        this._derivedSettingsSubs.clear();

        // Clear derived column state
        this._derived.clear();
        this._columnDependencies.clear();
        this._columnDependents.clear();
        this._valueSetters.clear();
        this._settingsGetter.clear();
        if (this._derivedErrorOnce) { this._derivedErrorOnce.clear(); this._derivedErrorOnce = null; }

        // Clear sort and dependency caches
        if (this._sortedRowCache) { this._sortedRowCache.clear(); this._sortedRowCache = null; }
        this._depClosureCache.clear();
        this._dependentsClosureCache.clear();
        this._fieldNamesCache = null;
        this._fieldSetCache = null;
        this._dtypes = null;
        this._rowIdToIndex = null;
        if (this._derivedNameToIndex) { this._derivedNameToIndex.clear(); }

        // Release Pool memory
        this._pool.clear();

        this.opts = null;
        this.master_schema = null;

        try {this._epochQueue.length = 0} catch(_){}
    }

}

/* ---------------------------------- Adapter ---------------------------------- */

export class ArrowAgGridAdapter {
    static focusedGrid = null;
    static LOAD_DELAY = 2000;

    constructor(context, engine, columnDefs, opts = {}, {name='portfolio', globalViews=[], enableFilterMemory=true, enablePasteOverride=false}={}) {
        if (!engine) throw new Error('ArrowAgGridAdapter: engine is required.');
        if (!Array.isArray(columnDefs)) throw new Error('ArrowAgGridAdapter: columnDefs array is required.');

        this.context = context;
        this.engine = engine;
        this.originalColumnDefs = columnDefs;


        this.opts = Object.assign({
            rowModelType: 'serverSide',
            suppressServerSideFullWidthLoadingRow: true,
            cacheBlockSize: Math.max(1, engine?.numRows?.() | 0),
            maxBlocksInCache: 2,
            refreshDebounceMs: 10,
            blockLoadDebounceMillis: 10,
            refreshMaxDelayMs: 250,
            rowBuffer: 0,
            settingsDict: null,
            maxConcurrentDatasourceRequests: -1,
            gridOptions: {}
        }, opts || {});

        this.globalViews = globalViews;
        this._projection = columnDefs.map(d => d.field || d.colId).filter(Boolean);
        this.enableFilterMemory = enableFilterMemory;

        this.dirty = { rows: new Set(), cols: new Set(), cells: new Set() };
        this._flushScheduled = false;
        this._pendingPostEditRefresh = false;

        this.name = name;
        this.grid$ = this.context.page.createStore(this.name, {
            initialized: false,
            lastIdx: null,
            pivotInitialized: false,
            presetLoaded: false,
            weight: 'grossSize'
        });

        this.api = null;
        this.gridOptions = null;
        this._gridWidget = null;
        this._flashTimers = new Map();
        this.domTimeout = false;
        this.pending_styles = new Map();

        this._bootstrapPollTimer = null;
        this._treeStateInitDone = false;
        this._firstEventTimer = null;
        this._firstEventEmitted = false;

        this.columnRegistry = new AgColumnRegistry(this);
        this.registerColumns(this.originalColumnDefs);
        this.filterManager = new ArrowAgGridFilter(this);
        this.searchManager = new GlobalSearchManager(this);

        this.emitter = this.context.page.emitter;


        this._memo = memoize;
        this._guessWidth = this._guessWidthRaw // this._memo(this._guessWidthRaw.bind(this));

        this._setupReactions();
        this.keyCoalescer = null;
        this.selector = null;

        this.id_rows_by_user = new Map();
        this.getRows = debouncePerArgs(this._getRows, 10, {leading:true, trailing:false});

        this._idProperty = this.engine._idProperty;
        this._idxProperty = this.engine._idxProperty;

        this._enablePasteOverride = !!enablePasteOverride;
        this._pasteOverrideManager = null;
    }

    _setupReactions() {
        const adapter = this;
        this._debounceRefresh = debounce(this._softRefresh.bind(this), this.opts.refreshDebounceMs);
        if (this?.context?.page?.quickSearch$) {
            this.context.page.quickSearch$.onChanges(async (q) => {
                adapter._notifyQuickSearch(q);
                await adapter._debounceRefresh();
            })
        }
    }

    /* -------------------------------- grid events ----------------------------- */

    _notifyFirstEvent(...args) {
        if (this._firstEventEmitted) return;
        this._firstEventEmitted = true;
        this.emitter.emitAsync(ArrowAgGridAdapter.FIRST_EVENT, ...args);
    };

    _notifyGeneric(key, ...args){this.emitter.emitAsync(key, ...args)};
    _notify = debouncePerArgs((...args) => this._notifyGeneric(...args), 275, {leading:true, trailing:true, keyIdx: 0});

    /* -------------------------------- triggers-------------------------------- */

    // Grid
    static FIRST_EVENT = "grid:first";
    static API_READY = "grid:ready";

    // Sorts & Filters
    static SORT_EVENT = "grid:sort"
    static FILTER_EVENT = "grid:filter"
    static FILTER_MODEL_EVENT = "grid:filter-model"
    static SEARCH_EVENT = "grid:search"
    static COLUMNS_EVENT = "grid:columns"
    static COLUMN_RESIZE_EVENT = "grid:column-resize"
    static ROW_COUNT_EVENT = "grid:row-count"

    _notifyFilterModel(...args) {this._notify(ArrowAgGridAdapter.FILTER_EVENT, ...args)}
    _notifyQuickSearch(...args) {this._notify(ArrowAgGridAdapter.FILTER_EVENT, ...args);}
    _notifyRowCount(...args) {
        const my_event = ArrowAgGridAdapter.ROW_COUNT_EVENT + ":" + this.name;
        this._notify(my_event, ...args);
    }

    // Keyboard
    static KEYDOWN_EVENT = "grid:keydown";

    // Clipboard
    static CLIPBOARD_EVENT = "grid:clipboard";
    static PASTE_START_EVENT = "grid:paste-start";
    static PASTE_END_EVENT = "grid:paste-end";

    // Rows
    static ROW_CLICK_EVENT = "grid:row-clicked";
    static ROW_DBL_CLICK_EVENT = "grid:row-dbl-click";
    static ROW_SELECT_EVENT = "grid:row-selected";

    // Columns
    static COL_HEADER_EVENT = "grid:header";

    // Misc
    static CONTEXT_EVENT = "grid:context-menu"
    static TOOLBAR_EVENT = "grid:toolbar";

    registerColumns(columnDefs) {
        if (!this.columnRegistry) this.columnRegistry = new AgColumnRegistry(this);
        this.engine.registerColumnsFromDefs(columnDefs, this.opts.settingsDict);
        this.columnRegistry.registerBatchColumns(columnDefs);

        /* Dynamically Add all market columns
            as dependencies to the ref-level column */

        // const allCols = [
        //     ...this.engine._fieldNames, // Physical
        //     ...this.engine.listDerivedColumns() // Virtual
        // ];

        // const dynamicDepConsumers = ['refLevel', 'refMktMkt', 'refMktDisp', 'refBid', 'refAsk', 'refMid'];
        // dynamicDepConsumers.forEach(colName => {
        //     if (this.engine._isDerived(colName)) {
        //         this.engine.addDependencies(colName, marketPriceCols);
        //     }
        // });
    }

    _getNewColumnIndex(key, _index, { toRight = true, breakPins = false, pinned = null } = {}) {
        const api = this.api;
        if (!api) return 0;

        const col = api.getColumn(key);
        if (!col) return 0;

        const side = pinned === 'left' || pinned === 'right' ? pinned : null;

        // Normalize the column's pinning so region math is deterministic.
        try { api.setColumnsPinned([key], side); } catch {}
        const all = api.getAllGridColumns() || [];
        if (!all.length) return 0;

        // Build the post-removal order the API uses when inserting at toIndex.
        const after = [];
        for (let i = 0; i < all.length; i++) {
            if (all[i] !== col) after.push(all[i]);
        }
        if (!after.length) return 0;

        // Partition counts by pinning on the 'after' list; AG Grid orders as left -> center -> right.
        let leftCount = 0, centerCount = 0, rightCount = 0;
        for (let i = 0; i < after.length; i++) {
            const p = typeof after[i].getPinned === 'function' ? after[i].getPinned() : null;
            if (p === 'left') leftCount++;
            else if (p === 'right') rightCount++;
            else centerCount++;
        }

        // Compute insertion index within 'after' (i.e., after the removal step AG Grid does internally).
        let toIndex;
        if (side === 'left') {
            toIndex = toRight ? leftCount : 0;
        } else if (side === 'right') {
            toIndex = toRight ? after.length : (leftCount + centerCount);
        } else if (breakPins) {
            toIndex = toRight ? after.length : 0;
        } else {
            // Center only: just after left pins or just before right pins.
            toIndex = toRight ? (leftCount + centerCount) : leftCount;
        }

        if (toIndex < 0) toIndex = 0;
        if (toIndex > after.length) toIndex = after.length;

        return toIndex;
    }

    moveColumn(key, index, opts = { breakPins: false, toRight: true, pinned: null, ensureScroll: null }) {
        const api = this.api;
        if (!api) return 0;

        const { pinned = null, ensureScroll = null } = opts || {};

        const toIndex = this._getNewColumnIndex(key, index, opts);
        api.moveColumns([key], toIndex);

        // Re-assert pinning in case grid options (e.g. lockPinned) changed it during move.
        if (pinned === 'left' || pinned === 'right') {
            try { api.setColumnsPinned([key], pinned); } catch {}
        } else {
            try { api.setColumnsPinned([key], null); } catch {}
        }

        if (ensureScroll !== null) {
            // Valid positions: 'start' | 'middle' | 'end' | 'auto' (default).
            try { api.ensureColumnVisible(key, ensureScroll === true ? undefined : ensureScroll); } catch {}
        }

        return toIndex;
    }

    static create(context, engine, columnDefs, opts = {}, {name='portfolio', globalViews=[], enableFilterMemory=true, enablePasteOverride=false}={}) {
        return new ArrowAgGridAdapter(context, engine, columnDefs, opts, {name:name, globalViews:globalViews, enableFilterMemory:enableFilterMemory, enablePasteOverride:enablePasteOverride});
    }

    async onKeyDown(e) {
        try {
            const event = e.event;
            const api = e.api;
            const key = event.key;
            const copyPressed = (event.ctrlKey || event.metaKey) && (key === 'c' || key === 'C') && (!event.shiftKey);
            if (!copyPressed) return;

            const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            if (now - lastCopyAt < THROTTLE_MS) return;
            lastCopyAt = now;

            event.preventDefault();

            const ranges = api.getCellRanges ? api.getCellRanges() : null;
            if (!ranges || ranges.length === 0) return;

            const displayedCount = this.engine.numRows();
            if (!displayedCount) return;

            const rowToColPos = new Map();    // ri -> Set(pos)
            const riToNode = new Map();       // ri -> RowNode
            const riVisualIdx = new Map();    // ri -> first-seen visual order index
            const visualOrder = [];           // array of ri in visual encounter order

            const colIdToPos = new Map();     // fallback position map if display order not available
            const posToColId = [];

            const displayCols = typeof api.getAllDisplayedColumns === 'function' ? api.getAllDisplayedColumns() : null;
            const colIdToDisplayPos = new Map();
            if (displayCols && displayCols.length) {
                for (let i = 0; i < displayCols.length; i++) {
                    const c = displayCols[i];
                    const colId = (c && (c.colId || (c.getColId && c.getColId()))) || c;
                    if (colId != null && !colIdToDisplayPos.has(colId)) colIdToDisplayPos.set(colId, i);
                }
            }

            for (let r = 0; r < ranges.length; r++) {
                const rng = ranges[r];
                if (!rng || !rng.columns || rng.columns.length === 0) continue;

                let startIdx = (rng.startRow && typeof rng.startRow.rowIndex === 'number') ? rng.startRow.rowIndex : 0;
                let endIdx = (rng.endRow && typeof rng.endRow.rowIndex === 'number') ? rng.endRow.rowIndex : (displayedCount - 1);
                if (startIdx > endIdx) {
                    const t = startIdx;
                    startIdx = endIdx;
                    endIdx = t;
                }
                if (startIdx < 0) startIdx = 0;
                if (endIdx >= displayedCount) endIdx = displayedCount - 1;

                const cols = rng.columns;
                const posList = new Array(cols.length);
                let posCnt = 0;

                for (let j = 0; j < cols.length; j++) {
                    const c = cols[j];
                    const colId = (c && (c.colId || (c.getColId && c.getColId()))) || c;
                    let pos;
                    if (colIdToDisplayPos.size && colIdToDisplayPos.has(colId)) {
                        pos = colIdToDisplayPos.get(colId);
                        if (posToColId[pos] === undefined) posToColId[pos] = colId;
                    } else {
                        pos = colIdToPos.get(colId);
                        if (pos === undefined) {
                            pos = posToColId.length;
                            colIdToPos.set(colId, pos);
                            posToColId.push(colId);
                        }
                    }
                    posList[posCnt++] = pos;
                }

                for (let i = startIdx; i <= endIdx; i++) {
                    const node = api.getDisplayedRowAtIndex(i);
                    if (!node || !node.data) continue;

                    const ri = (node.data[this._idxProperty] != null) ? node.data[this._idxProperty] :
                        (typeof node.rowIndex === 'number' ? node.rowIndex : i);

                    if (!riToNode.has(ri)) riToNode.set(ri, node);
                    if (!riVisualIdx.has(ri)) {
                        riVisualIdx.set(ri, visualOrder.length);
                        visualOrder.push(ri);
                    }

                    let set = rowToColPos.get(ri);
                    if (!set) {
                        set = new Set();
                        rowToColPos.set(ri, set);
                    }
                    for (let p = 0; p < posCnt; p++) set.add(posList[p]);
                }
            }
            if (rowToColPos.size === 0) return;

            const _rowKeys = new Array(rowToColPos.size);
            {
                let k = 0;
                for (const ri of rowToColPos.keys()) _rowKeys[k++] = ri;
            }
            const rowKeys = _rowKeys.toSorted((a, b) => a - b);

            const setsEqual = (a, b) => {
                if (a === b) return true;
                if (!a || !b || a.size !== b.size) return false;
                for (const v of a) if (!b.has(v)) return false;
                return true;
            };

            const groups = [];
            let prevSet = rowToColPos.get(rowKeys[0]);
            let running = [];
            for (let idx = 0; idx < rowKeys.length; idx++) {
                const ri = rowKeys[idx];
                const s = rowToColPos.get(ri);
                if (setsEqual(prevSet, s)) {
                    running.push(ri);
                } else {
                    let minIdx = Number.MAX_SAFE_INTEGER;
                    for (let k = 0; k < running.length; k++) { const v = riVisualIdx.get(running[k]); if (v < minIdx) minIdx = v; }
                    groups.push({rows: running.slice(), set: prevSet, orderKey: minIdx});
                    running.length = 0;
                    running.push(ri);
                    prevSet = s;
                }
            }
            if (running.length) {
                let minIdx = Number.MAX_SAFE_INTEGER;
                for (let k = 0; k < running.length; k++) { const v = riVisualIdx.get(running[k]); if (v < minIdx) minIdx = v; }
                groups.push({rows: running.slice(), set: prevSet, orderKey: minIdx});
            }

            groups.sort((a, b) => a.orderKey - b.orderKey);

            const comps = new Array(groups.length);
            for (let g = 0; g < groups.length; g++) {
                const rowsArr = groups[g].rows.slice().sort((a, b) => (riVisualIdx.get(a) - riVisualIdx.get(b)));
                const gSet = groups[g].set;

                const posArr = new Array(gSet.size);
                {
                    let i = 0;
                    for (const p of gSet) posArr[i++] = p;
                }
                posArr.sort((a, b) => a - b);

                const colIds = [];
                for (let i = 0; i < posArr.length; i++) {
                    const cid = posToColId[posArr[i]];
                    if (cid !== undefined) colIds.push(cid);
                }

                comps[g] = this.engine.getDisjointRowObjectsFast(rowsArr, colIds, {
                    includeId: false, useFormats: false, gridApi: api, useHeaders: true
                });
            }

            if (!comps.length) return;

            let h = new HyperTable(comps[0]);
            for (let i = 1; i < comps.length; i++) h = h.bulkOperations({add: comps[i]});

            const headers = (h.rowCount() > 1) || (h.fields().size > 1);
            const obj = h.toArray().map(x => x?.toJSON ? x.toJSON() : x);
            await writeObjectToClipboard(obj, {headers, addCommaToNumerics: true});
            flashSelection(this.selector, this.api);
        } catch (err) {
            console.error('Optimized copy failed:', err);
        }
    }

    mount(elementOrSelector) {
        const el = getElement(elementOrSelector);
        const adapter = this;
        const engine = this.engine;
        this.selector = elementOrSelector;
        this.element = el;

        let columnDefs = [];
        columnDefs = this._buildAgColumnDefs(this.originalColumnDefs);
        if (this.filterManager) columnDefs = columnDefs.map(col=>{
            if (col?.context?.dataType) return this.filterManager.configureFilter(col);
            return col;
        });

        const baseGridOptions = Object.assign({
            columnDefs,
            rowModelType: this.opts.rowModelType,
            cacheBlockSize: this.opts.cacheBlockSize,
            maxBlocksInCache: this.opts.maxBlocksInCache,
            maxConcurrentDatasourceRequests: this.opts.maxConcurrentDatasourceRequests,
            rowBuffer: this.opts.rowBuffer,
            suppressColumnVirtualisation: false,
            suppressRowClickSelection: true,
            asyncTransactionWaitMillis: 300,
            animateRows: false,
            serverSideEnableClientSideSort: true,
            getRowId: (params) => {
                const data = params && params.data;
                if (data && data[engine._idProperty] != null) return String(data[engine._idProperty]);
                if (typeof params.getRowIndex === 'function') return String(engine.getRowIdByIndex(params.getRowIndex()));
                const idx = params?.rowIndex != null ? params.rowIndex : 0;
                return String(engine.getRowIdByIndex(idx|0));
            },
            defaultColDef: {
                hide: true,
                sortingOrder: ['desc', 'asc', null],
                menuTabs: ["filterMenuTab"],
                resizable: true,
                sortable: true,
                filter: true,
                editable: false,
                floatingFilter: false,
                autoHeight: false,
                wrapText: false,
                minWidth:30,
                suppressMovable: false,
                suppressSizeToFit: true,
                suppressColumnsToolPanel: false,
                suppressSpanHeaderHeight: true,
                suppressHeaderFilterButton: true,
                enableCellChangeFlash: false,
                cellEditor: CustomCellEditor,
                suppressKeyboardEvent: (params) => {
                    return isArrowKey(params.event) || isCopyHotKey(params.event);
                },
                valueGetter: (p) => {
                    try {
                        const colId = p.column ? p.column.getColId() : (p.colId || p.colDef?.field);
                        const ri = p?.data?.[this._idxProperty];
                        if (ri == null || colId == null) return null;
                        return engine.getCell(ri | 0, colId, p);
                    } catch (e) {
                        return null;
                    }
                },
                valueSetter: nullishValueSetter,
            },
            suppressChangeDetection: true,
            suppressAggFuncInHeader: true,
            suppressMaintainUnsortedOrder: false,
            maintainColumnOrder: true,
            deltaSort: true,
            suppressAnimationFrame: false,
            suppressHeaderFocus: true,
            suppressAutoSize: true,
            suppressTouch: true,
            suppressColumnMoveAnimation: true,
            suppressRowHoverHighlight: true,
            enableRangeSelection: true,
            alwaysShowVerticalScroll: true,
            suppressClipboardPaste: false,
            suppressCutToClipboard: true,
            copyHeadersToClipboard: false,
            suppressScrollOnNewData: true,
            suppressFocusAfterRefresh: true,
            pagination: false,
            suppressMenuHide: false,
            enableCellTextSelection: false,
            stopEditingWhenCellsLoseFocus: true,
            enterNavigatesVertically: false,
            enterNavigatesVerticallyAfterEdit:true,
            skipHeaderOnAutoSize: true,
            tooltipShowDelay: 500,
            cellFlashDuration: 100,
            cellFadeDuration: 500,
            scrollbarWidth: 16,
            rowHeight: 28,
            headerHeight: 32,
            suppressContextMenu: false,
            overlayLoadingTemplate: '<div class="overlay-inner">Fetching portfolio...</div>',
            processDataFromClipboard: (params) => processDataFromClipboard(params, adapter),
            onColumnHeaderContextMenu: (e) => {
                e.api.showColumnFilter(e.column.colId);
            },
            onSortChanged: async (e) => {
                const sortModel = e.api.sortController.getSortModel();
                if (sortModel.length === 0) {
                    if (adapter.columnRegistry.hasColumnDef('sortIndex')) {
                        e.api.applyColumnState({
                            state: [
                                {colId: 'sortIndex', sort: 'desc'},
                                {colId: 'tnum', sort: 'asc'},
                            ],
                            applyOrder: false
                        });
                        try { e.api.refreshServerSide({ purge: false }); } catch {}
                    }
                }
                adapter._notify(ArrowAgGridAdapter.SORT_EVENT, e);
            },
            onFilterChanged: async (e) => {
                const idx = e?.api?.getFirstDisplayedRowIndex();
                try { e.api.refreshServerSide({ purge: false }); } catch {}
                adapter._notifyFilterModel(e);
                if (idx != null) e?.api?.ensureIndexVisible(idx, 'top');
            },
            onColumnVisible: async (e) => {
                try { this._processColumnVisibleEvent(e); } catch (err) {
                    console.error('[grid] columnVisible handler failed:', err);
                }
            },
            rowClassRules: (() => {
                // Hoist getters once at mount time — not per row per render.
                const _getDesig = engine.fieldSet().has('desigName') ? engine._getValueGetter('desigName') : null;
                const _getAssigned = engine.fieldSet().has('assignedTrader') ? engine._getValueGetter('assignedTrader') : null;
                const _getDesc = engine.fieldSet().has('description') ? engine._getValueGetter('description') : null;
                const _getRef = engine.fieldSet().has('refLevel') ? engine._getValueGetter('refLevel') : null;
                const _getDesigTraderId = engine.fieldSet().has('desigTraderId') || engine.columnDefs.has('desigTraderId')
                    ? engine._getValueGetter('desigTraderId') : null;
                const _getAssignedTrader = _getAssigned; // alias for clarity
                const _getIsReal = engine.fieldSet().has('isReal') || engine.columnDefs.has('isReal')
                    ? engine._getValueGetter('isReal') : null;

                return {
                    'error-row': (params) => {
                        if (!_getDesig) return;
                        const ri = params?.data?.[adapter._idxProperty];
                        if (ri == null) return;
                        if ((params.data?.desigName ?? _getDesig(ri)) == null) return true;
                        if (_getAssigned && (params.data?.assignedTrader ?? _getAssigned(ri)) == null) return true;
                        if (_getDesc && (params.data?.description ?? _getDesc(ri)) == null) return true;
                        if (_getRef && (params.data?.refLevel ?? _getRef(ri)) == null) return true;
                        return null;
                    },
                    'trader-name-matches': (params) => {
                        if (!_getDesigTraderId) return;
                        const ri = params?.data?.[adapter._idxProperty];
                        if (ri == null) return;
                        if (adapter.pending_styles.has(ri)) {
                            return coerceToNumber(adapter.pending_styles.get(ri)['claimed']) === 1;
                        }
                        const user = adapter?.context?.page?.userManager()?.userProfile$?.get('username');
                        const display = adapter?.context?.page?.userManager()?.userProfile$?.get('displayName');
                        const desig = params.data?.desigTraderId ?? _getDesigTraderId(ri);
                        const assigned = _getAssignedTrader ? (params.data?.assignedTrader ?? _getAssignedTrader(ri)) : null;
                        if (desig != null && desig?.toLowerCase() === user?.toLowerCase()) return true;
                        if (assigned != null && assigned?.toLowerCase() === display?.toLowerCase()) return true;
                        return false;
                    },
                    'removed-row': (params) => {
                        if (!_getIsReal) return;
                        const ri = params?.data?.[adapter._idxProperty];
                        if (ri == null) return;
                        if (adapter.pending_styles.has(ri)) {
                            return coerceToNumber(adapter.pending_styles.get(ri)['isReal']) === 0;
                        }
                        const real = params.data?.isReal ?? _getIsReal(ri);
                        return coerceToNumber(real) === 0;
                    }
                };
            })(),
            onColumnMoved: () => this._updateProjection(),
            onColumnPinned: () => this._updateProjection(),
            onDisplayedColumnsChanged: () => this._updateProjection(),
            onCellKeyDown: async (e) => {

                try {
                    const isDelete = e?.event?.key === 'Delete';
                    const isTyping = isTextInputTarget(e?.event?.target);
                    if (!isDelete || isTyping) {
                        return adapter.onKeyDown(e);
                    }

                    e.event.preventDefault();

                    const api = e?.api;
                    const colApi = e?.columnApi;
                    if (!api || !colApi) {
                        return adapter.onKeyDown(e);
                    }

                    const ranges = typeof api.getCellRanges === 'function' ? (api.getCellRanges() || []) : [];
                    const hasRange = Array.isArray(ranges) && ranges.length > 0;

                    if (!hasRange) {
                        const field = e?.colDef?.field;
                        if (!field) return adapter.onKeyDown(e);
                        const singlePayload = { ...e, data: { ...e.data, [field]: null }, __isDelete: true };
                        return adapter._processValueChange(singlePayload);
                    }

                    await clearSelectedCellsOnDelete(e, adapter);
                } catch (err) {
                    console.error('[grid] delete-key handler failed', err);
                    // Fail-safe: delegate to default handler
                    return adapter.onKeyDown(e);
                }
            },
            onCellValueChanged: async (e) => {
                await adapter._processValueChange(e)
            },
            sideBar: {
                toolPanels: [
                    {
                        id: 'treeColumnChooser',
                        labelDefault: 'Available Columns',
                        labelKey: 'treeColumnChooser',
                        iconKey: 'columns',
                        toolPanel: TreeColumnChooser,
                        toolPanelParams: {
                            context: adapter.context,
                            adapter: adapter,
                            toolbarId: 'treeColumnChooser',
                            gridName: adapter.name,
                            globalPresets: adapter.getGlobalPresets(),
                            outsideSaveBtnSelector: '#saveState-outside',
                            config: {enableFilterMemory: adapter.enableFilterMemory}
                        }
                    },
                ],
                position: "right",
            },
            statusBar: {
                statusPanels: [
                    {
                        statusPanel: ArrowSsrAggregationStatusBar,
                        key: "statusBarCount",
                        align: 'right',
                        statusPanelParams: { arrowEngine: this.engine, adapter: this }
                    },
                ]
            },
            icons: {
                overview: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 20 20" style="transform:rotate(90deg); opacity:0.75;"><path fill="currentColor" d="M4 4a2 2 0 0 1 2-2h4.586a1.5 1.5 0 0 1 1.06.44l3.915 3.914A1.5 1.5 0 0 1 16 7.414V16a2 2 0 0 1-2 2h-3.088q.088-.222.088-.476c0-.186-.04-.364-.11-.524H14a1 1 0 0 0 1-1V8h-3.5A1.5 1.5 0 0 1 10 6.5V3H6a1 1 0 0 0-1 1v4.094a1.46 1.46 0 0 0-1-.01zm7.5 3h3.293L11 3.207V6.5a.5.5 0 0 0 .5.5M4.878 9.282l.348 1.071a2.2 2.2 0 0 0 1.398 1.397l1.072.348l.022.006a.423.423 0 0 1 0 .798l-1.072.348a2.2 2.2 0 0 0-1.399 1.397l-.348 1.07a.423.423 0 0 1-.798 0l-.348-1.07a2.2 2.2 0 0 0-.652-.977a2.2 2.2 0 0 0-.747-.426l-1.072-.348a.423.423 0 0 1 0-.798l1.072-.348a2.2 2.2 0 0 0 1.377-1.397l.348-1.07a.423.423 0 0 1 .799 0m4.905 7.931l-.766-.248a1.58 1.58 0 0 1-.998-.998l-.25-.765a.302.302 0 0 0-.57 0l-.248.765a1.58 1.58 0 0 1-.984.998l-.765.248a.303.303 0 0 0 0 .57l.765.249a1.58 1.58 0 0 1 1 1.002l.248.764a.302.302 0 0 0 .57 0l.249-.764a1.58 1.58 0 0 1 .999-.999l.765-.248a.302.302 0 0 0 0-.57z"/></svg>',
                columns: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" style="transform:rotate(90deg); opacity:0.75;"><path fill="currentColor" d="M11.71 17.99A5.993 5.993 0 0 1 6 12c0-3.31 2.69-6 6-6c3.22 0 5.84 2.53 5.99 5.71l-2.1-.63a3.999 3.999 0 1 0-4.81 4.81zM22 12c0 .3-.01.6-.04.9l-1.97-.59c.01-.1.01-.21.01-.31c0-4.42-3.58-8-8-8s-8 3-8 8s3.58 8 8 8c.1 0 .21 0 .31-.01l.59 1.97c-.3.03-.6.04-.9.04c-5.52 0-10-4.48-10-10S6.48 2 12 2s10 4.48 10 10m-3.77 4.26l2.27-.76c.46-.15.45-.81-.01-.95l-7.6-2.28c-.38-.11-.74.24-.62.62l2.28 7.6c.14.47.8.48.95.01l.76-2.27l3.91 3.91c.2.2.51.2.71 0l1.27-1.27c.2-.2.2-.51 0-.71z"/></svg>',
                loading: {svg:'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2A10 10 0 1 0 22 12A10 10 0 0 0 12 2Zm0 18a8 8 0 1 1 8-8A8 8 0 0 1 12 20Z" opacity=".5"/><path fill="currentColor" d="M20 12h2A10 10 0 0 0 12 2V4A8 8 0 0 1 20 12Z"><animateTransform attributeName="transform" dur="1s" from="0 12 12" repeatCount="indefinite" to="360 12 12" type="rotate"/></path></svg>'},
            },
            // overlayLoadingTemplate: 'Loading data... one moment please.',
            onToolPanelVisibleChanged: (params) => {
                if (!params.visible) {
                    params?.api?.treeService?.clearSearch();
                } else {
                    params?.api?.treeService?.searchInput.focus();
                }
                adapter._notify(ArrowAgGridAdapter.TOOLBAR_EVENT, params);
            },
            onFirstDataRendered: (params) => {
                params?.api?.treeService?.completeInitialization();
                params?.api?.treeService?._initializeStateManagement();
                setTimeout(() => {
                    this.domTimeout = true;
                    adapter._notifyFirstEvent(params);
                }, ArrowAgGridAdapter.LOAD_DELAY)
            },
            getContextMenuItems: (params) => {
                adapter._notify(ArrowAgGridAdapter.CONTEXT_EVENT, params);
                const rng =params?.api?.getCellRanges();
                let row_count = 0;
                if (rng) {
                    row_count = Math.abs(rng[0]?.endRow?.rowIndex - rng[0]?.startRow?.rowIndex);
                    row_count = Math.max(0, row_count);
                }
                return [
                    {
                        name: row_count > 1 ? 'Claim All (desig)' : 'Claim Bond (desig)',
                        action: async () => {
                            const claimed = true;
                            const rng = params.api.getCellRanges()[0]
                            const sr = Math.min(rng.startRow.rowIndex, rng.endRow.rowIndex);
                            const er = Math.max(rng.startRow.rowIndex, rng.endRow.rowIndex);

                            const nodes = [];
                            const updates = [];

                            for (let i = sr; i <= er; i++) {
                                try {
                                    const node = params.api.getDisplayedRowAtIndex(i);
                                    const gui = params.api.getCellRendererInstances({rowNodes:[node]});
                                    if (gui.length) {
                                        gui[0].eGui.classList.toggle('bond-claimed', claimed);
                                    }
                                    nodes.push(node);
                                    const ri = node.data[this._idxProperty];
                                    const id_col = engine._idProperty;
                                    const id = node.data[id_col]
                                    if (ri == null) {
                                        console.error(node, i)
                                        continue;
                                    }
                                    updates.push({[this._idxProperty]:ri, [id_col]:id, claimed:1})
                                } catch(e) {
                                    console.error(e)
                                }
                            }
                            params.api.flashCells({rowNodes:nodes, columns:['claimed']});
                            await adapter.applyServerUpdateTransaction(updates, {emitAsEdit:true});
                        },
                        icon: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 20 20"><g fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"><path d="M11.5 6a2.5 2.5 0 1 0 0 5a2.5 2.5 0 0 0 0-5M7 8.5a4.5 4.5 0 1 1 9 0a4.5 4.5 0 0 1-9 0"/><path d="M2 4a2.5 2.5 0 0 1 5 0v3.272c0 .883.31 1.737.874 2.415l1.394 1.673a1 1 0 1 1-1.536 1.28l-1.394-1.673A5.77 5.77 0 0 1 5 7.272V4a.5.5 0 0 0-1 0v3.75a4 4 0 0 0 .47 1.882l3.412 6.397a1 1 0 0 1-1.764.942l-3.412-6.398A6 6 0 0 1 2 7.75z"/><path d="M7.5 12a1 1 0 0 1 1-1H11a6 6 0 0 1 6 6v2a1 1 0 1 1-2 0v-2a4 4 0 0 0-4-4H8.5a1 1 0 0 1-1-1M7 15.5a1 1 0 0 1 1 1V19a1 1 0 1 1-2 0v-2.5a1 1 0 0 1 1-1"/></g></svg>`

                    },
                    {
                        name: row_count > 1 ? 'Claim All (me)' : 'Claim Bond (me)',
                        action: async () => {
                            const claimed = true;
                            const rng = params.api.getCellRanges()[0]
                            const sr = Math.min(rng.startRow.rowIndex, rng.endRow.rowIndex);
                            const er = Math.max(rng.startRow.rowIndex, rng.endRow.rowIndex);

                            const nodes = [];
                            const updates = [];

                            for (let i = sr; i <= er; i++) {
                                try {
                                    const node = params.api.getDisplayedRowAtIndex(i);
                                    const gui = params.api.getCellRendererInstances({rowNodes:[node]});
                                    if (gui.length) {
                                        gui[0].eGui.classList.toggle('bond-claimed', claimed);
                                    }
                                    nodes.push(node);
                                    const ri = node.data[this._idxProperty];
                                    const id_col = engine._idProperty;
                                    const id = node.data[id_col]
                                    if (ri == null) {
                                        console.error(node, i)
                                        continue;
                                    }
                                    updates.push({[this._idxProperty]:ri, [id_col]:id, claimed:2})
                                } catch(e) {
                                    console.error(e)
                                }
                            }
                            params.api.flashCells({rowNodes:nodes, columns:['claimed']});
                            await adapter.applyServerUpdateTransaction(updates, {emitAsEdit:true});
                        },
                        icon: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 28 28"><path fill="currentColor" d="M14 4.26v7.99a.75.75 0 0 0 1.5 0V5.763c0-.4.325-.76.751-.763c.383-.002.75.363.75.795v8.955a.75.75 0 0 0 1.245.563l.186-.166c.24-.215.398-.357.629-.49c.266-.154.657-.305 1.377-.416c.664-.102 1.346-.012 1.829.25c.35.19.606.469.697.894l-2.144 1.5a1 1 0 0 0-.117.102l-2.535 2.704a12.8 12.8 0 0 0-2.128 3.07l-.377.762a1.75 1.75 0 0 1-1.568.974h-3.867c-.562 0-1.055-.262-1.326-.694a21 21 0 0 1-1.556-2.977c-.493-1.188-.846-2.443-.846-3.576V8.27c0-.424.32-.77.75-.77s.75.343.75.77v4.48a.75.75 0 0 0 1.5 0V5.795c0-.445.332-.795.747-.795c.421 0 .753.35.753.795v6.455a.75.75 0 1 0 1.5 0V4.272c0-.427.32-.772.752-.772c.44 0 .748.332.748.76m2.242-.76a2.2 2.2 0 0 0-.822.158Zm-.822.158A2.21 2.21 0 0 0 13.251 2c-1.1 0-1.912.742-2.167 1.658a2.2 2.2 0 0 0-.837-.158C8.887 3.5 8 4.644 8 5.795v.33A2.3 2.3 0 0 0 7.25 6C5.91 6 5 7.102 5 8.27v8.98c0 1.414.432 2.878.96 4.151a22 22 0 0 0 1.67 3.198c.572.912 1.572 1.398 2.598 1.398h3.867a3.25 3.25 0 0 0 2.913-1.81l.377-.762c.49-.99 1.122-1.902 1.877-2.708l2.482-2.647l2.436-1.706c.2-.14.32-.37.32-.614c0-1.239-.639-2.101-1.518-2.578c-.838-.454-1.87-.552-2.771-.413c-.736.113-1.272.276-1.71.497v-7.46c0-1.173-.955-2.304-2.26-2.296"/></svg>`

                    },
                    // {
                    //     name: 'Toggle Claims',
                    //     action: () => {
                    //         const rng = params.api.getCellRanges()[0]
                    //         const sr = Math.min(rng.startRow.rowIndex, rng.endRow.rowIndex);
                    //         const er = Math.max(rng.startRow.rowIndex, rng.endRow.rowIndex);
                    //         const ri = params.node.data.__srcIndex;
                    //         const colDef = params.column.colDef;
                    //         if (ri == null) return;
                    //         const old_value = coerceToBool(engine.getCell(ri, 'claimed'));
                    //         const new_value = old_value ? 0 : 1;
                    //         const nodes = [];
                    //         for (let i = sr; i <= er; i++) {
                    //             const node = params.api.getDisplayedRowAtIndex(i);
                    //             nodes.push(node);
                    //             const gui = params.api.getCellRendererInstances({rowNodes:[node]})?.[0]
                    //             gui.eGui.classList.toggle('bond-claimed', !!new_value);
                    //             requestAnimationFrame(async ()=>{
                    //                 await engine.applyValue(ri, 'claimed', new_value, colDef)
                    //                 adapter.markCell(ri, 'claimed');
                    //             })
                    //         }
                    //         params.api.flashCells({rowNodes:nodes, columns:['claimed']});
                    //     },
                    //     icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640" height="16" width="16"><path fill="currentColor" d="M569 79C559.6 69.6 544.4 69.6 535.1 79L493 121.1L485.6 116.1C465.8 103 442.6 96 418.9 96L359.7 96L359.3 96L215.7 96C183.9 96 153.4 108.6 130.8 131.1L127 135L39 223C29.6 232.4 29.6 247.6 39 256.9C48.4 266.2 63.6 266.3 72.9 256.9L160.9 168.9C143.9 151.9 143.9 151.9 160.9 168.9L164.8 165C178.4 151.6 196.7 144 215.8 144L262.1 144L170.4 235.7C154.8 251.3 154.8 276.6 170.4 292.3L171.2 293.1C218 340 294 340 340.9 293.1L368 266L465.8 363.8C481.4 379.4 481.4 404.7 465.8 420.4L456 430.2L425 399.2C415.6 389.8 400.4 389.8 391.1 399.2C381.8 408.6 381.7 423.8 391.1 433.1L419.1 461.1C401.6 471.5 381.9 477.8 361.5 479.6L313 431C303.6 421.6 288.4 421.6 279.1 431C269.8 440.4 269.7 455.6 279.1 464.9L294.1 479.9L290.3 479.9C254.2 479.9 219.6 465.6 194.1 440.1L193 439C183.6 429.6 168.4 429.6 159.1 439L71 527C61.6 536.4 61.6 551.6 71 560.9C80.4 570.2 95.6 570.3 104.9 560.9L176.9 488.9C209.1 514.1 248.9 527.9 290.2 527.9L349.7 527.9C398.5 527.9 445.3 508.5 479.8 474L499.7 454.1C500.9 452.9 502 451.8 503.1 450.6C503.8 450.1 504.4 449.5 505 448.9L601 352.9C610.4 343.5 610.4 328.3 601 319C591.6 309.7 576.4 309.6 567.1 319L521.3 364.8C517.1 352 510 339.9 499.8 329.7L385 215C375.6 205.6 360.4 205.6 351.1 215L307 259.1C280.5 285.6 238.5 287.1 210.3 263.7L309 165C322.4 151.6 340.6 144 359.6 143.9L368.1 143.9L368.3 143.9L419.1 143.9C433.3 143.9 447.2 148.1 459 156L482.7 172C492.2 178.3 504.9 177.1 513 169L569 113C578.4 103.6 578.4 88.4 569 79.1z"/></svg>`
                    // },
                    {
                        name: row_count > 1 ? 'Unclaim All' : 'Unclaim Bond',
                        action: async () => {
                            const claimed = false;
                            const rng = params.api.getCellRanges()[0]
                            const sr = Math.min(rng.startRow.rowIndex, rng.endRow.rowIndex);
                            const er = Math.max(rng.startRow.rowIndex, rng.endRow.rowIndex);

                            const nodes = [];
                            const updates = [];

                            for (let i = sr; i <= er; i++) {
                                try {
                                    const node = params.api.getDisplayedRowAtIndex(i);
                                    const gui = params.api.getCellRendererInstances({rowNodes:[node]});
                                    if (gui.length) {
                                        gui[0].eGui.classList.toggle('bond-claimed', claimed);
                                    }
                                    nodes.push(node);
                                    const ri = node.data[this._idxProperty];
                                    const id_col = engine._idProperty;
                                    const id = node.data[id_col]
                                    if (ri == null) {
                                        console.error(node, i)
                                        continue;
                                    }
                                    updates.push({[this._idxProperty]:ri, [id_col]:id, claimed:0})
                                } catch(e) {
                                    console.error(e)
                                }
                            }
                            params.api.flashCells({rowNodes:nodes, columns:['claimed']});
                            await adapter.applyServerUpdateTransaction(updates, {emitAsEdit:true});
                        },
                        icon: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 28 28"><path fill="currentColor" d="M3.28 2.22a.75.75 0 1 0-1.06 1.06l3.41 3.41c-.4.422-.63.993-.63 1.58v8.98c0 1.414.432 2.878.96 4.151a22 22 0 0 0 1.67 3.198c.572.912 1.572 1.398 2.597 1.398h3.867a3.25 3.25 0 0 0 2.914-1.81l.377-.762c.49-.99 1.122-1.902 1.877-2.708l.19-.203l5.267 5.267a.75.75 0 0 0 1.061-1.061zm15.112 17.232l-.224.239a12.8 12.8 0 0 0-2.128 3.07l-.377.762a1.75 1.75 0 0 1-1.569.974h-3.867c-.56 0-1.054-.262-1.325-.694a21 21 0 0 1-1.556-2.977c-.493-1.188-.846-2.443-.846-3.576V8.27c0-.2.07-.382.19-.519L8 9.061v3.689a.75.75 0 0 0 1.5 0v-2.19l1.5 1.5v.19a.75.75 0 0 0 .92.73zM8.152 4.97L9.5 6.318v-.523c0-.445.332-.795.747-.795c.421 0 .753.35.753.795v2.023l1.5 1.5V4.272c0-.427.32-.772.752-.772c.44 0 .748.332.748.76v6.558l1.497 1.497l.003-.065V5.763c0-.4.324-.76.75-.763c.384-.002.75.363.75.795v8.023l1.378 1.378l.054-.05c.24-.214.398-.356.629-.49c.266-.153.656-.304 1.377-.415c.663-.102 1.346-.012 1.829.25c.35.19.606.469.697.894l-2.144 1.5a1 1 0 0 0-.117.102l-.258.276l1.06 1.06l.239-.253l2.436-1.706c.2-.14.32-.37.32-.614c0-1.239-.639-2.101-1.518-2.578c-.839-.454-1.87-.552-2.772-.413c-.735.113-1.272.276-1.71.497v-7.46c0-1.173-.954-2.304-2.258-2.296a2.2 2.2 0 0 0-.822.158c-.249-.924-1.062-1.659-2.169-1.658c-1.1 0-1.912.742-2.167 1.658a2.2 2.2 0 0 0-.837-.158c-1.029 0-1.786.654-2.095 1.47"/></svg>`
                    },
                    {
                        name: 'Toggle Algo Assignment',
                        action: async () => {
                            const claimed = true;
                            const rng = params.api.getCellRanges()[0]
                            const sr = Math.min(rng.startRow.rowIndex, rng.endRow.rowIndex);
                            const er = Math.max(rng.startRow.rowIndex, rng.endRow.rowIndex);

                            const nodes = [];
                            const updates = [];

                            for (let i = sr; i <= er; i++) {
                                try {
                                    const node = params.api.getDisplayedRowAtIndex(i);
                                    nodes.push(node);
                                    const ri = node.data[this._idxProperty];
                                    const id_col = engine._idProperty;
                                    const id = node.data[id_col]
                                    if (ri == null) {
                                        console.error(node, i)
                                        continue;
                                    }
                                    const isAlgo = engine.getCell(ri, 'algoAssigned');
                                    if (isAlgo) {
                                        const desig = engine.getCell(ri, 'desigName');
                                        updates.push({[this._idxProperty]:ri, [id_col]:id, assignedTrader:desig})
                                    } else {
                                        updates.push({[this._idxProperty]:ri, [id_col]:id, assignedTrader: "ALGO"})
                                    }

                                } catch(e) {
                                    console.error(e)
                                }
                            }
                            params.api.flashCells({rowNodes:nodes, columns:['assignedTrader']});
                            await adapter.applyServerUpdateTransaction(updates, {emitAsEdit:true});
                        },
                        icon: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 20 20"><g fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"><path d="M11.5 6a2.5 2.5 0 1 0 0 5a2.5 2.5 0 0 0 0-5M7 8.5a4.5 4.5 0 1 1 9 0a4.5 4.5 0 0 1-9 0"/><path d="M2 4a2.5 2.5 0 0 1 5 0v3.272c0 .883.31 1.737.874 2.415l1.394 1.673a1 1 0 1 1-1.536 1.28l-1.394-1.673A5.77 5.77 0 0 1 5 7.272V4a.5.5 0 0 0-1 0v3.75a4 4 0 0 0 .47 1.882l3.412 6.397a1 1 0 0 1-1.764.942l-3.412-6.398A6 6 0 0 1 2 7.75z"/><path d="M7.5 12a1 1 0 0 1 1-1H11a6 6 0 0 1 6 6v2a1 1 0 1 1-2 0v-2a4 4 0 0 0-4-4H8.5a1 1 0 0 1-1-1M7 15.5a1 1 0 0 1 1 1V19a1 1 0 1 1-2 0v-2.5a1 1 0 0 1 1-1"/></g></svg>`

                    },
                    'separator',
                    ...(() => {

                        const lockSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h1V6q0-2.075 1.463-3.537T12 1t3.538 1.463T17 6v2h1q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22zm7.413-5.587Q14 15.825 14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17t1.413-.587M9 8h6V6q0-1.25-.875-2.125T12 3t-2.125.875T9 6z"/></svg>`;
                        const unlockSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M13.413 16.413Q14 15.825 14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17t1.413-.587M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h7V6q0-2.075 1.463-3.537T18 1q1.875 0 3.263 1.213T22.925 5.2q.05.325-.225.563T22 6t-.7-.175t-.4-.575q-.275-.95-1.062-1.6T18 3q-1.25 0-2.125.875T15 6v2h3q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22z"/></svg>`;
                        const _collectLockNodes = () => {
                            const rng3 = params.api.getCellRanges?.();
                            if (rng3 && rng3.length > 0) {
                                const sr3 = Math.min(rng3[0].startRow.rowIndex, rng3[0].endRow.rowIndex);
                                const er3 = Math.max(rng3[0].startRow.rowIndex, rng3[0].endRow.rowIndex);
                                const out = [];
                                for (let i = sr3; i <= er3; i++) {
                                    try { out.push(params.api.getDisplayedRowAtIndex(i)); } catch(_) {}
                                }
                                return out;
                            }
                            return params.node ? [params.node] : [];
                        };
                        const buildLockAction = (targetValue) => async () => {
                            const targetNodes = _collectLockNodes();
                            const nodes = [];
                            const updates = [];
                            for (const node of targetNodes) {
                                try {
                                    nodes.push(node);
                                    const ri = node.data[adapter._idxProperty];
                                    const id_col = engine._idProperty;
                                    const id = node.data[id_col];
                                    if (ri == null) continue;
                                    updates.push({[adapter._idxProperty]: ri, [id_col]: id, isLocked: targetValue});
                                } catch(e) { console.error(e); }
                            }
                            params.api.flashCells({rowNodes: nodes, columns: ['isLocked']});
                            await adapter.applyServerUpdateTransaction(updates, {emitAsEdit: true});
                        };
                            return [{
                            name: 'Lock Rows',
                                icon: `<span style="color:var(--red-500,#ef4444)">${lockSvg}</span>`,
                                action: buildLockAction(1),
                            }, {
                            name: 'Unlock Rows',
                                icon: `<span style="color:var(--gray-500,#6b7280)">${unlockSvg}</span>`,
                                action: buildLockAction(0),
                        }];
                    })(),
                    'copy', 'copyWithHeaders', 'export']
            },
        }, this.opts.gridOptions || {});

        this._gridWidget = createGrid(el, baseGridOptions);
        this.api = this._gridWidget;
        this.api.showLoadingOverlay()
        this.gridOptions = baseGridOptions;
        this.afterMount();
        return this;
    }

    _processValueChange(e) {
        if (!e) return;

        const colId =
            (e.column && typeof e.column.getColId === 'function' && e.column.getColId()) ||
            (e.colDef && e.colDef.field);

        if (!colId) return;
        if (!e.data || e?.data?.[this._idxProperty] == null) return
        const ri = e.data[this._idxProperty];
        if (ri < 0) return;

        const newValue = e.data[colId];
        const oldValue = this.engine.getCell(ri, colId);
        if (!e.__isDelete && !didCellValueActuallyChange(oldValue, newValue)) return

        const colDef =
            (e.column && typeof e.column.getColDef === 'function' && e.column.getColDef()) ||
            e.colDef ||
            {};

        requestAnimationFrame(async ()=>{
            if (!this.engine || !this.api) return; // guard against post-dispose RAF
            await this.engine.applyValue(ri, colId, newValue, colDef)
            try {
                if (colId != null && ri >= 0) this.markCell(ri, colId);
            } catch {}
        })
    }

    afterMount() {
        requestAnimationFrame(()=>{
            this._afterMount()
        })
    }

    _afterMount() {
        this.api.setGridOption('serverSideDatasource', this._makeDatasource());
        this.keyCoalescer = installKeyCoalescer(this, this.element, this.api);

        this._epochUnsub = this.engine.onEpochChange((epoch) => {
            this._handleEpochRefresh(epoch)
        });

        this._derivedDirtyUnsub = this.engine.onDerivedDirty((dirtyCols) => {
            this._handleDerivedDirty(dirtyCols)
        })

        this._setupUserMap();
        this._ensureStyles();
        this._setupHovers();
        // this._bootstrapInitialColumnState();

        // Paste override: attach manager if enabled
        if (this._enablePasteOverride) {
            this._pasteOverrideManager = new PasteOverrideManager(this, { enabled: true, createGrid });
            this._pasteOverrideManager.attach();
        }

        const adapter = this;
        if (!this.grid$.get("initialized")) {
            setTimeout(() => {
                adapter.grid$.set('initialized', true);
                adapter._notify(ArrowAgGridAdapter.API_READY, true);
                this._notifyRowCount();
            }, 60);
        }

        const headerBar = this.element.querySelector('.ag-root .ag-header');
        const popup = this.element.querySelector('.ag-root-wrapper');
        this.context.page.addEventListener(headerBar, 'contextmenu', (e) => e.preventDefault());
        this.context.page.addEventListener(popup, 'click', async (e) => {
            const h = e?.target?.closest('.ag-tab.ag-tab-selected span');
            if (h == null) return;
            try {
                e.preventDefault();
                const header = h.textContent.split("(").slice(-1)[0].replace(")", "");
                await writeStringToClipboard(header);
            } catch {}
        });
    }

    _handleEpochRefresh(epoch) {
        const api = this.api;
        const engine = this.engine;
        if (!api || !engine) return;

        // Global epoch = everything changed (table replaced, etc.)
        if (epoch.global) {
            try { api.refreshServerSide({ purge: false }); } catch {}
            return;
        }

        const colsChanged = epoch.colsChanged;
        const rowsChanged = epoch.rowsChanged;

        if (!colsChanged && !rowsChanged) return;

        // 1) If a sort key is affected, we need a full SSRM refresh
        //    because row ORDER may have changed.
        const sortCols = this._getCurrentSortCols();
        if (sortCols.size > 0 && colsChanged) {
            const cols = colsChanged === true ? null : (Array.isArray(colsChanged) ? colsChanged : [colsChanged]);
            if (colsChanged === true || cols.some(c => sortCols.has(c))) {
                try { api.refreshServerSide({ purge: false }); } catch {}
                return;
            }
        }

        // 2) If active filters touch changed columns, SSRM refresh
        //    because the FILTERED set may have changed.
        try {
            const fm = api.getFilterModel?.() || {};
            const filterKeys = Object.keys(fm);
            if (filterKeys.length && colsChanged) {
                const cols = colsChanged === true ? null : (Array.isArray(colsChanged) ? colsChanged : [colsChanged]);
                if (colsChanged === true || cols.some(c => fm.hasOwnProperty(c))) {
                    try { api.refreshServerSide({ purge: false }); } catch {}
                    return;
                }
            }
        } catch {}

        // 3) Otherwise: lightweight cell refresh.
        //    This is the fast path for settings changes, derived column
        //    recomputation, and overlay edits.  It tells AG Grid to
        //    re-query the valueGetter for specific cells without
        //    rebuilding the entire row model.
        const columns = colsChanged === true
            ? undefined  // undefined = all columns
            : (Array.isArray(colsChanged) ? colsChanged : [colsChanged]);

        let rowNodes;
        if (rowsChanged && rowsChanged !== true) {
            // Specific rows: resolve to AG RowNodes
            const ris = (rowsChanged instanceof Int32Array)
                ? rowsChanged
                : (Array.isArray(rowsChanged) ? rowsChanged : []);

            if (ris.length > 0 && ris.length < 500) {
                // Only resolve row nodes for small sets — above 500 it's
                // cheaper to let AG Grid refresh all visible rows.
                const nodes = [];
                for (let i = 0; i < ris.length; i++) {
                    try {
                        const id = engine.getRowIdByIndex(ris[i] | 0);
                        const node = api.getRowNode(id);
                        if (node) nodes.push(node);
                    } catch {}
                }
                if (nodes.length) rowNodes = nodes;
            }
            // else: rowNodes stays undefined → refreshes all visible rows
        }

        try {
            api.refreshCells({
                rowNodes,
                columns,
                force: true,          // force re-render even if AG thinks value hasn't changed
                suppressFlash: true    // settings changes shouldn't flash
            });
        } catch {}
    }

    _handleDerivedDirty(dirtyCols) {
        const api = this.api;
        if (!api || !dirtyCols || dirtyCols.size === 0) return;

        const needsRedraw = this._checkRowClassDependency(dirtyCols);

        if (needsRedraw) {
            try { api.redrawRows(); } catch {}
        }
    }

    _checkRowClassDependency(dirtyCols) {
        if (!this._rowClassDeps) {

            // Build once — the set of columns that rowClassRules read.
            this._rowClassDeps = new Set([
                'desigName', 'assignedTrader', 'description', 'refLevel',
                'desigTraderId', 'isReal', 'claimed', 'isLocked'
            ]);
        }
        for (const col of dirtyCols) {
            if (this._rowClassDeps.has(col)) return true;
        }
        return false;
    }


    setPasteOverride(flag) {
        const enabled = !!flag;
        this._enablePasteOverride = enabled;
        if (enabled && !this._pasteOverrideManager && this.element) {
            this._pasteOverrideManager = new PasteOverrideManager(this, { enabled: true, createGrid });
            this._pasteOverrideManager.attach();
        }
        if (this._pasteOverrideManager) {
            this._pasteOverrideManager.setEnabled(enabled);
        }
    }

    _setupHovers() {

        const adapter = this;
        const gridDiv = this.element;
        this._hoverController = new AbortController();
        const signal = this._hoverController.signal;

        const toggleRow = (idx, add) => {
            const nodes = gridDiv.querySelectorAll('.ag-row[row-index="' + idx + '"]');
            for (let i = 0; i < nodes.length; i++) {
                nodes[i].classList[add ? 'add' : 'remove']('row-hover-all');
            }
        }

        let lastIdx = null;
        gridDiv.addEventListener('mouseover', function (e) {
            const rowEl = e.target.closest('.ag-row');
            if (!rowEl || !gridDiv.contains(rowEl)) return;

            const idx = rowEl.getAttribute('row-index');
            if (idx == null || idx === lastIdx) return;

            if (lastIdx != null) toggleRow(lastIdx, false);
            toggleRow(idx, true);
            lastIdx = idx;
        }, { signal });

        this.element.addEventListener('mouseleave', function () {
            if (lastIdx != null) {
                toggleRow(lastIdx, false);
                lastIdx = null;
            }
        }, { signal });
    }

    _hasAnyDisplayedColumns(api) {
        const cols = api?.getAllDisplayedColumns?.();
        return Array.isArray(cols) && cols.length > 0;
    }

    _scheduleFirstEvent(params) {
        if (this._firstEventTimer != null || this._firstEventEmitted) return;
        const adapter = this;
        const p = params || { api: this.api };
        this._firstEventTimer = setTimeout(() => {
            adapter._firstEventTimer = null;
            adapter.domTimeout = true;
            adapter._notifyFirstEvent(p);
        }, ArrowAgGridAdapter.LOAD_DELAY);
    }

    _tryInitTreeServiceAndState(api) {
        if (!api || this._treeStateInitDone) return false;
        const tree = api.treeService;
        if (!tree) return false;

        let did = false;
        if (typeof tree.completeInitialization === 'function') {
            try { tree.completeInitialization(); did = true; } catch {}
        }
        if (typeof tree._initializeStateManagement === 'function') {
            try { tree._initializeStateManagement(); did = true; } catch {}
        }
        if (did) this._treeStateInitDone = true;
        return did;
    }

    _applyDefaultPresetIfHidden(api) {
        if (!api) return false;
        if (this._hasAnyDisplayedColumns(api)) {
            try { this.grid$.set('presetLoaded', true); } catch {}
            return true;
        }

        const presets = (typeof this.getGlobalPresets === 'function') ? this.getGlobalPresets() : [];
        const p0 = (Array.isArray(presets) && presets.length) ? presets[0] : null;
        if (p0 && Array.isArray(p0.columnState) && p0.columnState.length) {
            try { void this.loadGridState(p0); } catch {
                try { this.grid$.set('presetLoaded', true); } catch {}
            }
            return true;
        }
        const all = this.getAllFields();
        if (all && all.length) {
            try { api.setColumnsVisible(all, true); } catch {}
            return true;
        }

        return false;
    }

    _bootstrapInitialColumnState() {
        if (!this.api) return;
        if (this._bootstrapPollTimer != null) return;

        const adapter = this;
        const POLL_MS = 50;
        const APPLY_PRESET_AFTER_MS = 750;
        const start = (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now();

        const tick = () => {
            adapter._bootstrapPollTimer = null;
            const api = adapter.api;
            if (!api || (typeof api.isDestroyed === 'function' && api.isDestroyed())) return;
            adapter._scheduleFirstEvent({ api });
            if (adapter.grid$?.get?.('presetLoaded')) return;
            if (adapter._hasAnyDisplayedColumns(api)) return;
            adapter._tryInitTreeServiceAndState(api);
            if (adapter._hasAnyDisplayedColumns(api)) return;
            const now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
                ? performance.now()
                : Date.now();
            const elapsed = now - start;

            if (!adapter.grid$?.get?.('presetLoaded') && elapsed >= APPLY_PRESET_AFTER_MS) {
                adapter._applyDefaultPresetIfHidden(api);
                return;
            }

            adapter._bootstrapPollTimer = setTimeout(tick, POLL_MS);
        };
        this._bootstrapPollTimer = setTimeout(tick, 0);
    }



    _ensureStyles() {
        ensureStyles();
    }

    _setupUserMap() {
        if (!this.engine.columnDefs.has('desigTraderId')) return
        const n = this.engine.numRows();
        const mp = new Map();
        for (let i = 0; i < n; i++) {
            const id = this.engine.getRowIdByIndex(i);
            const traderId = this.engine.getCell(i, 'desigTraderId');
            if (id != null && traderId != null) mp.set(String(id), traderId);
        }
        this.id_rows_by_user = mp;
    }

    dispose() {
        // Clear trackers
        this.dirty.rows.clear();
        this.dirty.cols.clear();
        this._flashTimers.forEach(t => clearTimeout(t));
        this._flashTimers.clear();
        if (this._debounceRefresh?.cancel) {
            this._debounceRefresh.cancel();
        }
        this._debounceRefresh = null;

        if (this.searchManager?.dispose) {
            try {this.searchManager.clearCache()} catch(_){};
            this.searchManager.dispose();
            this.searchManager = null;
        }
        this.filterManager = null;
        this.columnRegistry = null;
        if (this.grid$?.dispose) {
            this.grid$.dispose();
        }
        this.originalColumnDefs = null;

        try { this._hoverController?.abort(); } catch {}

        if (this._bootstrapPollTimer) clearTimeout(this._bootstrapPollTimer);
        this._bootstrapPollTimer = null;

        if (this._firstEventTimer) clearTimeout(this._firstEventTimer);
        this._firstEventTimer = null;

        // Destroy paste override manager
        if (this._pasteOverrideManager) {
            try { this._pasteOverrideManager.destroy(); } catch {}
            this._pasteOverrideManager = null;
        }

        // Clean up module-global overlay DOM elements for this adapter's selector
        if (this.selector && overlayMap.has(this.selector)) {
            const el = overlayMap.get(this.selector);
            try { el?.remove(); } catch {}
            overlayMap.delete(this.selector);
        }
        if (overlayTimer) { clearTimeout(overlayTimer); overlayTimer = 0; }

        // Clear id_rows_by_user to free captured row data
        if (this.id_rows_by_user) { this.id_rows_by_user.clear(); this.id_rows_by_user = null; }

        // Clear pending_styles and row pool
        if (this.pending_styles) { this.pending_styles.clear(); this.pending_styles = null; }
        this._rowPool = null;
        this.viewportRows = null;

        if (this?.api?.treeService) this?.api?.treeService.destroy();
        if (this.keyCoalescer) this.keyCoalescer();
        if (this?._notify?.cancelAll) this._notify.cancelAll();

        if (this._epochUnsub) { try { this._epochUnsub(); } catch {} this._epochUnsub = null; }
        if (this._derivedDirtyUnsub) { try { this._derivedDirtyUnsub(); } catch {} this._derivedDirtyUnsub = null; }
        this._rowClassDeps = null;

        if (this.engine && typeof this.engine.dispose === 'function') {
            this.engine.dispose();
        }
        this.engine = null;

        if (this.api && !this.api.isDestroyed()) {
            this.api.destroy();
        }
        this.api = null;
    }


    getGlobalPresets() { return this.globalViews }

    _processColumnVisibleEvent(e) {
        if (!this.domTimeout) return
        if (!e?.api.treeService?.initialized) return;
        if (e.columns && e.columns.length) {
            e.columns.forEach(col => { this._processColumnVisible(col); })
        } else if (e.column) { this._processColumnVisible(e.column) }
    }

    _processColumnVisible(column, { suppressFlash = false } = {}) {
        const name = column.colId || column?.getColId?.() || column?.colDef?.field || column?.colDef?.colId;
        if (!name) return;

        this._updateProjection();

        if (column.visible) {
            // Move to rightmost unpinned spot & scroll into view (unchanged)
            if (this?.api?.treeService?._filterSelected !== true) {
                this.moveColumn(name, Number.MAX_SAFE_INTEGER, { breakPins: false, toRight: true });
            }
            const api = this.api;
            if (api && typeof api.ensureColumnVisible === 'function') {
                requestAnimationFrame(() => api.ensureColumnVisible(name, 'end'));
            }

            // Visual affordance (unchanged)
            if (!suppressFlash && this.grid$.get('presetLoaded')) this.flashColumnHeader(column);

            try {
                const w = (this._guessWidth(name) ?? 51) * 1.6;
                const guessed = Math.min(Math.max(GUESS_MIN_W, Math.round(w)), GUESS_MAX_W);
                this.api.setColumnWidths([{ key: name, newWidth: guessed }], false);
            } catch {}
        }
    }

    flashColumnHeader(column) {
        const colId = column?.getColId();
        if (!colId) return;
        if (this._flashTimers.has(colId)) return;

        const s = this.selector + ` .ag-header-cell[col-id="${colId}"]`;
        const dom = document.querySelector(s);
        if (!dom) return
        dom.classList.add('ag-header-flash');

        const grid = this;
        const _t = setTimeout(()=>{
            dom.classList.remove('ag-header-flash');
            grid._flashTimers.delete(column.getColId());
        }, 400)
        this._flashTimers.set(colId, _t);
    }

    getAllFields() { return Array.from(this.columnRegistry.columns.keys()); }
    getAllColumnDefs() { return this.api?.getGridOption('columnDefs') || Array.from(this.columnRegistry.columns.values()); }
    setColumnsVisible(columnIds, visible) { if (!this.api || !Array.isArray(columnIds)) return; this.api.setColumnsVisible(columnIds, !!visible); }
    applyColumnState(state, applyOrder = true) { if (!this.api || !state) return; this.api.applyColumnState({ state, applyOrder }); }

    getGridState() {
        // Keep capturing width in state (your code already does this)
        let state = this.api?.getColumnState?.();
        if (state) {
            state = state.map(col => {
                // Ensure width is numeric if present
                const w = (typeof col.width === 'number' && col.width > 0) ? (col.width|0) : undefined;
                return utils.pick(Object.assign({}, col, (w ? { width: w } : {})), ["colId", "hide", "pinned", "width", "sort"]);
            });
        }
        return {
            columnState: state || [],
            filterModel: this.api?.getFilterModel?.() || [],
            sortModel: this.api?.sortController?.getSortModel?.() || []
        };
    }

    async loadGridState(state) {
        if (!this.api || !state) return;

        const prevAuto = this.api.getGridOption('autoSizeStrategy');
        const prevSuppress = this.api.getGridOption('suppressAutoSize');

        // Compute this BEFORE the try block so it's visible in finally.
        const hasExplicitWidths = Array.isArray(state.columnState) &&
            state.columnState.some(c => typeof c.width === 'number' && c.width > 0);

        try {
            this.api.setGridOption('suppressAutoSize', true);
            this.api.setGridOption('autoSizeStrategy', undefined);

            if (state.columnState && state.columnState.length) {
                this.api.applyColumnState({
                    state: state.columnState,
                    applyOrder: true,
                });
            }

            if (state.filterModel) {
                try { this.api.setFilterModel(state.filterModel); } catch {}
            }
            if (state.sortModel && Array.isArray(state.sortModel)) {
                try { this.api.setSortModel(state.sortModel); } catch {}
            }

            const widths = [];
            const st = Array.isArray(state.columnState) ? state.columnState : [];
            for (let i = 0; i < st.length; i++) {
                const c = st[i];
                if (!c || !c.colId) continue;
                if (typeof c.width === 'number' && c.width > 0) {
                    widths.push({ key: c.colId, newWidth: c.width | 0 });
                }
            }
            if (widths.length) {
                try { this.api.setColumnWidths(widths, false); } catch {}
            }

            // Suppress autosize BEFORE refresh to prevent width clobbering.
            if (hasExplicitWidths) {
                try { this.api.setGridOption('autoSizeStrategy', undefined); } catch {}
                try { this.api.setGridOption('suppressAutoSize', true); } catch {}
            }

            try { this.api.refreshHeader(); } catch {}
            try { this.api.refreshServerSide({ purge: false }); } catch {}
            try { this.grid$.set('presetLoaded', true); } catch {}

        } finally {
            if (!hasExplicitWidths) {
                try { this.api.setGridOption('autoSizeStrategy', prevAuto); } catch {}
                try { this.api.setGridOption('suppressAutoSize', prevSuppress); } catch {}
            }
        }
    }

    addColumns(colDefs = [], { show = false } = {}) {
        if (!Array.isArray(colDefs) || !this.api) return;

        this.columnRegistry.registerBatchColumns(colDefs);
        this.engine.registerColumnsFromDefs?.(colDefs);

        if (this.filterManager) colDefs = colDefs.map(col=>{
            if (col?.context?.dataType) return this.filterManager.configureFilter(col);
            return col;
        });

        const current = this.api.getGridOption('columnDefs') || [];
        const ids = new Set(current.map(c => c.colId || c.field));
        const newDefs = [];

        for (const d of colDefs) {
            const id = d.colId || d.field;
            if (!id) continue;
            if (!ids.has(id)) newDefs.push(d);
        }

        if (newDefs.length === 0) return;

        // Save existing column state so setGridOption('columnDefs') doesn't reset it
        const savedState = this.api.getColumnState?.();

        const merged = current.concat(newDefs);
        this.api.setGridOption('columnDefs', merged);

        // Restore previous column state — new columns stay hidden by default
        if (savedState && savedState.length) {
            this.api.applyColumnState({ state: savedState, applyOrder: true });
        }

        this.api.dispatchEvent?.({ type: 'newColumnsLoaded' });

        if (show) {
            this.setColumnsVisible(newDefs.map(d => d.colId || d.field), true);
        }
    }

    clearFilters() {
        try { this?.api?.setFilterModel({}); } catch {}
    }

    _buildAgColumnDefs(userDefs) {
        const engine = this.engine;
        const adapter = this;
        const normalize = (d) => {
            const out = Object.assign({}, d);
            if (!out.colId && out.field) out.colId = out.field;
            if (!out.context) out.context = {};
            out.context.showInfo = true;
            if (!out.icons) out.icons = {};
            let filterName = out.headerName;
            if (filterName == null) {
                filterName = out.field;
            } else {
                filterName += ` (${out.field})`
            }
            if (!out.icons.filter) out.icons.filter = `<span>${filterName}</span></div>` //`<div class="ag-filter-lottie-wrapper"><lord-icon src="/assets/lottie/filter.json" trigger="in" class="ag-filter-lottie-icon current-color"></lord-icon><span>${filterName}</span></div>`
            const ctx = out.context;
            ctx.idProperty = engine._idProperty;
            const cs = out.cellClass || (ctx.cellClass);
            if (typeof cs == 'function') {
                out.cellClass = (params) => { return cs(params, engine, []); }
            }
            return out;
        };
        return userDefs.map(normalize);
    }

    _updateProjection() {
        if (!this.api || typeof this.api.getAllDisplayedColumns !== 'function') return;

        try {
            const cols = this.api.getAllDisplayedColumns();
            const names = new Array(cols.length);

            for (let i = 0; i < cols.length; i++) {
                names[i] = cols[i].getColId();
            }

            this._projection = names;

            if (this.engine && typeof this.engine.setActiveColumnsFromProjection === 'function') {
                this.engine.setActiveColumnsFromProjection(names);
            }
        } catch (e) {
            console.error(e)
            // keep old projection on error
        }
    }

    _evaluateQuickSearch(my_idx) {
        this.quickSearch = null;
        let idx = my_idx;
        if (this?.context?.page?.quickSearch$) {
            const quickSearch = this.context.page.quickSearch$.get('quickSearch');
            const colsForSearch = this.searchManager.colsForSearch({includeVisible:false});
            if (quickSearch && quickSearch.trim()) {
                idx = this.searchManager.filter(idx, colsForSearch, quickSearch);
                this.quickSearch = quickSearch
            }
        }
        return idx
    }

    splitFilterModel(filterModel) {
        const derivedFn = new Map();
        const realFilters = {}
        Object.keys(filterModel).forEach(col => {
            if (this.engine._isDerived(col)) {
                const dfn = this.columnRegistry.getColumnDef(col)?.context?.compute;
                if (dfn) derivedFn.set(col, dfn);
            } else {
                realFilters[col] = filterModel[col]
            }
        })

        this.grid$.set('filterModel', realFilters);
        this.grid$.set('derivedModel', derivedFn);
    }

    _makeDatasource() {
        const adapter = this;
        return {getRows: (params) => adapter.getRows(params)}
    }

    normalize_filter(v) {
        if (v == null) return 'null';
        const vl = v.toString().toLowerCase().trim();
        if (vl === '') return 'null';
        if (vl === '(blanks)') return 'null';
        if (vl === '*missing*') return 'null'
        if (vl === 'null') return 'null';
        if (v === CLEAR_SENTINEL) return 'CLEAR'
        return v
    }

    buildPredicate(col, model) {

        const engine = this.engine;
        const MS_PER_DAY = 86400000;
        const sameDay = (aMs, bMs) => {
            if (!Number.isFinite(aMs) || !Number.isFinite(bMs)) return false;
            return Math.floor(aMs / MS_PER_DAY) === Math.floor(bMs / MS_PER_DAY);
        };

        if (!model || typeof model !== 'object') return null;
        const ft = String(model.filterType || '').toLowerCase();

        const colDef = this.api.getColumnDef(col);
        const vp = colDef?.valueParser;

        if (model.operator && (model.condition1 || model.condition2)) {
            const a = this.buildPredicate(col, model.condition1);
            const b = this.buildPredicate(col, model.condition2);
            if (!a && !b) return null;
            return String(model.operator).toUpperCase() === 'OR'
                ? (ri) => (a && a(ri)) || (b && b(ri))
                : (ri) => (!a || a(ri)) && (!b || b(ri));
        }

        if (ft === 'text' || ft === 'agtextcolumnfilter') {
            const type = String(model.type || 'contains').toLowerCase();
            const q = String(model.filter ?? '').toLowerCase();
            if (type === 'blank') return (ri) => { const v = engine.getCell(ri, col); return v == null || v === ''; };
            if (type === 'notblank') return (ri) => { const v = engine.getCell(ri, col); return !(v == null || v === ''); };
            if (!q) return null;
            const get = (ri) => String(engine.getCell(ri, col) ?? '').toLowerCase();
            if (type === 'equals') return (ri) => get(ri) === q;
            if (type === 'notequals') return (ri) => get(ri) !== q;
            if (type === 'startswith') return (ri) => get(ri).startsWith(q);
            if (type === 'endswith') return (ri) => get(ri).endsWith(q);
            if (type === 'notcontains') return (ri) => get(ri).indexOf(q) === -1;
            return (ri) => get(ri).indexOf(q) !== -1;
        }

        if (ft === 'set' || ft === 'agsetcolumnfilter') {
            let vals = Array.isArray(model.values) ? new Set(model.values.map(String)) : null;
            if (vals == null) return null;
            vals = new Set(Array.from(vals).map(v => this.normalize_filter(v)));
            if (!vals || vals.size === 0) return null;
            const hasNull = vals.has('null');
            return (ri) => {
                let mv = engine.getCell(ri, col);
                if (vp) mv = vp({ 'value': mv });
                if (mv == null) return hasNull;
                return vals.has(String(mv));
            };
        }

        if (ft === 'date' || ft === 'agdatecolumnfilter') {
            const type = String(model.type || 'equals').toLowerCase();
            const toMs = (x) => (x instanceof Date) ? x.getTime() : (x ? new Date(x).getTime() : NaN);
            const v1 = toMs(model.dateFrom), v2 = toMs(model.dateTo);
            const getMs = (ri) => {
                const v = engine.getCell(ri, col);
                if (v == null) return NaN;
                if (v instanceof Date) return v.getTime();
                const n = Number(v);
                return Number.isFinite(n) && n > 0 ? n : toMs(v);
            };
            if (type === 'equals')      return (ri) => { const t = getMs(ri); return Number.isFinite(t) && sameDay(t, v1); };
            if (type === 'inrange')     return (ri) => { const t = getMs(ri); return Number.isFinite(t) && t >= v1 && t <= v2; };
            if (type === 'lessthan')    return (ri) => { const t = getMs(ri); return Number.isFinite(t) && t < v1; };
            if (type === 'greaterthan') return (ri) => { const t = getMs(ri); return Number.isFinite(t) && t > v1; };
            return null;
        }

        if (ft === 'number' || ft === 'agnumbercolumnfilter') {
            const type = String(model.type || 'equals').toLowerCase();
            const f = coerceToNumber(model.filter), t = coerceToNumber(model.filterTo);
            const get = (ri) => { const v = engine.getCell(ri, col); const n = (typeof v === 'number') ? v : Number(v); return Number.isFinite(n) ? n : NaN; };
            if (type === 'equals')                return (ri) => get(ri) === f;
            if (type === 'notequals')             return (ri) => get(ri) !== f;
            if (type === 'lessthan')              return (ri) => get(ri) < f;
            if (type === 'lessthanorequal')       return (ri) => get(ri) <= f;
            if (type === 'greaterthan')           return (ri) => get(ri) > f;
            if (type === 'greaterthanorequal')    return (ri) => get(ri) >= f;
            if (type === 'inrange')               return (ri) => { const x = get(ri); return x >= f && x <= t; };
            if (type === 'blank')                 return (ri) => { const v = engine.getCell(ri, col); return v == null || v === ''; };    // 0 is NOT blank
            if (type === 'notblank')              return (ri) => { const v = engine.getCell(ri, col); return !(v == null || v === ''); };
            return null;
        }
        return null;
    };

    _getOrBuildFullIndex(n) {
        if (this._fullIdx && this._fullIdx.length === n) return this._fullIdx;
        const a = new Int32Array(n);
        for (let i = 0; i < n; i++) a[i] = i;
        this._fullIdx = a;
        return a;
    }

    evaluateFilterModel(filterModel) {
        const engine = this.engine;
        const keys = Object.keys(filterModel || {});
        const n = engine.numRows() | 0;

        if (!keys.length || n === 0) return this._getOrBuildFullIndex(n);

        const intersect = (a, b) => {
            const na = a?.length|0, nb = b?.length|0;
            if (!na || !nb) return new Int32Array(0);
            const out = new Int32Array(Math.min(na, nb));
            let i=0, j=0, k=0;
            while (i<na && j<nb) {
                const x=a[i]|0, y=b[j]|0;
                if (x===y) { out[k++]=x; i++; j++; }
                else if (x<y) i++; else j++;
            }
            return out.subarray(0, k);
        };

        // 1) Attempt min/max pushdown only for numeric filters (if indexes are enabled)
        let candidate = null;
        for (let i = 0; i < keys.length; i++) {
            const col = keys[i]; const m = filterModel[col] || {};
            const ft = String(m.filterType || m.type || '').toLowerCase();
            const type = String(m.type || '').toLowerCase();
            if (ft === 'number' || ft === 'agnumbercolumnfilter') {
                const map = { lessthan:'<', lessthanorequal:'<=', greaterthan:'>', greaterthanorequal:'>=', equals:'==', inrange:'between' };
                const op = map[type];
                if (op) {
                    let next = null;
                    if (op === 'between') {
                        const lo = coerceToNumber(m.filter), hi = coerceToNumber(m.filterTo);
                        if (lo != null && hi != null) next = engine.queryRange({ column: col, op: 'between', lo, hi });
                    } else {
                        const v = coerceToNumber(m.filter);
                        if (v != null) next = engine.queryRange({ column: col, op, value: v });
                    }
                    if (next && next.length) {
                        candidate = candidate ? intersect(candidate, next) : next;
                        if (candidate.length === 0) return candidate;
                    }
                }
            }
        }

        // 2) Remaining predicates (text/set/date/number cases not handled by pushdown)
        const preds = [];
        for (let i = 0; i < keys.length; i++) {
            const p = this.buildPredicate(keys[i], filterModel[keys[i]]);
            if (p) preds.push(p);
        }
        if (!preds.length) return candidate || fullIndex(n);

        const base = candidate || fullIndex(n);
        const kept = new Int32Array(base.length); let k = 0;
        for (let i = 0; i < base.length; i++) {
            const ri = base[i] | 0;
            let ok = true;
            for (let j = 0; j < preds.length; j++) { if (!preds[j](ri)) { ok = false; break; } }
            if (ok) kept[k++] = ri;
        }
        return kept.subarray(0, k);
    };

    _getRows(params) {
        try {
            const engine = this.engine
            const req = params.request || {};
            const startRow = req.startRow | 0;
            const endRow   = req.endRow   | 0;
            const sortModel   = Array.isArray(req.sortModel) ? req.sortModel : [];
            const filterModel = req.filterModel || {};

            // Filter
            let idx = this.evaluateFilterModel(filterModel);
            idx = this._evaluateQuickSearch(idx);
            this.grid$.set('lastIdx', idx);
            this.splitFilterModel(filterModel);

            // Sort (engine-side)
            if (sortModel.length) {
                idx = engine.sortIndexByModel(sortModel, idx, this.columnRegistry.columns);
            }

            let total = idx.length | 0;
            const s = Math.max(0, Math.min(startRow, total));
            const e = Math.max(s, Math.min(endRow, total));
            const want = e - s;

            let rows = new Array(want);
            for (let i = 0; i < want; i++) {
                const src = idx[s + i];
                rows[i] = { [engine._idProperty]: engine.getRowIdByIndex(src), [this._idxProperty]: src };
            }

            this.viewportTotal  = total;
            this.viewportWindow = [s, e];
            this.viewportRows   = rows;

            params.success({ rowData: rows, rowCount: total});
            if (this.api.overlayService.isAlive()) {
                document.querySelector('.ag-overlay').style.opacity = '0 !important'
                setTimeout(()=>{
                    this.api.hideOverlay()
                },200)
            }
        } catch (err) {
            console.error('Datasource.getRows error:', err);
            params.fail();
        }
    }

    _applyCellRange(saved) {
        const api = this.api;
        if (saved && saved.length) {
            requestAnimationFrame(() => {
                for (const r of saved) {
                    api.addCellRange({
                        rowStartIndex: r.startRow?.rowIndex,
                        rowEndIndex:   r.endRow?.rowIndex,
                        columns: (r.columns || []).map(c => c.getColId ? c.getColId() : c)
                    });
                }
            });
        }
    }

    _buildIdxSignature(filterModel, quickSearch) {
        const keys = Object.keys(filterModel || {});
        keys.sort();
        let sig = quickSearch ? `q:${quickSearch}|` : 'q:|';
        for (let i = 0; i < keys.length; i++) {
            const k = keys[i];
            const m = filterModel[k];
            const t = m && typeof m === 'object' ? (m.filterType || m.type || '') : '';
            const f = m && typeof m === 'object' ? (m.filter ?? m.dateFrom ?? '') : (m ?? '');
            sig += `${k}:${t}:${String(f)}|`;
        }
        return sig;
    }

    _invalidateIdxCache() {
        this._idxCacheEpoch = (this._idxCacheEpoch | 0) + 1;
        this._idxCacheSig = '';
        this._idxCacheIdx = null;
    }

    /* _handleEpoch removed — was dead code (never registered as listener).
       Dependency expansion + cache clearing now handled by engine._emitEpochChange.
       Grid cell refresh handled by flush(). */

    _getCurrentSortCols() {
        try {
            const sm = this.api?.getColumnDefs ? (this.api?.getSortModel?.() || this.api?.sortController?.getSortModel?.() || []) : (this.api?.getSortModel?.() || []);
            const out = new Set();
            for (let i=0;i<sm.length;i++) {
                const m = sm[i] || {}; const col = m.colId ?? m.field ?? m.column ?? m.col;
                if (col) out.add(col);
            }
            return out;
        } catch { return new Set(); }
    }

    _resortIfSortKeyAffected(changedCols) {
        const api = this.api; if (!api) return false;
        const sortCols = this._getCurrentSortCols(); if (sortCols.size === 0) return false;
        if (changedCols === true) { api.refreshServerSide?.({ purge: false }); return true; }
        for (const c of (changedCols || [])) { if (sortCols.has(c)) { api.refreshServerSide?.({ purge: false }); return true; } }
        return false;
    }

    _softRefresh() {
        const api = this.api; if (!api) return false;
        try { api.refreshServerSide({ purge: false }); } catch {}
    }

    _hardRefresh() {
        const api = this.api; if (!api) return false;
        try { api.refreshServerSide({ purge: true }); } catch {}
    }

    markCell(rowIndex, colId) {
        this.dirty.rows.add(rowIndex|0);
        this.dirty.cols.add(colId);
        this._scheduleFlush();
    }

    markColumn(colId) { this.dirty.cols.add(colId); this._scheduleFlush(); }
    markRow(rowIndex) { this.dirty.rows.add(rowIndex|0); this._scheduleFlush(); }

    flush({ suppressFlash = 'auto' } = {}) {
        this._flushScheduled = false;
        const api = this.api;
        const engine = this.engine;
        if (!api || !engine) return;

        const rowNodes = this.dirty.rows.size
            ? Array.from(this.dirty.rows, r => {
                try { return api.getRowNode(engine.getRowIdByIndex(r)); } catch { return null; }
            }).filter(Boolean)
            : undefined;

        let columns;
        if (this.dirty.cols.size) {
            // getDependentsClosure returns a FLAT string array, not nested.
            // Expand dirty cols to include all transitive dependents.
            const expanded = engine.getDependentsClosure(Array.from(this.dirty.cols));
            const colSet = this.dirty.cols;
            for (let i = 0; i < expanded.length; i++) colSet.add(expanded[i]);
            columns = Array.from(colSet);
        }

        suppressFlash = (typeof rowNodes === 'undefined') || (columns && columns.length > 5);

        if (columns) {
            engine._clearCellCaches(columns);
            const dirtyDerived = columns.filter(c => engine._isDerived(c));
            if (dirtyDerived.length) engine._notifyDerivedDirty(dirtyDerived);
            // Note: _notifyColumnChanged must exist on your engine; if it
            // doesn't, remove this line — the original code called it but
            // it's not defined in the provided source.
            if (typeof engine._notifyColumnChanged === 'function') {
                engine._notifyColumnChanged(columns);
            }
        }

        api.refreshCells?.({ rowNodes, columns, force: true, suppressFlash });

        this.dirty.rows.clear();
        this.dirty.cols.clear();
        if (this.dirty.cells) this.dirty.cells.clear();
    }

    _scheduleFlush() {
        if (this._flushScheduled) return;
        this._flushScheduled = true;
        const run = () => {
            this._flushScheduled = false;
            try { this.flush({ suppressFlash: 'auto' }); } catch (e) { console.error(e); }
        };
        if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
        else setTimeout(run, 0);
    }

    /* -------------------------- transactions  ----------------------- */
    async applyServerTransaction(payloads, opts = {}) {
        const api = this.api;
        const engine = this.engine;
        if (!api || !engine) {
            const addLen = (payloads && Array.isArray(payloads.add)) ? payloads.add.length : (payloads?.add ? 1 : 0);
            const updLen = (payloads && Array.isArray(payloads.update)) ? payloads.update.length : (payloads?.update ? 1 : 0);
            const remLen = (payloads && Array.isArray(payloads.remove)) ? payloads.remove.length : (payloads?.remove ? 1 : 0);
            return {
                applied: 0,
                skipped: addLen + updLen + remLen,
                breakdown: { add:{applied:0,skipped:addLen}, update:{applied:0,skipped:updLen}, remove:{applied:0,skipped:remLen} }
            };
        }

        const {
            commit = false,
            includeAllColumns = false,
            freezeDerived = true,
            refresh = 'auto',
            clearSearchCache = true,
            emitAsEdit = true
        } = opts;

        if (clearSearchCache) {
            this.searchManager.clearCache();
        }

        // ---- Tunables (keep fast defaults) ----
        const SANITIZE_DEEP = false;     // set true to walk arrays/objects
        const SANITIZE_MAX_DEPTH = 2;    // only used if SANITIZE_DEEP === true
        const SANITIZE_FINITE = true;   // set true to also coerce ±Infinity → null

        // ---- Sanitizers (hot path friendly) ----
        const sanitizeScalar = (v) => {
            if (typeof v === 'number') {
                if (Number.isNaN(v)) return null;
                if (SANITIZE_FINITE && !Number.isFinite(v)) return null;
            }
            return v;
        };
        const sanitizeDeep = (v, depth, seen) => {
            // Only engaged if SANITIZE_DEEP is true; otherwise callers use sanitizeScalar
            if (depth > SANITIZE_MAX_DEPTH) return v;
            const s = sanitizeScalar(v);
            if (s !== v) return s;
            if (!SANITIZE_DEEP) return v;
            if (!v || typeof v !== 'object') return v;

            if (!seen) seen = new Set();
            if (seen.has(v)) return v;
            seen.add(v);

            if (Array.isArray(v)) {
                for (let i = 0; i < v.length; i++) {
                    const nv = sanitizeDeep(v[i], depth + 1, seen);
                    if (nv !== v[i]) v[i] = nv;
                }
                return v;
            }
            const keys = Object.keys(v);
            for (let i = 0; i < keys.length; i++) {
                const k = keys[i];
                const nv = sanitizeDeep(v[k], depth + 1, seen);
                if (nv !== v[k]) v[k] = nv;
            }
            return v;
        };
        const sanitizeVal = (v) => SANITIZE_DEEP ? sanitizeDeep(v, 0, null) : sanitizeScalar(v);

        const idKey = engine._idProperty || '__rowId';
        const toArray = (x) => Array.isArray(x) ? x : (x ? [x] : []);
        const addListRaw = toArray(payloads?.add);
        const updListRaw = toArray(payloads?.update);
        const remListRaw = toArray(payloads?.remove);
        const impactedCols = new Set();

        /* ----------------------------- normalize ids ----------------------------- */
        // Drop NaN ids instead of stringifying them to 'NaN'
        const remIdsSet = (function dedupeRemIds(arr){
            if (!arr.length) return new Set();
            const s = new Set();
            for (let i = 0; i < arr.length; i++) {
                const v = arr[i];
                const idRaw = (v && typeof v === 'object') ? v[idKey] : v;
                const idSan = (typeof idRaw === 'number' && Number.isNaN(idRaw)) ? null : idRaw;
                if (idSan != null) s.add(String(idSan));
            }
            return s;
        })(remListRaw);

        /* ------------------------------ preprocess add --------------------------- */
        // Inline sanitize while scanning keys; no extra pass. Also collect derived stubs.
        const isDerived = engine._isDerived.bind(engine);
        const addDerivedStubs = new Array(addListRaw.length);
        for (let i = 0; i < addListRaw.length; i++) {
            const row = addListRaw[i];
            const stub = Object.create(null);
            let has = false;
            if (row && typeof row === 'object') {
                const keys = Object.keys(row);
                for (let k = 0; k < keys.length; k++) {
                    const col = keys[k];
                    if (col === idKey || col === this._idxProperty) continue;
                    let val = row[col];
                    val = sanitizeVal(val);
                    if (val !== row[col]) row[col] = val; // mutate in place
                    if (isDerived(col)) { stub[col] = val; has = true; }
                }
            }
            addDerivedStubs[i] = has ? stub : null;
        }

        /* --------------------------------- remove -------------------------------- */
        let appliedRem = 0, skippedRem = 0;
        if (remIdsSet.size) {
            try {
                const remIds = Array.from(remIdsSet);
                // console.log('REMOVING:', remIdsSet)
                const { removedIndices } = engine.removeRowsById(remIds);

                appliedRem = removedIndices.length | 0;
                skippedRem = remIds.length - appliedRem;
            } catch (e) {
                console.error('removeRowsById failed:', e);
                skippedRem = remIdsSet.size;
            }
        }

        /* ---------------------------------- add ---------------------------------- */
        let appliedAdd = 0, skippedAdd = 0;
        let postAddUpdates = [];
        if (addListRaw.length) {
            try {
                const { addedIndices, addedIds } = engine.addRows(addListRaw);
                appliedAdd = addedIndices.length | 0;

                // Derived overlays (already sanitized above)
                if (appliedAdd) {
                    const out = [];
                    for (let i = 0; i < addListRaw.length; i++) {
                        const stub = addDerivedStubs[i];
                        if (!stub) continue;
                        const id = (addListRaw[i] && addListRaw[i][idKey] != null)
                            ? String(addListRaw[i][idKey])
                            : String(addedIds[i]);
                        out.push(Object.assign({ [idKey]: id }, stub));
                    }
                    postAddUpdates = out;
                }
                skippedAdd = addListRaw.length - appliedAdd;
            } catch (e) {
                console.error('addRows failed:', e);
                skippedAdd = addListRaw.length;
                postAddUpdates = [];
            }
        }

        /* --------------------------------- update -------------------------------- */
        // Sanitize per-field while folding; skip NaN ids.
        const mergedUpdatesById = new Map();
        const ingestUpdate = (src) => {
            if (!src || typeof src !== 'object') return;

            const rowIdRaw = src[idKey];
            if (rowIdRaw == null || (typeof rowIdRaw === 'number' && Number.isNaN(rowIdRaw))) return;

            const rowId = String(rowIdRaw);
            if (remIdsSet.has(rowId)) return;

            let dst = mergedUpdatesById.get(rowId);
            if (!dst) {
                dst = { [idKey]: rowId };
                mergedUpdatesById.set(rowId, dst);
            }

            const keys = Object.keys(src);
            for (let k = 0; k < keys.length; k++) {
                const col = keys[k];
                if (col === idKey || col === this._idxProperty) continue;

                let v = sanitizeVal(src[col]);

                if (v !== undefined) {
                    dst[col] = v;
                    impactedCols.add(col);
                }
            }
        };

        for (let i = 0; i < updListRaw.length; i++) ingestUpdate(updListRaw[i]);
        for (let i = 0; i < postAddUpdates.length; i++) ingestUpdate(postAddUpdates[i]);

        const mergedUpdates = mergedUpdatesById.size ? Array.from(mergedUpdatesById.values()) : [];
        let appliedUpd = 0, skippedUpd = 0;
        if (mergedUpdates.length) {
            const r = await this.applyServerUpdateTransaction(mergedUpdates, {
                includeAllColumns,
                commit,
                freezeDerived,
                emitAsEdit:emitAsEdit
            });
            appliedUpd = r.applied | 0;
            skippedUpd = r.skipped | 0;
        }

        const sortCols = this._getCurrentSortCols();
        let sortChanged = false;
        for (const c of impactedCols) {
            if (sortCols.has(c)) { sortChanged = true; break; }
        }

        /* ---------------------------- single grid refresh ------------------------ */
        const needRefresh = (refresh === 'always' || refresh === true) ||
            (refresh === 'auto' && (appliedAdd || appliedRem || skippedAdd || skippedRem)) || (sortChanged);
        if (needRefresh) {
            try { api.refreshServerSide?.({ purge: false }); } catch {}
        }

        return {
            applied: (appliedAdd + appliedUpd + appliedRem),
            skipped: (skippedAdd + skippedUpd + skippedRem),
            breakdown: {
                add:    { applied: appliedAdd, skipped: skippedAdd },
                update: { applied: appliedUpd, skipped: skippedUpd },
                remove: { applied: appliedRem, skipped: skippedRem }
            }
        };
    }

    getNodeByIndex(rowIndex) {
        if (!this?.api) return;
        const id = this.engine.getRowIdByIndex(rowIndex);
        return this.api.getRowNode(id)
    }

    async applyServerUpdateTransaction(updates, opts = {}) {
        const api = this.api;
        const engine = this.engine;
        if (!api || !engine) {
            const skipped = Array.isArray(updates) ? updates.length : (updates ? 1 : 0);
            return { applied: 0, skipped };
        }

        const {
            includeAllColumns = false,
            commit = false,
            freezeDerived = true,
            emitAsEdit = false,
            checkRedraw = true,
            callback = null
        } = opts;

        const list = Array.isArray(updates) ? updates : (updates ? [updates] : []);
        if (list.length === 0) return { applied: 0, skipped: 0 };

        const idKey = engine._idProperty || '__rowId';

        const changedCols = new Set();
        const changedRowsSet = new Set();
        const physicalColsChanged = new Set();

        let applied = 0, skipped = 0;

        // First pass: validate and fold
        const rowWorkMap = new Map();
        for (let i = 0; i < list.length; i++) {
            const payload = list[i];
            if (!payload || typeof payload !== 'object') { skipped++; continue; }
            const rowIdRaw = payload[idKey];
            if (rowIdRaw == null) { skipped++; continue; }
            const rowId = String(rowIdRaw);
            const rowIndex = engine.getRowIndexById(rowId);
            if (rowIndex < 0) { skipped++; continue; }
            let arr = rowWorkMap.get(rowIndex);
            if (!arr) { arr = []; rowWorkMap.set(rowIndex, arr); }
            arr.push(payload);
        }
        if (rowWorkMap.size === 0) return { applied: 0, skipped };

        const overlayCache = new Map();
        const ensureOv = (col) => {
            let ov = overlayCache.get(col);
            if (ov === undefined) {
                ov = engine._ensureOverlay(col);
                overlayCache.set(col, ov || null);
            }
            return ov;
        };

        const tracker = { changedCols, changedRows: changedRowsSet, physicalCols: physicalColsChanged };

        // ─── Track whether we need to broadcast edits to the server ───
        let broadcastedEdits = false;

        for (const [rowIndex, payloadsForRow] of rowWorkMap.entries()) {
            const ri = rowIndex | 0;
            const merged = Object.create(null);
            merged[idKey] = engine.getRowIdByIndex(ri);

            for (let p = 0; p < payloadsForRow.length; p++) {
                const src = payloadsForRow[p];
                const keys = Object.keys(src);
                for (let k = 0; k < keys.length; k++) {
                    const col = keys[k];
                    if (col === idKey || col === this._idxProperty) continue;
                    merged[col] = src[col];
                }
            }

            const cols = Object.keys(merged);
            for (let c = 0; c < cols.length; c++) {
                const col = cols[c];
                if (col === idKey) continue;
                const val = merged[col];

                // DERIVED column path (unchanged)
                if (engine._derived.has(col)) {
                    const def = engine._derived.get(col);
                    const inv = def && typeof def.inverse === 'function' ? def.inverse : null;

                    if (inv) {
                        let plan = null;
                        try {
                            const settingsFn = engine._settingsGetter ? engine._settingsGetter.get(col) : null;
                            plan = inv(ri, val, engine, settingsFn ? settingsFn() : undefined);
                            if (plan && typeof plan.then === 'function') plan = await plan;
                        } catch { plan = null; }
                        await engine._applyEditPlanBatch(ri, plan, tracker, {
                            deferUnfreeze: true,
                            // When emitAsEdit is true, let _applyEditPlanBatch also
                            // enqueue to the coalescer for physical columns in the plan
                            suppressEditSignals: !emitAsEdit,
                        });
                    } else {
                        const ov = ensureOv(col);
                        if (ov) engine._overlaySet(ov, ri, val);
                        if (freezeDerived) engine._markFrozen(col, ri);
                        changedCols.add(col);
                        changedRowsSet.add(ri);
                    }
                    continue;
                }

                // ─── PHYSICAL column path: set overlay + optionally broadcast ───
                const ov = ensureOv(col);
                if (ov) engine._overlaySet(ov, ri, val);
                physicalColsChanged.add(col);
                changedCols.add(col);
                changedRowsSet.add(ri);

                // *** THIS IS THE FIX ***
                // When emitAsEdit is true, enqueue to the edit coalescer so the
                // 'cell-edit' event fires → setupEmitter handler → publishMessage
                if (emitAsEdit) {
                    try {
                        engine._editCoalescer.silent(ri, col, val, engine);
                        broadcastedEdits = true;
                    } catch {}
                }
            }

            applied++;
        }

        // ─── Flush edit signals to the emitter if we enqueued any ───
        if (broadcastedEdits) {
            engine._ensureEditDrainScheduled();
        }

        // ── BATCH UNFREEZE ──
        if (physicalColsChanged.size > 0) {
            engine._batchUnfreezeDependents(physicalColsChanged, changedRowsSet);
        }

        // Build AG update objects
        const changedColsArr = Array.from(changedCols);
        const colsWanted = includeAllColumns ? engine._fieldNames : changedColsArr;
        const rowsForAg = [];
        if (changedRowsSet.size) {
            const it = changedRowsSet.values();
            const totalRows = changedRowsSet.size;
            for (let i = 0; i < totalRows; i++) {
                const ri = (it.next().value | 0);
                const obj = engine.getRowObject(ri, colsWanted);
                obj[this._idxProperty] = ri;
                obj[idKey] = engine.getRowIdByIndex(ri);
                rowsForAg.push(obj);
            }
        }
        if (rowsForAg.length) {
            await new Promise((resolve) => {
                try {
                    this.api.applyServerSideTransactionAsync({ update: rowsForAg }, () => resolve(true));
                } catch (_) { resolve(true); }
            });
        }

        if ((commit === true) || (commit === 'auto')) {
            try { await engine._materializeEdits({ commit, columns: changedCols.size ? changedCols : null }); } catch {}
        }

        // Emit coalesced epoch
        const colsArr = Array.from(changedCols);
        const rowsArr = (function toInt32(set) {
            const n = set.size | 0;
            const out = new Int32Array(n);
            let i = 0;
            for (const v of set) out[i++] = (v | 0);
            return out;
        })(changedRowsSet);

        try {
            engine._emitEpochChange({ rowsChanged: rowsArr, colsChanged: colsArr });
        } catch {
            try { for (let i = 0; i < rowsArr.length; i++) this.markRow(rowsArr[i]); } catch {}
        }

        // Refresh / redraw logic (unchanged)
        let requestRefresh = false;
        let redrawFlag = false;
        let redrawNodes;

        if (checkRedraw !== false) {
            try {
                for (let i = 0; i < colsArr.length; i++) {
                    if (this.columnRegistry.getColumnDef(colsArr[i])?.context?.triggerRedraw) { redrawFlag = true; break; }
                }
                if (redrawFlag) {
                    const nodes = [];
                    for (let i = 0; i < rowsArr.length; i++) {
                        if (!this.pending_styles.has(rowsArr[i])) {
                            const node = this.getNodeByIndex(rowsArr[i]);
                            if (node) nodes.push(node);
                        } else {
                            this.pending_styles.delete(rowsArr[i]);
                        }
                    }
                    if (nodes.length) { redrawNodes = nodes; requestRefresh = true; }
                }
            } catch {}
        }

        try {
            const fm = api.getFilterModel?.() || {};
            const filterKeys = Object.keys(fm);
            let hit = false;
            for (let i = 0; i < filterKeys.length; i++) {
                if (changedCols.has(filterKeys[i])) { hit = true; break; }
            }
            if (hit) requestRefresh = true;
        } catch {}

        if (requestRefresh) {
            api.refreshServerSide?.({ purge: false });
        }

        if (redrawFlag && redrawNodes) {
            setTimeout(() => {
                if (this.api && !this.api.isDestroyed?.()) {
                    this.api.redrawRows({ rowNodes: redrawNodes });
                }
            }, 0);
        }

        if (callback) {
            await callback({ applied, skipped, updates });
        }

        return { applied, skipped };
    }

    removeAllRowsInGrid({ clearFilters = false } = {}) {
        if (!this.engine) return this;
        this.engine.removeAllRows();
        this._fullIdx = null;
        if (clearFilters) { try { this.api?.setFilterModel({}); } catch {} }
        this._syncCacheBlockToEngineSize();
        try { this.api?.refreshServerSide({ purge: false }); } catch {}
        return this;
    }

    replaceAllRowsInGrid(rowsOrTable, engineOpts = {}, { clearFilters = false, clearSearchCache = true} = {}) {
        if (!this.engine) return { previousRowCount: 0, nextRowCount: 0 };
        if (clearSearchCache) { try { this.searchManager.clearCache(); } catch {} }
        const result = this.engine.replaceAllRows(rowsOrTable, engineOpts);
        this._fullIdx = null;
        if (clearFilters) { try { this.api?.setFilterModel({}); } catch {} }
        this._syncCacheBlockToEngineSize();
        this._scheduleFlush();
        try { this.api?.refreshServerSide({ purge: false }); } catch {}
        if (result.nextRowCount !== result.previousRowCount) {
            this._notifyRowCount()
        }
        return result;
    }

    swapAllRows(newRowsOrTable, engineOpts = {}, uiOpts = {}) {
        return this.replaceAllRowsInGrid(newRowsOrTable, engineOpts, uiOpts);
    }

    rebuildIndexesAndRefresh(columns = null) {
        try { this.engine?.rebuildAllIndexes?.(columns); } catch {}
        try { this.api?.refreshServerSide({ purge: false }); } catch {}
        return this;
    }

    _syncCacheBlockToEngineSize() {
        try {
            const n = Math.max(1, this.engine?.numRows?.() | 0);
            this.api?.setGridOption?.('cacheBlockSize', n);
        } catch {}
    }

    /* -------------------------------- utils --------------------------------- */

    getFirstNonNullCell(column, startRow=0, endRow=null, direction=-1){
        const name = this.engine._normalizeColumnSelector(column)[0];
        const n = endRow ?? this.engine.numRows();
        if (direction === 1) {
            for (let i=(n-1); i>=startRow; i--) {
                const node = this.api.getDisplayedRowAtIndex(i);
                if (!node) continue;
                const id = node.id;
                const ri = this.engine.getRowIndexById(id);
                const v = this.engine.getCell(ri, name);
                if (v != null && v !== '') {
                    return {value:v, rowIndex:i, id:id, srcIndex: ri};
                }
            }
        } else {
            for (let i=startRow; i<n; i++) {
                const node = this.api.getDisplayedRowAtIndex(i);
                if (!node) continue;
                const id = node.id;
                const ri = this.engine.getRowIndexById(id);
                const v = this.engine.getCell(ri, name);
                if (v != null && v !== '') {
                    return {value:v, rowIndex:i, id:id, srcIndex: ri};
                }
            }
        }
        return null
    }

    getFirstNullCell(column, startRow=0, endRow=null, direction=-1){
        const name = this.engine._normalizeColumnSelector(column)[0];
        const n = endRow ?? this.engine.numRows();
        if (direction === 1) {
            for (let i=(n-1); i>=startRow; i--) {
                const node = this.api.getDisplayedRowAtIndex(i);
                if (!node) continue;
                const id = node.id;
                const ri = this.engine.getRowIndexById(id);
                const v = this.engine.getCell(ri, name);
                if (v == null || v === '') {
                    return {value:v, rowIndex:i, id:id, srcIndex: ri};
                }
            }
        } else {
            for (let i=startRow; i<n; i++) {
                const node = this.api.getDisplayedRowAtIndex(i);
                if (!node) continue;
                const id = node.id;
                const ri = this.engine.getRowIndexById(id);
                const v = this.engine.getCell(ri, name);
                if (v == null || v === '') {
                    return {value:v, rowIndex:i, id:id, srcIndex: ri};
                }
            }
        }
        return null
    }

    getNextNullyCell(column, startRow=0, endRow=null, nully=false, direction=1) {
        if (nully) return this.getFirstNullCell(column, startRow, endRow, direction)
        return this.getFirstNonNullCell(column, startRow, endRow, direction);
    }

    _guessWidthRaw(colId, defaultMin = 50, force = false) {
        try {
            const col = this.api.getColumn(colId);
            const colDef = col && col.colDef;
            if (colDef?.width && !force) {
                return colDef.width;
            }

            const mapVal = v => (colDef?.valueFormatter ? colDef.valueFormatter({ value: v }) : v);
            const n = this.engine.numRows() | 0;
            const min = colDef?.minWidth || defaultMin;
            if (n === 0) return min;

            // Sample up to 200 evenly-spaced rows instead of reading ALL values
            const sampleSize = Math.min(n, 200);
            const step = n / sampleSize;
            const widths = [];
            for (let s = 0; s < sampleSize; s++) {
                const ri = Math.min((s * step) | 0, n - 1);
                const raw = this.engine.getCell(ri, colId);
                if (raw == null) continue;
                const formatted = mapVal(raw);
                if (formatted == null) continue;
                const w = measureText(String(formatted)).width;
                if (w > 0) widths.push(w);
            }

            if (!widths.length) return min;

            const l = quantileSeq(widths, 0.25);
            const u = quantileSeq(widths, 0.75);
            if (widths.length) {
                let ww = widths.filter(w => (l <= w) && (w <= u));
                if (ww.length < 2) return median(widths) || defaultMin;
                return mean(ww) || defaultMin;
            }
            return defaultMin;
        } catch {
            return defaultMin;
        }
    }
}

export default ArrowEngine;
