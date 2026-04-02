


import { license } from '@/config/licenses.js';
import { LicenseManager, createGrid} from 'ag-grid-enterprise';
LicenseManager.setLicenseKey(license);
import {
    MICRO_GRID_CONFIGS, MICRO_GRID_GROUPS, SEVERITY_COLOR_MAP,
    _parseTags, seedKnownTags, getAllKnownTags, _createPillEl,
    detectMatchMode, getTagsByCategory, getTagCustomColors,
    setTagColor, removeTagColor,
} from './microGridConfigs.js';
import {installKeyCoalescer} from "@/grids/js/arrow/keyCoalescer.js";
import {v4 as uuidv4} from "uuid";
import {debounce, asArray} from '@/utils/helpers.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _generate_trace() {
    try {
        return `micro-${uuidv4()}`;
    } catch (e) {}
    return `micro-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function _shallowEqual(a, b) {
    if (a === b) return true;
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length) return false;
    for (const k of keysA) {
        if (a[k] !== b[k]) return false;
    }
    return true;
}

function _upper(s) { return (s || '').toUpperCase(); }
function _topic(microName) { return `MICRO.${_upper(microName)}`; }
function _escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function _timestamp() {
    return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

// Pattern matchers for bulk add
const ISIN_RE = /^[A-Z]{2}[A-Z0-9]{9}[0-9]$/;
const TICKER_RE = /^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$/;

function _extractIdentifiers(text) {
    const tokens = text.split(/[\n\r,;\t|]+/).map(s => s.trim()).filter(Boolean);
    const results = [];
    const seen = new Set();
    for (const raw of tokens) {
        const words = raw.split(/\s+/);
        for (const w of words) {
            // Support tag:TICKER syntax — extract tag prefix before cleaning
            let tag = '';
            let token = w;
            const colonIdx = token.indexOf(':');
            if (colonIdx > 0) {
                const maybeTag = token.slice(0, colonIdx);
                const maybeId = token.slice(colonIdx + 1);
                // Only treat as tag:ticker if the part after colon looks like a ticker/ISIN
                const cleanedId = maybeId.replace(/[^A-Za-z0-9.]/g, '').toUpperCase();
                if (cleanedId && (ISIN_RE.test(cleanedId) || TICKER_RE.test(cleanedId))) {
                    tag = maybeTag.toLowerCase();
                    token = maybeId;
                }
            }

            const cleaned = token.replace(/[^A-Za-z0-9.]/g, '');
            if (!cleaned) continue;
            const upper = cleaned.toUpperCase();
            if (seen.has(upper)) continue;
            if (upper.length === 12 && ISIN_RE.test(upper)) {
                seen.add(upper);
                results.push({ column: 'isin', pattern: upper, ...(tag ? { tags: tag } : {}) });
            } else if (TICKER_RE.test(upper) && upper.length >= 1) {
                seen.add(upper);
                results.push({ column: 'ticker', pattern: upper, ...(tag ? { tags: tag } : {}) });
            }
        }
    }
    return results;
}

// ---------------------------------------------------------------------------
// UndoStack — per micro-grid edit history
// ---------------------------------------------------------------------------

class UndoStack {
    constructor(maxSize = 50) {
        this._stack = [];
        this._redoStack = [];
        this._max = maxSize;
    }
    push(entry) {
        this._stack.push(entry);
        if (this._stack.length > this._max) this._stack.shift();
        this._redoStack.length = 0; // clear redo on new action
    }
    undo() {
        const entry = this._stack.pop();
        if (entry) this._redoStack.push(entry);
        return entry || null;
    }
    redo() {
        const entry = this._redoStack.pop();
        if (entry) this._stack.push(entry);
        return entry || null;
    }
    get canUndo() { return this._stack.length > 0; }
    get canRedo() { return this._redoStack.length > 0; }
    clear() { this._stack.length = 0; this._redoStack.length = 0; }
}

// ---------------------------------------------------------------------------
// ChangeLog — rolling activity feed
// ---------------------------------------------------------------------------

class ChangeLog {
    constructor(userManager, maxEntries = 100) {
        this._entries = [];
        this._max = maxEntries;
        this._listeners = new Set();
        this.userManager = userManager;
    }
    add(type, detail, user = null) {
        user = user || this.userManager.displayName;
        const entry = { time: _timestamp(), type, detail, user };
        this._entries.push(entry);
        if (this._entries.length > this._max) this._entries.shift();
        for (const fn of this._listeners) fn(entry);
    }
    get entries() { return this._entries; }
    onEntry(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
    clear() { this._entries.length = 0; }
}

// ---------------------------------------------------------------------------
// MicroGridManager
// ---------------------------------------------------------------------------

export class MicroGridManager {

    constructor(page) {
        this._page = page;
        this._data = new Map();
        this._gridApis = new Map();
        this._openModals = new Map();
        this._activeTabs = new Map();
        this._subscribed = new Set();
        this._abortControllers = new Map();
        this._handlers = new Map();
        const user = this._page.userManager()

        /** @type {Map<string, UndoStack>} */
        this._undoStacks = new Map();

        /** @type {ChangeLog} */
        this._changeLog = new ChangeLog(user);

        /** @type {Map<string, HTMLElement>} context menu element refs */
        this._contextMenus = new Map();

        /** @type {Map<string, HTMLElement>} grid container refs for overlays */
        this._gridContainers = new Map();

        /** Reconnect indicator elements per group. @type {Map<string, HTMLElement>} */
        this._reconnectIndicators = new Map();

        /** Track pending publishes for optimistic reconciliation. @type {number} */
        this._pendingPublishes = 0;

        /** @type {Map<string, Set<Function>>} external data-change listeners */
        this._dataListeners = new Map();
    }

    // =========================================================================
    // External data access
    // =========================================================================

    /** Get current rows for a micro-grid. */
    getData(microName) {
        return this._data.get(microName) || [];
    }

    /** Register a callback fired whenever a micro-grid's data changes (snapshot or delta). Returns disposer. */
    onDataChange(microName, fn) {
        if (!this._dataListeners.has(microName)) this._dataListeners.set(microName, new Set());
        this._dataListeners.get(microName).add(fn);
        return () => this._dataListeners.get(microName)?.delete(fn);
    }

    _emitDataChange(microName) {
        const fns = this._dataListeners.get(microName);
        if (!fns || !fns.size) return;
        const rows = this._data.get(microName) || [];
        for (const fn of fns) { try { fn(rows); } catch (e) { console.error('[MicroGrid] dataChange listener error:', e); } }
    }

    // =========================================================================
    // Selection helpers
    // =========================================================================

    /** Get selected rows, falling back to the focused/active row if nothing is checked. */
    _getEffectiveSelection(microName) {
        const api = this._gridApis.get(microName);
        if (!api) return [];
        const sel = api.getSelectedRows();
        if (sel.length > 0) return sel;
        // Fallback: use focused cell's row
        const focused = api.getFocusedCell();
        if (focused) {
            const rowNode = api.getDisplayedRowAtIndex(focused.rowIndex);
            if (rowNode?.data && !rowNode.data._phantom) return [rowNode.data];
        }
        return [];
    }

    // =========================================================================
    // Undo stack helpers
    // =========================================================================

    _getUndo(microName) {
        if (!this._undoStacks.has(microName)) this._undoStacks.set(microName, new UndoStack());
        return this._undoStacks.get(microName);
    }

    _recordUndo(microName, type, forward, inverse) {
        this._getUndo(microName).push({ type, forward, inverse });
    }

    async performUndo(microName) {
        const stack = this._getUndo(microName);
        const entry = stack.undo();
        if (!entry) return;
        await this._applyPayload(microName, entry.inverse, true);
    }

    async performRedo(microName) {
        const stack = this._getUndo(microName);
        const entry = stack.redo();
        if (!entry) return;
        await this._applyPayload(microName, entry.forward, true);
    }

    // =========================================================================
    // Subscription
    // =========================================================================

    /** Subscribe all grids in a group without opening the modal UI. */
    async subscribeGroup(groupNameOrConfig) {
        const groupConfig = typeof groupNameOrConfig === 'string'
            ? MICRO_GRID_GROUPS[groupNameOrConfig]
            : groupNameOrConfig;
        if (!groupConfig) return;
        for (const gridName of groupConfig.grids) {
            await this.subscribe(gridName);
        }
    }

    async subscribe(microName) {
        if (this._subscribed.has(microName)) return;

        const sm = this._page.subscriptionManager();
        const topic = _topic(microName);

        const handlers = {
            subscribe: this._onSnapshot.bind(this, microName),
            publish: this._onDelta.bind(this, microName),
        };
        this._handlers.set(microName, handlers);

        sm.registerMessageHandler(topic, 'micro_subscribe', handlers.subscribe);
        sm.registerMessageHandler(topic, 'micro_publish', handlers.publish);

        this._subscribed.add(microName);
        await sm.microSubscribe(microName);
    }

    async unsubscribe(microName) {
        if (!this._subscribed.has(microName)) return;

        const sm = this._page.subscriptionManager();
        const topic = _topic(microName);
        const handlers = this._handlers.get(microName);

        if (handlers) {
            sm.unregisterMessageHandler(topic, 'micro_subscribe', handlers.subscribe);
            sm.unregisterMessageHandler(topic, 'micro_publish', handlers.publish);
            this._handlers.delete(microName);
        }

        this._subscribed.delete(microName);
        await sm.microUnsubscribe(microName).catch(() => {});
    }

    /**
     * Resubscribe all currently-subscribed micro-grids (e.g. after reconnect).
     */
    async resubscribeAll() {
        const names = [...this._subscribed];
        // Clear subscription state so subscribe() runs again
        for (const name of names) {
            const sm = this._page.subscriptionManager();
            const topic = _topic(name);
            const handlers = this._handlers.get(name);
            if (handlers) {
                sm.unregisterMessageHandler(topic, 'micro_subscribe', handlers.subscribe);
                sm.unregisterMessageHandler(topic, 'micro_publish', handlers.publish);
                this._handlers.delete(name);
            }
            this._subscribed.delete(name);
        }
        // Re-subscribe — will get fresh snapshots
        for (const name of names) {
            await this.subscribe(name);
        }
        this._changeLog.add('system', 'Reconnected and resubscribed');
    }

    // =========================================================================
    // Message handlers
    // =========================================================================

    _onSnapshot(microName, message) {
        const snapshot = message?.data?.snapshot;
        if (!Array.isArray(snapshot)) return;

        this._data.set(microName, [...snapshot]);
        seedKnownTags(snapshot);
        this._refreshGrid(microName);
        this._emitDataChange(microName);
        this._hideReconnectIndicator();
        this._changeLog.add('snapshot', `${snapshot.length} rows loaded`, '');
    }

    _onDelta(microName, message) {
        const payloads = message?.data?.payloads;
        if (!payloads) return;

        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;

        const pkCol = config.primaryKeys[0];
        let rows = this._data.get(microName) || [];

        const removeRows = payloads.remove || [];
        if (removeRows.length > 0) {
            const removeIds = new Set(removeRows.map(r => String(r[pkCol])));
            rows = rows.filter(r => !removeIds.has(String(r[pkCol])));
            this._changeLog.add('remove', `${removeRows.length} row(s) removed`);
        }

        const addRows = payloads.add || [];
        if (addRows.length > 0) {
            // Deduplicate: remove any existing rows with the same PK (optimistic UI may have already added them)
            const addIds = new Set(addRows.map(r => String(r[pkCol])));
            rows = rows.filter(r => !addIds.has(String(r[pkCol])));
            rows = [...rows, ...addRows];
            seedKnownTags(addRows);
            const patterns = addRows.map(r => r.pattern || '?').join(', ');
            this._changeLog.add('add', `${addRows.length} added: ${patterns}`);
        }

        const updateRows = payloads.update || [];
        if (updateRows.length > 0) {
            const updateMap = new Map(updateRows.map(r => [String(r[pkCol]), r]));
            rows = rows.map(r => {
                const upd = updateMap.get(String(r[pkCol]));
                return upd ? { ...r, ...upd } : r;
            });
            seedKnownTags(updateRows);
            this._changeLog.add('update', `${updateRows.length} row(s) updated`);
        }

        this._data.set(microName, rows);
        this._refreshGrid(microName);
        this._emitDataChange(microName);
    }

    _refreshGrid(microName) {
        const api = this._gridApis.get(microName);
        if (!api) return;
        const rows = this._data.get(microName) || [];

        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) { api.setGridOption('rowData', rows); return; }
        const pkCol = config.primaryKeys[0];

        // Build a map of current grid rows for diffing
        const currentRows = new Map();
        try {
            api.forEachNode(node => {
                if (node.data && !node.data._phantom) {
                    currentRows.set(String(node.data[pkCol]), node.data);
                }
            });
        } catch (_) {
            // Grid not ready — fall back to full replace
            api.setGridOption('rowData', rows);
            return;
        }

        // If grid is empty, use full replace (faster for initial load)
        if (currentRows.size === 0 && rows.length > 0) {
            api.setGridOption('rowData', rows);
            return;
        }

        const newRowMap = new Map(rows.map(r => [String(r[pkCol]), r]));
        const toAdd = [];
        const toUpdate = [];
        const toRemove = [];

        for (const [id, row] of newRowMap) {
            const old = currentRows.get(id);
            if (!old) {
                toAdd.push(row);
            } else if (!_shallowEqual(old, row)) {
                toUpdate.push(row);
            }
        }
        for (const [id, row] of currentRows) {
            if (!newRowMap.has(id)) {
                toRemove.push(row);
            }
        }

        if (toAdd.length || toUpdate.length || toRemove.length) {
            api.applyTransaction({ add: toAdd, update: toUpdate, remove: toRemove });
        }
    }

    // =========================================================================
    // Publish edits — with optimistic UI + undo recording
    // =========================================================================

    /**
     * Core publish with optimistic local apply.
     * @param {string} microName
     * @param {object} payloads  - { add:[], update:[], remove:[] }
     * @param {boolean} skipUndo - true when replaying undo/redo
     * @param {boolean} skipRefresh - true when the grid cell already reflects the
     *     change (e.g. inside onCellValueChanged) so a full setGridOption('rowData')
     *     would race with ag-grid's edit lifecycle and clobber the value.
     */
    async _applyPayload(microName, payloads, skipUndo = false, skipRefresh = false) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const pkCol = config.primaryKeys[0];

        // --- Optimistic local apply ---
        let rows = [...(this._data.get(microName) || [])];

        if (payloads.add?.length) {
            rows = [...rows, ...payloads.add];
        }
        if (payloads.update?.length) {
            const updateMap = new Map(payloads.update.map(r => [String(r[pkCol]), r]));
            rows = rows.map(r => {
                const upd = updateMap.get(String(r[pkCol]));
                return upd ? { ...r, ...upd } : r;
            });
        }
        if (payloads.remove?.length) {
            const removeIds = new Set(payloads.remove.map(r => String(r[pkCol])));
            rows = rows.filter(r => !removeIds.has(String(r[pkCol])));
        }

        this._data.set(microName, rows);

        if (!skipRefresh) {
            this._refreshGrid(microName);
        }
        this._emitDataChange(microName);

        // --- Send to server ---
        this._pendingPublishes++;
        try {
            const sm = this._page.subscriptionManager();
            await sm.microPublish(microName, payloads);
        } catch (e) {
            console.error('[MicroGrid] Publish failed, server will reconcile on next delta:', e);
        } finally {
            this._pendingPublishes--;
        }
    }

    async publishAdd(microName, row) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const filled = config.addRowDefaults ? config.addRowDefaults(row) : row;
        // Auto-detect match_mode from pattern
        if (filled.pattern !== undefined) {
            filled.match_mode = detectMatchMode(filled.pattern);
        }
        const pkCol = config.primaryKeys[0];
        const inverse = { remove: [{ [pkCol]: filled[pkCol] }] };
        this._recordUndo(microName, 'add', { add: [filled] }, inverse);
        await this._applyPayload(microName, { add: [filled] });
    }

    async publishBulkAdd(microName, rowArr) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const pkCol = config.primaryKeys[0];
        const filled = rowArr.map(r => {
            const f = config.addRowDefaults ? config.addRowDefaults(r) : r;
            if (f.pattern !== undefined) f.match_mode = detectMatchMode(f.pattern);
            return f;
        });
        if (filled.length === 0) return;
        const inverse = { remove: filled.map(r => ({ [pkCol]: r[pkCol] })) };
        this._recordUndo(microName, 'bulk_add', { add: filled }, inverse);
        await this._applyPayload(microName, { add: filled });
    }

    async publishUpdate(microName, row, oldRow = null, { skipRefresh = false } = {}) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const pkCol = config.primaryKeys[0];
        // Auto-detect match_mode when pattern changes
        if (row.pattern !== undefined) {
            row.match_mode = detectMatchMode(row.pattern);
        }
        // Normalize tags to comma-separated string — cell editors may return
        // arrays, whitespace-padded strings, or other non-canonical formats.
        if (row.tags !== undefined) {
            if (Array.isArray(row.tags)) {
                row.tags = row.tags.map(t => t.trim()).filter(Boolean).join(',');
            } else {
                row.tags = _parseTags(row.tags).join(',');
            }
            seedKnownTags([row]);
        }
        if (oldRow) {
            const inverseRow = { [pkCol]: row[pkCol] };
            for (const key of Object.keys(row)) {
                if (key !== pkCol && oldRow[key] !== undefined) inverseRow[key] = oldRow[key];
            }
            this._recordUndo(microName, 'update', { update: [row] }, { update: [inverseRow] });
        }
        await this._applyPayload(microName, { update: [row] }, false, skipRefresh);
    }

    async publishRemove(microName, row) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const pkCol = config.primaryKeys[0];
        // Store full row for undo
        const fullRow = (this._data.get(microName) || []).find(r => String(r[pkCol]) === String(row[pkCol]));
        if (fullRow) {
            this._recordUndo(microName, 'remove', { remove: [{ [pkCol]: row[pkCol] }] }, { add: [{ ...fullRow }] });
        }
        await this._applyPayload(microName, { remove: [{ [pkCol]: row[pkCol] }] });
    }

    async publishBulkRemove(microName, rowArr) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const pkCol = config.primaryKeys[0];
        const allRows = this._data.get(microName) || [];
        const removeIds = new Set(rowArr.map(r => String(r[pkCol])));
        const fullRows = allRows.filter(r => removeIds.has(String(r[pkCol]))).map(r => ({ ...r }));
        if (fullRows.length > 0) {
            this._recordUndo(microName, 'bulk_remove',
                { remove: rowArr.map(r => ({ [pkCol]: r[pkCol] })) },
                { add: fullRows }
            );
        }
        await this._applyPayload(microName, { remove: rowArr.map(r => ({ [pkCol]: r[pkCol] })) });
    }

    async publishBulkUpdate(microName, rowArr) {
        if (rowArr.length === 0) return;
        await this._applyPayload(microName, { update: rowArr });
    }

    // =========================================================================
    // Duplicate rows
    // =========================================================================

    async duplicateRows(microName, rows) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;
        const pkCol = config.primaryKeys[0];
        const dupes = rows.map(r => {
            const d = { ...r };
            d[pkCol] = _generate_trace();
            delete d.updated_at;
            delete d.updated_by;
            return d;
        });
        if (dupes.length === 0) return;
        const inverse = { remove: dupes.map(r => ({ [pkCol]: r[pkCol] })) };
        this._recordUndo(microName, 'duplicate', { add: dupes }, inverse);
        await this._applyPayload(microName, { add: dupes });
    }

    // =========================================================================
    // Copy to clipboard
    // =========================================================================

    copyRowsToClipboard(microName, rows, btnEl = null) {
        if (!rows || rows.length === 0) return;
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;

        const visibleCols = config.columns
                                   .filter(c => c.field && !c.hide && c.field !== 'id')
                                   .map(c => c.field);

        const header = visibleCols.join('\t');
        const body = rows.map(r => visibleCols.map(c => r[c] ?? '').join('\t')).join('\n');
        const text = header + '\n' + body;

        navigator.clipboard.writeText(text).then(() => {
            this._changeLog.add('copy', `${rows.length} row(s) copied to clipboard`);
            if (btnEl) {
                const orig = btnEl.textContent;
                btnEl.textContent = '✓';
                setTimeout(() => { btnEl.textContent = orig; }, 1500);
            }
        }).catch(e => {
            console.error('[MicroGrid] Clipboard write failed:', e);
        });
    }

    // =========================================================================
    // Filter Grid — apply patterns to main grid
    // =========================================================================

    filterMainGrid(microName) {
        const rows = this._data.get(microName) || [];
        if (rows.length === 0) return;

        const api = this._page.ptGrid.api;
        const patterns = rows.map(r => ({
            column: r.column || 'ticker',
            pattern: r.pattern || '',
            match_mode: r.match_mode || 'literal',
        })).filter(p => p.pattern);

        if (page._gridWidget || page.gridApi) {
            this._applyColumnFilters(api, patterns);
        }
        return;
    }

    _applyColumnFilters(gridApi, patterns) {
        // Group by column target
        const byCol = new Map();
        for (const p of patterns) {
            if (!byCol.has(p.column)) byCol.set(p.column, []);
            byCol.get(p.column).push(p.pattern);
        }

        // Apply as a text filter with OR-combined values
        for (const [col, values] of byCol) {
            try {
                const filterInstance = gridApi.getFilterInstance?.(col);
                if (filterInstance) {
                    filterInstance.setModel({
                        type: 'set',
                        values: values,
                    });
                }
            } catch (e) {
                console.warn(`[MicroGrid] Could not apply filter to column "${col}":`, e);
            }
        }
        try {
            gridApi.onFilterChanged?.();
        } catch (e) { /* ignore */ }

        this._changeLog.add('filter', `Applied ${patterns.length} pattern(s) to main grid`);
    }

    // =========================================================================
    // Modal / Tab UI
    // =========================================================================

    async openGroup(groupNameOrConfig) {
        const groupConfig = typeof groupNameOrConfig === 'string'
            ? MICRO_GRID_GROUPS[groupNameOrConfig]
            : groupNameOrConfig;

        if (!groupConfig) {
            console.error('[MicroGrid] Unknown group:', groupNameOrConfig);
            return;
        }

        const groupName = groupConfig.name;
        const existing = this._openModals.get(groupName);
        if (existing && existing.open) return;

        for (const gridName of groupConfig.grids) {
            await this.subscribe(gridName);
        }

        const mm = this._page.modalManager();
        const mgr = this;

        return mm.showCustom({
            title: groupConfig.displayName,
            modalBoxClass: 'micro-grid-modal-box',
            preventBackdropClick: false,
            setupContent(contentArea, dialog, closeWithValue) {
                mgr._openModals.set(groupName, dialog);
                mgr._buildTabbedUI(groupConfig, contentArea, dialog, closeWithValue);
            },
        }).then(result => {
            this._cleanupGroup(groupName);
            return result;
        });
    }

    _buildTabbedUI(groupConfig, contentArea, dialog, closeWithValue) {
        const groupName = groupConfig.name;
        const grids = groupConfig.grids;
        const ac = new AbortController();
        this._abortControllers.set(groupName, ac);
        const signal = ac.signal;

        // --- Reconnect indicator ---
        const reconnectBar = document.createElement('div');
        reconnectBar.className = 'micro-reconnect-bar';
        reconnectBar.style.display = 'none';
        reconnectBar.innerHTML = '<span class="micro-reconnect-dot"></span> Reconnecting…';
        contentArea.appendChild(reconnectBar);
        this._reconnectIndicators.set(groupName, reconnectBar);

        // --- Tab bar ---
        const tabBar = document.createElement('div');
        tabBar.className = 'micro-tab-bar';

        grids.forEach((gridName, idx) => {
            const config = MICRO_GRID_CONFIGS[gridName];
            const tab = document.createElement('div');
            tab.className = 'micro-tab' + (idx === 0 ? ' active' : '');
            tab.innerHTML = config?.displayName || gridName;
            tab.dataset.grid = gridName;
            tab.addEventListener('click', () => {
                tabBar.querySelectorAll('.micro-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this._switchTab(groupName, gridName, gridContainer);
            }, { signal });
            tabBar.appendChild(tab);
        });

        const t = document.createElement('div');
        t.className = 'micro-top-row';


        t.appendChild(tabBar);

        // --- Search bar ---
        const searchRow = document.createElement('div');
        searchRow.className = 'micro-search-row';
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search pattern, tags, notes…';
        searchInput.className = 'micro-search-input';
        searchRow.addEventListener('click', () => {
            searchInput.focus();
        });

        searchInput.addEventListener('input', debounce(() => {
            const g = this._activeTabs.get(groupName);
            if (!g) return
            const api = this._gridApis.get(g);
            if (!api) return;
            if (!searchInput.value || (searchInput.value.trim() === '')) {
                searchInput.classList.remove('active');
                api.setGridOption('quickFilterText', null);
            } else {
                searchInput.classList.add('active');
                api.setGridOption('quickFilterText', searchInput.value);
            }

        }, 100));
        searchRow.appendChild(searchInput);
        t.appendChild(searchRow);
        contentArea.appendChild(t);

        // --- Grid container ---
        const gridContainer = document.createElement('div');
        gridContainer.className = 'micro-grid-container';
        gridContainer.style.position = 'relative';
        contentArea.appendChild(gridContainer);
        this._gridContainers.set(groupName, gridContainer);

        // --- Status bar ---
        const statusBar = document.createElement('div');
        statusBar.className = 'micro-status-bar';
        contentArea.appendChild(statusBar);
        this._statusBars = this._statusBars || new Map();
        this._statusBars.set(groupName, statusBar);

        // --- Toolbar ---
        const toolbar = document.createElement('div');
        toolbar.className = 'micro-toolbar';

        const _btn = (text, cls, handler) => {
            const b = document.createElement('button');
            b.className = `btn btn-sm ${cls}`;
            b.textContent = text;
            b.addEventListener('click', handler, { signal });
            return b;
        };
        const _sep = () => { const s = document.createElement('span'); s.className = 'toolbar-sep'; return s; };
        const _spacer = () => { const s = document.createElement('span'); s.className = 'toolbar-spacer'; return s; };

        const activeGrid = () => this._activeTabs.get(groupName);

        toolbar.appendChild(_btn('+ Add', 'btn-primary', () => {
            const g = activeGrid(); if (g) this.publishAdd(g, {});
        }));
        toolbar.appendChild(_btn('Bulk Add', 'btn-outline', () => {
            const g = activeGrid(); if (g) this._showBulkAddOverlay(g, gridContainer);
        }));
        toolbar.appendChild(_btn('Duplicate', 'btn-outline', () => {
            const g = activeGrid(); if (!g) return;
            const sel = this._getEffectiveSelection(g); if (sel.length) this.duplicateRows(g, sel);
        }));
        toolbar.appendChild(_sep());
        toolbar.appendChild(_btn('+ Tag', 'btn-outline', () => {
            const g = activeGrid(); if (g) this._showTagBulkOverlay(g, gridContainer, 'add');
        }));
        toolbar.appendChild(_btn('− Tag', 'btn-outline', () => {
            const g = activeGrid(); if (g) this._showTagBulkOverlay(g, gridContainer, 'remove');
        }));
        toolbar.appendChild(_btn('Tags…', 'btn-outline', () => {
            const g = activeGrid(); if (g) this._showTagManagementPanel(g, gridContainer);
        }));
        toolbar.appendChild(_sep());
        const copyBtn = _btn('Copy', 'btn-outline', () => {
            const g = activeGrid(); if (!g) return;
            const sel = this._getEffectiveSelection(g);
            this.copyRowsToClipboard(g, sel.length ? sel : (this._data.get(g) || []), copyBtn);
        });
        toolbar.appendChild(copyBtn);
        toolbar.appendChild(_btn('Filter Grid', 'btn-outline', () => {
            const g = activeGrid(); if (g) this.filterMainGrid(g);
        }));
        toolbar.appendChild(_sep());
        toolbar.appendChild(_btn('Undo', 'btn-outline', () => {
            const g = activeGrid(); if (g) this.performUndo(g);
        }));
        toolbar.appendChild(_btn('Redo', 'btn-outline', () => {
            const g = activeGrid(); if (g) this.performRedo(g);
        }));
        toolbar.appendChild(_sep());
        toolbar.appendChild(_btn('Remove', 'btn-error btn-outline', () => {
            const g = activeGrid(); if (!g) return;
            const api = this._gridApis.get(g); if (!api) return;
            let sel = api.getSelectedRows();
            if (sel.length === 0) {
                // Try cell range selection
                try {
                    const rng = api.getCellRanges().map(x => {
                        return [...Array(x.endRow.rowIndex - x.startRow.rowIndex + 1).keys()].map(n => n + x.startRow.rowIndex);
                    })[0];
                    if (rng && rng.length) {
                        sel = rng.map(r => api.getDisplayedRowAtIndex(r).data).filter(Boolean);
                    }
                } catch (_) {}
            }
            if (sel.length === 0) {
                // Fallback to active row
                sel = this._getEffectiveSelection(g);
            }
            if (sel.length > 0) this._confirmRemove(g, sel, gridContainer);
        }));
        toolbar.appendChild(_spacer());

        // Activity feed toggle
        const logBtn = _btn('Log', 'btn-outline', () => {
            this._toggleChangeLog(gridContainer);
        });
        logBtn.title = 'Activity feed';
        toolbar.appendChild(logBtn);

        toolbar.appendChild(_btn('Close', '', () => closeWithValue('close')));

        contentArea.appendChild(toolbar);

        // --- Keyboard shortcuts ---
        const keyHandler = (e) => {
            if (e.ctrlKey || e.metaKey) {
                const g = activeGrid();
                if (!g) return;
                if (e.key === 'z' && !e.shiftKey) { e.preventDefault(); this.performUndo(g); }
                if (e.key === 'z' && e.shiftKey) { e.preventDefault(); this.performRedo(g); }
                if (e.key === 'y') { e.preventDefault(); this.performRedo(g); }
                if (e.key === 'c' && !e.target.closest('.ag-cell-edit-wrapper')) {
                    const sel = this._getEffectiveSelection(g);
                    if (sel.length) { e.preventDefault(); this.copyRowsToClipboard(g, sel); }
                }
            }
        };
        dialog.addEventListener('keydown', keyHandler, { signal });

        // --- Render first tab ---
        const firstGrid = grids[0];
        this._activeTabs.set(groupName, firstGrid);
        this._renderGrid(firstGrid, gridContainer, groupName);
    }

    // =========================================================================
    // Delete confirmation
    // =========================================================================

    _confirmRemove(microName, rows, container) {
        if (rows.length <= 2) {
            // Small deletes: no confirmation
            this.publishBulkRemove(microName, rows);
            return;
        }

        if (container.querySelector('.micro-confirm-overlay')) return;

        const overlay = document.createElement('div');
        overlay.className = 'micro-confirm-overlay';

        const msg = document.createElement('div');
        msg.style.cssText = 'font-size:14px;color:#fff;text-align:center;margin-bottom:12px;';
        msg.textContent = `Remove ${rows.length} selected rows?`;

        const actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:8px;justify-content:center;';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-sm';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => overlay.remove());

        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'btn btn-sm btn-error';
        confirmBtn.textContent = 'Remove';
        confirmBtn.addEventListener('click', async () => {
            await this.publishBulkRemove(microName, rows);
            overlay.remove();
        });

        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        overlay.appendChild(msg);
        overlay.appendChild(actions);
        container.appendChild(overlay);
        confirmBtn.focus();
    }

    // =========================================================================
    // Context menu (right-click)
    // =========================================================================

    _showContextMenu(microName, event, api) {
        this._dismissContextMenu();
        event.preventDefault();

        const config = MICRO_GRID_CONFIGS[microName];
        const pkCol = config.primaryKeys[0];
        const selected = this._getEffectiveSelection(microName);

        const menu = document.createElement('div');
        menu.className = 'micro-context-menu';
        menu.style.left = event.clientX + 'px';
        menu.style.top = event.clientY + 'px';

        const items = [
            { label: `Copy${selected.length ? ` (${selected.length})` : ''}`, action: () => {
                    this.copyRowsToClipboard(microName, selected.length ? selected : (this._data.get(microName) || []));
                }},
            { label: `Duplicate${selected.length ? ` (${selected.length})` : ''}`, action: () => {
                    if (selected.length) this.duplicateRows(microName, selected);
                }, disabled: selected.length === 0 },
            { divider: true },
            { label: `Remove${selected.length ? ` (${selected.length})` : ''}`, action: () => {
                    if (selected.length) {
                        const container = menu.closest('.micro-grid-container') || document.body;
                        this._confirmRemove(microName, selected, container);
                    }
                }, disabled: selected.length === 0, danger: true },
        ];

        for (const item of items) {
            if (item.divider) {
                const d = document.createElement('div');
                d.className = 'micro-ctx-divider';
                menu.appendChild(d);
                continue;
            }
            const row = document.createElement('div');
            row.className = 'micro-ctx-item' + (item.disabled ? ' disabled' : '') + (item.danger ? ' danger' : '');
            row.textContent = item.label;
            if (!item.disabled) {
                row.addEventListener('click', () => { this._dismissContextMenu(); item.action(); });
            }
            menu.appendChild(row);
        }

        document.body.appendChild(menu);
        this._contextMenus.set(microName, menu);

        // Dismiss on click outside — store listener ref for cleanup
        const dismiss = (e) => {
            if (!menu.contains(e.target)) {
                this._dismissContextMenu();
            }
        };
        menu._dismissListener = dismiss;
        setTimeout(() => document.addEventListener('click', dismiss, true), 0);
    }

    _dismissContextMenu() {
        for (const [, menu] of this._contextMenus) {
            try {
                if (menu._dismissListener) {
                    document.removeEventListener('click', menu._dismissListener, true);
                }
                menu.remove();
            } catch (e) { /* ignore */ }
        }
        this._contextMenus.clear();
    }

    // =========================================================================
    // Change log / activity feed
    // =========================================================================

    _toggleChangeLog(container) {
        const existing = container.querySelector('.micro-changelog-panel');
        if (existing) {
            if (existing._unsub) existing._unsub();
            existing.remove();
            return;
        }

        const panel = document.createElement('div');
        panel.className = 'micro-changelog-panel';

        const title = document.createElement('div');
        title.style.cssText = 'font-size:13px;font-weight:600;color:#fff;margin-bottom:6px;display:flex;justify-content:space-between;';
        title.innerHTML = 'Activity Log <span style="cursor:pointer;opacity:0.6" class="cl-close">×</span>';
        title.querySelector('.cl-close').addEventListener('click', () => {
            if (panel._unsub) panel._unsub();
            panel.remove();
        });

        const list = document.createElement('div');
        list.className = 'micro-changelog-list';

        const renderEntries = () => {
            list.innerHTML = '';
            const entries = this._changeLog.entries.slice(-30).reverse();
            for (const e of entries) {
                const row = document.createElement('div');
                row.className = 'micro-cl-entry';
                const icon = { add: '+', remove: '−', update: '~', snapshot: '⟳', copy: '📋', filter: '⚡', system: '⟲', tag: '#' }[e.type] || '•';
                row.innerHTML = `<span class="cl-time">${e.time}</span> <span class="cl-icon">${icon}</span> ${_escHtml(e.detail)}`;
                list.appendChild(row);
            }
            if (entries.length === 0) {
                list.innerHTML = '<div style="color:#666;font-size:12px;">No activity yet</div>';
            }
        };
        renderEntries();

        const unsub = this._changeLog.onEntry(() => renderEntries());
        panel._unsub = unsub;

        panel.appendChild(title);
        panel.appendChild(list);
        container.appendChild(panel);
    }

    // =========================================================================
    // Reconnect indicator
    // =========================================================================

    showReconnectIndicator() {
        for (const [, el] of this._reconnectIndicators) {
            el.style.display = 'flex';
        }
    }

    _hideReconnectIndicator() {
        for (const [, el] of this._reconnectIndicators) {
            el.style.display = 'none';
        }
    }

    // =========================================================================
    // Bulk Add overlay
    // =========================================================================

    _showBulkAddOverlay(microName, container) {
        if (container.querySelector('.micro-bulk-overlay')) return;

        const existingRows = this._data.get(microName) || [];
        const existingPatterns = new Set(existingRows.map(r => _upper(r.pattern || '')));

        const overlay = document.createElement('div');
        overlay.className = 'micro-bulk-overlay';

        const title = document.createElement('div');
        title.style.cssText = 'font-size:14px;font-weight:600;color:#fff;';
        title.textContent = 'Bulk Add — paste tickers / ISINs';

        const subtitle = document.createElement('div');
        subtitle.style.cssText = 'font-size:12px;color:#888;';
        subtitle.textContent = 'Comma/newline/tab separated. ISINs (12 chars) and ALL-CAPS tickers auto-detected. Duplicates ignored. Use tag:TICKER to auto-set tags (e.g. test:AAPL).';

        const textarea = document.createElement('textarea');
        textarea.placeholder = 'AAPL, MSFT, GOOGL\nUS0378331005\nTSLA NVDA\ntest:AMZN, watchlist:META';

        const preview = document.createElement('div');
        preview.className = 'bulk-preview';

        textarea.addEventListener('input', () => {
            const items = _extractIdentifiers(textarea.value);
            const deduped = items.filter(i => !existingPatterns.has(_upper(i.pattern)));
            if (deduped.length) {
                const labels = deduped.map(i => i.tags ? `${i.tags}:${i.pattern}` : i.pattern);
                preview.textContent = `Will add ${deduped.length} item(s): ${labels.join(', ')}`;
            } else {
                preview.textContent = items.length ? 'All items already exist.' : '';
            }
        });

        const actions = document.createElement('div');
        actions.className = 'bulk-actions';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-sm';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => overlay.remove());

        const importBtn = document.createElement('button');
        importBtn.className = 'btn btn-sm btn-primary';
        importBtn.textContent = 'Import';
        importBtn.addEventListener('click', () => {
            const items = _extractIdentifiers(textarea.value);
            const deduped = items.filter(i => !existingPatterns.has(_upper(i.pattern)));
            if (deduped.length > 0) this.publishBulkAdd(microName, deduped);
            overlay.remove();
        });

        actions.appendChild(cancelBtn);
        actions.appendChild(importBtn);
        overlay.appendChild(title);
        overlay.appendChild(subtitle);
        overlay.appendChild(textarea);
        overlay.appendChild(preview);
        overlay.appendChild(actions);
        container.appendChild(overlay);
        textarea.focus();
    }

    // =========================================================================
    // Bulk Tag overlay
    // =========================================================================

    _showTagBulkOverlay(microName, container, mode) {
        if (container.querySelector('.micro-tag-bulk-overlay')) return;

        const selected = this._getEffectiveSelection(microName);
        if (selected.length === 0) return;

        const config = MICRO_GRID_CONFIGS[microName];
        const pkCol = config.primaryKeys[0];

        const overlay = document.createElement('div');
        overlay.className = 'micro-tag-bulk-overlay';

        const title = document.createElement('div');
        title.style.cssText = 'font-size:14px;font-weight:600;color:#fff;';
        title.textContent = mode === 'add'
            ? `Add tag to ${selected.length} selected row(s)`
            : `Remove tag from ${selected.length} selected row(s)`;

        const inputRow = document.createElement('div');
        inputRow.className = 'tag-input-row';
        const tagInput = document.createElement('input');
        tagInput.type = 'text';
        tagInput.placeholder = 'Type tag (category:name) and press Enter';
        inputRow.appendChild(tagInput);

        const sugContainer = document.createElement('div');
        sugContainer.className = 'tag-suggestions';
        let chosenTag = '';

        const renderSuggestions = () => {
            sugContainer.innerHTML = '';
            const tags = mode === 'remove'
                ? _getTagsFromRows(selected)
                : [...getAllKnownTags()].sort();
            tags.forEach(tag => {
                const pill = _createPillEl(tag, false);
                pill.style.cursor = 'pointer';
                pill.addEventListener('click', () => { tagInput.value = tag; chosenTag = tag; });
                sugContainer.appendChild(pill);
            });
        };
        renderSuggestions();

        const applyTag = () => {
            if (!chosenTag) return;
            const updates = [];
            for (const row of selected) {
                const currentTags = _parseTags(row.tags);
                let newTags;
                if (mode === 'add') {
                    if (currentTags.includes(chosenTag)) continue;
                    newTags = [...currentTags, chosenTag];
                } else {
                    if (!currentTags.includes(chosenTag)) continue;
                    newTags = currentTags.filter(t => t !== chosenTag);
                }
                updates.push({ [pkCol]: row[pkCol], tags: newTags.join(',') });
            }
            if (updates.length > 0) {
                this.publishBulkUpdate(microName, updates);
                getAllKnownTags().add(chosenTag);
            }
            overlay.remove();
        };

        const actions = document.createElement('div');
        actions.className = 'bulk-actions';
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-sm';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => overlay.remove());
        const applyBtn = document.createElement('button');
        applyBtn.className = 'btn btn-sm btn-primary';
        applyBtn.textContent = mode === 'add' ? 'Add Tag' : 'Remove Tag';
        applyBtn.addEventListener('click', () => { chosenTag = tagInput.value.trim(); if (chosenTag) applyTag(); });

        actions.appendChild(cancelBtn);
        actions.appendChild(applyBtn);
        overlay.appendChild(title);
        overlay.appendChild(inputRow);
        overlay.appendChild(sugContainer);
        overlay.appendChild(actions);
        container.appendChild(overlay);
        tagInput.focus();
    }

    // =========================================================================
    // Tag management panel
    // =========================================================================

    _showTagManagementPanel(microName, container) {
        const existing = container.querySelector('.micro-tag-mgmt-panel');
        if (existing) { existing.remove(); return; }

        const panel = document.createElement('div');
        panel.className = 'micro-tag-mgmt-panel';


        const config = MICRO_GRID_CONFIGS[microName];
        const pkCol = config.primaryKeys[0];

        const renderPanel = () => {
            panel.innerHTML = '';
            // Re-read data each render to get fresh counts
            const allRows = this._data.get(microName) || [];

            const header = document.createElement('div');
            header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;';
            header.innerHTML = `<span style="font-size:14px;font-weight:600;color:#fff;">Tag Management</span>`;
            const closeX = document.createElement('span');
            closeX.textContent = '×';
            closeX.style.cssText = 'cursor:pointer;font-size:18px;color:#888;';
            closeX.addEventListener('click', () => panel.remove());
            header.appendChild(closeX);
            panel.appendChild(header);

            const cats = getTagsByCategory();
            const tagCounts = new Map();
            for (const row of allRows) {
                for (const t of _parseTags(row.tags)) {
                    tagCounts.set(t, (tagCounts.get(t) || 0) + 1);
                }
            }

            const sortedCats = [...cats.keys()].sort((a, b) => {
                if (a === '') return 1;
                if (b === '') return -1;
                return a.localeCompare(b);
            });

            for (const cat of sortedCats) {
                if (cat) {
                    const catH = document.createElement('div');
                    catH.style.cssText = 'font-size:10px;font-weight:600;color:#888;text-transform:uppercase;margin:8px 0 3px;letter-spacing:0.5px;';
                    catH.textContent = cat;
                    panel.appendChild(catH);
                }

                for (const tag of cats.get(cat)) {
                    const count = tagCounts.get(tag) || 0;
                    const row = document.createElement('div');
                    row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:3px 0;';

                    // Color picker swatch
                    const swatch = document.createElement('input');
                    swatch.type = 'color';
                    const customC = getTagCustomColors().get(tag);
                    swatch.value = customC ? customC.bg : _hslToHex(_tagHueForPicker(tag));
                    swatch.style.cssText = 'width:20px;height:20px;border:none;padding:0;cursor:pointer;background:transparent;flex-shrink:0;';
                    swatch.addEventListener('change', () => {
                        const hex = swatch.value;
                        setTagColor(tag, hex, _lighten(hex), _darken(hex));
                        this._refreshGrid(microName);
                        renderPanel();
                    });

                    const pill = _createPillEl(tag, false);
                    pill.style.flex = '1';

                    const countSpan = document.createElement('span');
                    countSpan.style.cssText = 'font-size:11px;color:#666;min-width:20px;text-align:right;';
                    countSpan.textContent = count;

                    // Rename button
                    const renameBtn = document.createElement('span');
                    renameBtn.textContent = '✎';
                    renameBtn.title = 'Rename tag';
                    renameBtn.style.cssText = 'cursor:pointer;font-size:13px;color:#888;';
                    renameBtn.addEventListener('click', () => {
                        const newName = prompt(`Rename tag "${tag}" to:`, tag);
                        if (newName && newName !== tag) this._renameTag(microName, tag, newName.trim());
                    });

                    // Delete button
                    const delBtn = document.createElement('span');
                    delBtn.textContent = '×';
                    delBtn.title = 'Remove from all rows';
                    delBtn.style.cssText = 'cursor:pointer;font-size:15px;color:#f87171;';
                    delBtn.addEventListener('click', () => {
                        this._removeTagGlobally(microName, tag);
                        renderPanel();
                    });

                    row.appendChild(swatch);
                    row.appendChild(pill);
                    row.appendChild(countSpan);
                    row.appendChild(renameBtn);
                    row.appendChild(delBtn);
                    panel.appendChild(row);
                }
            }

            if (cats.size === 0) {
                const empty = document.createElement('div');
                empty.style.cssText = 'color:#666;font-size:12px;';
                empty.textContent = 'No tags yet';
                panel.appendChild(empty);
            }
        };

        renderPanel();
        container.appendChild(panel);
    }

    _renameTag(microName, oldTag, newTag) {
        const config = MICRO_GRID_CONFIGS[microName];
        const pkCol = config.primaryKeys[0];
        const allRows = this._data.get(microName) || [];
        const updates = [];
        for (const row of allRows) {
            const tags = _parseTags(row.tags);
            const idx = tags.indexOf(oldTag);
            if (idx >= 0) {
                tags[idx] = newTag;
                updates.push({ [pkCol]: row[pkCol], tags: tags.join(',') });
            }
        }
        if (updates.length > 0) {
            this.publishBulkUpdate(microName, updates);
            getAllKnownTags().add(newTag);
            getAllKnownTags().delete(oldTag);
            // Transfer color
            const custom = getTagCustomColors().get(oldTag);
            if (custom) { setTagColor(newTag, custom.bg, custom.fg, custom.border); removeTagColor(oldTag); }
        }
    }

    _removeTagGlobally(microName, tag) {
        const config = MICRO_GRID_CONFIGS[microName];
        const pkCol = config.primaryKeys[0];
        const allRows = this._data.get(microName) || [];
        const updates = [];
        for (const row of allRows) {
            const tags = _parseTags(row.tags);
            if (tags.includes(tag)) {
                updates.push({ [pkCol]: row[pkCol], tags: tags.filter(t => t !== tag).join(',') });
            }
        }
        if (updates.length > 0) {
            this.publishBulkUpdate(microName, updates);
            getAllKnownTags().delete(tag);
            removeTagColor(tag);
        }
    }

    // =========================================================================
    // Grid rendering
    // =========================================================================

    _switchTab(groupName, gridName, container) {
        const currentGrid = this._activeTabs.get(groupName);
        if (currentGrid) this._destroyGrid(currentGrid);
        this._activeTabs.set(groupName, gridName);
        container.innerHTML = '';
        this._renderGrid(gridName, container, groupName);
    }

    _renderGrid(microName, container, groupName) {
        const config = MICRO_GRID_CONFIGS[microName];
        if (!config) return;

        const gridDiv = document.createElement('div');
        gridDiv.className = 'ag-theme-alpine';
        gridDiv.style.width = '100%';
        gridDiv.style.height = '100%';
        container.appendChild(gridDiv);

        const rows = this._data.get(microName) || [];
        seedKnownTags(rows);
        const mgr = this;
        const pkCol = config.primaryKeys[0];

        // Phantom quick-add row
        const phantomRow = config.addRowDefaults ? config.addRowDefaults({ _phantom: true }) : { _phantom: true };
        phantomRow[pkCol] = '__phantom__';
        phantomRow.pattern = '';
        phantomRow.notes = '';

        const gridOptions = {
            columnDefs: config.columns,
            rowData: rows,
            pinnedBottomRowData: [phantomRow],
            rowSelection: 'multiple',
            suppressRowClickSelection: false,
            animateRows: false,
            rowHeight: 28,
            getRowId: (params) => String(params.data[pkCol]),
            getRowStyle: (params) => {
                if (params.data?._phantom) return { fontStyle: 'italic', opacity: '0.5' };
                return null;
            },
            defaultColDef: {
                resizable: true,
                sortable: true,
                filter: true,
                suppressMovable: true,
                // Fallback valueSetter: ensures that columns which define a
                // valueGetter but no valueSetter (e.g. the tags column) can
                // still persist cell-editor values back to row data.  Without
                // this, ag-grid silently drops the edit and onCellValueChanged
                // never fires.  Column-level valueSetters override this.
                valueSetter: (params) => {
                    const field = params.colDef.field;
                    if (!field) return false;
                    // Normalize: if the editor returns an array for a field
                    // that stores comma-separated strings (tags), flatten it.
                    let val = params.newValue;
                    if (field === 'tags' && Array.isArray(val)) {
                        val = val.map(t => String(t).trim()).filter(Boolean).join(',');
                    }
                    const old = params.data[field];
                    // Coerce both sides to string for comparison so
                    // "a,b" vs ["a","b"] doesn't fool the equality check.
                    const oldStr = old == null ? '' : String(old);
                    const newStr = val == null ? '' : String(val);
                    if (oldStr === newStr) return false;
                    params.data[field] = val;
                    return true;
                },
            },
            singleClickEdit: false,
            stopEditingWhenCellsLoseFocus: true,
            suppressScrollOnNewData: true,
            pagination: false,
            enableRangeSelection: true,
            cacheQuickFilter: true,
            enterNavigatesVertically: false,
            enterNavigatesVerticallyAfterEdit:false,

            onCellEditingStopped: (event) => {
                if (event.valueChanged) return;  // Already handled by onCellValueChanged
                if (event.data?._phantom) return;
                if (event.colDef.field !== 'tags') return;


                let editorValue = event.newValue;
                if (editorValue === undefined || editorValue === null) return;

                let normalized;
                if (Array.isArray(editorValue)) {
                    normalized = editorValue.map(t => String(t).trim()).filter(Boolean).join(',');
                } else {
                    normalized = _parseTags(editorValue).join(',');
                }

                const currentTags = _parseTags(event.data.tags).join(',');
                if (normalized === currentTags) return;  // Genuinely unchanged

                // Manually write back and publish
                const oldTags = event.data.tags ?? '';
                event.data.tags = normalized;
                seedKnownTags([event.data]);

                const row = { [pkCol]: event.data[pkCol], tags: normalized };
                const oldRow = { [pkCol]: event.data[pkCol], tags: oldTags };
                mgr.publishUpdate(microName, row, oldRow, { skipRefresh: true });
            },
            onCellValueChanged: (event) => {
                const data = event.data;

                // Phantom row → promote to real add
                if (data._phantom) {
                    const newRow = { ...data };
                    delete newRow._phantom;
                    newRow[pkCol] = _generate_trace();
                    mgr.publishAdd(microName, newRow);

                    // Reset phantom
                    const resetPhantom = config.addRowDefaults ? config.addRowDefaults({ _phantom: true }) : { _phantom: true };
                    resetPhantom[pkCol] = '__phantom__';
                    resetPhantom.pattern = '';
                    resetPhantom.notes = '';
                    event.api.setGridOption('pinnedBottomRowData', [resetPhantom]);
                    return;
                }

                // Snapshot pre-edit values for undo.  ag-grid has already
                // mutated event.data, so only event.oldValue is reliable for
                // the edited column; every other column uses current data.
                const oldRow = {};
                for (const col of Object.keys(data)) {
                    oldRow[col] = event.oldValue !== undefined && col === event.colDef.field
                        ? event.oldValue : data[col];
                }

                const row = { [pkCol]: data[pkCol] };
                let newValue = event.newValue;

                // Normalize tag value — cell editors may hand back arrays,
                // whitespace-padded strings, etc.
                if (event.colDef.field === 'tags') {
                    if (Array.isArray(newValue)) {
                        newValue = newValue.map(t => t.trim()).filter(Boolean).join(',');
                    } else {
                        newValue = _parseTags(newValue).join(',');
                    }
                    // Write the canonical value back into the row object so the
                    // cell renderer sees a clean comma-separated string.
                    data.tags = newValue;
                }

                row[event.colDef.field] = newValue;
                mgr.publishUpdate(microName, row, oldRow, { skipRefresh: true });
            },
            onCellContextMenu: (event) => {
                if (event.data?._phantom) return;
                mgr._showContextMenu(microName, event.event, event.api);
            },
            onModelUpdated: () => {
                mgr._updateStatusBar(microName, groupName);
            },
            onSelectionChanged: () => {
                mgr._updateStatusBar(microName, groupName);
            },
        };

        const api = createGrid(gridDiv, gridOptions);
        this._gridApis.set(microName, api);
        this.keyCoalescer = installKeyCoalescer(this, gridDiv, api);
        this._updateStatusBar(microName, groupName);
    }

    _updateStatusBar(microName, groupName) {
        if (!this._statusBars) return;
        const bar = this._statusBars.get(groupName);
        if (!bar) return;

        const api = this._gridApis.get(microName);
        const total = (this._data.get(microName) || []).length;
        let displayed = total;
        let selected = 0;

        if (api) {
            try {
                selected = api.getSelectedRows().length;
                let count = 0;
                api.forEachNodeAfterFilter((node) => { if (!node.data?._phantom) count++; });
                displayed = count;
            } catch (e) { /* ignore */ }
        }

        const parts = [`${displayed} of ${total} rows`];
        if (selected > 0) parts.push(`${selected} selected`);
        if (displayed < total) parts.push('filtered');
        bar.textContent = parts.join(' · ');
    }

    _destroyGrid(microName) {
        const api = this._gridApis.get(microName);
        if (api) {
            try { api.destroy(); } catch (e) { /* ignore */ }
            this._gridApis.delete(microName);
        }
        this._dismissContextMenu();
    }


    // =========================================================================
    // Cleanup
    // =========================================================================

    async _cleanupGroup(groupName) {
        const groupConfig = MICRO_GRID_GROUPS[groupName];
        if (!groupConfig) return;

        const ac = this._abortControllers.get(groupName);
        if (ac) { ac.abort(); this._abortControllers.delete(groupName); }

        for (const gridName of groupConfig.grids) {
            this._destroyGrid(gridName);
            // Don't unsubscribe here — keep receiving data for derived columns.
            // Subscriptions are only torn down in destroy() during page cleanup.
        }

        this._openModals.delete(groupName);
        this._activeTabs.delete(groupName);
        this._gridContainers.delete(groupName);
        this._reconnectIndicators.delete(groupName);
        if (this._statusBars) this._statusBars.delete(groupName);
    }

    async destroy() {
        for (const [, ac] of this._abortControllers) {
            try { ac.abort(); } catch (e) { /* ignore */ }
        }
        this._abortControllers.clear();

        for (const [, dialog] of this._openModals) {
            try { if (dialog?.open) dialog.close(); } catch (e) { /* ignore */ }
        }

        for (const [name] of this._gridApis) {
            this._destroyGrid(name);
        }

        for (const microName of new Set(this._subscribed)) {
            await this.unsubscribe(microName).catch(() => {});
        }

        this._dismissContextMenu();
        this._data.clear();
        this._gridApis.clear();
        this._openModals.clear();
        this._activeTabs.clear();
        this._handlers.clear();
        this._subscribed.clear();
        this._undoStacks.clear();
        this._changeLog.clear();
        this._gridContainers.clear();
        this._reconnectIndicators.clear();
        this._dataListeners.clear();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function _getTagsFromRows(rows) {
    const tagSet = new Set();
    for (const row of rows) {
        for (const t of _parseTags(row.tags)) tagSet.add(t);
    }
    return [...tagSet].sort();
}

function _tagHueForPicker(tag) {
    let h = 0;
    const label = tag.includes(':') ? tag.split(':')[1] : tag;
    for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) & 0xFFFF;
    return h % 360;
}

function _hslToHex(hue) {
    // Convert HSL(hue, 55%, 25%) to hex
    const s = 0.55, l = 0.25;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
    const m = l - c / 2;
    let r, g, b;
    if (hue < 60) { r = c; g = x; b = 0; }
    else if (hue < 120) { r = x; g = c; b = 0; }
    else if (hue < 180) { r = 0; g = c; b = x; }
    else if (hue < 240) { r = 0; g = x; b = c; }
    else if (hue < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    const toHex = (v) => Math.round((v + m) * 255).toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function _lighten(hex) {
    // Lighten a hex color for text
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const l = (v) => Math.min(255, v + 140).toString(16).padStart(2, '0');
    return `#${l(r)}${l(g)}${l(b)}`;
}

function _darken(hex) {
    // Darken a hex color for border
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const d = (v) => Math.max(0, v + 40).toString(16).padStart(2, '0');
    return `#${d(r)}${d(g)}${d(b)}`;
}
