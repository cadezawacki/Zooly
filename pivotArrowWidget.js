
// pt/js/widgets/pivotWidget.js

import {BaseWidget} from './baseWidget.js';
import {combineLatest, mergeAll} from 'rxjs';
import {debounceTime, filter, map} from 'rxjs/operators';
import '../../css/pivot.css';
import * as utils from '@stdlib/utils';
import {pivotColumns} from "@/pt/js/grids/portfolio/portfolioColumns.js";
import {writeObjectToClipboard, getVenueCopyColumns, mapVenueFromRfq} from '@/utils/clipboardHelpers.js';
import {ArrowAgPivotAdapter} from '@/grids/js/arrow/arrowPivotEngine.js';
import {asArray, wait} from "@/utils/helpers.js";
import {ArrowAgGridAdapter} from "@/grids/js/arrow/arrowEngine.js";
import {ACTION_MAP} from "@/global/actionMap.js";
import interact from "interactjs";

function _escHtml(v) {
    if (v == null) return '';
    const s = typeof v === 'number' ? v.toLocaleString(undefined, {maximumFractionDigits: 4}) : String(v);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}


function wavgFromObjects(rows, value, weight) {
    let weightedSum = 0
    let weightSum = 0
    for (let i = 0, n = rows.length; i < n; i++) {
        const row = rows[i]
        const w = row[weight]
        weightedSum += row[value] * w
        weightSum += w
    }
    return weightSum !== 0 ? weightedSum / weightSum : NaN
}

function groupWavgFromObjects(rows, keyProp, valueProp, weightProp) {
    const metrics = normalizeMetrics(valueProp, weightProp)
    const m = metrics.length

    const weightsMode = classifyWeights(weightProp, m)

    const acc = new Map() // key -> { sumVW: Float64Array(m), sumW: Float64Array(m) }

    const keyIsArray = Array.isArray(keyProp)
    const keyProps = keyIsArray ? keyProp : null
    const keyPropsLen = keyIsArray ? keyProps.length : 0

    for (let r = 0, n = rows.length; r < n; r++) {
        const row = rows[r]

        let key
        if (keyIsArray) {
            key = ''
            for (let k = 0; k < keyPropsLen; k++) {
                if (k) key += '_'
                key += row[keyProps[k]]
            }
        } else {
            key = row[keyProp]
        }

        let state = acc.get(key)
        if (state === undefined) {
            state = { sumVW: new Float64Array(m), sumW: new Float64Array(m) }
            acc.set(key, state)
        }

        const sumVW = state.sumVW
        const sumW = state.sumW

        switch (weightsMode.type) {
            case 0: { // scalar numeric constant
                const wConst = weightsMode.wConst
                if (wConst === 0 || wConst == null) break
                for (let j = 0; j < m; j++) {
                    const v = row[metrics[j].prop]
                    if (v == null) continue
                    sumVW[j] += v * wConst
                    sumW[j] += wConst
                }
                break
            }

            case 1: { // scalar weight property name
                const w = row[weightsMode.wProp]
                if (w === 0 || w == null) break
                for (let j = 0; j < m; j++) {
                    const v = row[metrics[j].prop]
                    if (v == null) continue
                    sumVW[j] += v * w
                    sumW[j] += w
                }
                break
            }

            case 2: { // per-metric numeric constants
                const wArr = weightsMode.wArr
                for (let j = 0; j < m; j++) {
                    const w = wArr[j]
                    if (w === 0 || w == null) continue
                    const v = row[metrics[j].prop]
                    if (v == null) continue
                    sumVW[j] += v * w
                    sumW[j] += w
                }
                break
            }

            case 3: { // per-metric weight property names
                const wPropArr = weightsMode.wPropArr
                for (let j = 0; j < m; j++) {
                    const w = row[wPropArr[j]]
                    if (w === 0 || w == null) continue
                    const v = row[metrics[j].prop]
                    if (v == null) continue
                    sumVW[j] += v * w
                    sumW[j] += w
                }
                break
            }
        }
    }

    const out = Object.create(null)
    for (const [key, state] of acc) {
        const res = Object.create(null)
        const sumVW = state.sumVW
        const sumW = state.sumW
        for (let j = 0; j < m; j++) {
            const w = sumW[j]
            res[metrics[j].as] = w !== 0 ? (sumVW[j] / w) : NaN
        }
        out[key] = res
    }
    return out
}

function normalizeMetrics(valueProp, weightProp) {
    const valueArr = Array.isArray(valueProp) ? valueProp : [valueProp]

    // Allow explicit {prop, as} entries
    const metrics = new Array(valueArr.length)
    for (let i = 0; i < valueArr.length; i++) {
        const v = valueArr[i]
        if (v && typeof v === 'object') {
            const prop = v.prop
            const as = v.as != null ? v.as : prop
            metrics[i] = { prop, as }
        } else {
            metrics[i] = { prop: v, as: v }
        }
    }

    // Ensure unique output names; if duplicates, suffix using per-metric weight name if available
    const seen = new Map()
    const weightSuffixes = getWeightSuffixes(weightProp, metrics.length)

    for (let i = 0; i < metrics.length; i++) {
        let name = metrics[i].as
        const count = seen.get(name)
        if (count === undefined) {
            seen.set(name, 1)
            continue
        }

        // Duplicate: create deterministic suffix
        const suffix = weightSuffixes[i] != null ? String(weightSuffixes[i]) : String(count)
        name = name + '__' + suffix

        // If still colliding, keep bumping with numeric postfix
        let bump = 2
        while (seen.has(name)) {
            name = metrics[i].as + '__' + suffix + '__' + bump
            bump++
        }
        metrics[i].as = name

        seen.set(metrics[i].as, 1)
        seen.set(metrics[i].as.split('__')[0], count + 1) // keep original base count moving
    }

    return metrics
}

function getWeightSuffixes(weightProp, m) {
    // Only meaningful when weightProp is an array of strings (per-metric prop names)
    if (Array.isArray(weightProp) && weightProp.length === m && typeof weightProp[0] === 'string') {
        for (let i = 1; i < m; i++) {
            if (typeof weightProp[i] !== 'string') return null
        }
        return weightProp
    }
    return null
}

function classifyWeights(weightProp, m) {
    // type:
    // 0 -> scalar numeric const
    // 1 -> scalar prop name
    // 2 -> per-metric numeric consts
    // 3 -> per-metric prop names

    if (Array.isArray(weightProp)) {
        if (weightProp.length !== m) {
            throw new RangeError(`weightProp array length (${weightProp.length}) must match valueProp length (${m})`)
        }
        if (m === 0) return { type: 2, wArr: weightProp }

        const first = weightProp[0]
        const isStringArray = typeof first === 'string'
        const isNumberArray = typeof first === 'number'

        if (!isStringArray && !isNumberArray) {
            throw new TypeError('weightProp array must contain either strings (property names) or numbers (constants)')
        }

        for (let i = 1; i < m; i++) {
            const t = typeof weightProp[i]
            if (isStringArray) {
                if (t !== 'string') throw new TypeError('weightProp array must be all strings or all numbers (no mixing)')
            } else {
                if (t !== 'number') throw new TypeError('weightProp array must be all numbers or all strings (no mixing)')
            }
        }

        return isStringArray
            ? { type: 3, wPropArr: weightProp }
            : { type: 2, wArr: weightProp }
    }

    if (typeof weightProp === 'number') {
        return { type: 0, wConst: weightProp }
    }

    return { type: 1, wProp: weightProp }
}

//context, widgetId, manager, feederId, selector, config = {}
export class PivotWidget extends BaseWidget {
    constructor(context, widgetId, adapter, selector, config = {}) {
        super(context, widgetId, adapter, selector, config);

        // Debug
        this.tableId = config.tableId || "pivotWidget-table";
        this.pivotSelector = config.pivotSelector || '#pivot-grid-container';
        this.ptPivot = null;

        this.currentGroups = [];
        this.required_groups = [];
        this._applyRafId = 0;
        this._applyTries = 0;
        this._current_active_presets = new Set();
        this._em = this.context.page.emitter || this.context.emitter;

        // Subscription disposers drained in onCleanup
        this._disposers = [];
        // In-flight guard for applyAllBuckets
        this._applyInFlight = false;
        // RAF id for coalesced alignSkewButton
        this._alignRafId = 0;
        // Tracked timeout for onPresetLoaded
        this._presetTimeoutId = 0;

        this._first_activation = false;
        // In-flight guard for pivot lock toggle
        this._lockToggleInFlight = false;

    }

    // --- Widget Lifecycle ---

    async onInit() {

        const required_groups = ['userSide', 'QT'];
        this.required_groups = required_groups;
        const required_aggs = [
            {isPriced: {func: 'PERCENT_OF_ROW_COUNT', name: '%'}},
            {refSkew: {func: 'wavg', name: 'Skew', weight: 'grossSize'}},
            {grossSize: {func: 'sum', name: 'Gross'}},
            {refSkewPx: {func: 'wavg', name: 'refSkewPx', weight: 'grossSize', DROP_NULLS: false, FILL_NULL: 0, ZERO_AS_NULL: false}},
            {refSkewSpd: {func: 'wavg', name: 'refSkewSpd', weight: 'grossSize', DROP_NULLS: false, FILL_NULL: 0, ZERO_AS_NULL: false}},
            {skewScore: {func: 'wavg', name: 'skewScore', weight: 'grossSize', DROP_NULLS: true, ZERO_AS_NULL: true}},

            {current_skew: {func: 'wavg', name: 'current_skew', weight: 'grossSize', headerName: 'Current', DROP_NULLS: false, FILL_NULL: 0, ZERO_AS_NULL: false}},
            {input_skew: {func: 'wavg', name: 'input_skew', weight: 'grossSize', headerName: '∆'}},
            {proposed_skew: {func: 'wavg', name: 'proposed_skew', weight: 'grossSize', headerName: 'Prop'}},
            {isLocked: {func: 'mean', name: 'isLocked', headerName: '🔒', FILL_NULL: 0, ZERO_AS_NULL: false}},
        ];

        // SVG icons for lock states (sized for grid cells, colored via currentColor)
        const _lockSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"><path fill="currentColor" d="M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h1V6q0-2.075 1.463-3.537T12 1t3.538 1.463T17 6v2h1q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22zm7.413-5.587Q14 15.825 14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17t1.413-.587M9 8h6V6q0-1.25-.875-2.125T12 3t-2.125.875T9 6z"/></svg>`;
        const _semiSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"><path fill="currentColor" d="M13.413 16.413Q14 15.825 14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17t1.413-.587M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h7V6q0-2.075 1.463-3.537T18 1q1.875 0 3.263 1.213T22.925 5.2q.05.325-.225.563T22 6t-.7-.175t-.4-.575q-.275-.95-1.062-1.6T18 3q-1.25 0-2.125.875T15 6v2h3q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22z"/></svg>`;
        const _unlockSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"><path fill="currentColor" d="M13.413 16.413Q14 15.825 14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17t1.413-.587M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h7V6q0-2.075 1.463-3.537T18 1q1.875 0 3.263 1.213T22.925 5.2q.05.325-.225.563T22 6t-.7-.175t-.4-.575q-.275-.95-1.062-1.6T18 3q-1.25 0-2.125.875T15 6v2h3q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22z"/></svg>`;

        // Build lock column def for pivot isLocked toggle
        const lockColDef = {
            field: 'isLocked',
            headerName: '🔒',
            headerClass: ["text-center", 'small-header'],
            width: 50,
            maxWidth: 60,
            minWidth: 40,
            suppressMenu: true,
            sortable: false,
            cellClass: ["center-cell"],
            cellStyle: params => {
                const v = params.value;
                let c;
                if (v == null || v === 0) c = 'var(--gray-500, #6b7280)';
                else if (v === 1) c = 'var(--red-500, #ef4444)';
                else c = 'var(--amber-500, #f59e0b)';
                return {
                    cursor: 'pointer',
                    textAlign: 'center',
                    color: c,
                    fontWeight: 600,
                    userSelect: 'none',
                };
            },
            cellRenderer: params => {
                const v = params.value;
                if (v === 1) return `<span title="Fully locked — click to unlock">${_lockSvg}</span>`;
                if (v == null || v === 0) return `<span title="Unlocked — click to lock">${_unlockSvg}</span>`;
                return `<span title="Partially locked — click to lock all">${_semiSvg}</span>`;
            },
            onCellClicked: params => {
                this._handlePivotLockToggle(params);
            },
        };

        const allCustomDefs = [...(pivotColumns || []), lockColDef];

        this.ptPivot = new ArrowAgPivotAdapter(this.context, this.adapter, {
            requiredGroups: required_groups,
            requiredAggregations: required_aggs,
            customDefs: allCustomDefs
        });

        this.ptPivot.mount(this.pivotSelector);

        // Store SVG strings for reuse in context menu
        this._lockSvg = _lockSvg;
        this._semiSvg = _semiSvg;
        this._unlockSvg = _unlockSvg;

        // Set up context menu on pivot grid
        this._setupPivotContextMenu();

        this.currentGroups = [];
        this._initControls();
        this._cacheDom();
        await this._setupReactions();
        this._setupHotkeys();
        this._setupResizer();
        this.setupTooltips();
        this._em.emitSync('pivot-initialized');

    }

    _setupPivotContextMenu() {
        const pivot = this.ptPivot;
        if (!pivot?.api) return;
        const widget = this;
        pivot.api.setGridOption('getContextMenuItems', (params) => {
            const api = params.api;
            const ranges = api.getCellRanges();
            // Collect all visible row nodes in selection range
            const selectedNodes = [];
            if (ranges && ranges.length > 0) {
                for (const rng of ranges) {
                    const sr = Math.min(rng.startRow.rowIndex, rng.endRow.rowIndex);
                    const er = Math.max(rng.startRow.rowIndex, rng.endRow.rowIndex);
                    for (let i = sr; i <= er; i++) {
                        const node = api.getDisplayedRowAtIndex(i);
                        if (node?.data) selectedNodes.push(node);
                    }
                }
            } else if (params.node?.data) {
                selectedNodes.push(params.node);
            }
            if (selectedNodes.length === 0) return ['copy', 'copyWithHeaders', 'export'];

            // Determine lock state across all selected pivot rows
            const lockValues = selectedNodes.map(n => n.data.isLocked);
            const allLocked = lockValues.every(v => v === 1);
            const allUnlocked = lockValues.every(v => v == null || v === 0);

            const lockIcon = `<span style="color:var(--red-500,#ef4444)">${widget._lockSvg}</span>`;
            const unlockIcon = `<span style="color:var(--gray-500,#6b7280)">${widget._unlockSvg}</span>`;
            const semiIcon = `<span style="color:var(--amber-500,#f59e0b)">${widget._semiSvg}</span>`;

            const items = [];
            if (allUnlocked) {
                items.push({
                    name: 'Lock Rows',
                    icon: lockIcon,
                    action: () => widget._bulkLockFromContextMenu(selectedNodes, 1),
                });
            } else if (allLocked) {
                items.push({
                    name: 'Unlock Rows',
                    icon: unlockIcon,
                    action: () => widget._bulkLockFromContextMenu(selectedNodes, 0),
                });
            } else {
                items.push({
                    name: 'Toggle Locks',
                    icon: semiIcon,
                    action: () => {
                        // Per-row toggle: locked rows unlock, unlocked rows lock
                        for (const node of selectedNodes) {
                            const newVal = node.data.isLocked === 1 ? 0 : 1;
                            widget._handlePivotLockToggle({node, value: node.data.isLocked, _forceValue: newVal});
                        }
                    },
                });
                items.push({
                    name: 'Lock All',
                    icon: lockIcon,
                    action: () => widget._bulkLockFromContextMenu(selectedNodes, 1),
                });
                items.push({
                    name: 'Unlock All',
                    icon: unlockIcon,
                    action: () => widget._bulkLockFromContextMenu(selectedNodes, 0),
                });
            }
            items.push('separator');
            items.push('copy', 'copyWithHeaders', 'export');
            return items;
        });
    }

    async _bulkLockFromContextMenu(nodes, lockValue) {
        if (this._lockToggleInFlight) return;
        this._lockToggleInFlight = true;
        try {
            const engine = this.engine;
            const adapter = this.adapter;
            const id_col = engine._idProperty;
            const idx_col = adapter._idxProperty;
            const groupBy = this.ptPivot.pivotConfig?.groupBy || [];
            const updates = [];

            for (const node of nodes) {
                const pivotData = node.data || {};
                const groupFilters = {};
                for (const col of groupBy) {
                    if (pivotData[col] !== undefined && pivotData[col] !== null) {
                        groupFilters[col] = pivotData[col];
                    }
                }
                for (const col of this.required_groups) {
                    if (pivotData[col] !== undefined && pivotData[col] !== null) {
                        groupFilters[col] = pivotData[col];
                    }
                }
                const allCols = [...new Set([...Object.keys(groupFilters), 'isLocked', id_col])];
                const allRows = engine.getAllRows({columns: allCols});
                for (const row of allRows) {
                    let match = true;
                    for (const [col, val] of Object.entries(groupFilters)) {
                        if (String(row[col] ?? '') !== String(val ?? '')) { match = false; break; }
                    }
                    if (!match) continue;
                    const rid = row[id_col];
                    const ri = engine.getRowIndexById(rid);
                    if (ri == null) continue;
                    updates.push({[idx_col]: ri, [id_col]: rid, isLocked: lockValue});
                }
            }

            if (updates.length > 0) {
                await adapter.applyServerUpdateTransaction(updates, {emitAsEdit: true});
                const toast = this.context.page.toastManager();
                const action = lockValue === 1 ? 'Locked' : 'Unlocked';
                toast.info('Lock Toggle', `${action} ${updates.length} bond${updates.length !== 1 ? 's' : ''}.`);
            }
        } catch (err) {
            console.error('[pivot-lock-context]', err);
        } finally {
            this._lockToggleInFlight = false;
        }
    }

    /**
     * Handles clicking the isLocked cell on a pivot row.
     * Determines which bonds belong to that pivot group and toggles their lock state.
     * If all bonds are locked (mean === 1), unlocks all. Otherwise locks all.
     */
    async _handlePivotLockToggle(params) {
        if (this._lockToggleInFlight) return;
        this._lockToggleInFlight = true;

        try {
            const node = params.node;
            if (!node) return;

            const pivotData = node.data || {};
            const groupBy = this.ptPivot.pivotConfig?.groupBy || [];
            const engine = this.engine;
            const adapter = this.adapter;
            const id_col = engine._idProperty;
            const idx_col = adapter._idxProperty;

            // Determine the group filter criteria from the pivot row data
            const groupFilters = {};
            for (const col of groupBy) {
                if (pivotData[col] !== undefined && pivotData[col] !== null) {
                    groupFilters[col] = pivotData[col];
                }
            }
            // Also include the required groups (userSide, QT) which are always present
            for (const col of this.required_groups) {
                if (pivotData[col] !== undefined && pivotData[col] !== null) {
                    groupFilters[col] = pivotData[col];
                }
            }

            // Get all rows from the source engine that match the group filter
            const allCols = [...new Set([...Object.keys(groupFilters), 'isLocked', id_col])];
            const allRows = engine.getAllRows({columns: allCols});

            // Filter to only rows matching all group column values
            const matchingRows = allRows.filter(row => {
                for (const [col, val] of Object.entries(groupFilters)) {
                    if (String(row[col] ?? '') !== String(val ?? '')) return false;
                }
                return true;
            });

            if (matchingRows.length === 0) return;

            // Determine new lock value (support forced value from context menu)
            const newLockValue = params._forceValue != null
                ? params._forceValue
                : (pivotData.isLocked === 1 ? 0 : 1);

            // Build update payloads
            const updates = [];
            for (const row of matchingRows) {
                const rid = row[id_col];
                const ri = engine.getRowIndexById(rid);
                if (ri == null) continue;
                updates.push({[idx_col]: ri, [id_col]: rid, isLocked: newLockValue});
            }

            if (updates.length > 0) {
                await adapter.applyServerUpdateTransaction(updates, {emitAsEdit: true});
                const toast = this.context.page.toastManager();
                const action = newLockValue === 1 ? 'Locked' : 'Unlocked';
                toast.info('Lock Toggle', `${action} ${updates.length} bond${updates.length !== 1 ? 's' : ''}.`);
            }
        } catch (err) {
            console.error('[pivot-lock-toggle]', err);
        } finally {
            this._lockToggleInFlight = false;
        }
    }

    flipLock(force = null) {
        let v;
        if (force != null) {
            v = force
        } else {
            v = !this.context.page.page$.get('proceedsLocked');
        }
        if (this.tt) {
            if (v) {
                this.tt.enable();
                this.tt.flash();
            } else {
                this.tt.stopFlash();
                this.tt.disable();
            }
        }

        this.context.page.page$.set('proceedsLocked', v);
        return v
    }

    async _redistributeProceeds(solverOptions = {}) {
        if (this._redistInFlight) return;
        this._redistInFlight = true;
        this._redistStale = false;

        const page = this.context.page;
        const toast = page.toastManager();
        const mm = page.modalManager();
        const sm = page.subscriptionManager();
        const socketManager = page.socketManager();

        const roomContext = page._ptRaw?.context;
        if (!roomContext) {
            toast.error('Redistribute', 'No active portfolio context.');
            this._redistInFlight = false;
            return;
        }

        const portfolioRoom = roomContext.room;

        // Track portfolio publishes to detect stale data
        const onPortfolioPublish = () => { this._redistStale = true; };
        sm.registerMessageHandler(portfolioRoom, 'publish', onPortfolioPublish);

        const cleanup = () => {
            sm.unregisterMessageHandler(portfolioRoom, 'publish', onPortfolioPublish);
            this._redistInFlight = false;
        };

        toast.info('Proceeds Solver', 'Computing redistribution...');

        try {
            const result = await this._fetchRedistribution(socketManager, roomContext, solverOptions);
            await this._showRedistributeModal(result, roomContext, {mm, sm, toast, esc: _escHtml, cleanup, onPortfolioPublish});
        } catch (err) {
            console.error('[redistribute]', err);
            toast.error('Redistribute', `Error: ${err.message || err}`);
            cleanup();
        }
    }

    async _fetchRedistribution(socketManager, roomContext, solverOptions = {}) {
        const skews = this.engine.getAllRows({
            columns: [
                'tnum','side','quoteType',
                'refBid','refMid', 'refAsk',
                'relativeSkewTargetMkt', 'relativeSkewTargetSide', 'QT',
                'refSkew', 'newLevelPx', 'newLevelSpd', 'newLevel',
                'refMktMkt', 'refMktDisp'
            ]
        });

        // Build solverOptions payload — strip null/undefined values
        const cleanOpts = {};
        for (const [k, v] of Object.entries(solverOptions)) {
            if (v != null) cleanOpts[k] = v;
        }

        const response = await socketManager._sendWebSocketMessage({
            action: ACTION_MAP.get('redistribute'),
            context: roomContext,
            data: {params: {skews: skews}},
            solverOptions: cleanOpts,
        }, {wait: true, timeout: 15000});
        return response?.data;
    }

    async _showRedistributeModal(res, roomContext, {mm, sm, toast, esc, cleanup, onPortfolioPublish}) {

        // -- Handle error responses ------------------------------------------
        if (!res || res.error) {
            cleanup();
            const errorMsg = res?.error || 'Failed to compute redistribution.';
            await mm.show({
                title: 'Redistribute Proceeds — Error',
                body: `<div style="padding:12px;color:var(--danger, #e74c3c);">${esc(errorMsg)}</div>`,
                fields: null,
                buttons: [
                    {text: 'Close', value: 'close'},
                    {text: 'Re-run', value: 'rerun', class: 'btn-primary'}
                ],
                modalBoxClass: 'pt-redistribute-modal-error'
            }).then(action => {
                const SCFG_STORAGE_KEY = 'pt:solverConfig';
                const _readScfgCache = () => {
                    try { return JSON.parse(sessionStorage.getItem(SCFG_STORAGE_KEY)) || {}; } catch { return {}; }
                };
                const cached = _readScfgCache();
                // -- Solver-options state (lives across the modal lifecycle) ------
                const opts = {
                    isolate_traders: cached.isolate_traders ?? false,
                    max_individual_spread_skew_delta: cached.max_individual_spread_skew_delta ?? null,
                    max_individual_px_skew_delta: cached.max_individual_px_skew_delta ?? null,
                    match_pivot_groups: cached.match_pivot_groups ?? false,
                    group_columns: cached.group_columns ?? null,
                    allow_through_mid: cached.allow_through_mid ?? true,
                    lambda_param: cached.lambda_param ?? 0.25,
                    gamma: cached.gamma ?? 1.0,
                    trader_pull: cached.trader_pull ?? 1.0,
                    target_blend: cached.target_blend ?? 0.50,
                };
                if (action === 'rerun') this._redistributeProceeds(opts);
            });
            return;
        }

        const {detail, removed, summary, trader, updates, result, group_columns: resGroupCols} = res;

        // ── Diff view: compare vs previous run ────────────────────────────
        const _DIFF_KEY = 'pt:redistPrev';
        const _portfolioKey = roomContext?.room || 'default';
        const _groupSig = JSON.stringify(resGroupCols || ['desigName']);
        let _prevRun = null;
        let _diffAvailable = false;
        try {
            const raw = sessionStorage.getItem(_DIFF_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed.portfolioKey === _portfolioKey && parsed.groupSig === _groupSig) {
                    _prevRun = parsed;
                    _diffAvailable = true;
                }
            }
        } catch {}
        // Build lookup indexes from previous run
        const _prevDetailIdx = new Map();
        const _prevTraderIdx = new Map();
        if (_prevRun) {
            for (const b of (_prevRun.detail || [])) _prevDetailIdx.set(b.tnum, b);
            for (const t of (_prevRun.trader || [])) {
                const k = `${t.desigName || ''}|${t.side}|${t.quoteType}`;
                _prevTraderIdx.set(k, t);
            }
        }
        // Save current run for next comparison
        try {
            sessionStorage.setItem(_DIFF_KEY, JSON.stringify({
                portfolioKey: _portfolioKey,
                groupSig: _groupSig,
                detail: (detail || []).map(b => ({
                    tnum: b.tnum, skew: b.skew, bucket_effect: b.bucket_effect,
                    group_effect: b.group_effect, skew_delta: b.skew_delta,
                    proceeds_delta: b.proceeds_delta,
                })),
                trader: (trader || []).map(t => ({
                    desigName: t.desigName, side: t.side, quoteType: t.quoteType,
                    wavg_start_skew: t.wavg_start_skew, wavg_bucket_effect: t.wavg_bucket_effect,
                    wavg_trader_effect: t.wavg_trader_effect, wavg_skew_delta: t.wavg_skew_delta,
                    proceeds_delta: t.proceeds_delta,
                })),
            }));
        } catch {}

        // Determine the grouping key column(s) — defaults to desigName when not using pivot groups
        const _groupCols = (resGroupCols && resGroupCols.length > 0) ? resGroupCols : ['desigName'];
        const _groupLabel = _groupCols.length === 1
            ? (_groupCols[0] === 'desigName' ? 'Trader' : 'Group')
            : 'Group';
        // Helper to build composite key from a row using the group columns
        const _buildGroupKey = (row) => _groupCols.map(c => row[c] ?? '').join('|');
        // Helper to build display value from a row
        const _buildGroupDisplay = (row) => _groupCols.map(c => row[c] ?? '').join(' / ');
        if (!updates || updates.length === 0) {
            toast.info('Redistribute', 'No eligible rows for redistribution.');
            cleanup();
            return;
        }

        // -- Formatters ------------------------------------------------------
        const fmtNum = (v, decimals = 4, signed = false, percent=false) => {
            if (v == null || v === '' || !Number.isFinite(+v)) return '-';
            if (percent && Number(v) === 0) return '-';
            let r = (+v).toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
            r = signed ? (Number(v) > 0 ? '+' : '') + r : r;
            if (percent) r += '%';
            return r
        };

        const fmtK = (v, signed=false, prefix="", suffix="") => {
            if (v == null || !Number.isFinite(+v)) return '-';
            const n = +v;
            let r = ''
            if (Math.abs(n) >= 1_000_000) r= (n / 1_000_000).toFixed(1) + 'm';
            else if (Math.abs(n) >= 1_000) r = (n / 1_000).toFixed(1) + 'k';
            else  r = (n / 1_000).toFixed(1) + 'k';
            let base = '<div class="fmtk-wrapper">';
            if (prefix && prefix !== '') base += `<span>${prefix}</span>`;
            base += `<span>${(signed && n>0 ? '+' : "") + r}</span>`
            if (suffix && suffix !== '') base += `<span>${suffix}</span>`
            base += `</div>`;
            return base
        };

        // -- Delta colour helper (same sign convention as original) ----------
        const deltaColor = (v, qt, side) => {
            if (v == null || !Number.isFinite(+v) || +v === 0) return '';
            const n = +v;
            const good = 'color:var(--success-skew, #2ecc71);';
            const bad = 'color:var(--danger-skew, #e74c3c);';
            if (qt === 'PX' && side === 'SELL') return n > 0 ? good : bad;
            if (qt === 'PX' && side === 'BUY') return n < 0 ? good : bad;
            if (qt === 'SPD' && side === 'BUY') return n > 0 ? good : bad;
            if (qt === 'SPD' && side === 'SELL') return n < 0 ? good : bad;
            return '';
        };


        // ----------------------------------------------------------------------
        // 1) GLOBAL STATUS BAR (compact)
        // ----------------------------------------------------------------------
        let constantsHtml = '';
        if (result) {
            let status = Array.isArray(result) ? result[0] : result;
            status = status?.toString()?.toUpperCase() || 'FAILED';

            // Global status label
            const statusClass = status === 'OPTIMAL' ? 'solver-optimal' : status === 'PARTIAL' ? 'solver-partial' : 'solver-failed';
            constantsHtml += `<div class="solver-top-region" style="display:flex;align-items:center;gap:12px;padding:6px 0;">`;
            constantsHtml += `<span class="solver-result ${statusClass}">${status !== 'PARTIAL' ? 'ALL' : ''} ${esc(status)}</span>`;

            // Removal warnings (compact)
            if (Object.keys(removed).length) {

                const r = new Map();
                for (const rp of Object.entries(removed)) {
                    const idx = rp[0];
                    const rr = rp[1];
                    if (!r.has(rr)) r.set(rr, [0, []]);
                    const current = r.get(rr);
                    const new_count = current[0] + 1;
                    const ri = this.engine.getRowIndexById(idx);
                    const des = this.engine.getRowObject(ri, ['description'])?.description;
                    if (des) current[1].push(des);
                    r.set(rr, [new_count, current[1]]);
                }
                for (const rem of r.entries()) {
                    const reason = rem[0];
                    const count = rem[1][0];
                    const bonds = rem[1][1].join('\n');
                    constantsHtml += `<span class="removal-entry tooltip tooltip-right" data-tooltip="${bonds}" style="font-size:11px;opacity:0.7;"><span class="removal-reason">${reason}: </span><span class="removal-count">${count}</span></span>`;
                }
            }
            constantsHtml += `</div>`; // close solver-top-region
            }

        // Diagnostics from backend (if available)
        const diag = res.diagnostics || {};

        // Infeasibility diagnosis
        const _infReasons = diag.infeasibility_reasons || [];
        if (_infReasons.length > 0) {
            constantsHtml += `<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:6px;padding:8px 12px;margin:6px 0;">`;
            constantsHtml += `<div style="color:#ef4444;font-weight:600;font-size:12px;margin-bottom:4px;">Solver could not find a feasible solution</div>`;
            for (const reason of _infReasons) {
                constantsHtml += `<div style="color:#fca5a5;font-size:11px;padding:1px 0;">• ${esc(reason)}</div>`;
            }
            constantsHtml += `</div>`;
        }


        // ── Diff compare toggle ────────────────────────────────────────────
        const _diffToggleId = `rdm-diff-toggle-${Math.random().toString(36).slice(2,8)}`;
        const _diffSummaryId = `rdm-diff-summary-${Math.random().toString(36).slice(2,8)}`;
        constantsHtml += `<div style="display:flex;align-items:center;gap:8px;padding:2px 0;">`;
        constantsHtml += `<label style="display:flex;align-items:center;gap:4px;font-size:11px;opacity:${_diffAvailable ? '1' : '0.35'};cursor:${_diffAvailable ? 'pointer' : 'not-allowed'};">`;
        constantsHtml += `<input type="checkbox" id="${_diffToggleId}" ${_diffAvailable ? '' : 'disabled'} style="margin:0;"/>`;
        constantsHtml += `<span>Compare vs previous run</span></label>`;
        constantsHtml += `<span id="${_diffSummaryId}" style="font-size:11px;display:none;"></span>`;
        constantsHtml += `</div>`;

        // Diff annotation helper: returns a tiny HTML string showing the change
        let _diffEnabled = false;
        const _diffAnnotation = (cur, prev, fmt) => {
            if (!_diffEnabled || prev == null || cur == null) return '';
            const d = (+cur) - (+prev);
            if (!Number.isFinite(d) || Math.abs(d) < 0.005) return '';
            const arrow = d > 0 ? '↑' : '↓';
            const display = fmt ? fmt(Math.abs(d)) : Math.abs(d).toFixed(2);
            const color = 'rgba(148,163,184,0.7)';
            return `<div class="rdm-diff-ann" style="font-size:9px;color:${color};line-height:1;" title="change from previous run">${arrow}${display}</div>`;
        };
        // Keys that get diff annotations
        const _diffKeys = new Set(['wavg_start_skew','wavg_bucket_effect','wavg_trader_effect','wavg_skew_delta','proceeds_delta',
                                   'skew','bucket_effect','group_effect','skew_delta']);

        const CHEVRON_SVG = `<svg class="rdm-chevron" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 3L11 8L6 13" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

        const summaryDisplayCols = [
            {key: '_expand', label: '', width: 'var(--expand-width, 10px)'},
            {key: '_group_display', label: _groupLabel, tip: '', fmt: null},
            {key: 'side', label: 'Side', tip: ''},
            {key: 'quoteType', label: 'QT', tip: ''},
            {key: 'size', label: 'Size', tip: '', fmt: v => fmtK(v)},
            {key: 'dv01', label: 'DV01', tip: '', fmt: v => fmtK(v)},
            {key: 'risk_pct', label: 'Risk', tip: '% Share', fmt: v => fmtNum(v * 100, 0, false, true)},
            {key: 'wavg_liq_score', label: 'Liq', tip: '', fmt: v => fmtNum(v, 1, false)},
            {key: 'pct_bsr', label: '% BSR', tip: '', fmt: v => fmtNum(v * 100, 0, false, true)},
            {key: 'wavg_start_skew', label: 'Start', tip: 'Skew', fmt: v => fmtNum(v, 2, true)},
            {key: 'wavg_bucket_effect', label: 'Curve', tip: 'Effect', fmt: v => fmtNum(v, 2, true), isDelta: true},
            {key: 'wavg_trader_effect', label: 'Residual', tip: 'Effect', fmt: v => fmtNum(v, 2, true), isDelta: true},
            {key: 'wavg_anchor_adj', label: 'Adj', tip: 'Anchor', fmt: v => fmtNum(v, 2, true), isDelta: true, isAdj: true},
            {key: 'wavg_final_skew', label: 'Final', tip: 'Skew', fmt: v => fmtNum(v, 2, true), highlight: true},
            {key: 'wavg_skew_delta', label: 'Wavg Δ', tip: 'Delta', fmt: v => fmtNum(v, 2, true), isDelta: true},
            {key: 'proceeds_delta', label: 'PNL Δ', tip: 'Delta', fmt: v => fmtK(v, true, "$"), isPnl: true},
        ];

        // Header row
        const headerRow = summaryDisplayCols.map(c => {
            const w = c.width ? `width:${c.width};` : '';
            const tipHtml = c.tip ? `<div class="rdm-header-wrapper"><span class="rdm-tip">${c.tip}</span></div>` : '';
            const sortable = c.key !== '_expand';
            const sortAttr = sortable ? ` data-sort-key="${c.key}" style="${w}cursor:pointer;user-select:none;"` : ` style="${w}"`;
            return `<th${sortAttr}>${c.label}${tipHtml}</th>`;
        }).join('');

        // Index detail by (desigName, side, quoteType) for fast lookup
        const detailIndex = new Map();
        if (detail && detail.length) {
            for (const bond of detail) {
                const k = `${_buildGroupKey(bond)}|${bond.side}|${bond.quoteType}`;
                if (!detailIndex.has(k)) detailIndex.set(k, []);
                detailIndex.get(k).push(bond);
            }
        }

        // -- Bond detail sub-table builder -----------------------------------
        const bondDetailCols = [
            {key: 'description', label: 'Bond', tip: ''},
            // {key: 'side', label: '', tip: '', fmt: v => ''}, //gap
            // {key: 'quoteType', label: '', tip: '', fmt: v => ''}, //gap
            {key: 'grossSize', label: 'Size', tip: '', fmt: v => fmtK(v)},
            {key: 'grossDv01', label: 'DV01', tip: '', fmt: v => fmtK(v)},
            {key: 'risk_pct', label: 'Risk', tip: '% Share', fmt: v => fmtNum(v * 100, 1, false, true)},
            {key: 'avgLiqScore', label: 'Liq', tip: '', fmt: v => fmtNum(v, 1), className: (v) => `liq-td liq-${v.toFixed(0)}`},
            {key: 'bsr_pct', label: '% BSR', tip: '', fmt: v => fmtNum(v * 100, 0, false, true)},
            {key: 'skew', label: 'Start', tip: 'Skew', fmt: v => fmtNum(v, 2, true)},
            {key: 'bucket_effect', label: 'Curve', tip: 'Effect', fmt: v => fmtNum(v, 2, false), isDelta: true},
            {key: 'group_effect', label: 'Residual', tip: 'Effect', fmt: v => fmtNum(v, 2, false), isDelta: true},
            {key: 'anchor_adj', label: 'Adj', tip: 'Anchor', fmt: v => fmtNum(v, 2, false), isDelta: true, isAdj: true},
            {key: 'final_skew', label: 'Final', tip: 'Skew', fmt: v => fmtNum(v, 2, true), highlight: true},
            {key: 'skew_delta', label: 'Delta Δ', tip: 'Skew', fmt: v => fmtNum(v, 2, true), isDelta: true},
            {key: 'proceeds_delta', label: 'PNL Δ', tip: 'Delta', fmt: v => fmtK(v, true, "$"), isPnl: true},
            {key: 'binding_constraint', label: 'Bound', tip: 'Constraint', fmt: v => {
                if (!v) return '';
                const labels = {side: 'SIDE', trader: 'TRADER', mid: 'MID', cap: 'CAP', locked: 'LOCK'};
                const colors = {side: '#f59e0b', trader: '#8b5cf6', mid: '#3b82f6', cap: '#ef4444', locked: '#6b7280'};
                const label = labels[v] || v;
                const color = colors[v] || '#9ca3af';
                return `<span style="background:${color}22;color:${color};padding:1px 5px;border-radius:3px;font-size:10px;font-weight:600;">${label}</span>`;
            }},
        ];

        const buildBondRows = (bondList, qt, side) => {
            return bondList.map(bond => {
                const prev = _prevDetailIdx.get(bond.tnum);
                return '<tr>' + bondDetailCols.map(c => {
                    const raw = bond[c.key];
                    const display = c.fmt ? c.fmt(raw) : esc(raw);
                    let style = '';
                    if (c.isDelta && raw != null && Number.isFinite(+raw)) {
                        if (c.isAdj && Math.abs(+raw) > 0.1) {
                            style += 'color:#f87171;font-weight:600;';
                        } else {
                            style += deltaColor(raw, qt, side);
                            if (Math.abs(+raw) > 0.05) style += 'font-weight:600;';
                        }
                    }
                    if (c.isPnl && raw != null && Number.isFinite(+raw)) {
                        style += deltaColor(raw, 'PX', 'SELL');
                        if (Math.abs(+raw) > 2000) style += 'font-weight:600;';
                    }
                    if (c.highlight) style += 'font-weight:600;color:var(--rdm-accent);';
                    let cl = (c?.className ? c.className : '');
                    if (typeof cl === 'function') cl = cl(raw);
                    const diff = (_diffKeys.has(c.key) && prev) ? _diffAnnotation(raw, prev[c.key], c.fmt) : '';
                    return `<td style="${style}" class="${cl}">${display}${diff}</td>`;
                }).join('') + '</tr>';
            }).join('');
        };

        const buildBondTable = (bonds, qt, side) => {
            if (!bonds || !bonds.length) return '<div style="padding:8px;opacity:0.5;">No bonds in this bucket.</div>';

            // Sort bonds by |skew_delta| descending so biggest movers are on top
            const sorted = [...bonds].sort((a, b) => Math.abs(b.skew_delta ?? 0) - Math.abs(a.skew_delta ?? 0));
            const uid = `bdt-${Math.random().toString(36).slice(2, 8)}`;

            const bTh = bondDetailCols.map(c => {
                const tipHtml = c.tip ? `<div class="rdm-header-wrapper"><span class="rdm-tip">${c.tip}</span></div>` : '';
                return `<th data-sort-key="${c.key}" style="cursor:pointer;user-select:none;">${c.label}${tipHtml}</th>`;
            }).join('');

            return `<table class="rdm-bond-table rdm-bond-sortable" data-uid="${uid}" data-qt="${qt}" data-side="${side}"><thead><tr>${bTh}</tr></thead><tbody>${buildBondRows(sorted, qt, side)}</tbody></table>`;
        };


        // -- Build trader body rows with paired detail expansion rows ---------
        let _globalRowIdx = 0;  // unique across all groups
        const buildTraderRows = (trader) => {
            let bodyRowsHtml = '';
            trader.forEach((row, localIdx) => {
                const idx = _globalRowIdx++;
                const qt = row['quoteType'];
                const side = row['side'];
                const groupDisplay = _buildGroupDisplay(row);
                // Inject the display value so the column renderer can pick it up
                row['_group_display'] = groupDisplay;
                const rowId = `rdm-row-${idx}`;
                const detailId = `rdm-detail-${idx}`;
                const detailKey = `${_buildGroupKey(row)}|${side}|${qt}`;
                const bonds = detailIndex.get(detailKey) || [];

                // Summary row
                const _prevTrader = _prevTraderIdx.get(`${row.desigName || ''}|${side}|${qt}`);
                const cells = summaryDisplayCols.map(c => {
                    if (c.key === '_expand') {
                        return `<td style="width:24px;text-align:center;padding:4px 2px;">${CHEVRON_SVG}</td>`;
                    }
                    const raw = row[c.key];
                    const display = c.fmt ? c.fmt(raw) : esc(raw);
                    let style = '';
                    if (c.isDelta && raw != null && Number.isFinite(+raw)) {
                        if (c.isAdj && Math.abs(+raw) > 0.1) {
                            style += 'color:#f87171;font-weight:600;';
                        } else {
                            style += deltaColor(raw, qt, side);
                        }
                    }
                    if (c.isPnl && raw != null && Number.isFinite(+raw)) {
                        style += deltaColor(raw, 'PX', 'SELL');
                        if (Math.abs(+raw) > 2000) style += 'font-weight:600;';
                    }
                    if (c.highlight) style += 'font-weight:600;';
                    const diff = (_diffKeys.has(c.key) && _prevTrader) ? _diffAnnotation(raw, _prevTrader[c.key], c.fmt) : '';
                    return `<td style="${style}" class="td-${qt.toString().toLowerCase()} td-${side.toString().toLowerCase()}">${display}${diff}</td>`;
                }).join('');

                const stripeClass = localIdx % 2 === 0 ? 'rdm-stripe-even' : 'rdm-stripe-odd';
                bodyRowsHtml += `<tr id="${rowId}" class="rdm-summary-row ${stripeClass}" data-detail-id="${detailId}" data-key="${esc(detailKey)}">${cells}</tr>`;

                // Detail expansion row (hidden by default)
                const colSpan = summaryDisplayCols.length;
                const bondTableHtml = buildBondTable(bonds, qt, side);
                bodyRowsHtml += `<tr class="rdm-detail-wrapper-tr">
            <td colspan="${colSpan}">
                <div id="${detailId}" class="rdm-detail-outer">
                    <div class="rdm-detail-inner">
                        <div class="rdm-detail-label">${esc(groupDisplay)} - ${side} ${qt} - ${bonds.length} bond${bonds.length !== 1 ? 's' : ''}</div>
                        ${bondTableHtml}
                    </div>
                </div>
            </td>
        </tr>`;
            })
            return bodyRowsHtml;
        };
        // -- Group trader rows for separate tables -----------
        // In isolated mode: one table per (trader, qt, side)
        // Normal mode: one table per (qt, side)
        // Detect isolated mode from diagnostics OR from detail data having _isolation_group
        const _isIsolated = (diag?.isolated === true)
            || (detail && detail.length > 0 && detail[0]._isolation_group != null);
        if (_isIsolated) console.log('[redistribute] Isolated mode detected — building per-trader tables');
        const qtSideGroups = new Map();
        trader.forEach(row => {
            const key = _isIsolated
                ? `${_buildGroupKey(row)}|${row.quoteType}|${row.side}`
                : `${row.quoteType}|${row.side}`;
            if (!qtSideGroups.has(key)) qtSideGroups.set(key, []);
            qtSideGroups.get(key).push(row);
        });

        // Build per-basket summary lookup from summary data
        const _summaryByBasket = {};
        if (summary && summary.length) {
            summary.forEach(s => {
                _summaryByBasket[`${s.quoteType}|${s.side}`] = s;
            });
        }

        const buildGroupTable = (groupKey, rows, idx) => {
            // Parse key: either "QT|SIDE" or "TRADER|QT|SIDE"
            const parts = groupKey.split('|');
            let traderName = null, qt, side;
            if (_isIsolated && parts.length >= 3) {
                traderName = parts.slice(0, -2).join('|');  // handle trader names with pipes
                qt = parts[parts.length - 2];
                side = parts[parts.length - 1];
            } else {
                [qt, side] = parts;
            }
            const themeClass = qt === 'PX' ? 'rdm-theme-px' : 'rdm-theme-spd';
            const tbodyId = `rdm-tbody-${idx}`;
            const groupRows = buildTraderRows(rows);

            // Per-basket summary stats
            const bSummary = _summaryByBasket[`${qt}|${side}`] || {};
            const wavgStart = bSummary.wavg_start_skew;
            const wavgFinal = bSummary.wavg_final_skew;
            const wavgDelta = (wavgStart != null && wavgFinal != null) ? wavgFinal - wavgStart : null;

            // Compute PnL delta from the rows in THIS table
            const tablePnl = rows.reduce((s, r) => s + (r.proceeds_delta || 0), 0);

            // Title: "SPD / BUY" or "TraderA — SPD / BUY"
            const titleLabel = traderName
                ? `${esc(traderName)} — ${esc(qt)} / ${esc(side)}`
                : `${esc(qt)} / ${esc(side)}`;

            // Per-trader diagnostics in isolated mode
            let diagHtml = '';
            if (_isIsolated && diag?.per_group && traderName && diag.per_group[traderName]) {
                const dp = diag.per_group[traderName];
                diagHtml = `<div class="rdm-diag-row">
                    <span class="rdm-diag-item" title="Change in PnL from optimization"><b>PnL Δ</b> ${fmtK(dp.pnl_delta, true, '$')}</span>
                    <span class="rdm-diag-item" title="PnL sacrificed to keep bonds near current levels"><b>Stability cost</b> ${fmtK(dp.smoothness_penalty, false, '-$')}</span>
                    <span class="rdm-diag-item" title="PnL sacrificed to follow the charge curve shape"><b>Shaping cost</b> ${fmtK(dp.curve_penalty, false, '-$')}</span>
                    <span class="rdm-diag-item" title="Average skew change per bond"><b>Avg Δ</b> ${fmtNum(dp.avg_delta, 2)}bps</span>
                    <span class="rdm-diag-item" title="Largest move on any single bond"><b>Max Δ</b> ${fmtNum(dp.max_delta, 2)}bps</span>
                    <span class="rdm-diag-item" title="How many bonds moved more than 0.1 bps"><b>Moved</b> ${dp.bonds_moved_1 ?? '?'}/${dp.bonds_unlocked ?? '?'}</span>
                </div>`;
            } else if (!_isIsolated && diag && diag.status) {
                const dp = diag;
                diagHtml = `<div class="rdm-diag-row">
                    <span class="rdm-diag-item" title="Change in PnL from optimization"><b>PnL Δ</b> ${fmtK(dp.pnl_delta, true, '$')}</span>
                    <span class="rdm-diag-item" title="PnL sacrificed to keep bonds near current levels (higher anchoring = higher cost)"><b>Stability cost</b> ${fmtK(dp.smoothness_penalty, false, '-$')}</span>
                    <span class="rdm-diag-item" title="PnL sacrificed to follow the charge curve shape (higher curve shaping = higher cost)"><b>Shaping cost</b> ${fmtK(dp.curve_penalty, false, '-$')}</span>
                    <span class="rdm-diag-item" title="Average skew change per bond"><b>Avg Δ</b> ${fmtNum(dp.avg_delta, 2)}bps</span>
                    <span class="rdm-diag-item" title="Largest move on any single bond"><b>Max Δ</b> ${fmtNum(dp.max_delta, 2)}bps</span>
                    <span class="rdm-diag-item" title="How many bonds moved more than 0.1 bps"><b>Moved</b> ${dp.bonds_moved_1 ?? '?'}/${dp.bonds_unlocked ?? '?'}</span>
                </div>`;
            }

            // Apply Slice: include trader filter for isolated mode
            const applyAttrs = `data-bucket-qt="${qt}" data-bucket-side="${side}"${traderName ? ` data-bucket-trader="${esc(traderName)}"` : ''}`;

            return `<div class="rdm-group-section ${themeClass}">
                <div class="rdm-group-header">
                    <div class="rdm-group-label">${titleLabel}${!_isIsolated && wavgDelta != null ? ` <span class="rdm-group-delta" style="${deltaColor(wavgDelta, qt, side)}">Δ ${fmtNum(wavgDelta, 2, true)}</span>` : ''}</div>
                    <div class="rdm-group-actions">
                        <button class="rdm-bucket-apply-btn" ${applyAttrs}>Apply Slice</button>
                    </div>
                </div>
                ${diagHtml}
                <div class="rdm-table-wrapper">
                    <table class="rdm-summary-table" style="width:100%;border-collapse:collapse;">
                        <thead><tr>${headerRow}</tr></thead>
                        <tbody class="rdm-tbody" data-tbody-idx="${idx}">${groupRows}</tbody>
                    </table>
                </div>
            </div>`;
        };

        let groupTablesHtml = '';
        let groupIdx = 0;
        for (const [key, rows] of qtSideGroups) {
            groupTablesHtml += buildGroupTable(key, rows, groupIdx++);
        }
        console.log(`[redistribute] Built ${groupIdx} tables from ${trader.length} trader rows (isolated=${_isIsolated})`);


        // ----------------------------------------------------------------------
        // 4) ASSEMBLE FINAL MODAL HTML
        // ----------------------------------------------------------------------
        const staleId = `redist-stale`;
        const tableHtml = `
        <div class="redist-modal-body rdm-style-root" id="rdm-modal-body">
            ${constantsHtml}
            <div id="${staleId}" style="display:none;padding:6px 10px;background:rgba(234,179,8,0.12);color:#fbbf24;border-radius:4px;font-size:11px;margin:4px 0;">
                Portfolio data has changed since this was computed. Re-run recommended.
            </div>
            <div class="rdm-scroll-region">
            ${groupTablesHtml}
            </div>
        </div>
        <div class="rdm-legend-wrapper">
            <div class="rdm-table-legend">
                <span title="What the charge curve says this bond should move based on BSR/BSI/liquidity"><b>Curve</b> target shift from charge curve</span>
                    <span title="Everything the optimizer did beyond what the curve wanted: PnL chasing, constraints, trader rebalancing"><b>Residual</b> optimizer adjustment beyond curve</span>
                    <span title="Parallel shift to preserve starting weighted-avg skew (Anchor WAVG Skew)"><b>Adj</b> wavg skew anchor correction</span>
                    <span title="Curve + Residual + Adj = total change"><b>Delta</b> = Curve + Residual + Adj</span>
            </div>
        </div>
    `;


        // ----------------------------------------------------------------------
        // 5) SHOW MODAL  &  ATTACH EXPAND/COLLAPSE HANDLERS
        // ----------------------------------------------------------------------

        // Stale data polling (same as original)
        let staleCheckInterval = null;
        const startStaleCheck = () => {
            staleCheckInterval = setInterval(() => {
                const el = document.getElementById(staleId);
                if (el && this._redistStale) el.style.display = 'block';
            }, 500);
        };
        startStaleCheck();

        // Launch modal (returns a promise that resolves on button click)
        const modalPromise = mm.show({
            title: 'Redistribute Proceeds',
            body: tableHtml,
            fields: null,
            buttons: [
                {text: 'Cancel', value: 'cancel'},
                {text: 'Adjust Config', value: 'config'},
                {text: 'Re-run', value: 'rerun'},
                {text: 'Apply', value: 'apply', class: 'btn-primary'}
            ],
            modalBoxClass: 'pt-redistribute-modal'
        });

        const _setSortIndicator = (th, indicator) => {
            // Find or create the sort-arrow span (preserves tip spans)
            let arrow = th.querySelector('.rdm-sort-arrow');
            let inner = th.querySelector('.rdm-header-wrapper') ?? th;
            if (indicator) {
                if (!arrow) {
                    arrow = document.createElement('span');
                    arrow.className = 'rdm-sort-arrow';
                    inner.appendChild(arrow);
                }
                arrow.textContent = indicator;
            } else if (arrow) {
                arrow.remove();
            }
        };

        // -- Diff toggle listener -----------------------------------------------
        requestAnimationFrame(() => {
            const diffToggle = document.getElementById(_diffToggleId);
            if (diffToggle) {
                diffToggle.addEventListener('change', () => {
                    _diffEnabled = diffToggle.checked;
                    // Toggle visibility of all diff annotations
                    const anns = document.querySelectorAll('.rdm-diff-ann');
                    for (const el of anns) el.style.display = _diffEnabled ? '' : 'none';
                    // Show/hide summary
                    const sumEl = document.getElementById(_diffSummaryId);
                    if (sumEl) {
                        if (_diffEnabled && _prevRun) {
                            // Build a quick summary comparing wavg_skew_delta per bucket
                            const parts = [];
                            for (const cur of (summary || [])) {
                                const prev = (_prevRun.trader || []).find(p =>
                                    p.side === cur.side && p.quoteType === cur.quoteType && !p.desigName);
                                // Can't match overall summary easily, skip
                            }
                            sumEl.textContent = 'Showing changes from previous run';
                            sumEl.style.display = '';
                        } else {
                            sumEl.style.display = 'none';
                        }
                    }
                });
            }
            // Hide all diff annotations initially (toggle is off)
            for (const el of document.querySelectorAll('.rdm-diff-ann')) el.style.display = 'none';
        });

        // -- Attach row-expansion logic after DOM render ---------------------
        requestAnimationFrame(() => {
            const allTbodies = document.querySelectorAll('.rdm-tbody');
            if (!allTbodies.length) return;

            // FIFO expansion queue - max 2
            const expandedQueue = []; // array of {rowEl, detailEl}

            const collapseEntry = (entry) => {
                entry.rowEl.classList.remove('rdm-expanded');
                entry.detailEl.classList.remove('rdm-open');
            };

            const expandEntry = (entry) => {
                entry.rowEl.classList.add('rdm-expanded');
                // Force reflow before adding class so transition triggers
                void entry.detailEl.offsetHeight;
                entry.detailEl.classList.add('rdm-open');
            };

            allTbodies.forEach(tbody => {
                tbody.addEventListener('click', (e) => {
                    const summaryRow = e.target.closest('.rdm-summary-row');
                    if (!summaryRow) return;

                    const detailId = summaryRow.getAttribute('data-detail-id');
                    const detailEl = document.getElementById(detailId);
                    if (!detailEl) return;

                    const isOpen = summaryRow.classList.contains('rdm-expanded');

                    if (isOpen) {
                        collapseEntry({rowEl: summaryRow, detailEl});
                        const qIdx = expandedQueue.findIndex(e => e.rowEl === summaryRow);
                        if (qIdx !== -1) expandedQueue.splice(qIdx, 1);
                    } else {
                        if (expandedQueue.length >= 2) {
                            const oldest = expandedQueue.shift();
                            collapseEntry(oldest);
                        }
                        const entry = {rowEl: summaryRow, detailEl};
                        expandedQueue.push(entry);
                        expandEntry(entry);
                    }
                });
            });

            // -- Sortable column headers (per-group table) --------------------
            // Build a map from tbody index to its subset of trader rows
            const tbodyGroupMap = new Map();
            let gIdx = 0;
            for (const [, rows] of qtSideGroups) {
                tbodyGroupMap.set(String(gIdx++), rows);
            }

            allTbodies.forEach(tbody => {
                const thead = tbody.closest('table')?.querySelector('thead');
                if (!thead) return;
                const tIdx = tbody.getAttribute('data-tbody-idx');
                const groupRows = tbodyGroupMap.get(tIdx) || [];

                let sortKey = null;
                let sortAsc = true;
                const SORT_ASC = ' ▲';
                const SORT_DESC = ' ▼';

                const updateHeaderIndicators = () => {
                    thead.querySelectorAll('th[data-sort-key]').forEach(th => {
                        const key = th.getAttribute('data-sort-key')
                        _setSortIndicator(th, key === sortKey ? (sortAsc ? SORT_ASC : SORT_DESC) : null);
                    });
                };

                thead.addEventListener('click', (e) => {
                    const th = e.target.closest('th[data-sort-key]');
                    if (!th) return;

                    const key = th.getAttribute('data-sort-key');
                    if (sortKey === key) {
                        sortAsc = !sortAsc;
                    } else {
                        sortKey = key;
                        sortAsc = true;
                    }

                    const dir = sortAsc ? 1 : -1;
                    const sorted = [...groupRows].sort((a, b) => {
                        const av = a[key];
                        const bv = b[key];
                        if (av == null && bv == null) return 0;
                        if (av == null) return 1;
                        if (bv == null) return -1;
                        if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
                        return String(av).localeCompare(String(bv)) * dir;
                    });

                    // Collapse all expanded rows before re-render
                    while (expandedQueue.length) collapseEntry(expandedQueue.pop());

                    tbody.innerHTML = buildTraderRows(sorted);
                    updateHeaderIndicators();
                });
            });

            // -- Bucket-level Apply buttons ---------------------------------------
            document.querySelectorAll('.rdm-bucket-apply-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const qt = btn.getAttribute('data-bucket-qt');
                    const side = btn.getAttribute('data-bucket-side');
                    const trdr = btn.getAttribute('data-bucket-trader') || '';
                    const dialog = btn.closest('dialog');
                    if (dialog) dialog.close(`apply_bucket:${qt}/${side}/${trdr}`);
                });
            });

            // -- Sortable bond detail sub-table headers -----------------------
            const bondSortState = new Map(); // uid -> {key, asc}
            const modalBody = document.getElementById('rdm-modal-body');
            if (modalBody) {
                modalBody.addEventListener('click', (e) => {
                    const th = e.target.closest('.rdm-bond-sortable th[data-sort-key]');
                    if (!th) return;

                    const table = th.closest('.rdm-bond-sortable');
                    if (!table) return;

                    const uid = table.getAttribute('data-uid');
                    const qt = table.getAttribute('data-qt');
                    const side = table.getAttribute('data-side');
                    const key = th.getAttribute('data-sort-key');

                    const detailKey = th.closest('.rdm-detail-inner')
                        ?.closest('td')?.closest('tr')?.previousElementSibling
                        ?.getAttribute('data-key');
                    const bonds = detailKey ? (detailIndex.get(detailKey) || []) : [];
                    if (!bonds.length) return;

                    const prev = bondSortState.get(uid);
                    let asc = true;
                    if (prev && prev.key === key) asc = !prev.asc;
                    bondSortState.set(uid, {key, asc});

                    const dir = asc ? 1 : -1;
                    const sorted = [...bonds].sort((a, b) => {
                        const av = a[key];
                        const bv = b[key];
                        if (av == null && bv == null) return 0;
                        if (av == null) return 1;
                        if (bv == null) return -1;
                        if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
                        return String(av).localeCompare(String(bv)) * dir;
                    });

                    const bondTbody = table.querySelector('tbody');
                    if (bondTbody) bondTbody.innerHTML = buildBondRows(sorted, qt, side);

                    // Update bond header indicators
                    table.querySelectorAll('th[data-sort-key]').forEach(h => {
                        _setSortIndicator(h, h.getAttribute('data-sort-key') === key ? (asc ? ' ▲' : ' ▼') : null);
                    });
                });
            }
        });


        // -- Await user's button choice --------------------------------------
        const action = await modalPromise;
        clearInterval(staleCheckInterval);

        if (action === 'config') {
            cleanup();
            this.topLock.click();
            return;
        }

        if (action === 'rerun') {
            cleanup();
            const SCFG_STORAGE_KEY = 'pt:solverConfig';
            const _readScfgCache = () => {
                try { return JSON.parse(sessionStorage.getItem(SCFG_STORAGE_KEY)) || {}; } catch { return {}; }
            };
            const cached = _readScfgCache();
            // -- Solver-options state (lives across the modal lifecycle) ------
            const opts = {
                isolate_traders: cached.isolate_traders ?? false,
                anchor_wavg_skew: cached.anchor_wavg_skew ?? false,
                max_individual_spread_skew_delta: cached.max_individual_spread_skew_delta ?? null,
                max_individual_px_skew_delta: cached.max_individual_px_skew_delta ?? null,
                match_pivot_groups: cached.match_pivot_groups ?? false,
                group_columns: cached.group_columns ?? null,
                allow_through_mid: cached.allow_through_mid ?? true,
                lambda_param: cached.lambda_param ?? 0.25,
                gamma: cached.gamma ?? 1.0,
                trader_pull: cached.trader_pull ?? 1.0,
                target_blend: cached.target_blend ?? 0.50,
            };
            return this._redistributeProceeds(opts);
        }

        const isBucketApply = typeof action === 'string' && action.startsWith('apply_bucket:');
        if (action !== 'apply' && !isBucketApply) {
            cleanup();
            return;
        }

        // -- Determine bucket filter (null = apply all) ----------------------
        let bucketFilter = null;
        if (isBucketApply) {
            const filterParts = action.slice('apply_bucket:'.length).split('/');
            const bQt = filterParts[0];
            const bSide = filterParts[1];
            const bTrader = filterParts[2] || '';
            bucketFilter = {quoteType: bQt, side: bSide, trader: bTrader || null};
        }

        // -- Stale-data double confirmation (same as original) ---------------
        if (this._redistStale) {
            const bucketLabel = bucketFilter
                ? ` for ${bucketFilter.trader ? bucketFilter.trader + ' ' : ''}${bucketFilter.quoteType}/${bucketFilter.side}`
                : '';
            const confirm = await mm.show({
                title: 'Stale Data Warning',
                body: `<div style="padding:12px;">
                <p style="color:var(--danger, #e74c3c);font-weight:600;margin-bottom:8px;">Portfolio data has changed since this redistribution was computed.</p>
                <p>Applying${bucketLabel} may produce unexpected results.</p>
            </div>`,
                fields: null,
                buttons: [
                    {text: 'Cancel', value: 'cancel'},
                    {text: 'Re-run Instead', value: 'rerun'},
                    {text: 'Apply Anyway', value: 'apply', class: 'btn-danger'}
                ],
                modalBoxClass: 'pt-redistribute-modal-stale'
            });

            if (confirm === 'rerun') {
                cleanup();
                const SCFG_STORAGE_KEY = 'pt:solverConfig';
                const _readScfgCache = () => {
                    try { return JSON.parse(sessionStorage.getItem(SCFG_STORAGE_KEY)) || {}; } catch { return {}; }
                };
                const cached = _readScfgCache();
                // -- Solver-options state (lives across the modal lifecycle) ------
                const opts = {
                    isolate_traders: cached.isolate_traders ?? false,
                    max_individual_spread_skew_delta: cached.max_individual_spread_skew_delta ?? null,
                    max_individual_px_skew_delta: cached.max_individual_px_skew_delta ?? null,
                    match_pivot_groups: cached.match_pivot_groups ?? false,
                    group_columns: cached.group_columns ?? null,
                    allow_through_mid: cached.allow_through_mid ?? true,
                    lambda_param: cached.lambda_param ?? 0.25,
                    gamma: cached.gamma ?? 1.0,
                    trader_pull: cached.trader_pull ?? 1.0,
                    target_blend: cached.target_blend ?? 0.50,
                };
                return this._redistributeProceeds(opts);
            }
            if (confirm !== 'apply') {
                cleanup();
                return;
            }
        }


        // ----------------------------------------------------------------------
        // 6) APPLY UPDATES
        // ----------------------------------------------------------------------

        // Filter updates to a specific bucket if a bucket apply was triggered
        let applicableUpdates = updates;
        if (bucketFilter && detail && detail.length) {
            const id_col_name = this.engine._idProperty;
            const bucketTnums = new Set();
            for (const bond of detail) {
                const qtMatch = bond.quoteType === bucketFilter.quoteType;
                const sideMatch = bond.side === bucketFilter.side;
                const traderMatch = !bucketFilter.trader || bond.desigName === bucketFilter.trader;
                if (qtMatch && sideMatch && traderMatch) {
                    bucketTnums.add(bond[id_col_name]);
                }
            }
            applicableUpdates = updates.filter(u => bucketTnums.has(u[id_col_name]));
        }

        if (!applicableUpdates || applicableUpdates.length === 0) {
            const label = bucketFilter ? ` in ${bucketFilter.quoteType}/${bucketFilter.side}` : '';
            toast.info('Redistribute', `No rows to update${label}.`);
            cleanup();
            return;
        }

        const engine = this.engine;
        const adapter = this.adapter;

        const id_col = engine._idProperty;
        const idx_col = adapter._idxProperty;
        const update_payloads = [];
        for (let i = 0; i < applicableUpdates.length; i++) {
            const update = applicableUpdates[i];
            const rid = update?.[id_col];
            const ri = engine.getRowIndexById(rid);
            if (update?.refSkew !== undefined) {
                update_payloads.push({[idx_col]: ri, [id_col]: rid, refSkew: update['refSkew']});
            } else if (update?.newLevelPx !== undefined) {
                update_payloads.push({[idx_col]: ri, [id_col]: rid, newLevelPx: update['newLevelPx']});
            } else if (update?.newLevelSpd !== undefined) {
                update_payloads.push({[idx_col]: ri, [id_col]: rid, newLevelSpd: update['newLevelSpd']});
            }
        }
        await adapter.applyServerUpdateTransaction(update_payloads, {emitAsEdit: true});
        const bucketMsg = bucketFilter
            ? ` (${bucketFilter.trader ? bucketFilter.trader + ' ' : ''}${bucketFilter.quoteType}/${bucketFilter.side})`
            : '';
        toast.success('Proceeds Solver', `Applied redistribution to ${update_payloads.length} row(s)${bucketMsg}.`);
        cleanup();
    }

    setupTooltips() {
        const flipLock = this.flipLock.bind(this);
        const lockState = () => this.context.page.page$.get('proceedsLocked');

        const modalManager = this.context.page.modalManager();
        this.context.page.addEventListener(this.topLock, 'click', async () => {

            // -- SessionStorage helpers for solver config persistence --------
            const SCFG_STORAGE_KEY = 'pt:solverConfig';
            const _readScfgCache = () => {
                try { return JSON.parse(sessionStorage.getItem(SCFG_STORAGE_KEY)) || {}; } catch { return {}; }
            };
            const _writeScfgCache = (obj) => {
                try { sessionStorage.setItem(SCFG_STORAGE_KEY, JSON.stringify(obj)); } catch {}
            };
            const cached = _readScfgCache();

            // -- Solver-options state (lives across the modal lifecycle) ------
            const opts = {
                isolate_traders: cached.isolate_traders ?? false,
                max_individual_spread_skew_delta: cached.max_individual_spread_skew_delta ?? null,
                max_individual_px_skew_delta: cached.max_individual_px_skew_delta ?? null,
                match_pivot_groups: cached.match_pivot_groups ?? false,
                group_columns: cached.group_columns ?? null,
                allow_through_mid: cached.allow_through_mid ?? true,
                // Slider-controlled params
                lambda_param: cached.lambda_param ?? 0.25,
                gamma: cached.gamma ?? 1.0,
                trader_pull: cached.trader_pull ?? 1.0,
                target_blend: cached.target_blend ?? 0.50,
            };

            // -- Slider helper ------------------------------------------------
            const _slider = (id, label, desc, val, min, max, step, fmt) => {
                return `
                <div class="scfg-slider-row" id="${id}-row">
                    <div class="scfg-slider-header">
                        <span class="scfg-slider-label">${label}</span>
                        <span class="scfg-slider-value" id="${id}-val">${fmt(val)}</span>
                    </div>
                    <input type="range" min="${min}" max="${max}" value="${val}" step="${step}" id="${id}" class="range range-xs scfg-range-input">
                    <div class="scfg-slider-desc">${desc}</div>
                </div>`;
            };
            const _fmtDec = v => (+v).toFixed(2);

            // -- Config panel HTML -------------------------------------------
            const MODAL_STYLE = `
    <style>

    </style>`;

            const configBody = `
    ${MODAL_STYLE}
    <div class="scfg">
        <div class="scfg-notice">
            Results are previewed before applying — nothing changes until you confirm.
        </div>

        <div class="scfg-flex-wrapper">
            <div class="scfg-flex-left">
                <!-- Toggles -->
                <div class="scfg-section">
                    <div class="scfg-toggle-row">
                        <label class="scfg-toggle-row-wrapper">
                            <input type="checkbox" id="scfg-crossing" ${opts.allow_through_mid !== false ? 'checked' : ''}/>
                            <span for="scfg-crossing">Allow Crossing Mids</span>
                        </label>
                        <span class="scfg-tag">Recommended</span>
                    </div>
                    <div class="scfg-toggle-row">
                        <label class="scfg-toggle-row-wrapper">
                            <input type="checkbox" id="scfg-match-pivot-groups" ${opts.match_pivot_groups ? 'checked' : ''}/>
                            <span for="scfg-match-pivot-groups">Match Pivot Groups</span>
                        </label>
                        <span class="scfg-tag">group buckets</span>
                    </div>
                    <div class="scfg-toggle-row">
                        <label class="scfg-toggle-row-wrapper">
                            <input type="checkbox" id="scfg-isolate" ${opts.isolate_traders ? 'checked' : ''}/>
                            <span for="scfg-isolate">Solve per trader</span>
                        </label>
                        <span class="scfg-tag">Each trader optimized independently</span>
                    </div>
                    <div class="scfg-toggle-row">
                        <label class="scfg-toggle-row-wrapper">
                            <input type="checkbox" id="scfg-anchor-wavg" ${opts.anchor_wavg_skew ? 'checked' : ''}/>
                            <span for="scfg-anchor-wavg">Anchor WAVG Skew</span>
                        </label>
                        <span class="scfg-tag">Preserve starting weighted-avg skew</span>
                    </div>
                </div>
        
                <div class="scfg-divider"></div>
        
                <!-- Run From X -->
                <div class="scfg-section">
                    <div class="scfg-section-label">Run From <span style="font-weight:400;opacity:0.55;text-transform:none;letter-spacing:0;">&mdash; override starting skew per basket (blank = live skew)</span></div>
                    <div class="scfg-fields" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
                <div class="scfg-field ${cached.rfx_buy_spd != null ? '' : 'scfg-faded'}" data-field="rfx_buy_spd">
                            <label>BUY SPD <span class="scfg-unit">bps</span></label>
                    <input type="number" id="scfg-rfx-buy-spd" step="any" placeholder="live" ${cached.rfx_buy_spd != null ? `value="${cached.rfx_buy_spd}"` : ''}/>
                        </div>
                <div class="scfg-field ${cached.rfx_sell_spd != null ? '' : 'scfg-faded'}" data-field="rfx_sell_spd">
                            <label>SELL SPD <span class="scfg-unit">bps</span></label>
                    <input type="number" id="scfg-rfx-sell-spd" step="any" placeholder="live" ${cached.rfx_sell_spd != null ? `value="${cached.rfx_sell_spd}"` : ''}/>
                        </div>
                <div class="scfg-field ${cached.rfx_buy_px != null ? '' : 'scfg-faded'}" data-field="rfx_buy_px">
                            <label>BUY PX <span class="scfg-unit">pts</span></label>
                    <input type="number" id="scfg-rfx-buy-px" step="any" placeholder="live" ${cached.rfx_buy_px != null ? `value="${cached.rfx_buy_px}"` : ''}/>
                        </div>
                <div class="scfg-field ${cached.rfx_sell_px != null ? '' : 'scfg-faded'}" data-field="rfx_sell_px">
                            <label>SELL PX <span class="scfg-unit">pts</span></label>
                    <input type="number" id="scfg-rfx-sell-px" step="any" placeholder="live" ${cached.rfx_sell_px != null ? `value="${cached.rfx_sell_px}"` : ''}/>
                        </div>
                    </div>
                </div>
        
                <div class="scfg-divider"></div>
        
                <!-- Skew caps -->
                <div class="scfg-section">
                    <div class="scfg-section-label">Skew Constraints <span style="font-weight:400;opacity:0.55;text-transform:none;letter-spacing:0;">&mdash; leave blank for uncapped</span></div>
                    <div class="scfg-fields">
                        <div class="scfg-field ${opts.max_individual_spread_skew_delta != null ? '' : 'scfg-faded'}" data-field="max_individual_spread_skew_delta">
                            <label>Max individual SPD skew <span class="scfg-unit">bps</span></label>
                            <input type="number" id="scfg-max-ind-spd" step="any" placeholder="—" ${opts.max_individual_spread_skew_delta != null ? `value="${opts.max_individual_spread_skew_delta}"` : ''}/>
                        </div>
                        <div class="scfg-field ${opts.max_individual_px_skew_delta != null ? '' : 'scfg-faded'}" data-field="max_individual_px_skew_delta">
                            <label>Max individual PX skew <span class="scfg-unit">pts</span></label>
                            <input type="number" id="scfg-max-ind-px" step="any" placeholder="—" ${opts.max_individual_px_skew_delta != null ? `value="${opts.max_individual_px_skew_delta}"` : ''}/>
                        </div>
                    </div>
                </div>
            </div>
        
            <div class="scfg-flex-right">
            
                <!-- Solver tuning sliders -->
                <div class="scfg-section">
                    <div class="scfg-section-label scfg-tuner-label">
                        <span>Solver Tuning</span>
                        <span class="reset-wrapper">
                            <button class="scfg-reset-btn" id="scfg-reset" title="Reset sliders to defaults">Reset</button>
                            <label class="scfg-link-toggle">
                                <input type="checkbox" id="scfg-linked" ${cached.linked_sliders ? 'checked' : ''}/>
                                <span>Link</span>
                                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
                            </label>
                        </span>
                    </div>
                    <div class="scfg-sliders">
                        ${_slider('scfg-lambda', 'Anchoring', 
                            'How much should bonds stay near their current levels? Higher = less change.',
                            opts.lambda_param, 0, 3, 0.01, _fmtDec)}
                        ${_slider('scfg-gamma', 'Curve Shaping', 
                            'How strongly should the charge curve reshape where each bond ends up?',
                            opts.gamma, 0, 3, 0.01, _fmtDec)}
                        ${_slider('scfg-trader-pull', 'Trader Rebalancing', 
                            'How aggressively should proceeds shift between traders toward fair share?',
                            opts.trader_pull, 0, 5, 0.01, _fmtDec)}
                        ${_slider('scfg-target-blend', 'Target Aggressiveness', 
                            'What does "fair share" mean? 0 = keep as-is, 1 = full risk-based reallocation.',
                            opts.target_blend, 0, 1, 0.01, _fmtDec)}
                    </div>
                </div>
            </div>
        </div>
        <div class="scfg-warning-slot"><div class="scfg-warning-msg" id="scfg-warning" style="display:none;"></div>
    </div>`;

            // Show config modal
            const modalPromise = modalManager.show({
                title: 'Redistribute Proceeds',
                body: configBody,
                modalClass: 'proceeds-modal',
                buttons: [
                    {text: 'Cancel', value: 'cancel'},
                    {text: 'Config Model', value: 'config', class: 'btn-secondary'},
                    {text: 'Run Solver', value: 'proceed', class: 'btn-primary'}
                ]
            });

            // -- Attach live listeners after DOM paint ------------------------
            requestAnimationFrame(() => {

                const crossingEl = document.getElementById('scfg-crossing');
                if (crossingEl) {
                    crossingEl.addEventListener('change', () => {
                        opts.allow_through_mid = !crossingEl.checked;
                    });
                }

                // Match Pivot Groups toggle
                const matchPivotEl = document.getElementById('scfg-match-pivot-groups');
                if (matchPivotEl) {
                    matchPivotEl.addEventListener('change', () => {
                        opts.match_pivot_groups = matchPivotEl.checked;
                        if (matchPivotEl.checked) {
                            // Collect current pivot group columns, excluding always-present and
                            // invalid groups (desigName is already the default trader grouping;
                            // derived/computed columns don't exist on the backend DataFrame)
                            const excludeGroups = new Set(['quoteType', 'QT', 'side', 'userSide', 'desigName']);
                            const allCols = new Set(Object.keys(this.adapter?.engine?._colDefs || {}));
                            const pivotGroups = (this.ptPivot?.pivotConfig?.groupBy || [])
                                .filter(g => !excludeGroups.has(g))
                                .filter(g => allCols.size === 0 || allCols.has(g));
                            opts.group_columns = pivotGroups.length > 0 ? pivotGroups : null;
                        } else {
                            opts.group_columns = null;
                        }
                    });
                }

                // Isolate by group toggle
                const isolateEl = document.getElementById('scfg-isolate');
                if (isolateEl) {
                    isolateEl.addEventListener('change', () => {
                        opts.isolate_traders = isolateEl.checked;
                    });
                }

                // Anchor WAVG Skew toggle
                const anchorWavgEl = document.getElementById('scfg-anchor-wavg');
                if (anchorWavgEl) {
                    anchorWavgEl.addEventListener('change', () => {
                        opts.anchor_wavg_skew = anchorWavgEl.checked;
                    });
                }

                // Run From X: parse overrides into starting_skew_override dict
                const _rfxPairs = [
                    ['scfg-rfx-buy-spd',  'BUY|SPD',  'rfx_buy_spd'],
                    ['scfg-rfx-sell-spd', 'SELL|SPD',  'rfx_sell_spd'],
                    ['scfg-rfx-buy-px',   'BUY|PX',   'rfx_buy_px'],
                    ['scfg-rfx-sell-px',  'SELL|PX',   'rfx_sell_px'],
                ];
                const _syncRfx = () => {
                    const overrides = {};
                    for (const [elId, key, cacheKey] of _rfxPairs) {
                        const input = document.getElementById(elId);
                        if (!input) continue;
                        const wrapper = input.closest('.scfg-field');
                        const raw = input.value.trim();
                        if (raw === '' || isNaN(parseFloat(raw))) {
                            if (wrapper) wrapper.classList.add('scfg-faded');
                            opts[cacheKey] = null;
                        } else {
                            overrides[key] = parseFloat(raw);
                            opts[cacheKey] = parseFloat(raw);
                            if (wrapper) wrapper.classList.remove('scfg-faded');
                        }
                    }
                    opts.starting_skew_override = Object.keys(overrides).length > 0 ? overrides : null;
                    _writeScfgCache(opts);
                };
                for (const [elId] of _rfxPairs) {
                    const input = document.getElementById(elId);
                    if (input) {
                        input.addEventListener('input', _syncRfx);
                        input.addEventListener('change', _syncRfx);
                    }
                }
                // Initialize opts from cached RFX values
                _syncRfx();

                // -- Solver tuning sliders ------------------------------------
                const _lambdaEl = document.getElementById('scfg-lambda');
                const _gammaEl  = document.getElementById('scfg-gamma');
                const _tpEl     = document.getElementById('scfg-trader-pull');
                const _blendEl  = document.getElementById('scfg-target-blend');
                const _linkedEl = document.getElementById('scfg-linked');
                const _warnEl   = document.getElementById('scfg-warning');

                const _sliderEls = {
                    lambda_param: _lambdaEl,
                    gamma:        _gammaEl,
                    trader_pull:  _tpEl,
                    target_blend: _blendEl,
                };

                // Warning state detection — fixed slot, no height jump
                const _checkWarnings = () => {
                    if (!_warnEl) return;
                    const l = opts.lambda_param, g = opts.gamma, tp = opts.trader_pull, tb = opts.target_blend;
                    let msg = '', level = '';

                    if (l === 0 && g === 0 && tp === 0) {
                        msg = 'All tuning off — only the side constraint limits changes.';
                        level = 'warn';
                    } else if (l > 1.5 && g === 0 && tp === 0) {
                        msg = 'Anchoring is on but nothing is pushing bonds to move — result will match starting.';
                        level = 'err';
                    } else if (tb === 0 && tp > 0) {
                        msg = 'Trader rebalancing is on but targets match starting — nothing to rebalance.';
                        level = 'warn';
                    } else if (l > 2 * (g + tp + 0.01)) {
                        msg = 'Anchoring is much stronger than reshaping — changes will be small / not move.';
                        level = 'warn';
                    }

                    if (msg) {
                        _warnEl.style.display = 'block';
                        _warnEl.textContent = msg;
                        _warnEl.className = `scfg-warning-msg ${level}`;
                    } else {
                        _warnEl.style.display = 'none';
                        _warnEl.className = 'scfg-warning-msg';
                    }
                };

                // Reset button
                const _DEFAULTS = { lambda_param: 0.25, gamma: 1.0, trader_pull: 1.0, target_blend: 0.50 };
                const _resetBtn = document.getElementById('scfg-reset');
                if (_resetBtn) {
                    _resetBtn.addEventListener('click', () => {
                        for (const [key, val] of Object.entries(_DEFAULTS)) {
                            opts[key] = val;
                            _syncSliderUI(key);
                        }
                        _budget = _DEFAULTS.lambda_param + _DEFAULTS.gamma + _DEFAULTS.trader_pull;
                        _checkWarnings();
                        _writeScfgCache({...opts, linked_sliders: _linked});
                    });
                }

                // Lock-step: λ opposes (γ + tp). Moving λ up pushes γ+tp down, and vice versa.
                // Budget = λ + γ + tp stays constant.
                let _linked = cached.linked_sliders ?? false;
                let _budget = opts.lambda_param + opts.gamma + opts.trader_pull;
                if (_budget < 0.1) _budget = 3.0; // sensible default

                const _syncSliderUI = (key) => {
                    const el = _sliderEls[key];
                    const valEl = el ? document.getElementById(el.id + '-val') : null;
                    if (el) el.value = opts[key];
                    if (valEl) valEl.textContent = opts[key].toFixed(2);
                };

                const _onSliderMove = (changedKey) => {
                    const el = _sliderEls[changedKey];
                    if (!el) return;
                    const newVal = +el.value;
                    opts[changedKey] = newVal;

                    if (_linked && changedKey !== 'target_blend') {
                        const others = ['lambda_param', 'gamma', 'trader_pull'].filter(k => k !== changedKey);
                        const remaining = Math.max(0, _budget - newVal);
                        const oldOtherSum = others.reduce((s, k) => s + opts[k], 0);

                        if (oldOtherSum > 0.001) {
                            // Distribute remaining proportionally
                            for (const k of others) {
                                const share = opts[k] / oldOtherSum;
                                const maxVal = +_sliderEls[k].max;
                                opts[k] = Math.min(maxVal, Math.max(0, +(remaining * share).toFixed(2)));
                            }
                        } else {
                            // Equal split
                            for (const k of others) {
                                const maxVal = +_sliderEls[k].max;
                                opts[k] = Math.min(maxVal, Math.max(0, +(remaining / others.length).toFixed(2)));
                            }
                        }
                        for (const k of others) _syncSliderUI(k);
                    }

                    // Update display for the changed slider
                    const valEl = document.getElementById(el.id + '-val');
                    if (valEl) valEl.textContent = newVal.toFixed(2);

                    _checkWarnings();
                    _writeScfgCache({...opts, linked_sliders: _linked});
                };

                // Bind sliders
                for (const [key, el] of Object.entries(_sliderEls)) {
                    if (!el) continue;
                    el.addEventListener('input', () => _onSliderMove(key));
                    el.addEventListener('change', () => _onSliderMove(key));
                }

                // Link toggle
                if (_linkedEl) {
                    _linkedEl.addEventListener('change', () => {
                        _linked = _linkedEl.checked;
                        // Recalculate budget from current values when linking
                        _budget = opts.lambda_param + opts.gamma + opts.trader_pull;
                        if (_budget < 0.1) _budget = 3.0;
                        _writeScfgCache({...opts, linked_sliders: _linked});
                    });
                }

                // Initial warning check
                _checkWarnings();

                // Numeric inputs for skew caps — fade in/out on value presence
                const fieldPairs = [
                    ['scfg-max-ind-spd', 'max_individual_spread_skew_delta'],
                    ['scfg-max-ind-px',  'max_individual_px_skew_delta'],
                ];
                for (const [elId, optKey] of fieldPairs) {
                    const input = document.getElementById(elId);
                    if (!input) continue;
                    const wrapper = input.closest('.scfg-field');
                    const sync = () => {
                        const raw = input.value.trim();
                        if (raw === '' || isNaN(parseFloat(raw))) {
                            opts[optKey] = null;
                            if (wrapper) wrapper.classList.add('scfg-faded');
                        } else {
                            opts[optKey] = Math.abs(parseFloat(raw));
                            if (wrapper) wrapper.classList.remove('scfg-faded');
                        }
                    };
                    input.addEventListener('input', sync);
                    input.addEventListener('change', sync);
                }

                // Persist all option changes to sessionStorage
                const _persistOnChange = () => _writeScfgCache(opts);


                if (crossingEl) crossingEl.addEventListener('change', _persistOnChange);
                if (matchPivotEl) matchPivotEl.addEventListener('change', _persistOnChange);
                if (isolateEl) isolateEl.addEventListener('change', _persistOnChange);
                for (const [elId] of fieldPairs) {
                    const input = document.getElementById(elId);
                    if (input) {
                        input.addEventListener('input', _persistOnChange);
                        input.addEventListener('change', _persistOnChange);
                    }
                }
            });

            const confirmResult = await modalPromise;

            if (confirmResult === 'proceed') {
                await this._redistributeProceeds(opts);
            } else if (confirmResult === 'config') {
                const mgr = this.context.page._microGridManager;
                if (mgr) {
                   await mgr.openGroup({
                        name: 'redist_params',
                        displayName: 'Solver Config',
                        grids: ['redist_params'],
                    });
                   this.topLock.click();
                }
            }
        })

        this._disposers.push(this.context.page.page$.onValueChanged('proceedsLocked', (cur) => {
            if (cur) {
                this.lockIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 48 48"><defs><mask id="SVGtDIhpJdd"><g fill="none" stroke-linejoin="round" stroke-width="4"><rect width="36" height="22" x="6" y="22" fill="#fff" stroke="#fff" rx="2"/><path stroke="#fff" stroke-linecap="round" d="M14 22v-8c0-5.523 4.477-10 10-10s10 4.477 10 10v8"/><path stroke="#000" stroke-linecap="round" d="M24 30v6"/></g></mask></defs><path fill="currentColor" d="M0 0h48v48H0z" mask="url(#SVGtDIhpJdd)"/></svg>`;
                this.topLock.classList.add('locked')
            } else {
                this.lockIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 48 48"><g fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="4"><rect width="34" height="22" x="7" y="22.048" rx="2"/><path stroke-linecap="round" d="M14 22v-7.995c-.005-5.135 3.923-9.438 9.086-9.954S32.967 6.974 34 12.006M24 30v6"/></g></svg>`;
                this.topLock.classList.remove('locked')
            }
        }));
    }

    _cacheDom() {
        this.filterToggle = document.getElementById('filter-pivot-by-grid');
        this.colorizeToggle = document.getElementById('pivot-colorize-toggle');
        this.bulkSkewMode = document.getElementById('pivot-bulkskew-toggle');
        this.lockTotalsToggle = document.getElementById('pivot-locktotals-toggle');
        this.gridDiv = document.querySelector(this.pivotSelector);
        this.bulkSkewClear = document.getElementById('pivot-bulk-skew-clear');
        this.bulkSkewApply = document.getElementById('pivot-bulk-skew-apply');
        this.venueCopy = document.getElementById('venue-copy-btn');
        this.venueDropdown = document.querySelector('#venue-caret');
        this.venueOptions = this.venueDropdown ? this.venueDropdown.querySelectorAll('.quote-types') : [];
        this.widgetDom = document.getElementById('pivotWidget');
        this.bulkBar = document.querySelector('.pt-left');
        this.controls = document.querySelector('.pivot-controls');
        this.presetButtons = document.getElementById("pivot-quick-presets");
        this.contentRow = document.getElementById('content-row');
        this.refSection = document.querySelector('.upper-ref-market-section');
        this.topBar = document.getElementById('pivot-topbar');
        this.pivotWeight = document.getElementById('pivot-weight-group');
        this.weightNotional = document.getElementById('weightNotional');
        this.weightDv01 = document.getElementById('weightDv01');
        this.weightCount = document.getElementById('weightCount');
        this.topApply = document.getElementById('pivot-top-apply');
        this.topLock = document.getElementById('pivot-top-lock');
        this.lockIcon = document.querySelector('.icon-wrap');
        this.topBtnWrapper = document.getElementById('pivot-top-btn-wrapper')
        this.topDiscard = document.getElementById('pivot-top-discard');

        // Skew mode elements
        this.skewModeGroup = document.getElementById('pivot-skew-mode');
        this.pctMarketWrap = document.getElementById('pivot-pct-market-wrap');
        this.pctMarketSelect = document.getElementById('pivot-pct-market');
    }

    _updatePendingBar(count = 0) {
        const n = Number(count) | 0;
        const has = n > 0;
        if (this.topApply) this.topApply.disabled = !has;
        // if (this.topDiscard) this.topDiscard.disabled = !has;

        if (this.topApply) {
            if (has) {
                this.topApply.textContent = `Apply ${n} Pending Skew${n > 1 ? 's' : ''}`;
                if (this.topDiscard) this.topDiscard.textContent = 'Discard';
            } else {
                this.topApply.textContent = `No Pending Skews`
                if (this.topDiscard) this.topDiscard.textContent = 'Clear';
            }
        }
    }

    currentWidth() {
        return this.ptPivot.element.getBoundingClientRect().width
    }

    maxWidth() {
        return this.contentRow.getBoundingClientRect().width - (this.controls.getBoundingClientRect().width + this.refSection.getBoundingClientRect().width)
    }

    proposeWidth() {
        if (!this.ptPivot) return
        let headerWidth = Array.from(this.ptPivot.element.querySelectorAll('.ag-header-cell')).reduce((acc, dom) => {
            acc += (dom.getBoundingClientRect().width ?? 0)
            return acc;
        }, 0);
        if (!headerWidth) return;
        headerWidth += 30 // sidebar + small buffer
        const maxWidth = this.maxWidth();
        return Math.max(400, Math.min(headerWidth, maxWidth));
    }

    async alignWidthIfNeeded() {
        const c = this.currentWidth()
        const p = this.proposeWidth();
        if (c && p && (p > c)) this.ptPivot.setWidth(p);
    }

    alignWidth() {
        const w = this.proposeWidth();
        if (w != null) this.ptPivot.setWidth(w);
    }

    alignWidthOnTap() {
        const c = this.currentWidth()
        const p = this.proposeWidth();
        const m = this.maxWidth();
        if (Math.abs(c - p) <= 50) {
            this.ptPivot.setWidth(m);
        } else {
            this.ptPivot.setWidth(p);
        }
    }

    _setupResizer() {
        const target = document.querySelector(".pivot-controls");
        const grid = this.ptPivot.element;
        const tbl = this.ptPivot;
        this.interactable = interact(target).resizable({
            edges: {top: false, left: true, bottom: false, right: true},
            listeners: {
                move: function (event) {
                    let base = event.client.x - grid.getBoundingClientRect().left;
                    tbl.setWidth(base)
                }
            }
        });

        // Expand pivot grid to fill available width when the container resizes
        const widget = this;
        let resizeRafId = 0;
        this._resizeObserver = new ResizeObserver(() => {
            if (resizeRafId) return;
            resizeRafId = requestAnimationFrame(() => {
                resizeRafId = 0;
                const maxW = widget.maxWidth();
                if (maxW > 0) tbl.setWidth(maxW);
            });
        });
        if (this.contentRow) this._resizeObserver.observe(this.contentRow);
    }


    /* ------------------------ GroupBy persistence helpers ------------------------ */
    _groupByStorageKey() {
        const k = (this.context?.portfolio_key || this.context?.url_context || 'default').toUpperCase();
        return `pt:pivotGroupBy:${k}`;
    }

    _readGroupByCache() {
        try {
            const raw = localStorage.getItem(this._groupByStorageKey());
            const v = raw ? JSON.parse(raw) : [];
            return Array.isArray(v) ? v : [];
        } catch {
            return [];
        }
    }

    _writeGroupByCache(values) {
        try {
            const dedup = Array.from(new Set((values || []).filter(Boolean)));
            localStorage.setItem(this._groupByStorageKey(), JSON.stringify(dedup));
        } catch {
        }
    }

    _validGroupFields() {
        // Mirrors your existing getAllValidGroupDefs filter.  [oai_citation:0‡Pivot.txt](sediment://file_00000000215061f6be6fe4f880c65325)
        const defs = this.ptPivot.getAllValidGroupDefs().filter(cd => !(cd?.context?.suppressColumnMenu && cd?.context?.suppressColumnMenu.includes('pivotWidget-table')));
        const fields = new Set(defs.map(cd => cd.field));
        return {defs, fields};
    }

    // async _applyGroupByFromCache() {
    //     const pivot = this.ptPivot;    //     if (!pivot?.api) return false;    //    //     const { fields } = this._validGroupFields();    //     const saved = this._readGroupByCache().filter(v => fields.has(v));    //     if (!saved.length) return true; // nothing to do, but grid is ready    //    //     const MAX_GROUPS = 4;    //     const next = saved.slice(0, MAX_GROUPS);    //    //     const curSet = new Set(this.currentGroups || []);    //     const nextSet = new Set(next);    //     const removed = Array.from(curSet).filter(x => !nextSet.has(x));    //     const added = Array.from(nextSet).filter(x => !curSet.has(x));    //    //     if (added.length || removed.length) {    //         await pivot.addRemovePivotGroups({ added, removed });    //         this.currentGroups = next;    //     }    //     if (this.groupBySelect?.addSelections) this.groupBySelect.addSelections(next);    //     return true;    // }
    // _scheduleApplyGroupByFromCache() {    //     // Bounded RAF retry in case API becomes ready a tick later.    //     const MAX_TRIES = 60; // ~1s worst-case    //     const tick = async () => {    //         this._applyTries++;    //         const ok = await this._applyGroupByFromCache();    //         if (ok || this._applyTries >= MAX_TRIES) { this._applyRafId = 0; return; }    //         this._applyRafId = requestAnimationFrame(tick);    //     };    //     this._applyRafId = requestAnimationFrame(tick);    // }
    async _setupReactions() {
        const pivot = this.ptPivot;
        const page$ = this.context.page.page$;
        const widget = this;

        this._disposers.push(this.engine.onColumnEvent('columnResized', () => widget._scheduleAlignSkewButton()));
        const isInitialized = pivot.grid$.pick('pivotInitialized');

        if (isInitialized && pivot.api) {
            widget.api = pivot.api;
            await this.onPresetLoaded()
        } else {
            this._disposers.push(this._em.once('pivot-initialized', async () => {
                widget.api = pivot.api;
                await this.onPresetLoaded()
            }));
        }

        this._disposers.push(this._em.once('pivot-initialized', async () => {
            this.setBulkBars(this.page.page$.get('bulkSkewMode'))
        }));

        this._disposers.push(this._em.on(ArrowAgGridAdapter.COLUMNS_EVENT, () => {
            widget._scheduleAlignSkewButton()
        }));

        // this.context.page.addEventListener(this.filterToggle, 'input', (v) => {
        //     page$.set('linkedPivotFilters', v?.target?.checked ?? false)        // }, {passive: true});
        // Link toggles
        this.context.page.linkStoreToInput(this.filterToggle, 'linkedPivotFilters', page$, {persist:true, storageKey:'linkedPivotFilters'});
        this.context.page.linkStoreToInput(this.lockTotalsToggle, 'lockPivotTotals', page$, {persist: true, storageKey: 'lockPivotTotals'});
        this.context.page.linkStoreToInput(this.colorizeToggle, 'colorizePivot', page$, {persist: true, storageKey: 'colorizePivot'});

        // React to changes
        this._disposers.push(this.context.page.page$.onValueChanged('linkedPivotFilters', (link) => pivot.setRespectSourceFilters(link)));
        this._disposers.push(this.context.page.page$.onValueChanged('lockPivotTotals', (ev) => pivot.setLockGridTotals(ev)));

        // Skew mode: outright vs percent
        this._setupSkewModeControls(pivot);

        // initialize adapter with current persisted values
        pivot.setRespectSourceFilters(!!page$.get('linkedPivotFilters'));
        pivot.setLockGridTotals(!!page$.get('lockPivotTotals'));
        this.context.page.addEventListener(this.bulkSkewClear, 'click', (v) => {
            if (pivot._bucketState.size === 0) return pivot.input_clear_on_each_row();
            pivot.resetAllBuckets();
        });

        this.context.page.addEventListener(this.bulkSkewApply, 'click', async () => {
            if (this._applyInFlight) return;
            this._applyInFlight = true;
            try {
                await pivot.applyAllBuckets();
            } finally {
                this._applyInFlight = false;
            }
        });

        if (this.venueOptions) {
            const pg = this.context.page
            this.venueOptions.forEach(item => {
                this.context.page.addEventListener(item, 'mousedown', async (e) => {
                    const venue = e.currentTarget.getAttribute('data-venue');
                    if (venue) {
                        await widget.writeVenueToClipboard(venue);
                        await pg.send_full_push("Submission", true);
                    } else {
                        widget.context.page.toastManager().error('Submission Copy', `Failed to write to clipboard.`);
                    }
                });
            });
        }

        if (this.topApply) {
            this.context.page.addEventListener(this.topApply, 'click', async () => {
                if (this._applyInFlight) return;
                this._applyInFlight = true;
                try {
                    await pivot.applyAllBuckets();
                    this._updatePendingBar(0);
                } finally {
                    this._applyInFlight = false;
                }
            });
        }
        if (this.topDiscard) {
            this.context.page.addEventListener(this.topDiscard, 'click', () => {
                if (pivot._bucketState.size === 0) {
                    return pivot.input_clear_on_each_row()
                }
                pivot.resetAllBuckets();
                this._updatePendingBar(0);
            });
        }

        this.context.page.addEventListener(this.presetButtons, 'click', async (e) => {
            const nearestBtn = e.target.closest('.pill-button');
            if (nearestBtn) {
                const groupsThen = [...(pivot.pivotConfig.groupBy || [])];
                const group = nearestBtn.getAttribute('data-column');

                if (group === 'CLEAR') {
                    widget.clearActiveQuickPreset();
                    await pivot.updateGroups([], {hard: true});
                } else {
                    if (!e.ctrlKey) {
                        widget.clearActiveQuickPreset();
                        this._current_active_presets.add(nearestBtn);
                        nearestBtn.classList.toggle('active', true);
                        await pivot.updateGroups([group], {hard: true});
                    } else {
                        if (this._current_active_presets.has(nearestBtn)) {
                            this._current_active_presets.delete(nearestBtn);
                            await pivot.removeGroups([group], {hard: true});
                            nearestBtn.classList.toggle('active', false);
                        } else {
                            this._current_active_presets.add(nearestBtn);
                            await pivot.addGroups([group], {hard: true});
                            nearestBtn.classList.toggle('active', true);
                        }
                    }
                }
                const groupsNow = pivot.pivotConfig.groupBy;
                const groupsSet = new Set(groupsNow);
                const removals = groupsThen.filter(x => !groupsSet.has(x))
                pivot.api.setColumnsVisible(removals, false);
                pivot.api.setColumnsVisible(groupsNow, true);
            }
        });

        this.context.page.addEventListener(this.venueCopy, 'click', async () => {
            if (this._applyInFlight) return;
            this._applyInFlight = true;
            try {
                const venue = widget.context.page.portfolioMeta$.get("venue") || 'ib';
                await widget.writeVenueToClipboard(venue);
                await widget.context.page.send_full_push("Submission", true);
            } finally {
                this._applyInFlight = false;
            }
        });

        const applyBtn = this.bulkSkewApply;
        this._disposers.push(this.ptPivot.onComputed(() => {
            applyBtn.classList.toggle('dirtySkews', pivot._bucketState.size > 0);
        }));

        const node_event = 'treeColumnChooser-pivot-NODE_TOGGLE';
        this._disposers.push(this._em.on(node_event, () => {
            widget.clearActiveQuickPreset()
        }))

        this._disposers.push(this.ptPivot.onBucketChange((info) => {
            widget._updatePendingBar(info?.size || 0);
        }));

        this._disposers.push(this.ptPivot.onGroupChange((info) => {
            widget.clearSpan();
            widget._scheduleAlignSkewButton();
        }));
        // initialize banner from current state
        widget._updatePendingBar(this.ptPivot.getPendingBucketCount ? this.ptPivot.getPendingBucketCount() : 0);

    }

    clearSpan() {
        this.ptPivot.resetAllBuckets();
        this._updatePendingBar(0);
    }

    clearActiveQuickPreset() {
        const a = this.presetButtons.querySelectorAll('.active');
        if (a) a.forEach(aa => aa.classList.toggle('active', false));
        this._current_active_presets.clear();
        this.clearSpan();
    }

    setBulkBars(v, realign=true) {
            const pivot = this.ptPivot;
            this.ptPivot.domTimeout = false;
            pivot?.api.setColumnsVisible(['current_skew', 'input_skew', 'proposed_skew'], v);
            pivot.api.setColumnsPinned(['current_skew', 'input_skew', 'proposed_skew'], 'left');
            pivot.api.moveColumns(['current_skew', 'input_skew', 'proposed_skew'], pivot.api.getColumnDefs().length - 3);

            pivot?.api.setColumnsVisible(['isLocked'], v);
            pivot.api.moveColumns(['isLocked'], pivot.api.getColumnDefs().length - 4);
            pivot.api.setColumnsPinned(['isLocked'], 'left');

            requestAnimationFrame(async ()=>{
                if (v && realign) await this.alignWidthIfNeeded();
                requestAnimationFrame(()=> {
                    this.topApply.classList.toggle('disabled', !v);
                    this.topDiscard.classList.toggle('disabled', !v);
                    this.topLock.classList.toggle('disabled', !v);
                    this._scheduleAlignSkewButton();
                    pivot.api.setColumnsPinned(['isLocked'], 'left');
                });
            });
        }

    async onPresetLoaded() {
        const page$ = this.context.page.page$;
        this.context.page.linkStoreToInput(this.bulkSkewMode, 'bulkSkewMode', page$, {persist: true, storageKey: 'bulkSkewMode', cb: this.setBulkBars.bind(this)});
        // this.alignWidth();
    }

    async writeVenueToClipboard(venue) {
        let trueVenue = this.context.page.portfolioMeta$.get("venue") || 'ib';
        if (trueVenue !== 'ib') {
            trueVenue = mapVenueFromRfq(trueVenue)
        }
        const clean_venue = mapVenueFromRfq(venue)
        const cols_needed = {...getVenueCopyColumns(clean_venue, false)};
        if (cols_needed) {
            if (this.context.page.page$.get('activeQuoteType') === 'client') {
                if ('newLevel' in cols_needed) {
                    cols_needed['newLevelDisplay'] = cols_needed.newLevel;
                    delete cols_needed.newLevel
                }
            }
            const fields = Object.keys(cols_needed);
            fields.push('isReal', 'assignedTrader');
            if (!fields.includes('grossSize')) {
                fields.push('grossSize')
            }

            const data = this.adapter.engine.getColumns(fields)
            const headers = {};

            Object.entries(cols_needed).forEach(([k, v]) => {
                if (!Array.isArray(v)) {
                    headers[k] = v;
                } else {
                    headers[k] = v[0];
                }
            });
            const reformat = data.filter(row => {
                return !(
                    (row.isReal == null) ||
                    (row.isReal === 0) ||
                    (row.grossSize == null) ||
                    (row.grossSize === 0) ||
                    (row.assignedTrader === "REMOVED")
                )
            }).map(row => {
                const clean = {};
                Object.entries(row).forEach(([k, v]) => {
                    if (!(k in cols_needed)) return;
                    if (!Array.isArray(cols_needed[k])) {
                        clean[k] = v;
                    } else {
                        clean[k] = cols_needed[k][1](v) || '';
                    }
                })
                return clean
            });
            const removedCount = data.length - reformat.length;
            await writeObjectToClipboard(reformat, {headerOverride: headers, addCommaToNumerics: clean_venue === 'ib'});

            const mismatchVenue = (clean_venue !== trueVenue) && (clean_venue !== "ib")
            let msg = `Copied data for: ${clean_venue.toUpperCase()}`;
            if (mismatchVenue) {
                msg = msg + `<br>Venue Expected: ${trueVenue.toUpperCase()}`
            }
            if (removedCount > 0) {
                msg = msg + `<br>**REMOVED ${removedCount} bonds**`
            }
            if (!mismatchVenue) {
                this.context.page.toastManager().success('Submission Copy', msg)
            } else {
                this.context.page.toastManager().warning('Submission Copy - Mismatched Venue', msg)
            }
        } else {
            this.context.page.toastManager().error('Submission Copy', `Failed to write to clipboard.`)
        }
    }

    _setupSkewModeControls(pivot) {
        if (!this.skewModeGroup) return;
        const widget = this;

        // Populate percent market select with available markets
        this._populatePctMarketOptions();

        // Radio change: outright vs percent
        this.context.page.addEventListener(this.skewModeGroup, 'change', (e) => {
            const mode = e.target.value;
            pivot.setSkewMode(mode);
            // Clear pending skews when switching mode to avoid confusion
            pivot.resetAllBuckets();
            widget._updatePendingBar(0);
        });

        // Market select for percent width source
        if (this.pctMarketSelect) {
            this.context.page.addEventListener(this.pctMarketSelect, 'change', (e) => {
                pivot.setSkewPctMarket(e.target.value || null);
            });
        }
    }

    _populatePctMarketOptions() {
        if (!this.pctMarketSelect) return;
        const page = this.context.page;
        const marketMap = page.marketDataMap;
        if (!marketMap || !marketMap.size) return;
        // Keep the default "(Ref Mkt)" option, then add available markets
        const frag = document.createDocumentFragment();
        const def = document.createElement('option');
        def.value = '';
        def.textContent = '(Ref Mkt)';
        frag.appendChild(def);
        for (const [key, meta] of marketMap.entries()) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = meta?.abbr || meta?.label || key;
            frag.appendChild(opt);
        }
        this.pctMarketSelect.innerHTML = '';
        this.pctMarketSelect.appendChild(frag);
    }

    _setupHotkeys() {
        const page$ = this.context.page.page$;
        const widget = this;
        this.context.page.addEventListener(document, 'keydown', async (event) => {

            // Ctrl+Shift+f to toggle grid link
            if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'f') {
                event.preventDefault();
                if (this.filterToggle) {
                    const newValue = !this.filterToggle.checked;
                    this.filterToggle.checked = newValue;
                    page$.set('linkedPivotFilters', newValue)
                }
            } else if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'b') {
                event.preventDefault();
                if (this.bulkSkewMode) {
                    const newValue = !this.bulkSkewMode.checked;
                    this.bulkSkewMode.checked = newValue;
                    page$.set('bulkSkewMode', newValue)
                }
            }

            // Ctrl+shift+c to copy
            else if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'c') {
                event.preventDefault();
                event.stopPropagation();
                if (widget._applyInFlight) return;
                widget._applyInFlight = true;
                try {
                    const venue = widget.context.page.portfolioMeta$.get("venue") || 'ib';
                    await widget.writeVenueToClipboard(venue);
                    await widget.context.page.send_full_push("Submission", true);
                } finally {
                    widget._applyInFlight = false;
                }
            }

        });
    }

    async onActivate() {
        this.ptPivot._locked = false;
        this.ptPivot.isActive = true;
        await this._scheduleAlignSkewButton();
    }

    _scheduleAlignSkewButton() {
        if (this._alignRafId) return;
        this._alignRafId = requestAnimationFrame(async () => {
            this._alignRafId = 0;
            await this.alignSkewButton();
        });
    }

    async alignSkewButton() {

        if (!this.ptPivot?.api) return;
        const current_skew = this.ptPivot.api.getColumn('current_skew');
        const input_skew = this.ptPivot.api.getColumn('input_skew');
        const prop_skew = this.ptPivot.api.getColumn('proposed_skew');
        if (!current_skew || !input_skew || !prop_skew) return;

        const barrier = this.bulkBar.getBoundingClientRect().width || 0
        const cost = this.topBtnWrapper.getBoundingClientRect().width || 0;

        const max_skew = Math.max(0, (barrier || 0) - (cost || 0));

        const minLeft = Math.max(0, Math.min(current_skew.left, input_skew.left, prop_skew.left));
        this.topBtnWrapper.style.transform = `translateX(${Math.min(max_skew, minLeft - 10)}px)`;

        const w = current_skew.getActualWidth() + input_skew.getActualWidth() + prop_skew.getActualWidth();
        this.topBtnWrapper.style.width = `${w - 65}px`;

    }

    onResumeSubscriptions() {
        this.ptPivot.hardRefresh({force: true});
        if (!this._first_activation) {
            this._first_activation = true;
            this.setBulkBars(this.page.page$.get("bulkSkewMode"));
        }
    }

    onDeactivate() {
        // console.log('[PivotWidget] deactivated');
        // this.ptPivot._locked = true;        // this.ptPivot.isActive = false;
    }

    async onCleanup() {
        // Drain subscription disposers
        for (const dispose of this._disposers) {
            try {
                if (typeof dispose === 'function') dispose();
            } catch (_) {
            }
        }
        this._disposers.length = 0;

        if (this._alignRafId) {
            cancelAnimationFrame(this._alignRafId);
            this._alignRafId = 0;
        }
        if (this._presetTimeoutId) {
            clearTimeout(this._presetTimeoutId);
            this._presetTimeoutId = 0;
        }
        if (this._resizeObserver) {
            try { this._resizeObserver.disconnect(); } catch(_) {}
            this._resizeObserver = null;
        }

        try {
            this.api?.destroy();
        } catch (_) {
        }
        try {
            this.ptPivot.dispose();
        } catch (_) {
        }


        if (this.interactable) {
            try {
                this.interactable.unset();
            } catch (_) {
            }
            this.interactable = null;
        }
        if (this.tt) {
            try {
                this.context.page.tooltipManager()?.remove?.('proceed-lock-tooltip');
            } catch (_) {
            }
            this.tt = null;
        }
        try {
            if (this._applyRafId) cancelAnimationFrame(this._applyRafId);
            this._applyRafId = 0;
            this._applyTries = 0;
        } finally {
            if (this.dynamicController) this.dynamicController.abort();
        }
        if (this.widgetDom) {
            this.widgetDom.innerHTML = '';
        }
    }

    // --- UI & Controls ---

    onRender() {
        this.widgetDiv.innerHTML = `
        <div class="pivot-controls">
            <div class="pivot-controls-top">
                <div class="pivot-control-group">
                    <label>
                        <div class="piv-left">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><path fill="currentColor" d="M10 8V3h9q.825 0 1.413.588T21 5v3zM5 21q-.825 0-1.412-.587T3 19v-9h5v11zM3 8V5q0-.825.588-1.412T5 3h3v5zm9.85 11l.875.9q.275.275.275.688t-.3.712q-.275.275-.7.275t-.7-.275l-2.6-2.6q-.3-.3-.3-.7t.3-.7l2.6-2.6q.275-.275.688-.287t.712.287q.275.275.288.688t-.263.712l-.875.9H15q.825 0 1.413-.587T17 15v-2.2l-.9.9q-.275.275-.7.275t-.7-.275t-.275-.7t.275-.7l2.6-2.6q.3-.3.7-.3t.7.3l2.6 2.6q.275.275.275.7t-.275.7t-.7.275t-.7-.275l-.9-.9V15q0 1.65-1.175 2.825T15 19z"/></svg>
                            <p>Pivot Portfolio</p>
                        </div>
                        <div class="piv-right">
                            <button class="tooltip-left" id="pivotSettings" data-tooltip="Settings">
                                <lord-icon src="/assets/lottie/cog-hover-2.json" trigger="hover" class="current-color" stroke="currentColor" style="width: 16px; height: 16px;"></lord-icon>
                            </button>
                        </div>
                    </label>
                </div>
                <div id="pivot-quick-presets">
                    <div id="pivot-quick-main">
                        <div class="colored-pill indigo-pill pill-button btn" data-column="assignedTrader">Assigned</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="desigName">Desig</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="lastEditUser">LastEdit</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="isDnt">DNT</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="bvalSubAssetClass">Asset</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="hasPriced">Priced</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="liqScoreCombinedGroup">Liq</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="firmPositionBucket">Firm</div>
                        <div class="colored-pill indigo-pill pill-button btn" data-column="algoPositionBucket">Algo</div>
                    </div>
                    <div id="pivot-quick-clear">
                        <div class="colored-pill gold-pill pill-button btn" data-column="CLEAR">Clear Groups</div>
                        <div class="pivot-quick-groups-title">Quick Groups</div>
                    </div>
                </div>
            </div>
            <div class="pivot-controls-middle">
                <div class="pivot-control-group pivot-skew-mode-group">
                    <span class="label-text skew-mode-label">Skew Mode</span>
                    <div id="pivot-skew-mode" class="pivot-skew-mode-radios">
                        <label class="skew-mode-option">
                            <input type="radio" name="pivotSkewMode" value="outright" checked />
                            <span>Outright</span>
                        </label>
                        <label class="skew-mode-option">
                            <input type="radio" name="pivotSkewMode" value="percent" />
                            <span>%&nbsp;Width</span>
                            <div id="pivot-pct-market-wrap" class="pivot-pct-market-wrap">
                                <select id="pivot-pct-market" class="pivot-pct-market-select">
                                    <option value="">(Ref Mkt)</option>
                                </select>
                            </div>
                        </label>
                    </div>
                </div>
            </div>
            <div class="pivot-controls-bottom">
                <div class="pivot-control-group middle-wrapper">
                                        
                    <div class="pivot-control-group pivot-toggle">
                        <label class="label">
                            <span class="label-text">Use grid filters?</span>
                            <input id="filter-pivot-by-grid" type="checkbox" class="checkbox"/>
                        </label>
                        <label class="label">
                            <span class="label-text">Colorize</span>
                            <input id="pivot-colorize-toggle" type="checkbox" class="checkbox"/>
                        </label>
                        <label class="label">
                            <span class="label-text tooltip tooltip-top" data-tooltip="Grand totals ignore filters; top rows still follow them.">Guard grid totals?</span>
                            <input id="pivot-locktotals-toggle" type="checkbox" class="checkbox"/>
                        </label>
                    </div>
                </div>
                <div class="pivot-control-group bottom-wrapper">
                    <div class="pivot-control-group bulkskew-wrapper">
                        <div class="pivot-control-group pivot-toggle pivot-switch" id="pivot-bulkskew-toggle-outer">
                            <label class="label">
                                <input id="pivot-bulkskew-toggle" type="checkbox" class="toggle"/>
                                <span class="label-text">Bulk Skew Mode</span>
                            </label>
                        </div>
                        <div id="bulk-skew-apply-btn" style="display:none;">
                            <button class="btn btn-primary btn-sm" id="pivot-bulk-skew-apply">APPLY</button>
                            <button class="btn btn-primary btn-sm" id="pivot-bulk-skew-clear">CLEAR</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div id="pivot-grid-container" class="ag-theme-alpine fill-parent">
            <div id="pivot-topbar" class="pivot-topbar">
                <div class="pt-left">
                    <div id="pivot-top-btn-wrapper">
                        <button id="pivot-top-discard" class="btn btn-xs disabled">Discard</button>
                        <button id="pivot-top-apply" class="btn btn-xs disabled" disabled>Apply Pending Skews</button>
                        <button id="pivot-top-lock" class="btn btn-xs">
                             <span class="icon-wrap">
                                 <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 20 20"><path fill="currentColor" d="M8.693 2.058c.404-.058.77-.065 1.13.006c.28.057.529.157.765.282a1.42 1.42 0 0 0-.566 1.401l-.008-.004a1.5 1.5 0 0 0-.487-.207c-.14-.028-.322-.036-.612.005a1.25 1.25 0 0 0-.861.537c-.21.301-.359.75-.405 1.357c-.003.16-.028.647-.065 1.315H9.3a.75.75 0 0 1 0 1.5H7.498a506 506 0 0 1-.35 5.607v.005l-.01.12c-.091 1.115-.226 2.772-1.76 3.666l-.006.003c-.997.57-2.18.384-3.107-.08a.751.751 0 0 1 .67-1.341c.671.335 1.287.349 1.69.12c.807-.472.92-1.326 1.027-2.61c.05-.645.174-2.635.286-4.522l.058-.968H4.801a.75.75 0 0 1 0-1.5h1.28l.006-.093a169 169 0 0 0 .062-1.224v-.025L6.15 5.4l.001-.053c.056-.785.257-1.529.67-2.124A2.75 2.75 0 0 1 8.677 2.06zm1.264 7.101c.31-.132.635-.173.957-.1c.312.07.555.235.736.402c.277.256.505.622.657.867l.074.116a36 36 0 0 0 .76 1.354l1.985-1.985l.53.174a.6.6 0 0 1 .219.14c.07.06.11.14.14.22l.17.527l-2.147 2.15a46 46 0 0 1 1.02 1.798l.061.103c.053.09.102.171.158.258q.116.176.202.257q.043.04.064.05q.019.011.02.01h.006l.033-.006c.018-.01.026-.019.033-.025a1 1 0 0 0 .127-.176a.75.75 0 0 1 1.279.783c-.227.37-.448.601-.804.78a1.6 1.6 0 0 1-1.056.096a1.7 1.7 0 0 1-.733-.423c-.29-.275-.51-.65-.64-.872l-.05-.087l-.028-.05a38 38 0 0 0-.788-1.4l-2.662 2.66a.75.75 0 0 1-1.06-1.06l2.825-2.825a46 46 0 0 1-.947-1.671l-.087-.14c-.058-.091-.108-.172-.167-.259a1.6 1.6 0 0 0-.212-.262l-.042-.035l-.012.01l-.066.076l-.022.025l-.174.202a.75.75 0 1 1-1.133-.983l.164-.19l.022-.026a1.8 1.8 0 0 1 .415-.38l.032-.024a1 1 0 0 1 .14-.079M17.484 6a.3.3 0 0 1 .285.201l.25.766a1.58 1.58 0 0 0 .999.998l.765.248l.016.004a.302.302 0 0 1 0 .57l-.766.248a1.58 1.58 0 0 0-.999.998l-.249.766a.3.3 0 0 1-.46.145a.3.3 0 0 1-.11-.145l-.25-.766a1.58 1.58 0 0 0-.997-1.002l-.766-.248A.3.3 0 0 1 15 8.498a.3.3 0 0 1 .202-.285l.766-.248a1.58 1.58 0 0 0 .983-.998l.248-.766A.3.3 0 0 1 17.484 6m-3.006-6a.43.43 0 0 1 .4.282l.348 1.072a2.2 2.2 0 0 0 1.399 1.396l1.071.349l.022.005a.426.426 0 0 1 .205.643a.42.42 0 0 1-.205.154l-1.072.349a2.21 2.21 0 0 0-1.398 1.396l-.349 1.072a.424.424 0 0 1-.643.204l-.02-.015a.43.43 0 0 1-.136-.19l-.347-1.07a2.2 2.2 0 0 0-1.398-1.402l-1.073-.349a.424.424 0 0 1 0-.797l1.072-.349a2.21 2.21 0 0 0 1.377-1.396L14.08.282a.42.42 0 0 1 .4-.282"/></svg>
                            </span>
                           
                        </button>
                    </div>
                    <span id="pivot-pending-count" class="pending-indicator"></span>
                </div>
            </div>
        </div>`;
    }


    _initControls() {
        const {defs} = this._validGroupFields();
        const groupByOptions = defs.map(cd => ({
            label: cd?.context?.aggregationName ?? cd.headerName ?? cd.field,
            value: cd.field,
            group: cd.context?.customColumnGroup || 'General'
        }));

        const t2 = document.getElementById('portfolio-top-two-rows'); // z-index shim if present
        const pivot = this.ptPivot;
        const widget = this;

        const vc = document.getElementById('venue-caret');
        const gg = document.getElementById('portfolio-pricing-grid');
        if (vc && this.context?.page?.addEventListener) {
            this.context.page.addEventListener(vc, 'mousedown', () => {
                if (t2) t2.style.zIndex = "2";
                if (gg) gg.style.zIndex = "0";
            });
            const vcb = vc.querySelector('*');
            if (vcb) {
                this.context.page.addEventListener(vcb, 'blur', () => {
                    if (t2) t2.style.zIndex = "0";
                    if (gg) gg.style.zIndex = "auto";
                });
            }
        }
    }
}
