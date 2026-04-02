

import {writeObjectToClipboard} from "@/utils/clipboardHelpers.js";

export const TYPE_TO_COLOR = Object.freeze({
    client: 'blue',
    timing: 'orange',
    market: 'purple',
    trader: 'blue',
    status: 'green',
    warning: 'orange',
    error: 'red',
    info: 'gray',
    portfolio: 'teal'
});

const DEFAULT_NULL_POLICY = 'hide';
const ANIM_MS = 160;
const DISTINCT_DEBOUNCE_MS = 120;
const FRAME_BUDGET_MS = 5.0;

const CLASSNAMES = Object.freeze({
    pill: 'pt-pill',
    colored: 'colored-pill',
    tooltip: 'tooltip-top',
    dismissBtn: 'pill-dismiss',
    hidden: 'pill-hidden',
    fading: 'pill-fading',
    overflow: 'pill-overflow',
    bounding: 'pill-box',
    clickable: 'pill-clickable'
});

const _COLOR_KEEP = new Set([CLASSNAMES.colored, CLASSNAMES.pill]);
let _uid = 0;
const _now = () => performance.now();

/* ---------- tiny utils ---------- */
function _toPx(v) { return typeof v === 'number' ? `${v}px` : (v || ''); }
function _qs(x) { return typeof x === 'string' ? document.querySelector(x) : x || null; }
function _trimStr(x) { return x == null ? '' : (x + '').trim(); }
function _isEmpty(v) { return v == null || v === '' || (Array.isArray(v) && v.length === 0); }
function _isMissingValue(v) { return v == null || (typeof v === 'string' && v.trim() === ''); }
function _toFiniteNumber(v) {
    if (v == null) return null;
    const n = typeof v === 'number' ? v : +v;
    return Number.isFinite(n) ? n : null;
}
function _timeUnitToMs(unit) {
    const u = String(unit || 'days').toLowerCase();
    if (u === 'ms' || u === 'millisecond' || u === 'milliseconds') return 1;
    if (u === 's' || u === 'sec' || u === 'second' || u === 'seconds') return 1000;
    if (u === 'm' || u === 'min' || u === 'minute' || u === 'minutes') return 60 * 1000;
    if (u === 'h' || u === 'hr' || u === 'hour' || u === 'hours') return 60 * 60 * 1000;
    return 24 * 60 * 60 * 1000;
}
function _looksLikeTimeOnly(str) { return /^\d{1,2}:\d{2}(:\d{2}(\.\d{1,3})?)?$/.test(str); }
function _toTimestamp(v, now = Date.now()) {
    if (v == null) return null;
    if (v instanceof Date) {
        const t = v.getTime();
        return Number.isFinite(t) ? t : null;
    }
    if (typeof v === 'number') {
        if (!Number.isFinite(v)) return null;
        const abs = Math.abs(v);
        if (abs < 1e11) return Math.trunc(v * 1000);
        return Math.trunc(v);
    }
    const raw = String(v).trim();
    if (!raw) return null;
    if (/^-?\d+(\.\d+)?$/.test(raw)) {
        const n = +raw;
        if (!Number.isFinite(n)) return null;
        const abs = Math.abs(n);
        if (abs < 1e11) return Math.trunc(n * 1000);
        return Math.trunc(n);
    }
    if (_looksLikeTimeOnly(raw)) {
        const d = new Date(now);
        const p = raw.split(':');
        const h = +p[0], m = +p[1];
        let sec = 0, ms = 0;
        if (p[2]) {
            const sPart = p[2];
            if (sPart.includes('.')) {
                const ps = sPart.split('.');
                sec = +ps[0];
                ms = +(ps[1] + '00').slice(0, 3);
            } else sec = +sPart;
        }
        d.setHours(h, m, sec, ms);
        const t = d.getTime();
        return Number.isFinite(t) ? t : null;
    }
    const t = Date.parse(raw);
    return Number.isFinite(t) ? t : null;
}
function _raf(fn) { requestAnimationFrame(fn); }
function _debounce(fn, wait) {
    let timerId = null, lastArgs = null;
    function debounced(...args) {
        lastArgs = args;
        if (timerId !== null) return;
        timerId = setTimeout(() => {
            timerId = null;
            const a = lastArgs;
            lastArgs = null;
            fn.apply(this, a);
        }, wait);
    }
    debounced.cancel = () => {
        if (timerId !== null) { clearTimeout(timerId); timerId = null; }
        lastArgs = null;
    };
    return debounced;
}
function _applyColorClass(el, color) {
    const list = el.classList;
    for (let i = list.length - 1; i >= 0; i--) {
        const c = list.item(i);
        if (!_COLOR_KEEP.has(c) && c.endsWith('-pill')) {
            list.remove(c);
        }
    }
    if (color) {
        list.add(`${color}-pill`);
        list.add(CLASSNAMES.colored);
    }
}
function isFunc(x) { return typeof x === 'function'; }
function ensure_list(x) { return Array.isArray(x) ? x : [x]; }

/* ---------- Pill Recipes ---------- */
export const PILL_RECIPES = Object.freeze({
    distinct: {
        type: 'portfolio',
        valueGetter: async (data) => {
            const seen = new Set(data?.filter(v => v != null) || []);
            return { distinct: seen.size, seenValues: Array.from(seen) };
        },
        condition: (data, pill) => {
            return data.distinct === 1
        }
    },
    notDistinct: {
        type: 'portfolio',
        valueGetter: async (data) => {
            const seen = new Set(data?.filter(v => v != null) || []);
            return { distinct: seen.size, seenValues: Array.from(seen) };
        },
        condition: (data, pill) => {
            return data.distinct > 1
        }
    },
    count: {
        type: 'portfolio',
        valueGetter: async (data, pill) => {
            const test = pill.equals || ((v) => v != null);
            return data?.filter(test).length || 0;
        },
        valueFormatter: (count, pill) => {
            if (count < (pill.min || 1)) return null;
            return typeof pill.labelWhenMet === 'function' ? pill.labelWhenMet(count) : `${count} items`;
        }
    },
    mismatch: {
        type: 'error',
        valueGetter: async (data, pill) => {
            const [left, right] = data || [[], []];
            let count = 0;
            for (let i = 0; i < Math.max(left.length, right.length); i++) {
                if (pill.ignoreWhenBothMissing && _isMissingValue(left[i]) && _isMissingValue(right[i])) continue;
                if (left[i] !== right[i]) count++;
            }
            return count >= (pill.min || 1) ? count : null;
        },
        valueFormatter: (count, pill) => count ? `${pill.columns[0]} ≠ ${pill.columns[1]}: ${count}` : null
    },
    staleTime: {
        type: 'warning',
        valueGetter: async (data, pill) => {
            const now = Date.now();
            const unitMs = _timeUnitToMs(pill.unit || 'days');
            let count = 0;
            for (const val of (data || [])) {
                const ts = _toTimestamp(val, now);
                if (ts == null) continue;
                const age = (now - ts) / unitMs;
                if (age > (pill.threshold || 7)) count++;
            }
            return count >= (pill.min || 1) ? { count } : null;
        },
        valueFormatter: (raw, pill) => raw ? `Stale ${pill.label || 'items'}: ${raw.count}` : null
    },
    duplicates: {
        type: 'warning',
        valueGetter: async (data, pill) => {
            const counts = new Map();
            let duplicateRows = 0;
            for (const v of (data || [])) {
                if (pill.ignoreMissing && _isMissingValue(v)) continue;
                const key = typeof v === 'string' ? v.trim() : v;
                counts.set(key, (counts.get(key) || 0) + 1);
            }
            for (const n of counts.values()) if (n > 1) duplicateRows += n - 1;
            return duplicateRows >= (pill.min || 1) ? { duplicateRows } : null;
        },
        valueFormatter: (raw) => raw ? `Duplicates: ${raw.duplicateRows}` : null
    },
    meta: {
        valueGetter: async (data, pill) => {
            const key = Array.isArray(pill.columns) ? pill.columns[0] : pill.columns;
            if (key == null) return;
            return pill.mgr.context.page._metaStore.get(key)
        },
        source: 'store'
    },
    custom: {
        // Acts as a passthrough for pills that provide their own inline valueGetter
        valueGetter: async (data, pill) => data
    },
});

/* ---------- Pill ---------- */
export class Pill {
    constructor(manager, opts) {
        this.mgr = manager;
        this.id = opts.id || `pill_${++_uid}`;
        this.type = opts.type || null;
        this.color = opts.color || (this.type ? manager.typeToColor[this.type] : null);
        this.columns = opts.columns || null;
        this.classes = Array.isArray(opts.classes) ? opts.classes.slice(0, 12) : [];
        this.minWidth = opts.minWidth || null;
        this.maxWidth = opts.maxWidth || null;
        this.nullPolicy = opts.nullPolicy || DEFAULT_NULL_POLICY;
        this.source = opts.source || null;

        this.tooltip = opts.tooltip || null;
        this.tooltipConfig = opts.tooltipConfig || null;
        this.modal = opts.modal || null;

        this.valueGetter = opts.valueGetter || (async (data) => data);
        this.valueFormatter = opts.valueFormatter || (async (val) => val);
        this.condition = opts.condition || (async (val) => val != null);

        this.label = opts.label || null;
        this.onClick = isFunc(opts.onClick) ? opts.onClick : null;

        this.hiddenByForce = false;
        this.isDismissing = false;
        this.dismissed = false;
        this.container = null;
        this.root = null;
        this.textNode = null;
        this.labelRoot = null;

        this._mounted = false;
        this._lastDisplay = '';
        this._lastVisible = false;
        this._controller = new AbortController();
        this.signal = this._controller.signal;

        this.update = _debounce(async () => await this._updateFromSource(), DISTINCT_DEBOUNCE_MS);
        this._createDom();
        this._createTooltip();
    }

    _createTooltip(data) {
        if (this.tooltip == null) return
        let txt;
        if (typeof this.tooltip === 'function') {
            txt = this.tooltip(data)
        } else {
            txt = this.tooltip;
        }
        this.tooltipHandle = this.mgr.tooltipManager.add({
            id: `${this.id}-tooltip`,
            targets: this.root,
            content: txt,
            fitContent: true,
            contextMenu: { enabled: false },
            ...this.tooltipConfig
        });
    }

    _createDom() {
        const el = document.createElement('div');
        el.className = `${CLASSNAMES.pill} ${CLASSNAMES.colored} ${CLASSNAMES.bounding}`;
        if (this.classes && this.classes.length) {
            el.classList.add(...ensure_list(this.classes));
        }

        el.classList.add(CLASSNAMES.hidden);

        _applyColorClass(el, this.color);
        if (this.minWidth) el.style.minWidth = _toPx(this.minWidth);
        if (this.maxWidth) el.style.maxWidth = _toPx(this.maxWidth);

        const labelNode = document.createElement('div');
        labelNode.className = 'pill-label';
        this.labelRoot = labelNode;
        el.append(labelNode);

        const tn = document.createTextNode('');
        el.appendChild(tn);
        this.textNode = tn;

        if (this.modal || this.onClick) {
            el.style.cursor = 'pointer';
            el.classList.add(CLASSNAMES.clickable)
            el.addEventListener('click', (e) => {
                if (this.onClick) {
                    this.onClick(e, this);
                }
                if (this.modal) {
                    if (typeof this.modal == 'function') {
                        this.modal(e, this);
                    } else if (typeof this.modal == 'object') {
                        const d = this.modal;
                        this.mgr.createInfoModal(d.title, d.info, d.type || 'info', d?.subtitle);
                    } else {
                        this.mgr.createInfoModal(this.id, this.modal, 'info');
                    }
                }
            }, {signal: this.signal});
        }

        this.root = el;
    }

    async _updateFromSource() {
        if (!this._mounted || !this.root || !this.columns) return;

        let data = null;
        let columns = null;

        if (typeof this.columns === 'function') {
            columns = this.columns(this)
        } else {
            columns = this.columns;
        }

        columns = ensure_list(columns);
        if (columns && (this.source !== 'store')) {
            data = await this.mgr.engine.getColumnValues(columns);
        }

        const rawValues = await this.valueGetter(data, this);

        if (await this.condition(rawValues, this)) {
            const displayStr = await this.valueFormatter(rawValues, this);

            let currentLabel = this.label;
            if (isFunc(currentLabel)) {
                currentLabel = currentLabel(displayStr, rawValues);
            }
            if (currentLabel != null) {
                this.labelRoot.textContent = currentLabel.toString();
            }

            this.mgr._enqueueDomWrite(this, { display: displayStr, raw: rawValues });
        } else {
            this._applyVisibility(false);
        }
    }

    _applyVisibility(visible) {
        if (!this.root || this.isDismissing) return;
        const v = visible && !this.dismissed && !this.hiddenByForce;

        this._lastVisible = v;

        if (v) {
            this.root.classList.remove(CLASSNAMES.hidden);
            this.root.style.display = '';
        } else {
            this.root.classList.add(CLASSNAMES.hidden);
            this.root.style.display = 'none';
        }
    }

    async mount(container, { prepend = false } = {}) {
        try {
            const parent = _qs(container) || this.mgr.defaultContainer;
            if (!parent) {
                throw new Error(`Pill#${this.id}: container not found.`);
            }
            if (this._mounted && this.container === parent) {
                return this;
            }

            if (this._mounted && this.root && this.root.parentNode) {
                this.root.parentNode.removeChild(this.root);
            }

            const frag = document.createDocumentFragment();
            frag.appendChild(this.root);

            if (prepend) {
                parent.prepend(frag);
            } else {
                parent.appendChild(frag);
            }

            this.container = parent;
            this._mounted = true;
            await this._updateFromSource();
            return this;
        } catch (err) {
            console.error(err);
            return this;
        }
    }

    unmount() {
        if (!this._mounted) {
            return this;
        }
        if (this.root && this.root.parentNode) {
            this.root.parentNode.removeChild(this.root);
        }
        this._mounted = false;
        this.container = null;
        return this;
    }

    destroy() {
        this.unmount();
        if (this.update && this.update.cancel) {
            this.update.cancel();
        }
        this._controller.abort();
        this.root = null;
        this.textNode = null;
        this.mgr.pills.delete(this.id);
    }
}

/* ---------- PillManager ---------- */
export default class PillManager {
    constructor(context, opts = {}) {
        this.context = context;
        this.defaultContainer = _qs(opts.defaultContainer);
        this.engine = opts.engine;
        this.typeToColor = opts.typeToColor || TYPE_TO_COLOR;
        this.frameBudgetMs = opts.frameBudgetMs || FRAME_BUDGET_MS;

        this.pills = new Map();
        this._domQueue = new Map();
        this._domScheduled = false;
        this._canAnimateFlag = true;

        this.tooltipManager = this.context.page.tooltipManager()

        if (this.engine && (typeof this.engine.onEpochChange === 'function')) {
            this.engine.onEpochChange((colsChanged) => this._fanoutEngineDirty(colsChanged));
        }
    }

    async createPill(recipeName, opts = {}) {
        try {
            const recipe = PILL_RECIPES[recipeName];
            if (!recipe) throw new Error(`Unknown pill recipe: ${recipeName}`);

            const mergedOpts = {
                ...recipe,
                ...opts,
                columns: isFunc(recipe.watchColumns) ? recipe.watchColumns(opts) : (opts.columns || recipe.columns)
            };

            const p = new Pill(this, mergedOpts);
            this.pills.set(p.id, p);

            const d = opts?.dom ?? this.defaultContainer;
            if (d != null) {
                await p.mount(d);
            }
            return p;
        } catch (e) {
            console.error(e);
            return null;
        }
    }

    _fanoutEngineDirty(changes) {
        const changed = changes?.colsChanged;
        if (!changed) return;
        const isGlobal = changed === true;
        const changedSet = isGlobal ? null : (changed instanceof Set ? changed : new Set(Array.isArray(changed) ? changed : [changed]));
        for (const pill of this.pills.values()) {
            if (!pill.columns || pill.source === 'store') continue;
            if (isGlobal) { pill.update(); continue; }
            const cols = ensure_list(pill.columns);
            if (cols.some(c => changedSet.has(c))) {
                pill.update();
            }
        }
    }

    _enqueueDomWrite(pill, state) {
        this._domQueue.set(pill.id, { pill, state });
        if (!this._domScheduled) {
            this._domScheduled = true;
            _raf(() => this._flushDomQueue());
        }
    }

    _flushDomQueue() {
        const start = performance.now();
        this._domScheduled = false;
        this._canAnimateFlag = (this._domQueue.size <= 12);

        for (const [id, { pill, state }] of this._domQueue) {
            this._domQueue.delete(id);
            if (!pill.root || pill.isDismissing) continue;

            const displayStr = state.display;
            const shouldShow = !_isEmpty(displayStr) && !pill.dismissed && !pill.hiddenByForce;

            if (displayStr !== pill._lastDisplay) {
                pill.textNode.nodeValue = displayStr;
                pill._lastDisplay = displayStr;
            }

            if (shouldShow !== pill._lastVisible) {
                pill._applyVisibility(shouldShow);
            }

            if ((performance.now() - start) >= this.frameBudgetMs) {
                this._domScheduled = true;
                _raf(() => this._flushDomQueue());
                break;
            }
        }

        if (this._domQueue.size === 0) {
            this._canAnimateFlag = true;
        }
    }

    canAnimate() { return this._canAnimateFlag; }

    getModalIcon(type) {
        if (type === 'info') {
            return `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><path fill="currentColor" d="M12 17q.425 0 .713-.288T13 16v-4q0-.425-.288-.712T12 11t-.712.288T11 12v4q0 .425.288.713T12 17m0-8q.425 0 .713-.288T13 8t-.288-.712T12 7t-.712.288T11 8t.288.713T12 9m0 13q-2.075 0-3.9-.788t-3.175-2.137T2.788 15.9T2 12t.788-3.9t2.137-3.175T8.1 2.788T12 2t3.9.788t3.175 2.137T21.213 8.1T22 12t-.788 3.9t-2.137 3.175t-3.175 2.138T12 22m0-2q3.35 0 5.675-2.325T20 12t-2.325-5.675T12 4T6.325 6.325T4 12t2.325 5.675T12 20m0-8"/></svg>`;
        }
        return '';
    }

    createInfoModal(title, content, type = "info", subtitle=null, classList=null, options={}) {
        const enableCopy = options.copyToClipboard !== false;

        const modal = document.createElement('dialog');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-box pt-info-box">
                <div class="modal-header-wrapper" style="position:relative">
                    ${this.getModalIcon(type)}
                    <h3 class="font-bold">${title || 'Information'}</h3>
                </div>
                <div class="modal-header-subtitle">${subtitle ? subtitle : ''}</div>
                <div class="modal-body-wrapper">
                    ${content || '<p>No details provided.</p>'}
                </div>
            </div>
            <form method="dialog" class="modal-backdrop">
                <button class="modal-closer">close</button>
            </form>
        `;
        if (classList) {
            classList = Array.isArray(classList) ? classList : [classList];
            classList.forEach(cls => {
                modal.classList.add(cls);
            })
        }
        document.body.appendChild(modal);

        const absig = new AbortController();
        const signal = absig.signal;

        // ── Copy-to-clipboard button ──
        if (enableCopy) {
            const table = modal.querySelector('.modal-body-wrapper table');
            if (table) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.title = 'Copy to clipboard';
                btn.style.cssText = 'position:absolute;top:0;right:0;background:none;border:1px solid var(--border-color,#ccc);border-radius:6px;padding:4px 6px;cursor:pointer;display:inline-flex;align-items:center;gap:4px;font-size:12px;color:inherit;opacity:0.7;transition:opacity 0.15s,border-color 0.15s';
                btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>`;
                btn.addEventListener('mouseenter', () => { btn.style.opacity = '1'; });
                btn.addEventListener('mouseleave', () => { if (!btn.dataset.copied) btn.style.opacity = '0.7'; });

                btn.addEventListener('click', async () => {
                    try {
                        const rows = [];
                        for (const tr of table.rows) {
                            const cells = [];
                            for (const td of tr.cells) cells.push(td.textContent);
                            rows.push(cells);
                        }
                        await writeObjectToClipboard(rows, { headers: false });

                        btn.dataset.copied = '1';
                        btn.style.opacity = '1';
                        btn.style.borderColor = 'var(--success-color,#22c55e)';
                        btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--success-color,#22c55e)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg><span style="color:var(--success-color,#22c55e)">Copied</span>`;

                        setTimeout(() => {
                            delete btn.dataset.copied;
                            btn.style.opacity = '0.7';
                            btn.style.borderColor = '';
                            btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>`;
                        }, 2000);
                    } catch (e) {
                        console.error('Clipboard write failed:', e);
                    }
                }, { signal });

                modal.querySelector('.modal-header-wrapper').appendChild(btn);
            }
        }

        const close_el = modal.querySelector('.modal-closer');

        this.context.page.addEventListener(close_el, 'click', () => {
            modal.close();
            absig.abort();
        }, { once: true, signal: signal});
        this.context.page.addEventListener(modal, 'close', () => {
            modal.remove();
            absig.abort();
        }, { once: true, signal: signal});
        modal.showModal();

    }

    destroyAll() {
        for (const pill of this.pills.values()) {
            pill.destroy();
        }
        this.pills.clear();
        this._domQueue.clear();
    }
}
