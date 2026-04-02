

import {v4 as uuidv4} from "uuid";

/**
 * Micro-Grid configuration declarations.
 *
 * Each config defines:
 *   - name:         unique identifier matching the backend MicroGridConfig.name
 *   - displayName:  human-readable title for the tab header
 *   - primaryKeys:  array of PK column names
 *   - columns:      AG Grid column definitions
 *   - addRowDefaults: function that fills default values for new rows
 *
 * Groups bundle multiple micro-grids into a single tabbed modal.
 */

export const SEVERITY_COLOR_MAP = {
    low:   '#FFFF00',   // yellow
    med:   '#FFBF00',   // amber
    high:  '#FF0000',   // red
    other: '#800080',   // purple
};

const BASE_COLORS = Array.from(Object.values(SEVERITY_COLOR_MAP));

// ============================================================================
// Regex / glob auto-detection
// ============================================================================

const REGEX_INDICATORS = /[\\^$*+?.()|[\]{}]/;

export function detectMatchMode(pattern) {
    if (!pattern || typeof pattern !== 'string') return 'literal';
    if (REGEX_INDICATORS.test(pattern)) return 'regex';
    return 'literal';
}

export function isValidRegex(pattern) {
    try { new RegExp(pattern); return true; } catch { return false; }
}

// ============================================================================
// Display mode options — multi-select
// ============================================================================

const DISPLAY_MODES = ['column', 'highlight_row', 'overview_warnings'];
const DISPLAY_MODE_LABELS = {
    column: 'Column',
    highlight_row: 'Row',
    overview_warnings: 'Overview',
};
const DISPLAY_MODE_COLORS = {
    column:            { bg: 'hsl(210,50%,25%)', fg: 'hsl(210,80%,80%)', border: 'hsl(210,50%,35%)' },
    highlight_row:     { bg: 'hsl(45,50%,25%)',  fg: 'hsl(45,80%,80%)',  border: 'hsl(45,50%,35%)' },
    overview_warnings: { bg: 'hsl(0,50%,25%)',   fg: 'hsl(0,80%,80%)',   border: 'hsl(0,50%,35%)' },
};

export class DisplayModeCellEditor {
    init(params) {
        this._selected = new Set(
            (params.value || '').split(',').map(s => s.trim()).filter(Boolean)
        );
        this._el = document.createElement('div');
        this._el.style.cssText = 'padding:6px 8px;display:flex;flex-direction:column;gap:4px;min-width:160px;';
        this._render();
    }

    _render() {
        this._el.innerHTML = '';
        for (const mode of DISPLAY_MODES) {
            const label = document.createElement('label');
            label.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;font-size:12px;color:#ccc;padding:2px 0;';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = this._selected.has(mode);
            cb.addEventListener('change', () => {
                if (cb.checked) this._selected.add(mode);
                else this._selected.delete(mode);
            });
            label.appendChild(cb);
            const c = DISPLAY_MODE_COLORS[mode];
            const pill = document.createElement('span');
            pill.style.cssText = `display:inline-flex;align-items:center;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:500;background:${c.bg};color:${c.fg};border:1px solid ${c.border};`;
            pill.textContent = DISPLAY_MODE_LABELS[mode];
            label.appendChild(pill);
            this._el.appendChild(label);
        }
    }

    getGui() { return this._el; }
    afterGuiAttached() {}
    getValue() { return [...this._selected].join(','); }
    isPopup() { return true; }
    destroy() {}
}

// ============================================================================
// Custom AG Grid cell editor: native color picker
// ============================================================================

export class ColorPickerEditor {
    init(params) {
        this._value = params.value || '#FFFFFF';
        this._el = document.createElement('div');
        this._el.style.cssText = 'display:flex;align-items:center;gap:6px;padding:2px 4px;height:100%;';

        this._input = document.createElement('input');
        this._input.type = 'color';
        this._input.value = this._value;
        this._input.style.cssText = 'width:32px;height:26px;border:none;padding:0;cursor:pointer;background:transparent;';

        this._label = document.createElement('span');
        this._label.style.cssText = 'font-size:12px;color:#ccc;';
        this._label.textContent = this._value;

        this._input.addEventListener('input', () => {
            this._value = this._input.value;
            this._label.textContent = this._value;
        });

        this._el.appendChild(this._input);
        this._el.appendChild(this._label);
    }
    getGui() { return this._el; }
    afterGuiAttached() { this._input.focus(); this._input.click(); }
    getValue() { return this._value; }
    isPopup() { return false; }
    destroy() {}
}

// ============================================================================
// Tags system — hierarchical with custom colors
// ============================================================================

const ALL_KNOWN_TAGS = new Set();
const TAG_CUSTOM_COLORS = new Map();   // tag -> {bg, fg, border}

export function _parseTags(val) {
    if (!val || typeof val !== 'string') return [];
    return val.split(',').map(s => s.trim()).filter(Boolean);
}

export function _parseTagCategory(tag) {
    const idx = tag.indexOf(':');
    if (idx > 0) return { category: tag.slice(0, idx), label: tag.slice(idx + 1) };
    return { category: null, label: tag };
}

function _tagHue(tag) {
    const { label } = _parseTagCategory(tag);
    let h = 0;
    for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) & 0xFFFF;
    return h % 360;
}

function _tagColors(tag) {
    const custom = TAG_CUSTOM_COLORS.get(tag);
    if (custom) return custom;
    const hue = _tagHue(tag);
    return {
        bg: `hsl(${hue},55%,25%)`,
        fg: `hsl(${hue},80%,80%)`,
        border: `hsl(${hue},50%,35%)`,
    };
}

function _generate_trace() {
    try {
        return `micro-${uuidv4()}`;
    } catch (e) {}
    return `micro-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function _createPillEl(tag, showRemove = false, onRemove = null, className=null) {
    const pill = document.createElement('span');
    pill.className = 'micro-tag-pill-wrapper';
    const c = _tagColors(tag);
    pill.classList.add('micro-tag-pill');
    if (className) pill.classList.add(className);
    const { category, label } = _parseTagCategory(tag);
    if (category) {
        const catSpan = document.createElement('span');
        catSpan.style.cssText = 'opacity:0.6;font-size:10px;margin-right:1px;';
        catSpan.textContent = category + ':';
        pill.appendChild(catSpan);
        pill.appendChild(document.createTextNode(label));
    } else {
        pill.textContent = tag;
    }

    if (showRemove && onRemove) {
        const x = document.createElement('span');
        x.textContent = '×';
        x.style.cssText = 'cursor:pointer;margin-left:2px;font-size:13px;line-height:1;';
        x.addEventListener('click', (e) => { e.stopPropagation(); onRemove(); });
        pill.appendChild(x);
    }
    return pill;
}

export function getAllKnownTags() { return ALL_KNOWN_TAGS; }
export function getTagCustomColors() { return TAG_CUSTOM_COLORS; }

export function setTagColor(tag, bg, fg, border) {
    TAG_CUSTOM_COLORS.set(tag, { bg, fg, border });
}

export function removeTagColor(tag) {
    TAG_CUSTOM_COLORS.delete(tag);
}

export function seedKnownTags(rows) {
    for (const row of rows) {
        if (row.tags) {
            for (const t of _parseTags(row.tags)) ALL_KNOWN_TAGS.add(t);
        }
    }
}

/**
 * Group known tags by category. Returns Map<category|'', string[]>.
 */
export function getTagsByCategory() {
    const cats = new Map();
    for (const tag of ALL_KNOWN_TAGS) {
        const { category } = _parseTagCategory(tag);
        const key = category || '';
        if (!cats.has(key)) cats.set(key, []);
        cats.get(key).push(tag);
    }
    for (const arr of cats.values()) arr.sort();
    return cats;
}

// ============================================================================
// Tags cell renderer (pills display)
// ============================================================================

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

export function tagsCellRenderer(params) {
    if (!params.value) return '';
    const tags = _parseTags(params.value);
    return tags.map(tag => {
        let c;
        if (params.data?.color && !BASE_COLORS.includes(params.data.color)) {
            const base = params.data.color;
            c = {bg: base, fg: _lighten(base), border: _darken(base)}
        } else {
            c = _tagColors(tag);
        }
        const { category, label } = _parseTagCategory(tag);
        const catHtml = category ? `<span style="opacity:0.6;font-size:9px;">${_escHtml(category)}:</span>` : '';
        return `<span class="micro-tag-pill" style="display:inline-flex;align-items:center;gap:1px;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:500;background:${c.bg};color:${c.fg};border:1px solid ${c.border};margin-right:3px;">${catHtml}${_escHtml(label)}</span>`;
    }).join('');
}

function _escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ============================================================================
// Custom AG Grid cell editor: tags pill editor
// ============================================================================

export class TagsCellEditor {
    init(params) {
        this._params = params;
        this._tags = _parseTags(params.value);
        this._el = document.createElement('div');
        this._el.className = 'micro-tags-editor';
        this._rendering = false;
        this._disposed = false;
        this._pillEls = [];
        this._input = null;
        this._sugWrap = null;
        this._render();
    }

    _render() {
        if (this._rendering || this._disposed) return;
        this._rendering = true;

        // Detach old event listeners by removing elements explicitly
        this._cleanup();

        this._tags.forEach((tag, i) => {
            const pill = _createPillEl(tag, true, () => {
                this._tags.splice(i, 1);
                this._render();
            });
            this._pillEls.push(pill);
            this._el.appendChild(pill);
        });

        const input = document.createElement('input');
        input.type = 'text';
        input.id = 'micro-tag-input'
        input.placeholder = '+ tag (category:name)';
        input.className = 'micro-tag-input';
        this._input = input;

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ',') {
                e.preventDefault();
                e.stopPropagation();
                const val = input.value.trim();
                if (val && !this._tags.includes(val)) {
                    this._tags.push(val);
                    ALL_KNOWN_TAGS.add(val);
                    input.value = '';
                    this._render();
                } else if (!val && e.key === 'Enter') {
                    // Empty input + Enter → commit tags and close editor
                    this._params.stopEditing();
                }
            } else if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                this._params.stopEditing();
            }
        });
        this._el.appendChild(input);

        // Suggestions grouped by category
        if (ALL_KNOWN_TAGS.size > 0) {
            const missing = [...ALL_KNOWN_TAGS].filter(t => !this._tags.includes(t));
            if (missing.length > 0) {
                const wrap = document.createElement('div');
                wrap.style.cssText = 'width:100%;display:flex;flex-wrap:wrap;gap:3px;margin-top:2px;opacity:0.5;';
                missing.slice(0, 12).forEach(tag => {
                    const sug = _createPillEl(tag, false);
                    sug.style.cursor = 'pointer';
                    sug.style.opacity = '0.6';
                    sug.addEventListener('click', () => {
                        if (!this._tags.includes(tag)) {
                            this._tags.push(tag);
                            this._render();
                        }
                    });
                    wrap.appendChild(sug);
                });
                this._sugWrap = wrap;
                this._el.appendChild(wrap);
            }
        }

        this._rendering = false;

        requestAnimationFrame(() => {
            if (!this._disposed && this._input) this._input.focus();
        });
    }

    _cleanup() {
        // Remove pill elements explicitly to detach listeners
        for (const pill of this._pillEls) pill.remove();
        this._pillEls = [];
        if (this._input) { this._input.remove(); this._input = null; }
        if (this._sugWrap) { this._sugWrap.remove(); this._sugWrap = null; }
    }

    getGui() { return this._el; }
    afterGuiAttached() {
        if (this._input) this._input.focus();
    }
    getValue() { return this._tags.join(','); }
    isPopup() { return true; }
    destroy() {
        this._disposed = true;
        this._cleanup();
    }
}

// ============================================================================
// Tags filter — hierarchical grouping by category
// ============================================================================

export class TagsFilter {
    init(params) {
        this._params = params;
        this._filterTags = new Set();
        this._el = document.createElement('div');
        this._el.style.cssText = 'padding:8px;min-width:200px;max-height:280px;overflow-y:auto;';
        this._render();
    }

    _render() {
        this._el.innerHTML = '';
        const cats = getTagsByCategory();

        if (cats.size === 0) {
            this._el.textContent = 'No tags yet';
            return;
        }

        // Clear button
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.style.cssText = 'font-size:11px;margin-bottom:6px;cursor:pointer;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;padding:2px 8px;';
        clearBtn.addEventListener('click', () => {
            this._filterTags.clear();
            this._render();
            this._params.filterChangedCallback();
        });
        this._el.appendChild(clearBtn);

        // Render each category
        const sortedCats = [...cats.keys()].sort((a, b) => {
            if (a === '') return 1;
            if (b === '') return -1;
            return a.localeCompare(b);
        });

        for (const cat of sortedCats) {
            const tags = cats.get(cat);
            if (cat) {
                const catHeader = document.createElement('div');
                catHeader.style.cssText = 'font-size:10px;font-weight:600;color:#888;text-transform:uppercase;margin:6px 0 2px;letter-spacing:0.5px;';
                catHeader.textContent = cat;
                this._el.appendChild(catHeader);
            }
            for (const tag of tags) {
                const row = document.createElement('label');
                row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;font-size:12px;color:#ccc;';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = this._filterTags.has(tag);
                cb.addEventListener('change', () => {
                    if (cb.checked) this._filterTags.add(tag);
                    else this._filterTags.delete(tag);
                    this._params.filterChangedCallback();
                });
                row.appendChild(cb);
                row.appendChild(_createPillEl(tag, false));
                this._el.appendChild(row);
            }
        }
    }

    getGui() { return this._el; }
    doesFilterPass(params) {
        if (this._filterTags.size === 0) return true;
        const rowTags = _parseTags(params.data?.tags);
        return rowTags.some(t => this._filterTags.has(t));
    }
    isFilterActive() { return this._filterTags.size > 0; }
    getModel() {
        if (this._filterTags.size === 0) return null;
        return { tags: [...this._filterTags] };
    }
    setModel(model) {
        this._filterTags = new Set(model?.tags || []);
        this._render();
    }
    afterGuiAttached() { this._render(); }
    destroy() {}
}

// ============================================================================
// Pattern cell renderer — teal highlight for regex patterns
// ============================================================================

function patternCellRenderer(params) {
    const val = params.value;
    if (!val) return '';
    const mode = detectMatchMode(val);
    if (mode === 'regex') {
        const valid = isValidRegex(val);
        const color = valid ? '#2dd4bf' : '#f87171';  // teal or red for invalid
        const title = valid ? 'Regex pattern' : 'Invalid regex';
        return `<span style="color:${color};font-family:monospace;font-size:12px;" title="${title}">${_escHtml(val)}</span>`;
    }
    return _escHtml(val);
}

// ============================================================================
// Column definitions: hot_tickers
// ============================================================================

// ============================================================================
// Category badge colors for redist_params
// ============================================================================

const REDIST_CATEGORY_COLORS = {
    charge_strength: { bg: 'hsl(200,50%,22%)', fg: 'hsl(200,80%,78%)', border: 'hsl(200,50%,32%)' },
    risk_weight:     { bg: 'hsl(270,50%,22%)', fg: 'hsl(270,80%,78%)', border: 'hsl(270,50%,32%)' },
    solver_default:  { bg: 'hsl(140,50%,22%)', fg: 'hsl(140,80%,78%)', border: 'hsl(140,50%,32%)' },
};

function _redistCategoryRenderer(params) {
    const val = params.value;
    if (!val) return '';
    const c = REDIST_CATEGORY_COLORS[val] || { bg: '#333', fg: '#ccc', border: '#555' };
    const label = val.replace(/_/g, ' ');
    return `<span style="display:inline-flex;align-items:center;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:500;background:${c.bg};color:${c.fg};border:1px solid ${c.border};text-transform:uppercase;letter-spacing:0.3px;">${_escHtml(label)}</span>`;
}

function _redistValueRenderer(params) {
    const v = params.value;
    if (v == null || v === '') return '-';
    const n = +v;
    if (!Number.isFinite(n)) return '-';
    const row = params.data;
    const max = row?.max_value;
    const min = row?.min_value;
    const k = row?.param_key;
    if (max != null && max <= 1.0 && min != null && min >= 0.0 && (k !== 'buffer_mode') && (k !== 'normalize_objective_by_bucket')) {
        return (n * 100).toFixed(1) + '%';
    }
    return n % 1 === 0 ? n.toFixed(0) : n.toFixed(4);
}

export const MICRO_GRID_CONFIGS = {

    hot_tickers: {
        name: 'hot_tickers',
        displayName: `<div class="hot-ticker-modal">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M9.616 3.642c1.058-1.839 3.71-1.839 4.768 0l8.043 13.988c1.054 1.833-.27 4.12-2.384 4.12H3.957c-2.115 0-3.438-2.287-2.384-4.12zM12 8.25a.75.75 0 0 1 .75.75v4a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75m.568 9.25a.75.75 0 0 0-1.115-1.003l-.01.011a.75.75 0 0 0 1.114 1.004z" clip-rule="evenodd"/></svg>
            <span>User Flags</span>
            </div>`,
        primaryKeys: ['id'],
        columns: [
            {
                headerCheckboxSelection: true,
                headerCheckboxSelectionFilteredOnly: true,
                checkboxSelection: true,
                width: 42,
                maxWidth: 42,
                pinned: 'left',
                suppressHeaderMenuButton: true,
                sortable: false,
                filter: false,
                resizable: false,
                editable: false,
                suppressColumnsToolPanel: true,
                headerName: '',
            },
            {
                field: 'id',
                hide: true,
                suppressColumnsToolPanel: true,
            },
            {
                field: 'column',
                headerName: 'Column',
                editable: true,
                width: 120,
                cellEditor: 'agSelectCellEditor',
                cellEditorParams: {
                    values: ['ticker', 'isin', 'cusip', 'description'],
                },
            },
            {
                field: 'pattern',
                headerName: 'Pattern',
                editable: true,
                flex: 1,
                minWidth: 140,
                cellEditor: 'agTextCellEditor',
                cellRenderer: patternCellRenderer,
            },
            {
                field: 'match_mode',
                headerName: 'Match Mode',
                editable: false,
                width: 100,
                hide: true,
                suppressColumnsToolPanel: false,
            },
            {
                field: 'severity',
                headerName: 'Severity',
                editable: true,
                width: 100,
                cellEditor: 'agSelectCellEditor',
                cellEditorParams: {
                    values: ['low', 'med', 'high', 'other'],
                },
                cellStyle: (params) => {
                    const color = SEVERITY_COLOR_MAP[params.value];
                    return color
                        ? { backgroundColor: color + '30', borderLeft: `3px solid ${color}` }
                        : null;
                },
            },
            {
                field: 'display',
                headerName: 'Display',
                editable: true,
                width: 160,
                cellEditor: DisplayModeCellEditor,
                cellRenderer: (params) => {
                    if (!params.value) return '';
                    const modes = params.value.split(',').map(s => s.trim()).filter(Boolean);
                    return modes.map(m => {
                        const label = DISPLAY_MODE_LABELS[m] || m;
                        const color = DISPLAY_MODE_COLORS[m] || { bg: '#333', fg: '#ccc', border: '#555' };
                        return `<span class="display-mode-hot-ticker ${m}-mode">${_escHtml(label)}</span>`;
                    }).join('');
                },
                filter: false,
            },
            {
                field: 'color',
                headerName: 'Color',
                editable: true,
                width: 90,
                cellRenderer: (params) => {
                    if (!params.value) return '';
                    return `<span style="display:inline-block;width:16px;height:16px;border-radius:3px;background:${params.value};vertical-align:middle;margin-right:6px;border:1px solid ${_lighten(params.value)}"></span>${_escHtml(params.value)}`;
                },
                cellEditor: ColorPickerEditor,
            },
            {
                field: 'tags',
                headerName: 'Tags',
                editable: true,
                width: 160,
                cellRenderer: tagsCellRenderer,
                cellEditor: TagsCellEditor,
                filter: TagsFilter,
                suppressKeyboardEvent: (params) => {
                    if (!params.editing) return false;
                    const key = params.event.key;
                    return key === 'Enter' || key === ',' || key === 'Backspace';
                },
            },
            {
                field: 'notes',
                headerName: 'Notes',
                editable: true,
                width: 150,
                cellEditor: 'agTextCellEditor',
            },
            {
                field: 'updated_at',
                headerName: 'Updated',
                editable: false,
                width: 140,
                sort: 'desc',
            },
            {
                field: 'updated_by',
                headerName: 'By',
                editable: false,
                width: 80,
            },
        ],
        addRowDefaults: (row = {}) => ({
            id: _generate_trace(),
            column: 'ticker',
            pattern: '',
            match_mode: 'literal',
            severity: 'low',
            display: 'column',
            color: SEVERITY_COLOR_MAP['low'],
            tags: '',
            notes: '',
            updated_at: '',
            updated_by: '',
            ...row,
        }),
    },

    redist_params: {
        name: 'redist_params',
        displayName: `<div style="display:flex;align-items:center;gap:6px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M19.14 12.94c.04-.3.06-.61.06-.94c0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.49.49 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.48.48 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.49.49 0 0 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61zM12 15.6A3.6 3.6 0 1 1 12 8.4a3.6 3.6 0 0 1 0 7.2"/></svg>
            <span>Solver Config</span>
        </div>`,
        primaryKeys: ['param_key'],
        columns: [
            {
                field: 'category',
                headerName: 'Category',
                editable: false,
                width: 140,
                cellRenderer: _redistCategoryRenderer,
                sort: 'asc',
                sortIndex: 0,
            },
            {
                field: 'param_key',
                headerName: 'Parameter',
                editable: false,
                width: 160,
                cellStyle: { fontWeight: 600, fontFamily: 'monospace', fontSize: '12px' },
            },
            {
                field: 'param_value',
                headerName: 'Value',
                editable: true,
                width: 120,
                cellEditor: 'agNumberCellEditor',
                cellRenderer: _redistValueRenderer,
                cellStyle: (params) => {
                    const row = params.data;
                    const v = +params.value;
                    const min = row?.min_value;
                    const max = row?.max_value;
                    if (min != null && v < min) return { color: '#f87171' };
                    if (max != null && v > max) return { color: '#f87171' };
                    return { color: 'var(--rdm-accent, #60a5fa)', fontWeight: 600 };
                },
            },
            {
                field: 'description',
                headerName: 'Description',
                editable: false,
                flex: 1,
                minWidth: 200,
                cellStyle: { opacity: 0.7, fontSize: '11px' },
            },
            {
                field: 'min_value',
                headerName: 'Min',
                editable: false,
                width: 70,
                hide: true,
                suppressColumnsToolPanel: false,
            },
            {
                field: 'max_value',
                headerName: 'Max',
                editable: false,
                width: 70,
                hide: true,
                suppressColumnsToolPanel: false,
            },
            {
                field: 'updated_at',
                headerName: 'Updated',
                editable: false,
                width: 140,
            },
            {
                field: 'updated_by',
                headerName: 'By',
                editable: false,
                width: 80,
            },
        ],
        addRowDefaults: null,  // no manual row adds — seeded by backend
    },
};

export const MICRO_GRID_GROUPS = {

    pt_tools: {
        name: 'pt_tools',
        displayName: 'Bond Flags',
        grids: ['hot_tickers'],
    },
};
