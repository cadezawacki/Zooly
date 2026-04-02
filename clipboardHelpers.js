
// clipboard-utils.js
import * as clipboard from "clipboard-polyfill";
import copy from "copy-to-clipboard";
import {NumberFormatter} from "@/utils/NumberFormatter.js";


/* ================================
   Public API
   ================================ */

export async function writeObjectToClipboard(input, options = {}) {
    const cfg = withDefaults(options);
    const type = cfg.objectTypeOverride || detectType(input);

    const { headers, rows2D, headerNames } = normalizeTo2D(input, type);

    const sliced = sliceRowsAndColumns(rows2D, headers, cfg);
    const withHeaders = cfg.headers ? applyHeaderOverrides(sliced, headerNames, cfg) : sliced;

    const matrix = cfg.transpose ? transpose(withHeaders) : withHeaders;
    const headerRow = cfg.headers ? matrix[0] : null;

    const formatted = formatMatrix(matrix, {
        headerRow,
        addCommaToNumerics: cfg.addCommaToNumerics,
        valueFormatter: cfg.valueFormatter,
        nullOverride: cfg.nullOverride,
        treatEmptyStringAsNull: cfg.treatEmptyStringAsNull
    });

    switch ((cfg.asFormat || "excel").toLowerCase()) {
        case "excel":
            await copyAsExcel(formatted, cfg.delimiterOverride ?? "\t");
            break;
        case "csv":
            await copyAsDelimited(formatted, cfg.delimiterOverride ?? ",");
            break;
        case "json":
            await copyAsJSON(formatted, cfg.headers);
            break;
        case "ascii":
            await copyAsASCII(formatted);
            break;
        default:
            throw new Error(`Unsupported asFormat: ${cfg.asFormat}`);
    }
}

export async function writeStringToClipboard(strOrObj, options = {}) {
    const text = typeof strOrObj === "string" ? strOrObj : safeJSONStringify(strOrObj);
    await bestEffortClipboardWrite({ "text/plain": text });
}

export async function writeHeadersToClipboard(input, options = {}) {
    const { format = "delimited", delimiter = ",", objectTypeOverride } = options || {};
    const type = objectTypeOverride || detectType(input);
    const { headerNames } = normalizeTo2D(input, type);

    let out = "";
    if (format === "delimited") {
        out = headerNames.join(delimiter);
    } else if (format === "python") {
        out = `[${headerNames.map(quotePy).join(", ")}]`;
    } else if (format === "rows") {
        out = headerNames.join("\n");
    } else {
        throw new Error(`Unsupported header format: ${format}`);
    }
    await bestEffortClipboardWrite({ "text/plain": out });
}

/* ================================
   Config, Detection, Normalization
   ================================ */

function withDefaults(options) {
    return {
        rows: null,
        columns: null,
        transpose: false,
        headers: true,
        headerOverride: null,
        asFormat: "excel",
        delimiterOverride: undefined,
        objectTypeOverride: null,
        addCommaToNumerics: false,
        valueFormatter: null, // function or { headerName: fn }
        nullOverride: "",
        treatEmptyStringAsNull: false,
        ...options
    };
}

function detectType(input) {
    if (input == null) return "null";
    if (isArrowTable(input)) return "arrow";
    if (Array.isArray(input)) {
        if (input.length === 0) return "array-of-objects";
        if (isPlainObject(input[0])) return "array-of-objects";
        if (Array.isArray(input[0])) return "array-of-arrays";
    }
    if (input instanceof Map) return "map";
    if (isPlainObject(input)) return "object";
    if (typeof input === "string") {
        const trimmed = input.trim();
        if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
            try { JSON.parse(trimmed); return "json-string"; } catch { /* fall through */ }
        }
    }
    if (typeof input === "object") return "json"; // generic JSON-like
    return "unknown";
}

function isPlainObject(v) {
    return Object.prototype.toString.call(v) === "[object Object]";
}

function isArrowTable(t) {
    // Loose detection for apache arrow-js table
    return t && typeof t === "object" && ("schema" in t) && ("numRows" in t) && (typeof t.getChild === "function" || typeof t.getChildAt === "function");
}

/* Normalize any supported input into:
   - headers: string[] | null
   - rows2D: string[][] (raw values, not yet formatted)
   - headerNames: string[] (original header names)
*/
function normalizeTo2D(input, type) {
    switch (type) {
        case "null": return { headers: [], rows2D: [], headerNames: [] };
        case "json-string": return normalizeTo2D(JSON.parse(input), detectType(JSON.parse(input)));
        case "json": return normalizeFromJSON(input);
        case "object": return normalizeFromObject(input);
        case "array-of-objects": return normalizeFromArrayOfObjects(input);
        case "array-of-arrays": return normalizeFromArrayOfArrays(input);
        case "map": return normalizeFromMap(input);
        case "arrow": return normalizeFromArrow(input);
        default:
            // Fallback: try JSON
            return normalizeFromJSON(input);
    }
}

function normalizeFromObject(obj) {
    // Object values may be single values or arrays; columns are object keys
    const keys = Object.keys(obj);
    const maxLen = Math.max(...keys.map(k => toArray(obj[k]).length), 0);
    const rows = new Array(maxLen).fill(null).map((_, i) => keys.map(k => toArray(obj[k])[i]));
    return { headers: keys, rows2D: rows, headerNames: keys };
}

function normalizeFromArrayOfObjects(arr) {
    // Values must be single; gather union of keys in order of first occurrence
    const headerSet = [];
    for (const row of arr) {
        for (const k of Object.keys(row)) if (!headerSet.includes(k)) headerSet.push(k);
    }
    const rows = arr.map(obj => headerSet.map(k => obj[k]));
    return { headers: headerSet, rows2D: rows, headerNames: headerSet };
}

function normalizeFromArrayOfArrays(arr) {
    if (arr.length === 0) return { headers: [], rows2D: [], headerNames: [] };
    // Assume first row is data; no headers inferred
    const maxCols = Math.max(...arr.map(r => r.length));
    const padded = arr.map(r => {
        const row = new Array(maxCols);
        for (let i = 0; i < maxCols; i++) row[i] = r[i];
        return row;
    });
    const headers = new Array(maxCols).fill(null).map((_, i) => `Col${i + 1}`);
    return { headers, rows2D: padded, headerNames: headers };
}

function normalizeFromMap(map) {
    // Keys become columns; values are single or arrays
    const keys = Array.from(map.keys());
    const maxLen = Math.max(...keys.map(k => toArray(map.get(k)).length), 0);
    const rows = new Array(maxLen).fill(null).map((_, i) => keys.map(k => toArray(map.get(k))[i]));
    return { headers: keys, rows2D: rows, headerNames: keys };
}

function normalizeFromJSON(jsonish) {
    // Try to coerce various shapes
    if (Array.isArray(jsonish)) {
        if (jsonish.length === 0) return { headers: [], rows2D: [], headerNames: [] };
        if (isPlainObject(jsonish[0])) return normalizeFromArrayOfObjects(jsonish);
        if (Array.isArray(jsonish[0])) return normalizeFromArrayOfArrays(jsonish);
    }
    if (isPlainObject(jsonish)) return normalizeFromObject(jsonish);
    // Scalar → single cell
    return { headers: ["Value"], rows2D: [[jsonish]], headerNames: ["Value"] };
}

function normalizeFromArrow(table) {
    const names = table.schema.fields.map(f => f.name);
    const cols = names.map(name => {
        const col = table.getChild ? table.getChild(name) : table.getChildAt(names.indexOf(name));
        return col?.toArray ? Array.from(col.toArray()) : [];
    });
    const maxLen = Math.max(...cols.map(c => c.length), 0);
    const rows = new Array(maxLen).fill(null).map((_, i) => cols.map(c => c[i]));
    return { headers: names, rows2D: rows, headerNames: names };
}

/* ================================
   Slicing, Header overrides, Transpose
   ================================ */

function sliceRowsAndColumns(rows2D, headers, cfg) {
    const rowIdxs = coerceRowSelection(cfg.rows, rows2D.length);
    const { indices: colIdxs, names: colNames } = coerceColumnSelection(cfg.columns, headers);
    const out = [];
    if (cfg.headers) out.push(colNames);
    for (const r of rowIdxs) {
        const src = rows2D[r] || [];
        out.push(colIdxs.map(c => src[c]));
    }
    return out;
}

function coerceRowSelection(rows, total) {
    if (rows == null) return range(0, total);
    if (Number.isInteger(rows)) return [clamp(rows, 0, total - 1)];
    if (Array.isArray(rows)) {
        return rows
            .filter(n => Number.isInteger(n))
            .map(n => clamp(n, 0, total - 1));
    }
    throw new Error("rows must be null, integer, or array of integers");
}

function coerceColumnSelection(columns, headers) {
    const total = headers.length;
    if (columns == null) return { indices: range(0, total), names: headers.slice() };
    if (typeof columns === "string") {
        const idx = headers.indexOf(columns);
        if (idx === -1) throw new Error(`Column not found: ${columns}`);
        return { indices: [idx], names: [headers[idx]] };
    }
    if (Array.isArray(columns)) {
        const indices = columns.map(c => {
            const i = headers.indexOf(String(c));
            if (i === -1) throw new Error(`Column not found: ${c}`);
            return i;
        });
        return { indices, names: indices.map(i => headers[i]) };
    }
    throw new Error("columns must be null, string, or array of strings");
}

function applyHeaderOverrides(matrix, headerNames, cfg) {
    if (!cfg.headers) return matrix;
    if (!cfg.headerOverride) return matrix;
    const out = matrix.slice();
    const hdr = matrix[0].slice();
    if (Array.isArray(cfg.headerOverride)) {
        if (cfg.columns && Array.isArray(cfg.columns) && cfg.headerOverride.length !== cfg.columns.length) {
            throw new Error("headerOverride length must match selected columns");
        }
        for (let i = 0; i < hdr.length; i++) hdr[i] = cfg.headerOverride[i] ?? hdr[i];
    } else if (isPlainObject(cfg.headerOverride)) {
        for (let i = 0; i < hdr.length; i++) {
            const current = hdr[i];
            if (Object.prototype.hasOwnProperty.call(cfg.headerOverride, current)) {
                hdr[i] = cfg.headerOverride[current];
            }
        }
    } else {
        throw new Error("headerOverride must be an array or object");
    }
    out[0] = hdr;
    return out;
}

function transpose(matrix) {
    if (matrix.length === 0) return matrix;
    const rows = matrix.length;
    const cols = Math.max(...matrix.map(r => r.length));
    const t = new Array(cols).fill(null).map(() => new Array(rows));
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) t[c][r] = matrix[r][c];
    return t;
}

/* ================================
   Formatting
   ================================ */

function formatMatrix(matrix, { headerRow, addCommaToNumerics, valueFormatter, nullOverride, treatEmptyStringAsNull }) {
    // Build per-column value formatters
    const perColFormatter = Array.isArray(headerRow) && isPlainObject(valueFormatter)
        ? headerRow.reduce((acc, name, i) => { acc[i] = typeof valueFormatter[name] === "function" ? valueFormatter[name] : null; return acc; }, {})
        : {};

    const perColNull = Array.isArray(headerRow) && isPlainObject(nullOverride)
        ? headerRow.reduce((acc, name, i) => { acc[i] = name in nullOverride ? nullOverride[name] : undefined; return acc; }, {})
        : {};

    const globalFormatter = typeof valueFormatter === "function" ? valueFormatter : null;
    const globalNull = !isPlainObject(nullOverride) ? nullOverride : "";

    const out = new Array(matrix.length);
    for (let r = 0; r < matrix.length; r++) {
        const row = matrix[r] || [];
        const formattedRow = new Array(row.length);
        for (let c = 0; c < row.length; c++) {
            const raw = row[c];
            let v = normalizeCell(raw, { treatEmptyStringAsNull, globalNull, colNull: perColNull[c] });
            v = addCommaToNumerics && isFiniteNumber(v) ? numberWithCommas(Number(v)) : v;
            const fmt = perColFormatter[c] || globalFormatter;
            formattedRow[c] = typeof fmt === "function" ? guardFormat(fmt, v, r, c) : toCellString(v);
        }
        out[r] = formattedRow;
    }
    return out;
}

function normalizeCell(value, { treatEmptyStringAsNull, globalNull, colNull }) {
    const nullReplacement = colNull !== undefined ? colNull : (globalNull !== undefined ? globalNull : "");
    if (value === null || value === undefined) return nullReplacement ?? "";
    if (treatEmptyStringAsNull && value === "") return nullReplacement ?? "";
    return value;
}

function toCellString(v) {
    if (v == null) return "";
    if (typeof v === "string") return v;
    if (typeof v === "number" && Number.isFinite(v)) return String(v);
    if (typeof v === "boolean") return v ? "TRUE" : "FALSE";
    // Flatten simple objects/arrays
    if (Array.isArray(v) || isPlainObject(v)) return safeJSONStringify(v);
    return String(v);
}

function guardFormat(fn, value, rowIndex, colIndex) {
    try { return toCellString(fn(value, { rowIndex, colIndex })); }
    catch { return toCellString(value); }
}

function isFiniteNumber(v) {
    if (typeof v === "number") return Number.isFinite(v);
    if (typeof v === "string" && v.trim() !== "") return Number.isFinite(Number(v));
    return false;
}

function numberWithCommas(n) {
    // Avoid locale for determinism and speed
    const parts = String(n).split(".");
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return parts.join(".");
}

/* ================================
   Output encoders
   ================================ */

async function copyAsExcel(matrix, delimiter) {
    // Sanitize for TSV/plain channel
    const tsv = matrix.map(row => row.map(sanitizeCellForTSV).join(delimiter)).join("\r\n");
    const html = excelHTML(matrix);
    await bestEffortClipboardWrite({ "text/html": html, "text/plain": tsv });
}

async function copyAsDelimited(matrix, delimiter) {
    const text = matrix.map(row => row.map(s => csvEscape(String(s), delimiter)).join(delimiter)).join("\r\n");
    await bestEffortClipboardWrite({ "text/plain": text });
}

async function copyAsJSON(matrix, hasHeader) {
    let out;
    if (hasHeader && matrix.length >= 1) {
        const [hdr, ...rows] = matrix;
        const objs = rows.map(r => {
            const o = {};
            for (let i = 0; i < hdr.length; i++) o[hdr[i]] = r[i];
            return o;
        });
        out = safeJSONStringify(objs);
    } else {
        out = safeJSONStringify(matrix);
    }
    await bestEffortClipboardWrite({ "application/json": out, "text/plain": out });
}

async function copyAsASCII(matrix) {
    const widths = colWidths(matrix);
    const text = matrix.map(row =>
        row.map((cell, i) => padRight(String(cell ?? ""), widths[i])).join(" | ")
    ).join("\n");
    await bestEffortClipboardWrite({ "text/plain": text });
}

/* ================================
   Clipboard write (secure + fallback)
   ================================ */

async function bestEffortClipboardWrite(mimeToData) {
    const types = Object.keys(mimeToData);
    const blobs = {};
    for (const t of types) blobs[t] = asBlob(mimeToData[t], t);

    try {
        const item = new clipboard.ClipboardItem(Object.fromEntries(Object.entries(blobs)));
        await clipboard.write([item]);
    } catch (err) {
        // Fallbacks for insecure hosts
        if (mimeToData["text/html"]) {
            try { copy(mimeToData["text/html"], { format: "text/html" }); return; } catch { /* fall through */ }
        }
        const plain = mimeToData["text/plain"] ?? mimeToData["application/json"] ?? firstValue(mimeToData);
        copy(String(plain), { format: "text/plain" });
    }
}

/* ================================
   HTML & helpers
   ================================ */

function excelHTML(matrix) {
    const body = matrix.map(row =>
        `<tr style="height:15.0pt">${row.map(cell => `<td class="xl65">${escapeHTML(String(cell ?? ""))}</td>`).join("")}</tr>`
    ).join("");
    return `<!DOCTYPE html>
<html xmlns:v="urn:schemas-microsoft-com:vml"
      xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:x="urn:schemas-microsoft-com:office:excel"
      xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta http-equiv=Content-Type content="text/html; charset=utf-8">
<meta name=ProgId content=Excel.Sheet>
<meta name=Generator content="Microsoft Excel 15">
<style>
.xl24{height:15.0pt;white-space:nowrap;mso-wrap-style:none;}
.xl65{height:15.0pt;white-space:nowrap;mso-wrap-style:none;}
</style>
</head>
<!doctype> <body link="#467886" vlink="#96607D"> <!doctype>
<body>
<table border="0" cellpadding="0" cellspacing="0">
${body}
</table>
</body>
</html>`;
}

function csvEscape(s, delimiter) {
    const needsQuotes = s.includes(delimiter) || s.includes("\n") || s.includes("\r") || s.includes('"');
    let out = s.replace(/"/g, '""');
    return needsQuotes ? `"${out}"` : out;
}

function sanitizeCellForTSV(s) {
    return String(s).replace(/[\n\r\t]+/g, " ");
}

function toTSV(matrix) {
    return matrix.map((row) => row.join('\t')).join('\n');
}

function escapeHTML(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function toCSV(matrix) {
    function esc(val) {
        const s = String(val ?? '');
        if (s.includes('"') || s.includes(',') || s.includes('\n')) return `"${s.replace(/"/g, '""')}"`;
        return s;
    }
    return matrix.map((row) => row.map(esc).join(',')).join('\n');
}

function toHtmlTable(matrix) {
    const rows = matrix
        .map((row) => `<tr>${row.map((c) => `<td>${escapeHTML(c)}</td>`).join('')}</tr>`)
        .join('');
    return `<table>${rows}</table>`;
}

function toArray(v) { return Array.isArray(v) ? v : [v]; }

function range(a, b) { const out = new Array(b - a); for (let i = 0; i < out.length; i++) out[i] = a + i; return out; }

function clamp(n, min, max) { return Math.max(min, Math.min(max, n)); }

function colWidths(matrix) {
    const cols = Math.max(...matrix.map(r => r.length), 0);
    const widths = new Array(cols).fill(0);
    for (const row of matrix) for (let c = 0; c < cols; c++) widths[c] = Math.max(widths[c], String(row[c] ?? "").length);
    return widths;
}

function padRight(s, w) {
    if (s.length >= w) return s;
    return s + " ".repeat(w - s.length);
}

function safeJSONStringify(v) {
    const seen = new WeakSet();
    return JSON.stringify(v, function (k, val) {
        if (typeof val === "object" && val !== null) {
            if (seen.has(val)) return "[Circular]";
            seen.add(val);
        }
        if (val instanceof Map) return Object.fromEntries(val);
        return val;
    });
}

function asBlob(data, mime) { return new Blob([data], { type: mime }); }

function firstValue(obj) { for (const k in obj) return obj[k]; return ""; }

function quotePy(s) {
    const q = String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    return `'${q}'`;
}

/* ================================
   Venue Defaults
   ================================ */

export function mapVenueFromRfq(venue) {
    if (venue == null) {
        return 'ib'
    } else if (venue === 'ib') {
        return venue
    } else if (/BX|BBG|BLOOMBERG/i.test(venue)) {
        return 'bbg'
    } else if (/MX|mx|MarketAxess|MXUSCRPTTrading|MarketAxessEUPT/i.test(venue)) {
        return 'mx'
    } else if (/TW|tw|TradewebEUPT|Tradeweb|TradewebCORIPT|CORI/i.test(venue)) {
        return 'tw'
    } else if (/TRUMID|tm|trmud|/i.test(venue)) {
        return 'tm'
    } else {
        window.page.toastManager().error('Unknown venue, copying generic fields!', 'Copy Warning')
        return 'ib'
    }
}

export function getVenueCopyColumns(venue, normalize=true) {
    let columns;
    if (normalize) {
        venue = mapVenueFromRfq(venue);
    }
    switch (venue) {
        case 'mx':
            columns = {
                'isin': 'Identifier',
                'newLevel': ['Level', (lvl) => Number(parseFloat(lvl).toFixed(3))],
                'grossSize': ['Sz(000s)', (size) => size / 1000],
                'originalSide': ['Side', (side) => side.toUpperCase() === 'BUY' ? 'BID' : 'OFFER'],
                'tnum': 'Inquiry #'
            }
            break;
        case 'tw':
            columns = {
                'tnum': 'tnum',
                'newLevel': ['Quote', (lvl) => Number(parseFloat(lvl).toFixed(3))],
                'isin': 'ISIN',
            }
            break;
        case 'bbg':
            columns = {
                'isin': 'Identifier',
                'newLevel': ['Quote', (lvl) => Number(parseFloat(lvl).toFixed(3))],
                'grossSize': 'Amount',
                'originalSide': ['Side', (side) => `DLR ${side}`]
            }
            break;
        case 'tm':
            columns = {
                'isin': 'ID',
                'newLevel': ['Quote', (lvl) => Number(parseFloat(lvl).toFixed(3))],
            }
            break;
        case 'ib':
            columns = {
                'desigFullName': 'Trader',
                'description': 'Bond',
                'originalSide': ['Barc Side', (side) => side.toUpperCase() === 'BUY' ? 'Bid' : 'Offer'],
                'grossSize': ['Notional', (size) => NumberFormatter.numberWithCommas(size)],
                'newLevel': 'Level',
            }
            break;
    }
    return columns
}
