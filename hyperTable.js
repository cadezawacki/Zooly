
import * as arrow from 'apache-arrow';
import * as aq from 'arquero';
import { SparsePatchCodec } from '@/utils/sparsePatchCodec.js';
import { OptimizedColumnarCodec } from '@/global/optimizedColumnarCodec.js';
import * as fletch from '@uwdata/flechette';
import { CompressionType, setCompressionCodec } from '@uwdata/flechette';
import * as lz4 from 'lz4js';

setCompressionCodec(CompressionType.LZ4_FRAME, {
    encode: (data) => lz4.compress(data),
    decode: (data) => lz4.decompress(data)
});

window.fletch = fletch;
window.arrow = arrow;

// ------------------------------------------------------------------------------
// Constants & basic helpers
// ------------------------------------------------------------------------------

const KEY_DELIM = '§';
const NULL_VALUES = [null, undefined];

// Small IN lists are faster with linear scan than Set allocations
const IN_SET_LINEAR_LIMIT = 8;

// For take fallback: if we have too many “runs” of rows, use per-column builders instead
const TAKE_SLICE_RUN_LIMIT = 1024;

function _isFn(v) {
    return typeof v === 'function';
}
function _isObj(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
}
function _isDate(v) {
    return v instanceof Date;
}
function _escapeRegex(text) {
    return String(text).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function _hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
}

function _bitIsSet(bitmap, bitIndex) {
    return (bitmap[bitIndex >> 3] & (1 << (bitIndex & 7))) !== 0;
}

function _getChild(tbl, name) {
    if (!tbl) return null;
    if (typeof tbl.getChild === 'function') return tbl.getChild(name);
    if (typeof tbl.getColumn === 'function') return tbl.getColumn(name);
    return null;
}

function _makeTableFromVectors(children, schema = null) {
    // Arrow Table constructor supports preserving schema: new Table(schema, {col: vec, ...})
    // If schema is omitted, it infers from vectors.
    try {
        if (schema && schema.fields) {
            return new arrow.Table(schema, children);
        }
    } catch {
        // fall through to inferred schema
    }
    return new arrow.Table(children);
}

function _normalizeOp(op) {
    const raw = String(op ?? '').trim();
    const upper = raw.toUpperCase();

    // Common symbol aliases
    if (upper === '=' || upper === '==' || upper === '===') return 'EQ';
    if (upper === '!=' || upper === '!==') return 'NEQ';
    if (upper === '<') return 'LT';
    if (upper === '<=') return 'LTE';
    if (upper === '>') return 'GT';
    if (upper === '>=') return 'GTE';

    // Slight spelling variants
    if (upper === 'ISNULL') return 'IS_NULL';
    if (upper === 'ISNOTNULL') return 'IS_NOT_NULL';
    if (upper === 'NOTIN') return 'NOT_IN';
    if (upper === 'STARTSWITH') return 'STARTS_WITH';
    if (upper === 'ENDSWITH') return 'ENDS_WITH';

    return upper;
}

function _schemaFieldMap(schema) {
    const m = new Map();
    if (!schema || !schema.fields) return m;
    for (const f of schema.fields) m.set(f.name, f);
    return m;
}

function _arrowTypeIsInt64(t) {
    return t && t.typeId === arrow.Type.Int && t.bitWidth === 64;
}
function _arrowTypeIsTimestamp(t) {
    return t && t.typeId === arrow.Type.Timestamp;
}
function _arrowTypeIsDate(t) {
    return t && (t.typeId === arrow.Type.Date);
}
function _arrowTypeIsBool(t) {
    return t && t.typeId === arrow.Type.Bool;
}
function _arrowTypeIsUtf8(t) {
    return t && t.typeId === arrow.Type.Utf8;
}
function _arrowTypeIsFloat(t) {
    return t && t.typeId === arrow.Type.Float;
}
function _arrowTypeIsInt(t) {
    return t && t.typeId === arrow.Type.Int;
}
function _arrowTypeIsNumeric(t) {
    return _arrowTypeIsInt(t) || _arrowTypeIsFloat(t);
}

function _makeValueCoercer(type) {
    // Produces a function that coerces JS inputs to what Arrow builders/vectors expect.
    // Keep it permissive to preserve backwards compatibility.
    if (!type) return (v) => v;

    if (_arrowTypeIsUtf8(type)) {
        return (v) => {
            if (v === null || v === undefined) return null;
            if (typeof v === 'string') return v;
            if (typeof v === 'bigint') return v.toString();
            if (_isDate(v)) return String(v); // preserve legacy String(Date) behavior
            if (typeof v === 'object') {
                try { return JSON.stringify(v); } catch { return String(v); }
            }
            return String(v);
        };
    }

    if (_arrowTypeIsBool(type)) {
        return (v) => {
            if (v === null || v === undefined) return null;
            return Boolean(v);
        };
    }

    if (_arrowTypeIsNumeric(type)) {
        // For int64/timestamp we should avoid lossy Number casts. Keep BigInt if possible.
        if (_arrowTypeIsInt64(type)) {
            return (v) => {
                if (v === null || v === undefined) return null;
                if (typeof v === 'bigint') return v;
                // If user supplies a safe integer, allow BigInt conversion
                if (typeof v === 'number' && Number.isFinite(v)) {
                    // only convert safe integers; otherwise stringify via builder? we'll best-effort
                    if (Number.isSafeInteger(v)) return BigInt(v);
                    // best effort: keep as number; builder may coerce or throw depending on Arrow version
                    return v;
                }
                if (typeof v === 'string' && v.trim() !== '') {
                    try { return BigInt(v); } catch { /* fallthrough */ }
                }
                return v;
            };
        }

        if (_arrowTypeIsTimestamp(type) || _arrowTypeIsDate(type)) {
            // Arrow JS may represent timestamps/dates as number, Date, or {seconds/nanoseconds} depending on library/version.
            // Keep Date objects as-is, allow numbers, allow objects (pass through).
            return (v) => {
                if (v === null || v === undefined) return null;
                if (_isDate(v)) return v;
                if (typeof v === 'number' && Number.isFinite(v)) return v;
                return v;
            };
        }

        return (v) => {
            if (v === null || v === undefined) return null;
            if (typeof v === 'number') return Number.isNaN(v) ? null : v;
            if (typeof v === 'bigint') return Number(v);
            if (typeof v === 'string') {
                const n = Number(v);
                return Number.isNaN(n) ? null : n;
            }
            return null;
        };
    }

    // Default: pass through
    return (v) => (v === undefined ? null : v);
}

function _coerceFilterOperand(fieldType, op, value) {
    const normOp = _normalizeOp(op);

    if (normOp === 'IS_NULL' || normOp === 'IS_NOT_NULL') return null;

    const coerce = _makeValueCoercer(fieldType);

    if (normOp === 'IN' || normOp === 'NOT_IN') {
        if (!Array.isArray(value)) throw new Error(`Operator ${normOp} expects an array value`);
        const coerced = value.map((v) => coerce(v));
        return coerced;
    }

    if (normOp === 'BETWEEN') {
        if (!Array.isArray(value) || value.length !== 2) throw new Error('BETWEEN expects [min, max]');
        return [coerce(value[0]), coerce(value[1])];
    }

    if (normOp === 'LIKE' || normOp === 'ILIKE' || normOp === 'MATCH') {
        // Allow RegExp directly; otherwise compile from string.
        if (value instanceof RegExp) return value;
        const flags = normOp === 'ILIKE' ? 'i' : '';
        const pat = normOp === 'MATCH' ? String(value) : _escapeRegex(value);
        return new RegExp(pat, flags);
    }

    if (normOp === 'CONTAINS' || normOp === 'STARTS_WITH' || normOp === 'ENDS_WITH') {
        if (value === null || value === undefined) return null;
        return String(value);
    }

    // EQ/NEQ etc
    return coerce(value);
}

// ------------------------------------------------------------------------------
// Nested Map key trie utilities (no composite string keys)
// ------------------------------------------------------------------------------

function _trieEnsure(map, key) {
    let child = map.get(key);
    if (!child) {
        child = new Map();
        map.set(key, child);
    }
    return child;
}

/**
 * Insert a value into a nested-Map trie at a tuple key.
 * For k-length tuple, the leaf is stored in the last Map with lastKey -> value.
 */
function _trieSet(root, keys, value) {
    if (keys.length === 0) throw new Error('Cannot set trie value with empty key tuple');
    let node = root;
    for (let i = 0; i < keys.length - 1; i++) {
        node = _trieEnsure(node, keys[i]);
    }
    node.set(keys[keys.length - 1], value);
}

/**
 * Get a value from a nested-Map trie at a tuple key.
 */
function _trieGet(root, keys) {
    if (keys.length === 0) return undefined;
    let node = root;
    for (let i = 0; i < keys.length - 1; i++) {
        node = node.get(keys[i]);
        if (!node) return undefined;
    }
    return node.get(keys[keys.length - 1]);
}

function _trieGetFromVectors(root, vectors, rowIndex) {
    const k = vectors.length;
    if (k === 0) return undefined;
    let node = root;
    for (let i = 0; i < k - 1; i++) {
        const v = vectors[i].get(rowIndex);
        node = node.get(v);
        if (!node) return undefined;
    }
    const last = vectors[k - 1].get(rowIndex);
    return node.get(last);
}

function _trieHasFromVectors(root, vectors, rowIndex) {
    return _trieGetFromVectors(root, vectors, rowIndex) !== undefined;
}

// ------------------------------------------------------------------------------
// Arrow <-> objects ingestion helpers (array-of-objects -> Table)
// ------------------------------------------------------------------------------

function _arrayToArrowDict(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return new arrow.Table({});
    }

    const first = rows[0];
    if (!_isObj(first)) {
        throw new Error('Array source must be an array of objects');
    }

    // Determine column names from union of keys seen (backwards compatible with sparse rows)
    const colSet = new Set();
    for (const r of rows) {
        if (_isObj(r)) {
            for (const k of Object.keys(r)) colSet.add(k);
        }
    }
    const colNames = Array.from(colSet);

    // First pass: stats
    const stats = new Map();
    for (const name of colNames) {
        stats.set(name, {
            hasNull: false,
            hasString: false,
            hasNumber: false,
            hasBoolean: false,
            hasDate: false
        });
    }

    for (const r of rows) {
        if (!_isObj(r)) continue;
        for (const name of colNames) {
            const v = r[name];
            const st = stats.get(name);
            if (v === null || v === undefined) {
                st.hasNull = true;
                continue;
            }
            if (_isDate(v)) {
                st.hasDate = true;
                continue;
            }
            const t = typeof v;
            if (t === 'string') st.hasString = true;
            else if (t === 'number') {
                if (Number.isNaN(v)) st.hasString = true;
                else st.hasNumber = true;
            } else if (t === 'boolean') st.hasBoolean = true;
            else if (t === 'bigint') st.hasString = true;
            else if (t === 'object') st.hasString = true;
            else st.hasString = true;
        }
    }

    // Choose Arrow types using legacy rules
    const fields = [];
    const builders = new Map();

    for (const name of colNames) {
        const st = stats.get(name);
        let type;
        if (st.hasDate && !st.hasString && !st.hasNumber) {
            type = new arrow.DateMillisecond();
        } else if (st.hasNumber && !st.hasString && !st.hasBoolean) {
            type = new arrow.Float64();
        } else if (st.hasBoolean && !st.hasString && !st.hasNumber) {
            type = new arrow.Bool();
        } else {
            type = new arrow.Utf8();
        }

        fields.push(new arrow.Field(name, type, true));
        const b = arrow.makeBuilder({ type, nullValues: NULL_VALUES });
        builders.set(name, b);
    }

    // Second pass: append values (coerce where needed)
    for (const r of rows) {
        const rowObj = _isObj(r) ? r : {};
        for (const name of colNames) {
            const b = builders.get(name);
            const field = fields.find((f) => f.name === name);
            const type = field.type;

            let v = rowObj[name];

            if (v === undefined) v = null;

            if (v !== null && v !== undefined) {
                if (type.typeId === arrow.Type.Utf8) {
                    // Preserve old String() conversion behavior for mixed columns
                    if (typeof v === 'bigint') v = v.toString();
                    else if (typeof v === 'object' && !_isDate(v)) {
                        try { v = JSON.stringify(v); } catch { v = String(v); }
                    } else v = String(v);
                } else if (type.typeId === arrow.Type.Float) {
                    if (typeof v !== 'number') {
                        const n = Number(v);
                        v = Number.isNaN(n) ? null : n;
                    } else if (Number.isNaN(v)) {
                        v = null;
                    }
                } else if (type.typeId === arrow.Type.Bool) {
                    v = Boolean(v);
                } else if (type.typeId === arrow.Type.Date) {
                    // allow Date or numeric ms; leave as-is and let Arrow builder coerce if needed
                }
            }

            b.append(v);
        }
    }

    const children = {};
    for (const f of fields) {
        const b = builders.get(f.name);
        children[f.name] = b.finish().toVector();
        if (typeof b.clear === 'function') b.clear();
    }

    const schema = new arrow.Schema(fields);
    return _makeTableFromVectors(children, schema);
}

// ------------------------------------------------------------------------------
// Columnar schema decoding helpers
// ------------------------------------------------------------------------------

function getArrowType(typeNameOrId) {
    // Best-effort mapping for serialized schema objects.
    // If already a DataType instance, return it.
    if (typeNameOrId && typeof typeNameOrId === 'object' && typeof typeNameOrId.typeId === 'number') {
        return typeNameOrId;
    }
    const t = String(typeNameOrId ?? '').toLowerCase();

    // common strings
    if (t.includes('utf') || t === 'string') return new arrow.Utf8();
    if (t === 'bool' || t === 'boolean') return new arrow.Bool();
    if (t === 'float64' || t === 'double' || t === 'number') return new arrow.Float64();
    if (t === 'float32') return new arrow.Float32();
    if (t === 'int32') return new arrow.Int32();
    if (t === 'int16') return new arrow.Int16();
    if (t === 'int8') return new arrow.Int8();
    if (t === 'uint32') return new arrow.Uint32();
    if (t === 'uint16') return new arrow.Uint16();
    if (t === 'uint8') return new arrow.Uint8();
    if (t === 'int64') return new arrow.Int64();
    if (t === 'date' || t === 'datemillisecond') return new arrow.DateMillisecond();
    if (t === 'dateday') return new arrow.DateDay();

    // numeric ids (Arrow.Type enum)
    const asNum = Number(typeNameOrId);
    if (Number.isInteger(asNum)) {
        // For many types, we can't reconstruct without more metadata. Default to Utf8.
        return new arrow.Utf8();
    }

    return new arrow.Utf8();
}

function _arrowFromObjects(data) {
    // Expected format:
    // { schema: { fields: [{name, type, nullable, metadata?}], metadata? }, columns: { [name]: values } }
    if (!data || !_isObj(data) || !data.schema || !data.columns) {
        throw new Error('Invalid columnar object format: missing schema/columns');
    }

    const schemaObj = data.schema;
    const colsObj = data.columns;

    const schemaMeta = schemaObj.metadata && _isObj(schemaObj.metadata)
        ? new Map(Object.entries(schemaObj.metadata).map(([k, v]) => [String(k), String(v)]))
        : undefined;

    const fields = [];
    for (const f of schemaObj.fields || []) {
        const name = f.name;
        const type = getArrowType(f.type);
        const nullable = Boolean(f.nullable ?? true);
        const fieldMeta = f.metadata && _isObj(f.metadata)
            ? new Map(Object.entries(f.metadata).map(([k, v]) => [String(k), String(v)]))
            : undefined;
        fields.push(new arrow.Field(name, type, nullable, fieldMeta));
    }

    const schema = new arrow.Schema(fields, schemaMeta);

    const children = {};
    for (const field of fields) {
        const colData = colsObj[field.name] ?? [];
        // Use vectorFromArray with explicit type so arrays are coerced correctly
        children[field.name] = arrow.vectorFromArray(colData, field.type);
    }

    return _makeTableFromVectors(children, schema);
}

// ------------------------------------------------------------------------------
// Mask application (row selection) with take + robust fallback
// ------------------------------------------------------------------------------

function _maskCountOnes(mask) {
    let c = 0;
    for (let i = 0; i < mask.length; i++) c += mask[i] ? 1 : 0;
    return c;
}

function _maskToIndices(mask, count) {
    const out = new Uint32Array(count);
    let p = 0;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) out[p++] = i;
    }
    return out;
}

function _maskToRuns(mask) {
    // Returns array of [start, end) where mask is 1 contiguously
    const runs = [];
    let i = 0;
    const n = mask.length;
    while (i < n) {
        while (i < n && !mask[i]) i++;
        if (i >= n) break;
        const start = i;
        while (i < n && mask[i]) i++;
        runs.push([start, i]);
    }
    return runs;
}

function _tableTakeFallbackByRuns(tbl, mask) {
    // Build by concatenating slices over contiguous runs
    const runs = _maskToRuns(mask);
    if (runs.length === 0) return tbl.slice(0, 0);

    // If runs are excessive, build by per-column builders instead (more predictable)
    if (runs.length > TAKE_SLICE_RUN_LIMIT) {
        return _tableTakeFallbackByBuilders(tbl, mask);
    }

    let out = null;
    for (const [s, e] of runs) {
        const part = tbl.slice(s, e);
        out = out ? out.concat(part) : part;
    }
    return out || tbl.slice(0, 0);
}

function _tableTakeFallbackByBuilders(tbl, mask) {
    const count = _maskCountOnes(mask);
    const indices = _maskToIndices(mask, count);

    const schema = tbl.schema;
    const children = {};
    for (const field of schema.fields) {
        const vec = _getChild(tbl, field.name);
        const b = arrow.makeBuilder({ type: field.type, nullValues: NULL_VALUES });
        for (let j = 0; j < indices.length; j++) {
            b.append(vec.get(indices[j]));
        }
        children[field.name] = b.finish().toVector();
        if (typeof b.clear === 'function') b.clear();
    }
    return _makeTableFromVectors(children, schema);
}

function _applyMask(tbl, mask) {
    const n = tbl.numRows;
    if (mask.length !== n) throw new Error(`Mask length ${mask.length} != table rows ${n}`);

    const count = _maskCountOnes(mask);
    if (count === 0) {
        return tbl.slice(0, 0);
    }
    if (count === n) {
        return tbl;
    }

    // Preferred: use Table.take(indicesVector) if available
    const indices = _maskToIndices(mask, count);
    const indicesVector = arrow.vectorFromArray(indices, new arrow.Uint32());

    if (typeof tbl.take === 'function') {
        return tbl.take(indicesVector);
    }

    // Robust fallback for Arrow versions without `take`
    return _tableTakeFallbackByRuns(tbl, mask);
}

// ------------------------------------------------------------------------------
// Vectorized filter engine (mask builder)
// ------------------------------------------------------------------------------

function _compileFilterSpec(spec, schema) {
    const fieldMap = _schemaFieldMap(schema);

    function compileNode(node) {
        if (node === null || node === undefined) {
            return { kind: 'true' };
        }
        if (_isFn(node)) {
            return { kind: 'fn', fn: node };
        }
        if (Array.isArray(node)) {
            return { kind: 'and', parts: node.map(compileNode) };
        }
        if (!_isObj(node)) {
            throw new Error(`Invalid filter spec node: ${String(node)}`);
        }

        // New-style leaf: { field, op, value }
        if (_hasOwn(node, 'field') && _hasOwn(node, 'op')) {
            const field = node.field;
            const op = _normalizeOp(node.op);
            if (!fieldMap.has(field)) {
                throw new Error(`Unknown field '${field}' in filter spec`);
            }
            const fieldType = fieldMap.get(field).type;
            const value = _hasOwn(node, 'value') ? node.value : undefined;
            const coerced = _coerceFilterOperand(fieldType, op, value);

            const leaf = {
                kind: 'leaf',
                field,
                op,
                value: coerced,
                _raw: node
            };

            // Precompute IN lookups
            if (op === 'IN' || op === 'NOT_IN') {
                if (!Array.isArray(coerced)) throw new Error(`${op} expects array`);
                leaf.hasNullInList = coerced.some((v) => v === null);
                if (coerced.length > IN_SET_LINEAR_LIMIT) leaf.valueSet = new Set(coerced);
                else leaf.valueList = coerced.slice();
            }

            return leaf;
        }

        // Logical nodes: { AND: [...] }, { OR: [...] }, { NOT: spec }
        const keys = Object.keys(node);
        if (keys.length === 1) {
            const k = keys[0].toUpperCase();
            if (k === 'AND' || k === 'OR') {
                const parts = node[keys[0]];
                if (!Array.isArray(parts)) throw new Error(`${k} expects an array`);
                return { kind: k.toLowerCase(), parts: parts.map(compileNode) };
            }
            if (k === 'NOT') {
                return { kind: 'not', part: compileNode(node[keys[0]]) };
            }

            // Legacy leaf style: { colName: { OP: value } }
            const col = keys[0];
            if (!fieldMap.has(col)) {
                throw new Error(`Unknown field '${col}' in legacy filter spec`);
            }
            const leafObj = node[col];
            if (!_isObj(leafObj)) {
                throw new Error(`Legacy leaf for '${col}' must be an object`);
            }
            const ops = Object.keys(leafObj);
            if (ops.length !== 1) throw new Error(`Legacy leaf for '${col}' must have exactly one operator`);
            const op = _normalizeOp(ops[0]);
            const fieldType = fieldMap.get(col).type;
            const coerced = _coerceFilterOperand(fieldType, op, leafObj[ops[0]]);

            const leaf = {
                kind: 'leaf',
                field: col,
                op,
                value: coerced,
                _raw: node
            };
            if (op === 'IN' || op === 'NOT_IN') {
                if (!Array.isArray(coerced)) throw new Error(`${op} expects array`);
                leaf.hasNullInList = coerced.some((v) => v === null);
                if (coerced.length > IN_SET_LINEAR_LIMIT) leaf.valueSet = new Set(coerced);
                else leaf.valueList = coerced.slice();
            }
            return leaf;
        }

        throw new Error(`Invalid filter spec object shape: ${JSON.stringify(node)}`);
    }

    return compileNode(spec);
}

function _getAccessor(tbl, field, cache) {
    let acc = cache.get(field);
    if (acc) return acc;

    const vec = _getChild(tbl, field);
    if (!vec) throw new Error(`Column '${field}' not found`);

    const type = vec.type;
    const typeId = type?.typeId;

    acc = {
        field,
        vec,
        type,
        typeId,
        data: vec.data || [],
        // dictionary helpers
        isDict: typeId === arrow.Type.Dictionary,
        dict: null,
        dictTypeId: null,
        indicesVec: null
    };

    if (acc.isDict) {
        acc.dict = vec.dictionary;
        acc.dictTypeId = acc.dict?.type?.typeId;
        acc.indicesVec = vec.indices || null;
    }

    cache.set(field, acc);
    return acc;
}

function _evalLeafToNewMask(tbl, leaf, accessorCache) {
    const n = tbl.numRows;
    const mask = new Uint8Array(n);
    const acc = _getAccessor(tbl, leaf.field, accessorCache);
    _applyLeafIntoMask(acc, leaf, mask, 'OR'); // OR into empty mask => compute fresh
    return mask;
}

function _andMasksInPlace(a, b) {
    for (let i = 0; i < a.length; i++) a[i] = a[i] & b[i];
    return a;
}
function _orMasksInPlace(a, b) {
    for (let i = 0; i < a.length; i++) a[i] = a[i] | b[i];
    return a;
}
function _invertMaskInPlace(m) {
    for (let i = 0; i < m.length; i++) m[i] = m[i] ? 0 : 1;
    return m;
}

function _evalCompiledSpecToMask(tbl, compiled, accessorCache) {
    if (!compiled || compiled.kind === 'true') {
        const m = new Uint8Array(tbl.numRows);
        m.fill(1);
        return m;
    }

    if (compiled.kind === 'fn') {
        // Build mask from predicate function (still avoids full table materialization)
        const m = new Uint8Array(tbl.numRows);
        for (let i = 0; i < tbl.numRows; i++) {
            const row = typeof tbl.get === 'function' ? tbl.get(i) : null;
            m[i] = compiled.fn(row ?? {}) ? 1 : 0;
        }
        return m;
    }

    if (compiled.kind === 'leaf') {
        return _evalLeafToNewMask(tbl, compiled, accessorCache);
    }

    if (compiled.kind === 'and') {
        const parts = compiled.parts || [];
        if (parts.length === 0) {
            const m = new Uint8Array(tbl.numRows);
            m.fill(1);
            return m;
        }
        let m = _evalCompiledSpecToMask(tbl, parts[0], accessorCache);
        for (let i = 1; i < parts.length; i++) {
            const next = _evalCompiledSpecToMask(tbl, parts[i], accessorCache);
            m = _andMasksInPlace(m, next);
        }
        return m;
    }

    if (compiled.kind === 'or') {
        const parts = compiled.parts || [];
        if (parts.length === 0) {
            return new Uint8Array(tbl.numRows); // all zeros
        }
        let m = _evalCompiledSpecToMask(tbl, parts[0], accessorCache);
        for (let i = 1; i < parts.length; i++) {
            const next = _evalCompiledSpecToMask(tbl, parts[i], accessorCache);
            m = _orMasksInPlace(m, next);
        }
        return m;
    }

    if (compiled.kind === 'not') {
        const inner = _evalCompiledSpecToMask(tbl, compiled.part, accessorCache);
        return _invertMaskInPlace(inner);
    }

    throw new Error(`Unknown compiled filter node kind: ${compiled.kind}`);
}

function _applyLeafIntoMask(acc, leaf, mask, mode /* 'AND'|'OR' */) {
    const vec = acc.vec;
    const type = acc.type;
    const op = leaf.op;

    const n = mask.length;

    // Null ops can be done via null bitmap checks for most vectors
    if (op === 'IS_NULL' || op === 'IS_NOT_NULL') {
        // Use chunk iteration to read nullBitmap
        let row = 0;
        for (const data of acc.data) {
            const len = data.length ?? 0;
            const offset = data.offset ?? 0;
            const nullBitmap = data.nullBitmap;
            const hasBitmap = !!nullBitmap;

            for (let j = 0; j < len; j++, row++) {
                if (mode === 'AND' && !mask[row]) continue;
                if (mode === 'OR' && mask[row]) continue;

                let isValid = true;
                if (hasBitmap) {
                    isValid = _bitIsSet(nullBitmap, offset + j);
                } else {
                    // If no bitmap, assume valid
                    isValid = true;
                }

                const matches = op === 'IS_NULL' ? !isValid : isValid;
                if (mode === 'AND') mask[row] = matches ? 1 : 0;
                else if (mode === 'OR') mask[row] = matches ? 1 : 0;
            }
        }
        return;
    }

    // Dictionary-encoded UTF8: evaluate against dictionary once
    if (acc.isDict && acc.dict && acc.dictTypeId === arrow.Type.Utf8) {
        _applyLeafIntoMask_DictUtf8(acc, leaf, mask, mode);
        return;
    }

    // Fast path: numeric primitives (excluding tricky timestamp/int64 encodings)
    if (_arrowTypeIsNumeric(type) && !_arrowTypeIsTimestamp(type) && !_arrowTypeIsDate(type)) {
        // If int64 uses BigInt64Array in this Arrow build, we can still do a fast path
        if (_arrowTypeIsInt64(type)) {
            _applyLeafIntoMask_Int64(acc, leaf, mask, mode);
            return;
        }
        _applyLeafIntoMask_Numeric(acc, leaf, mask, mode);
        return;
    }

    // Fast-ish path: bool bitpacked
    if (_arrowTypeIsBool(type)) {
        _applyLeafIntoMask_Bool(acc, leaf, mask, mode);
        return;
    }

    // Fallback: row-wise get() with proper null handling
    for (let i = 0; i < n; i++) {
        if (mode === 'AND' && !mask[i]) continue;
        if (mode === 'OR' && mask[i]) continue;

        const v = vec.get(i);
        const matches = _evalScalarOp(v, leaf);
        mask[i] = matches ? 1 : 0;
    }
}

function _evalScalarOp(v, leaf) {
    const op = leaf.op;
    const val = leaf.value;

    if (op === 'EQ') {
        if (val === null) return v === null;
        return v === val;
    }
    if (op === 'NEQ') {
        if (val === null) return v !== null;
        return v !== val;
    }

    // Treat null as non-matching for comparisons
    if (v === null || v === undefined) return false;

    if (op === 'LT') return v < val;
    if (op === 'LTE') return v <= val;
    if (op === 'GT') return v > val;
    if (op === 'GTE') return v >= val;

    if (op === 'BETWEEN') {
        const a = val[0];
        const b = val[1];
        return v >= a && v <= b;
    }

    if (op === 'IN') {
        if (leaf.valueSet) return leaf.valueSet.has(v);
        if (leaf.valueList) {
            for (let i = 0; i < leaf.valueList.length; i++) if (leaf.valueList[i] === v) return true;
            return false;
        }
        return Array.isArray(val) ? val.includes(v) : false;
    }

    if (op === 'NOT_IN') {
        if (leaf.valueSet) return !leaf.valueSet.has(v);
        if (leaf.valueList) {
            for (let i = 0; i < leaf.valueList.length; i++) if (leaf.valueList[i] === v) return false;
            return true;
        }
        return Array.isArray(val) ? !val.includes(v) : true;
    }

    if (op === 'LIKE' || op === 'ILIKE' || op === 'MATCH') {
        const pat = (val instanceof RegExp) ? val : new RegExp(op === 'MATCH' ? String(val) : _escapeRegex(val), op === 'ILIKE' ? 'i' : '');
        return pat.test(String(v));
    }

    if (op === 'CONTAINS') return String(v).includes(String(val));
    if (op === 'STARTS_WITH') return String(v).startsWith(String(val));
    if (op === 'ENDS_WITH') return String(v).endsWith(String(val));

    // Unknown op => false
    return false;
}

function _applyLeafIntoMask_Numeric(acc, leaf, mask, mode) {
    const op = leaf.op;

    // Handle EQ/NEQ null as validity checks
    if ((op === 'EQ' || op === 'NEQ') && leaf.value === null) {
        // EQ null => IS_NULL, NEQ null => IS_NOT_NULL
        const fake = { ...leaf, op: op === 'EQ' ? 'IS_NULL' : 'IS_NOT_NULL' };
        _applyLeafIntoMask(acc, fake, mask, mode);
        return;
    }

    // Pre-extract constants
    const v0 = leaf.value;
    const betweenA = op === 'BETWEEN' ? leaf.value[0] : null;
    const betweenB = op === 'BETWEEN' ? leaf.value[1] : null;

    let row = 0;
    for (const data of acc.data) {
        const len = data.length ?? 0;
        const offset = data.offset ?? 0;
        const values = data.values;
        const nullBitmap = data.nullBitmap;
        const hasBitmap = !!nullBitmap;

        for (let j = 0; j < len; j++, row++) {
            if (mode === 'AND' && !mask[row]) continue;
            if (mode === 'OR' && mask[row]) continue;

            const valid = hasBitmap ? _bitIsSet(nullBitmap, offset + j) : true;
            if (!valid) {
                // null does not match numeric comparisons or IN unless list contains null
                let matches = false;
                if (op === 'IN') matches = Boolean(leaf.hasNullInList);
                else if (op === 'NOT_IN') matches = !leaf.hasNullInList;
                mask[row] = matches ? 1 : 0;
                continue;
            }

            const x = values[offset + j];

            let matches = false;
            switch (op) {
                case 'EQ': matches = x === v0; break;
                case 'NEQ': matches = x !== v0; break;
                case 'LT': matches = x < v0; break;
                case 'LTE': matches = x <= v0; break;
                case 'GT': matches = x > v0; break;
                case 'GTE': matches = x >= v0; break;
                case 'BETWEEN': matches = x >= betweenA && x <= betweenB; break;
                case 'IN':
                    if (leaf.valueSet) matches = leaf.valueSet.has(x);
                    else if (leaf.valueList) {
                        matches = false;
                        for (let k = 0; k < leaf.valueList.length; k++) { if (leaf.valueList[k] === x) { matches = true; break; } }
                    } else matches = Array.isArray(v0) ? v0.includes(x) : false;
                    break;
                case 'NOT_IN':
                    if (leaf.valueSet) matches = !leaf.valueSet.has(x);
                    else if (leaf.valueList) {
                        matches = true;
                        for (let k = 0; k < leaf.valueList.length; k++) { if (leaf.valueList[k] === x) { matches = false; break; } }
                    } else matches = Array.isArray(v0) ? !v0.includes(x) : true;
                    break;
                default:
                    // unsupported numeric op => fallback scalar
                    matches = _evalScalarOp(x, leaf);
                    break;
            }

            mask[row] = matches ? 1 : 0;
        }
    }
}

function _applyLeafIntoMask_Int64(acc, leaf, mask, mode) {
    // Int64/Timestamp may be represented as BigInt64Array (or as pairs in Int32Array in some stacks).
    // If values are BigInt64Array we can do a fast comparison; otherwise fall back to vec.get.
    const op = leaf.op;

    // Null comparisons
    if ((op === 'EQ' || op === 'NEQ') && leaf.value === null) {
        const fake = { ...leaf, op: op === 'EQ' ? 'IS_NULL' : 'IS_NOT_NULL' };
        _applyLeafIntoMask(acc, fake, mask, mode);
        return;
    }

    // If user provided number for Int64 filter value, try to coerce to BigInt for stable compare
    let v0 = leaf.value;
    if (typeof v0 === 'number' && Number.isSafeInteger(v0)) v0 = BigInt(v0);

    // BETWEEN
    let betweenA = null;
    let betweenB = null;
    if (op === 'BETWEEN') {
        betweenA = leaf.value[0];
        betweenB = leaf.value[1];
        if (typeof betweenA === 'number' && Number.isSafeInteger(betweenA)) betweenA = BigInt(betweenA);
        if (typeof betweenB === 'number' && Number.isSafeInteger(betweenB)) betweenB = BigInt(betweenB);
    }

    // IN lists: build set of BigInt when possible
    let set = leaf.valueSet;
    let list = leaf.valueList;
    let hasNull = leaf.hasNullInList;

    const canUseBigIntSet = (op === 'IN' || op === 'NOT_IN') && (set || list);

    // Try fast path if underlying values are BigInt64Array
    let row = 0;
    for (const data of acc.data) {
        const len = data.length ?? 0;
        const offset = data.offset ?? 0;
        const values = data.values;
        const nullBitmap = data.nullBitmap;
        const hasBitmap = !!nullBitmap;

        const isBigIntArray =
            typeof BigInt64Array !== 'undefined' &&
            values instanceof BigInt64Array;

        if (!isBigIntArray) {
            // fallback for this chunk: scalar
            for (let j = 0; j < len; j++, row++) {
                if (mode === 'AND' && !mask[row]) continue;
                if (mode === 'OR' && mask[row]) continue;

                const v = acc.vec.get(row);
                const matches = _evalScalarOp(v, leaf);
                mask[row] = matches ? 1 : 0;
            }
            continue;
        }

        for (let j = 0; j < len; j++, row++) {
            if (mode === 'AND' && !mask[row]) continue;
            if (mode === 'OR' && mask[row]) continue;

            const valid = hasBitmap ? _bitIsSet(nullBitmap, offset + j) : true;
            if (!valid) {
                let matches = false;
                if (op === 'IN') matches = Boolean(hasNull);
                else if (op === 'NOT_IN') matches = !hasNull;
                mask[row] = matches ? 1 : 0;
                continue;
            }

            const x = values[offset + j]; // BigInt

            let matches = false;
            switch (op) {
                case 'EQ': matches = x === v0; break;
                case 'NEQ': matches = x !== v0; break;
                case 'LT': matches = x < v0; break;
                case 'LTE': matches = x <= v0; break;
                case 'GT': matches = x > v0; break;
                case 'GTE': matches = x >= v0; break;
                case 'BETWEEN': matches = x >= betweenA && x <= betweenB; break;
                case 'IN': {
                    if (set) matches = set.has(x);
                    else if (list) {
                        matches = false;
                        for (let k = 0; k < list.length; k++) { if (list[k] === x) { matches = true; break; } }
                    } else matches = false;
                    break;
                }
                case 'NOT_IN': {
                    if (set) matches = !set.has(x);
                    else if (list) {
                        matches = true;
                        for (let k = 0; k < list.length; k++) { if (list[k] === x) { matches = false; break; } }
                    } else matches = true;
                    break;
                }
                default:
                    matches = _evalScalarOp(x, leaf);
                    break;
            }

            mask[row] = matches ? 1 : 0;
        }
    }

    // If IN values were numbers, our set/list may not match BigInt values; for safety,
    // fallback scalar in that case (rare)
    if (canUseBigIntSet && (set || list)) {
        // ok
    }
}

function _applyLeafIntoMask_Bool(acc, leaf, mask, mode) {
    const op = leaf.op;

    // Null comparisons
    if ((op === 'EQ' || op === 'NEQ') && leaf.value === null) {
        const fake = { ...leaf, op: op === 'EQ' ? 'IS_NULL' : 'IS_NOT_NULL' };
        _applyLeafIntoMask(acc, fake, mask, mode);
        return;
    }

    const target = Boolean(leaf.value);

    let row = 0;
    for (const data of acc.data) {
        const len = data.length ?? 0;
        const offset = data.offset ?? 0;
        const values = data.values; // bitpacked Uint8Array
        const nullBitmap = data.nullBitmap;
        const hasBitmap = !!nullBitmap;

        for (let j = 0; j < len; j++, row++) {
            if (mode === 'AND' && !mask[row]) continue;
            if (mode === 'OR' && mask[row]) continue;

            const valid = hasBitmap ? _bitIsSet(nullBitmap, offset + j) : true;
            if (!valid) {
                mask[row] = 0;
                continue;
            }

            const bitIndex = offset + j;
            const b = _bitIsSet(values, bitIndex); // true/false
            let matches = false;

            switch (op) {
                case 'EQ': matches = b === target; break;
                case 'NEQ': matches = b !== target; break;
                default:
                    matches = _evalScalarOp(b, leaf);
                    break;
            }

            mask[row] = matches ? 1 : 0;
        }
    }
}

function _applyLeafIntoMask_DictUtf8(acc, leaf, mask, mode) {
    const op = leaf.op;
    const dict = acc.dict;
    const vec = acc.vec;

    // Build (and cache per-leaf per-dict identity) a dictionary match structure
    if (!leaf._dictCache) leaf._dictCache = new WeakMap();
    let cache = leaf._dictCache.get(dict);
    if (!cache) {
        // dictionary values
        const dictSize = dict.length;
        const dictValues = new Array(dictSize);
        for (let i = 0; i < dictSize; i++) dictValues[i] = dict.get(i);

        // value -> codes map (for EQ/IN)
        const codeByValue = new Map();
        for (let i = 0; i < dictSize; i++) {
            // if duplicates, first wins (fine)
            if (!codeByValue.has(dictValues[i])) codeByValue.set(dictValues[i], i);
        }

        cache = { dictSize, dictValues, codeByValue };
        leaf._dictCache.set(dict, cache);
    }

    // Precompute dict matches for pattern-like ops
    let dictMatch = null;
    if (op === 'LIKE' || op === 'ILIKE' || op === 'MATCH') {
        const re = leaf.value instanceof RegExp
            ? leaf.value
            : new RegExp(op === 'MATCH' ? String(leaf.value) : _escapeRegex(leaf.value), op === 'ILIKE' ? 'i' : '');
        dictMatch = new Uint8Array(cache.dictSize);
        for (let i = 0; i < cache.dictSize; i++) {
            dictMatch[i] = re.test(String(cache.dictValues[i])) ? 1 : 0;
        }
    } else if (op === 'CONTAINS' || op === 'STARTS_WITH' || op === 'ENDS_WITH') {
        const needle = String(leaf.value ?? '');
        dictMatch = new Uint8Array(cache.dictSize);
        for (let i = 0; i < cache.dictSize; i++) {
            const s = String(cache.dictValues[i]);
            if (op === 'CONTAINS') dictMatch[i] = s.includes(needle) ? 1 : 0;
            else if (op === 'STARTS_WITH') dictMatch[i] = s.startsWith(needle) ? 1 : 0;
            else dictMatch[i] = s.endsWith(needle) ? 1 : 0;
        }
    }

    // Scan rows - use precomputed dictMatch for pattern ops when available
    const n = mask.length;
    if (dictMatch && acc.indices) {
        // Fast path: use dictionary index lookup instead of per-row vec.get + regex
        const indices = acc.indices;
        for (let i = 0; i < n; i++) {
            if (mode === 'AND' && !mask[i]) continue;
            if (mode === 'OR' && mask[i]) continue;
            const code = indices.get(i);
            mask[i] = (code != null && dictMatch[code]) ? 1 : 0;
        }
    } else {
        // Fallback: scalar evaluation per row
        for (let i = 0; i < n; i++) {
            if (mode === 'AND' && !mask[i]) continue;
            if (mode === 'OR' && mask[i]) continue;
            const v = vec.get(i);
            mask[i] = _evalScalarOp(v, leaf) ? 1 : 0;
        }
    }
}

// Public debug helper (backwards compatible)
function _vectorMask(tbl, spec) {
    const compiled = _compileFilterSpec(spec, tbl.schema);
    const accessors = new Map();
    const mask = _evalCompiledSpecToMask(tbl, compiled, accessors);
    return mask;
}

// Export for debugging in browser (guarded below too)
if (typeof window !== 'undefined') {
    window._vectorMask = _vectorMask;
}

// ------------------------------------------------------------------------------
// Aggregation utilities (streaming accumulators)
// ------------------------------------------------------------------------------

function _normalizeAggFunc(name) {
    const s = String(name ?? '').trim().toLowerCase();
    if (s === 'avg') return 'mean';
    if (s === 'average') return 'mean';
    if (s === 'count()') return 'count';
    return s;
}

function _createAggState(aggFunc) {
    const f = _normalizeAggFunc(aggFunc);

    if (f === 'sum' || f === 'percent_of_total') return { kind: f, sum: 0 };
    if (f === 'count') return { kind: f, count: 0 };
    if (f === 'mean') return { kind: f, sum: 0, count: 0 };
    if (f === 'min') return { kind: f, has: false, v: null };
    if (f === 'max') return { kind: f, has: false, v: null };
    if (f === 'first') return { kind: f, has: false, v: null };
    if (f === 'last') return { kind: f, has: false, v: null };
    if (f === 'std') return { kind: f, count: 0, mean: 0, m2: 0 };
    if (f === 'weighted_avg') return { kind: f, sumW: 0, sumWX: 0 };

    // default: sum
    return { kind: 'sum', sum: 0 };
}

function _aggUpdate(state, x, w = 1) {
    if (x === null || x === undefined) return;

    switch (state.kind) {
        case 'sum':
        case 'percent_of_total': {
            const n = Number(x);
            if (!Number.isFinite(n)) return;
            state.sum += n;
            break;
        }
        case 'count':
            state.count += 1;
            break;
        case 'mean': {
            const n = Number(x);
            if (!Number.isFinite(n)) return;
            state.sum += n;
            state.count += 1;
            break;
        }
        case 'min':
            if (!state.has || x < state.v) { state.v = x; state.has = true; }
            break;
        case 'max':
            if (!state.has || x > state.v) { state.v = x; state.has = true; }
            break;
        case 'first':
            if (!state.has) { state.v = x; state.has = true; }
            break;
        case 'last':
            state.v = x;
            state.has = true;
            break;
        case 'std': {
            const n = Number(x);
            if (!Number.isFinite(n)) return;
            state.count += 1;
            const delta = n - state.mean;
            state.mean += delta / state.count;
            const delta2 = n - state.mean;
            state.m2 += delta * delta2;
            break;
        }
        case 'weighted_avg': {
            const wx = Number(x);
            const ww = Number(w);
            if (!Number.isFinite(wx) || !Number.isFinite(ww)) return;
            state.sumW += ww;
            state.sumWX += ww * wx;
            break;
        }
        default:
            break;
    }
}

function _aggFinalize(state, totalForPercent = null) {
    switch (state.kind) {
        case 'sum': return state.sum;
        case 'count': return state.count;
        case 'mean': return state.count ? state.sum / state.count : null;
        case 'min': return state.has ? state.v : null;
        case 'max': return state.has ? state.v : null;
        case 'first': return state.has ? state.v : null;
        case 'last': return state.has ? state.v : null;
        case 'std': {
            if (state.count < 2) return null;
            const variance = state.m2 / (state.count - 1);
            return Math.sqrt(variance);
        }
        case 'weighted_avg': {
            return state.sumW ? state.sumWX / state.sumW : null;
        }
        case 'percent_of_total': {
            const total = Number(totalForPercent);
            if (!Number.isFinite(total) || total === 0) return 0;
            return (state.sum / total) * 100;
        }
        default:
            return null;
    }
}

// ------------------------------------------------------------------------------
// Pivot implementation (streaming, trie keys)
// ------------------------------------------------------------------------------

function _fastGroupBy(tbl, groupCols, values, aggFunc, fillValue, weightColumn = null) {
    // Simple groupby used when pivot columns are empty.
    const schema = tbl.schema;
    const fieldMap = _schemaFieldMap(schema);

    const groups = [];
    const root = new Map();

    const groupVectors = groupCols.map((c) => _getChild(tbl, c));
    const valueVectors = values.map((c) => _getChild(tbl, c));
    const weightVec = weightColumn ? _getChild(tbl, weightColumn) : null;

    function getOrCreateGroup(rowIndex) {
        if (groupCols.length === 0) {
            if (groups.length === 0) {
                const st = { keyVals: [], aggs: values.map(() => _createAggState(aggFunc)) };
                groups.push(st);
            }
            return groups[0];
        }

        let node = root;
        for (let i = 0; i < groupVectors.length - 1; i++) {
            const kv = groupVectors[i].get(rowIndex);
            node = _trieEnsure(node, kv);
        }
        const lastVal = groupVectors[groupVectors.length - 1].get(rowIndex);
        let st = node.get(lastVal);
        if (!st) {
            const keyVals = groupVectors.map((v) => v.get(rowIndex));
            st = { keyVals, aggs: values.map(() => _createAggState(aggFunc)) };
            node.set(lastVal, st);
            groups.push(st);
        }
        return st;
    }

    // Totals for percent_of_total
    const totals = aggFunc === 'percent_of_total' ? new Float64Array(values.length) : null;

    for (let r = 0; r < tbl.numRows; r++) {
        const g = getOrCreateGroup(r);
        const w = weightVec ? weightVec.get(r) : 1;
        for (let vi = 0; vi < values.length; vi++) {
            const x = valueVectors[vi].get(r);
            _aggUpdate(g.aggs[vi], x, w);
            if (totals) {
                const n = Number(x);
                if (Number.isFinite(n)) totals[vi] += n;
            }
        }
    }

    // Build output columns with builders
    const outFields = [];
    const outChildren = {};

    // group columns
    for (let i = 0; i < groupCols.length; i++) {
        const name = groupCols[i];
        const type = fieldMap.get(name)?.type || new arrow.Utf8();
        outFields.push(new arrow.Field(name, type, true));
        outChildren[name] = arrow.makeBuilder({ type, nullValues: NULL_VALUES });
    }

    // aggregated columns
    for (let vi = 0; vi < values.length; vi++) {
        const outName = values[vi];
        let outType = new arrow.Float64();
        if (_normalizeAggFunc(aggFunc) === 'count') outType = new arrow.Int32();
        // min/max/first/last inherit input type
        if (['min', 'max', 'first', 'last'].includes(_normalizeAggFunc(aggFunc))) {
            outType = fieldMap.get(values[vi])?.type || new arrow.Utf8();
        }
        outFields.push(new arrow.Field(outName, outType, true));
        outChildren[outName] = arrow.makeBuilder({ type: outType, nullValues: NULL_VALUES });
    }

    // Append rows
    for (const g of groups) {
        for (let i = 0; i < groupCols.length; i++) {
            outChildren[groupCols[i]].append(g.keyVals[i]);
        }
        for (let vi = 0; vi < values.length; vi++) {
            const total = totals ? totals[vi] : null;
            const v = _aggFinalize(g.aggs[vi], total);
            outChildren[values[vi]].append(v === undefined ? fillValue : v);
        }
    }

    // Finish builders
    const finalChildren = {};
    for (const f of outFields) {
        const b = outChildren[f.name];
        finalChildren[f.name] = b.finish().toVector();
        if (typeof b.clear === 'function') b.clear();
    }
    const outSchema = new arrow.Schema(outFields);
    return _makeTableFromVectors(finalChildren, outSchema);
}

function _pivotArrowTable(tbl, config, options = {}) {
    const {
        index = [],
        columns = [],
        values = [],
        aggFunc = 'sum',
        fillValue = null,
        weightColumn = null
    } = config;

    const { grandTotal = true, grandTotalGroups = null } = options;

    if (!index.length && !columns.length) {
        throw new Error('Must specify at least index or columns for pivot operation');
    }
    if (!values.length) {
        throw new Error('Must specify at least one value column for pivot operation');
    }

    // If no pivot columns: groupby only
    if (!columns.length) {
        return _fastGroupBy(tbl, index, values, aggFunc, fillValue, weightColumn);
    }

    const schema = tbl.schema;
    const fieldMap = _schemaFieldMap(schema);

    const indexVecs = index.map((c) => _getChild(tbl, c));
    const colVecs = columns.map((c) => _getChild(tbl, c));
    const valueVecs = values.map((c) => _getChild(tbl, c));
    const weightVec = weightColumn ? _getChild(tbl, weightColumn) : null;

    // Pivot key interner: nested maps -> pivotId, and store labels only once per unique pivot tuple
    const pivotRoot = new Map();
    const pivotLabels = [];
    const pivotIdToValues = [];
    let nextPivotId = 0;

    function internPivotId(rowIndex) {
        let node = pivotRoot;
        for (let i = 0; i < colVecs.length - 1; i++) {
            const kv = colVecs[i].get(rowIndex);
            node = _trieEnsure(node, kv);
        }
        const lastVal = colVecs[colVecs.length - 1].get(rowIndex);
        let id = node.get(lastVal);
        if (id === undefined) {
            id = nextPivotId++;
            node.set(lastVal, id);

            // Build label & tuple values once
            const tuple = colVecs.map((v) => v.get(rowIndex));
            pivotIdToValues[id] = tuple;
            pivotLabels[id] = tuple.map((x) => String(x)).join(KEY_DELIM);
        }
        return id;
    }

    // Index groups trie -> group state, preserve insertion order by pushing on create
    const groups = [];
    const groupRoot = new Map();

    function getOrCreateGroup(rowIndex) {
        if (indexVecs.length === 0) {
            if (groups.length === 0) {
                const st = { indexVals: [], pivots: new Map() };
                groups.push(st);
            }
            return groups[0];
        }
        let node = groupRoot;
        for (let i = 0; i < indexVecs.length - 1; i++) {
            const kv = indexVecs[i].get(rowIndex);
            node = _trieEnsure(node, kv);
        }
        const lastVal = indexVecs[indexVecs.length - 1].get(rowIndex);
        let st = node.get(lastVal);
        if (!st) {
            const idxVals = indexVecs.map((v) => v.get(rowIndex));
            st = { indexVals: idxVals, pivots: new Map() };
            node.set(lastVal, st);
            groups.push(st);
        }
        return st;
    }

    // Totals for percent_of_total: per pivotId and per value index
    const wantsPercent = _normalizeAggFunc(aggFunc) === 'percent_of_total';
    const totalsByPivot = wantsPercent ? new Map() : null;

    function ensureTotals(pivotId) {
        if (!totalsByPivot) return null;
        let arr = totalsByPivot.get(pivotId);
        if (!arr) {
            arr = new Float64Array(values.length);
            totalsByPivot.set(pivotId, arr);
        }
        return arr;
    }

    // Grand total groups
    const gtGroups = [];
    const gtRoot = new Map();
    const gtCols = grandTotalGroups
        ? (Array.isArray(grandTotalGroups) ? grandTotalGroups : [grandTotalGroups])
        : [];

    const gtVecs = gtCols.map((c) => _getChild(tbl, c));

    function getOrCreateGrandTotalGroup(rowIndex) {
        if (!grandTotal) return null;
        if (gtVecs.length === 0) {
            if (gtGroups.length === 0) {
                const st = { gtVals: [], pivots: new Map() };
                gtGroups.push(st);
            }
            return gtGroups[0];
        }
        let node = gtRoot;
        for (let i = 0; i < gtVecs.length - 1; i++) {
            const kv = gtVecs[i].get(rowIndex);
            node = _trieEnsure(node, kv);
        }
        const lastVal = gtVecs[gtVecs.length - 1].get(rowIndex);
        let st = node.get(lastVal);
        if (!st) {
            const vals = gtVecs.map((v) => v.get(rowIndex));
            st = { gtVals: vals, pivots: new Map() };
            node.set(lastVal, st);
            gtGroups.push(st);
        }
        return st;
    }

    function getOrCreatePivotAgg(map, pivotId) {
        let cell = map.get(pivotId);
        if (!cell) {
            cell = values.map(() => _createAggState(aggFunc));
            map.set(pivotId, cell);
        }
        return cell;
    }

    // Single pass over input rows
    for (let r = 0; r < tbl.numRows; r++) {
        const pivotId = internPivotId(r);
        const g = getOrCreateGroup(r);
        const cellAggs = getOrCreatePivotAgg(g.pivots, pivotId);

        const w = weightVec ? weightVec.get(r) : 1;

        for (let vi = 0; vi < values.length; vi++) {
            const x = valueVecs[vi].get(r);
            _aggUpdate(cellAggs[vi], x, w);

            if (wantsPercent) {
                const tarr = ensureTotals(pivotId);
                const n = Number(x);
                if (Number.isFinite(n)) tarr[vi] += n;
            }
        }

        if (grandTotal) {
            const gt = getOrCreateGrandTotalGroup(r);
            const gtCell = getOrCreatePivotAgg(gt.pivots, pivotId);
            for (let vi = 0; vi < values.length; vi++) {
                const x = valueVecs[vi].get(r);
                _aggUpdate(gtCell[vi], x, w);
            }
        }
    }

    // Pivot column order: legacy behavior sorts lexicographically by label
    const pivotOrder = Array.from({ length: nextPivotId }, (_, i) => i);
    pivotOrder.sort((a, b) => {
        const sa = pivotLabels[a] ?? '';
        const sb = pivotLabels[b] ?? '';
        return sa < sb ? -1 : sa > sb ? 1 : 0;
    });

    // Output column definitions
    const outFields = [];
    const builders = {};

    // Index columns first
    for (const col of index) {
        const t = fieldMap.get(col)?.type || new arrow.Utf8();
        outFields.push(new arrow.Field(col, t, true));
        builders[col] = arrow.makeBuilder({ type: t, nullValues: NULL_VALUES });
    }

    // Pivoted value columns
    const outValueCols = [];
    for (const pid of pivotOrder) {
        const label = pivotLabels[pid] ?? '';
        for (const vcol of values) {
            const outName = `${vcol}_${label}`;
            outValueCols.push({ outName, pid, vcol });
        }
    }

    for (const item of outValueCols) {
        // Match legacy inference roughly:
        // - sum/mean/std/weighted_avg/percent => Float64
        // - count => Int32
        // - min/max/first/last => input type
        const f = _normalizeAggFunc(aggFunc);
        let t = new arrow.Float64();
        if (f === 'count') t = new arrow.Int32();
        if (['min', 'max', 'first', 'last'].includes(f)) {
            t = fieldMap.get(item.vcol)?.type || new arrow.Utf8();
        }
        outFields.push(new arrow.Field(item.outName, t, true));
        builders[item.outName] = arrow.makeBuilder({ type: t, nullValues: NULL_VALUES });
    }

    // Extra grand total group columns (legacy: appended, null for main rows)
    const extraGtCols = [];
    if (grandTotal && gtCols.length) {
        for (const c of gtCols) {
            if (!index.includes(c)) {
                extraGtCols.push(c);
                const t = fieldMap.get(c)?.type || new arrow.Utf8();
                outFields.push(new arrow.Field(c, t, true));
                builders[c] = arrow.makeBuilder({ type: t, nullValues: NULL_VALUES });
            }
        }
    }

    function appendMainRow(g) {
        // index cols
        for (let i = 0; i < index.length; i++) {
            builders[index[i]].append(g.indexVals[i]);
        }

        // pivot value cols
        for (const item of outValueCols) {
            const pid = item.pid;
            const vIdx = values.indexOf(item.vcol);
            const cell = g.pivots.get(pid);
            if (!cell) {
                builders[item.outName].append(fillValue);
                continue;
            }
            const total = wantsPercent ? (totalsByPivot.get(pid)?.[vIdx] ?? 0) : null;
            const out = _aggFinalize(cell[vIdx], total);
            builders[item.outName].append(out === undefined ? fillValue : out);
        }

        // extra GT cols: null for main rows
        for (const c of extraGtCols) builders[c].append(null);
    }

    function appendGrandRow(gt) {
        // index cols: fill with group values if included, else "Grand Total"
        for (let i = 0; i < index.length; i++) {
            const col = index[i];
            const gi = gtCols.indexOf(col);
            if (gi >= 0) builders[col].append(gt.gtVals[gi]);
            else builders[col].append('Grand Total');
        }

        // pivot values: percent_of_total => 100 when there is data, else fill
        for (const item of outValueCols) {
            const pid = item.pid;
            const vIdx = values.indexOf(item.vcol);
            const cell = gt.pivots.get(pid);
            if (!cell) {
                builders[item.outName].append(fillValue);
                continue;
            }
            if (wantsPercent) {
                // Legacy: grand totals are 100 if any value exists, otherwise fill
                const hasAny = cell[vIdx] && typeof cell[vIdx].sum === 'number' ? cell[vIdx].sum !== 0 : true;
                builders[item.outName].append(hasAny ? 100 : fillValue);
            } else {
                const out = _aggFinalize(cell[vIdx], null);
                builders[item.outName].append(out === undefined ? fillValue : out);
            }
        }

        // extra GT cols: write group values
        for (const c of extraGtCols) {
            const gi = gtCols.indexOf(c);
            builders[c].append(gi >= 0 ? gt.gtVals[gi] : 'Grand Total');
        }
    }

    for (const g of groups) appendMainRow(g);
    if (grandTotal) {
        for (const gt of gtGroups) appendGrandRow(gt);
    }

    // Finish builders
    const finalChildren = {};
    for (const f of outFields) {
        const b = builders[f.name];
        finalChildren[f.name] = b.finish().toVector();
        if (typeof b.clear === 'function') b.clear();
    }

    const outSchema = new arrow.Schema(outFields);
    return _makeTableFromVectors(finalChildren, outSchema);
}

// Multi-agg pivot (kept backwards compatible: uses multiple aggFuncs producing extra columns)
function _pivotTableMultiAgg(tbl, config, options = {}) {
    const {
        index = [],
        columns = [],
        values = [],
        aggFuncs = ['sum', 'mean'],
        fillValue = null,
        sortValues = true,
        weightColumn = null
    } = config;

    // Implement by running single-agg pivot per aggFunc and then concatenating columns.
    // This is not the most efficient, but preserves existing behavior and keeps code size manageable.
    const pivots = aggFuncs.map((fn) =>
        _pivotArrowTable(tbl, { index, columns, values, aggFunc: fn, fillValue, weightColumn }, options)
    );

    // Merge columns: keep index columns from first pivot, then add each pivot's value columns with suffix aggFunc
    const base = pivots[0];
    const outChildren = {};
    const outFields = [];

    // index cols
    for (const col of index) {
        const vec = _getChild(base, col);
        const field = base.schema.fields.find((f) => f.name === col);
        outChildren[col] = vec;
        outFields.push(field);
    }

    for (let ai = 0; ai < aggFuncs.length; ai++) {
        const fn = aggFuncs[ai];
        const t = pivots[ai];
        for (const field of t.schema.fields) {
            if (index.includes(field.name)) continue;
            const outName = `${field.name}__${fn}`;
            outChildren[outName] = _getChild(t, field.name);
            outFields.push(new arrow.Field(outName, field.type, true));
        }
    }

    const outSchema = new arrow.Schema(outFields);
    return _makeTableFromVectors(outChildren, outSchema);
}

// ------------------------------------------------------------------------------
// BigInt recast helper (fixes old broken implementation)
// ------------------------------------------------------------------------------

export function recastBigIntsToInt32(table) {
    if (!table) return table;
    const schema = table.schema;
    if (!schema || !schema.fields || schema.fields.length === 0) return table;

    let changed = false;

    const newFields = [];
    const children = {};

    for (const field of schema.fields) {
        const vec = _getChild(table, field.name);
        if (!vec) continue;

        if (_arrowTypeIsInt64(field.type)) {
            // Rebuild as Int32 (lossy if values exceed 32-bit; caller responsibility)
            changed = true;
            const outType = new arrow.Int32();
            newFields.push(new arrow.Field(field.name, outType, true, field.metadata));

            const b = arrow.makeBuilder({ type: outType, nullValues: NULL_VALUES });
            for (let i = 0; i < table.numRows; i++) {
                const v = vec.get(i);
                if (v === null || v === undefined) b.append(null);
                else b.append(Number(v));
            }
            children[field.name] = b.finish().toVector();
            if (typeof b.clear === 'function') b.clear();
        } else {
            newFields.push(field);
            children[field.name] = vec;
        }
    }

    if (!changed) return table;

    const newSchema = new arrow.Schema(newFields, schema.metadata);
    return _makeTableFromVectors(children, newSchema);
}

// ------------------------------------------------------------------------------
// HyperTable class
// ------------------------------------------------------------------------------

export class HyperTable {
    #baseArrow;         // committed Arrow table
    #plan = [];         // lazy operations queue
    #pk = [];           // primary key columns
    #sk = [];           // secondary key columns (reserved)
    #cached = null;     // materialized cache (same as #baseArrow after evaluate)
    #immutable = false; // immutability mode

    // light caches (cleared on commit)
    #fieldSet = null;           // Set<string>
    #accessorCache = null;      // Map<string, accessor> for current cached table
    #accessorCacheTable = null; // pointer equality guard

    metaData = {};      // parsed schema metadata
    _patchCodec;        // SparsePatchCodec instance
    #columnarCodec;     // OptimizedColumnarCodec instance

    constructor(src = null, options = {}) {
        const { immutable = false } = options || {};
        this.#immutable = Boolean(immutable);

        this._patchCodec = new SparsePatchCodec();
        this.#columnarCodec = new OptimizedColumnarCodec();

        let explicitMetadata = null;

        if (src === null) {
            this.#baseArrow = new arrow.Table({});
        } else if (src instanceof Uint8Array) {
            // Prefer tableFromIPC if available; otherwise RecordBatchReader
            if (typeof arrow.tableFromIPC === 'function') {
                this.#baseArrow = arrow.tableFromIPC(src);
            } else {
                this.#baseArrow = arrow.RecordBatchReader.from(src).readAll().concat();
            }
        } else if (Array.isArray(src)) {
            this.#baseArrow = _arrayToArrowDict(src);
        } else if (src instanceof arrow.Table) {
            this.#baseArrow = src;
        } else if (src?.constructor?.name === '_Table') {
            // Arquero table
            try {
                this.#baseArrow = src.toArrow();
            } catch {
                this.#baseArrow = src;
            }
        } else if (_isObj(src) && 'schema' in src && 'columns' in src) {
            explicitMetadata = src.schema?.metadata ?? null;
            this.#baseArrow = _arrowFromObjects(src);
        } else {
            throw new Error('Unsupported constructor argument type');
        }

        // Parse schema metadata (and optionally merge explicit metadata if Arrow couldn't attach it)
        const schemaMeta = this.#baseArrow?.schema?.metadata;
        this.metaData = this.#parseMetadata(schemaMeta);

        if (explicitMetadata && Object.keys(this.metaData).length === 0) {
            // Use caller-provided metadata (best-effort)
            this.metaData = this.#parseMetadata(new Map(Object.entries(explicitMetadata).map(([k, v]) => [k, String(v)])));
        }

        if (this.metaData?.primary_keys && Array.isArray(this.metaData.primary_keys)) {
            this.#pk = this.metaData.primary_keys.slice();
        }

        // Invalidate caches
        this.#cached = null;
        this.#fieldSet = null;
        this.#accessorCache = null;
        this.#accessorCacheTable = null;
    }

    // ----------------------------------------------------------------------------
    // Metadata and schema (no implicit evaluation)
    // ----------------------------------------------------------------------------

    #parseMetadata(metadataMap) {
        const meta = {};
        if (!metadataMap || typeof metadataMap[Symbol.iterator] !== 'function') return meta;

        try {
            for (const [k, v] of metadataMap) {
                const key = String(k);
                const val = String(v);
                try {
                    meta[key] = JSON.parse(val);
                } catch {
                    meta[key] = val;
                }
            }
        } catch {
            // ignore
        }
        return meta;
    }

    schema() {
        return this.#baseArrow.schema;
    }

    fields() {
        if (this.#fieldSet) return Array.from(this.#fieldSet);

        const schema = this.#baseArrow.schema;
        const names = schema?.fields?.map((f) => f.name) ?? [];
        this.#fieldSet = new Set(names);
        return names;
    }

    hasField(field) {
        if (!this.#fieldSet) this.fields();
        return this.#fieldSet.has(field);
    }

    rowCount() {
        // No implicit evaluation; report committed base
        return this.#baseArrow.numRows;
    }

    get length() {
        return this.rowCount();
    }

    immutable(flag = null) {
        if (flag === null) return this.#immutable;
        this.#immutable = Boolean(flag);
        return this;
    }

    // Explicit cache control (production-friendly)
    dropCache() {
        this.#cached = null;
        this.#fieldSet = null;
        this.#accessorCache = null;
        this.#accessorCacheTable = null;
        return this;
    }

    // Force evaluation/commit; alias for evaluate()
    commit() {
        return this.evaluate();
    }

    // Materialize into a fresh HyperTable; optionally deep-copy via IPC roundtrip
    materialize({ copy = false } = {}) {
        if (!copy) {
            return new HyperTable(this.toArrow()).setPrimaryKey(this.#pk);
        }
        const ipc = this.toIPC();
        const ht = new HyperTable(ipc);
        if (this.#pk.length) ht.setPrimaryKey(this.#pk);
        if (this.#immutable) ht.immutable(true);
        return ht;
    }

    // ----------------------------------------------------------------------------
    // Evaluation & plan management
    // ----------------------------------------------------------------------------

    #clonePlan(plan) {
        // Fuse adjacent compatible ops (like old behavior), preserving order.
        const out = [];
        for (const op of plan) {
            const prev = out[out.length - 1];
            if (!prev) {
                out.push(op);
                continue;
            }
            if (op.type === 'filter' && prev.type === 'filter') {
                prev.vecSpecs.push(...op.vecSpecs);
                prev.predFns.push(...op.predFns);
                continue;
            }
            if (op.type === 'update' && prev.type === 'update') {
                prev.rows.push(...op.rows);
                continue;
            }
            if (op.type === 'insert' && prev.type === 'insert') {
                prev.rows.push(...op.rows);
                continue;
            }
            out.push(op);
        }
        return out;
    }

    #evaluate() {
        if (this.#plan.length === 0) {
            // Cache is just baseArrow; keep cached pointer for backward compatibility
            this.#cached = this.#baseArrow;
            return;
        }

        const originalPlan = this.#plan;
        const ops = this.#clonePlan(this.#plan);

        try {
            let tbl = this.#baseArrow;

            for (const op of ops) {
                if (op.type === 'filter') {
                    tbl = this.#applyFilter(tbl, op);
                } else if (op.type === 'update') {
                    tbl = this.#applyUpdate(tbl, op);
                } else if (op.type === 'insert') {
                    tbl = this.#applyInsert(tbl, op);
                } else {
                    throw new Error(`Unknown operation type: ${op.type}`);
                }
            }

            // Commit evaluated result into baseArrow to fix the old “evaluate resets to base” bug
            this.#baseArrow = tbl;
            this.#cached = tbl;
            this.#plan = [];

            // Invalidate caches tied to old table
            this.dropCache();
            this.#cached = this.#baseArrow; // re-set after dropCache cleared it
        } catch (error) {
            // Restore original plan and clear cache on error
            this.#plan = originalPlan;
            this.#cached = null;

            const errorMsg = `HyperTable evaluation failed: ${error.message}`;
            // eslint-disable-next-line no-console
            console.error(errorMsg, { plan: this.#plan, error });
            throw new Error(errorMsg);
        }
    }

    evaluate() {
        this.#evaluate();
        return this;
    }

    lazy() {
        // No-op; operations are already lazy by default
        return this;
    }

    clear() {
        this.#baseArrow = new arrow.Table({});
        this.#plan = [];
        this.#pk = [];
        this.#cached = null;
        this.metaData = {};
        return this.dropCache();
    }

    clone() {
        const cloned = new HyperTable(this.#baseArrow);
        cloned.#pk = this.#pk.slice();
        cloned.metaData = { ...this.metaData };
        if (this.#immutable) cloned.immutable(true);
        return cloned;
    }

    #forkWith(op) {
        if (this.#immutable) {
            const fork = new HyperTable(this.#baseArrow, { immutable: true });
            fork.#pk = this.#pk.slice();
            fork.#sk = this.#sk.slice();
            fork.metaData = { ...this.metaData };
            fork.#plan = this.#plan.slice();
            fork.#plan.push(op);
            return fork;
        }

        // mutable mode
        this.#plan.push(op);
        this.#cached = null;
        return this;
    }

    // ----------------------------------------------------------------------------
    // Primary keys
    // ----------------------------------------------------------------------------

    setPrimaryKey(keys) {
        const pk = Array.isArray(keys) ? keys : [keys];
        for (const k of pk) {
            if (!this.hasField(k)) {
                throw new Error(`Primary key column '${k}' not found in schema`);
            }
        }
        this.#pk = pk.slice();
        this.metaData = { ...(this.metaData || {}), primary_keys: this.#pk.slice() };

        // Best-effort: store metadata into Arrow schema
        try {
            const schema = this.#baseArrow.schema;
            const md = new Map(schema.metadata || []);
            md.set('primary_keys', JSON.stringify(this.#pk));
            const newSchema = new arrow.Schema(schema.fields, md);
            // Rewrap table with new schema
            const children = {};
            for (const f of schema.fields) children[f.name] = _getChild(this.#baseArrow, f.name);
            this.#baseArrow = _makeTableFromVectors(children, newSchema);
            if (this.#cached) this.#cached = this.#baseArrow;
        } catch {
            // ignore (not all Arrow builds allow schema replacement)
        }

        return this;
    }

    getPrimaryKey() {
        return this.#pk.slice();
    }

    // ----------------------------------------------------------------------------
    // Filter validation & translation (kept for compatibility)
    // ----------------------------------------------------------------------------

    _validateFilterSpec(spec) {
        if (spec === null || spec === undefined) return true;
        if (_isFn(spec)) return true;
        if (typeof spec === 'string') return true;
        if (Array.isArray(spec)) {
            spec.forEach((s) => this._validateFilterSpec(s));
            return true;
        }
        if (!_isObj(spec)) {
            throw new Error('Filter specification must be an object, array, function, or string expression');
        }

        // new-style leaf
        if (_hasOwn(spec, 'field') && _hasOwn(spec, 'op')) {
            if (!this.hasField(spec.field)) throw new Error(`Unknown field '${spec.field}'`);
            return true;
        }

        const keys = Object.keys(spec);
        if (keys.length === 1) {
            const k = keys[0].toUpperCase();
            if (k === 'AND' || k === 'OR') {
                const parts = spec[keys[0]];
                if (!Array.isArray(parts)) throw new Error(`${k} expects an array`);
                parts.forEach((p) => this._validateFilterSpec(p));
                return true;
            }
            if (k === 'NOT') {
                return this._validateFilterSpec(spec[keys[0]]);
            }
            // legacy leaf
            const col = keys[0];
            if (!this.hasField(col)) throw new Error(`Unknown field '${col}'`);
            const leaf = spec[col];
            if (!_isObj(leaf)) throw new Error(`Leaf for '${col}' must be an object`);
            const ops = Object.keys(leaf);
            if (ops.length !== 1) throw new Error(`Leaf for '${col}' must have exactly one operator`);
            return true;
        }

        throw new Error('Invalid filter specification shape');
    }

    #buildArqueroFilterFromExpression(expr) {
        // Sanitize: only allow safe property-access expressions (field comparisons).
        // Reject anything that looks like code injection (semicolons, function calls, assignments).
        if (/[;{}=]|(\b(function|class|import|export|eval|return|var|let|const|new|delete|void|typeof|instanceof)\b)/.test(expr)) {
            throw new Error(`Unsafe filter expression rejected: ${expr}`);
        }
        try {
            // eslint-disable-next-line no-new-func
            return new Function('row', `with(row) { return ${expr}; }`);
        } catch (error) {
            throw new Error(`Invalid filter expression: ${expr}. ${error.message}`);
        }
    }

    #translateAgGridFilter(filterModel) {
        // Backwards compatible translation used by caller
        if (!filterModel || !filterModel.colId) return null;

        const field = filterModel.colId;
        const type = filterModel.type;

        const getValue = (row) => row[field];

        switch (type) {
            case 'equals': return (row) => getValue(row) === filterModel.filter;
            case 'notEqual': return (row) => getValue(row) !== filterModel.filter;
            case 'contains': return (row) => String(getValue(row) ?? '').includes(String(filterModel.filter));
            case 'notContains': return (row) => !String(getValue(row) ?? '').includes(String(filterModel.filter));
            case 'startsWith': return (row) => String(getValue(row) ?? '').startsWith(String(filterModel.filter));
            case 'endsWith': return (row) => String(getValue(row) ?? '').endsWith(String(filterModel.filter));
            case 'lessThan': return (row) => getValue(row) < filterModel.filter;
            case 'lessThanOrEqual': return (row) => getValue(row) <= filterModel.filter;
            case 'greaterThan': return (row) => getValue(row) > filterModel.filter;
            case 'greaterThanOrEqual': return (row) => getValue(row) >= filterModel.filter;
            case 'inRange':
                return (row) => {
                    const v = getValue(row);
                    return v >= filterModel.filter && v <= filterModel.filterTo;
                };
            case 'blank': return (row) => getValue(row) == null;
            case 'notBlank': return (row) => getValue(row) != null;
            default:
                // eslint-disable-next-line no-console
                console.warn(`Unsupported AG Grid filter type: ${type}`);
                return null;
        }
    }

    #convertToArqueroFilter(spec) {
        // Keep this for compatibility. We apply the predicate directly (not via Arquero),
        // so it must work on Arrow row proxies too.
        if (_isFn(spec)) return spec;

        if (typeof spec === 'string') {
            return this.#buildArqueroFilterFromExpression(spec);
        }

        if (Array.isArray(spec)) {
            const parts = spec.map((s) => this.#convertToArqueroFilter(s));
            return (d) => parts.every((p) => p(d));
        }

        if (!_isObj(spec)) {
            throw new Error('Invalid filter specification for Arquero conversion');
        }

        if (_hasOwn(spec, 'field') && _hasOwn(spec, 'op')) {
            // translate leaf
            const field = spec.field;
            const op = _normalizeOp(spec.op);
            const value = spec.value;

            switch (op) {
                case 'EQ': return (d) => d[field] === value;
                case 'NEQ': return (d) => d[field] !== value;
                case 'LT': return (d) => d[field] < value;
                case 'LTE': return (d) => d[field] <= value;
                case 'GT': return (d) => d[field] > value;
                case 'GTE': return (d) => d[field] >= value;
                case 'IN': {
                    const set = new Set(Array.isArray(value) ? value : []);
                    return (d) => set.has(d[field]);
                }
                case 'NOT_IN': {
                    const set = new Set(Array.isArray(value) ? value : []);
                    return (d) => !set.has(d[field]);
                }
                case 'BETWEEN':
                    return (d) => d[field] >= value[0] && d[field] <= value[1];
                case 'LIKE': {
                    const re = value instanceof RegExp ? value : new RegExp(_escapeRegex(value));
                    return (d) => re.test(String(d[field] ?? ''));
                }
                case 'MATCH': {
                    const re = value instanceof RegExp ? value : new RegExp(String(value));
                    return (d) => re.test(String(d[field] ?? ''));
                }
                case 'ILIKE': {
                    const re = value instanceof RegExp ? value : new RegExp(_escapeRegex(value), 'i');
                    return (d) => re.test(String(d[field] ?? ''));
                }
                case 'CONTAINS': return (d) => String(d[field] ?? '').includes(String(value));
                case 'STARTS_WITH': return (d) => String(d[field] ?? '').startsWith(String(value));
                case 'ENDS_WITH': return (d) => String(d[field] ?? '').endsWith(String(value));
                case 'IS_NULL': return (d) => d[field] == null;
                case 'IS_NOT_NULL': return (d) => d[field] != null;
                default:
                    throw new Error(`Unsupported operator: ${op}`);
            }
        }

        const keys = Object.keys(spec);
        if (keys.length !== 1) {
            throw new Error('Invalid filter spec shape for Arquero conversion');
        }

        const k = keys[0].toUpperCase();
        if (k === 'AND' || k === 'OR') {
            const parts = spec[keys[0]].map((s) => this.#convertToArqueroFilter(s));
            return (d) => (k === 'AND' ? parts.every((p) => p(d)) : parts.some((p) => p(d)));
        }
        if (k === 'NOT') {
            const p = this.#convertToArqueroFilter(spec[keys[0]]);
            return (d) => !p(d);
        }

        // legacy leaf
        const col = keys[0];
        const leaf = spec[col];
        const opKey = Object.keys(leaf)[0];
        const op = _normalizeOp(opKey);
        return this.#convertToArqueroFilter({ field: col, op, value: leaf[opKey] });
    }

    // ----------------------------------------------------------------------------
    // Apply operations (filter/update/insert)
    // ----------------------------------------------------------------------------

    #ensureAccessorCacheFor(tbl) {
        if (this.#accessorCacheTable === tbl && this.#accessorCache) return this.#accessorCache;
        this.#accessorCache = new Map();
        this.#accessorCacheTable = tbl;
        return this.#accessorCache;
    }

    #applyFunctionalFilters(tbl, predFns) {
        if (!predFns || predFns.length === 0) return tbl;

        const n = tbl.numRows;
        const mask = new Uint8Array(n);

        for (let i = 0; i < n; i++) {
            const row = typeof tbl.get === 'function' ? tbl.get(i) : null;
            let ok = true;
            for (let j = 0; j < predFns.length; j++) {
                if (!predFns[j](row ?? {})) {
                    ok = false;
                    break;
                }
            }
            mask[i] = ok ? 1 : 0;
        }

        return _applyMask(tbl, mask);
    }

    #applyVectorizedFilters(tbl, vecSpecs) {
        if (!vecSpecs || vecSpecs.length === 0) return tbl;

        const accessors = this.#ensureAccessorCacheFor(tbl);

        let combinedMask = null;
        for (const spec of vecSpecs) {
            const compiled = _compileFilterSpec(spec, tbl.schema);
            const mask = _evalCompiledSpecToMask(tbl, compiled, accessors);
            if (!combinedMask) combinedMask = mask;
            else _andMasksInPlace(combinedMask, mask);
        }

        return combinedMask ? _applyMask(tbl, combinedMask) : tbl;
    }

    #applyFilter(tbl, op) {
        let result = tbl;

        if (op.vecSpecs && op.vecSpecs.length > 0) {
            result = this.#applyVectorizedFilters(result, op.vecSpecs);
        }

        if (op.predFns && op.predFns.length > 0) {
            result = this.#applyFunctionalFilters(result, op.predFns);
        }

        return result;
    }

    #rowsToTable(rows, schema) {
        const fields = schema.fields || [];
        const builders = new Map();

        for (const f of fields) {
            const b = arrow.makeBuilder({ type: f.type, nullValues: NULL_VALUES });
            builders.set(f.name, b);
        }

        // Precompute coercers per field
        const coercers = new Map();
        for (const f of fields) {
            coercers.set(f.name, _makeValueCoercer(f.type));
        }

        for (const r of rows) {
            // Support both plain objects and Arrow StructRow-like values
            for (const f of fields) {
                const name = f.name;
                const b = builders.get(name);
                const coerce = coercers.get(name);

                let v = null;
                try {
                    v = r ? r[name] : null;
                } catch {
                    v = null;
                }
                if (v === undefined) v = null;
                b.append(coerce(v));
            }
        }

        const children = {};
        for (const f of fields) {
            const b = builders.get(f.name);
            children[f.name] = b.finish().toVector();
            if (typeof b.clear === 'function') b.clear();
        }

        return _makeTableFromVectors(children, schema);
    }

    #applyUpdate(tbl, op) {
        const updates = op.rows || [];
        if (updates.length === 0) return tbl;

        const pks = this.#pk;
        if (!pks || pks.length === 0) {
            throw new Error('Primary keys must be set for update operation');
        }

        const schema = tbl.schema;
        const fieldMap = _schemaFieldMap(schema);

        // Build update trie keyed by PK tuple. Update semantics: last write wins (overwrite).
        const root = new Map();
        const patchRows = [];

        function overwriteObject(target, source) {
            for (const k of Object.keys(target)) delete target[k];
            Object.assign(target, source);
        }

        for (const u of updates) {
            if (!_isObj(u)) continue;

            // Ensure all PK values are present
            let ok = true;
            const keyVals = new Array(pks.length);
            for (let i = 0; i < pks.length; i++) {
                const pk = pks[i];
                if (!_hasOwn(u, pk)) { ok = false; break; }
                keyVals[i] = u[pk];
            }
            if (!ok) continue;

            // Insert / overwrite in trie
            let node = root;
            for (let i = 0; i < keyVals.length - 1; i++) node = _trieEnsure(node, keyVals[i]);
            const lastKey = keyVals[keyVals.length - 1];

            let existing = node.get(lastKey);
            if (!existing) {
                existing = { ...u };
                node.set(lastKey, existing);
                patchRows.push(existing);
            } else {
                overwriteObject(existing, u);
            }
        }

        if (patchRows.length === 0) return tbl;

        // Scan base table to find which row indices are updated (O(n), memory O(m))
        const pkVectors = pks.map((pk) => _getChild(tbl, pk));
        const rowIndexToPatch = new Map();

        for (let i = 0; i < tbl.numRows; i++) {
            const patch = _trieGetFromVectors(root, pkVectors, i);
            if (patch) rowIndexToPatch.set(i, patch);
        }

        if (rowIndexToPatch.size === 0) return tbl;

        // Determine updated columns union (excluding unknown columns)
        const updatedCols = new Set();
        for (const patch of patchRows) {
            for (const k of Object.keys(patch)) {
                if (fieldMap.has(k)) updatedCols.add(k);
            }
        }

        // Build new children
        const children = {};
        for (const field of schema.fields) {
            const name = field.name;
            const vec = _getChild(tbl, name);

            if (!updatedCols.has(name)) {
                children[name] = vec;
                continue;
            }

            const coerce = _makeValueCoercer(field.type);

            // Fast patch for single-chunk numeric columns without nulls and no null patches
            const isNumeric = _arrowTypeIsNumeric(field.type) && !_arrowTypeIsInt64(field.type) && !_arrowTypeIsTimestamp(field.type) && !_arrowTypeIsDate(field.type);
            const canTryFast =
                isNumeric &&
                vec &&
                vec.data &&
                vec.data.length === 1 &&
                (vec.nullCount ?? 0) === 0 &&
                vec.data[0] &&
                vec.data[0].offset === 0 &&
                vec.data[0].values &&
                typeof vec.data[0].values.slice === 'function';

            let didFast = false;

            if (canTryFast) {
                const valuesArr = vec.data[0].values;
                const outArr = valuesArr.slice(); // typed array copy
                let ok = true;

                for (const [rowIdx, patch] of rowIndexToPatch.entries()) {
                    if (!_hasOwn(patch, name)) continue;
                    const v = coerce(patch[name]);
                    if (v === null || v === undefined) { ok = false; break; }
                    outArr[rowIdx] = v;
                }

                if (ok) {
                    children[name] = arrow.makeVector(outArr);
                    didFast = true;
                }
            }

            if (didFast) continue;

            // Builder fallback (handles nulls + complex types)
            const b = arrow.makeBuilder({ type: field.type, nullValues: NULL_VALUES });

            for (let i = 0; i < tbl.numRows; i++) {
                const patch = rowIndexToPatch.get(i);
                if (patch && _hasOwn(patch, name)) {
                    b.append(coerce(patch[name]));
                } else {
                    b.append(vec.get(i));
                }
            }

            children[name] = b.finish().toVector();
            if (typeof b.clear === 'function') b.clear();
        }

        // Preserve schema (metadata, field metadata)
        return _makeTableFromVectors(children, schema);
    }

    #applyInsert(tbl, op) {
        const rows = op.rows || [];
        if (rows.length === 0) return tbl;

        const schema = tbl.schema;
        const newRowsTable = this.#rowsToTable(rows, schema);

        // Concatenate tables (Arrow-native)
        return tbl.concat(newRowsTable);
    }

    // ----------------------------------------------------------------------------
    // Public operations
    // ----------------------------------------------------------------------------

    filter(spec, options = {}) {
        const { strategy = 'auto', translateAg = false } = options || {};

        if (spec === null || spec === undefined) {
            return this;
        }

        let useStrategy = String(strategy || 'auto').toLowerCase();

        // AG Grid translation produces predicate functions
        if (translateAg) {
            const agFn = this.#translateAgGridFilter(spec);
            if (!agFn) return this;
            return this.#forkWith({ type: 'filter', vecSpecs: [], predFns: [agFn] });
        }

        // String expression => predicate
        if (typeof spec === 'string') {
            const pred = this.#buildArqueroFilterFromExpression(spec);
            return this.#forkWith({ type: 'filter', vecSpecs: [], predFns: [pred] });
        }

        if (_isFn(spec)) {
            return this.#forkWith({ type: 'filter', vecSpecs: [], predFns: [spec] });
        }

        // Validate spec without implicit evaluation
        this._validateFilterSpec(spec);

        if (useStrategy === 'auto') {
            // Prefer vectorized for object specs; fall back to functional only for functions/expressions (handled above)
            useStrategy = 'vectorized';
        }

        if (useStrategy === 'vectorized') {
            return this.#forkWith({ type: 'filter', vecSpecs: [spec], predFns: [] });
        }

        if (useStrategy === 'arquero') {
            // Backwards compatible: convert to Arquero predicate, but we apply it directly
            const pred = this.#convertToArqueroFilter(spec);
            return this.#forkWith({ type: 'filter', vecSpecs: [], predFns: [pred] });
        }

        // functional
        return this.#forkWith({ type: 'filter', vecSpecs: [], predFns: [spec] });
    }

    filterAgGrid(filterModel) {
        return this.filter(filterModel, { translateAg: true });
    }

    where(spec, options = {}) {
        return this.filter(spec, options);
    }

    quickFilter(query, columns = null) {
        if (query === null || query === undefined || String(query).trim() === '') return this;

        const q = String(query);
        const cols = columns ? (Array.isArray(columns) ? columns : [columns]) : this.fields();

        // OR across columns using CONTAINS (case-insensitive via ILIKE regex)
        const escaped = _escapeRegex(q);
        const spec = {
            OR: cols.map((c) => ({ field: c, op: 'ILIKE', value: escaped }))
        };
        return this.filter(spec, { strategy: 'vectorized' });
    }

    whereMatch(fieldOrQuery, patternOrColumns = null, caseSensitive = false) {
        // Backwards compatible overload behavior:
        // (query, columns, caseSensitive) OR (field, pattern, flags)
        if (typeof patternOrColumns === 'string' || patternOrColumns instanceof RegExp) {
            const field = fieldOrQuery;
            const pattern = patternOrColumns;
            const flags = caseSensitive ? '' : 'i';
            const re = pattern instanceof RegExp ? pattern : new RegExp(String(pattern), flags);
            return this.filter((row) => re.test(String(row[field] ?? '')));
        }

        const query = String(fieldOrQuery ?? '');
        const cols = patternOrColumns
            ? (Array.isArray(patternOrColumns) ? patternOrColumns : [patternOrColumns])
            : this.fields();

        const re = new RegExp(_escapeRegex(query), caseSensitive ? '' : 'i');
        return this.filter((row) => cols.some((c) => re.test(String(row[c] ?? ''))));
    }

    update(rows, options = {}) {
        const updateRows = Array.isArray(rows) ? rows : [rows];
        if (updateRows.length === 0) return this;
        return this.#forkWith({ type: 'update', rows: updateRows });
    }

    insert(rows) {
        const insertRows = Array.isArray(rows) ? rows : [rows];
        if (insertRows.length === 0) return this;
        return this.#forkWith({ type: 'insert', rows: insertRows });
    }

    upsert(rows, primaryKeys = null) {
        return this.batchUpdate(Array.isArray(rows) ? rows : [rows], primaryKeys, { upsert: true });
    }

    delete(criteria, primaryKeys = null) {
        // Backwards compatible behavior: returns a NEW HyperTable (does not mutate this)
        const pks = primaryKeys || this.#pk;
        if (!pks || pks.length === 0) {
            throw new Error('Primary keys required for delete operation');
        }

        const criteriaRows = Array.isArray(criteria) ? criteria : [criteria];
        if (criteriaRows.length === 0) return this;

        // Evaluate current view (commit)
        this.#evaluate();
        const tbl = this.#cached;

        // Build trie of delete keys (tuple -> true)
        const root = new Map();
        for (const c of criteriaRows) {
            if (!_isObj(c)) continue;
            let ok = true;
            const keyVals = new Array(pks.length);
            for (let i = 0; i < pks.length; i++) {
                const pk = pks[i];
                if (!_hasOwn(c, pk)) { ok = false; break; }
                keyVals[i] = c[pk];
            }
            if (!ok) continue;
            _trieSet(root, keyVals, true);
        }

        const pkVecs = pks.map((pk) => _getChild(tbl, pk));
        const n = tbl.numRows;
        const keepMask = new Uint8Array(n);

        for (let i = 0; i < n; i++) {
            keepMask[i] = _trieHasFromVectors(root, pkVecs, i) ? 0 : 1;
        }

        const outTbl = _applyMask(tbl, keepMask);
        const result = new HyperTable(outTbl);
        result.setPrimaryKey(this.#pk);
        if (this.#immutable) result.immutable(true);
        return result;
    }

    batchUpdate(updates, primaryKeys = null, options = {}) {
        const {
            upsert = true,
            validateKeys = true,
            deduplicate = true
        } = options || {};

        if (!Array.isArray(updates) || updates.length === 0) {
            return this.#immutable ? new HyperTable(this.toArrow()) : this;
        }

        const pks = primaryKeys || this.#pk;
        if (!pks || pks.length === 0) {
            throw new Error('Primary keys required for batch update. Set them on the table or provide them as a parameter.');
        }

        // Evaluate current view (commit)
        this.#evaluate();
        let currentTbl = this.#cached;

        // Build update trie keyed by PK. For deduplicate=true, merge; else last wins overwrite.
        const root = new Map();
        const patches = [];

        function overwriteObject(target, source) {
            for (const k of Object.keys(target)) delete target[k];
            Object.assign(target, source);
        }

        for (const update of updates) {
            if (!_isObj(update)) continue;

            let ok = true;
            const keyVals = new Array(pks.length);
            for (let i = 0; i < pks.length; i++) {
                const pk = pks[i];
                if (!_hasOwn(update, pk)) { ok = false; break; }
                keyVals[i] = update[pk];
            }

            if (!ok) {
                if (validateKeys) {
                    throw new Error(`Update object is missing a required primary key. Required: ${pks.join(', ')}`);
                }
                continue;
            }

            // insert/merge in trie
            let node = root;
            for (let i = 0; i < keyVals.length - 1; i++) node = _trieEnsure(node, keyVals[i]);
            const lastKey = keyVals[keyVals.length - 1];

            let existing = node.get(lastKey);
            if (!existing) {
                existing = { ...update };
                node.set(lastKey, existing);
                patches.push(existing);
            } else if (deduplicate) {
                Object.assign(existing, update);
            } else {
                overwriteObject(existing, update);
            }
        }

        if (patches.length === 0) {
            const out = new HyperTable(currentTbl);
            out.setPrimaryKey(this.#pk);
            if (this.#immutable) out.immutable(true);
            return out;
        }

        const updatesToApply = [];
        const insertsToApply = [];

        if (upsert) {
            // Partition patches by scanning base table once (O(n) time, O(m) memory)
            const found = new Set();
            const pkVecs = pks.map((pk) => _getChild(currentTbl, pk));

            for (let i = 0; i < currentTbl.numRows; i++) {
                const patch = _trieGetFromVectors(root, pkVecs, i);
                if (patch) found.add(patch);
            }

            for (const p of patches) {
                if (found.has(p)) updatesToApply.push(p);
                else insertsToApply.push(p);
            }
        } else {
            updatesToApply.push(...patches);
        }

        if (updatesToApply.length > 0) {
            currentTbl = this.#applyUpdate(currentTbl, { type: 'update', rows: updatesToApply });
        }
        if (insertsToApply.length > 0) {
            currentTbl = this.#applyInsert(currentTbl, { type: 'insert', rows: insertsToApply });
        }

        const finalTable = new HyperTable(currentTbl);
        finalTable.setPrimaryKey(this.#pk);
        if (this.#immutable) finalTable.immutable(true);
        return finalTable;
    }

    // ----------------------------------------------------------------------------
    // Serialization
    // ----------------------------------------------------------------------------

    toArrow() {
        this.#evaluate();
        return this.#cached;
    }

    toIPC() {
        this.#evaluate();
        if (typeof this.#cached.serialize === 'function') {
            return this.#cached.serialize();
        }
        if (typeof arrow.tableToIPC === 'function') {
            return arrow.tableToIPC(this.#cached);
        }
        throw new Error('Arrow IPC serialization is not available in this Arrow build');
    }

    toArray() {
        this.#evaluate();
        return this.#cached.toArray();
    }

    toObject() {
        this.#evaluate();
        return this.#cached.toArray().map((row) => (row && typeof row.toJSON === 'function' ? row.toJSON() : row));
    }

    toJSON() {
        return this.toObject();
    }

    toColumnar(includeSchema = true) {
        this.#evaluate();
        const rows = this.toObject();
        return this.#columnarCodec.encodeDataFrame(rows, includeSchema);
    }

    decodeDelta(delta) {
        return this.#columnarCodec.decodeDataFrame(delta);
    }

    encodeDelta(changes, includeFull = false, includeSchema = true) {
        return this.#columnarCodec.encodeDelta(changes, includeFull, includeSchema);
    }

    applyPatch(delta, applyChanges = true) {
        const patch = this.decodeDelta(delta);
        if (!applyChanges) return patch;
        return this.applySparsePatch(patch, true);
    }

    applySparsePatch(delta, applyChanges = true) {
        if (!applyChanges) return delta;
        // Expected delta: { add, update, remove } etc
        return this.bulkOperations(delta);
    }

    // ----------------------------------------------------------------------------
    // Analytics
    // ----------------------------------------------------------------------------

    pivot(config = {}, options = {}) {
        this.#evaluate();

        const {
            index = [],
            columns = [],
            values = [],
            aggFunc = 'sum',
            fillValue = null,
            sortValues = true,
            weightColumn = null
        } = config;

        const {
            grandTotal = true,
            grandTotalGroups = null
        } = options;

        if (!index.length && !columns.length) {
            throw new Error('Must specify at least index or columns for pivot operation');
        }
        if (!values.length) {
            throw new Error('Must specify at least one value column for pivot operation');
        }

        const available = this.#cached.schema.fields.map((f) => f.name);
        const allCols = [...index, ...columns, ...values];
        if (weightColumn) allCols.push(weightColumn);
        for (const c of allCols) {
            if (!available.includes(c)) {
                throw new Error(`Column '${c}' not found in table. Available columns: ${available.join(', ')}`);
            }
        }

        let pivotResult;
        if (Array.isArray(aggFunc)) {
            pivotResult = _pivotTableMultiAgg(this.#cached, {
                index,
                columns,
                values,
                aggFuncs: aggFunc,
                fillValue,
                sortValues,
                weightColumn
            }, options);
        } else {
            pivotResult = _pivotArrowTable(this.#cached, {
                index,
                columns,
                values,
                aggFunc,
                fillValue,
                weightColumn
            }, { grandTotal, grandTotalGroups });
        }

        const result = new HyperTable(pivotResult);

        if (this.#pk.length > 0 && this.#pk.every((pk) => index.includes(pk))) {
            result.setPrimaryKey(this.#pk);
        } else if (index.length > 0) {
            result.setPrimaryKey(index);
        }

        if (this.#immutable) result.immutable(true);
        return result;
    }

    pivotTable(indexCol, pivotCol, valueCol, aggFunc = 'sum', options = {}) {
        const config = {
            index: Array.isArray(indexCol) ? indexCol : [indexCol],
            columns: Array.isArray(pivotCol) ? pivotCol : [pivotCol],
            values: Array.isArray(valueCol) ? valueCol : [valueCol],
            aggFunc
        };
        if (aggFunc === 'weighted_avg' && options.weightColumn) {
            config.weightColumn = options.weightColumn;
        }
        return this.pivot(config, options);
    }

    pivotTableMultiAgg(indexCol, pivotCol, valueCol, aggFuncs = ['sum', 'mean'], options = {}) {
        const config = {
            index: Array.isArray(indexCol) ? indexCol : [indexCol],
            columns: Array.isArray(pivotCol) ? pivotCol : [pivotCol],
            values: Array.isArray(valueCol) ? valueCol : [valueCol],
            aggFunc: aggFuncs
        };
        if (aggFuncs.includes('weighted_avg') && options.weightColumn) {
            config.weightColumn = options.weightColumn;
        }
        return this.pivot(config, options);
    }

    groupBy(groupCols, aggregations = {}) {
        // Keep Arquero semantics, but avoid Arrow->objects materialization (use aq.fromArrow)
        this.#evaluate();

        const cols = Array.isArray(groupCols) ? groupCols : [groupCols];
        const aqTbl = aq.fromArrow(this.#cached);

        const aqAggs = {};
        for (const [newCol, spec] of Object.entries(aggregations)) {
            if (typeof spec === 'string') {
                aqAggs[newCol] = spec;
            } else if (_isObj(spec) && spec.op && spec.field) {
                aqAggs[newCol] = spec.op === 'count' ? 'count()' : `${spec.op}(d => d.${spec.field})`;
            }
        }

        const grouped = Object.keys(aqAggs).length > 0
            ? aqTbl.groupby(cols).rollup(aqAggs)
            : aqTbl.groupby(cols).count();

        return new HyperTable(grouped.toArrow()).setPrimaryKey(cols);
    }

    orderBy(columns, directions = []) {
        // Keep Arquero semantics, but avoid Arrow->objects materialization (use aq.fromArrow)
        this.#evaluate();

        const sortSpec = {};
        if (typeof columns === 'string') {
            sortSpec[columns] = directions[0] || 'asc';
        } else if (Array.isArray(columns)) {
            columns.forEach((col, i) => {
                sortSpec[col] = directions[i] || 'asc';
            });
        } else if (_isObj(columns)) {
            Object.assign(sortSpec, columns);
        } else {
            throw new Error('Invalid columns specification for orderBy');
        }

        const aqTbl = aq.fromArrow(this.#cached);

        const aqSortCols = Object.keys(sortSpec);
        const aqSortDirs = Object.values(sortSpec).map((dir) =>
            String(dir).toLowerCase() === 'desc' ? 'desc' : 'asc'
        );

        const sorted = aqTbl.orderby(aqSortCols, aqSortDirs);
        const out = new HyperTable(sorted.toArrow()).setPrimaryKey(this.#pk);
        if (this.#immutable) out.immutable(true);
        return out;
    }

    // Zero-copy row proxy: reads directly from Arrow columns on demand
    // instead of materializing the full JS object array.
    _rowProxy(table, rowIndex) {
        const proxy = {};
        for (const field of table.schema.fields) {
            Object.defineProperty(proxy, field.name, {
                get: () => table.getChild(field.name)?.get(rowIndex),
                enumerable: true
            });
        }
        return proxy;
    }

    map(mapFn) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        const result = new Array(n);
        for (let i = 0; i < n; i++) {
            result[i] = mapFn(this._rowProxy(tbl, i), i);
        }
        return result;
    }

    reduce(reduceFn, initialValue) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        let acc = initialValue;
        for (let i = 0; i < n; i++) {
            acc = reduceFn(acc, this._rowProxy(tbl, i), i);
        }
        return acc;
    }

    forEach(fn) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        for (let i = 0; i < n; i++) {
            fn(this._rowProxy(tbl, i), i);
        }
    }

    find(predicate) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        for (let i = 0; i < n; i++) {
            const row = this._rowProxy(tbl, i);
            if (predicate(row, i)) return row;
        }
        return undefined;
    }

    findIndex(predicate) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        for (let i = 0; i < n; i++) {
            if (predicate(this._rowProxy(tbl, i), i)) return i;
        }
        return -1;
    }

    every(predicate) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        for (let i = 0; i < n; i++) {
            if (!predicate(this._rowProxy(tbl, i), i)) return false;
        }
        return true;
    }

    some(predicate) {
        this.#evaluate();
        const tbl = this.#cached;
        const n = tbl.numRows;
        for (let i = 0; i < n; i++) {
            if (predicate(this._rowProxy(tbl, i), i)) return true;
        }
        return false;
    }

    // Bulk column read: returns typed array for numeric columns, avoiding
    // full row materialization when only a single column is needed.
    getColumnValues(colName) {
        this.#evaluate();
        const col = this.#cached.getChild(colName);
        if (!col) return [];
        return col.toArray();
    }

    select(columns) {
        const columnList = Array.isArray(columns) ? columns : [columns];
        if (columnList.length === 0) throw new Error('Must specify at least one column to select');

        this.#evaluate();

        const available = this.#cached.schema.fields.map((f) => f.name);
        for (const col of columnList) {
            if (!available.includes(col)) {
                throw new Error(`Column '${col}' not found. Available columns: ${available.join(', ')}`);
            }
        }

        const selectedTable = this.#cached.select(columnList);
        const result = new HyperTable(selectedTable);

        if (this.#pk.length > 0 && this.#pk.every((pk) => columnList.includes(pk))) {
            result.setPrimaryKey(this.#pk);
        }
        if (this.#immutable) result.immutable(true);
        return result;
    }

    selectAt(indices) {
        const idx = Array.isArray(indices) ? indices : [indices];
        this.#evaluate();
        const selectedTable = this.#cached.selectAt(idx);
        const result = new HyperTable(selectedTable);
        if (this.#immutable) result.immutable(true);
        return result;
    }

    filterColumns(columns) {
        return this.select(columns);
    }

    dropColumns(columns) {
        this.#evaluate();
        const drop = Array.isArray(columns) ? columns : [columns];
        const keep = this.#cached.schema.fields.map((f) => f.name).filter((c) => !drop.includes(c));
        return this.select(keep);
    }

    head(n = 5) {
        this.#evaluate();
        const t = this.#cached.slice(0, n);
        const out = new HyperTable(t);
        if (this.#pk.length && this.#pk.every((k) => out.hasField(k))) out.setPrimaryKey(this.#pk);
        if (this.#immutable) out.immutable(true);
        return out;
    }

    tail(n = 5) {
        this.#evaluate();
        const start = Math.max(0, this.#cached.numRows - n);
        const t = this.#cached.slice(start, this.#cached.numRows);
        const out = new HyperTable(t);
        if (this.#pk.length && this.#pk.every((k) => out.hasField(k))) out.setPrimaryKey(this.#pk);
        if (this.#immutable) out.immutable(true);
        return out;
    }

    slice(start = 0, end = null) {
        this.#evaluate();
        const e = end === null ? this.#cached.numRows : end;
        const t = this.#cached.slice(start, e);
        const out = new HyperTable(t);
        if (this.#pk.length && this.#pk.every((k) => out.hasField(k))) out.setPrimaryKey(this.#pk);
        if (this.#immutable) out.immutable(true);
        return out;
    }

    concat(other) {
        this.#evaluate();
        const otherArrow = other instanceof HyperTable ? other.toArrow() : other;
        if (!(otherArrow instanceof arrow.Table)) throw new Error('concat expects a HyperTable or Arrow Table');
        const t = this.#cached.concat(otherArrow);
        const out = new HyperTable(t);
        if (this.#pk.length) out.setPrimaryKey(this.#pk);
        if (this.#immutable) out.immutable(true);
        return out;
    }

    debug(label = '') {
        this.#evaluate();
        // eslint-disable-next-line no-console
        console.debug(`[HyperTable] ${label}`, {
            rows: this.#cached.numRows,
            cols: this.#cached.numCols,
            pk: this.#pk,
            schema: this.#cached.schema?.fields?.map((f) => ({ name: f.name, typeId: f.type?.typeId, nullable: f.nullable }))
        });
        return this;
    }

    info() {
        this.#evaluate();
        return {
            rows: this.#cached.numRows,
            cols: this.#cached.numCols,
            fields: this.#cached.schema.fields.map((f) => f.name),
            pk: this.#pk.slice(),
            metaData: { ...this.metaData }
        };
    }

    crosstab(indexCol, pivotCol, fillValue = 0) {
        // Arrow-native count column (no toObject)
        this.#evaluate();
        const idx = Array.isArray(indexCol) ? indexCol : [indexCol];
        const piv = Array.isArray(pivotCol) ? pivotCol : [pivotCol];

        const n = this.#cached.numRows;
        const ones = new Int32Array(n);
        ones.fill(1);
        const countVec = arrow.makeVector(ones);

        const schema = this.#cached.schema;
        const children = {};
        for (const f of schema.fields) children[f.name] = _getChild(this.#cached, f.name);
        children.__count__ = countVec;

        const newSchema = new arrow.Schema([...schema.fields, new arrow.Field('__count__', new arrow.Int32(), true)], schema.metadata);
        const tableWithCount = new HyperTable(_makeTableFromVectors(children, newSchema));

        return tableWithCount.pivot({
            index: idx,
            columns: piv,
            values: ['__count__'],
            aggFunc: 'sum',
            fillValue
        });
    }

    merge(other, strategy = 'upsert', primaryKeys = null) {
        const pks = primaryKeys || this.#pk;
        if (!pks || pks.length === 0) throw new Error('Primary keys required for merge operation.');

        other.#evaluate?.();
        const otherData = other instanceof HyperTable ? other.toObject() : other;
        if (!Array.isArray(otherData) || otherData.length === 0) return this;

        switch (String(strategy).toLowerCase()) {
            case 'update':
                return this.batchUpdate(otherData, pks, { upsert: false });
            case 'upsert':
                return this.batchUpdate(otherData, pks, { upsert: true });
            case 'insert_only':
                return this.insert(otherData);
            default:
                throw new Error(`Unknown merge strategy: ${strategy}`);
        }
    }

    bulkOperations(operations) {
        let result = this;

        const groupedOps = { updates: [], inserts: [], deletes: [] };

        const adds = operations?.add || operations?.adds;
        const updates = operations?.update || operations?.updates;
        const removes = operations?.remove || operations?.removes;

        if (adds && Array.isArray(adds)) {
            adds.forEach((op) => groupedOps.inserts.push(...(Array.isArray(op) ? op : [op])));
        }
        if (updates && Array.isArray(updates)) {
            updates.forEach((op) => groupedOps.updates.push(...(Array.isArray(op) ? op : [op])));
        }
        if (removes && Array.isArray(removes)) {
            removes.forEach((op) => groupedOps.deletes.push(...(Array.isArray(op) ? op : [op])));
        }

        if (groupedOps.deletes.length > 0) result = result.delete(groupedOps.deletes);
        if (groupedOps.updates.length > 0) result = result.batchUpdate(groupedOps.updates);
        if (groupedOps.inserts.length > 0) result = result.insert(groupedOps.inserts);

        return result;
    }

    // ----------------------------------------------------------------------------
    // Diff / map (optimized: avoid full-table toObject in diff)
    // ----------------------------------------------------------------------------

    toMap(primaryKeys = null) {
        const pks = primaryKeys || this.#pk;
        if (!pks || pks.length === 0) throw new Error('Primary keys required for toMap');

        this.#evaluate();

        const map = new Map();
        const pkVecs = pks.map((pk) => _getChild(this.#cached, pk));

        for (let i = 0; i < this.#cached.numRows; i++) {
            const key = pkVecs.map((v) => v.get(i)).join(KEY_DELIM);
            const row = this.#cached.get(i);
            map.set(key, row && typeof row.toJSON === 'function' ? row.toJSON() : row);
        }
        return map;
    }

    diff(other, primaryKeys = null) {
        const pks = primaryKeys || this.#pk;
        if (!pks || pks.length === 0) throw new Error('Primary keys required for diff');

        this.#evaluate();
        other.#evaluate?.();

        const a = this.#cached;
        const b = other instanceof HyperTable ? other.#cached : other;
        if (!(b instanceof arrow.Table)) throw new Error('diff expects a HyperTable or Arrow Table');

        const aPkVecs = pks.map((pk) => _getChild(a, pk));
        const bPkVecs = pks.map((pk) => _getChild(b, pk));

        // Build b index trie -> rowIndex
        const bRoot = new Map();
        for (let i = 0; i < b.numRows; i++) {
            const keyVals = bPkVecs.map((v) => v.get(i));
            _trieSet(bRoot, keyVals, i);
        }

        // Track removals by building a set of keys from a and matching in b
        const removed = [];
        const added = [];
        const updated = [];

        // Helper to compare rows by columns
        const cols = a.schema.fields.map((f) => f.name);

        // Walk A: removed or updated
        for (let i = 0; i < a.numRows; i++) {
            const keyVals = aPkVecs.map((v) => v.get(i));
            const bIdx = _trieGet(bRoot, keyVals);

            if (bIdx === undefined) {
                // removed: return PK fields only (legacy did full objects in some cases, but PK is typical)
                const obj = {};
                for (let k = 0; k < pks.length; k++) obj[pks[k]] = keyVals[k];
                removed.push(obj);
                continue;
            }

            // compare values
            const change = {};
            let changed = false;

            for (const col of cols) {
                const av = _getChild(a, col)?.get(i);
                const bv = _getChild(b, col)?.get(bIdx);
                if (av !== bv) {
                    change[col] = bv;
                    changed = true;
                }
            }

            if (changed) {
                // include PK fields for updates
                for (let k = 0; k < pks.length; k++) change[pks[k]] = keyVals[k];
                updated.push(change);
            }
        }

        // Walk B: added
        // Build a membership trie for A keys (O(n) but avoids object conversion)
        const aRoot = new Map();
        for (let i = 0; i < a.numRows; i++) {
            const keyVals = aPkVecs.map((v) => v.get(i));
            _trieSet(aRoot, keyVals, true);
        }

        for (let i = 0; i < b.numRows; i++) {
            const keyVals = bPkVecs.map((v) => v.get(i));
            if (_trieGet(aRoot, keyVals) === undefined) {
                const row = b.get(i);
                added.push(row && typeof row.toJSON === 'function' ? row.toJSON() : row);
            }
        }

        return { add: added, update: updated, remove: removed };
    }

    static fromMap(map, primaryKeys, schema = null) {
        if (!(map instanceof Map)) throw new Error('fromMap expects a Map');
        const rows = Array.from(map.values());
        const ht = new HyperTable(rows);
        if (primaryKeys) ht.setPrimaryKey(primaryKeys);
        return ht;
    }

    static fromJSON(json, pk = null) {
        const data = typeof json === 'string' ? JSON.parse(json) : json;
        const ht = new HyperTable(data);
        if (pk) ht.setPrimaryKey(pk);
        return ht;
    }

    static fromArrowIPC(buffer, pk = null) {
        const ht = new HyperTable(buffer);
        if (pk) ht.setPrimaryKey(pk);
        return ht;
    }

    static fromArrow(table, pk = null) {
        const ht = new HyperTable(table);
        if (pk) ht.setPrimaryKey(pk);
        return ht;
    }

    split(specs, { strict = true } = {}) {
        this.#evaluate();
        const n = this.#cached.numRows;
        const used = strict ? new Uint8Array(n) : null;

        const outs = specs.map(() => null);

        specs.forEach((spec, idx) => {
            let mask = _vectorMask(this.#cached, spec);

            if (strict) {
                for (let i = 0; i < n; i++) if (used[i]) mask[i] = 0;
                for (let i = 0; i < n; i++) if (mask[i]) used[i] = 1;
            }

            outs[idx] = HyperTable.fromArrow(_applyMask(this.#cached, mask)).setPrimaryKey(this.#pk);
        });

        return outs;
    }
}

// Default export
export default HyperTable;
window.HyperTable = HyperTable // debug




