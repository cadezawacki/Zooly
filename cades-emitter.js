

import { CozyEvent } from 'cozyevent';

export class CadesEmitter {
    constructor(options = {}) {
        const {
            emitter = null,
            defaultRegistrar = 'cozy-sync',
            maxListeners = 0,
        } = options;

        this._owned = !emitter;
        this._emitter = emitter || new CozyEvent();

        this._registrars = new Map();
        this._installDefaultRegistrars();

        this._defaultRegistrar = defaultRegistrar;

        this._registry = new Map();
        this._wrapIndex = new WeakMap();

        if (typeof this._emitter.setMaxListeners === 'function' && Number.isInteger(maxListeners)) {
            this._emitter.setMaxListeners(maxListeners);
        }

        this._onAbortBound = (rec) => {
            try { this.offToken(rec); } catch (_) {}
        };

        this._destroyed = false;
    }

    useRegistrar(name, impl) {
        if (!name || typeof name !== 'string') throw new TypeError('Registrar name must be a string');
        if (!impl || typeof impl !== 'object') throw new TypeError('Registrar impl must be an object');

        const required = ['on', 'once', 'off', 'emit', 'removeAll'];
        for (const k of required) {
            if (typeof impl[k] !== 'function') throw new TypeError(`Registrar missing function: ${k}`);
        }
        this._registrars.set(name, impl);
        return this;
    }

    setDefaultRegistrar(name) {
        this._ensureRegistrar(name);
        this._defaultRegistrar = name;
        return this;
    }

    on(event, a, b, opts) {
        return this._subscribe('on', event, a, b, opts);
    }

    once(event, a, b, opts) {
        return this._subscribe('once', event, a, b, opts);
    }

    onMany(entries, opts) {
        const tokens = [];
        if (Array.isArray(entries)) {
            for (const item of entries) {
                if (Array.isArray(item) && item.length >= 2) {
                    tokens.push(this.on(item[0], item[1], undefined, opts));
                } else if (item && typeof item === 'object') {
                    const { event, handler, context, method, once } = item;
                    const r = once ? this.once(event, context ?? handler, method ?? handler, opts)
                        : this.on(event, context ?? handler, method ?? handler, opts);
                    tokens.push(r);
                }
            }
        } else if (entries && typeof entries === 'object') {
            for (const [evt, handler] of Object.entries(entries)) {
                tokens.push(this.on(evt, handler, undefined, opts));
            }
        } else {
            throw new TypeError('onMany expects an array or object');
        }
        return tokens;
    }

    emit(event, ...args) {
        const { registrar = this._defaultRegistrar, async = undefined } = (args.length && typeof args[args.length - 1] === 'object' && args[args.length - 1] && args[args.length - 1]._emitOptions === true)
            ? args.pop()
            : {};

        const r = this._ensureRegistrar(registrar);
        return r.emit(event, args, async);
    }

    emitAsync(event, ...args) {
        return this.emit(event, ...args, { _emitOptions: true, registrar: 'cozy-async', async: true });
    }

    emitSync(event, ...args) {
        return this.emit(event, ...args, { _emitOptions: true, registrar: 'cozy-sync', async: false });
    }

    offToken(token) {
        this._assertNotDestroyed();
        this._validateToken(token);
        const r = this._ensureRegistrar(token.registrar);
        r.off(token.event, token.wrapped);

        this._removeRecord(token);
        return true;
    }

    off(event, handlerOrToken, context) {
        this._assertNotDestroyed();
        if (handlerOrToken && handlerOrToken.__cadesToken === true) {
            return this.offToken(handlerOrToken);
        }

        const { wrapped, record } = this._findWrapped(event, handlerOrToken, context);
        if (!wrapped) return false;

        const r = this._ensureRegistrar(record.registrar);
        r.off(event, wrapped);
        this._removeRecord(record);
        return true;
    }

    clear(event = undefined, registrar = undefined) {
        this._assertNotDestroyed();
        if (!registrar) {
            for (const reg of this._registrars.values()) reg.removeAll(event ?? undefined);
            if (event == null) {
                this._registry.clear();
            } else {
                this._registry.delete(event);
            }
            return;
        }
        const reg = this._ensureRegistrar(registrar);
        reg.removeAll(event ?? undefined);
        if (event == null) {
            for (const [evt, set] of this._registry.entries()) {
                for (const rec of set) {
                    if (rec.registrar === registrar) set.delete(rec);
                }
                if (set.size === 0) this._registry.delete(evt);
            }
        } else {
            const set = this._registry.get(event);
            if (set) {
                for (const rec of set) {
                    if (rec.registrar === registrar) set.delete(rec);
                }
                if (set.size === 0) this._registry.delete(event);
            }
        }
    }

    lookup(event) {
        this._assertNotDestroyed();
        if (event && typeof event !== 'string') throw new TypeError('event must be a string');
        if (event == null) {
            const out = {};
            for (const [evt, set] of this._registry.entries()) {
                out[evt] = Array.from(set).map(this._toLookup);
            }
            return out;
        }
        const set = this._registry.get(event);
        return set ? Array.from(set).map(this._toLookup) : [];
    }

    has(event, handler, context) {
        return !!this._findWrapped(event, handler, context).wrapped;
    }

    listenerCount(event) {
        if (event && typeof event !== 'string') throw new TypeError('event must be a string');
        if (event == null) {
            let n = 0;
            for (const set of this._registry.values()) n += set.size;
            return n;
        }
        const set = this._registry.get(event);
        return set ? set.size : 0;
    }

    events() {
        return Array.from(this._registry.keys());
    }

    createScope() {
        this._assertNotDestroyed();
        const tokens = [];
        return {
            on: (e, a, b, o) => tokens.push(this.on(e, a, b, o)),
            once: (e, a, b, o) => tokens.push(this.once(e, a, b, o)),
            emit: (e, ...args) => this.emit(e, ...args),
            clear: () => { for (const t of tokens.splice(0)) this.offToken(t); },
        };
    }

    destroy() {
        if (this._destroyed) return;
        try {
            this.clear();
        } catch (e) {
            console.error('Emitter destroy/clear failed:', e);
            try { this._registry?.clear?.() || (this._events = {}); } catch (_) {}
        } finally {
            this._wrapIndex = new WeakMap();
            this._destroyed = true;
        }
    }

    _installDefaultRegistrars() {
        const e = this._emitter;
        this._registrars.set('cozy-sync', {
            on: (evt, fn) => (e.on ? e.on(evt, fn) : e.addListener(evt, fn)),
            once: (evt, fn) => (e.once ? e.once(evt, fn) : this._polyfillOnce(e, evt, fn)),
            off: (evt, fn) => (e.off ? e.off(evt, fn) : (e.removeListener && e.removeListener(evt, fn))),
            emit: (evt, args) => (e.emit ? e.emit(evt, ...args) : false),
            removeAll: (evt) => {
                if (typeof e.removeAll === 'function') return evt == null ? e.removeAll() : e.removeAll(evt);
                if (typeof e.removeAllListeners === 'function') return evt == null ? e.removeAllListeners() : e.removeAllListeners(evt);
            },
        });
        this._registrars.set('cozy-async', {
            on: (evt, fn) => (e.on ? e.on(evt, fn) : e.addListener(evt, fn)),
            once: (evt, fn) => (e.once ? e.once(evt, fn) : this._polyfillOnce(e, evt, fn)),
            off: (evt, fn) => (e.off ? e.off(evt, fn) : (e.removeListener && e.removeListener(evt, fn))),
            emit: (evt, args) => {
                if (typeof e.emitAsync === 'function') return e.emitAsync(evt, ...args);

                // Fallback: Trigger synchronously on the native emitter but resolve asynchronously
                if (typeof e.emit === 'function') e.emit(evt, ...args);

                const set = this._registry.get(evt);
                if (!set || set.size === 0) return Promise.resolve();

                const promises = [];
                for (const rec of set) {
                    promises.push(Promise.resolve().then(() => rec.wrapped(...args)));
                }
                return Promise.all(promises);
            },
            removeAll: (evt) => {
                if (typeof e.removeAll === 'function') return evt == null ? e.removeAll() : e.removeAll(evt);
                if (typeof e.removeAllListeners === 'function') return evt == null ? e.removeAllListeners() : e.removeAllListeners(evt);
            },
        });
    }

    _subscribe(kind, event, a, b, opts) {
        this._assertNotDestroyed();
        if (!event || typeof event !== 'string') throw new TypeError('event must be a non-empty string');

        const { registrar = this._defaultRegistrar, signal, debounce } = opts || {};
        const r = this._ensureRegistrar(registrar);

        let context = null;
        let original;

        if (typeof a === 'function' && b === undefined) {
            original = a;
        } else if (a && (typeof a === 'object' || typeof a === 'function') && (typeof b === 'function' || typeof b === 'string')) {
            context = a;
            original = typeof b === 'string' ? this._resolveMethod(context, b) : b;
        } else {
            throw new TypeError('Invalid arguments: expected (event, handler) or (event, context, handler|methodName)');
        }

        if (typeof original !== 'function') throw new TypeError('Handler must be a function');

        let wrapped = context ? this._wrapContext(original, context) : original;

        if (debounce !== undefined) {
            wrapped = this._debounce(wrapped, debounce);
        }

        const record = {
            __cadesToken: true,
            event,
            original,
            wrapped: null,
            context: context || null,
            registrar,
            once: (kind === 'once'),
            get handler() { return this.original; },
        };

        if (wrapped.cancel) {
            record._cancel = wrapped.cancel;
            record.flush = wrapped.flush;
        }

        // SURGICAL FIX: Prevent 'once' memory leak by self-destructing from the registry on execution
        if (kind === 'once') {
            const userWrapped = wrapped;
            wrapped = (...args) => {
                this._removeRecord(record);
                return userWrapped(...args);
            };
        }

        record.wrapped = wrapped;

        if (kind === 'once') r.once(event, wrapped);
        else r.on(event, wrapped);

        let set = this._registry.get(event);
        if (!set) { set = new Set(); this._registry.set(event, set); }
        set.add(record);

        this._indexWrap(record);

        if (signal && typeof signal.addEventListener === 'function') {
            const abortHandler = () => this._onAbortBound(record);
            record._abortHandler = abortHandler;
            record._signal = signal;
            signal.addEventListener('abort', abortHandler, { once: true });
        }

        return Object.freeze(record);
    }

    _wrapContext(original, context) {
        return function wrappedListener(...args) {
            return original.apply(context, args);
        };
    }

    _resolveMethod(context, name) {
        const fn = context[name];
        if (typeof fn !== 'function') throw new TypeError(`Method "${name}" not found on context`);
        return fn;
    }

    _indexWrap(rec) {
        let ctxMap = this._wrapIndex.get(rec.original);
        if (!ctxMap) { ctxMap = new Map(); this._wrapIndex.set(rec.original, ctxMap); }

        const key = rec.context || null;
        let evtMap = ctxMap.get(key);
        if (!evtMap) { evtMap = new Map(); ctxMap.set(key, evtMap); }

        // SURGICAL FIX: Use Set to prevent collisions if the same function is registered twice
        let evtSet = evtMap.get(rec.event);
        if (!evtSet) { evtSet = new Set(); evtMap.set(rec.event, evtSet); }

        evtSet.add(rec.wrapped);
    }

    _findWrapped(event, original, context) {
        if (!event || typeof event !== 'string') return { wrapped: null, record: null };
        const set = this._registry.get(event);
        if (!set || set.size === 0) return { wrapped: null, record: null };

        if (original && typeof original === 'function') {
            const ctxMap = this._wrapIndex.get(original);
            const key = context || null;
            const evtMap = ctxMap && ctxMap.get(key);
            const evtSet = evtMap && evtMap.get(event);

            // SURGICAL FIX: Iterate the Set to find the exact colliding reference
            if (evtSet) {
                for (const wrapped of evtSet) {
                    for (const rec of set) {
                        if (rec.wrapped === wrapped) return { wrapped, record: rec };
                    }
                }
            }
        }
        return { wrapped: null, record: null };
    }

    _removeRecord(record) {
        if (record._cancel) { record._cancel(); record._cancel = null; }

        const set = this._registry.get(record.event);
        if (set) {
            set.delete(record);
            if (set.size === 0) this._registry.delete(record.event);
        }

        const ctxMap = this._wrapIndex.get(record.original);
        if (ctxMap) {
            const key = record.context || null;
            const evtMap = ctxMap.get(key);
            if (evtMap) {
                const evtSet = evtMap.get(record.event);
                if (evtSet) {
                    evtSet.delete(record.wrapped);
                    if (evtSet.size === 0) evtMap.delete(record.event);
                }
                if (evtMap.size === 0) ctxMap.delete(key);
            }
            if (ctxMap.size === 0) this._wrapIndex.delete(record.original);
        }

        if (record._abortHandler && record._signal && typeof record._signal.removeEventListener === 'function') {
            try { record._signal.removeEventListener('abort', record._abortHandler); } catch (_) {}
        }
    }

    _toLookup(rec) {
        return {
            event: rec.event,
            registrar: rec.registrar,
            once: rec.once,
            handler: rec.original,
            context: rec.context,
        };
    }

    _debounce(fn, config) {
        const wait = typeof config === 'number' ? config : config.wait;
        const leading = typeof config === 'object' && !!config.leading;
        const trailing = typeof config === 'object' && config.trailing === false ? false : true;

        if (!Number.isFinite(wait) || wait < 0) throw new TypeError('debounce wait must be a non-negative number');

        let timer = null;
        let lastArgs = null;
        let leadFired = false;

        function debounced(...args) {
            lastArgs = args;

            if (leading && !timer) {
                leadFired = true;
                fn(...args);
                timer = setTimeout(() => { flush(); }, wait);
                return;
            }

            leadFired = false;
            clearTimeout(timer);
            timer = setTimeout(() => { flush(); }, wait);
        }

        function flush() {
            clearTimeout(timer);
            timer = null;
            if (trailing && lastArgs && !leadFired) fn(...lastArgs);
            lastArgs = null;
            leadFired = false;
        }

        debounced.cancel = () => {
            clearTimeout(timer);
            timer = null;
            lastArgs = null;
            leadFired = false;
        };

        debounced.flush = () => {
            if (timer !== null) flush();
        };

        return debounced;
    }

    _polyfillOnce(emitter, event, fn) {
        const onceFn = (...args) => {
            try { fn(...args); } finally {
                if (emitter.off) emitter.off(event, onceFn);
                else if (emitter.removeListener) emitter.removeListener(event, onceFn);
            }
        };
        if (emitter.on) emitter.on(event, onceFn);
        else if (emitter.addListener) emitter.addListener(event, onceFn);
    }

    _ensureRegistrar(name) {
        const r = this._registrars.get(name);
        if (!r) throw new Error(`Unknown registrar: ${name}`);
        return r;
    }

    _validateToken(t) {
        if (!t || t.__cadesToken !== true || typeof t.event !== 'string' || typeof t.wrapped !== 'function') {
            throw new TypeError('Invalid subscription token');
        }
    }

    _assertNotDestroyed() {
        if (this._destroyed) throw new Error('CadesEmitter is destroyed');
    }
}
