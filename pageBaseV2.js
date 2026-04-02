

import {
    defer, from, timer, firstValueFrom, fromEvent,
    merge, map, debounceTime, Subject, takeUntil, tap, filter, take
} from 'rxjs';
import { throttleTime } from 'rxjs/operators';
import { ObservableDictionary } from '@/utils/ObservableDictionary.js';
import { v4 as uuidv4 } from 'uuid';
import { retryOperation } from '@/utils/helpers.js';
import { CadesEmitter } from "@/utils/cades-emitter.js";
import {ArrowAgGridAdapter} from "@/grids/js/arrow/arrowEngine.js";

function _arrow_to_col_dtype(dtype) {
    switch(dtype){
        case 'float64': return 'float';
        case 'float32': return 'float';
        case 'int32':   return 'integer';
        case 'uint32':  return 'integer';
        case 'int16':   return 'integer';
        case 'uint16':  return 'integer';
        case 'uint8':   return 'integer';
        case 'int8':    return 'integer';
        default:        return 'text'
    }
}

function changeFavicon(src) {
    const link = document.createElement('link'),
        oldLink = document.getElementById('dynamic-favicon');
    link.id = 'dynamic-favicon';
    link.rel = 'shortcut icon';
    link.href = src;
    if (oldLink) {
        document.head.removeChild(oldLink);
    }
    document.head.appendChild(link);
}

export class PageBase {
    static _shared = new Map();
    static _sharedContext = new Map();
    static _persistedStores = new Set();
    static _pages = new Map();

    constructor(name, { url_context = null, context = {}, config = {}, container=null} = {}) {
        this.initialized = false;
        this.container = container;

        this.context = {
            name: name,
            page: this,
            isFrame: name === 'frame',
            url_context: url_context,
            ...context
        };

        this.name = name;

        this._listeners = new Map();             // uuid -> AbortController
        this._handlers = new Map();              // room -> type -> Set(handlers)
        this._rooms = new Set();                 // subscribed (local view)
        this._stores = new Map();                // key -> { dict, subs:Set<Subscription> }
        this._sharedKeys = new Set();            // shared store keys created by this page (uppercase)
        this._sharedContextKeys = new Set();     // shared context keys created by this page (uppercase)
        this._breakpoints = new Map();           // name -> breakpoint controller with destroy()
        this._roleGuards = [];                   // [{ el, role }]

        this._subscriptions = [];                // rxjs subscriptions not tied to a specific store
        this._storageListeners = new Map();      // (key+type) -> fn
        this._persistentStorageMap = new Map();  // storageKey -> storeKey (uppercase)

        this._timeouts = new Set();
        this._intervals = new Set();
        this._rafMap = new Map();                // key -> rafId

        // Accessible
        this._shared = PageBase._shared;
        this._sharedContext = PageBase._sharedContext;

        this.context.destroy$ = new Subject();
        this.page$ = null;

        this._ensureErrorOverlay();
        this.emitter = new CadesEmitter();

        // this.eagerColumnCacheEnabled = true;
        // this.eagerColumns = new EagerColumnCache(name);
        // this._eagerColTrackers = [];
    }

    // ===== Lifecycle Hooks (meant to be overridden by subclasses) =====
    async onBeforeInit()   {}
    async onInit()         {}
    async onAfterInit()    {}
    async onCacheDom()     {}
    async onBeforeBind()   {}
    async onBind()         {}
    async onAfterBind()    {}
    async onReady()        {}
    async onError(e)       {}
    async onBeforeCleanup(){}
    async onCleanup()      {}
    async onAfterCleanup() {}
    async setupHotkeys()   {}

    async init() {
        if (this._initialized) return;
        this._initialized = true;
        await this._init();

        await this._onCacheDom();
        await this.onCacheDom();

        await this.onBeforeInit();
        await this.onInit();
        await this.onAfterInit();

        await this.onBeforeBind();
        await this._onBind();
        await this.onBind();
        await this.onAfterBind();

        await this.setupHotkeys();
        await this.setWindowTitle();
        await this.setFavicon();

        await this._onReady();
        await this.onReady();
        document.getElementById("drawer-content").style.display = "flex";
    }

    // ===== State Helpers =====
    isInitialized() { return this.page$?.get('initialized'); }
    isAlive() { return this.page$ && this.page$.get('alive'); }
    isConnected() { return this.getSharedContext('socketManager')?.isConnected ?? false; }

    // ===== Managers =====
    modalManager()        { return this.getSharedContext('modalManager') ?? null; }
    serialManager()       { return this.getSharedContext('serialManager') ?? null; }
    socketManager()       { return this.getSharedContext('socketManager') ?? null; }
    settingsManager()     { return this.getSharedContext('settingsManager') ?? null; }
    subscriptionManager() { return this.getSharedContext('subscriptionManager') ?? null; }
    themeManager()        { return this.getSharedContext('themeManager') ?? null; }
    toastManager()        { return this.getSharedContext('toastManager') ?? null; }
    userManager()         { return this.getSharedContext('userManager') ?? null; }
    colorManager()        { return this.getSharedContext('colorManager') ?? null; }
    scratchPad()          { return this.getSharedContext('scratchPad') ?? null; }
    tooltipManager()      { return this.getSharedContext('tooltipManager') ?? null; }

    // ===== Shared Access =====
    getPersistedStoreKeys() { return PageBase._persistedStores; }

    _setupEmits() {

    }

    // ———— Error Boundary ————
    _ensureErrorOverlay() {
        if (typeof document === 'undefined') return;
        if (document.getElementById('pb-error')) return;
        const o = document.createElement('div');
        o.id = 'pb-error'; Object.assign(o.style, {
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
            color: '#fff', display: 'none', alignItems: 'center', justifyContent: 'center',
            flexDirection: 'column', zIndex: 9999
        });
        const m = document.createElement('div'); m.id = 'pb-error-msg';
        const b = document.createElement('button'); b.textContent = 'Retry';
        b.onclick = () => { o.style.display = 'none'; this.init(); };
        o.append(m, b);
        document.body.append(o);
    }
    handleError(phase, e) {
        console.error(`Error ${phase}`, e);
        const o = typeof document !== 'undefined' && document.getElementById('pb-error');
        const m = typeof document !== 'undefined' && document.getElementById('pb-error-msg');
        if (o && m) { m.textContent = `Error in ${phase}: ${e.message}`; o.style.display = 'flex'; }
    }

    // ———— Offline Support ————
    _setupOfflineSupport() {
        if (typeof window === 'undefined') return;
        this.addEventListener(window, 'online', () => this.onOnline());
        this.addEventListener(window, 'offline', () => this.onOffline());
    }
    onOnline()  { this._rooms.forEach(r => this.subscribeToPageRoom(r).catch(e => { console.error('Resubscription failed for room', r, e); })); }
    onOffline() {}

    // ———— Role Guards ————
    requireRole(elSelOrEl, role) {
        const el = typeof elSelOrEl === 'string' ? document.querySelector(elSelOrEl) : elSelOrEl;
        this._roleGuards.push({ el, role });
        this._applyRoles();
    }
    _applyRoles() {
        const roles = this.userManager()?.getRoles() || [];
        this._roleGuards.forEach(({ el, role }) => { if (el) el.style.display = roles.includes(role) ? '' : 'none'; });
    }
    _setupRoleGuardWatcher() {
        const um = this.userManager();
        if (um?.role$) {
            const sub = um.role$.subscribe(roles => this._checkRoles(roles));
            this._subscriptions.push(sub);
        }
    }
    _checkRoles(roles = this.userManager()?.getRoles() || []) {
        this._roleGuards.forEach(({ el, role }) => this._applyRoles(el, role, roles));
    }

    // ———— ObservableDictionary Stores ————
    createStore(key, initial = null, opts = {}) {
        const K = String(key).toUpperCase();
        if (!this._stores.has(K)) {

            if (opts?.version != null) {
                initial['_version_'] = opts.version;
            }

            if (opts.persist) {
                PageBase._persistedStores.add(opts.storageKey);
                this._persistentStorageMap.set(opts.storageKey, K);
                const storageType = opts?.storageType ?? 'local';

                this.onStorage(opts.storageKey, (storageEvent) => {
                    try {
                        const raw = storageEvent.newValue;
                        if (raw == null) return;
                        const newValue = JSON.parse(raw);
                        const storeKey = this._persistentStorageMap.get(opts.storageKey);
                        if (this.sharedStoreExists(storeKey)) {
                            const store = this.getSharedStore(storeKey, {createOnMissing:true});
                            store.update(newValue);
                        }
                    } catch (e) { console.warn('Storage event error:', e); }
                }, storageType);
            }

            const dict = new ObservableDictionary(initial, opts);
            this._stores.set(K, { dict, subs: new Set() });
        }
        return this._stores.get(K).dict;
    }
    subscribeToDict(key, obs, fn) {
        const K = String(key).toUpperCase();
        let e = this._stores.get(K);
        if (!e) e = this._shared.get(K);
        if (!e) throw new Error(`Store ${key} not found`);
        const sub = obs.subscribe(fn);
        e.subs.add(sub);
        return sub;
    }

    untilDestroyed() { return takeUntil(this.context.destroy$); }

    safeSubscribe(observable$, next, error, complete) {
        const sub = observable$.pipe(this.untilDestroyed()).subscribe(next, error, complete);
        this._subscriptions.push(sub);
        return sub;
    }

    fromEvent$(target, type, options) {
        return fromEvent(target, type, options).pipe(this.untilDestroyed());
    }

    clearAllSubscriptions() {
        this._stores.forEach(({ dict, subs }) => {
            subs.forEach(s => { try { s.unsubscribe(); } catch {} });
            // Only dispose dictionaries that actually have a disposer
            if (dict && typeof dict.dispose === 'function') {
                try { dict.dispose(); } catch {}
            }
        });
        this._stores.clear();
    }

    serializeState() {
        const snap = {};
        this._stores.forEach(({ dict }, key) => {
            try {
                // Only include ObservableDictionary snapshots
                if (dict && typeof dict.getValue === 'function') {
                    snap[key] = Array.from(dict.getValue().entries());
                }
            } catch {}
        });
        return snap;
    }

    restoreState(snapshot) {
        for (const [k, entries] of Object.entries(snapshot || {})) {
            const dict = this.createStore(k, new Map(entries));
            dict.update(new Map(entries));
        }
    }

    createSharedContext(key, value = null) {
        const k = String(key).toUpperCase();
        if (!PageBase._sharedContext.has(k)) {
            PageBase._sharedContext.set(k, value);
        }
        const e = PageBase._sharedContext.get(k);
        this._sharedContextKeys.add(k);
        return e;
    }

    getSharedContext(key, fallback = null) {
        const k = String(key).toUpperCase();
        return PageBase._sharedContext.get(k) ?? fallback;
    }

    setWindowTitle(title = "Portfolio Webtool") {
        document.title = title;
    }

    setFavicon(src="/assets/ico/pt.ico") {
        if (!src.startsWith('/assets/ico/')) {
            src = `/assets/ico/${src}`
        }
        if (!src.endsWith('.ico')) {
            src = `${src}.ico`
        }
        return changeFavicon(src)
    }

    destroySharedContext(key) {
        const k = String(key).toUpperCase();
        if (this._sharedContextKeys.has(k)) {
            PageBase._sharedContext.delete(k);
            this._sharedContextKeys.delete(k);
        }
    }

    createSharedStore(key, initial = null, opts = {}) {
        const K = String(key).toUpperCase();
        initial = initial ?? {};

        if (opts?.version != null) {
            initial['_version_'] = opts.version;
        }

        if (opts.persist) {
            PageBase._persistedStores.add(opts.storageKey);
            this._persistentStorageMap.set(opts.storageKey, K);
            const storageType = opts?.storageType ?? 'local';
            this.onStorage(opts.storageKey, (storageEvent) => {
                try {
                    const raw = storageEvent.newValue;
                    if (raw == null) return;
                    const newValue = JSON.parse(raw);
                    const storeKey = this._persistentStorageMap.get(opts.storageKey);
                    if (this.sharedStoreExists(storeKey)) {
                        const store = this.getSharedStore(storeKey, {createOnMissing:true});
                        store.update(newValue);
                    }
                } catch (e) { console.warn('Storage event error:', e); }
            }, storageType);
        }
        if (!PageBase._shared.has(K)) {
            const dict = new ObservableDictionary(initial, opts);
            PageBase._shared.set(K, { dict, ref: 0 });
        }

        const e = PageBase._shared.get(K);
        if (!this._sharedKeys.has(K)) {
            e.ref++;
            this._sharedKeys.add(K);
        }
        return e.dict;
    }

    _disposeShared(K) {
        const e = PageBase._shared.get(K);
        if (!e) return;
        e.ref--;
        if (e.ref <= 0) { try { e.dict.dispose(); } catch {} PageBase._shared.delete(K); }
    }

    sharedStoreExists(key) {
        if (!key) return false;
        return PageBase._shared.has(String(key).toUpperCase());
    }

    getSharedStore(key, { createOnMissing = false } = {}) {
        if (!key) return;
        const K = String(key).toUpperCase();
        if (this.sharedStoreExists(K)) return PageBase._shared.get(K).dict;
        if (createOnMissing) return this.createSharedStore(K);
    }

    async _getSharedStoreWithRetry(key) {
        const result = this.getSharedStore(key);
        if (result) return result;
        throw new Error();
    }

    async getSharedStoreWithRetry(key, { maxAttempts = 5, retryTimer = 1000 } = {}) {
        const f = async () => this._getSharedStoreWithRetry(key);
        return await retryOperation(f, maxAttempts, retryTimer);
    }

    _createChangeAggregator(sourcesObject) {
        const changeStreams = Object.entries(sourcesObject).map(([name, sourceDict]) =>
            sourceDict.changes$.pipe(map(change => ({ source: name, change })))
        );
        return merge(...changeStreams);
    }

    createDebouncedStore(name, ms, initial = null, opts = {}) {
        const source = this.createStore(name, initial, opts);
        if (!source) return null;
        const debouncedSource = source.pipe(debounceTime(ms));
        const baseDebouncedKey = `${name}_DEBOUNCED_${ms}ms`.toUpperCase();
        let finalDebouncedKey = baseDebouncedKey, counter = 0;
        while (this._stores.has(finalDebouncedKey)) { counter++; finalDebouncedKey = `${baseDebouncedKey}_${counter}`; }
        const subs = new Set();
        const sub = debouncedSource.subscribe();
        subs.add(sub);
        this._stores.set(finalDebouncedKey, { dict: debouncedSource, subs });
        return debouncedSource;
    }

    createThrottledStore(name, ms, initial = null, opts = {}) {
        const source = this.createStore(name, initial, opts);
        if (!source) return null;
        const throttledSource = source.pipe(throttleTime(ms));
        const baseThrottledKey = `${name}_THROTTLED_${ms}ms`.toUpperCase();
        let finalThrottledKey = baseThrottledKey, counter = 0;
        while (this._stores.has(finalThrottledKey)) { counter++; finalThrottledKey = `${baseThrottledKey}_${counter}`; }
        const subs = new Set();
        const sub = throttledSource.subscribe();
        subs.add(sub);
        this._stores.set(finalThrottledKey, { dict: throttledSource, subs });
        return throttledSource;
    }

    createPipedStore(name, initial = null, opts = {}, ...operators) {
        const source = this.createStore(name, initial, opts);
        if (!source) return null;
        if (operators.length === 0) return source;

        const pipedSource = source.pipe(...operators);
        const basePipedKey = `${name}_PIPED`.toUpperCase();
        let finalPipedKey = basePipedKey, counter = 0;
        while (this._stores.has(finalPipedKey)) { counter++; finalPipedKey = `${basePipedKey}_${counter}`; }
        const subs = new Set();
        const sub = pipedSource.subscribe();
        subs.add(sub);
        this._stores.set(finalPipedKey, { dict: pipedSource, subs });
        return pipedSource;
    }

    onMergedChange(sourcesObject, fn) {
        const sourceChanges$ = this._createChangeAggregator(sourcesObject);
        const s = sourceChanges$.subscribe(({ source, change }) => fn(source, change));
        this._subscriptions.push(s);
        return s;
    }

    // ———— DOM & Events ————
    _onCacheDom() {
        this.drawer = typeof document !== 'undefined' ? document.getElementById("drawer-content") : null;
    }

    addEventListener(target, type, fn, opts = {}, u = null) {
        if (u == null) u = uuidv4();
        let a;
        if (!this._listeners.has(u)) {
            a = new AbortController();
            this._listeners.set(u, a);
        } else {
            a = this._listeners.get(u);
        }
        const { signal } = a;
        const options = { ...opts, signal };
        target.addEventListener(type, fn, options);
        return u;
    }

    removeEventListener(uuid) {
        if (this._listeners.has(uuid)) {
            const controller = this._listeners.get(uuid);
            try { controller.abort(); } catch {}
            this._listeners.delete(uuid);
            console.debug(`Page '${this.context.name}' removed listener: ${uuid}`);
        }
    }

    _updateStorageListeners(e) {
        const key = e.key;
        const storageType = e.storageArea === sessionStorage ? 'session' : 'local';
        const k = key + storageType;
        if (this._storageListeners.has(k)) {
            try { this._storageListeners.get(k)(e); } catch (err) { console.warn('Storage event error:', err); }
        }
    }

    onStorage(key, fn, type = 'local', opts = {}, u = 'storageListener') {
        this._storageListeners.set(key + type, fn);
        if (!this._listeners.has(u)) {
            this.addEventListener(window, 'storage', (e) => this._updateStorageListeners(e), opts, u);
        }
    }

    getElement(elementOrSelector) {
        const el = (typeof elementOrSelector === 'string')
            ? (typeof document !== 'undefined' ? document.querySelector(elementOrSelector) : null)
            : elementOrSelector;
        if (!el) throw new Error('Element not found');
        return el
    }

    isAsync(fn) {
        return fn.constructor.name === 'AsyncFunction';
    }

    linkStoreToInput(_dom, key=null, store=null, {default_val=false, persist = false, storageKey = null, persistOpts = {}, cb = null } = {}) {
        if (!_dom) throw Error('No DOM')
        const dom = this.getElement(_dom);
        const tag = dom.tagName;
        if (tag !== 'INPUT') throw Error('Must be an <input> tag to link to input')

        let my_store = store;
        let my_key = key;
        if ((my_key == null) && (dom.id !== '')){
            my_key = dom.id
        }
        if ((my_key == null) || (my_key === '')) {
            throw Error("Linked storage needs a DOM Id or passed key to store")
        }

        if (!this._stores.has(my_key)) {
            if (my_store != null) {
                this._stores.set(my_key, { my_store, subs: new Set() });
            } else {
                my_store = this.createStore(my_key, {[my_key]: default_val});
            }
        }

        if (!my_store.has(my_key)) {
            my_store.set([my_key], default_val)
        }

        if (persist) {
            let my_storageKey = storageKey || key;
            my_store.bindPersistentKeys([my_key], {
                ...{
                    namespace: this.name,
                    key: my_storageKey,
                    storage: 'local',
                    pruneNullish: true,
                    reconcile: 'replace',
                    listen: true
                },
                ...persistOpts
            });
        }

        dom.checked = my_store.get(my_key);
        this.addEventListener(dom, 'input', (v) => {
            my_store.set(my_key, v?.target?.checked ?? false)
        });

        if (typeof cb === 'function') {
            requestAnimationFrame(()=>{
                if (this.isAsync(cb)) {
                    my_store.onValueChanged(my_key, async (cur, prev) => {
                        dom.checked = cur;
                        return await cb(cur, prev);

                    });
                } else {
                    my_store.onValueChanged(my_key, (cur, prev) => {
                        dom.checked = cur;
                        return cb(cur, prev);
                    });
                }
            })
        } else {
            my_store.onValueChanged(my_key, (cur, prev) => {
                dom.checked = cur;
            });
        }



    }

    onClickOutside(elOrSelector, handler, { once = false } = {}) {
        const el = typeof elOrSelector === 'string' ? document.querySelector(elOrSelector) : elOrSelector;
        if (!el) return null;
        const uuid = this.addEventListener(document, 'mousedown', (ev) => {
            const target = ev.target;
            if (!el.contains(target)) {
                try { handler(ev); } catch (e) { this.handleError('onClickOutside', e); }
                if (once) this.removeEventListener(uuid);
            }
        });
        return uuid;
    }

    qs(selector, scope = document) { try { return (scope || document).querySelector(selector); } catch { return null; } }

    qsa(selector, scope = document) { try { return Array.from((scope || document).querySelectorAll(selector)); } catch { return []; } }


    // ———— WebSocket Message Handlers ————
    addMessageHandler(room, type, handler) {
        const r = String(room).toUpperCase(), t = String(type).toLowerCase();
        if (!this._handlers.has(r)) this._handlers.set(r, new Map());
        const m = this._handlers.get(r);
        if (!m.has(t)) m.set(t, new Set());
        const s = m.get(t);
        if (s.has(handler)) return;
        s.add(handler);
        this.subscriptionManager()?.registerMessageHandler(r, t, handler);
    }
    onMessage(room, type, handler) { this.addMessageHandler(room, type, handler); }

    removeMessageHandler(room, type, handler) {
        const r = String(room).toUpperCase(), t = String(type).toLowerCase();
        const m = this._handlers.get(r);
        if (!m || !m.get(t)?.delete(handler)) return;
        this.subscriptionManager()?.unregisterMessageHandler(r, t, handler);
        if (m.get(t).size === 0) { m.delete(t); if (m.size === 0) this._handlers.delete(r); }
    }
    removeAllMessageHandlers() {
        this._handlers.forEach((types, r) => {
            types.forEach((hs, t) => { hs.forEach(h => this.subscriptionManager()?.unregisterMessageHandler(r, t, h)); });
        });
        this._handlers.clear();
    }

    // --- Timing ---
    setTimeoutScoped(fn, ms) {
        const id = setTimeout(() => { this._timeouts.delete(id); try { fn(); } catch (e) { this.handleError('timeout', e); } }, ms);
        this._timeouts.add(id);
        return id;
    }

    setIntervalScoped(fn, ms) {
        const id = setInterval(() => { try { fn(); } catch (e) { this.handleError('interval', e); } }, ms);
        this._intervals.add(id);
        return id;
    }

    clearTimers() {
        for (const id of this._timeouts) { try { clearTimeout(id); } catch {} }
        for (const id of this._intervals) { try { clearInterval(id); } catch {} }
        this._timeouts.clear();
        this._intervals.clear();
    }

    sleep(ms) { return new Promise(r => this.setTimeoutScoped(r, ms)); }

    scheduleRaf(key, fn) {
        const k = String(key);
        if (this._rafMap.has(k)) {
            return this._rafMap.get(k);
        }
        const id = requestAnimationFrame(() => {
            this._rafMap.delete(k);
            try { fn(); } catch (e) { this.handleError('raf', e); }
        });
        this._rafMap.set(k, id);
        return id;
    }

    frame() {
        return PageBase._pages.get('frame')
    }

    // ===== Internal Hooks =====
    async _init() {

        if (PageBase._pages.has(this.name)) {
            await PageBase._pages.get(this.name).cleanup();
        }
        PageBase._pages.set(this.name, this)

        this.page$ = this.createStore(this.name, {
            initialized: false,
            alive: true,
            domAnimationFinished: false
        });

        this.initialized$ = this.page$.pick('initialized');
        this._setupEmits();

        await this._listenToIntroDom()

        this.page$.onValueChanged('domAnimationFinished', () => {
            this.setTimeoutScoped(() => {
                const drawer = typeof document !== 'undefined' && document.querySelector('.drawer');
                if (drawer) drawer.classList.add('initialized');
            }, 1000);
        });
    }

    _listenToIntroDom(container, cb=null) {
        if (!container) return
        return Promise.all(
            container.getAnimations({ subtree: true })
                .filter((a)=> {
                    try {
                        if (a.animationName == null) return false
                        const el = a?.effect?.target;
                        if (!el) return false
                        const iter_count = window.getComputedStyle(el).animationIterationCount;
                        if (iter_count === 'infinite') return false
                        return true
                    } catch (_) {
                        return false
                    }
                }).map((animation) => animation.finished)
        ).then(() => {
            if (typeof cb == 'function') cb()
        });
    }

    create_generic_field(field, template, engine) {
        const t = {...template};
        t.field = field;
        t.headerName = field;
        let dtype = engine.table.getChild(field)?.data?.[0]?.type || 'LargeUtf8'
        dtype = dtype.toString().toLowerCase();
        t.context = t?.context ?? {};
        t.context.dataType = _arrow_to_col_dtype(dtype);
        return t
    }

    async updateRoomFilters(room, context = {}, options = {}) {
        const upperRoom = String(room || '').toUpperCase();
        if (!upperRoom) return Promise.reject('Invalid room name');

        const subManager = this.subscriptionManager();
        if (!subManager) return Promise.reject('SubscriptionManager not available');

        if (!this._rooms.has(upperRoom)) {
            return subManager.subscribeToRoom(upperRoom, context, options);
        }
        try {
            await subManager.updateSubscriptionFilter(upperRoom, context, options);
        } catch (error) {
            console.error(`Page '${this.context.name}': Failed to subscribe to room '${upperRoom}':`, error);
            this._rooms.delete(upperRoom);
            return Promise.reject(error);
        }
        return Promise.resolve();
    }

    _getInitialColumns(grid_id) {
        let base = null;

        // 1. Try persisted user preset from TreeColumnChooser
        try {
            if (grid_id) {
                const stored = localStorage.getItem(`treeColumnChooser_presets_${grid_id}`);
                if (stored) {
                    const presets = JSON.parse(stored);
                    let preset = presets.find(p => p?.metaData?.isDefault && !p?.metaData?.isGlobal);
                    if (!preset && presets.length) preset = presets[0];
                    if (preset?.columnState) {
                        const visible = preset.columnState
                                               .filter(c => c.hide === false)
                                               .map(c => c.colId);
                        if (visible.length > 0) base = visible;
                    }
                }
            }
        } catch {}

        // 2. Fallback
        if (!base) {
            try {
                const globalPresets = ArrowAgGridAdapter.prototype.getGlobalPresets();
                if (globalPresets?.length && globalPresets[0].columnState) {
                    base = globalPresets[0].columnState
                                            .filter(c => c.hide === false)
                                            .map(c => c.colId);
                    if (!base.length) base = null;
                }
            } catch {}
        }

        // 3. No info at all — null
        if (!base) return null;

        // 4. Merge in eagerly-cached columns
        if (this.eagerColumnCacheEnabled && grid_id) {
            return this.eagerColumns.mergeWith(grid_id, base);
        }
        return base;
    }

    trackEagerColumns(engine, gridId, thresholdMs = 30_000) {
        if (!this.eagerColumnCacheEnabled || !engine || !gridId) return;

        const deadline = Date.now() + thresholdMs;
        const off = engine.onColumnsLoaded(({ columns }) => {
            if (!columns || !columns.length) return;
            if (Date.now() > deadline) { try { off(); } catch {} return; }
            this.eagerColumns.add(gridId, columns);
        });

        // Auto-stop tracking after the threshold expires
        const tid = setTimeout(() => { try { off(); } catch {} }, thresholdMs);

        this._eagerColTrackers.push(() => {
            clearTimeout(tid);
            try { off(); } catch {}
        });
    }

    async _onBind() {
        if (!this.drawer) return; // safety: drawer may not exist in some pages
        this.drawer.querySelectorAll("[class*=\"tooltip-\"]").forEach((tooltip) => {
            this.addEventListener(tooltip, 'mousedown', function () {
                tooltip.classList.add('tooltip-clicked');
                const removeClickedClass = () => {
                    tooltip.classList.remove('tooltip-clicked');
                    tooltip.removeEventListener('mouseleave', removeClickedClass);
                };
                tooltip.addEventListener('mouseleave', removeClickedClass, { once: true });
            });
        });
    }

    async _onReady() {
        this.page$.set('initialized', true);

        if ((this.name !== 'frame') && (this.name !== 'homepage')) {
            this.frame().toggleSidebar(false);
        }

        const page_body = (
            this.container?.children && this.container.children.length
                ? this.container.children[0]?.querySelector('div')
                : this.container?.querySelector('div')
        )

        if (page_body && page_body.style.display === 'none') {
            await this._listenToIntroDom(page_body, () => {
                requestAnimationFrame(()=>{
                    page_body.style.display = '';
                    requestAnimationFrame(()=> {
                        this.page$.set('domAnimationFinished', true);
                    });
                })
            });
        }
    }

    async _onError(e) {
        this.page$.set('alive', false);
        this.handleError('init', e);
    }

    async cleanup() {
        try {
            await this.onBeforeCleanup();
            await this._cleanup();
            await this.onCleanup();
            await this.onAfterCleanup();
        } catch (e) {
            await this._onError(e);
            await this.onError(e);
        } finally {
            await this._afterCleanup();
        }
    }

    async _afterCleanup() {
        this._initialized = false;
        if (!this.page$) return;
        this.initialized = false;
        this.page$.clear();
        this.page$.dispose();
        this.page$ = null;
        this.context = null;
    }

    async _cleanup() {
        // Stop eager column trackers
        if (this._eagerColTrackers) {
            for (let i = 0; i < this._eagerColTrackers.length; i++) {
                try { this._eagerColTrackers[i](); } catch {}
            }
            this._eagerColTrackers.length = 0;
        }

        try {
            this.toastManager().clearAllToasts();
        } catch (e) { console.error('Cleanup clearAllToasts failed:', e); }

        try {
            this.removeAllMessageHandlers();
            for (const r of this._rooms) {
                const sm = this.subscriptionManager();
                try { await sm?.unsubscribeFromRoom(r); } catch (e) { console.error('Cleanup room unsubscribe failed:', e); }
            }
            this._rooms.clear();
        } catch (e) { console.error('Cleanup message handlers/rooms failed:', e); }

        try {
            if (this.context?.destroy$) {
                this.context.destroy$.next();
                this.context.destroy$.complete();
            }
        } catch (e) { console.error('Cleanup destroy$ failed:', e); }

        try {
            this._listeners.forEach(c => { try { c.abort(); } catch {} });
            this._listeners.clear();
            this._storageListeners.clear();
        } catch (e) { console.error('Cleanup DOM listeners failed:', e); }

        try {
            this.clearTimers();
            for (const [k, id] of this._rafMap) { try { cancelAnimationFrame(id); } catch {} }
            this._rafMap.clear();
        } catch (e) { console.error('Cleanup timers failed:', e); }

        try {
            this._subscriptions.forEach(s => { try { s.unsubscribe(); } catch {} });
            this._subscriptions = [];
        } catch (e) { console.error('Cleanup subscriptions failed:', e); }

        try {
            this._stores.forEach(({ dict, subs }) => {
                subs.forEach(s => { try { s.unsubscribe(); } catch {} });
                if (dict && typeof dict.dispose === 'function') { try { dict.dispose(); } catch {} }
            });
            this._stores.clear();
        } catch (e) { console.error('Cleanup stores failed:', e); }

        try {
            this._sharedKeys.forEach(k => this._disposeShared(k));
            this._sharedKeys.clear();
            this._sharedContextKeys.forEach(k => this.destroySharedContext(k));
            this._sharedContextKeys.clear();
        } catch (e) { console.error('Cleanup shared stores/contexts failed:', e); }

        try {
            this._breakpoints.forEach(bp => { try { bp.destroy?.(); } catch {} });
            this._breakpoints.clear();
        } catch (e) { console.error('Cleanup breakpoints failed:', e); }

        try {
            if (this.emitter) {
                this.emitter.clear();
                this.emitter?.destroy();
            }
        } catch (e) { console.error('Cleanup emitter failed:', e); }
    }
}


export class EagerColumnCache {
    constructor(pageName) {
        this._pageName = pageName;
    }

    _key(gridId) { return `eagerCols:${this._pageName}:${gridId}`; }

    _read(gridId) {
        try {
            const raw = sessionStorage.getItem(this._key(gridId));
            return raw ? JSON.parse(raw) : [];
        } catch { return []; }
    }

    _write(gridId, cols) {
        try {
            sessionStorage.setItem(this._key(gridId), JSON.stringify(cols));
        } catch {}
    }

    /** Return the cached column list for a grid. */
    list(gridId) { return this._read(gridId); }

    /** Add columns (de-duplicated) to the cache for a grid. */
    add(gridId, columns) {
        const set = new Set(this._read(gridId));
        for (let i = 0; i < columns.length; i++) set.add(columns[i]);
        this._write(gridId, Array.from(set));
    }

    /** Remove specific columns from the cache for a grid. */
    remove(gridId, columns) {
        const rm = new Set(columns);
        this._write(gridId, this._read(gridId).filter(c => !rm.has(c)));
    }

    /** Clear all cached columns for a grid. */
    clear(gridId) {
        try { sessionStorage.removeItem(this._key(gridId)); } catch {}
    }

    /** Merge cached columns into a base column set. Returns de-duplicated array or null. */
    mergeWith(gridId, baseColumns) {
        const cached = this._read(gridId);
        if (!cached.length) return baseColumns;
        if (!baseColumns) return null; // null = "send everything", no point adding more
        const set = new Set(baseColumns);
        for (let i = 0; i < cached.length; i++) set.add(cached[i]);
        return Array.from(set);
    }
}

window.PageBase = PageBase;
export default PageBase
