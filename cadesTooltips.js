
import {writeStringToClipboard} from '@/utils/clipboardHelpers.js';

function ensure_list(x) { return Array.isArray(x) ? x : [x]; }

export class CadesTooltips {
    static VERSION = '7.0.0';

    static THEMES = {
        dark: {
            bg: 'linear-gradient(145deg, #1e2028 0%, #14161c 100%)',
            bgSolid: '#191b21',
            text: '#f0f4fc',
            textMuted: 'rgba(240, 244, 252, 0.65)',
            border: 'rgba(255, 255, 255, 0.08)',
            shadow: '0 20px 60px -15px rgba(0,0,0,0.6), 0 8px 20px -10px rgba(0,0,0,0.4)',
            radius: '14px'
        },
        light: {
            bg: 'linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)',
            bgSolid: '#ffffff',
            text: '#1e293b',
            textMuted: 'rgba(30, 41, 59, 0.65)',
            border: 'rgba(0, 0, 0, 0.08)',
            shadow: '0 20px 60px -15px rgba(0,0,0,0.15), 0 8px 20px -10px rgba(0,0,0,0.1)',
            radius: '14px'
        }
    };

    static ANIMATIONS = {
        'shift-away': {
            in: {
                top: [{transform: 'translateY(-10px)', opacity: 0}, {transform: 'translateY(0)', opacity: 1}],
                bottom: [{transform: 'translateY(10px)', opacity: 0}, {transform: 'translateY(0)', opacity: 1}],
                left: [{transform: 'translateX(-10px)', opacity: 0}, {transform: 'translateX(0)', opacity: 1}],
                right: [{transform: 'translateX(10px)', opacity: 0}, {transform: 'translateX(0)', opacity: 1}]
            },
            options: {duration: 200, easing: 'cubic-bezier(0.16, 1, 0.3, 1)'}
        },
        'shift-toward': {
            in: {
                top: [{transform: 'translateY(10px)', opacity: 0}, {transform: 'translateY(0)', opacity: 1}],
                bottom: [{transform: 'translateY(-10px)', opacity: 0}, {transform: 'translateY(0)', opacity: 1}],
                left: [{transform: 'translateX(10px)', opacity: 0}, {transform: 'translateX(0)', opacity: 1}],
                right: [{transform: 'translateX(-10px)', opacity: 0}, {transform: 'translateX(0)', opacity: 1}]
            },
            options: {duration: 200, easing: 'cubic-bezier(0.16, 1, 0.3, 1)'}
        },
        scale: {
            in: [{transform: 'scale(0.8)', opacity: 0}, {transform: 'scale(1)', opacity: 1}],
            options: {duration: 150, easing: 'cubic-bezier(0.16, 1, 0.3, 1)'}
        },
        perspective: {
            in: {
                top: [{transform: 'perspective(700px) translateY(-10px) rotateX(60deg)', opacity: 0}, {transform: 'perspective(700px) translateY(0) rotateX(0)', opacity: 1}],
                bottom: [{transform: 'perspective(700px) translateY(10px) rotateX(-60deg)', opacity: 0}, {transform: 'perspective(700px) translateY(0) rotateX(0)', opacity: 1}],
                left: [{transform: 'perspective(700px) translateX(-10px) rotateY(-60deg)', opacity: 0}, {transform: 'perspective(700px) translateX(0) rotateY(0)', opacity: 1}],
                right: [{transform: 'perspective(700px) translateX(10px) rotateY(60deg)', opacity: 0}, {transform: 'perspective(700px) translateX(0) rotateY(0)', opacity: 1}]
            },
            options: {duration: 250, easing: 'cubic-bezier(0.16, 1, 0.3, 1)'}
        },
        fade: {
            in: [{opacity: 0}, {opacity: 1}],
            options: {duration: 150, easing: 'ease-out'}
        }
    };

    constructor(context, options = {}) {
        this._instanceId = 'ctt_' + (++CadesTooltips._instanceCounter) + '_';
        this.context = context;
        this._opts = {
            zIndex: options.zIndex ?? 99996,
            distance: options.distance ?? 8,
            padding: options.padding ?? 2,
            showDelay: options.showDelay ?? 100,
            hideDelay: options.hideDelay ?? 100,
            theme: options.theme || 'auto',
            animation: options.animation || 'shift-away',
            maxWidth: options.maxWidth || '600px',
            maxHeight: options.maxHeight || '400px',
            followCursor: options.followCursor ?? false,
            spotlight: options.spotlight ?? false,
            debug: options.debug ?? false,
            maxPopWindows: options.maxPopWindows ?? 1,
            singleton: options.singleton ?? false,
            appendTo: options.appendTo || null,

            flashDefaults: {
                color: options.flashColor || 'var(--teal-500)',
                opacity: options.flashOpacity ?? 0.5,
                spread: options.flashSpread ?? 12,
                duration: options.flashDuration ?? 1400,
                easing: options.flashEasing || 'cubic-bezier(0.2, 0.85, 0.2, 1)',
                iterations: options.flashIterations ?? 'infinite',
                interval: options.flashInterval ?? 0,
                once: options.flashOnce ?? false
            },

            flashStorageKey: options.flashStorageKey || 'ctt-flash-dismissed',

            hintDefaults: {
                hideOnOutOfView: options.hintHideOnOutOfView ?? true,
                positionStrategy: options.hintPositionStrategy || 'absolute'
            }
        };
        if (Array.isArray(options.delay)) {
            this._opts.showDelay = options.delay[0] ?? 0;
            this._opts.hideDelay = options.delay[1] ?? 10;
        } else if (typeof options.delay === 'number') {
            this._opts.showDelay = options.delay;
            this._opts.hideDelay = options.delay;
        }

        this._opts.contextMenu = this._normalizeContextMenu(options.contextMenu, true);
        this._opts.popDefaults = this._normalizePopConfig(options.popDefaults || options.popWindow || options.pop, true);

        this._tooltips = new Map();
        this._elementMap = new WeakMap();
        this._groups = new Map();

        this._active = null;
        this._activeEl = null;
        this._activePlacement = 'top';
        this._activeTrigger = '';
        this._activeVirtualRect = null;
        this._locked = false;
        this._lockedMode = 'none';
        this._lockedHidden = false;
        this._resolvedTheme = null; // Cached resolved theme for 'auto' mode

        this._hover = {target: false, tooltip: false, bridge: false};
        this._focus = false;

        this._cursorPos = {x: 0, y: 0};
        this._followRAF = 0;

        this._timers = {show: 0, hide: 0, copy: 0};
        this._rafId = 0;

        this._bubble = null;
        this._hidingBubble = null;
        this._panel = null;
        this._inner = null;
        this._caret = null;
        this._bridge = null;
        this._spotlight = null;
        this._currentAnimation = null;

        this._connectedTargets = new WeakSet();


        this._singletonPrev = null;

        this._pins = new Map();
        this._pinBubbleMap = new WeakMap();

        this._hints = [];
        this._hintMarks = new WeakMap();
        this._hintByTarget = new WeakMap();
        this._hintObserver = null;

        this._ctx = {open: false, el: null, items: [], tooltipId: null, scope: '', x: 0, y: 0, targetEl: null};
        this._ctxRegistry = new Map();
        this._registerBuiltInContextMenuItems();

        this._pops = new Map();
        this._popElMap = new WeakMap();
        this._popDrag = null;
        this._popCountByTooltip = new Map();

        this._tour = {active: false, steps: [], index: -1};

        this._listeners = new Map();
        this._flashRules = new Map();
        this._zStack = this._opts.zIndex;

        this._destroyed = false;

        this._resizeObserver = null;
        this._resizeObservedEl = null;

        this._mutationObserver = null;
        this._initMutationObserver();

        this._deferredQueue = [];
        this._deferredScheduled = false;

        this._bindHandlers();
        this._attachListeners();

        if (this._opts.debug) {
            console.log(`[CadesTooltips:${this._instanceId}] Initialized`, this._opts);
        }

        this._themeUnsub = this.context.page.global$.get("theme").onChanges(ch => {
            this.refreshTheme(ch.current.get("theme"))
        })
    }

    static _instanceCounter = 0;

    _getContainer() {
        return this._opts.appendTo instanceof Element ? this._opts.appendTo : document.body;
    }

    _getScrollParent(el) {
        if (!el || el === document.body || el === document.documentElement) return null;
        let cur = el.parentElement;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            const style = getComputedStyle(cur);
            const overflow = style.overflow + style.overflowX + style.overflowY;
            if (/auto|scroll|overlay/.test(overflow)) return cur;
            cur = cur.parentElement;
        }
        return null;
    }

    _getClipRect(el) {
        const scrollParent = this._getScrollParent(el);
        if (!scrollParent) {
            return {top: 0, left: 0, bottom: window.innerHeight, right: window.innerWidth};
        }
        const r = scrollParent.getBoundingClientRect();
        return {
            top: Math.max(0, r.top),
            left: Math.max(0, r.left),
            bottom: Math.min(window.innerHeight, r.bottom),
            right: Math.min(window.innerWidth, r.right)
        };
    }

    _isInClipRect(el) {
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        const clip = this._getClipRect(el);
        return rect.bottom > clip.top && rect.top < clip.bottom &&
            rect.right > clip.left && rect.left < clip.right;
    }

    _observeTargetResize(el) {
        if (!el || !('ResizeObserver' in window)) return;
        if (this._resizeObservedEl === el) return;
        this._unobserveTargetResize();
        if (!this._resizeObserver) {
            this._resizeObserver = new ResizeObserver(() => {
                if (this._active && this._bubble && this._activeEl) {
                    this._positionActive();
                }
            });
        }
        this._resizeObserver.observe(el);
        this._resizeObservedEl = el;
    }

    _unobserveTargetResize() {
        if (this._resizeObservedEl && this._resizeObserver) {
            try {
                this._resizeObserver.unobserve(this._resizeObservedEl);
            } catch (_) {
            }
        }
        this._resizeObservedEl = null;
    }

    _initMutationObserver() {
        if (!('MutationObserver' in window)) return;
        this._mutationObserver = new MutationObserver(mutations => {
            if (this._destroyed) return;
            let hasRemovals = false;
            for (const m of mutations) {
                if (m.removedNodes.length) {
                    hasRemovals = true;
                    break;
                }
            }
            if (!hasRemovals) return;

            for (const [, tooltip] of this._tooltips) {
                for (const t of tooltip.targets) {
                    if (t.isConnected) this._connectedTargets.add(t);
                }
            }

            if (this._active && this._activeEl && !this._activeEl.isConnected) {
                this._hideTooltip(true);
            }
            for (const inst of [...this._pins.values()]) {
                if (inst.target && !inst.target.isConnected) this.unpin(inst.id);
            }
            for (const [id, tooltip] of this._tooltips) {
                const wasEverConnected = tooltip.targets.some(t => this._connectedTargets.has(t));
                if (wasEverConnected && tooltip.targets.every(t => !t.isConnected)) {
                    this.remove(id);
                }
            }
        });
        this._mutationObserver.observe(document.body, {childList: true, subtree: true});
    }

    _deferSetup(fn) {
        this._deferredQueue.push(fn);
        if (this._deferredScheduled) return;
        this._deferredScheduled = true;
        const schedule = typeof requestIdleCallback === 'function'
            ? cb => requestIdleCallback(cb, {timeout: 200})
            : cb => setTimeout(cb, 1);
        schedule(() => {
            this._deferredScheduled = false;
            const batch = this._deferredQueue.splice(0);
            for (const task of batch) {
                if (!this._destroyed) task();
            }
        });
    }

    add(config) {
        if (this._destroyed) {
            console.error('tt is destroyed')
            return null;
        }
        if (!config || typeof config !== 'object') {
            console.error('tt config is none')
            return null;
        }

        config.id = config.id || this._generateId();
        const id = config.id;
        const targets = this._resolveTargets(config.target ?? config.targets);
        if (!targets.length) {
            console.error('[CadesTooltips] No valid targets for:', config.target ?? config.targets);
            return null;
        }

        let showDelay = config.showDelay ?? null;
        let hideDelay = config.hideDelay ?? null;
        if (Array.isArray(config.delay)) {
            showDelay = config.delay[0] ?? null;
            hideDelay = config.delay[1] ?? null;
        } else if (typeof config.delay === 'number') {
            showDelay = config.delay;
            hideDelay = config.delay;
        }

        const tooltip = {
            id,
            targets,
            enabled: config.enabled !== false,

            content: config.render ?? config.content ?? config.text ?? "",
            html: config.html === true || config.contentIsHTML === true,

            placement: this._normalizePlacement(config.placement || config.direction || 'top'),
            offset: config.offset ?? this._opts.distance,
            flip: config.flip !== false,
            arrow: config.arrow !== false,
            fitContent: config.fitContent ?? false,

            theme: config.theme || this._opts.theme,
            themeClass: config.themeClass || '',
            themeColors: config.themeColors || null,
            animation: config.animation || this._opts.animation,
            animationOut: config.animationOut || null,
            animationInKeyframe: config.animationInKeyframe || null,
            animationOutKeyframe: config.animationOutKeyframe || null,
            animationDuration: config.animationDuration ?? null,
            maxWidth: config.maxWidth || this._opts.maxWidth,
            maxHeight: config.maxHeight || this._opts.maxHeight,

            showDelay,
            hideDelay,

            trigger: String(config.trigger || 'hover focus').toLowerCase(),
            hideOnClick: config.hideOnClick ?? true,
            hideOnEsc: config.hideOnEsc ?? true,
            interactive: config.interactive ?? false,
            followCursor: config.followCursor ?? false,
            sticky: config.sticky ?? false,
            windowOnly: config.windowOnly ?? false,
            singleton: config.singleton ?? null,

            lock: config.lock ?? config.locked ?? false,

            title: config.title || '',

            contextMenu: this._normalizeContextMenu(config.contextMenu),
            pop: this._normalizePopConfig(config.popWindow || config.pop),
            maxPopWindows: config.maxPopWindows ?? null,

            virtual: config.virtual ?? false,

            spotlight: config.spotlight ?? false,
            spotlightPadding: config.spotlightPadding ?? 8,
            className: config.className ?? null,

            media: config.media || null,
            progress: config.progress ?? null,
            actions: Array.isArray(config.actions) ? config.actions : [],

            flash: this._normalizeFlash(config),
            flashStorage: config.flashStorage ?? false,
            hint: this._normalizeHint(config.hint),

            group: config.group || null,

            onShow: config.onShow || null,
            onHide: config.onHide || null,
            onAction: config.onAction || null,
            onClickOutside: config.onClickOutside || null,

            _flashClass: `ctt-flash-${id.replace(/[^a-z0-9]/gi, '')}`,
            _flashApplied: false,
            _hintShown: false,
            _popped: false,
            _enabledBeforePop: null,
            _popCount: 0
        };

        if (tooltip.className) {
            tooltip.className = Array.isArray(tooltip.className) ? tooltip.className : [tooltip.className];
        }

        this._tooltips.set(id, tooltip);

        for (const el of targets) {
            this._elementMap.set(el, id);
        }

        if (tooltip.group) {
            if (!this._groups.has(tooltip.group)) this._groups.set(tooltip.group, new Set());
            this._groups.get(tooltip.group).add(id);
        }

        if (tooltip.flash && tooltip.flash.enabled && tooltip.enabled) {
            this._applyFlash(tooltip);
        }
        if (tooltip.hint && tooltip.hint.enabled && tooltip.enabled) {
            this._deferSetup(() => this._createHints(tooltip));
        }

        if (tooltip.windowOnly) {
            tooltip.targets.forEach(t => {
                t.classList.toggle('tooltip-window-only', true);
            })
        }

        this._emit('add', {id, tooltip});
        if (this._opts.debug) {
            console.log(`[CadesTooltips:${this._instanceId}] Added:`, id);
        }

        return new CadesTooltipHandle(this, id);
    }

    addMany(configs) {
        if (this._destroyed || !Array.isArray(configs)) return [];
        const handles = [];
        const flashBatch = [];
        const hintBatch = [];

        for (const config of configs) {
            if (!config || typeof config !== 'object') continue;

            config.id = config.id || this._generateId();
            const id = config.id;
            const targets = this._resolveTargets(config.target ?? config.targets);
            if (!targets.length) continue;

            let sd = config.showDelay ?? null;
            let hd = config.hideDelay ?? null;
            if (Array.isArray(config.delay)) {
                sd = config.delay[0] ?? null;
                hd = config.delay[1] ?? null;
            } else if (typeof config.delay === 'number') {
                sd = config.delay;
                hd = config.delay;
            }

            const tooltip = {
                id, targets, enabled: config.enabled !== false,
                content: config.render ?? config.content ?? config.text ?? "",
                html: config.html === true || config.contentIsHTML === true,
                placement: this._normalizePlacement(config.placement || config.direction || 'top'),
                offset: config.offset ?? this._opts.distance,
                flip: config.flip !== false,
                arrow: config.arrow !== false,
                fitContent: config.fitContent ?? false,
                theme: config.theme || this._opts.theme,
                themeClass: config.themeClass || '',
                themeColors: config.themeColors || null,
                animation: config.animation || this._opts.animation,
                animationOut: config.animationOut || null,
                animationInKeyframe: config.animationInKeyframe || null,
                animationOutKeyframe: config.animationOutKeyframe || null,
                animationDuration: config.animationDuration ?? null,
                maxWidth: config.maxWidth || this._opts.maxWidth,
                maxHeight: config.maxHeight || this._opts.maxHeight,
                showDelay: sd, hideDelay: hd,
                trigger: String(config.trigger || 'hover focus').toLowerCase(),
                hideOnClick: config.hideOnClick ?? true,
                hideOnEsc: config.hideOnEsc ?? true,
                interactive: config.interactive ?? false,
                followCursor: config.followCursor ?? false,
                sticky: config.sticky ?? false,
                windowOnly: config.windowOnly ?? false,
                singleton: config.singleton ?? null,
                lock: config.lock ?? config.locked ?? false,
                title: config.title || '',
                contextMenu: this._normalizeContextMenu(config.contextMenu),
                pop: this._normalizePopConfig(config.popWindow || config.pop),
                maxPopWindows: config.maxPopWindows ?? null,
                virtual: config.virtual ?? false,
                spotlight: config.spotlight ?? false,
                spotlightPadding: config.spotlightPadding ?? 8,
                className: config.className ?? null,
                media: config.media || null,
                progress: config.progress ?? null,
                actions: Array.isArray(config.actions) ? config.actions : [],
                flash: this._normalizeFlash(config),
                flashStorage: config.flashStorage ?? false,
                hint: this._normalizeHint(config.hint),
                group: config.group || null,
                onShow: config.onShow || null,
                onHide: config.onHide || null,
                onAction: config.onAction || null,
                onClickOutside: config.onClickOutside || null,
                _flashClass: `ctt-flash-${id.replace(/[^a-z0-9]/gi, '')}`,
                _flashApplied: false, _hintShown: false,
                _popped: false, _enabledBeforePop: null, _popCount: 0
            };

            if (tooltip.className) {
                tooltip.className = Array.isArray(tooltip.className) ? tooltip.className : [tooltip.className];
            }
            this._tooltips.set(id, tooltip);
            for (const el of targets) this._elementMap.set(el, id);
            if (tooltip.group) {
                if (!this._groups.has(tooltip.group)) this._groups.set(tooltip.group, new Set());
                this._groups.get(tooltip.group).add(id);
            }
            if (tooltip.flash.enabled && tooltip.enabled) {
                flashBatch.push(tooltip);
            }

            if (tooltip.hint.enabled && tooltip.hint.showOnLoad && tooltip.enabled) {
                hintBatch.push(tooltip);
            }

            if (tooltip.windowOnly) {
                tooltip.targets.forEach(t => t.classList.add('tooltip-window-only'));
            }
            handles.push(new CadesTooltipHandle(this, id));
        }

        if (flashBatch.length || hintBatch.length) {
            this._deferSetup(() => {
                for (const t of flashBatch) this._applyFlash(t);
                for (const t of hintBatch) this._createHints(t);
                this._syncFlashStyleElement();
            });
        }

        return handles;
    }

    remove(id) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip) return false;

        if (this._active?.id === id) this.hide(true);
        this.unpin(id);
        this._collapseAllPopWindows(id);

        this._removeFlash(tooltip);
        this._removeFlashRule(tooltip._flashClass);
        this._removeHints(tooltip.id);

        if (tooltip.group && this._groups.has(tooltip.group)) {
            this._groups.get(tooltip.group).delete(id);
        }

        for (const el of tooltip.targets) {
            if (this._elementMap.get(el) === id) this._elementMap.delete(el);
        }

        this._tooltips.delete(id);
        this._emit('remove', {id});
        return true;
    }

    update(id, config) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip || !config) return false;

        if (config.content !== undefined || config.render !== undefined) {
            tooltip.content = config.render ?? config.content ?? tooltip.content;
        }
        if (config.html !== undefined) tooltip.html = config.html === true;
        if (config.title !== undefined) tooltip.title = config.title;
        if (config.theme !== undefined) tooltip.theme = config.theme;
        if (config.themeClass !== undefined) tooltip.themeClass = config.themeClass;
        if (config.themeColors !== undefined) tooltip.themeColors = config.themeColors;
        if (config.animation !== undefined) tooltip.animation = config.animation;
        if (config.animationOut !== undefined) tooltip.animationOut = config.animationOut;
        if (config.animationInKeyframe !== undefined) tooltip.animationInKeyframe = config.animationInKeyframe;
        if (config.animationOutKeyframe !== undefined) tooltip.animationOutKeyframe = config.animationOutKeyframe;
        if (config.placement !== undefined) tooltip.placement = this._normalizePlacement(config.placement);
        if (config.offset !== undefined) tooltip.offset = config.offset;
        if (config.flip !== undefined) tooltip.flip = config.flip !== false;
        if (config.spotlight !== undefined) tooltip.spotlight = config.spotlight;
        if (config.interactive !== undefined) tooltip.interactive = config.interactive;
        if (config.followCursor !== undefined) tooltip.followCursor = config.followCursor;
        if (config.sticky !== undefined) tooltip.sticky = config.sticky;
        if (config.lock !== undefined || config.locked !== undefined) tooltip.lock = (config.lock ?? config.locked) ?? false;
        if (config.contextMenu !== undefined) tooltip.contextMenu = this._normalizeContextMenu(config.contextMenu);
        if (config.popWindow !== undefined || config.pop !== undefined) tooltip.pop = this._normalizePopConfig(config.popWindow || config.pop);
        if (config.virtual !== undefined) tooltip.virtual = config.virtual;
        if (config.progress !== undefined) tooltip.progress = config.progress;
        if (config.actions !== undefined) tooltip.actions = Array.isArray(config.actions) ? config.actions : [];
        if (config.hint !== undefined) tooltip.hint = this._normalizeHint(config.hint);
        if (config.flashStorage !== undefined) tooltip.flashStorage = config.flashStorage;
        if (config.showDelay !== undefined) tooltip.showDelay = config.showDelay;
        if (config.hideDelay !== undefined) tooltip.hideDelay = config.hideDelay;
        if (config.delay !== undefined) {
            if (Array.isArray(config.delay)) {
                tooltip.showDelay = config.delay[0] ?? null;
                tooltip.hideDelay = config.delay[1] ?? null;
            } else if (typeof config.delay === 'number') {
                tooltip.showDelay = config.delay;
                tooltip.hideDelay = config.delay;
            }
        }
        if (config.windowOnly !== undefined) tooltip.windowOnly = config.windowOnly;
        if (config.maxPopWindows !== undefined) tooltip.maxPopWindows = config.maxPopWindows;
        if (config.arrow !== undefined) tooltip.arrow = config.arrow;
        if (config.fitContent !== undefined) tooltip.fitContent = config.fitContent;
        if (config.hideOnEsc !== undefined) tooltip.hideOnEsc = config.hideOnEsc;
        if (config.onClickOutside !== undefined) tooltip.onClickOutside = config.onClickOutside;
        if (config.animationDuration !== undefined) tooltip.animationDuration = config.animationDuration;
        if (config.singleton !== undefined) tooltip.singleton = config.singleton;
        if (config.flash != null) {

            tooltip.flash = this._normalizeFlash(config, tooltip.flash);

            if (tooltip?.flash?.enabled && tooltip.enabled) {
                this.flash(id, {flash:tooltip.flash});
            } else {
                this._removeFlash(tooltip);
            }

        }

        if (this._active?.id === id && this._bubble) {
            if (config.content !== undefined || config.render !== undefined) {
                this._resolveAndRenderContent(this._inner, tooltip, this._activeEl);
            } else {
                this._renderContentInto(this._inner, tooltip);
            }
            this._applyThemeTo(this._bubble, this._panel, tooltip.theme, tooltip);
            this._applySizingTo(this._panel, tooltip);
            if (tooltip.progress != null) this._updateProgressIn(this._inner, tooltip.progress);

            this._measureAndPositionActive();
        }

        const pin = this._pins.get(id);
        if (pin) {
            this._resolveAndRenderContent(pin.inner, tooltip, pin.target);
            this._applyThemeTo(pin.bubble, pin.panel, tooltip.theme, tooltip);
            this._applySizingTo(pin.panel, tooltip);
            if (tooltip.progress != null) this._updateProgressIn(pin.inner, tooltip.progress);
            this._measureAndPositionPinned(pin);
        }

        this._emit('update', {id, config});
        return true;
    }

    show(id, options = {}) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip || !tooltip.enabled) return false;

        const target = options.target instanceof Element ? options.target : tooltip.targets[0];
        if (!target?.isConnected) return false;

        if (options.pin) {
            this._showPinnedTooltip(tooltip, target, {lock: options.lock ?? true, trigger: 'api'});
            return true;
        }

        if (options.lock) {
            this._locked = true;
            this._lockedMode = 'tooltip';
        }
        if (options.virtualPoint && typeof options.virtualPoint.x === 'number' && typeof options.virtualPoint.y === 'number') {
            this._activeVirtualRect = this._virtualRectFromPoint(options.virtualPoint.x, options.virtualPoint.y);
        } else {
            this._activeVirtualRect = null;
        }

        this._showTooltip(tooltip, target, 'api');
        return true;
    }

    hide(force = false) {
        if (!this._active) return false;
        if (this._locked && !force) return false;
        this._hideTooltip();
        return true;
    }

    toggle(id, options = {}) {
        if (options.pin) {
            const pinned = this._pins.has(id);
            if (pinned) return this.unpin(id);
            return this.show(id, options);
        }
        if (this._active?.id === id) return this.hide(true);
        return this.show(id, options);
    }


    lock(locked = true, mode = 'tooltip') {
        this._locked = !!locked;
        this._lockedMode = this._locked ? (mode === 'outside' ? 'outside' : 'tooltip') : 'none';
        return this;
    }

    unlock() {
        this._locked = false;
        this._lockedMode = 'none';
        return this;
    }


    registerContextMenuItem(item) {
        if (!item || typeof item !== 'object') return false;
        const id = String(item.id || '').trim();
        if (!id) return false;

        this._ctxRegistry.set(id, {
            id,
            label: String(item.label || id),
            icon: item.icon == null ? '' : String(item.icon),
            enabled: typeof item.enabled === 'function' ? item.enabled : null,
            handler: typeof item.handler === 'function' ? item.handler : null
        });

        return true;
    }

    isPopped(id) {
        return this._countPopWindows(id) > 0;
    }

    pop(id, options = {}) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip) return false;

        const max = tooltip.maxPopWindows ?? this._opts.maxPopWindows;
        const openCount = this._countPopWindows(id);

        const existing = this._pops.get(id);
        if (existing) {
            if (openCount >= max) {
                this._focusPop(existing);
                return true;
            }
        }

        if (openCount >= max) return false;

        const target = options.target instanceof Element ? options.target : tooltip.targets[0];
        if (!target?.isConnected) return false;

        this._closeContextMenu();

        if (this._active?.id === id) this.hide(true);
        this.unpin(id);

        if (!tooltip._popped) {
            tooltip._popped = true;
            tooltip._enabledBeforePop = tooltip.enabled;
            tooltip.enabled = false;
            this._setPoppedTargets(tooltip, true);
            this._removeFlash(tooltip);
            this._removeHints(id);
        }

        const popKey = max > 1 ? `${id}__pop_${++tooltip._popCount}` : id;

        const inst = this._createPopInstance(tooltip, target, options, popKey);
        inst.popKey = popKey;
        inst.tooltipId = id;
        this._pops.set(popKey, inst);
        this._emit('pop', {id, target, tooltip});

        return true;
    }

    unpop(id) {
        const inst = this._pops.get(id);
        if (!inst) return false;

        this._closeContextMenu();
        this._endPopDrag(true);

        try {
            if (inst.el?.parentNode) inst.el.parentNode.removeChild(inst.el);
        } catch (_) {
        }

        try {
            this._popElMap.delete(inst.el);
        } catch (_) {
        }
        this._pops.delete(id);

        const tooltipId = inst.tooltipId || id;
        const tooltip = this._tooltips.get(tooltipId);
        if (tooltip) {
            const remainingCount = this._countPopWindows(tooltipId);
            if (remainingCount === 0) {
                const restoreEnabled = tooltip._enabledBeforePop;
                tooltip._enabledBeforePop = null;
                tooltip._popped = false;
                tooltip.enabled = restoreEnabled !== null ? !!restoreEnabled : true;
                this._setPoppedTargets(tooltip, false);

                if (tooltip.enabled) {
                    if (tooltip.flash.enabled) this._applyFlash(tooltip);
                    if (tooltip.hint.enabled && tooltip.hint.showOnLoad) this._createHints(tooltip);
                }
            }
        }

        this._emit('unpop', {id: tooltipId});
        return true;
    }


    enable(id) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip) return false;

        if (tooltip._popped) {
            tooltip._enabledBeforePop = true;
            return true;
        }

        tooltip.enabled = true;
        if (tooltip.flash.enabled) this._applyFlash(tooltip);
        if (tooltip.hint.enabled && tooltip.hint.showOnLoad) this._createHints(tooltip);
        return true;
    }


    disable(id) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip) return false;

        if (tooltip._popped) {
            tooltip._enabledBeforePop = false;
            return true;
        }

        tooltip.enabled = false;
        if (this._active?.id === id) this.hide(true);
        this.unpin(id);
        this._removeFlash(tooltip);
        this._removeHints(id);
        return true;
    }


    has(id) {
        return this._tooltips.has(id);
    }

    get(id) {
        return this._tooltips.get(id) || null;
    }

    getAll() {
        return Array.from(this._tooltips.keys());
    }

    getActive() {
        if (!this._active) return null;
        return {id: this._active.id, target: this._activeEl, tooltip: this._active};
    }

    getFromElement(el) {
        return this._elementMap.get(el) || null;
    }

    pin(id, options = {}) {
        return this.show(id, {...options, pin: true});
    }

    unpin(id) {
        const inst = this._pins.get(id);
        if (!inst) return false;
        this._removePinnedInstance(inst);
        this._pins.delete(id);
        this._emit('unpin', {id});
        return true;
    }

    getPinned() {
        return Array.from(this._pins.keys());
    }

    showGroup(groupName, options = {}) {
        const group = this._groups.get(groupName);
        if (!group) return false;

        for (const id of group) {
            this.pin(id, {lock: options.lock ?? true});
        }
        return true;
    }

    hideGroup(groupName) {
        const group = this._groups.get(groupName);
        if (!group) return false;

        for (const id of group) {
            this.unpin(id);
        }
        return true;
    }

    getGroup(groupName) {
        const group = this._groups.get(groupName);
        return group ? Array.from(group) : [];
    }

    startTour(steps) {
        if (this._tour.active) this.endTour();

        this._tour.steps = Array.isArray(steps) ? steps.filter(s => s && s.target && s.content) : [];
        if (!this._tour.steps.length) return false;

        this._tour.active = true;
        this._tour.index = -1;
        this._locked = true;
        this._lockedMode = 'tour';

        this._emit('tourStart', {steps: this._tour.steps});
        return this.nextStep();
    }

    nextStep() {
        if (!this._tour.active) return false;
        this._tour.index++;
        if (this._tour.index >= this._tour.steps.length) {
            this.endTour();
            return false;
        }
        this._showTourStep(this._tour.index);
        return true;
    }

    prevStep() {
        if (!this._tour.active || this._tour.index <= 0) return false;
        this._tour.index--;
        this._showTourStep(this._tour.index);
        return true;
    }

    goToStep(index) {
        if (!this._tour.active || index < 0 || index >= this._tour.steps.length) return false;
        this._tour.index = index;
        this._showTourStep(index);
        return true;
    }

    endTour() {
        if (!this._tour.active) return false;

        this._tour.active = false;
        this._tour.steps = [];
        this._tour.index = -1;
        this._locked = false;
        this._lockedMode = 'none';

        this.hide(true);
        this._hideSpotlight();

        this._emit('tourEnd');
        return true;
    }

    getTourProgress() {
        if (!this._tour.active) return null;
        return {
            active: true,
            current: this._tour.index,
            total: this._tour.steps.length,
            step: this._tour.steps[this._tour.index] || null
        };
    }

    on(event, callback) {
        if (typeof callback !== 'function') {
            return () => {
            };
        }
        if (!this._listeners.has(event)) this._listeners.set(event, new Set());
        this._listeners.get(event).add(callback);
        return () => this.off(event, callback);
    }

    off(event, callback) {
        const listeners = this._listeners.get(event);
        if (listeners) listeners.delete(callback);
        return this;
    }

    once(event, callback) {
        const wrapper = (...args) => {
            this.off(event, wrapper);
            callback(...args);
        };
        return this.on(event, wrapper);
    }

    setTheme(theme) {
        this._opts.theme = theme;
        if (this._active && this._bubble) this._applyThemeTo(this._bubble, this._panel, theme, this._active);
        return this;
    }

    refreshTheme(newTheme) {
        // Cache the resolved theme so all future tooltip shows use it,
        // even if no bubble exists right now.
        // - refreshTheme()        → re-detect from DOM
        // - refreshTheme('dark')  → force to 'dark'
        // - refreshTheme('light') → force to 'light'
        if (newTheme === 'dark' || newTheme === 'light') {
            this._resolvedTheme = newTheme;
        } else {
            this._resolvedTheme = this._detectThemeFromDOM();
        }

        const theme = this._opts.theme;

        // Active tooltip bubble
        if (this._active && this._bubble) {
            this._applyThemeTo(this._bubble, this._panel, theme, this._active);
        }

        // Pinned instances
        for (const inst of [...this._pins.values()]) {
            if (inst.bubble && inst.tooltip) {
                this._applyThemeTo(inst.bubble, inst.panel, inst.tooltip.theme || theme, inst.tooltip);
            }
        }

        // Pop windows
        for (const pop of this._pops.values()) {
            if (pop.el && pop.tooltip) {
                this._applyThemeTo(pop.el, pop.el.querySelector('.ctt-pop-body'), pop.tooltip.theme || theme, pop.tooltip);
            }
        }


        return this;
    }

    setAnimation(animation) {
        this._opts.animation = animation;
        return this;
    }

    static registerTheme(name, theme) {
        CadesTooltips.THEMES[name] = theme;
    }

    static registerAnimation(name, animation) {
        CadesTooltips.ANIMATIONS[name] = animation;
    }

    hideAll() {
        this.hide(true);
        for (const id of this._pins.keys()) this.unpin(id);
        this._removeAllHints();
        return this;
    }

    reset() {
        for (const [id, tooltip] of this._tooltips) {
            if (tooltip._popped) this._collapseAllPopWindows(id);
            if (tooltip.flash?.enabled) this._applyFlash(tooltip);
            this._removeHints(id);
            tooltip._hintShown = false;
            if (tooltip.hint.enabled && tooltip.hint.showOnLoad) this._createHints(tooltip);
        }
        return this;
    }

    reposition() {
        if (this._active && this._bubble) this._measureAndPositionActive();
        for (const inst of this._pins.values()) this._measureAndPositionPinned(inst);
        this._repositionHints(true);
        return this;
    }

    setContent(id, content) {
        return this.update(id, {content});
    }

    setProgress(id, progress) {
        return this.update(id, {progress});
    }

    destroy() {
        if (this._destroyed) return;
        this._destroyed = true;

        if (this._tour.active) this.endTour();

        this._hideTooltip(true);

        for (const inst of [...this._pins.values()]) {
            this._removePinnedInstance(inst);
        }
        this._pins.clear();

        this._endPopDrag(true);
        for (const inst of [...this._pops.values()]) {
            try {
                if (inst.el?.parentNode) inst.el.parentNode.removeChild(inst.el);
            } catch (_) {
            }
            try {
                this._popElMap.delete(inst.el);
            } catch (_) {
            }
        }
        this._pops.clear();
        this._popCountByTooltip.clear();

        for (const [, t] of this._tooltips) {
            if (t && t._popped) this._setPoppedTargets(t, false);
            if (t) {
                t._popped = false;
                t._enabledBeforePop = null;
            }
        }

        this._closeContextMenu(true);

        this._removeAllHints();

        for (const [, tooltip] of this._tooltips) {
            this._removeFlash(tooltip);
        }
        this._flashRules.clear();
        this._syncFlashStyleElement();

        this._clearTimers();
        if (this._copyOverlay) {
            this._copyOverlay.remove();
            this._copyOverlay = null;
        }
        this._removeBubble();
        if (this._spotlight?.parentNode) {
            this._spotlight.parentNode.removeChild(this._spotlight);
            this._spotlight = null;
        }
        this._detachListeners();

        if (this._themeUnsub) {
            try {
                this._themeUnsub();
            } catch (_) {
            }
            this._themeUnsub = null;
        }

        if (this._hintObserver) {
            try {
                this._hintObserver.disconnect();
            } catch (_) {
            }
            this._hintObserver = null;
        }

        this._unobserveTargetResize();
        if (this._resizeObserver) {
            try {
                this._resizeObserver.disconnect();
            } catch (_) {
            }
            this._resizeObserver = null;
        }

        if (this._mutationObserver) {
            try {
                this._mutationObserver.disconnect();
            } catch (_) {
            }
            this._mutationObserver = null;
        }


        this._deferredQueue.length = 0;

        this._tooltips.clear();
        this._groups.clear();

        this._emit('destroy');
        this._listeners.clear();
        if (this._opts.debug) {
            console.log(`[CadesTooltips:${this._instanceId}] Destroyed`);
        }
    }

    cleanup() {
        if (this._destroyed) return;

        // End tour if active
        if (this._tour.active) this.endTour();

        // Hide active tooltip
        this._hideTooltip(true);

        // Remove all pinned instances
        for (const inst of [...this._pins.values()]) {
            this._removePinnedInstance(inst);
        }
        this._pins.clear();

        // Close all pop windows
        this._endPopDrag(true);
        for (const inst of [...this._pops.values()]) {
            try {
                if (inst.el?.parentNode) inst.el.parentNode.removeChild(inst.el);
            } catch (_) {
            }
            try {
                this._popElMap.delete(inst.el);
            } catch (_) {
            }
        }
        this._pops.clear();
        this._popCountByTooltip.clear();

        // Reset popped state on all tooltips
        for (const [, t] of this._tooltips) {
            if (t && t._popped) this._setPoppedTargets(t, false);
            if (t) {
                t._popped = false;
                t._enabledBeforePop = null;
            }
        }

        // Close context menu
        this._closeContextMenu(true);

        // Remove all hints and flash
        this._removeAllHints();
        for (const [, tooltip] of this._tooltips) {
            this._removeFlash(tooltip);
        }
        this._flashRules.clear();
        this._syncFlashStyleElement();

        // Clear timers and bubble
        this._clearTimers();
        this._removeBubble();
        if (this._spotlight?.parentNode) {
            this._spotlight.parentNode.removeChild(this._spotlight);
            this._spotlight = null;
        }

        // Disconnect observers (will be re-created on demand)
        if (this._themeUnsub) {
            try {
                this._themeUnsub();
            } catch (_) {
            }
            this._themeUnsub = null;
        }

        if (this._hintObserver) {
            try {
                this._hintObserver.disconnect();
            } catch (_) {
            }
            this._hintObserver = null;
        }
        this._unobserveTargetResize();
        if (this._resizeObserver) {
            try {
                this._resizeObserver.disconnect();
            } catch (_) {
            }
            this._resizeObserver = null;
        }

        // Flush deferred queue
        this._deferredQueue.length = 0;

        // Clear all registered tooltips and groups
        this._tooltips.clear();
        this._groups.clear();
        this._elementMap = new WeakMap();
        this._connectedTargets = new WeakSet();

        // Reset active state
        this._active = null;
        this._activeEl = null;
        this._activeTrigger = '';
        this._activeVirtualRect = null;
        this._activePlacement = 'top';
        this._locked = false;
        this._lockedMode = 'none';
        this._lockedHidden = false;
        this._hover = {target: false, tooltip: false, bridge: false};
        this._focus = false;

        if (this._mutationObserver) {
            try {
                this._mutationObserver.disconnect();
            } catch (_) {
            }
            this._mutationObserver = null;
        }
        // Re-init MutationObserver so it's ready for new tooltips
        this._initMutationObserver();

        if (this._opts.debug) {
            console.log(`[CadesTooltips:${this._instanceId}] Cleaned up`);
        }
    }

    get isDestroyed() {
        return this._destroyed;
    }

    _bindHandlers() {
        this._onPointerOver = this._onPointerOver.bind(this);
        this._onPointerMove = this._onPointerMove.bind(this);
        this._onPointerOut = this._onPointerOut.bind(this);
        this._onContextMenu = this._onContextMenu.bind(this);
        this._onWindowMouseOut = this._onWindowMouseOut.bind(this);
        this._onWindowBlur = this._onWindowBlur.bind(this);
        this._onVisibilityChange = this._onVisibilityChange.bind(this);
        this._onPopPointerMove = this._onPopPointerMove.bind(this);
        this._onPopPointerUp = this._onPopPointerUp.bind(this);
        this._onPointerLeave = this._onPointerLeave.bind(this);
        this._onPointerDown = this._onPointerDown.bind(this);
        this._onFocusIn = this._onFocusIn.bind(this);
        this._onFocusOut = this._onFocusOut.bind(this);
        this._onKeyDown = this._onKeyDown.bind(this);
        this._onScroll = this._onScroll.bind(this);
        this._onResize = this._onResize.bind(this);
    }

    _attachListeners() {
        document.addEventListener('pointerover', this._onPointerOver, true);
        document.addEventListener('pointerout', this._onPointerOut, true);
        document.addEventListener('pointermove', this._onPointerMove, {passive: true});
        document.addEventListener('pointerleave', this._onPointerLeave, true);
        document.addEventListener('contextmenu', this._onContextMenu, true);
        document.addEventListener('pointerdown', this._onPointerDown, true);
        document.addEventListener('focusin', this._onFocusIn, true);
        document.addEventListener('focusout', this._onFocusOut, true);
        document.addEventListener('keydown', this._onKeyDown, true);

        window.addEventListener('mouseout', this._onWindowMouseOut, true);
        window.addEventListener('blur', this._onWindowBlur, true);
        document.addEventListener('visibilitychange', this._onVisibilityChange, true);

        window.addEventListener('scroll', this._onScroll, {passive: true, capture: true});
        window.addEventListener('resize', this._onResize, {passive: true});
    }

    _detachListeners() {
        document.removeEventListener('pointerover', this._onPointerOver, true);
        document.removeEventListener('pointerout', this._onPointerOut, true);
        document.removeEventListener('pointermove', this._onPointerMove);
        document.removeEventListener('pointerleave', this._onPointerLeave, true);
        document.removeEventListener('contextmenu', this._onContextMenu, true);
        document.removeEventListener('pointerdown', this._onPointerDown, true);
        document.removeEventListener('focusin', this._onFocusIn, true);
        document.removeEventListener('focusout', this._onFocusOut, true);
        document.removeEventListener('keydown', this._onKeyDown, true);
        window.removeEventListener('mouseout', this._onWindowMouseOut, true);
        window.removeEventListener('blur', this._onWindowBlur, true);
        document.removeEventListener('visibilitychange', this._onVisibilityChange, true);
        window.removeEventListener('scroll', this._onScroll, true);
        window.removeEventListener('resize', this._onResize);
    }

    _onPointerOver(e) {
        if (this._destroyed || this._tour.active) return;

        const target = e.target;
        if (!(target instanceof Element)) return;

        if (this._bubble?.contains(target)) {
            this._hover.tooltip = true;
            this._clearTimers(["hide"]);
            return;
        }

        if (this._bridge?.contains(target)) {
            this._hover.bridge = true;
            this._clearTimers(["hide"]);
            return;
        }

        if (this._locked && this._active && this._activeTrigger === "click") return;

        const id = this._findTooltipTarget(target);
        if (!id) {
            return;
        }

        const tooltip = this._tooltips.get(id);
        if (!tooltip?.enabled) return;
        if (tooltip.windowOnly) return;
        if (!tooltip.trigger.includes("hover")) return;

        this._hover.target = true;
        this._clearTimers(["hide"]);

        const el = this._getTooltipTarget(target, tooltip);
        if (this._active?.id === id && this._activeEl === el) return;

        this._activeVirtualRect = null;
        this._scheduleShow(tooltip, el);
    }

    _onPointerMove(e) {
        this._cursorPos = {x: e.clientX, y: e.clientY};

        if (this._destroyed) return;
        if (!this._active || !this._bubble) return;

        if (this._activeTrigger === "hover") {
            const target = e.target;
            if (target instanceof Element) {
                const overTarget = !!(this._activeEl && this._activeEl.contains && this._activeEl.contains(target));
                const overTooltip = this._bubble.contains(target);
                const overBridge = !!(this._bridge && this._bridge.contains(target));

                const isHovering = overTarget || overTooltip || overBridge;

                this._hover.target = overTarget;
                this._hover.tooltip = overTooltip;
                this._hover.bridge = overBridge;

                if (!isHovering && !this._focus) {
                    this._scheduleHide();
                } else if (isHovering) {
                    this._clearTimers(["hide"]);
                }
            }
        }

        if (this._active && this._active.followCursor && this._bubble) {
            if (!this._followRAF) {
                this._followRAF = requestAnimationFrame(() => {
                    this._followRAF = 0;
                    if (this._active && this._bubble) {
                        this._positionAtCursor(this._bubble, this._panel, this._caret);
                    }
                });
            }
        }
    }


    _onPointerOut(e) {
        if (this._destroyed) return;

        const from = e.target;
        if (!(from instanceof Element)) return;

        // If no tooltip is active yet (during show delay), still clear hover
        // state and cancel the pending show timer
        if (!this._active || !this._bubble) {
            this._hover.target = false;
            this._clearTimers(["show"]);
            return;
        }

        const fromInTarget = !!(this._activeEl && this._activeEl.contains && this._activeEl.contains(from));
        const fromInBubble = this._bubble.contains(from);
        const fromInBridge = !!(this._bridge && this._bridge.contains(from));
        if (!fromInTarget && !fromInBubble && !fromInBridge) return;

        const to = e.relatedTarget instanceof Element ? e.relatedTarget : null;
        const toInTarget = !!(to && this._activeEl && this._activeEl.contains && this._activeEl.contains(to));
        const toInBubble = !!(to && this._bubble.contains(to));
        const toInBridge = !!(to && this._bridge && this._bridge.contains(to));

        if (toInTarget || toInBubble || toInBridge) return;

        this._hover.target = false;
        this._hover.tooltip = false;
        this._hover.bridge = false;

        if (!this._focus) {
            this._scheduleHide();
        }
    }

    _onContextMenu(e) {
        if (this._destroyed) return;

        const target = e.target;
        if (!(target instanceof Element)) return;

        const popEl = target.closest('.ctt-pop');
        if (popEl && this._popElMap.has(popEl)) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }

        const hit = this._resolveContextTooltipFromNode(target);
        if (!hit) {
            this._closeContextMenu();
            return;
        }

        const tooltip = this._tooltips.get(hit.id);
        if (!tooltip) return;

        const cm = this._getContextMenuConfig(tooltip);
        if (!cm) return;

        e.preventDefault();
        e.stopPropagation();

        this._openContextMenu(e.clientX, e.clientY, tooltip, hit);
    }

    _onWindowMouseOut(e) {
        if (this._destroyed) return;
        if (e && e.relatedTarget) return;
        this._closeContextMenu();
        if (this._active && !this._locked && !this._tour.active) this.hide(true);
    }

    _onWindowBlur() {
        if (this._destroyed) return;
        this._closeContextMenu();
        if (this._active && !this._locked && !this._tour.active) this.hide(true);
    }

    _onVisibilityChange() {
        if (this._destroyed) return;
        if (!document.hidden) return;
        this._closeContextMenu();
        if (this._active && !this._locked && !this._tour.active) this.hide(true);
    }

    _onPopPointerMove(e) {
        const st = this._popDrag;
        if (!st || !st.active) return;

        const inst = this._pops.get(st.id);
        if (!inst) return;

        const dx = e.clientX - st.startX;
        const dy = e.clientY - st.startY;

        if (st.type === 'move') {
            const left = st.startLeft + dx;
            const top = st.startTop + dy;
            this._setPopRect(inst, left, top, st.startW, st.startH);
            return;
        }

        let left = st.startLeft;
        let top = st.startTop;
        let w = st.startW;
        let h = st.startH;

        if (st.dir.includes('e')) w = st.startW + dx;
        if (st.dir.includes('s')) h = st.startH + dy;
        if (st.dir.includes('w')) {
            w = st.startW - dx;
            left = st.startLeft + dx;
        }
        if (st.dir.includes('n')) {
            h = st.startH - dy;
            top = st.startTop + dy;
        }

        const clamped = this._clampPopSize(inst, w, h);
        w = clamped.w;
        h = clamped.h;

        if (st.dir.includes('w')) left = st.startLeft + (st.startW - w);
        if (st.dir.includes('n')) top = st.startTop + (st.startH - h);

        this._setPopRect(inst, left, top, w, h);
    }

    _onPopPointerUp() {
        this._endPopDrag();
    }

    _onPointerLeave(e) {
        if (this._destroyed) return;
        if (!this._active) {
            this._hover.target = false;
            this._clearTimers(["show"]);
            return;
        }

        const target = e.target;
        if (!(target instanceof Element)) return;

        if (this._bubble && (target === this._bubble || this._bubble.contains(target))) {
            this._hover.tooltip = false;
            if (!this._hover.target && !this._hover.bridge && !this._focus) {
                this._scheduleHide();
            }
            return;
        }

        if (this._activeEl && (target === this._activeEl || this._activeEl.contains(target))) {
            const to = e.relatedTarget instanceof Element ? e.relatedTarget : null;
            const toInBubble = !!(to && this._bubble && this._bubble.contains(to));
            const toInBridge = !!(to && this._bridge && this._bridge.contains(to));
            if (!toInBubble && !toInBridge) {
                this._hover.target = false;
                if (!this._hover.tooltip && !this._hover.bridge && !this._focus) {
                    this._scheduleHide();
                }
            }
        }
    }

    _onPointerDown(e) {
        if (this._destroyed) return;

        if (e.button !== 0) return;

        const target = e.target;
        if (!(target instanceof Element)) return;

        if (this._ctx?.el && this._ctx.open) {
            if (this._ctx.el.contains(target)) {
                this._handleContextMenuPointerDown(e, target);
                return;
            }
            this._closeContextMenu();
        }

        if (this._handlePopPointerDown(e, target)) return;

        if (this._tour.active && this._bubble?.contains(target)) {
            const btn = target.closest('[data-tour-action]');
            if (btn) {
                const action = btn.dataset.tourAction;
                if (action === 'next') {
                    this.nextStep();
                } else if (action === 'prev') {
                    this.prevStep();
                } else if (action === 'end') this.endTour();
                return;
            }
        }

        const bubbleEl = target.closest('.ctt-bubble');
        if (bubbleEl) {
            if (bubbleEl === this._bubble && this._active) {
                if (this._handleActionClick(target, this._active, 'active')) return;
            } else {
                const pinId = this._pinBubbleMap.get(bubbleEl);
                if (pinId) {
                    const inst = this._pins.get(pinId);
                    if (inst && inst.tooltip) {
                        if (this._handleActionClick(target, inst.tooltip, 'pinned', inst)) return;
                    }
                }
            }
            const popEl = target.closest('.ctt-pop');
            if (popEl) {
                const popId = this._popElMap.get(popEl);
                if (popId) {
                    const inst = this._pops.get(popId);
                    if (inst && inst.tooltip) {
                        if (this._handleActionClick(target, inst.tooltip, 'pop', inst)) return;
                    }
                }
            }

            if (this._active && this._locked && this._lockedMode === 'tooltip' && !this._tour.active && this._bubble?.contains(target)) {
                if (!target.closest('[data-action-id]')) {
                    this.hide(true);
                    return;
                }
            }
        }

        const id = this._findTooltipTarget(target);
        if (id) {
            const tooltip = this._tooltips.get(id);

            if (tooltip?.enabled && tooltip.windowOnly) {
                const el = this._getTooltipTarget(target, tooltip);

                if (e.detail >= 2) {
                    this._collapseAllPopWindows(tooltip.id);
                    return;
                }

                const max = tooltip.maxPopWindows ?? this._opts.maxPopWindows;
                const openCount = this._countPopWindows(tooltip.id);
                if (openCount >= max) return;

                this.pop(tooltip.id, {target: el, point: {x: e.clientX, y: e.clientY}});
                return;
            }

            if (tooltip?.enabled && tooltip.trigger.includes('click')) {
                const el = this._getTooltipTarget(target, tooltip);

                if (e.detail >= 2) {
                    this._collapseAllPopWindows(tooltip.id);
                    return;
                }

                if (this._active?.id === id && this._activeEl === el && this._activeTrigger === 'click') {
                    this.hide(true);
                    return;
                }

                this._activeVirtualRect = tooltip.virtual ? this._virtualRectFromPoint(e.clientX, e.clientY) : null;
                this._locked = true;
                this._lockedMode = 'outside';
                this._showTooltip(tooltip, el, 'click');
                return;
            }

            if (tooltip?.hideOnClick === true) {
                this.hide(true);
                return;
            }
        }

        if (this._active && !this._bubble?.contains(target) && !(this._activeEl && this._activeEl.contains && this._activeEl.contains(target))) {

            if (this._active.onClickOutside) {
                this._active.onClickOutside({tooltip: this._active, target: this._activeEl, event: e});
            }
            if (!this._locked && this._active.hideOnClick) {
                this.hide(true);
            } else if (this._locked && this._activeTrigger === 'click' && this._active.hideOnClick) this.hide(true);
        }
        for (const inst of [...this._pins.values()]) {
            if (!inst || inst.locked) continue;
            if (!inst.tooltip?.hideOnClick) continue;
            if (inst.bubble?.contains(target)) continue;
            if (inst.target?.contains && inst.target.contains(target)) continue;
            this.unpin(inst.id);
        }
    }

    _handleActionClick(targetEl, tooltip, scope, inst) {
        if (!tooltip?.actions?.length) return false;
        const btn = targetEl.closest('[data-action-id]');
        if (!btn) return false;

        const actionId = btn.dataset.actionId;
        const action = tooltip.actions.find(a => a.id === actionId || a.label === actionId);
        if (!action) return true;

        try {
            if (action.onClick) action.onClick(action, tooltip);
        } catch (err) {
            console.error('[CadesTooltips] Action onClick error:', err);
        }

        try {
            if (tooltip.onAction) tooltip.onAction(action, tooltip);
        } catch (err) {
            console.error('[CadesTooltips] onAction error:', err);
        }

        this._emit('action', {action, tooltip});

        if (action.closeOnClick !== false) {
            if (scope === 'pinned' && inst) {
                this.unpin(inst.id);
            } else if (scope === 'pop' && inst) {
                this.unpop(inst.id);
            } else {
                this.hide(true);
            }
        }

        return true;
    }


    _normalizeContextMenu(cfg, isGlobal = false) {
        const defaultItems = ['copy', 'pop', 'lock', 'close-windows'];
        if (cfg == null) {
            return isGlobal ? {enabled: false, items: defaultItems} : null;
        }

        if (cfg === true) return {enabled: true, items: defaultItems};
        if (cfg === false) return {enabled: false, items: []};

        if (typeof cfg === 'object') {
            const enabled = cfg.enabled ?? true;
            const items = Array.isArray(cfg.items) && cfg.items.length ? cfg.items : defaultItems;
            return {enabled: !!enabled, items};
        }

        return isGlobal ? {enabled: false, items: defaultItems} : null;
    }

    _asPxNumber(v, fallback) {
        if (typeof v === 'number' && Number.isFinite(v)) return v;
        if (typeof v === 'string') {
            const m = v.trim().match(/^(-?\d+(?:\.\d+)?)/);
            if (m) {
                const n = parseFloat(m[1]);
                if (Number.isFinite(n)) return n;
            }
        }
        return fallback;
    }

    _normalizePopConfig(cfg, isGlobal = false) {
        const defaults = {
            title: '',
            minWidth: 240,
            minHeight: 140,
            maxWidth: 900,
            maxHeight: 700,
            width: 380,
            height: 240,
            resizable: true
        };

        if (cfg == null) return isGlobal ? defaults : null;
        if (cfg === true) return defaults;
        if (cfg === false) return isGlobal ? {...defaults} : null;
        if (typeof cfg !== 'object') return isGlobal ? defaults : null;

        const out = {...defaults};

        if (cfg.title != null) out.title = String(cfg.title);
        if (cfg.minWidth != null) out.minWidth = this._asPxNumber(cfg.minWidth, defaults.minWidth);
        if (cfg.minHeight != null) out.minHeight = this._asPxNumber(cfg.minHeight, defaults.minHeight);
        if (cfg.maxWidth != null) out.maxWidth = this._asPxNumber(cfg.maxWidth, defaults.maxWidth);
        if (cfg.maxHeight != null) out.maxHeight = this._asPxNumber(cfg.maxHeight, defaults.maxHeight);
        if (cfg.width != null) out.width = this._asPxNumber(cfg.width, defaults.width);
        if (cfg.height != null) out.height = this._asPxNumber(cfg.height, defaults.height);
        if (cfg.resizable != null) out.resizable = cfg.resizable !== false;

        out.maxWidth = Math.max(out.maxWidth, out.minWidth);
        out.maxHeight = Math.max(out.maxHeight, out.minHeight);

        out.width = Math.min(Math.max(out.width, out.minWidth), out.maxWidth);
        out.height = Math.min(Math.max(out.height, out.minHeight), out.maxHeight);

        return out;
    }

    _getContextMenuConfig(tooltip) {
        const g = this._opts.contextMenu;
        const t = tooltip?.contextMenu;

        const enabled = (t == null) ? (g?.enabled ?? false) : (t.enabled ?? false);
        if (!enabled) return null;

        const defaultItems = ['copy', 'pop', 'lock', 'close-windows'];
        const items = (t && Array.isArray(t.items) && t.items.length) ? t.items : (g?.items || defaultItems);
        return {enabled: true, items};
    }

    _ensureContextMenu() {
        if (this._ctx.el) return;

        const el = document.createElement('div');
        el.className = 'ctt-ctx';
        el.setAttribute('role', 'menu');
        el.setAttribute('aria-hidden', 'true');
        el.dataset.cttInstance = this._instanceId;
        this._getContainer().appendChild(el);

        this._ctx.el = el;
    }

    _resolveContextTooltipFromNode(node) {
        const popEl = node.closest('.ctt-pop');
        if (popEl) {
            const popId = this._popElMap.get(popEl);
            if (popId) {
                const inst = this._pops.get(popId);
                return {id: popId, scope: 'pop', targetEl: inst?.target || null, popEl};
            }
        }

        const bubbleEl = node.closest('.ctt-bubble');
        if (bubbleEl) {
            if (bubbleEl === this._bubble && this._active) {
                return {id: this._active.id, scope: 'active', targetEl: this._activeEl || null, bubbleEl};
            }
            const pinId = this._pinBubbleMap.get(bubbleEl);
            if (pinId) {
                const inst = this._pins.get(pinId);
                return {id: pinId, scope: 'pinned', targetEl: inst?.target || null, bubbleEl};
            }
        }

        const id = this._findTooltipTarget(node);
        if (id) return {id, scope: 'target', targetEl: node};

        return null;
    }

    _openContextMenu(clientX, clientY, tooltip, hit) {
        this._ensureContextMenu();
        const el = this._ctx.el;
        if (!el) return;

        const cm = this._getContextMenuConfig(tooltip);
        if (!cm) return;

        const items = [];
        let html = '';

        for (const itemRef of cm.items) {
            const itemId = (typeof itemRef === 'string') ? itemRef : String(itemRef?.id || '');
            if (!itemId) continue;

            const def = this._ctxRegistry.get(itemId) || null;
            const rawLabel = (typeof itemRef === 'object' && itemRef && itemRef.label != null) ? itemRef.label : (def?.label || itemId);
            const rawIcon = (typeof itemRef === 'object' && itemRef && itemRef.icon != null) ? itemRef.icon : (def?.icon || '');

            let label = String(rawLabel);
            let icon = String(rawIcon);
            if (itemId === 'pop') {
                const popped = tooltip._popped || this._pops.has(tooltip.id);
                const max = tooltip.maxPopWindows ?? this._opts.maxPopWindows;
                const openCount = this._countPopWindows(tooltip.id);
                label = popped ? 'Restore tooltip' : 'Pop window';
                if (!popped && openCount >= max) label += ' (max reached)';
                if (!icon) icon = popped ? '↩' : '⬈';
            }
            if (itemId === 'lock') {
                const isLocked = this._locked && this._active?.id === tooltip.id;
                label = isLocked ? 'Unlock' : 'Lock';
                if (!icon) icon = isLocked ? '🔓' : '🔒';
            }
            if (itemId === 'close-windows') {
                const openCount = this._countPopWindows(tooltip.id);
                if (openCount === 0) continue;
                label = `Close windows (${openCount})`;
                if (!icon) icon = '✕';
            }

            let enabled = true;
            if (def?.enabled) {
                try {
                    enabled = !!def.enabled({tooltip, hit, ctt: this});
                } catch (_) {
                    enabled = true;
                }
            } else if (typeof itemRef === 'object' && itemRef && typeof itemRef.enabled === 'function') {
                try {
                    enabled = !!itemRef.enabled({tooltip, hit, ctt: this});
                } catch (_) {
                    enabled = true;
                }
            }

            items.push({id: itemId, label, icon, enabled, handler: def?.handler || (typeof itemRef === 'object' ? itemRef.handler : null)});

            html += `<div class="ctt-ctx-item" role="menuitem" data-ctx-id="${this._escapeAttr(itemId)}" aria-disabled="${enabled ? 'false' : 'true'}">`;
            html += `<span class="ctt-ctx-icon">${this._escapeHTML(icon || '')}</span>`;
            html += `<span class="ctt-ctx-label">${this._escapeHTML(label)}</span>`;
            html += `</div>`;
        }

        this._ctx.items = items;
        this._ctx.tooltipId = tooltip.id;
        this._ctx.scope = hit.scope;
        this._ctx.targetEl = hit.targetEl || null;
        this._ctx.x = clientX;
        this._ctx.y = clientY;

        el.innerHTML = html || '';
        el.classList.add('ctt-ctx-open');
        el.setAttribute('aria-hidden', 'false');

        el.style.left = `${clientX}px`;
        el.style.top = `${clientY}px`;

        const r = el.getBoundingClientRect();
        const pad = 8;
        const vw = window.innerWidth;
        const vh = window.innerHeight;

        const left = Math.min(Math.max(pad, clientX), Math.max(pad, vw - r.width - pad));
        const top = Math.min(Math.max(pad, clientY), Math.max(pad, vh - r.height - pad));

        el.style.left = `${left}px`;
        el.style.top = `${top}px`;

        this._ctx.open = true;
    }

    _closeContextMenu(forceRemove = false) {
        if (!this._ctx) return;

        this._ctx.open = false;
        this._ctx.items = [];
        this._ctx.tooltipId = null;
        this._ctx.scope = '';
        this._ctx.targetEl = null;

        const el = this._ctx.el;
        if (!el) return;

        el.classList.remove('ctt-ctx-open');
        el.setAttribute('aria-hidden', 'true');
        el.style.left = '';
        el.style.top = '';

        if (forceRemove) {
            try {
                if (el.parentNode) el.parentNode.removeChild(el);
            } catch (_) {
            }
            this._ctx.el = null;
        }
    }

    async _animateCopy(dom, success = true) {
        if (this._timers.copy) return;

        const overlay = document.createElement("div");
        overlay.classList.add("tooltip-target-overlay");
        dom.appendChild(overlay);

        if (success) {
            overlay.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="m9.55 18l-5.7-5.7l1.425-1.425L9.55 15.15l9.175-9.175L20.15 7.4L9.55 18Z"/></svg>`;
            overlay.classList.add('success');
        } else {
            overlay.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M12 17q.425 0 .713-.288T13 16q0-.425-.288-.713T12 15q-.425 0-.713.288T11 16q0 .425.288.713T12 17Zm0-4q.425 0 .713-.288T13 12V8q0-.425-.288-.713T12 7q-.425 0-.713.288T11 8v4q0 .425.288.713T12 13Z"/></svg>`;
            overlay.classList.add('error');
        }

        this._copyOverlay = overlay;
        this._timers.copy = setTimeout(() => {
            overlay.remove();
            this._copyOverlay = null;
            this._timers.copy = 0;
        }, 1500);

    }


    _handleContextMenuPointerDown(e, target) {
        if (!this._ctx?.open || !this._ctx.el) return;

        const itemEl = target.closest('[data-ctx-id]');
        if (!itemEl || !this._ctx.el.contains(itemEl)) return;

        e.preventDefault();
        e.stopPropagation();

        const id = itemEl.dataset.ctxId;
        if (!id) return;

        const item = this._ctx.items.find(it => it.id === id);
        const tooltip = this._tooltips.get(this._ctx.tooltipId);

        if (!item || !tooltip || item.enabled === false) {
            this._closeContextMenu();
            return;
        }

        if (id === 'copy') {
            const text = this._getTooltipCopyText(tooltip);
            const copyTarget = tooltip.targets?.[0];
            writeStringToClipboard(text).then(() => {
                if (copyTarget) this._animateCopy(copyTarget, true);
            }).catch(() => {
                if (copyTarget) this._animateCopy(copyTarget, false);
            });
            this._closeContextMenu();
            return;
        }

        if (id === 'pop') {
            const popped = tooltip._popped || this._pops.has(tooltip.id);
            if (popped) {
                this._collapseAllPopWindows(tooltip.id);
            } else {
                const max = tooltip.maxPopWindows ?? this._opts.maxPopWindows;
                const openCount = this._countPopWindows(tooltip.id);
                if (openCount < max) {
                    this.pop(tooltip.id, {point: {x: this._ctx.x, y: this._ctx.y}});
                }
            }
            this._closeContextMenu();
            return;
        }

        if (id === 'lock') {
            if (this._locked && this._active?.id === tooltip.id) {
                this.unlock();
            } else {
                if (this._active?.id === tooltip.id) {
                    this.lock(true, 'tooltip');
                }
            }
            this._closeContextMenu();
            return;
        }

        if (id === 'close-windows') {
            this._collapseAllPopWindows(tooltip.id);
            this._closeContextMenu();
            return;
        }

        const handler = item.handler;
        if (typeof handler === 'function') {
            const hit = {id: tooltip.id, scope: this._ctx.scope, targetEl: this._ctx.targetEl};
            Promise.resolve().then(() => handler({tooltip, ctt: this, hit})).catch(err => {
                console.error('[CadesTooltips] Context menu handler error:', err);
            });
        }

        this._closeContextMenu();
    }

    _getTooltipCopyText(tooltip) {
        if (!tooltip) return '';

        if (this._active?.id === tooltip.id && this._inner) {
            const text = (this._inner.textContent || '').trim();
            if (text) return text;
        }

        const raw = tooltip.content == null ? '' : String(tooltip.content);
        if (!tooltip.html) return raw.trim();

        const tmp = document.createElement('div');
        tmp.innerHTML = raw;
        return (tmp.textContent || '').trim();
    }

    _registerBuiltInContextMenuItems() {
        this._ctxRegistry.set('copy', {id: 'copy', label: 'Copy contents', icon: '⧉', enabled: null, handler: null});
        this._ctxRegistry.set('pop', {id: 'pop', label: 'Pop window', icon: '⬈', enabled: null, handler: null});
        this._ctxRegistry.set('lock', {id: 'lock', label: 'Lock', icon: '🔒', enabled: null, handler: null});
        this._ctxRegistry.set('close-windows', {id: 'close-windows', label: 'Close windows', icon: '✕', enabled: null, handler: null});
    }

    _setPoppedTargets(tooltip, on) {
        if (!tooltip?.targets?.length) return;
        for (const el of tooltip.targets) {
            if (!el || !el.classList) continue;
            el.classList.toggle('ctt-popped-target', !!on);
            if (on) {
                el.setAttribute('data-ctt-popped', '1');
            } else {
                el.removeAttribute('data-ctt-popped');
            }
        }
    }

    _createPopInstance(tooltip, target, options = {}, popKey = null) {
        const cfg = {
            ...(this._opts.popDefaults || {}),
            ...(tooltip.pop || {}),
            ...(options.pop || {}),
        };

        const merged = this._normalizePopConfig(cfg, true);

        const el = document.createElement('div');
        el.className = 'ctt-pop';
        el.style.zIndex = String(++this._zStack);

        const header = document.createElement('div');
        header.className = 'ctt-pop-header';

        const title = document.createElement('div');
        title.className = 'ctt-pop-title';
        title.textContent = String(options.title || merged.title || tooltip.title || tooltip.id);

        const close = document.createElement('button');
        close.className = 'ctt-pop-close';
        close.type = 'button';
        close.setAttribute('aria-label', 'Close');
        close.textContent = '×';

        header.appendChild(title);
        header.appendChild(close);

        const panel = document.createElement('div');
        panel.className = 'ctt-panel';

        const inner = document.createElement('div');
        inner.className = 'ctt-inner';

        panel.appendChild(inner);

        el.appendChild(header);
        el.appendChild(panel);

        const resizable = merged.resizable !== false;
        if (resizable) {
            const rSE = document.createElement('div');
            rSE.className = 'ctt-pop-resize ctt-pop-resize-se';
            rSE.dataset.resizeDir = 'se';
            const rSW = document.createElement('div');
            rSW.className = 'ctt-pop-resize ctt-pop-resize-sw';
            rSW.dataset.resizeDir = 'sw';
            const rNE = document.createElement('div');
            rNE.className = 'ctt-pop-resize ctt-pop-resize-ne';
            rNE.dataset.resizeDir = 'ne';
            const rNW = document.createElement('div');
            rNW.className = 'ctt-pop-resize ctt-pop-resize-nw';
            rNW.dataset.resizeDir = 'nw';
            el.appendChild(rSE);
            el.appendChild(rSW);
            el.appendChild(rNE);
            el.appendChild(rNW);
        }

        el.dataset.cttInstance = this._instanceId;

        this._resolveAndRenderContent(inner, tooltip, target);
        this._applyThemeTo(el, panel, tooltip.theme, tooltip);

        this._getContainer().appendChild(el);

        this._popElMap.set(el, popKey || tooltip.id);

        const inst = {
            id: tooltip.id,
            el,
            header,
            panel,
            inner,
            close,
            tooltip,
            target,
            minW: merged.minWidth,
            minH: merged.minHeight,
            maxW: merged.maxWidth,
            maxH: merged.maxHeight
        };

        const initial = this._getPopInitialRect(tooltip, target, merged, options);
        this._setPopRect(inst, initial.left, initial.top, initial.w, initial.h);

        return inst;
    }

    _getPopInitialRect(tooltip, target, cfg, options = {}) {
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const pad = 14;

        let left = pad;
        let top = pad;

        if (options.point && typeof options.point.x === 'number' && typeof options.point.y === 'number') {
            left = options.point.x;
            top = options.point.y;
        } else if (this._active?.id === tooltip.id && this._bubble) {
            const r = this._bubble.getBoundingClientRect();
            left = r.left;
            top = r.top;
        } else {
            const r = target.getBoundingClientRect();
            left = r.left + 8;
            top = r.top + 8;
        }

        let w = cfg.width;
        let h = cfg.height;

        left = Math.min(Math.max(pad, left), Math.max(pad, vw - w - pad));
        top = Math.min(Math.max(pad, top), Math.max(pad, vh - h - pad));

        return {left, top, w, h};
    }

    _focusPop(inst) {
        if (!inst?.el) return;
        inst.el.style.zIndex = String(++this._zStack);
    }

    _handlePopPointerDown(e, target) {
        const popEl = target.closest('.ctt-pop');
        if (!popEl) return false;

        const id = this._popElMap.get(popEl);
        if (!id) return false;

        const inst = this._pops.get(id);
        if (!inst) return false;

        this._focusPop(inst);

        if (target.closest('.ctt-pop-close')) {
            this.unpop(id);
            return true;
        }

        const handle = target.closest('[data-resize-dir]');
        if (handle && popEl.contains(handle)) {
            const dir = handle.dataset.resizeDir || 'se';
            this._beginPopDrag(inst, e, 'resize', dir);
            return true;
        }

        if (target.closest('.ctt-pop-header')) {
            this._beginPopDrag(inst, e, 'move', '');
            return true;
        }

        return false;
    }

    _beginPopDrag(inst, e, type, dir) {
        if (!inst?.el) return;
        if (this._popDrag) this._endPopDrag();

        e.preventDefault();

        const r = inst.el.getBoundingClientRect();

        this._popDrag = {
            active: true,
            id: inst.popKey || inst.id,
            type,
            dir,
            startX: e.clientX,
            startY: e.clientY,
            startLeft: r.left,
            startTop: r.top,
            startW: r.width,
            startH: r.height
        };

        window.addEventListener('pointermove', this._onPopPointerMove, {passive: true, capture: true});
        window.addEventListener('pointerup', this._onPopPointerUp, {passive: true, capture: true});
        window.addEventListener('pointercancel', this._onPopPointerUp, {passive: true, capture: true});
    }

    _endPopDrag(force = false) {
        const st = this._popDrag;
        if (!st && !force) return;
        this._popDrag = null;
        window.removeEventListener('pointermove', this._onPopPointerMove, true);
        window.removeEventListener('pointerup', this._onPopPointerUp, true);
        window.removeEventListener('pointercancel', this._onPopPointerUp, true);
    }

    _clampPopSize(inst, w, h) {
        const minW = inst.minW || 240;
        const minH = inst.minH || 140;
        const maxW = inst.maxW || 900;
        const maxH = inst.maxH || 700;

        const ww = Math.min(Math.max(w, minW), maxW);
        const hh = Math.min(Math.max(h, minH), maxH);

        return {w: ww, h: hh};
    }

    _setPopRect(inst, left, top, w, h) {
        if (!inst?.el) return;

        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const pad = 10;

        const size = this._clampPopSize(inst, w, h);
        w = size.w;
        h = size.h;

        left = Math.min(Math.max(pad, left), Math.max(pad, vw - w - pad));
        top = Math.min(Math.max(pad, top), Math.max(pad, vh - h - pad));

        inst.el.style.left = `${left}px`;
        inst.el.style.top = `${top}px`;
        inst.el.style.width = `${w}px`;
        inst.el.style.height = `${h}px`;
    }


    _onFocusIn(e) {
        if (this._destroyed || this._tour.active) return;

        const target = e.target;
        if (!(target instanceof Element)) return;

        const id = this._findTooltipTarget(target);
        if (!id) return;

        const tooltip = this._tooltips.get(id);
        if (!tooltip?.enabled) return;
        if (tooltip.windowOnly) return;
        if (!tooltip.trigger.includes("focus")) return;

        this._focus = true;
        const el = this._getTooltipTarget(target, tooltip);

        this._activeVirtualRect = null;
        this._showTooltip(tooltip, el, "focus");
    }

    _onFocusOut(e) {
        if (this._destroyed || !this._active) return;
        if (!this._focus) return;

        const target = e.target;
        if (!(target instanceof Element)) return;

        const fromTarget = !!(this._activeEl && this._activeEl.contains && this._activeEl.contains(target));
        const fromBubble = !!(this._bubble && this._bubble.contains(target));
        if (!fromTarget && !fromBubble) return;

        const related = e.relatedTarget;
        if (related instanceof Element) {
            if (this._activeEl && this._activeEl.contains(related)) return;
            if (this._bubble?.contains(related)) return;
        }

        this._focus = false;
        if (!this._hover.target && !this._hover.tooltip && !this._hover.bridge) {
            this._scheduleHide();
        }
    }

    _onKeyDown(e) {
        if (this._destroyed) return;

        if (e.key === 'Escape') {
            if (this._ctx?.open) {
                this._closeContextMenu();
                return;
            }

            this._endPopDrag(true);

            if (this._tour.active) {
                this.endTour();
            } else {
                if (this._active && this._active.hideOnEsc === false) return;
                this.hideAll();
                this._locked = false;
                this._lockedMode = 'none';
            }
            return;
        }

        if (this._tour.active) {
            if (e.key === 'ArrowRight' || e.key === 'Enter') {
                e.preventDefault();
                this.nextStep();
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                this.prevStep();
            }
        }
    }

    _onScroll() {
        if (this._destroyed) return;

        this._closeContextMenu();

        if (this._rafId) return;
        if (!this._active && !this._pins.size) return;

        this._rafId = requestAnimationFrame(() => {
            this._rafId = 0;

            if (this._active && this._bubble) {
                if (!this._activeEl?.isConnected) {
                    this._hideTooltip(true);
                } else if (!this._active.sticky && !this._tour.active) {
                    if (!this._locked) {
                        this._hideTooltip();
                        return;
                    }
                    if (!this._isInClipRect(this._activeEl)) {
                        this._bubble.style.visibility = 'hidden';
                        this._lockedHidden = true;
                    } else {
                        if (this._lockedHidden) {
                            this._bubble.style.visibility = '';
                            this._lockedHidden = false;
                        }
                        this._positionActive();
                    }
                } else if (this._active.followCursor) {
                    this._positionAtCursor(this._bubble, this._panel, this._caret);
                } else {
                    this._positionActive();
                }
            }

            for (const inst of [...this._pins.values()]) {
                if (!inst.target?.isConnected) {
                    this.unpin(inst.id);
                    continue;
                }
                if (!inst.sticky && !this._isInClipRect(inst.target)) {
                    inst.bubble.style.visibility = 'hidden';
                    inst._scrollHidden = true;
                    continue;
                }
                if (inst._scrollHidden) {
                    inst.bubble.style.visibility = '';
                    inst._scrollHidden = false;
                }
                this._positionPinned(inst);
            }
        });
    }

    _onResize() {
        this._closeContextMenu();
        if (this._active && this._bubble) {
            if (this._active.followCursor) {
                this._positionAtCursor(this._bubble, this._panel, this._caret);
            } else {
                this._positionActive();
            }
        }
        for (const inst of this._pins.values()) this._positionPinned(inst);
        this._repositionHints(true);
    }

    _scheduleShow(tooltip, target) {
        this._clearTimers(["show"]);

        const delay = tooltip.showDelay ?? this._opts.showDelay;

        if (delay > 0) {
            this._timers.show = setTimeout(() => {
                if (this._destroyed) return;
                if (!this._hover.target) return;
                if (!target.isConnected) return;
                if (!this._tooltips.has(tooltip.id)) return;
                this._showTooltip(tooltip, target, "hover");
            }, delay);
        } else {
            this._showTooltip(tooltip, target, "hover");
        }
    }

    _scheduleHide() {
        if (this._locked || this._tour.active) return;

        this._clearTimers(["show", "hide"]);
        const delay = this._active?.hideDelay ?? this._opts.hideDelay;
        this._timers.hide = setTimeout(() => {
            if (this._destroyed || !this._active) return;
            if (this._locked || this._focus) return;
            const isHovering = this._hover.target || this._hover.tooltip || this._hover.bridge;
            if (isHovering) return;
            this._hideTooltip();
        }, delay);
    }

    _showTooltip(tooltip, target, trigger) {
        if (this._destroyed || !tooltip || !target?.isConnected) return;
        if (!this._isInClipRect(target) && trigger !== 'api' && trigger !== 'tour') return;

        const isSingleton = tooltip.singleton ?? this._opts.singleton;
        const wasSameTooltip = this._active && this._active.id === tooltip.id;
        const prevPlacement = this._activePlacement;
        const prevBubbleLeft = this._bubble ? parseFloat(this._bubble.style.left) || 0 : 0;
        const prevBubbleTop = this._bubble ? parseFloat(this._bubble.style.top) || 0 : 0;

        if (this._active && this._active.id !== tooltip.id) {
            if (this._locked && this._lockedMode === 'tooltip' && !this._tour.active) return;
            if (isSingleton && this._bubble) {
                this._singletonPrev = {tooltip: this._active, target: this._activeEl};
                this._active = null;
                this._activeEl = null;
            } else {
                this._hideTooltip(true);
            }
        }

        this._active = tooltip;
        this._activeEl = target;
        this._activeTrigger = trigger;

        if (!this._tour.active && tooltip.lock && !this._locked) {
            this._locked = true;
            this._lockedMode = 'tooltip';
        }

        this._ensureBubble();
        const bubble = this._bubble;
        bubble.style.zIndex = String(++this._zStack);

        const singletonMoving = isSingleton && this._singletonPrev;
        if (!singletonMoving) {
            bubble.style.visibility = "hidden";
            bubble.style.left = "-9999px";
            bubble.style.top = "-9999px";
            bubble.classList.add("ctt-visible");
            bubble.classList.remove("ctt-open");
            bubble.setAttribute("aria-hidden", "false");
        }

        this._resolveAndRenderContent(this._inner, tooltip, target);

        this._applyThemeTo(bubble, this._panel, tooltip.theme, tooltip);
        this._applySizingTo(this._panel, tooltip);

        if (this._caret) {
            this._caret.style.display = tooltip.arrow ? '' : 'none';
        }

        if (tooltip.fitContent) {
            this._panel.style.width = 'fit-content';
        } else {
            this._panel.style.width = '';
        }

        if (tooltip.followCursor) {
            this._positionAtCursor(bubble, this._panel, this._caret);
        } else {
            void this._panel.offsetWidth;
            this._positionActive();
        }

        if (singletonMoving) {
            const newLeft = parseFloat(bubble.style.left) || 0;
            const newTop = parseFloat(bubble.style.top) || 0;
            const dx = prevBubbleLeft - newLeft;
            const dy = prevBubbleTop - newTop;
            if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
                bubble.animate([
                    {transform: `translate(${dx}px, ${dy}px)`},
                    {transform: 'translate(0, 0)'}
                ], {duration: 200, easing: 'cubic-bezier(0.16, 1, 0.3, 1)'});
            }
        }

        if (tooltip.spotlight) this._showSpotlight(target, tooltip.spotlightPadding);

        if (tooltip.flash?.enabled && tooltip._flashApplied) {
            this._removeFlash(tooltip);
            this._dismissFlash(tooltip);
        }
        if (tooltip.hint?.enabled) {
            this._removeHints(tooltip.id);
        }

        bubble.classList.toggle('ctt-interactive', !!(tooltip.interactive || this._locked || this._tour.active));
        bubble.style.visibility = '';

        if (singletonMoving && this._singletonPrev?.tooltip?.className) {
            bubble.classList.remove(...this._singletonPrev.tooltip.className);
        }
        if (tooltip.className) {
            bubble.classList.add(...tooltip.className);
        }

        if (!singletonMoving) {
            this._animateBubble(bubble, tooltip.animation, this._activePlacement, tooltip);
        }

        this._singletonPrev = null;
        this._observeTargetResize(target);

        if (tooltip.onShow) tooltip.onShow(tooltip, target);
        this._emit('show', {id: tooltip.id, target, trigger, tooltip});

        if (this._opts.debug) {
            console.log(`[CadesTooltips:${this._instanceId}] Show:`, tooltip.id, trigger);
        }
    }

    _hideTooltip(immediate = false) {
        if (!this._active) return;

        const tooltip = this._active;
        const target = this._activeEl;
        const placement = this._activePlacement;

        if (this._currentAnimation) {
            this._currentAnimation.cancel();
            this._currentAnimation = null;
        }

        this._hideSpotlight();
        this._unobserveTargetResize();

        const bubble = this._bubble;

        this._active = null;
        this._activeEl = null;
        this._activeTrigger = '';
        this._activeVirtualRect = null;
        this._hover = {target: false, tooltip: false, bridge: false};
        this._focus = false;

        this._clearTimers(['show', 'hide']);

        if (!this._tour.active) {
            this._locked = false;
            this._lockedMode = 'none';
            this._lockedHidden = false;
        }

        if (bubble) {
            if (immediate) {
                this._removeBubble();
            } else {
                const animBubble = bubble;
                if (this._hidingBubble) this._hidingBubble.remove();
                this._hidingBubble = animBubble;
                this._bubble = null;
                this._panel = null;
                this._inner = null;
                this._caret = null;
                this._bridge = null;
                const safetyTimeout = setTimeout(() => {
                    if (animBubble.parentNode) animBubble.parentNode.removeChild(animBubble);
                    this._hidingBubble = null;
                }, 2000);
                this._animateBubbleOut(animBubble, tooltip.animation, placement, tooltip).then(() => {
                    clearTimeout(safetyTimeout);
                    if (animBubble.parentNode) animBubble.parentNode.removeChild(animBubble);
                    this._hidingBubble = null;
                }).catch(() => {
                    clearTimeout(safetyTimeout);
                    if (animBubble.parentNode) animBubble.parentNode.removeChild(animBubble);
                    this._hidingBubble = null;
                });
            }
        }

        if (tooltip.flash?.enabled && (!tooltip?.flash?.once)) {
            this._applyFlash(tooltip);
        }

        if (tooltip.onHide) {
            tooltip.onHide(tooltip, target);
        }
        this._emit('hide', {id: tooltip.id, target, tooltip});

        if (this._opts.debug) {
            console.log('[CadesTooltips] Hide:', tooltip.id);
        }
    }

    _showPinnedTooltip(tooltip, target, options = {}) {
        if (this._destroyed || !tooltip || !target?.isConnected) return;

        let inst = this._pins.get(tooltip.id);
        if (!inst) {
            inst = this._createPinnedInstance(tooltip.id);
            this._pins.set(tooltip.id, inst);
        }

        inst.id = tooltip.id;
        inst.tooltip = tooltip;
        inst.target = target;
        inst.locked = options.lock ?? true;
        inst.sticky = tooltip.sticky ?? false;

        inst.bubble.style.zIndex = String(++this._zStack);
        this._prepareBubbleForMeasure(inst.bubble);

        this._resolveAndRenderContent(inst.inner, tooltip, target);
        this._applyThemeTo(inst.bubble, inst.panel, tooltip.theme, tooltip);
        this._applySizingTo(inst.panel, tooltip);

        this._measureAndPositionPinned(inst);

        inst.bubble.classList.add('ctt-visible');
        inst.bubble.classList.toggle('ctt-interactive', !!tooltip.interactive);
        inst.bubble.style.visibility = '';

        this._animateBubble(inst.bubble, tooltip.animation, inst.placement || 'top', tooltip);

        this._emit('pin', {id: tooltip.id, target, tooltip});
    }

    _createPinnedInstance(id) {
        const bubble = document.createElement("div");
        bubble.className = "ctt-bubble";
        bubble.dataset.cttInstance = this._instanceId;
        bubble.setAttribute("role", "tooltip");
        bubble.setAttribute("aria-hidden", "true");

        const panel = document.createElement("div");
        panel.className = "ctt-panel";
        const inner = document.createElement("div");
        inner.className = "ctt-inner";
        const caret = document.createElement("div");
        caret.className = "ctt-caret";
        const bridge = document.createElement("div");
        bridge.className = "ctt-bridge";

        panel.appendChild(inner);
        bubble.appendChild(panel);
        bubble.appendChild(caret);
        bubble.appendChild(bridge);

        this._getContainer().appendChild(bubble);

        bubble.style.zIndex = String(++this._zStack);
        this._pinBubbleMap.set(bubble, id);

        return {id, bubble, panel, inner, caret, bridge, tooltip: null, target: null, locked: true, sticky: false, placement: "top"};
    }

    _removePinnedInstance(inst) {
        if (!inst) return;
        try {
            this._pinBubbleMap.delete(inst.bubble);
        } catch (_) {
        }
        if (inst.bubble?.parentNode) {
            inst.bubble.parentNode.removeChild(inst.bubble);
        }
    }

    _ensureBubble() {
        if (this._bubble) return;
        if (this._hidingBubble) {
            this._hidingBubble.remove();
            this._hidingBubble = null;
        }

        const bubble = document.createElement("div");
        bubble.className = "ctt-bubble";
        bubble.dataset.cttInstance = this._instanceId;
        bubble.setAttribute("role", "tooltip");
        bubble.setAttribute("aria-hidden", "true");

        const panel = document.createElement("div");
        panel.className = "ctt-panel";

        const inner = document.createElement("div");
        inner.className = "ctt-inner";

        const caret = document.createElement("div");
        caret.className = "ctt-caret";

        const bridge = document.createElement("div");
        bridge.className = "ctt-bridge";

        panel.appendChild(inner);
        bubble.appendChild(panel);
        bubble.appendChild(caret);
        bubble.appendChild(bridge);

        bubble.style.zIndex = String(++this._zStack);
        this._getContainer().appendChild(bubble);

        this._bubble = bubble;
        this._panel = panel;
        this._inner = inner;
        this._caret = caret;
        this._bridge = bridge;
    }

    _removeBubble() {
        if (this._currentAnimation) {
            this._currentAnimation.cancel();
            this._currentAnimation = null;
        }
        if (this._bubble?.parentNode) {
            this._bubble.parentNode.removeChild(this._bubble);
        }
        this._bubble = null;
        this._panel = null;
        this._inner = null;
        this._caret = null;
        this._bridge = null;
    }

    _prepareBubbleForMeasure(bubble) {
        if (!bubble) return;
        bubble.classList.add('ctt-visible');
        bubble.classList.remove('ctt-open');
        bubble.style.visibility = 'hidden';
        bubble.style.left = '-9999px';
        bubble.style.top = '-9999px';
        bubble.style.opacity = '';
        bubble.style.transform = '';
        bubble.setAttribute('aria-hidden', 'false');
    }

    _applySizingTo(panel, tooltip) {
        if (!panel) return;
        panel.style.maxWidth = tooltip.maxWidth || 'none';
        panel.style.maxHeight = tooltip.maxHeight || 'none';
        panel.style.width = tooltip.fitContent ? 'fit-content' : '';
    }

    _resolveAndRenderContent(inner, tooltip, target) {
        if (!inner) return;
        if (!tooltip._renderEpoch) tooltip._renderEpoch = 0;
        const renderEpoch = ++tooltip._renderEpoch;
        const content = tooltip.content;

        if (typeof content === "function") {
            try {
                const result = content(target, tooltip);
                if (result instanceof Promise || (result && typeof result.then === "function")) {
                    inner.innerHTML = '<div class="ctt-loading"><div class="ctt-spinner"></div></div>';
                    Promise.resolve(result).then(resolved => {
                        if (tooltip._renderEpoch !== renderEpoch) return;
                        this._renderResolvedContent(inner, tooltip, resolved);
                    }).catch(err => {
                        if (tooltip._renderEpoch !== renderEpoch) return;
                        this._renderStringContent(inner, tooltip, "Failed to load content");
                        console.error("[CadesTooltips] Async content error:", err);
                    });
                } else {
                    this._renderResolvedContent(inner, tooltip, result);
                }
            } catch (err) {
                this._renderStringContent(inner, tooltip, "Render error");
                console.error("[CadesTooltips] Content function error:", err);
            }
        } else {
            this._renderStringContent(inner, tooltip, content);
        }
    }

    _renderResolvedContent(inner, tooltip, result) {
        if (!inner) return;
        if (result instanceof Element || result instanceof DocumentFragment) {
            inner.textContent = "";
            this._renderChrome(inner, tooltip);
            inner.appendChild(result);
            this._renderActions(inner, tooltip);
        } else {
            this._renderStringContent(inner, tooltip, result == null ? "" : String(result));
        }
    }

    _renderStringContent(inner, tooltip, text) {
        if (!inner) return;
        let html = "";

        if (tooltip.media) {
            if (tooltip.media.type === "image" && tooltip.media.src) {
                html += `<div class="ctt-media"><img src="${this._escapeAttr(tooltip.media.src)}" alt="${this._escapeAttr(tooltip.media.alt || "")}" /></div>`;
            } else if (tooltip.media.type === "video" && tooltip.media.src) {
                html += `<div class="ctt-media"><video src="${this._escapeAttr(tooltip.media.src)}" ${tooltip.media.autoplay ? "autoplay" : ""} ${tooltip.media.loop ? "loop" : ""} muted playsinline></video></div>`;
            }
        }

        if (tooltip.progress != null) {
            html += `<div class="ctt-progress"><div class="ctt-progress-fill" style="width:${Math.max(0, Math.min(100, parseFloat(tooltip.progress) || 0))}%"></div></div>`;
        }

        if (tooltip.html) {
            html += String(text);
        } else {
            html += `<div class="ctt-text">${this._escapeHTML(text)}</div>`;
        }

        if (tooltip.actions?.length) {
            html += '<div class="ctt-actions">';
            for (const action of tooltip.actions) {
                const variant = action.variant ? `ctt-action-${action.variant}` : "";
                const actionId = action.id || action.label || "";
                html += `<button class="ctt-action ${variant}" data-action-id="${this._escapeAttr(actionId)}">${this._escapeHTML(action.label || "Action")}</button>`;
            }
            html += "</div>";
        }

        inner.innerHTML = html;
    }

    _renderChrome(inner, tooltip) {
        let html = "";
        if (tooltip.media) {
            if (tooltip.media.type === "image" && tooltip.media.src) {
                html += `<div class="ctt-media"><img src="${this._escapeAttr(tooltip.media.src)}" alt="${this._escapeAttr(tooltip.media.alt || "")}" /></div>`;
            } else if (tooltip.media.type === "video" && tooltip.media.src) {
                html += `<div class="ctt-media"><video src="${this._escapeAttr(tooltip.media.src)}" ${tooltip.media.autoplay ? "autoplay" : ""} ${tooltip.media.loop ? "loop" : ""} muted playsinline></video></div>`;
            }
        }
        if (tooltip.progress != null) {
            html += `<div class="ctt-progress"><div class="ctt-progress-fill" style="width:${Math.max(0, Math.min(100, parseFloat(tooltip.progress) || 0))}%"></div></div>`;
        }
        if (html) inner.insertAdjacentHTML("afterbegin", html);
    }

    _renderActions(inner, tooltip) {
        if (!tooltip.actions?.length) return;
        let html = '<div class="ctt-actions">';
        for (const action of tooltip.actions) {
            const variant = action.variant ? `ctt-action-${action.variant}` : "";
            const actionId = action.id || action.label || "";
            html += `<button class="ctt-action ${variant}" data-action-id="${this._escapeAttr(actionId)}">${this._escapeHTML(action.label || "Action")}</button>`;
        }
        html += "</div>";
        inner.insertAdjacentHTML("beforeend", html);
    }

    _renderContentInto(inner, tooltip) {
        if (!inner) return;
        const content = tooltip.content;
        if (typeof content === "function") return;
        this._renderStringContent(inner, tooltip, content == null ? "" : String(content));
    }

    _updateProgressIn(inner, progress) {
        const fill = inner?.querySelector?.('.ctt-progress-fill');
        if (fill) fill.style.width = `${Math.max(0, Math.min(100, parseFloat(progress) || 0))}%`;
    }

    _resolveAnimKeyframes(anim, placement, direction) {
        const frames = anim[direction] || anim['in'];
        if (!frames) return null;

        if (frames && typeof frames === 'object' && !Array.isArray(frames)) {
            const dirFrames = frames[placement] || frames.top || Object.values(frames)[0];
            if (Array.isArray(dirFrames)) return dirFrames;
            return null;
        }
        return Array.isArray(frames) ? frames : null;
    }

    _reverseKeyframes(keyframes) {
        if (!keyframes || !keyframes.length) return [{opacity: 0}];
        return keyframes.slice().reverse().map(f => {
            const copy = {...f};
            delete copy.offset;
            return copy;
        });
    }

    _animateBubble(bubble, animName, placement, tooltip) {
        if (!bubble) return;

        if (tooltip?.animationInKeyframe) {
            if (bubble === this._bubble && this._currentAnimation) {
                this._currentAnimation.cancel();
                this._currentAnimation = null;
            }
            const origins = {top: '50% 100%', bottom: '50% 0%', left: '100% 50%', right: '0% 50%'};
            bubble.style.transformOrigin = origins[placement] || '50% 50%';
            bubble.style.animation = `${tooltip.animationInKeyframe} 200ms cubic-bezier(0.16, 1, 0.3, 1) forwards`;
            bubble.classList.add('ctt-open');
            const onEnd = () => {
                bubble.style.animation = '';
            };
            bubble.addEventListener('animationend', onEnd, {once: true});
            return;
        }

        const anim = CadesTooltips.ANIMATIONS[animName] || CadesTooltips.ANIMATIONS['shift-away'];

        if (bubble === this._bubble && this._currentAnimation) {
            this._currentAnimation.cancel();
            this._currentAnimation = null;
        }

        const origins = {top: '50% 100%', bottom: '50% 0%', left: '100% 50%', right: '0% 50%'};
        bubble.style.transformOrigin = origins[placement] || '50% 50%';

        const keyframes = this._resolveAnimKeyframes(anim, placement, 'in');
        if (!keyframes) {
            bubble.classList.add('ctt-open');
            return;
        }

        const opts = {...anim.options, fill: 'forwards'};
        if (tooltip?.animationDuration != null) opts.duration = tooltip.animationDuration;

        const handle = bubble.animate(keyframes, opts);

        if (bubble === this._bubble) this._currentAnimation = handle;

        handle.onfinish = () => {
            bubble.classList.add('ctt-open');
            try {
                handle.commitStyles();
                handle.cancel();
            } catch (_) {
            }
            if (bubble === this._bubble) this._currentAnimation = null;
        };
        handle.oncancel = () => {
            bubble.classList.add('ctt-open');
            if (bubble === this._bubble) this._currentAnimation = null;
        };
    }

    _animateBubbleOut(bubble, animName, placement, tooltip) {
        if (!bubble) return Promise.resolve();

        if (bubble._activeAnimation) {
            bubble._activeAnimation.cancel();
            bubble._activeAnimation = null;
        }

        if (tooltip?.animationOutKeyframe) {
            const origins = {top: '50% 100%', bottom: '50% 0%', left: '100% 50%', right: '0% 50%'};
            bubble.style.transformOrigin = origins[placement] || '50% 50%';
            return new Promise(resolve => {
                bubble.style.animation = `${tooltip.animationOutKeyframe} 150ms ease-in forwards`;
                const fallback = setTimeout(() => {
                    bubble.removeEventListener('animationend', onEnd);
                    resolve();
                }, 2000);
                const onEnd = () => {
                    clearTimeout(fallback);
                    bubble.removeEventListener('animationend', onEnd);
                    bubble.style.animation = '';
                    resolve();
                };
                bubble.addEventListener('animationend', onEnd);
            });
        }

        const outAnimName = tooltip?.animationOut || animName;
        const outAnim = CadesTooltips.ANIMATIONS[outAnimName] || CadesTooltips.ANIMATIONS['shift-away'];

        const origins = {top: '50% 100%', bottom: '50% 0%', left: '100% 50%', right: '0% 50%'};
        bubble.style.transformOrigin = origins[placement] || '50% 50%';

        let keyframes;
        if (outAnim.out) {
            keyframes = this._resolveAnimKeyframes(outAnim, placement, 'out');
        }
        if (!keyframes) {
            const inKeyframes = this._resolveAnimKeyframes(outAnim, placement, 'in');
            keyframes = this._reverseKeyframes(inKeyframes);
        }

        if (!keyframes || !keyframes.length) return Promise.resolve();

        const baseDuration = tooltip?.animationDuration ?? outAnim.options?.duration ?? 150;
        const duration = Math.round(baseDuration * 0.75);

        return new Promise(resolve => {
            const handle = bubble.animate(keyframes, {
                duration,
                easing: 'cubic-bezier(0.4, 0, 1, 1)',
                fill: 'forwards'
            });
            handle.onfinish = resolve;
            handle.oncancel = resolve;
        });
    }

    _measureAndPositionActive() {
        void this._panel?.offsetWidth;
        this._positionActive();
    }

    _positionActive() {
        const placement = this._positionBubble(this._bubble, this._panel, this._caret, this._bridge, this._active, this._activeEl, this._activeVirtualRect);
        if (placement) this._activePlacement = placement;
    }

    _measureAndPositionPinned(inst) {
        void inst.panel?.offsetWidth;
        this._positionPinned(inst);
    }

    _positionPinned(inst) {
        const placement = this._positionBubble(inst.bubble, inst.panel, inst.caret, inst.bridge, inst.tooltip, inst.target, null);
        if (placement) inst.placement = placement;
    }

    _positionBubble(bubble, panel, caret, bridge, tooltip, targetEl, virtualRect) {
        if (!bubble || !panel || !tooltip || !targetEl) return '';

        const clip = this._getClipRect(targetEl);
        const pad = this._opts.padding;
        const offset = tooltip.offset ?? this._opts.distance;

        const targetRect = virtualRect || targetEl.getBoundingClientRect();

        // Measure the BUBBLE (the actual positioned box), not just the panel.
        // This accounts for CSS padding, borders, and any extra chrome.
        const bubbleRect = bubble.getBoundingClientRect();
        const bw = bubbleRect.width || panel.getBoundingClientRect().width;
        const bh = bubbleRect.height || panel.getBoundingClientRect().height;

        const placements = Array.isArray(tooltip.placement) && tooltip.placement.length ? tooltip.placement : ['top'];

        const placement = tooltip.flip ? this._getBestPlacement(placements, targetRect, bw, bh, clip, pad, offset) : placements[0];

        let x = 0;
        let y = 0;

        switch (placement) {
            case 'top':
                x = targetRect.left + targetRect.width / 2 - bw / 2;
                y = targetRect.top - bh - offset;
                break;
            case 'bottom':
                x = targetRect.left + targetRect.width / 2 - bw / 2;
                y = targetRect.bottom + offset;
                break;
            case 'left':
                x = targetRect.left - bw - offset;
                y = targetRect.top + targetRect.height / 2 - bh / 2;
                break;
            case 'right':
                x = targetRect.right + offset;
                y = targetRect.top + targetRect.height / 2 - bh / 2;
                break;
        }

        x = Math.max(clip.left + pad, Math.min(x, clip.right - bw - pad));
        y = Math.max(clip.top + pad, Math.min(y, clip.bottom - bh - pad));

        bubble.style.left = `${Math.round(x)}px`;
        bubble.style.top = `${Math.round(y)}px`;

        this._setPlacementClass(bubble, placement);

        if (tooltip.interactive) {
            bubble.classList.toggle('ctt-interactive', true);
        } else {
            bubble.classList.toggle('ctt-interactive', false);
        }

        if (caret) {
            caret.style.display = tooltip.arrow === false ? 'none' : '';
        }

        this._positionCaret(caret, placement, targetRect, x, y, bw, bh);
        this._positionBridge(bridge, placement, targetRect, x, y, bw, bh, offset);

        return placement;
    }

    _setPlacementClass(bubble, placement) {
        if (!bubble) return;
        bubble.classList.remove('ctt-place-top', 'ctt-place-bottom', 'ctt-place-left', 'ctt-place-right');
        bubble.classList.add(`ctt-place-${placement}`);
    }

    _positionAtCursor(bubble, panel, caret) {
        if (!bubble || !panel) return;

        const pw = panel.offsetWidth;
        const ph = panel.offsetHeight;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const pad = this._opts.padding;

        let x = this._cursorPos.x + 15;
        let y = this._cursorPos.y + 15;

        if (x + pw > vw - pad) x = this._cursorPos.x - pw - 15;
        if (y + ph > vh - pad) y = this._cursorPos.y - ph - 15;

        x = Math.max(pad, x);
        y = Math.max(pad, y);

        bubble.style.left = `${Math.round(x)}px`;
        bubble.style.top = `${Math.round(y)}px`;

        if (caret) {
            caret.style.display = 'none';
        }
    }

    _positionCaret(caret, placement, targetRect, bubbleX, bubbleY, pw, ph) {
        if (!caret) return;
        if (caret.style.display === 'none') return;

        caret.style.display = '';

        const CARET = 12;
        const MIN = 16;

        let left = '';
        let top = '';

        const cx = targetRect.left + targetRect.width / 2;
        const cy = targetRect.top + targetRect.height / 2;

        if (placement === 'top' || placement === 'bottom') {
            let l = cx - bubbleX - CARET / 2;
            l = Math.max(MIN, Math.min(l, pw - CARET - MIN));
            left = `${Math.round(l)}px`;
        } else {
            let t = cy - bubbleY - CARET / 2;
            t = Math.max(MIN, Math.min(t, ph - CARET - MIN));
            top = `${Math.round(t)}px`;
        }

        caret.style.left = left;
        caret.style.top = top;
    }

    _positionBridge(bridge, placement, targetRect, bubbleX, bubbleY, pw, ph, offset) {
        if (!bridge) return;

        const b = bridge.style;
        b.left = b.top = b.width = b.height = '';

        const gap = offset + 12;

        switch (placement) {
            case 'top': {
                const spanW = Math.max(pw, targetRect.width, 80);
                const spanX = Math.min(0, targetRect.left - bubbleX);
                b.left = `${Math.max(-20, spanX - 10)}px`;
                b.top = `${ph}px`;
                b.width = `${spanW + 20}px`;
                b.height = `${gap}px`;
                break;
            }
            case 'bottom': {
                const spanW = Math.max(pw, targetRect.width, 80);
                const spanX = Math.min(0, targetRect.left - bubbleX);
                b.left = `${Math.max(-20, spanX - 10)}px`;
                b.top = `${-gap}px`;
                b.width = `${spanW + 20}px`;
                b.height = `${gap}px`;
                break;
            }
            case 'left': {
                const spanH = Math.max(ph, targetRect.height, 80);
                const spanY = Math.min(0, targetRect.top - bubbleY);
                b.left = `${pw}px`;
                b.top = `${Math.max(-20, spanY - 10)}px`;
                b.width = `${gap}px`;
                b.height = `${spanH + 20}px`;
                break;
            }
            case 'right': {
                const spanH = Math.max(ph, targetRect.height, 80);
                const spanY = Math.min(0, targetRect.top - bubbleY);
                b.left = `${-gap}px`;
                b.top = `${Math.max(-20, spanY - 10)}px`;
                b.width = `${gap}px`;
                b.height = `${spanH + 20}px`;
                break;
            }
        }
    }

    _getBestPlacement(placements, targetRect, bw, bh, clip, pad, offset) {
        if (placements == null) return 'top';
        placements = ensure_list(placements);
        for (const p of placements) {
            if (this._fitsPlacement(p, targetRect, bw, bh, clip, pad, offset)) return p;
        }

        const opposites = {top: 'bottom', bottom: 'top', left: 'right', right: 'left'};
        for (const p of placements) {
            const opp = opposites[p];
            if (opp && this._fitsPlacement(opp, targetRect, bw, bh, clip, pad, offset)) return opp;
        }

        return placements[0] || 'top';
    }

    _fitsPlacement(placement, rect, bw, bh, clip, pad, offset) {
        switch (placement) {
            case 'top':
                return rect.top - offset - bh >= clip.top + pad;
            case 'bottom':
                return rect.bottom + offset + bh <= clip.bottom - pad;
            case 'left':
                return rect.left - offset - bw >= clip.left + pad;
            case 'right':
                return rect.right + offset + bw <= clip.right - pad;
            default:
                return true;
        }
    }

    _resolveThemeName(themeName) {
        if (themeName === 'auto' || !themeName) {
            // Use cached value if available (set by refreshTheme)
            if (this._resolvedTheme) return this._resolvedTheme;
            return this._detectThemeFromDOM();
        }
        return themeName;
    }

    _detectThemeFromDOM() {
        const dt = (
            document.documentElement?.getAttribute('data-theme') ||
            document.body?.getAttribute('data-theme') ||
            ''
        ).toLowerCase().trim();
        return dt.includes('dark') ? 'dark' : 'light';
    }

    _applyThemeTo(bubble, panel, themeName, tooltip) {
        if (!bubble || !panel) return;

        bubble.classList.remove('ctt-theme-dark', 'ctt-theme-light', 'ctt-glow', 'ctt-gradient');

        if (bubble._prevThemeClass) {
            bubble.classList.remove(bubble._prevThemeClass);
        }
        if (tooltip?.themeClass) {
            bubble.classList.add(tooltip.themeClass);
            bubble._prevThemeClass = tooltip.themeClass;
        } else {
            bubble._prevThemeClass = null;
        }

        if (tooltip?.themeColors) {
            const c = tooltip.themeColors;
            if (c.bg) bubble.style.setProperty('--ctt-bg', c.bg);
            if (c.bgSolid) bubble.style.setProperty('--ctt-bg-solid', c.bgSolid);
            if (c.text) bubble.style.setProperty('--ctt-text', c.text);
            if (c.textMuted) bubble.style.setProperty('--ctt-text-muted', c.textMuted);
            if (c.border) bubble.style.setProperty('--ctt-border', c.border);
            if (c.shadow) bubble.style.setProperty('--ctt-shadow', c.shadow);
            if (c.radius) bubble.style.setProperty('--ctt-radius', c.radius);
            panel.style.backdropFilter = '';
            panel.style.webkitBackdropFilter = '';
            return;
        }

        const resolved = this._resolveThemeName(themeName);

        // Always apply the resolved class so the bubble carries the correct
        // theme explicitly -- not relying on CSS cascade alone.
        bubble.classList.add(`ctt-theme-${resolved}`);

        if (themeName === 'auto' || !themeName) {
            // Strip inline overrides so the class-driven styles take precedence
            const vars = ['--ctt-bg', '--ctt-bg-solid', '--ctt-text', '--ctt-text-muted', '--ctt-border', '--ctt-shadow', '--ctt-radius', '--ctt-glow'];
            for (const v of vars) bubble.style.removeProperty(v);
        }

        panel.style.backdropFilter = '';
        panel.style.webkitBackdropFilter = '';
    }

    _showSpotlight(target, padding = 8) {
        if (!target) return;

        if (!this._spotlight) {
            this._spotlight = document.createElement('div');
            this._spotlight.className = 'ctt-spotlight';
            this._spotlight.dataset.cttInstance = this._instanceId;
            this._getContainer().appendChild(this._spotlight);
        }

        const rect = target.getBoundingClientRect();
        const s = this._spotlight.style;

        s.setProperty('--spotlight-x', `${rect.left - padding}px`);
        s.setProperty('--spotlight-y', `${rect.top - padding}px`);
        s.setProperty('--spotlight-w', `${rect.width + padding * 2}px`);
        s.setProperty('--spotlight-h', `${rect.height + padding * 2}px`);

        this._spotlight.classList.add('ctt-spotlight-visible');
    }

    _hideSpotlight() {
        if (this._spotlight) this._spotlight.classList.remove('ctt-spotlight-visible');
    }

    _normalizeHint(hint) {
        if (hint == null) return {enabled: false};
        if (typeof hint === 'string') hint = {text: hint, showOnLoad: true};
        if (hint === true) hint = {text: 'New', showOnLoad: true};
        if (typeof hint !== 'object') return {enabled: false};

        return {
            enabled: true,
            text: hint.text || hint.content || 'New',
            showOnLoad: hint.showOnLoad !== false,
            placement: this._normalizePlacement(hint.placement || hint.direction || 'top'),
            hideOnOutOfView: hint.hideOnOutOfView ?? this._opts.hintDefaults.hideOnOutOfView,
            positionStrategy: hint.positionStrategy || this._opts.hintDefaults.positionStrategy
        };
    }

    _ensureHintObserver() {
        if (this._hintObserver) return;

        if (!('IntersectionObserver' in window)) return;

        this._hintObserver = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                const set = this._hintByTarget.get(entry.target);
                if (!set) continue;

                for (const inst of set) {
                    if (!inst) continue;
                    if (!inst.hideOnOutOfView) continue;

                    if (entry.isIntersecting) {
                        inst.element.classList.remove('ctt-hint-hidden');
                        inst.element.classList.add('ctt-hint-visible');
                    } else {
                        inst.element.classList.remove('ctt-hint-visible');
                        inst.element.classList.add('ctt-hint-hidden');
                    }
                }
            }
        }, {root: null, threshold: 0});
    }

    _createHints(tooltip) {
        if (tooltip._hintShown) return;

        for (const target of tooltip.targets) {
            if (!target.isConnected) continue;

            let marks = this._hintMarks.get(target);
            if (!marks) {
                marks = new Set();
                this._hintMarks.set(target, marks);
            }
            if (marks.has(tooltip.id)) continue;
            marks.add(tooltip.id);

            const hint = document.createElement('div');
            hint.className = 'ctt-hint ctt-hint-hidden';
            hint.innerHTML = `<span class="ctt-hint-dot"></span><span class="ctt-hint-text">${this._escapeHTML(tooltip.hint.text)}</span>`;
            hint.dataset.cttInstance = this._instanceId;
            this._getContainer().appendChild(hint);

            const inst = {
                id: tooltip.id,
                target,
                element: hint,
                placement: tooltip.hint.placement,
                hideOnOutOfView: tooltip.hint.hideOnOutOfView,
                positionStrategy: tooltip.hint.positionStrategy
            };

            this._hints.push(inst);

            let set = this._hintByTarget.get(target);
            if (!set) {
                set = new Set();
                this._hintByTarget.set(target, set);
            }
            set.add(inst);

            this._repositionHint(inst);

            const visibleNow = !inst.hideOnOutOfView || this._isInViewport(target);
            if (visibleNow) {
                hint.classList.remove('ctt-hint-hidden');
                hint.classList.add('ctt-hint-visible');
            }

            if (inst.hideOnOutOfView) {
                this._ensureHintObserver();
                this._hintObserver?.observe?.(target);
            }
        }

        tooltip._hintShown = true;
    }

    _removeHints(tooltipId) {
        for (let i = this._hints.length - 1; i >= 0; i--) {
            const hint = this._hints[i];
            if (hint.id !== tooltipId) continue;

            const set = this._hintByTarget.get(hint.target);
            if (set) {
                set.delete(hint);
                // Only unobserve if no other hints remain on this target
                if (set.size === 0) {
                    try {
                        this._hintObserver?.unobserve?.(hint.target);
                    } catch (_) {
                    }
                }
            }

            hint.element.remove();
            const marks = this._hintMarks.get(hint.target);
            if (marks) marks.delete(hint.id);

            this._hints.splice(i, 1);
        }
    }

    _removeAllHints() {
        for (const hint of this._hints) {
            try {
                this._hintObserver?.unobserve?.(hint.target);
            } catch (_) {
            }
            hint.element.remove();
            const marks = this._hintMarks.get(hint.target);
            if (marks) marks.delete(hint.id);
        }
        this._hintByTarget = new WeakMap();
        this._hints = [];
    }

    _repositionHints(force) {
        if (!force) return;

        for (let i = this._hints.length - 1; i >= 0; i--) {
            const hint = this._hints[i];
            if (!hint.target.isConnected) {
                hint.element.remove();
                this._hints.splice(i, 1);
                continue;
            }
            this._repositionHint(hint);
        }
    }

    _repositionHint(hint) {
        const rect = hint.target.getBoundingClientRect();
        const hw = hint.element.offsetWidth;
        const hh = hint.element.offsetHeight;

        const hintClip = {top: 0, left: 0, bottom: window.innerHeight, right: window.innerWidth};
        const placement = this._getBestPlacement(hint.placement, rect, hw, hh, hintClip, 8, 6);

        let x = 0;
        let y = 0;

        switch (placement) {
            case 'top':
                x = rect.left + rect.width / 2 - hw / 2;
                y = rect.top - hh - 6;
                break;
            case 'bottom':
                x = rect.left + rect.width / 2 - hw / 2;
                y = rect.bottom + 6;
                break;
            case 'left':
                x = rect.left - hw - 6;
                y = rect.top + rect.height / 2 - hh / 2;
                break;
            case 'right':
                x = rect.right + 6;
                y = rect.top + rect.height / 2 - hh / 2;
                break;
        }

        if (hint.positionStrategy === 'absolute') {
            const sx = window.scrollX ?? 0;
            const sy = window.scrollY ?? 0;
            hint.element.style.left = `${Math.round(x + sx)}px`;
            hint.element.style.top = `${Math.round(y + sy)}px`;
            hint.element.style.position = 'absolute';
        } else {
            hint.element.style.left = `${Math.round(x)}px`;
            hint.element.style.top = `${Math.round(y)}px`;
            hint.element.style.position = 'fixed';
        }
    }

    _normalizeFlash(config, existing = null) {
        let flash = config.flash;
        existing = existing || (this._tooltips.get(config.id)?.flash ?? {})

        if (!flash) {
            flash = {...existing, enabled: false};
        }
        if (flash === true) {
            flash = {...existing, enabled: true};
        }

        if (typeof flash !== 'object'){
            flash = {};
        }

        // Backwards compat: intensity -> opacity
        if ('intensity' in flash && !('opacity' in flash)) {
            flash.opacity = flash.intensity;
        }

        // Merge: incoming flash fields -> existing tooltip flash -> global defaults
        const base = existing ? existing : {};
        const defaults = this._opts.flashDefaults;

        const pickNum = (key) => {
            for (const src of [flash, base, defaults]) {
                if (src && Number.isFinite(src[key])) return src[key];
            }
            return defaults[key];
        };

        const color = flash.color || base.color || defaults.color;
        const opacity = Math.max(0, Math.min(1, pickNum('opacity')));
        const spread = Math.max(0, pickNum('spread'));
        const duration = Math.max(0, pickNum('duration'));
        const interval = Math.max(0, pickNum('interval'));
        const easing = flash.easing || base.easing || defaults.easing;
        const iterations = (flash.iterations ?? base.iterations ?? defaults.iterations);
        const once = flash.once ?? base.once ?? defaults.once ?? false;
        const enabled = flash.enabled ?? base.enabled ?? false;

        return {
            enabled: enabled,
            color,
            opacity,
            spread,
            duration,
            interval,
            easing: String(easing),
            iterations: iterations === 'infinite' ? 'infinite' : String(iterations),
            once: !!once
        };
    }

    _applyFlash(tooltip) {
        if (tooltip._flashApplied) return;
        if (!tooltip.flash?.enabled) return;
        if (this._isFlashDismissed(tooltip)) return;
        this._ensureFlashRule(tooltip);

        for (const target of tooltip.targets) {
            target.classList.add('ctt-flash', tooltip._flashClass);
        }
        tooltip._flashApplied = true;

        if (tooltip.flash?.once) {
            const delay = tooltip.flash.duration + tooltip.flash.interval;
            tooltip._flashOnceTimer = setTimeout(() => {
                tooltip._flashOnceTimer = 0;
                this._removeFlash(tooltip);
                this._dismissFlash(tooltip);
            }, delay);
        }
    }

    _removeFlash(tooltip) {
        if (tooltip._flashOnceTimer) {
            clearTimeout(tooltip._flashOnceTimer);
            tooltip._flashOnceTimer = 0;
        }
        if (!tooltip._flashApplied) return;
        for (const target of tooltip.targets) {
            target.classList.remove('ctt-flash', tooltip._flashClass);
        }
        tooltip._flashApplied = false;
    }

    _ensureFlashRule(tooltip) {
        const rgb = this._parseColor(tooltip.flash.color);
        const c1 = `rgba(${rgb.r},${rgb.g},${rgb.b},${tooltip.flash.opacity})`;
        const c2 = `rgba(${rgb.r},${rgb.g},${rgb.b},0)`;

        const className = tooltip._flashClass;
        const rule = `  
.${className} {  --ctt-flash-c1: ${c1};  
  --ctt-flash-c2: ${c2};  
  --ctt-flash-spread: ${tooltip.flash.spread}px;  
  --ctt-flash-duration: ${tooltip.flash.duration}ms;  
  --ctt-flash-easing: ${tooltip.flash.easing};  
  --ctt-flash-iterations: ${tooltip.flash.iterations};  
  --ctt-flash-interval: ${tooltip.flash.interval}ms;
}`.trim();

        this._flashRules.set(className, rule);
        this._syncFlashStyleElement();
    }

    _removeFlashRule(className) {
        this._flashRules.delete(className);
        this._syncFlashStyleElement();
    }

    _syncFlashStyleElement() {
        const STYLE_ID = 'ctt-flash-rules-' + this._instanceId;
        let style = document.getElementById(STYLE_ID);
        if (!style) {
            style = document.createElement('style');
            style.id = STYLE_ID;
            document.head.appendChild(style);
        }

        if (this._flashRules.size === 0) {
            if (style) style.remove();
            return;
        }
        style.textContent = Array.from(this._flashRules.values()).join('\n\n');
    }

    _getFlashStore(tooltip) {
        const mode = tooltip?.flashStorage;
        if (mode === 'local') return window.localStorage;
        if (mode === 'session') return window.sessionStorage;
        return null;
    }

    _readFlashDismissed(store) {
        if (!store) return {};
        try {
            const raw = store.getItem(this._opts.flashStorageKey);
            if (!raw) return {};
            return JSON.parse(raw);
        } catch (_) {
            return {};
        }
    }

    _writeFlashDismissed(store, map) {
        if (!store) return;

        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const todayMs = today.getTime();

        const pruned = {};
        for (const [id, ts] of Object.entries(map)) {
            const d = new Date(ts);
            if (!isNaN(d.getTime()) && d.getTime() >= todayMs) {
                pruned[id] = ts;
            }
        }

        try {
            store.setItem(this._opts.flashStorageKey, JSON.stringify(pruned));
        } catch (_) {
        }
    }

    _isFlashDismissed(tooltip) {
        const store = this._getFlashStore(tooltip);
        if (!store) return false;
        const map = this._readFlashDismissed(store);
        return tooltip.id in map;
    }

    _dismissFlash(tooltip) {
        const store = this._getFlashStore(tooltip);
        if (!store) return;
        const map = this._readFlashDismissed(store);
        map[tooltip.id] = new Date().toISOString();
        this._writeFlashDismissed(store, map);
    }

    _showTourStep(index) {
        let step, target;
        const totalSteps = this._tour.steps.length;
        while (index < totalSteps) {
            step = this._tour.steps[index];
            if (!step) return;
            target = null;
            if (typeof step.target === 'string') {
                try {
                    target = document.querySelector(step.target);
                } catch (_) {
                }
            } else if (step.target instanceof Element) {
                target = step.target;
            }
            if (target?.isConnected) break;
            index++;
        }
        if (!target?.isConnected) {
            this.endTour();
            return;
        }
        this._tour.index = index;

        const total = this._tour.steps.length;
        const progress = ((index + 1) / total) * 100;

        const tourTooltip = {
            id: '__tour__',
            targets: [target],
            enabled: true,
            content: this._buildTourContent(step, index, total, progress),
            html: true,
            placement: this._normalizePlacement(step.placement || step.direction || 'bottom'),
            offset: step.offset ?? 16,
            flip: true,
            theme: step.theme || 'auto',
            animation: step.animation || 'scale',
            maxWidth: step.maxWidth || '380px',
            maxHeight: '',
            trigger: 'manual',
            hideOnClick: true,
            interactive: true,
            followCursor: false,
            sticky: true,
            virtual: false,
            spotlight: step.spotlight !== false,
            spotlightPadding: step.spotlightPadding ?? 12,
            media: null,
            progress: null,
            actions: [],
            flash: {enabled: false},
            hint: {enabled: false}
        };

        target.scrollIntoView({behavior: 'smooth', block: 'center', inline: 'center'});
        requestAnimationFrame(() => {
            this._activeVirtualRect = null;
            this._showTooltip(tourTooltip, target, 'tour');
            requestAnimationFrame(() => this._positionActive());
        });

        this._emit('tourStep', {index, step, total});
    }

    _buildTourContent(step, index, total, progress) {
        return `<div class="ctt-tour">
            <div class="ctt-tour-progress">
                <div class="ctt-tour-progress-fill" style="width:${Math.max(0, Math.min(100, parseFloat(progress) || 0))}%">
                </div>  
            </div>
            <div class="ctt-tour-header">
                <span class="ctt-tour-step">Step ${index + 1} of ${total}</span>  
                <button class="ctt-tour-close" data-tour-action="end">✕</button>
            </div>${step.title ? `<div class="ctt-tour-title">${this._escapeHTML(step.title)}</div>` : ''}
            <div class="ctt-tour-body">${step.html ? step.content : this._escapeHTML(step.content)}</div>  
            <div class="ctt-tour-footer">${index > 0 ? '<button class="ctt-tour-btn" data-tour-action="prev">← Back</button>' : '<span></span>'}  
            ${index < total - 1
                ? '<button class="ctt-tour-btn ctt-tour-btn-primary" data-tour-action="next">Next →</button>'
                : '<button class="ctt-tour-btn ctt-tour-btn-primary" data-tour-action="end">✓ Done</button>'
            }  
            </div>
        </div>`;
    }

    _countPopWindows(tooltipId) {
        let count = 0;
        for (const [key, inst] of this._pops) {
            const tid = inst.tooltipId || key;
            if (tid === tooltipId || key === tooltipId || key.startsWith(tooltipId + '__pop_')) count++;
        }
        return count;
    }

    _collapseAllPopWindows(tooltipId) {
        const popsToClose = [];
        for (const [key, inst] of this._pops) {
            const tid = inst.tooltipId || key;
            if (tid === tooltipId || key === tooltipId || key.startsWith(tooltipId + '__pop_')) {
                popsToClose.push({...inst, _popKey: key});
            }
        }
        if (!popsToClose.length) return;

        const tooltip = this._tooltips.get(tooltipId);
        const target = tooltip?.targets?.[0];
        const targetRect = target?.getBoundingClientRect?.();
        const targetX = targetRect ? targetRect.left + targetRect.width / 2 : window.innerWidth / 2;
        const targetY = targetRect ? targetRect.top + targetRect.height / 2 : window.innerHeight / 2;

        for (const inst of popsToClose) {
            if (!inst.el) {
                this.unpop(inst._popKey);
                continue;
            }

            const rect = inst.el.getBoundingClientRect();
            const dx = targetX - (rect.left + rect.width / 2);
            const dy = targetY - (rect.top + rect.height / 2);

            const popKey = inst._popKey;
            const handle = inst.el.animate([
                {transform: 'translate(0, 0) scale(1)', opacity: 1},
                {transform: `translate(${dx}px, ${dy}px) scale(0.1)`, opacity: 0}
            ], {
                duration: 300,
                easing: 'cubic-bezier(0.4, 0, 1, 1)',
                fill: 'forwards'
            });

            handle.onfinish = () => this.unpop(popKey);
            handle.oncancel = () => this.unpop(popKey);
        }
    }

    flash(id, options = {}) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip) return false;
        if (!tooltip.enabled) return false;

        if (!tooltip.flash?.enabled || Object.keys(options).length) {
            // Invalidate stale rule so _ensureFlashRule regenerates with new values
            tooltip.flash = tooltip.flash || {};
            tooltip.flash.enabled = true;
            this._removeFlashRule(tooltip._flashClass);
            tooltip._flashApplied = false;
        }

        const store = this._getFlashStore(tooltip);
        if (store) {
            const map = this._readFlashDismissed(store);
            delete map[id];
            this._writeFlashDismissed(store, map);
        }

        this._applyFlash(tooltip);
        return true;
    }

    stopFlash(id) {
        const tooltip = this._tooltips.get(id);
        if (!tooltip) return false;
        this._removeFlash(tooltip);
        tooltip.flash = tooltip.flash || {};
        tooltip.flash.enabled = false;
        return true;
    }

    _emit(event, data = {}) {
        const listeners = this._listeners.get(event);
        if (!listeners) return;

        for (const cb of listeners) {
            try {
                cb(data);
            } catch (e) {
                console.error('[CadesTooltips] Event error:', e);
            }
        }
    }

    _resolveTargets(target) {
        if (!target) return [];
        if (target instanceof Element) return [target];
        if (typeof target === 'string') {
            try {
                return Array.from(document.querySelectorAll(target));
            } catch (_) {
            }
        }
        if (Array.isArray(target)) return target.filter(t => t instanceof Element);
        return [];
    }

    _findTooltipTarget(el) {
        let current = el;
        while (current) {
            const id = this._elementMap.get(current);
            if (id && this._tooltips.has(id)) return id;
            current = current.parentElement;
        }
        return null;
    }

    _getTooltipTarget(el, tooltip) {
        for (const target of tooltip.targets) {
            if (target.contains(el)) return target;
        }
        return tooltip.targets[0];
    }

    _normalizePlacement(placement) {
        if (Array.isArray(placement)) return placement.map(p => this._normalizeSinglePlacement(p)).filter(Boolean);
        const p = this._normalizeSinglePlacement(placement);
        return p ? [p] : ['top'];
    }

    _normalizeSinglePlacement(p) {
        const s = (typeof p === 'string' ? p : '').toLowerCase();
        return ['top', 'bottom', 'left', 'right'].includes(s) ? s : '';
    }

    _virtualRectFromPoint(x, y) {
        const px = Math.round(x);
        const py = Math.round(y);
        return {left: px, top: py, right: px, bottom: py, width: 0, height: 0};
    }

    _generateId() {
        let id = 'ctt_' + Math.random().toString(36).slice(2, 9);
        while (this._tooltips.has(id)) id = 'ctt_' + Math.random().toString(36).slice(2, 9);
        return id;
    }

    _clearTimer(name) {
        if (this._timers[name]) {
            clearTimeout(this._timers[name]);
            this._timers[name] = 0;
        }
    }

    _clearTimers(names = null) {
        if (names == null) {
            for (const key of Object.keys(this._timers)) {
                this._clearTimer(key);
            }
            if (this._rafId) {
                cancelAnimationFrame(this._rafId);
                this._rafId = 0;
            }
            if (this._followRAF) {
                cancelAnimationFrame(this._followRAF);
                this._followRAF = 0;
            }
        } else {
            const list = Array.isArray(names) ? names : [names];
            for (const key of list) {
                this._clearTimer(key);
            }
        }
    }

    _isInViewport(el) {
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        return rect.top < window.innerHeight && rect.bottom > 0 && rect.left < window.innerWidth && rect.right > 0;
    }


    _escapeHTML(str) {
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/\n/g, '<br>');
    }

    _escapeAttr(str) {
        return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    _parseColor(color) {
        let probe;
        if (!this._colorCache) this._colorCache = new Map();
        if (this._colorCache.has(color)) return this._colorCache.get(color);
        try {
            probe = document.createElement('span');
            probe.style.color = color;
            probe.style.position = 'fixed';
            probe.style.left = '-9999px';
            probe.style.top = '-9999px';
            probe.style.width = '0';
            probe.style.height = '0';
            probe.style.overflow = 'hidden';
            document.body.appendChild(probe);
            const computed = getComputedStyle(probe).color;
            probe.remove();
            const m = computed.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
            if (m) {
                const result = {r: +m[1], g: +m[2], b: +m[3]};
                this._colorCache.set(color, result);
                return result;
            }
        } catch (_) {
            if (probe?.parentNode) probe.parentNode.removeChild(probe);
        }
        return {r: 124, g: 92, b: 255};
    }

}

export class CadesTooltipHandle {
    constructor(manager, id) {
        this._mgr = manager;
        this.id = id;
        this._unsubs = [];
    }

    /** Show the tooltip */
    show(options = {}) {
        return this._mgr.show(this.id, options);
    }

    /** Hide the tooltip */
    hide(force = false) {
        if (this._mgr._active?.id === this.id) return this._mgr.hide(force);
        return false;
    }

    /** Toggle visibility */
    toggle(options = {}) {
        return this._mgr.toggle(this.id, options);
    }

    /** Update tooltip configuration */
    update(config) {
        return this._mgr.update(this.id, config);
    }

    /** Remove the tooltip */
    remove() {
        for (const unsub of this._unsubs) unsub();
        this._unsubs.length = 0;
        return this._mgr.remove(this.id);
    }

    /** Enable the tooltip */
    enable() {
        return this._mgr.enable(this.id);
    }

    /** Disable the tooltip */
    disable() {
        return this._mgr.disable(this.id);
    }

    /** Lock the tooltip on screen */
    lock() {
        if (this._mgr._active?.id === this.id) {
            this._mgr.lock(true, 'tooltip');
            return true;
        }
        return this.show({lock: true});
    }

    /** Unlock the tooltip */
    unlock() {
        if (this._mgr._active?.id === this.id) {
            this._mgr.unlock();
            return true;
        }
        return false;
    }

    /** Start flashing the target element(s) */
    flash(options = {}) {
        return this._mgr.flash(this.id, options);
    }

    /** Stop flashing */
    stopFlash() {
        return this._mgr.stopFlash(this.id);
    }

    /** Pin the tooltip */
    pin(options = {}) {
        return this._mgr.pin(this.id, options);
    }

    /** Unpin the tooltip */
    unpin() {
        return this._mgr.unpin(this.id);
    }

    /** Pop out as a window */
    pop(options = {}) {
        return this._mgr.pop(this.id, options);
    }

    /** Close popped window */
    unpop() {
        return this._mgr.unpop(this.id);
    }

    /** Check if popped */
    get isPopped() {
        return this._mgr.isPopped(this.id);
    }

    /** Set content */
    setContent(content) {
        return this._mgr.setContent(this.id, content);
    }

    /** Set progress */
    setProgress(progress) {
        return this._mgr.setProgress(this.id, progress);
    }

    /** Get the underlying tooltip config */
    get config() {
        return this._mgr.get(this.id);
    }

    /** Check if tooltip exists */
    get exists() {
        return this._mgr.has(this.id);
    }

    /** Check if this tooltip is currently active/visible */
    get isActive() {
        return this._mgr._active?.id === this.id;
    }

    /** Collapse all pop windows for this tooltip */
    collapseWindows() {
        this._mgr._collapseAllPopWindows(this.id);
    }

    /** Listen to events for this tooltip */
    on(event, callback) {
        const unsub = this._mgr.on(event, (data) => {
            if (data.id === this.id || data.tooltip?.id === this.id) callback(data);
        });
        this._unsubs.push(unsub);
        return unsub;
    }
}
