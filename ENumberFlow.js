

import './ENumberFlow.css';

const SVG_ARROW_UP = `
      <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="19" x2="12" y2="5"></line>
        <polyline points="5 12 12 5 19 12"></polyline>
      </svg>`;

const SVG_CIRCLE_ICON = `
      <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="8"></circle>
      </svg>`;

const ORDINAL_SUFFIXES = ['th', 'st', 'nd', 'rd'];

// number-flow-inspired spring easing — fast start, gentle overshoot, smooth settle
const BUTTER_EASE = 'cubic-bezier(0.16, 1, 0.3, 1)';

class DigitRoller {
    constructor() {
        this._viewport = document.createElement('span');
        this._viewport.className = 'enf-dv';

        this._track = document.createElement('span');
        this._track.className = 'enf-dt';

        for (let i = 0; i < 30; i++) {
            const dd = document.createElement('span');
            dd.className = 'enf-dd';
            dd.textContent = String(i % 10);
            this._track.appendChild(dd);
        }

        this._viewport.appendChild(this._track);

        this._currentDigit = 0;
        this._duration = 500;
        this._easing = BUTTER_EASE;

        this._normalizeTimer = null;
        this._onTransitionEnd = null;

        this._rafId = 0;
        this._rafToken = 0;

        // Stagger delay for cascade effect (set externally by DigitEngine)
        this._staggerDelay = 0;

        // Start at canonical home position for digit 0 (index 10)
        this._setIndexImmediate(10);
    }

    get element() {
        return this._viewport;
    }

    invalidateMeasure() {
        // em-based; no measurement needed
    }

    _clearNormalize() {
        if (this._normalizeTimer) {
            clearTimeout(this._normalizeTimer);
            this._normalizeTimer = null;
        }
        if (this._onTransitionEnd) {
            this._track.removeEventListener('transitionend', this._onTransitionEnd);
            this._onTransitionEnd = null;
        }
    }

    _cancelPendingRaf() {
        if (this._rafId) {
            cancelAnimationFrame(this._rafId);
            this._rafId = 0;
        }
        this._rafToken++;
    }

    _setIndexImmediate(index) {
        this._cancelPendingRaf();
        this._track.style.transition = 'none';
        this._track.style.transform = `translate3d(0, -${index*16}px, 0)`;
    }

    _setIndexAnimated(index, duration, easing) {
        this._cancelPendingRaf();

        const dur = (duration | 0) > 0 ? (duration | 0) : 0;
        if (dur <= 0) {
            this._setIndexImmediate(index);
            return;
        }

        const eas = easing || BUTTER_EASE;
        const delay = this._staggerDelay || 0;

        this._track.style.transition = `transform ${dur}ms ${eas} ${delay}ms`;

        // FIX [B1]: Replaced getBoundingClientRect() with getComputedStyle() read.
        // getBoundingClientRect() forces a FULL synchronous layout recalculation
        // (reflow) every time it's called. With 10+ digits updating per frame,
        // that's 10+ forced reflows -- catastrophic for performance.
        // getComputedStyle().transition forces only a style recalc (much cheaper)
        // which is sufficient to arm the CSS transition after a style change.
        // eslint-disable-next-line no-unused-expressions
        getComputedStyle(this._track).transition;

        const token = this._rafToken;
        this._rafId = requestAnimationFrame(() => {
            if (token !== this._rafToken) return;
            this._rafId = 0;
            this._track.style.transform = `translate3d(0, -${index*16}px, 0)`;
        });
    }

    /**
     * Set the displayed digit with optional animation.
     * @param {number} digit 0-9
     * @param {boolean} animate
     * @param {number} direction 1 = increasing, -1 = decreasing, 0 = neutral
     * @param {number} [duration]
     * @param {string} [easing]
     */
    setDigit(digit, animate, direction, duration, easing) {
        digit = digit | 0;
        const prevDigit = this._currentDigit;
        this._currentDigit = digit;

        const dur = duration !== undefined ? duration : this._duration;
        const eas = easing || this._easing;

        if (prevDigit === digit) {
            return;
        }

        if (!animate || dur <= 0) {
            this.snapTo(digit);
            return;
        }

        this._clearNormalize();

        const canonicalIndex = digit + 10;

        let targetIndex;
        if (direction === 0) {
            targetIndex = canonicalIndex;
        } else if (direction > 0) {
            targetIndex = (digit > prevDigit) ? canonicalIndex : (digit + 20);
        } else {
            targetIndex = (digit < prevDigit) ? canonicalIndex : digit;
        }

        this._setIndexAnimated(targetIndex, dur, eas);

        if (targetIndex === canonicalIndex) return;

        this._onTransitionEnd = (ev) => {
            if (ev.propertyName !== 'transform') return;
            this._clearNormalize();
            this._setIndexImmediate(canonicalIndex);
        };

        this._track.addEventListener('transitionend', this._onTransitionEnd);

        const stagger = this._staggerDelay || 0;
        this._normalizeTimer = setTimeout(() => {
            this._clearNormalize();
            this._setIndexImmediate(canonicalIndex);
        }, dur + stagger + 200);
    }

    /**
     * Instantly show a digit with no animation.
     */
    snapTo(digit) {
        digit = digit | 0;
        this._currentDigit = digit;
        this._clearNormalize();
        this._setIndexImmediate(digit + 10);
    }

    destroy() {
        this._clearNormalize();
        this._cancelPendingRaf();
    }
}

const _formatCache = new Map();

class DigitEngine {
    constructor() {
        this._wrapper = document.createElement('span');
        this._wrapper.className = 'enf-engine';

        this._prefixEl = document.createElement('span');
        this._prefixEl.classList.add('enf-sym', 'enf-prefix');
        this._suffixEl = document.createElement('span');
        this._suffixEl.classList.add('enf-sym', 'enf-suffix');

        this._slotsContainer = document.createElement('span');
        this._slotsContainer.classList.add('enf-sym', 'enf-slot');
        this._slotsContainer.style.display = 'flex';
        this._slotsContainer.style.opacity = '1';

        this._wrapper.appendChild(this._prefixEl);
        this._wrapper.appendChild(this._slotsContainer);
        this._wrapper.appendChild(this._suffixEl);

        // Slots: array of { type: 'digit'|'symbol', char: string, node: Element, roller?: DigitRoller }
        this._slots = [];
        this._pendingAnims = []; // track entry/exit animations for cleanup

        this._prevChars = [];
        this._duration = 500;
        this._easing = BUTTER_EASE;
    }

    get element() {
        return this._wrapper;
    }

    setTiming(duration, easing) {
        this._duration = (duration !== undefined && duration !== null) ? duration : 600;
        this._easing = easing || BUTTER_EASE;
    }


    setPrefix(text) {
        const t = text || '';
        if (this._prefixEl.textContent !== t) this._prefixEl.textContent = t;
    }

    setSuffix(text) {
        const t = text || '';
        if (this._suffixEl.textContent !== t) this._suffixEl.textContent = t;
    }

    /**
     * Format a number using a cached Intl.NumberFormat.
     * @returns {string} The formatted string
     */
    formatNumber(value, style, minFD, maxFD, signDisplay) {
        const key = formatKey(style, minFD, maxFD, signDisplay);
        let nf = _formatCache.get(key);
        if (!nf) {
            const opts = {};
            if (style) opts.style = style;
            if (minFD !== undefined) opts.minimumFractionDigits = minFD;
            if (maxFD !== undefined) opts.maximumFractionDigits = maxFD;
            if (signDisplay) opts.signDisplay = signDisplay;
            nf = new Intl.NumberFormat(undefined, opts);
            _formatCache.set(key, nf);
        }
        return nf.format(value);
    }

    /**
     * Render a formatted string with animation.
     * @param {string} formatted The formatted number string (e.g. "1,234.56" or "--")
     * @param {number} direction 1 = increasing, -1 = decreasing, 0 = neutral
     */
    render(formatted, direction) {
        const newChars = formatted.split('');
        const oldChars = this._prevChars;
        const oldSlots = this._slots;

        if (newChars.length === 0 && oldChars.length === 0) return;

        // Skip all work when nothing changed
        if (newChars.length === oldChars.length) {
            let same = true;
            for (let i = 0; i < newChars.length; i++) {
                if (newChars[i] !== oldChars[i]) { same = false; break; }
            }
            if (same) return;
        }

        const aligned = this._alignChars(oldChars, newChars);

        const newSlots = [];
        const frag = document.createDocumentFragment();
        const dur = this._duration;
        const eas = this._easing;

        const exitingSlots = [];

        // Count how many digits will actually change (for stagger calculation)
        let changingDigitCount = 0;
        for (let i = 0; i < aligned.length; i++) {
            const { oldChar, newChar, oldSlotIndex } = aligned[i];
            if (newChar === null) continue;
            const isDigit = newChar >= '0' && newChar <= '9';
            if (oldChar === null) { changingDigitCount++; continue; }
            const oldSlot = (oldSlotIndex !== null) ? oldSlots[oldSlotIndex] : null;
            if (isDigit && oldSlot && oldSlot.type === 'digit' && oldSlot.char !== newChar) changingDigitCount++;
        }

        // Stagger: spread 0-60ms across changing digits for a cascade effect
        const maxStagger = Math.min(60, dur * 0.12);
        const staggerStep = changingDigitCount > 1 ? maxStagger / (changingDigitCount - 1) : 0;
        let changingIdx = 0;

        for (let i = 0; i < aligned.length; i++) {
            const { oldChar, newChar, oldSlotIndex } = aligned[i];
            const isDigit = newChar !== null && newChar >= '0' && newChar <= '9';

            if (newChar === null) {
                if (oldSlotIndex !== null && oldSlots[oldSlotIndex]) {
                    exitingSlots.push(oldSlots[oldSlotIndex]);
                }
                continue;
            }

            if (oldChar === null) {
                const staggerDelay = staggerStep * changingIdx++;
                if (isDigit) {
                    const roller = new DigitRoller();
                    roller.snapTo(parseInt(newChar, 10));
                    const node = roller.element;
                    this._animateEntry(node, dur, eas, staggerDelay);
                    newSlots.push({ type: 'digit', char: newChar, node, roller });
                    frag.appendChild(node);
                } else {
                    const sym = document.createElement('span');
                    sym.className = 'enf-sym';
                    sym.textContent = newChar;
                    this._animateEntry(sym, dur, eas, staggerDelay);
                    newSlots.push({ type: 'symbol', char: newChar, node: sym, roller: null });
                    frag.appendChild(sym);
                }
            } else {
                const oldSlot = (oldSlotIndex !== null) ? oldSlots[oldSlotIndex] : null;

                if (isDigit && oldSlot && oldSlot.type === 'digit' && oldSlot.roller) {
                    const newDigitVal = parseInt(newChar, 10);
                    if (oldSlot.char !== newChar) {
                        const staggerDelay = staggerStep * changingIdx++;
                        oldSlot.roller._staggerDelay = staggerDelay;
                        oldSlot.roller.setDigit(newDigitVal, dur > 0, direction, dur, eas);
                    }
                    oldSlot.char = newChar;
                    newSlots.push(oldSlot);
                    frag.appendChild(oldSlot.node);
                } else if (!isDigit && oldSlot && oldSlot.type === 'symbol') {
                    if (oldSlot.char !== newChar) {
                        oldSlot.node.textContent = newChar;
                        oldSlot.char = newChar;
                    }
                    newSlots.push(oldSlot);
                    frag.appendChild(oldSlot.node);
                } else {
                    if (oldSlot) exitingSlots.push(oldSlot);

                    const staggerDelay = staggerStep * changingIdx++;
                    if (isDigit) {
                        const roller = new DigitRoller();
                        roller.snapTo(parseInt(newChar, 10));
                        const node = roller.element;
                        this._animateEntry(node, dur, eas, staggerDelay);
                        newSlots.push({ type: 'digit', char: newChar, node, roller });
                        frag.appendChild(node);
                    } else {
                        const sym = document.createElement('span');
                        sym.className = 'enf-sym';
                        sym.textContent = newChar;
                        this._animateEntry(sym, dur, eas, staggerDelay);
                        newSlots.push({ type: 'symbol', char: newChar, node: sym, roller: null });
                        frag.appendChild(sym);
                    }
                }
            }
        }

        // Capture old width for smooth width transition
        const container = this._slotsContainer;
        const hadContent = oldChars.length > 0;
        let oldWidth;
        if (hadContent && (newChars.length !== oldChars.length)) {
            oldWidth = container.offsetWidth;
        }

        // Replace container contents in a single DOM operation
        // FIX [B4]: Use replaceChildren() instead of textContent='' + appendChild().
        // replaceChildren() is a single synchronous DOM mutation vs two separate ones,
        // reducing layout thrashing.
        container.replaceChildren(frag);

        // Animate width change smoothly when digit count changes
        if (oldWidth !== undefined) {
            const newWidth = container.offsetWidth;
            if (oldWidth !== newWidth) {
                container.style.width = oldWidth + 'px';
                // Force style recalc then animate to new width
                // eslint-disable-next-line no-unused-expressions
                getComputedStyle(container).width;
                container.style.width = newWidth + 'px';
                // Clear explicit width after transition completes
                const onDone = () => {
                    container.removeEventListener('transitionend', onDone);
                    container.style.width = '';
                };
                container.addEventListener('transitionend', onDone);
                // Safety timeout to clear width if transitionend doesn't fire
                setTimeout(onDone, dur + 100);
            }
        }

        // FIX [B3 cont]: Now animate exits. Exiting nodes need to be re-appended
        // to the container in an overlay position so they can visually fade out
        // without affecting the layout of new content.
        for (let i = 0; i < exitingSlots.length; i++) {
            this._animateExit(exitingSlots[i], dur, eas);
        }

        this._slots = newSlots;
        this._prevChars = newChars;
    }

    /**
     * Immediately show a formatted string without animation.
     */
    snapTo(formatted) {
        const chars = formatted.split('');
        const newSlots = [];
        const frag = document.createDocumentFragment();

        // Destroy old rollers
        for (let i = 0; i < this._slots.length; i++) {
            if (this._slots[i].roller) this._slots[i].roller.destroy();
        }

        for (let i = 0; i < chars.length; i++) {
            const c = chars[i];
            const isDigit = c >= '0' && c <= '9';
            if (isDigit) {
                const roller = new DigitRoller();
                roller.snapTo(parseInt(c, 10));
                newSlots.push({ type: 'digit', char: c, node: roller.element, roller });
                frag.appendChild(roller.element);
            } else {
                const sym = document.createElement('span');
                sym.className = 'enf-sym';
                sym.textContent = c;
                newSlots.push({ type: 'symbol', char: c, node: sym, roller: null });
                frag.appendChild(sym);
            }
        }

        this._slotsContainer.replaceChildren(frag);
        this._slots = newSlots;
        this._prevChars = chars;
    }

    /**
     * Align old and new character arrays for right-to-left digit matching.
     * Integer digits (before decimal point) align from the right.
     * Fraction digits (after decimal point) align from the left.
     */
    _alignChars(oldChars, newChars) {
        const oldDecIdx = oldChars.indexOf('.');
        const newDecIdx = newChars.indexOf('.');

        const oldInt = oldDecIdx >= 0 ? oldChars.slice(0, oldDecIdx) : oldChars.slice();
        const newInt = newDecIdx >= 0 ? newChars.slice(0, newDecIdx) : newChars.slice();
        const oldFrac = oldDecIdx >= 0 ? oldChars.slice(oldDecIdx) : [];
        const newFrac = newDecIdx >= 0 ? newChars.slice(newDecIdx) : [];

        const result = [];

        const oldIntLen = oldInt.length;
        const newIntLen = newInt.length;
        const maxIntLen = Math.max(oldIntLen, newIntLen);

        for (let i = 0; i < maxIntLen; i++) {
            const oldRightOffset = maxIntLen - 1 - i;

            const oldIdx = oldIntLen - 1 - oldRightOffset;
            const newIdx = newIntLen - 1 - oldRightOffset;

            const oc = oldIdx >= 0 ? oldInt[oldIdx] : null;
            const nc = newIdx >= 0 ? newInt[newIdx] : null;
            const oldSlotIdx = oldIdx >= 0 ? oldIdx : null;

            result.push({ oldChar: oc, newChar: nc, oldSlotIndex: oldSlotIdx });
        }

        const oldFracLen = oldFrac.length;
        const newFracLen = newFrac.length;
        const maxFracLen = Math.max(oldFracLen, newFracLen);

        for (let i = 0; i < maxFracLen; i++) {
            const oc = i < oldFracLen ? oldFrac[i] : null;
            const nc = i < newFracLen ? newFrac[i] : null;
            const oldSlotIdx = i < oldFracLen ? (oldDecIdx + i) : null;

            result.push({ oldChar: oc, newChar: nc, oldSlotIndex: oldSlotIdx });
        }

        return result;
    }

    _animateEntry(node, duration, easing, staggerDelay) {
        if (duration <= 0) return;
        const anim = node.animate(
            [
                { opacity: 0, transform: 'translateY(50%) scale(0.85)', filter: 'blur(2px)' },
                { opacity: 1, transform: 'translateY(0) scale(1)', filter: 'blur(0px)' },
            ],
            {
                duration: Math.min(duration, 350),
                easing: BUTTER_EASE,
                fill: 'forwards',
                delay: staggerDelay || 0,
            }
        );
        this._pendingAnims.push(anim);
        anim.onfinish = () => {
            const idx = this._pendingAnims.indexOf(anim);
            if (idx >= 0) this._pendingAnims.splice(idx, 1);
        };
        // FIX [B5]: Handle animation cancellation to prevent memory leaks.
        // If a node is removed from DOM mid-animation (e.g., rapid updates),
        // onfinish never fires and the entry leaks in _pendingAnims.
        anim.oncancel = () => {
            const idx = this._pendingAnims.indexOf(anim);
            if (idx >= 0) this._pendingAnims.splice(idx, 1);
        };
    }

    _animateExit(slot, duration, easing) {
        const node = slot.node;
        if (!node) {
            if (slot.roller) slot.roller.destroy();
            return;
        }
        if (duration <= 0) {
            if (slot.roller) slot.roller.destroy();
            if (node.parentNode) node.parentNode.removeChild(node);
            return;
        }

        // FIX [B3 cont]: For exit animations to be visible, the node must be
        // in the DOM. Re-append it to the slots container with absolute positioning
        // so it overlays without affecting layout flow.
        node.style.position = 'absolute';
        node.style.pointerEvents = 'none';
        this._slotsContainer.appendChild(node);

        const anim = node.animate(
            [
                { opacity: 1, transform: 'translateY(0) scale(1)', filter: 'blur(0px)' },
                { opacity: 0, transform: 'translateY(-50%) scale(0.85)', filter: 'blur(2px)' },
            ],
            {
                duration: Math.min(duration, 280),
                easing: 'cubic-bezier(0.4, 0, 1, 1)',
                fill: 'forwards',
            }
        );
        this._pendingAnims.push(anim);

        const cleanup = () => {
            const idx = this._pendingAnims.indexOf(anim);
            if (idx >= 0) this._pendingAnims.splice(idx, 1);
            if (slot.roller) slot.roller.destroy();
            if (node.parentNode) node.parentNode.removeChild(node);
        };

        anim.onfinish = cleanup;
        // FIX [B5 cont]: Also handle cancellation on exit animations.
        anim.oncancel = cleanup;
    }

    destroy() {
        for (let i = 0; i < this._pendingAnims.length; i++) {
            this._pendingAnims[i].cancel();
        }
        this._pendingAnims = [];
        for (let i = 0; i < this._slots.length; i++) {
            if (this._slots[i].roller) this._slots[i].roller.destroy();
        }
        this._slots = [];
        this._prevChars = [];
    }
}

// --- RAF Batching ------------------------------------------------------------

let RAF_SCHEDULED = false;
let RAF_QUEUE = [];
let RAF_QUEUE_ALT = [];

function rafEnqueue(instance) {
    RAF_QUEUE.push(instance);
    if (!RAF_SCHEDULED) {
        RAF_SCHEDULED = true;
        requestAnimationFrame(rafFlush);
    }
}

function rafFlush() {
    RAF_SCHEDULED = false;

    const processing = RAF_QUEUE;
    RAF_QUEUE = RAF_QUEUE_ALT;
    RAF_QUEUE_ALT = processing;

    for (let i = 0; i < processing.length; i++) {
        const inst = processing[i];
        inst._isUpdatePending = false;
        if (inst._disabled || inst._destroyed) continue;
        inst._performActualUpdate(inst._pendingRawValue);
    }

    processing.length = 0;

    // FIX [B6]: Guard against re-entrant scheduling.
    // If _performActualUpdate triggers a synchronous update() call (e.g., from
    // a user callback), RAF_QUEUE gets items added during the flush. The existing
    // check handles this, but we also need to avoid double-scheduling if
    // rafFlush is already on the microtask queue.
    if (RAF_QUEUE.length && !RAF_SCHEDULED) {
        RAF_SCHEDULED = true;
        requestAnimationFrame(rafFlush);
    }
}

// --- Utilities ---------------------------------------------------------------

function isNil(v) {
    return v === null || v === undefined;
}

function formatKey(style, minFD, maxFD, signDisplay) {
    const s = style || '';
    const minS = minFD === undefined ? '' : String(minFD);
    const maxS = maxFD === undefined ? '' : String(maxFD);
    const signS = signDisplay || '';
    return `${s}\0${minS}\0${maxS}\0${signS}`;
}

// --- ENumberFlow -------------------------------------------------------------

export class ENumberFlow {
    constructor(targetElementOrSelector, options = {}) {
        this._disabled = false;
        this._destroyed = false;

        if (!targetElementOrSelector) {
            console.error('ENumberFlow: Target element or selector is required.');
            this._disabled = true;
            return;
        }

        let targetElement;
        if (typeof targetElementOrSelector === 'string') {
            targetElement = document.querySelector(targetElementOrSelector);
        } else {
            targetElement = targetElementOrSelector;
        }

        if (!targetElement) {
            console.error(`ENumberFlow: Target element not found for "${targetElementOrSelector}".`);
            this._disabled = true;
            return;
        }

        this.defaultConfig = {
            initialValue: 0,
            displayMode: 'number',
            displayStyle: 'default',
            showSign: null,
            magnitudeSuffixes: { tn: 1e12, bn: 1e9, mio: 1e6, k: 1e3 },
            timeBaseUnit: 'ms',
            timeUnits: {
                year: 31536000000, month: 2592000000, week: 604800000,
                day: 86400000, hour: 3600000, min: 60000, s: 1000, ms: 1,
            },
            timeMaxDecimals: 1,
            minimumFractionDigits: undefined,
            maximumFractionDigits: undefined,
            valueBasedBackgroundColor: {
                enabled: false,
                positive: 'rgba(52, 211, 153, 0.2)',
                negative: 'rgba(239, 68, 68, 0.2)',
                zero: 'rgba(107, 114, 128, 0.2)',
                target: 'value',
            },
            valueBasedIcon: {
                enabled: false,
                positive: SVG_ARROW_UP,
                negative: SVG_ARROW_UP,
                zero: SVG_CIRCLE_ICON,
                target: 'value',
            },
            negativeFormat: 'minus',
            valueBasedTextColor: {
                enabled: false,
                positive: '#10B981',
                negative: '#EF4444',
                zero: '#6B7280',
                target: 'value',
            },
            prefix: '',
            suffix: '',
            duration: 600,
            ease: BUTTER_EASE,
            clockThresholds: [],
            pillStyle: {
                positiveBg: 'trend-up',
                negativeBg: 'trend-down',
                charHeight: '0.85em',
                maskHeight: '0.3em',
                arrowStrokeWidth: 3,
            },
            digitalClockStyle: {
                fontSize: '1.75rem',
                separatorColor: '#333',
            },
            initialClockTime: null,
            clockOptionsToStart: null,
            animationSquash: {
                enabled: true,
                minAnimatedIntervalMs: 250,
                busySpinDurationMs: 10,
            },
        };

        const _nestedKeys = [
            'valueBasedBackgroundColor',
            'valueBasedIcon',
            'valueBasedTextColor',
            'magnitudeSuffixes',
            'timeUnits',
            'pillStyle',
            'digitalClockStyle',
            'animationSquash'
        ];

        this.config = { ...this.defaultConfig };
        for (const key of _nestedKeys) {
            if (options && options[key]) {
                this.config[key] = { ...this.defaultConfig[key], ...options[key] };
            } else {
                this.config[key] = { ...this.defaultConfig[key] };
            }
        }

        if (options) {
            for (const key of Object.keys(options)) {
                if (!_nestedKeys.includes(key)) {
                    this.config[key] = options[key];
                }
            }
        }

        this._sortedMagnitudeSuffixes = [];
        this._sortedTimeUnits = [];
        this._rebuildCaches();

        this._createElements();
        targetElement.appendChild(this.container);

        this.currentValue = null;
        this.currentDisplayValue = null;
        this.externalColorValue = null;
        this.timer = { id: null, startTime: 0, currentTime: 0, endTime: null, direction: 'down', running: false };

        this._isUpdatePending = false;
        this._pendingRawValue = null;

        this._nfLastDisplayValue = undefined;
        this._lastFormatted = '';

        this._pillBgAppliedClass = '';
        this._pillArrowTransform = '';

        this._clockSeparators = [];
        this._clockLastH = null;
        this._clockLastM = null;
        this._clockLastS = null;

        this._iconHTML = null;
        this._iconRotation = 0;
        this._iconTimerId = null;

        this._animLastFullSpinAt = 0;
        this._nfLastSpinDuration = this.config.duration;
        this._nfLastSpinEasing = this.config.ease;

        // FIX [B7]: Store reference to the clock interval's expected next tick
        // for drift compensation (used by startClock).
        this._clockExpectedTick = 0;

        if (this.config.displayStyle === 'digitalClock' || this.config.displayMode === 'time') {
            if (this.config.initialValue instanceof Date) {
                this.config.initialClockTime = (this.config.initialValue.getHours() * 3600 + this.config.initialValue.getMinutes() * 60 + this.config.initialValue.getSeconds());
            } else {
                this.config.initialClockTime = this.config.initialValue;
            }
        }

        this.update(this.config.initialValue);
    }

    _rebuildCaches() {
        const mag = this.config.magnitudeSuffixes || {};
        const magEntries = Object.entries(mag);
        magEntries.sort((a, b) => b[1] - a[1]);
        this._sortedMagnitudeSuffixes = magEntries;

        const units = this.config.timeUnits || {};
        const unitEntries = Object.entries(units);
        unitEntries.sort((a, b) => b[1] - a[1]);
        this._sortedTimeUnits = unitEntries;
    }

    _createElements() {
        const config = this.config;

        if (config.displayStyle === 'pillPercentage') {
            this.container = document.createElement('span');
            this.container.className = `enf-pill-percentage`;
            this.container.style.fontSize = config.pillStyle.fontSize;

            this.trendWrapper = document.createElement('div');
            this.trendWrapper.className = 'enf-trend-wrapper';

            this.arrowElement = document.createElement('span');
            this.arrowElement.className = 'enf-pill-arrow';
            this.arrowElement.innerHTML = SVG_ARROW_UP;
            const svgInArrow = this.arrowElement.querySelector('svg');
            if (svgInArrow) svgInArrow.style.strokeWidth = config.pillStyle.arrowStrokeWidth;
            this.trendWrapper.appendChild(this.arrowElement);

            this._engine = new DigitEngine();
            this._engine.setTiming(config.duration, config.ease);
            this.trendWrapper.appendChild(this._engine.element);

            this.container.appendChild(this.trendWrapper);

            this.trendPeriod = document.createElement('span');
            this.trendPeriod.className = 'trend-period';
            this.container.appendChild(this.trendPeriod);

            this.iconElement = null;
            this._hourEngine = null;
            this._minuteEngine = null;
            this._secondEngine = null;
        } else if (config.displayStyle === 'digitalClock') {
            this.container = document.createElement('div');
            this.container.className = 'enf-digital-clock-container';
            this.container.style.fontSize = config.digitalClockStyle.fontSize;

            this._clockSeparators = [];

            const d = config.duration > 0 ? Math.min(config.duration, 300) : 0;

            const createEngine = () => {
                const eng = new DigitEngine();
                eng.setTiming(d, config.ease);
                return eng;
            };

            const createSeparator = () => {
                const sep = document.createElement('span');
                sep.textContent = ':';
                sep.className = 'enf-clock-separator';
                sep.style.color = config.digitalClockStyle.separatorColor;
                this._clockSeparators.push(sep);
                return sep;
            };

            this._hourEngine = createEngine();
            this._minuteEngine = createEngine();
            this._secondEngine = createEngine();

            this.container.appendChild(this._hourEngine.element);
            this.container.appendChild(createSeparator());
            this.container.appendChild(this._minuteEngine.element);
            this.container.appendChild(createSeparator());
            this.container.appendChild(this._secondEngine.element);

            this._engine = null;
            this.iconElement = null;
            this.arrowElement = null;
        } else {
            this.container = document.createElement('div');
            this.container.className = 'enhanced-number-flow-container';

            this.iconElement = document.createElement('span');
            this.iconElement.className = 'enf-icon';
            this.container.appendChild(this.iconElement);

            this._engine = new DigitEngine();
            this._engine.setTiming(config.duration, config.ease);
            this.container.appendChild(this._engine.element);

            this.arrowElement = null;
            this._hourEngine = null;
            this._minuteEngine = null;
            this._secondEngine = null;
        }
    }

    _getComparisonValue(targetType) {
        const configObject = this.config[targetType];
        const configTarget = configObject ? configObject.target : 'value';
        if (configTarget === 'external' && this.externalColorValue !== null) {
            return this.externalColorValue;
        }
        return this.currentValue;
    }

    _applyStylesAndIcon() {
        const config = this.config;
        let activeThreshold = null;

        if (this.timer.running && config.clockThresholds && config.clockThresholds.length > 0) {
            const thresholds = config.clockThresholds;
            const checkValue = this.timer.currentTime;
            if (this.timer.direction === 'down') {
                for (let i = 0; i < thresholds.length; i++) {
                    const t = thresholds[i];
                    if (checkValue <= t.value) {
                        activeThreshold = t;
                        break;
                    }
                }
            } else {
                for (let i = 0; i < thresholds.length; i++) {
                    const t = thresholds[i];
                    if (checkValue >= t.value) {
                        activeThreshold = t;
                        break;
                    }
                }
            }
        }

        if (config.displayStyle === 'pillPercentage') {
            const pillConf = config.pillStyle;
            const valueToCompare = this._getComparisonValue('valueBasedBackgroundColor');

            let newBgClass = (valueToCompare > 0) ? pillConf.positiveBg : pillConf.negativeBg;
            if (activeThreshold && activeThreshold.background_class) {
                newBgClass = activeThreshold.background_class;
            }

            if (newBgClass !== this._pillBgAppliedClass) {
                if (this._pillBgAppliedClass) this.container.classList.remove(this._pillBgAppliedClass);
                if (newBgClass) this.container.classList.add(newBgClass);
                this._pillBgAppliedClass = newBgClass;
            }

            if (this.arrowElement) {
                const rotateDeg = (valueToCompare > 0 || Object.is(valueToCompare, 0)) ? 0 : -180;
                const newTransform = `rotate(${rotateDeg}deg)`;
                if (newTransform !== this._pillArrowTransform) {
                    this.arrowElement.style.transform = newTransform;
                    this._pillArrowTransform = newTransform;
                }
            }
            return;
        }

        if (config.displayStyle === 'digitalClock') {
            let newBgColor = '';
            if (activeThreshold && activeThreshold.background) {
                newBgColor = activeThreshold.background;
            }
            if (this.container.style.backgroundColor !== newBgColor) {
                this.container.style.backgroundColor = newBgColor;
            }

            let newTextColor = '';
            let newSeparatorColor = config.digitalClockStyle.separatorColor;
            if (activeThreshold && activeThreshold.text) {
                newTextColor = activeThreshold.text;
                newSeparatorColor = activeThreshold.text;
            }

            if (this._hourEngine && this._hourEngine.element.style.color !== newTextColor) this._hourEngine.element.style.color = newTextColor;
            if (this._minuteEngine && this._minuteEngine.element.style.color !== newTextColor) this._minuteEngine.element.style.color = newTextColor;
            if (this._secondEngine && this._secondEngine.element.style.color !== newTextColor) this._secondEngine.element.style.color = newTextColor;

            const seps = this._clockSeparators;
            for (let i = 0; i < seps.length; i++) {
                const s = seps[i];
                if (s.style.color !== newSeparatorColor) s.style.color = newSeparatorColor;
            }
            return;
        }

        // Default display style
        const bgConf = config.valueBasedBackgroundColor;
        const bgValueToCompare = this._getComparisonValue('valueBasedBackgroundColor');
        let newBgColor = '';
        if (activeThreshold && activeThreshold.background !== undefined) {
            newBgColor = activeThreshold.background;
        } else if (bgConf.enabled) {
            if (bgValueToCompare > 0) newBgColor = bgConf.positive;
            else if (bgValueToCompare < 0) newBgColor = bgConf.negative;
            else newBgColor = bgConf.zero;
        }
        if (this.container.style.backgroundColor !== newBgColor) {
            this.container.style.backgroundColor = newBgColor;
        }

        if (this._engine) {
            const textConf = config.valueBasedTextColor;
            const textValueToCompare = this._getComparisonValue('valueBasedTextColor');
            let newTextColor = '';

            if (activeThreshold && activeThreshold.text !== undefined) {
                newTextColor = activeThreshold.text;
            } else if (textConf.enabled) {
                if (textValueToCompare > 0) newTextColor = textConf.positive;
                else if (textValueToCompare < 0) newTextColor = textConf.negative;
                else newTextColor = textConf.zero;
            }

            if (this._engine.element.style.color !== newTextColor) {
                this._engine.element.style.color = newTextColor;
            }
        }

        if (this.iconElement) {
            const iconConf = config.valueBasedIcon;
            const iconValueToCompare = this._getComparisonValue('valueBasedIcon');
            let newIconHTML = '';
            let iconRotation = 0;

            if (activeThreshold && activeThreshold.icon !== undefined) {
                newIconHTML = activeThreshold.icon;
            } else if (iconConf.enabled) {
                if (iconValueToCompare > 0) {
                    newIconHTML = iconConf.positive;
                } else if (iconValueToCompare < 0) {
                    newIconHTML = iconConf.negative;
                    iconRotation = -180;
                } else {
                    newIconHTML = iconConf.zero;
                }
            }
            this._updateIconContent(newIconHTML, iconRotation);
        }
    }

    _updateIconContent(newIconHTML, rotation = 0) {
        if (!this.iconElement) return;

        const html = newIconHTML || '';
        const rot = rotation || 0;
        const newTransform = `rotate(${rot}deg)`;

        if (this._iconHTML === html && this._iconRotation === rot) {
            if (!html) {
                if (this.iconElement.style.display !== 'none') this.iconElement.style.display = 'none';
            } else if (this.iconElement.style.display === 'none') {
                this.iconElement.style.display = 'flex';
                this.iconElement.style.transform = newTransform;
            }
            return;
        }

        this._iconHTML = html;
        this._iconRotation = rot;

        this.iconElement.classList.add('enf-icon-fading');

        if (html && this.iconElement.style.display === 'none') {
            this.iconElement.style.display = 'flex';
        }

        if (this._iconTimerId) {
            clearTimeout(this._iconTimerId);
            this._iconTimerId = null;
        }

        this._iconTimerId = setTimeout(() => {
            if (this._destroyed) return;
            this.iconElement.innerHTML = this._iconHTML;
            this.iconElement.style.transform = `rotate(${this._iconRotation}deg)`;
            if (!this._iconHTML) {
                this.iconElement.style.display = 'none';
            }
            this.iconElement.classList.remove('enf-icon-fading');
            this._iconTimerId = null;
        }, 150);
    }

    _formatNumberWithMagnitude(numValue, confMinDigits, confMaxDigits) {
        const absValue = Math.abs(numValue);
        let divisor = 1;
        let suffix = this.config.suffix || '';
        let autoMinDigits = 0;
        let autoMaxDigits = 0;

        const entries = this._sortedMagnitudeSuffixes;
        for (let i = 0; i < entries.length; i++) {
            const s = entries[i][0];
            const val = entries[i][1];
            if (absValue >= val) {
                suffix = s;
                divisor = val;
                if (val >= 1e9) {
                    autoMinDigits = 2;
                    autoMaxDigits = 2;
                } else if (val >= 1e6) {
                    autoMinDigits = 1;
                    autoMaxDigits = 1;
                } else {
                    autoMinDigits = 0;
                    autoMaxDigits = 0;
                }
                break;
            }
        }

        return {
            value: numValue / divisor,
            s: suffix,
            minF: confMinDigits !== undefined ? confMinDigits : autoMinDigits,
            maxF: confMaxDigits !== undefined ? confMaxDigits : autoMaxDigits,
        };
    }

    _formatTimeDuration(numValueInBaseUnit, baseUnit, confMinDigits, confMaxDigits) {
        const units = this.config.timeUnits;
        let valueInMs = numValueInBaseUnit * (units[baseUnit] || 1);
        let displayVal = valueInMs;
        let suffix = 'ms';

        const entries = this._sortedTimeUnits;
        for (let i = 0; i < entries.length; i++) {
            const unitName = entries[i][0];
            const unitMillis = entries[i][1];
            if (!unitMillis) continue;
            if (Math.abs(valueInMs) >= unitMillis) {
                displayVal = valueInMs / unitMillis;
                suffix = unitName;
                break;
            }
        }

        if (numValueInBaseUnit < 0 && displayVal > 0) displayVal *= -1;

        return {
            value: displayVal,
            s: suffix,
            minF: confMinDigits !== undefined ? confMinDigits : 0,
            maxF: confMaxDigits !== undefined ? confMaxDigits : this.config.timeMaxDecimals,
        };
    }

    update(rawValue) {
        if (this._disabled || this._destroyed) return;
        this._pendingRawValue = rawValue;
        if (!this._isUpdatePending) {
            this._isUpdatePending = true;
            rafEnqueue(this);
        }
    }

    _performActualUpdate(rawValue) {
        if (this._disabled || this._destroyed) return;

        if (isNil(rawValue)) {
            this.currentValue = 0;
            this.currentDisplayValue = '--';

            if (this._engine) {
                this._engine.setPrefix('');
                this._engine.setSuffix('');
                this._engine.render('--', 0);
                this._nfLastDisplayValue = this.currentDisplayValue;
            } else if (this.config.displayStyle === 'digitalClock') {
                if (this._hourEngine) this._hourEngine.render('--', 0);
                if (this._minuteEngine) this._minuteEngine.render('--', 0);
                if (this._secondEngine) this._secondEngine.render('--', 0);
                this._clockLastH = null;
                this._clockLastM = null;
                this._clockLastS = null;
            }

            this._applyStylesAndIcon();
            return;
        }

        this.currentValue = (typeof rawValue === 'string') ? parseFloat(rawValue) : rawValue;
        if (Number.isNaN(this.currentValue)) this.currentValue = 0;

        const config = this.config;

        if (config.displayStyle === 'digitalClock') {
            let totalSeconds;
            if (rawValue instanceof Date) {
                totalSeconds = rawValue.getHours() * 3600 + rawValue.getMinutes() * 60 + rawValue.getSeconds();
            } else {
                totalSeconds = Math.max(0, Math.floor(this.currentValue));
            }

            const hours = Math.floor(totalSeconds / 3600) % 24;
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const seconds = totalSeconds % 60;

            const pad = (n) => String(n).padStart(2, '0');

            if (this._hourEngine && hours !== this._clockLastH) {
                const dir = this._clockLastH !== null ? (hours > this._clockLastH ? 1 : hours < this._clockLastH ? -1 : 0) : 0;
                this._hourEngine.render(pad(hours), dir);
                this._clockLastH = hours;
            }
            if (this._minuteEngine && minutes !== this._clockLastM) {
                const dir = this._clockLastM !== null ? (minutes > this._clockLastM ? 1 : minutes < this._clockLastM ? -1 : 0) : 0;
                this._minuteEngine.render(pad(minutes), dir);
                this._clockLastM = minutes;
            }
            if (this._secondEngine && seconds !== this._clockLastS) {
                const dir = this._clockLastS !== null ? (seconds > this._clockLastS ? 1 : seconds < this._clockLastS ? -1 : 0) : 0;
                this._secondEngine.render(pad(seconds), dir);
                this._clockLastS = seconds;
            }

            this._applyStylesAndIcon();
            return;
        }

        if (!this._engine) {
            this._applyStylesAndIcon();
            return;
        }

        const displayValueToFormat = this.currentValue;
        let finalDisplayValue = displayValueToFormat;
        let prefix = config.prefix || '';
        let suffix = config.suffix || '';

        let minFD = config.minimumFractionDigits;
        let maxFD = config.maximumFractionDigits;
        let signDisplay = config.showSign || 'auto';
        let style;

        if (config.displayStyle === 'pillPercentage') {
            signDisplay = config.showSign || 'never';

            if (config.displayMode === 'percentage') {
                style = 'percent';
                if (minFD === undefined) minFD = 0;
                if (maxFD === undefined) maxFD = 1;
            } else if (config.displayMode === 'number') {
                const res = this._formatNumberWithMagnitude(displayValueToFormat, minFD, maxFD);
                finalDisplayValue = res.value;
                suffix = res.s;
                minFD = res.minF;
                maxFD = res.maxF;
            } else {
                if (minFD === undefined) minFD = 0;
                if (maxFD === undefined) maxFD = 2;
            }

            finalDisplayValue = Math.abs(finalDisplayValue);
        } else {
            if (config.displayMode === 'percentage') {
                style = 'percent';
                if (minFD === undefined) minFD = 2;
                if (maxFD === undefined) maxFD = 2;
            } else if (config.displayMode === 'number') {
                const res = this._formatNumberWithMagnitude(displayValueToFormat, minFD, maxFD);
                finalDisplayValue = res.value;
                suffix = res.s;
                minFD = res.minF;
                maxFD = res.maxF;
            } else if (config.displayMode === 'time') {
                const res = this._formatTimeDuration(displayValueToFormat, config.timeBaseUnit, minFD, maxFD);
                finalDisplayValue = res.value;
                suffix = res.s;
                minFD = res.minF;
                maxFD = res.maxF;
            } else if (config.displayMode === 'ordinal') {
                const vAbs = Math.abs(Math.floor(displayValueToFormat));
                const lastTwo = vAbs % 100;
                const lastOne = vAbs % 10;
                suffix = (lastTwo >= 11 && lastTwo <= 13) ? 'th'
                    : lastOne === 1 ? 'st'
                        : lastOne === 2 ? 'nd'
                            : lastOne === 3 ? 'rd'
                                : 'th';
                minFD = 0;
                maxFD = 0;
            }
        }

        if (config.displayStyle !== 'pillPercentage' && this.currentValue < 0) {
            if (config.negativeFormat === 'parentheses') {
                prefix = `(${prefix}`;
                suffix = `${suffix})`;
                finalDisplayValue = Math.abs(finalDisplayValue);
                signDisplay = config.showSign || 'never';
            } else if (config.negativeFormat === 'minusParentheses') {
                prefix = `(-${prefix}`;
                suffix = `${suffix})`;
                finalDisplayValue = Math.abs(finalDisplayValue);
            }
        }

        this.currentDisplayValue = finalDisplayValue;

        this._engine.setPrefix(prefix);
        this._engine.setSuffix(suffix);

        const formatted = this._engine.formatNumber(finalDisplayValue, style, minFD, maxFD, signDisplay);

        if (Object.is(finalDisplayValue, this._nfLastDisplayValue)
            && formatted === this._lastFormatted) {
            this._applyStylesAndIcon();
            return;
        }

        const direction = finalDisplayValue > this._nfLastDisplayValue ? 1
            : finalDisplayValue < this._nfLastDisplayValue ? -1
                : 0;

        const now = performance.now();
        const baseDuration = config.duration > 0 ? config.duration : 0;
        const easing = config.ease;
        let effectiveDuration = baseDuration;

        const squash = config.animationSquash;
        if (squash && squash.enabled && baseDuration > 0) {
            const minInterval = (squash.minAnimatedIntervalMs !== undefined && squash.minAnimatedIntervalMs !== null)
                ? squash.minAnimatedIntervalMs
                : baseDuration;

            let busyDuration = (squash.busySpinDurationMs !== undefined && squash.busySpinDurationMs !== null)
                ? squash.busySpinDurationMs
                : 0;

            if (busyDuration <= 0) busyDuration = 80;
            if (minInterval > 0 && (now - this._animLastFullSpinAt) < minInterval) {
                effectiveDuration = busyDuration;
            } else {
                effectiveDuration = baseDuration;
                this._animLastFullSpinAt = now;
            }

            if (effectiveDuration < 0) effectiveDuration = 0;
            if (effectiveDuration > baseDuration) effectiveDuration = baseDuration;
        }

        this._engine.setTiming(effectiveDuration, easing);
        this._engine.render(formatted, direction);

        this._nfLastDisplayValue = finalDisplayValue;
        this._lastFormatted = formatted;
        this._nfLastSpinDuration = effectiveDuration;
        this._nfLastSpinEasing = easing;

        this._applyStylesAndIcon();
    }

    setValue(newValue) {
        this.update(newValue);
    }

    setConfig(newOptions) {
        if (this._disabled || this._destroyed) return;

        const _nestedKeys = [
            'valueBasedBackgroundColor',
            'valueBasedIcon',
            'valueBasedTextColor',
            'magnitudeSuffixes',
            'timeUnits',
            'pillStyle',
            'digitalClockStyle',
            'animationSquash'
        ];

        for (const key of _nestedKeys) {
            if (newOptions && newOptions[key]) {
                this.config[key] = { ...this.config[key], ...newOptions[key] };
            }
        }
        if (newOptions) {
            for (const key of Object.keys(newOptions)) {
                if (!_nestedKeys.includes(key)) {
                    this.config[key] = newOptions[key];
                }
            }
        }

        if (newOptions && (newOptions.magnitudeSuffixes || newOptions.timeUnits)) {
            this._rebuildCaches();
        }

        const config = this.config;

        if (this._engine) {
            if (newOptions && (newOptions.duration !== undefined || newOptions.ease !== undefined)) {
                this._engine.setTiming(config.duration, config.ease);
                this._nfLastSpinDuration = config.duration > 0 ? config.duration : 0;
                this._nfLastSpinEasing = config.ease;
                this._animLastFullSpinAt = 0;
            }
        }

        if (config.displayStyle === 'digitalClock') {
            const d = config.duration > 0 ? Math.min(config.duration, 300) : 0;
            if (this._hourEngine) {
                if (newOptions && (newOptions.duration !== undefined || newOptions.ease !== undefined)) {
                    this._hourEngine.setTiming(d, config.ease);
                }
            }
            if (this._minuteEngine) {
                if (newOptions && (newOptions.duration !== undefined || newOptions.ease !== undefined)) {
                    this._minuteEngine.setTiming(d, config.ease);
                }
            }
            if (this._secondEngine) {
                if (newOptions && (newOptions.duration !== undefined || newOptions.ease !== undefined)) {
                    this._secondEngine.setTiming(d, config.ease);
                }
            }
        }

        if (config.displayStyle === 'pillPercentage' && this.container) {
            this.container.style.fontSize = config.pillStyle.fontSize;

            if (this.arrowElement) {
                const svgInArrow = this.arrowElement.querySelector('svg');
                if (svgInArrow) svgInArrow.style.strokeWidth = config.pillStyle.arrowStrokeWidth;
            }
        } else if (config.displayStyle === 'digitalClock' && this.container) {
            this.container.style.fontSize = config.digitalClockStyle.fontSize;
            const seps = this._clockSeparators;
            for (let i = 0; i < seps.length; i++) {
                seps[i].style.color = config.digitalClockStyle.separatorColor;
            }
        }

        this.update(this.currentValue !== null ? this.currentValue : 0);
    }

    setExternalColorValue(value) {
        if (this._disabled || this._destroyed) return;
        if (Object.is(this.externalColorValue, value)) return;
        this.externalColorValue = value;
        this._applyStylesAndIcon();
    }

    startClock(options = {}) {
        if (this._disabled || this._destroyed) return;

        this.stopClock();
        this.config.clockOptionsToStart = { ...this.config.clockOptionsToStart, ...options };

        let clockValueToStartWith;
        if (options.startTime !== undefined) {
            clockValueToStartWith = options.startTime;
        } else if (this.timer.currentTime !== null && options.resume) {
            clockValueToStartWith = this.timer.currentTime;
        } else {
            clockValueToStartWith = this.config.initialClockTime !== null ? this.config.initialClockTime : this.currentValue;
        }

        if (clockValueToStartWith instanceof Date) {
            this.timer.startTime = clockValueToStartWith.getHours() * 3600 + clockValueToStartWith.getMinutes() * 60 + clockValueToStartWith.getSeconds();
        } else {
            this.timer.startTime = (typeof clockValueToStartWith === 'number') ? clockValueToStartWith : 0;
        }

        this.timer.currentTime = this.timer.startTime;
        this.timer.direction = options.direction || this.timer.direction || 'down';
        this.timer.endTime = (options.endTime !== undefined) ? options.endTime : this.timer.endTime;
        if (options.thresholds) this.config.clockThresholds = options.thresholds;

        this.update(this.timer.currentTime);
        this.timer.running = true;

        // FIX [B7]: Replace setInterval with setTimeout + drift compensation.
        // setInterval(fn, 1000) accumulates drift because each callback's execution
        // time adds to the interval. Over 60 seconds, clocks can drift 100-500ms.
        // setTimeout with drift compensation measures the actual elapsed time and
        // adjusts the next timeout to stay synchronized.
        this._clockExpectedTick = Date.now() + 1000;

        const tick = () => {
            if (!this.timer.running) return;

            const increment = 1;

            if (this.timer.direction === 'down') {
                this.timer.currentTime -= increment;
                if (this.timer.endTime !== null && this.timer.currentTime < this.timer.endTime) {
                    this.timer.currentTime = this.timer.endTime;
                    this.stopClock();
                }
            } else {
                this.timer.currentTime += increment;
                if (this.timer.endTime !== null && this.timer.currentTime > this.timer.endTime) {
                    this.timer.currentTime = this.timer.endTime;
                    this.stopClock();
                }
            }

            if (this.timer.running) {
                this.update(this.timer.currentTime);

                // Schedule next tick with drift compensation
                const now = Date.now();
                const drift = now - this._clockExpectedTick;
                this._clockExpectedTick += 1000;
                const nextDelay = Math.max(0, 1000 - drift);
                this.timer.id = setTimeout(tick, nextDelay);
            }
        };

        this.timer.id = setTimeout(tick, 1000);
    }

    stopClock() {
        if (this.timer.id) {
            // FIX [B7 cont]: Use clearTimeout instead of clearInterval
            clearTimeout(this.timer.id);
            this.timer.id = null;
        }
        this.timer.running = false;
        this._applyStylesAndIcon();
    }

    resetClock(newStartTime) {
        this.stopClock();

        let timeToResetTo;
        if (newStartTime !== undefined) {
            timeToResetTo = newStartTime;
        } else {
            timeToResetTo = this.config.initialClockTime !== null ? this.config.initialClockTime : 0;
        }

        if (timeToResetTo instanceof Date) {
            this.timer.currentTime = timeToResetTo.getHours() * 3600 + timeToResetTo.getMinutes() * 60 + timeToResetTo.getSeconds();
        } else {
            this.timer.currentTime = (typeof timeToResetTo === 'number') ? timeToResetTo : 0;
        }

        this.timer.startTime = this.timer.currentTime;
        this.update(this.timer.currentTime);
    }

    play() {
        if (this._disabled || this._destroyed) return;

        if (!this.timer.running) {
            const optsToUse = (this.timer.currentTime !== null && this.timer.startTime !== this.timer.currentTime)
                ? { ...this.config.clockOptionsToStart, resume: true }
                : this.config.clockOptionsToStart;
            this.startClock(optsToUse || {});
        }
    }

    pause() {
        this.stopClock();
    }

    randomize() {
        if (typeof this.update !== 'function') return;

        let randomVal;
        const config = this.config;
        const mode = config.displayMode;
        const style = config.displayStyle;

        if (style === 'digitalClock') {
            randomVal = Math.floor(Math.random() * 86400);
            if (this.timer.running) {
                this.stopClock();
            }
        } else if (mode === 'percentage') {
            randomVal = Math.random() * 2 - 1;
        } else if (mode === 'ordinal') {
            randomVal = Math.floor(Math.random() * 100) + 1;
        } else if (mode === 'time') {
            randomVal = Math.random() * 7200;
            if (config.timeBaseUnit === 'ms') randomVal *= 1000;
            else if (config.timeBaseUnit === 'min') randomVal /= 60;
        } else {
            const currentNumericValue = (typeof this.currentValue === 'number') ? this.currentValue : 0;
            let maxMagnitude = 1000;
            if (Math.abs(currentNumericValue) > 1e9) maxMagnitude = 2e12;
            else if (Math.abs(currentNumericValue) > 1e6) maxMagnitude = 2e9;
            else if (Math.abs(currentNumericValue) > 1e3) maxMagnitude = 2e6;

            randomVal = (Math.random() * maxMagnitude * 2 - maxMagnitude);
            if (Math.random() < 0.3 && maxMagnitude > 2000) {
                randomVal = (Math.random() * 2000 - 1000);
            } else if (maxMagnitude <= 2000) {
                randomVal = (Math.random() * maxMagnitude) * (Math.random() < 0.5 ? 1 : -1);
            }
        }

        this.update(randomVal);
    }

    destroy() {
        if (this._destroyed) return;

        this.stopClock();

        if (this._iconTimerId) {
            clearTimeout(this._iconTimerId);
            this._iconTimerId = null;
        }

        if (this._engine) this._engine.destroy();
        if (this._hourEngine) this._hourEngine.destroy();
        if (this._minuteEngine) this._minuteEngine.destroy();
        if (this._secondEngine) this._secondEngine.destroy();

        this._destroyed = true;
    }
}
