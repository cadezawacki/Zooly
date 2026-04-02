
import {roundTo as rounderTo, roundToUp as rounderToUp, roundToDown as rounderToDown} from 'round-to';
import { isNumber, toNumber, isBoolean } from 'lodash';

// Utility.formatNumber(x, {postfix: "%", sigFigs(normal: 1)})
// window.coerceToNumber = coerceToNumber;
export const CLEAR_SENTINEL = -9999999999;

export const roundToUp = rounderToUp;
export const roundToDown = rounderToDown;

export function roundToNumeric(value, sigFigs) {
    const v = NumberFormatter.formatNumber(value, {sigFigs: {global: sigFigs}});
    return coerceToNumeric(v)
}

export function coerceToNumeric(value, {onNaN=null, emptyStringIsZero=false}={}){return coerceToNumber(value, onNaN, emptyStringIsZero)}

export function coerceToNumber(value, onNaN = null, emptyStringIsZero = false, suppressPercent=false) {
    if (value == null) return onNaN;
    if (value === CLEAR_SENTINEL) return CLEAR_SENTINEL;
    switch (typeof value) {
        case 'number':
            return value === value ? value : onNaN;
        case 'bigint':
            return Number(value);
        case 'string': {
            let s = value.trim();
            if (s === '') return emptyStringIsZero ? 0 : onNaN;
            if (s.indexOf(',') !== -1) s = s.replace(/,/g, '');
            if (s.indexOf('..') !== -1) s = s.replace(/\.{2,}/g, ".");
            let isPercent = false;
            if (!suppressPercent && s.indexOf('%') !== -1) {
                s = s.replace(/%/g, '');
                isPercent = true;
            }
            const n = +s;
            return n === n ? (isPercent ? n/100 : n) : onNaN;
        }
        case 'boolean':
            return value ? 1 : 0;
        case 'symbol':
            return onNaN;
        default:
            try {
                const n = +value;
                return n === n ? n : onNaN;
            } catch {
                return onNaN;
            }
    }
}

export function coerceToBool(v, onNaN=null) {
    if (isBoolean(v)) return Boolean(v)
    if ((typeof v === 'string') && ['true','false', 'y', 'n', '✓', '✗', 'x'].includes(v.toLowerCase())) {
        v = v.toLowerCase();
        return (v === 'true') || (v === 'y') || (v === '✓')
    }
    const t = coerceToNumber(v, onNaN);
    return t !== null ? Boolean(t) : false
}

export function roundTo(x, s=3) {
    const t = coerceToNumber(x);
    if (typeof x === 'bigint') {
        x = Number(x);
    }
    return t === null ? null : rounderTo(toNumber(x), s).toLocaleString()
}

export class NumberFormatter {
    static defaultConfig = {
        showSign: false,
        prefix: '',
        postFix: '',
        sigFigs: {
            global: null, //overrides all
            subOne: null,
            normal: 4,    // 0 means all digits
            thousand: 2,
            million: 1,
            billion: 1
        },
        thresholds: {
            thousand: 1000,
            million: 1000000,
            billion: 1000000000
        },
        units: {
            thousand: 'k',
            million: 'mio',
            billion: 'bn'
        },
        showCommas: true,
        useDivisor: true
    }

    static formatNumber(number, customConfig = {}) {
        number = Number(number);

        if (Number.isNaN(number) || !Number.isFinite(number)) {
            return String(number);
        }

        const config = { ...this.defaultConfig, ...customConfig };
        const absNum = Math.abs(number);
        const sign = number < 0 ? '-' : (config.showSign ? '+' : '');
        const prefix = config.prefix || '';
        const postfix = config.postfix || '';
        const spacing = ' '.repeat(config.spacing || 0);
        const useDivisor = config.useDivisor ?? true;
        const showCommas = config.showCommas ?? true;

        if (typeof config.sigFigs !== 'object') {
            const g = config.sigFigs;
            config.sigFigs = {global: g};
        }

        let r;
        if (absNum < config.thresholds.thousand) {

            const my_sigs = (config.sigFigs.global ?? ((absNum < 1) && (absNum !== 0) ? config.sigFigs.subOne : null ) ?? config.sigFigs.normal ?? 0);
            r = `${prefix}${sign}${this._formatWithSigFigs(absNum, my_sigs)}${postfix}`;
        }

        else if (absNum < config.thresholds.million) {
            r = `${prefix}${sign}${this._formatWithSigFigs(absNum / (useDivisor ? 1000 : 1), config.sigFigs.global || config.sigFigs.thousand)}${spacing}${(useDivisor ? config.units.thousand : '')|| ''}${postfix}`;
        }

        else if (absNum < config.thresholds.billion) {
            r = `${prefix}${sign}${this._formatWithSigFigs(absNum / (useDivisor ? 1000000 : 1), config.sigFigs.global || config.sigFigs.million)}${spacing}${(useDivisor ? config.units.million : '')|| ''}${postfix}`;
        }

        else {
            r = `${prefix}${sign}${this._formatWithSigFigs(absNum / (useDivisor ? 1000000000 : 1), config.sigFigs.global || config.sigFigs.billion)}${spacing}${(useDivisor ? config.units.billion : '') || ''}${postfix}`;
        }

        return showCommas ? NumberFormatter.numberWithCommas(r) : r
    }

    static _formatWithSigFigs(num, sigFigs) {
        if (sigFigs === -1) return num.toString();
        return Number(num).toFixed(sigFigs);
    }

    static numberWithCommas(x) {
        return x.toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ",");
    }
}

export class IntlNumberFormatter {
    static defaultConfig = {
        showSign: false,
        prefix: '',
        postfix: '',
        sigFigs: {
            normal: 4,    // Number of decimal places
            thousand: 2,
            million: 1,
            billion: 1
        },
        thresholds: {
            thousand: 1000,
            million: 1000000,
            billion: 1000000000
        },
        units: {
            thousand: 'k',
            million: 'mio',
            billion: 'bn'
        },
        locale: 'en-US',
        spacing: 0
    };

    static formatNumber(number, customConfig = {}) {
        // Merge default config with custom config
        const config = { ...this.defaultConfig, ...customConfig };

        // Handle non-numeric values
        if (Number.isNaN(number) || !Number.isFinite(number)) {
            return String(number);
        }

        const absNum = Math.abs(number);
        const spacing = ' '.repeat(config.spacing || 0);

        // Determine scale, unit, and significant digits based on number magnitude
        let scale = 1;
        let unit = '';
        let fractionDigits = config.sigFigs.normal;

        if (absNum >= config.thresholds.billion) {
            scale = 1000000000;
            unit = config.units.billion;
            fractionDigits = config.sigFigs.billion;
        } else if (absNum >= config.thresholds.million) {
            scale = 1000000;
            unit = config.units.million;
            fractionDigits = config.sigFigs.million;
        } else if (absNum >= config.thresholds.thousand) {
            scale = 1000;
            unit = config.units.thousand;
            fractionDigits = config.sigFigs.thousand;
        }

        // Create Intl.NumberFormat with appropriate options
        const formatter = new Intl.NumberFormat(config.locale, {
            style: 'decimal',
            minimumFractionDigits: fractionDigits === 0 ? undefined : fractionDigits,
            maximumFractionDigits: fractionDigits === 0 ? 20 : fractionDigits, // If 0, allow many digits
            signDisplay: config.showSign ? 'exceptZero' : 'auto',
            useGrouping: true
        });

        // Format the scaled number and combine with prefix, unit, and postfix
        return `${config.prefix}${formatter.format(number / scale)}${spacing}${unit}${config.postfix}`;
    }


    static getIntlOptions(number, customConfig = {}) {
        const config = { ...this.defaultConfig, ...customConfig };
        const absNum = Math.abs(number);

        // Determine scale and unit for display
        let scale = 1;
        let unit = '';

        if (absNum >= config.thresholds.billion) {
            scale = 1000000000;
            unit = config.units.billion;
        } else if (absNum >= config.thresholds.million) {
            scale = 1000000;
            unit = config.units.million;
        } else if (absNum >= config.thresholds.thousand) {
            scale = 1000;
            unit = config.units.thousand;
        }

        // Determine significant digits
        let fractionDigits = config.sigFigs.normal;
        if (absNum >= config.thresholds.billion) {
            fractionDigits = config.sigFigs.billion;
        } else if (absNum >= config.thresholds.million) {
            fractionDigits = config.sigFigs.million;
        } else if (absNum >= config.thresholds.thousand) {
            fractionDigits = config.sigFigs.thousand;
        }

        return {
            // Standard Intl.NumberFormat options
            style: 'decimal',
            notation: 'standard',
            minimumFractionDigits: fractionDigits === 0 ? undefined : fractionDigits,
            maximumFractionDigits: fractionDigits === 0 ? 20 : fractionDigits,
            signDisplay: config.showSign ? 'exceptZero' : 'auto',
            useGrouping: true,

            // Additional metadata for transforming the number and adding units
            _scale: scale,
            _unit: unit,
            _prefix: config.prefix,
            _postfix: config.postfix,
            _spacing: config.spacing || 0
        };
    }
}
