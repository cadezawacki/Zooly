
import {LRUCache} from 'mnemonist';
window.LRUCache = LRUCache;

export const CLEAR_SENTINEL = -9999999999;
window.CLEAR_SENTINEL = CLEAR_SENTINEL

const LRU_CACHE_SIZE = 0;
const PATTERN_CACHE_SIZE = 0;
const BOOL_LRU_SIZE = 0;
const DATE_LRU_SIZE = 0;

const MAX_CACHE_KEY_LEN = 0;
const MAX_STR_KEY_LEN = 96;

const CC_0 = 48, CC_9 = 57;
const CC_DOT = 46;
const CC_PLUS = 43;
const CC_MINUS = 45;
const CC_COMMA = 44;
const CC_SPACE = 32;
const CC_UNDERSCORE = 95;
const CC_APOSTROPHE = 39;
const CC_TAB = 9;
const CC_NBSP = 160;
const CC_LPAREN = 40;
const CC_RPAREN = 41;
const CC_PERCENT = 37;
const CC_E_LO = 101, CC_E_HI = 69;

const _lru = LRU_CACHE_SIZE > 0 ? new LRUCache(LRU_CACHE_SIZE) : null;
function _cacheGet(k){ if(!_lru) return undefined; return _lru.get(k);}
function _cacheSet(k,v){ if(!_lru) return; _lru.set(k,v);}

function _isFiniteNumber(n){ return typeof n==='number' && n===n && n!==Infinity && n!==-Infinity; }
function _normalizeMinusCharCode(c){ return (c===0x2212||c===0x2013||c===0x2014)?CC_MINUS:c; }
function _toLowerAscii(ch){ const code=ch.charCodeAt(0); return (code>=65&&code<=90)?String.fromCharCode(code+32):ch; }
function _suffixMultiplier(tokenLower){
    switch(tokenLower){
        case 'k': case 'm': case 'thousand': return 1e3;
        case 'mm': case 'mil': case 'mio': case 'million': return 1e6;
        case 'b': case 'bn': case 'bil': case 'billion': return 1e9;
        case 't': case 'tn': case 'trillion': return 1e12;
        case '%': case 'percent': case 'pct': return 0.01;
        default: return 1;
    }
}
function pf(n) {return Math.round(n * 1e15) / 1e15}
function _pad2(n){ return n < 10 ? '0' + n : '' + n; }
function _pad3(n){ return n < 10 ? '00' + n : (n < 100 ? '0' + n : '' + n); }

function _toLowerAsciiStr(s){
    let out = '', c = 0;
    for (let i = 0, n = s.length; i < n; i++) {
        c = s.charCodeAt(i);
        if (c >= 65 && c <= 90) out += String.fromCharCode(c + 32);
        else out += s.charAt(i);
    }
    return out;
}
function _extractTrailingAlphaToken(s){
    let i=s.length-1;
    for(;i>=0;i--){
        const c=s.charCodeAt(i);
        if(c===CC_SPACE||c===CC_TAB||c===CC_NBSP||c===CC_COMMA||c===CC_UNDERSCORE||c===CC_APOSTROPHE) continue;
        break;
    }
    let end=i;
    for(;i>=0;i--){
        const c=s.charCodeAt(i);
        const isUpper=c>=65&&c<=90, isLower=c>=97&&c<=122, isPct=c===CC_PERCENT;
        if(!(isUpper||isLower||isPct)) break;
    }
    const start=i+1;
    if(start<=end){
        const token=s.slice(start,end+1);
        let out=''; for(let j=0,n=token.length;j<n;j++) out+=_toLowerAscii(token[j]);
        return out;
    }
    return '';
}
function _parseLooseNumericString(s){
    s=s.trim(); if(s.length===0) return NaN;
    if(s.length<=MAX_CACHE_KEY_LEN){ const hit=_cacheGet(s); if(hit!==undefined) return hit; }

    let i=0,n=s.length, accountingNegative=false;
    if(n>=2 && s.charCodeAt(0)===CC_LPAREN && s.charCodeAt(n-1)===CC_RPAREN){ accountingNegative=true; i=1; n=n-1; }

    let negative=false, seenDigit=false, seenDot=false, seenExp=false, expSignAllowed=false, expHasDigit=false;
    const out=new Array(n); let outLen=0;

    for(;i<n;i++){
        let c=_normalizeMinusCharCode(s.charCodeAt(i));
        if(c===CC_SPACE||c===CC_TAB||c===CC_NBSP||c===CC_UNDERSCORE||c===CC_COMMA||c===CC_APOSTROPHE) continue;
        if(!seenDigit && !seenDot){
            if(c===CC_PLUS) continue;
            if(c===CC_MINUS){ negative=true; continue; }
        }
        if(c>=CC_0 && c<=CC_9){ out[outLen++]=s.charAt(i); seenDigit=true; break; }
        if(c===CC_DOT && !seenDot){ out[outLen++]='.'; seenDot=true; break; }
    }

    for(i=i+1;i<n;i++){
        let c=_normalizeMinusCharCode(s.charCodeAt(i));
        if(c===CC_SPACE||c===CC_TAB||c===CC_NBSP||c===CC_UNDERSCORE||c===CC_COMMA||c===CC_APOSTROPHE) continue;
        if(c>=CC_0 && c<=CC_9){ out[outLen++]=s.charAt(i); if(seenExp) expHasDigit=true; else seenDigit=true; continue; }
        if(c===CC_DOT){ if(!seenExp && !seenDot){ out[outLen++]='.'; seenDot=true; } continue; }
        if((c===CC_E_LO||c===CC_E_HI) && !seenExp && seenDigit){ out[outLen++]='e'; seenExp=true; expSignAllowed=true; expHasDigit=false; continue; }
        if(seenExp && expSignAllowed && (c===CC_PLUS||c===CC_MINUS)){ out[outLen++]=(c===CC_MINUS?'-':'+'); expSignAllowed=false; continue; }
        if(seenExp) expSignAllowed=false;
    }

    if(outLen===0){ const res=NaN; if(s.length<=MAX_CACHE_KEY_LEN) _cacheSet(s,res); return res; }

    if(out[0]==='.') { for(let k=outLen;k>0;k--) out[k]=out[k-1]; out[0]='0'; outLen++; }
    else if(outLen>=2 && out[0]==='-' && out[1]==='.') { for(let k=outLen;k>1;k--) out[k]=out[k-1]; out[1]='0'; outLen++; }

    if(outLen>0 && out[outLen-1]==='e'){ outLen--; }
    else if(seenExp && !expHasDigit){ while(outLen>0 && out[outLen-1]!=='e') outLen--; if(outLen>0 && out[outLen-1]==='e') outLen--; }

    const numStr=out.slice(0,outLen).join('');
    let val=+numStr;
    if(!_isFiniteNumber(val)) val=NaN;
    if(_isFiniteNumber(val) && (negative||accountingNegative)) val=-val;

    if(_isFiniteNumber(val)){
        const suffix=_extractTrailingAlphaToken(s);
        if(suffix){
            const mul=_suffixMultiplier(suffix);
            if(mul!==1){
                val=val*mul;
                if(!Number.isFinite(val)) val=NaN;
                else if (mul < 1) val = pf(val);
            }
        }
    }

    if(s.length<=MAX_CACHE_KEY_LEN) _cacheSet(s,val);
    return val;
}

export function coerceToNumber(value, opts){
    if (value === CLEAR_SENTINEL) return opts?.onCLEAR!==undefined ? opts.onCLEAR : CLEAR_SENTINEL;
    const nullishAsZero=!!(opts&&opts.nullishAsZero);
    const t=typeof value;
    if(t==='number') return _isFiniteNumber(value)?value:(opts?.onNaN!==undefined?opts.onNaN:NaN);
    if(t==='boolean') return value?1:0;
    if(t==='bigint'){ const v=Number(value); return _isFiniteNumber(v)?v:(opts?.onNaN!==undefined?opts.onNaN:NaN); }
    if(t==='string'){ if(value.length===0) return (opts?.onNaN!==undefined?opts.onNaN:NaN); return _parseLooseNumericString(value); }
    if(value==null) return nullishAsZero?0:(opts?.onNaN!==undefined?opts.onNaN:NaN);

    const prim=(t==='object'||t==='function')?(value.valueOf!==Object.prototype.valueOf?value.valueOf():value):value;
    if(typeof prim==='number') return _isFiniteNumber(prim)?prim:(opts?.onNaN!==undefined?opts.onNaN:NaN);
    if(typeof prim==='bigint'){ const v=Number(prim); return _isFiniteNumber(v)?v:(opts?.onNaN!==undefined?opts.onNaN:NaN); }
    if(typeof prim==='string') return _parseLooseNumericString(prim);
    return (opts?.onNaN!==undefined?opts.onNaN:NaN);
}
export function clearCoercionCache(){ if(_lru) _lru.clear(); }

//////////////////////////////
// ---- Formatting ----    //
//////////////////////////////

const FORMAT_CACHE_DEFAULT = 0; // per-config formatted-string LRU (0 = off)
const MAX_FRACTION_DEN = 8;

const CURRENCY_SYMBOL = Object.freeze({
    USD:'$', DOLLAR:'$', USA: "$", US: "$", "$":"$",
    AUD:'A$', CAD:'C$', NZD:'NZ$', HKD:'HK$', SGD:'S$',
    EUR:'€', EURO:'€','€':'€',
    GBP:'£', GDP:'£', POUND:'£', GB:'£', "£":"£",
    JPY:'¥', YEN:'¥','¥':'¥', CNY:'¥', CNH:'¥',
    KRW:'₩', INR:'₹', RUB:'₽', CHF:'CHF',
    SEK:'kr', NOK:'kr', DKK:'kr',
    MXN:'$', BRL:'R$', ZAR:'R', TRY:'₺', TWD:'NT$', IDR:'Rp'
});

const LOCALE_SEPS = Object.freeze({
    'en': { thousandsSep: ',', decimalSep: '.' },
    'eu': { thousandsSep: '.', decimalSep: ',' }
});

function _autoSigFigs(v) {
    if (isInteger(v)) return 0
    return _autoSigFigsFloat(v)
}

function _autoSigFigsFloat(absVal){
    if(absVal>=1e12) return 3;
    if(absVal>=1e9)  return 3;
    if(absVal>=1e6)  return 3;
    if(absVal>=1e3)  return 3;
    if(absVal>=10)   return 3;
    if(absVal>=1)    return 3;
    return 3;
}

// Compiled caches
const _thresholdCache = typeof WeakMap!=='undefined' ? new WeakMap() : null; // cfg -> fn
const _formatCache = typeof WeakMap!=='undefined' ? new WeakMap() : null;     // cfg -> LRU Map
const _patternCache = PATTERN_CACHE_SIZE > 0 ? new LRUCache(PATTERN_CACHE_SIZE) : null;

function _abs(n){ return n<0?-n:n; }

function _pow10(n){
    switch(n){
        case 0: return 1; case 1: return 10; case 2: return 100; case 3: return 1000; case 4: return 10000;
        case 5: return 100000; case 6: return 1000000; case 7: return 10000000; case 8: return 100000000;
        case 9: return 1000000000; case 10: return 10000000000; case 11: return 100000000000; case 12: return 1000000000000;
        default: return Math.pow(10, n);
    }
}

function _applyRounding(v, dp, mode){
    if(!Number.isFinite(v)) return v;
    const m = mode==='up' ? Math.ceil : (mode==='down' ? Math.floor : Math.round);
    if(dp===0) return m(v);
    if(dp>0){
        const scale = _pow10(dp);
        return m(v*scale)/scale;
    } else {
        const pow = _pow10(-dp);
        return m(v/pow)*pow;
    }
}

// Grouping utilities
function _groupInteger3(intStr, sep){
    let len=intStr.length;
    if(len<=3) return intStr;
    const cap = len + ((len-1)/3|0);
    const out = new Array(cap|0); let o=cap|0; let i=len; let c=0;
    while(i>0){
        if(c===3){ out[--o]=sep; c=0; } else { out[--o]=intStr.charAt(--i); c++; }
    }
    return out.slice(o).join('');
}
function _groupIntegerVar(intStr, sep, sizes){
    // sizes: e.g., [3,2] => first group 3 from right, then 2,2,...
    if(!sizes || sizes.length===0 || (sizes.length===1 && sizes[0]===3)) return _groupInteger3(intStr, sep);
    let i=intStr.length, first=true, sizeIdx=0, curSize=sizes[0], count=0;
    let chunks=[]; let partEnd=i;
    while(i>0){
        i--; count++;
        if(count===curSize){
            chunks.push(intStr.slice(i, partEnd));
            partEnd = i;
            count=0;
            if(first){ first=false; sizeIdx=1; curSize=sizes[1]||sizes[0]; } else { curSize=sizes[sizeIdx]||curSize; }
        }
    }
    if(partEnd>0) chunks.push(intStr.slice(0, partEnd));
    chunks.reverse();
    return chunks.join(sep);
}

function _trimDecimals(decStr, minKeep){
    let end = decStr.length;
    while(end>minKeep && decStr.charCodeAt(end-1)===48) end--;
    return end===0 ? '' : decStr.slice(0,end);
}

// Pattern compilation & formatting
function _compilePattern(pat){
    let hasGroup=false, dot=-1, intZeros=0, i=0, n=pat.length;
    for(; i<n; i++){
        const ch=pat.charAt(i);
        if(ch===','){ hasGroup=true; continue; }
        if(ch==='.'){ dot=i; break; }
        if(ch==='0') intZeros++;
    }
    let minDec=0, maxDec=0;
    if(dot!==-1){
        for(i=dot+1;i<n;i++){
            const ch=pat.charAt(i);
            if(ch==='0'){ minDec++; maxDec++; }
            else if(ch==='#'){ maxDec++; }
        }
    }
    return { hasGroup, intZeros, minDec, maxDec };
}

function _formatWithPattern(absVal, sign, compiled, thousandsSep, decimalSep, groupSizes){
    const maxDec = compiled.maxDec;
    const scale = maxDec>0 ? _pow10(maxDec) : 1;
    let temp = maxDec>0 ? Math.round(absVal*scale)/scale : Math.round(absVal);
    let s = temp.toString();
    let dotPos = s.indexOf('.');
    let intPart = dotPos===-1 ? s : s.slice(0,dotPos);
    let decPart = dotPos===-1 ? '' : s.slice(dotPos+1);

    // int zeros
    if(intPart.length<compiled.intZeros){
        const pad = compiled.intZeros - intPart.length;
        let z=''; for(let k=0;k<pad;k++) z+='0';
        intPart = z + intPart;
    }
    if(compiled.hasGroup) intPart = _groupIntegerVar(intPart, thousandsSep, groupSizes);

    if(maxDec>0){
        if(decPart.length<compiled.minDec){
            let z=''; for(let k=decPart.length;k<compiled.minDec;k++) z+='0';
            decPart += z;
        } else if(decPart.length>maxDec){
            decPart = decPart.slice(0,maxDec);
        }
    } else decPart='';

    return sign + (decPart ? intPart + decimalSep + decPart : intPart);
}

// Unicode fraction approx
function _bestUnicodeFraction(x){
    const UNI = {
        '1/2':'½','1/3':'⅓','2/3':'⅔','1/4':'¼','3/4':'¾',
        '1/5':'⅕','2/5':'⅖','3/5':'⅗','4/5':'⅘',
        '1/6':'⅙','5/6':'⅚','1/7':'⅐',
        '1/8':'⅛','3/8':'⅜','5/8':'⅝','7/8':'⅞',
        '1/9':'⅑','1/10':'⅒'
    };
    let bestN=0, bestD=1, bestErr=1;
    for(let d=2; d<=MAX_FRACTION_DEN; d++){
        const n = Math.round(x*d);
        const err = Math.abs(x - n/d);
        if(err<bestErr){ bestErr=err; bestN=n; bestD=d; if(err===0) break; }
    }
    if(bestN===0) return '';
    const key = bestN+'/'+bestD;
    return UNI[key] || key;
}

// Threshold compilation (sigFigs as dp)
function _compileThreshold(cfg){
    const sf = cfg.sigFigs;
    if(typeof sf==='number' && sf===sf) return function(){ return sf|0; };
    if(typeof sf!=='object' || sf==null) return function(){ return 0; };

    const rules = [];
    for(const k in sf){
        const val = sf[k]|0;
        const s = k.replace(/\s+/g,'');
        let idx;
        if((idx=s.indexOf('-'))>0){
            const a = Number(s.slice(0,idx)), b=Number(s.slice(idx+1));
            if(a===a && b===b) rules.push({type: 'range', a:a, b:b, dp: val});
        } else if(s.endsWith('+')){
            const a = Number(s.slice(0,-1)); if(a===a) rules.push({type:'ge', a:a, dp:val});
        } else if(s.startsWith('>=')){
            const a = Number(s.slice(2)); if(a===a) rules.push({type:'ge', a:a, dp:val});
        } else if(s.startsWith('<=')){
            const b = Number(s.slice(2)); if(b===b) rules.push({type:'le', b:b, dp:val});
        } else if(s.startsWith('>')){
            const a = Number(s.slice(1)); if(a===a) rules.push({type:'gt', a:a, dp:val});
        } else if(s.startsWith('<')){
            const b = Number(s.slice(1)); if(b===b) rules.push({type:'lt', b:b, dp:val});
        }
    }
    rules.sort((r1,r2)=>{
        const rank = {range:0, ge:1, le:1, gt:2, lt:2};
        const d = rank[r1.type]-rank[r2.type];
        if(d) return d;
        if(r1.type==='ge'||r1.type==='gt') return r2.a - r1.a;
        if(r1.type==='le'||r1.type==='lt') return r1.b - r2.b;
        return (r1.b-r1.a)-(r2.b-r1.a);
    });
    return function(n){
        for(let i=0;i<rules.length;i++){
            const r=rules[i];
            if(r.type==='range'){ if(n>=r.a && n<r.b) return r.dp; }
            else if(r.type==='ge'){ if(n>=r.a) return r.dp; }
            else if(r.type==='le'){ if(n<=r.b) return r.dp; }
            else if(r.type==='gt'){ if(n>r.a) return r.dp; }
            else if(r.type==='lt'){ if(n<r.b) return r.dp; }
        }
        return 0;
    };
}

function _getDpForValue(cfg, absVal){
    if(!_thresholdCache) return _compileThreshold(cfg)(absVal);
    let fn = _thresholdCache.get(cfg);
    if(!fn){ fn = _compileThreshold(cfg); _thresholdCache.set(cfg, fn); }
    return fn(absVal);
}

function _resolveSeps(cfg){
    if(cfg && cfg.thousandsSep && cfg.decimalSep) return {thousandsSep:cfg.thousandsSep, decimalSep:cfg.decimalSep};
    if(cfg && cfg.locale && LOCALE_SEPS[cfg.locale]) return LOCALE_SEPS[cfg.locale];
    if (!cfg && navigator?.languages?.includes('en')) return LOCALE_SEPS.en;
    if (!cfg && navigator?.languages?.includes('eu')) return LOCALE_SEPS.eu;
    return LOCALE_SEPS.en;
}

function _applyCurrency(cfg){
    if(!cfg || !cfg.asCurrency) return null;
    let sym = null;
    if(cfg.asCurrency===true) sym = CURRENCY_SYMBOL.USD;
    else if(typeof cfg.asCurrency==='string'){
        if(cfg.asCurrency.length===1 || cfg.asCurrency==='CHF') sym = cfg.asCurrency;
        else {
            const code = cfg.asCurrency.toUpperCase();
            sym = CURRENCY_SYMBOL[code] || code;
            if (cfg.currencySpace === 'auto') {
                cfg.currencySpace = sym === code
            }
        }
    }
    return sym;
}

function _autoDivisor(absVal){
    if(absVal>=1e12) return 'tn';
    if(absVal>=1e9)  return 'bn';
    if(absVal>=1e6)  return 'mio';
    if(absVal>=1e3)  return 'k';
    return '';
}
function _applyDivisorExplicit(v, div){
    switch(div){
        case 'k': return {value: v/1e3, unit:'k'};
        case 'mio': return {value: v/1e6, unit:'mio'};
        case 'bn': return {value: v/1e9, unit:'bn'};
        case 'tn': return {value: v/1e12, unit:'tn'};
        default: return {value:v, unit:''};
    }
}

const _allFormatLRUSlots = new Set();

// Per-config LRU
function _getFormatLRU(cfg){
    return null // THIS IS BROKEN AND EXPLODES THE MEMORY
    const cap = (cfg && typeof cfg.formatCacheSize==='number') ? (cfg.formatCacheSize|0) : FORMAT_CACHE_DEFAULT;
    if(!_formatCache || cap<=0) return null;
    let slot = _formatCache.get(cfg);
    if(!slot || slot._cap!==cap){
        slot = { map:new LRUCache(cap), _cap: cap };
        _formatCache.set(cfg, slot);
        _allFormatLRUSlots.add(slot);
    }
    return slot;
}
function _fmtCacheGet(slot, key){
    if(!slot) return undefined;
    const m=slot.map;
    return m.get(key);
}
function _fmtCacheSet(slot, key, value){
    if(!slot) return;
    const m=slot.map;
    m.set(key, value);
}

// Make a compact cache key based on post-rounding value and key flags
function _makeCacheKey(absKey, flags){
    return absKey + '|' + flags;
}

function _canonicalAbsKey(rounded, dp){
    if(dp<=0) return String(Math.abs(rounded)|0);
    // toFixed alloc already used later; for cache we recompute small dp only
    return Math.abs(rounded).toFixed(dp);
}

function _composeFlags(isNeg, dp, roundMode, group, asFrac, paren, signPlus, unit, patternId){
    // Use 1-char flags to stay short
    // n/p for sign; r:0/1/2; g:0/1; f:0/1; p:0/1; s:0/1; u:unit; d:dp; t:patternId
    return (isNeg?'n':'p') +
        'r' + (roundMode==='up'?1:(roundMode==='down'?2:0)) +
        'g' + (group?1:0) +
        'f' + (asFrac?1:0) +
        'p' + (paren?1:0) +
        's' + (signPlus?1:0) +
        'u' + unit +
        'd' + dp +
        't' + (patternId||'');
}

export function formatNumber(value, cfg){
    if (value === CLEAR_SENTINEL) return 'CLEAR';
    cfg = {...cfg} || {};

    // 1) Coerce
    let num = (typeof value==='number' || typeof value==='bigint' || typeof value==='boolean') ? coerceToNumber(value) : coerceToNumber(value, { nullishAsZero: cfg.nullishAsZero===true });
    let isNaNVal = !(num===num);

    // 2) NaN policy
    if(isNaNVal){
        const onNaN = cfg.onNaN;
        if(onNaN === 'zero' || onNaN === 0 || onNaN === '0') num = 0;
        else if(typeof onNaN === 'string') return onNaN;
        else if(onNaN === null) return null;
        else return '';
    }

    // 3) Percent scaling
    let suffixParts = [];
    if(cfg.asPercent){ num = num * 100; suffixParts.push('%'); }
    if(cfg.spacing) {suffixParts.push(' ')}

    // 4) Auto/Explicit divisor (default auto)
    let divisor = (!cfg.divisor)?'' : (cfg.divisor || 'auto');
    divisor = divisor === true ? 'auto' : divisor;
    let chosenUnit = '';
    if(divisor==='auto'){ chosenUnit = _autoDivisor(Math.abs(num)); }
    else if(typeof divisor==='string'){ chosenUnit = divisor; if(chosenUnit==='') chosenUnit=''; }
    if(chosenUnit){
        const res = _applyDivisorExplicit(num, chosenUnit);
        num = res.value;
        chosenUnit = res.unit; // normalize
        if(chosenUnit) suffixParts.push(chosenUnit);
    }

    // Apply Defaults
    cfg = {...cfg};
    cfg.currencySpace=cfg.currencySpace!==undefined?cfg.currencySpace:'auto';
    cfg.group=cfg.group!==undefined?cfg.group:true;
    cfg.signOnZero=cfg.signOnZero!==undefined?cfg.signOnZero:cfg.showSign;


    // 5) Clamp/underflow display (evaluate on |num|)
    const absPreRound = Math.abs(num);
    if(cfg.clampBelow!==undefined && cfg.showBelow!==undefined && absPreRound>0 && absPreRound<cfg.clampBelow){
        const out = (cfg.showBelow || '<' + String(cfg.clampBelow));
        const currencySym = _applyCurrency(cfg);
        const {thousandsSep, decimalSep} = _resolveSeps(cfg);
        const space = cfg.currencySpace ? ' ' : '';
        const pre = (currencySym?currencySym+(space):'') + (cfg.prefix||'');
        const suf = (cfg.suffix||'') + (suffixParts.length?suffixParts.join(''):'');
        const sign = (cfg.showSign && !cfg.parenthesesOnNegative) ? '+' : ''; // tiny positives
        const core = sign + out;
        if(cfg.returnParts) return { prefix: pre, body: core, suffix: suf };
        return pre + core + suf;
    }

    // 6) Zero policy (after scaling/divisor/clamp)
    if(num===0 && cfg.onZero!==undefined){
        if(typeof cfg.onZero === 'string') {
            if (cfg.onZero === 'null') return null;
            if (cfg.onZero === 'zero') return cfg.signOnZero ? '+0' : 0;
            return cfg.onZero;
        }
        if(cfg.onZero === null) return null;
        if (cfg.onZero === 0) return cfg.signOnZero ? '+0' : 0
        return cfg.onZero;
    }

    // 7) Determine dp from sigFigs/threshold (negative dp => powers of 10)
    const absVal = Math.abs(num);
    let dp = 3;
    if(typeof cfg.sigFigs === 'number' && cfg.sigFigs===cfg.sigFigs){
        dp = cfg.sigFigs|0;
    } else if(cfg.sigFigs){
        dp = _getDpForValue(cfg, absVal);
    } else if (cfg.sigFigs !== null) {
        dp = _autoSigFigs(absVal);
    }

    const roundMode = cfg.roundStrategy==='up'?'up':(cfg.roundStrategy==='down'?'down':'auto');

    // 8) Rounding
    let rounded = _applyRounding(num, dp, roundMode);

    // 9) Negative zero normalization
    if(cfg.normalizeNegativeZero!==false){
        if(rounded===0) rounded = 0; // remove -0
    }

    // 10) Resolve separators and currency prefix
    const {thousandsSep, decimalSep} = _resolveSeps(cfg);
    const currencySym = _applyCurrency(cfg);
    const space = cfg.currencySpace ? ' ' : '';

    // 11) Sign handling for display
    let isNeg = rounded<0;
    let showPlus = (!!cfg.showSign) && (!isNeg ? (rounded!==0 || !!cfg.signOnZero) : false);

    // 12) Pattern compilation if provided
    let patternId = '';
    let compiledPattern = null;
    if(cfg.stringFmt){
        patternId = cfg.stringFmt;
        compiledPattern = _patternCache.get(patternId);
        if(!compiledPattern){ compiledPattern = _compilePattern(patternId); _patternCache.set(patternId, compiledPattern); }
    }

    // 13) Per-config LRU (opt-in or default)
    const lru = _getFormatLRU(cfg);
    const unitForKey = chosenUnit || '';
    const dpForDisplay = dp<0 ? 0 : dp|0;
    const absKey = _canonicalAbsKey(rounded, dpForDisplay);
    const cacheFlags = _composeFlags(isNeg, dpForDisplay, roundMode, !!cfg.group, !!cfg.asFraction, !!cfg.parenthesesOnNegative, showPlus, unitForKey, compiledPattern?patternId:'');
    const cacheKey = _makeCacheKey(absKey, cacheFlags);

    const cached = _fmtCacheGet(lru, cacheKey);
    if(cached !== undefined){
        if(cfg.returnParts){
            // Recompute prefix/suffix only (cheap), cached contains just BODY
            const pre = (currencySym?currencySym+(space):'') + (cfg.prefix||'');
            const suf = (cfg.suffix||'') + (suffixParts.length?suffixParts.join(''):'');
            return { prefix: pre, body: cached, suffix: suf };
        }
        const pre = (currencySym?currencySym+(space):'') + (cfg.prefix||'');
        const suf = (cfg.suffix||'') + (suffixParts.length?suffixParts.join(''):'');
        return pre + cached + suf;
    }

    // 14) Body formatting (pattern/asFraction/standard)
    let bodyStr = '';
    if(compiledPattern){
        const absRounded = Math.abs(rounded);
        let core = _formatWithPattern(absRounded, '', compiledPattern, thousandsSep, decimalSep, cfg.groupSizes);
        if(isNeg && cfg.parenthesesOnNegative){ bodyStr = '(' + core + ')'; }
        else if(isNeg){ bodyStr = '-' + core; }
        else if(showPlus){ bodyStr = '+' + core; }
        else { bodyStr = core; }
    } else if(cfg.asFraction && dp>=0){
        const absRounded = Math.abs(rounded);
        const whole = Math.trunc(absRounded);
        const frac = absRounded - whole;
        const glyph = _bestUnicodeFraction(frac);
        const intStr = cfg.group ? _groupIntegerVar(String(whole), thousandsSep, cfg.groupSizes) : String(whole);
        let core = glyph ? (whole>0 ? intStr + ' ' + glyph : glyph) : intStr;
        if(isNeg && cfg.parenthesesOnNegative){ bodyStr = '(' + core + ')'; }
        else if(isNeg){ bodyStr = '-' + core; }
        else if(showPlus){ bodyStr = '+' + core; }
        else { bodyStr = core; }
    } else {
        // Fast integer path if dpForDisplay === 0
        let core;
        if(dpForDisplay===0){
            const intAbs = Math.abs(rounded)|0;
            const intStr = cfg.group ? _groupIntegerVar(String(intAbs), thousandsSep, cfg.groupSizes) : String(intAbs);
            core = intStr;
        } else {
            let s = Math.abs(rounded).toFixed(dpForDisplay);
            const dot = s.indexOf('.');
            let intPart = dot===-1 ? s : s.slice(0,dot);
            let decPart = dot===-1 ? '' : s.slice(dot+1);
            if(cfg.group) intPart = _groupIntegerVar(intPart, thousandsSep, cfg.groupSizes);
            const trimOpt = (cfg.trimSigFigs===undefined?true:cfg.trimSigFigs);
            if(trimOpt!==false){
                const minKeep = typeof trimOpt==='number' && trimOpt>=0 ? (trimOpt|0) : 0;
                decPart = _trimDecimals(decPart, minKeep);
            }
            core = decPart ? intPart + decimalSep + decPart : intPart;
        }
        if(isNeg && cfg.parenthesesOnNegative){ bodyStr = '(' + core + ')'; }
        else if(isNeg){ bodyStr = '-' + core; }
        else if(showPlus){ bodyStr = '+' + core; }
        else { bodyStr = core; }
    }

    // 15) Compose prefix/suffix
    const prefix = (currencySym?currencySym+(space):'') + (cfg.prefix||'');
    if(cfg.suffix) suffixParts.push(cfg.suffix);
    const finalSuffix = suffixParts.length ? suffixParts.join('') : '';

    // 16) Store in LRU (store only BODY; prefix/suffix vary per call rarely but are cheap)
    _fmtCacheSet(lru, cacheKey, bodyStr);

    if(cfg.returnParts) return { prefix, body: bodyStr, suffix: finalSuffix };
    return prefix + bodyStr + finalSuffix;
}

export function clearFormatCache(cfg){
    if(!_formatCache) return;
    if(cfg){
        const slot=_formatCache.get(cfg);
        if(slot) slot.map.clear();
    } else {
        // Clear all known slots (WeakMap is not iterable)
        for (const slot of _allFormatLRUSlots) slot.map.clear();
    }
}
export function clearAllFormatCaches(){
    for (const slot of _allFormatLRUSlots) slot.map.clear();
    _patternCache.clear();
}

/* ===========================
   coerceToBool
=========================== */


const _boolLRU = BOOL_LRU_SIZE > 0 ? new LRUCache(BOOL_LRU_SIZE) : null;

function _boolCacheGet(k){
    if(!_boolLRU) return undefined;
    return _boolLRU.get(k);
}
function _boolCacheSet(k, v){
    if(!_boolLRU) return;
    _boolLRU.set(k, v);
}

/* Default truth tables (ASCII lowercase) */
const _DEFAULT_TRUTHY = new Set(['1','true','t','yes','y','on','enable','enabled','active','ok','affirmative','aye','yup','sure','✓','✔','✅','👍', 'valid', 'success']);
const _DEFAULT_FALSEY = new Set(['0','false','f','no','n','off','disable','disabled','inactive','error','nay','✗','✘','❌','👎','null','nil','none','na','n/a','', ' ', 'blank', '(blanks)']);

export function coerceToBool(value, opts){
    // opts: {
    //   nullishAsFalse?: boolean (default true),
    //   stringNumericPolicy?: 'auto'|'strict'|'none' (default 'auto'),
    //   truthy?: string[] (lower/upper ignored),
    //   falsey?: string[] (lower/upper ignored)
    // }
    const o = opts || Object.create(null);
    const nullishAsFalse = o.nullishAsFalse !== false; // default true
    const strNumPolicy = o.stringNumericPolicy || 'auto';
    const truthy = o.truthy ? new Set(o.truthy.map(x => _toLowerAsciiStr(String(x).trim()))) : _DEFAULT_TRUTHY;
    const falsey = o.falsey ? new Set(o.falsey.map(x => _toLowerAsciiStr(String(x).trim()))) : _DEFAULT_FALSEY;

    const t = typeof value;
    if (t === 'boolean') return value;
    if (value == null) return nullishAsFalse ? false : false; // always boolean return

    if (t === 'number') {
        if (value !== value) return false; // NaN -> false
        return value !== 0;
    }
    if (t === 'bigint') return value !== 0n;

    // Strings & others -> string
    let s = (t === 'string') ? value : (value.valueOf !== Object.prototype.valueOf ? String(value.valueOf()) : String(value));
    if (s.length === 0) return false;

    // Trim and lowercase once; cache small strings
    s = s.trim();
    if (s.length <= MAX_STR_KEY_LEN) {
        const hit = _boolCacheGet(s);
        if (hit !== undefined) return hit;
    }
    const sl = _toLowerAsciiStr(s);

    // Direct table hits
    if (truthy.has(sl)) { if (s.length <= MAX_STR_KEY_LEN) _boolCacheSet(s, true); return true; }
    if (falsey.has(sl)) { if (s.length <= MAX_STR_KEY_LEN) _boolCacheSet(s, false); return false; }

    // Numeric-looking?
    if (strNumPolicy !== 'none') {
        let i = 0, n = sl.length, hasDigit = false, hasDot = false, hasOther = false, neg = false;
        for (; i < n; i++) {
            const c = sl.charCodeAt(i);
            if (c === 45 /*-*/) { if (hasDigit || hasDot) { hasOther = true; break; } neg = true; continue; }
            if (c >= 48 && c <= 57) { hasDigit = true; continue; }
            if (c === 46 /*.*/) { if (hasDot) { hasOther = true; break; } hasDot = true; continue; }
            if (c === 32 /*space*/ || c === 9 /*tab*/ || c === 44 /*,*/ || c === 95 /*_*/) continue;
            hasOther = true; break;
        }
        if (hasDigit && !hasOther) {
            if (strNumPolicy === 'strict') {
                const v = Number(sl);
                const res = (v === 1);
                if (s.length <= MAX_STR_KEY_LEN) _boolCacheSet(s, res);
                return res;
            } else {
                const v = Number(sl);
                const res = (v !== 0 && v === v); // NaN -> false
                if (s.length <= MAX_STR_KEY_LEN) _boolCacheSet(s, res);
                return res;
            }
        }
    }

    // Fallback: non-empty string -> true
    if (s.length <= MAX_STR_KEY_LEN) _boolCacheSet(s, true);
    return true;
}

/* ===========================
   coerceToDate
=========================== */

const _dateLRU = DATE_LRU_SIZE > 0 ? new LRUCache(DATE_LRU_SIZE) : null;
function _dateCacheGet(k){ if(!_dateLRU) return undefined; return _dateLRU.get(k)}
function _dateCacheSet(k,v){ if(!_dateLRU) return; _dateLRU.set(k,v); }

/* Month/weekday tables */
const _MON_SHORT = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
const _MON_LONG  = ['january','february','march','april','may','june','july','august','september','october','november','december'];

/* Heuristics for epoch units */
function _epochAutoToMs(n){
    const an = Math.abs(n);
    // >= 1e12 -> ms (year >= 2001), >= 1e10 -> seconds (year up to ~2286), else seconds
    return an >= 1e12 ? n : (n * 1000);
}

/* Parse ISO-like fast: YYYY-MM-DD[ T]HH:mm[:ss][.SSS][Z|±HH[:MM]] */
function _parseIsoFast(s){
    // Require at least YYYY-MM
    if (s.length < 7) return NaN;
    const y = (s.charCodeAt(0)-48)*1000 + (s.charCodeAt(1)-48)*100 + (s.charCodeAt(2)-48)*10 + (s.charCodeAt(3)-48);
    if (y < 0 || y !== y) return NaN;
    if (s.charAt(4) !== '-' && s.charAt(4) !== '/') return NaN;
    const m = (s.charCodeAt(5)-48)*10 + (s.charCodeAt(6)-48);
    if (m < 1 || m > 12) return NaN;

    let d = 1, H = 0, M = 0, S = 0, ms = 0, i = 7;
    if (s.charAt(i) === '-' || s.charAt(i) === '/') {
        d = (s.charCodeAt(i+1)-48)*10 + (s.charCodeAt(i+2)-48);
        i += 3;
    }
    // Time?
    if (i < s.length && (s.charAt(i) === 'T' || s.charAt(i) === ' ')) {
        i++;
        // HH
        H = (s.charCodeAt(i)-48)*10 + (s.charCodeAt(i+1)-48); i += 2;
        if (s.charAt(i) === ':') {
            M = (s.charCodeAt(i+1)-48)*10 + (s.charCodeAt(i+2)-48); i += 3;
            if (s.charAt(i) === ':') {
                S = (s.charCodeAt(i+1)-48)*10 + (s.charCodeAt(i+2)-48); i += 3;
            }
        }
        // .ms
        if (s.charAt(i) === '.') {
            i++;
            let start = i, end = i;
            while (end < s.length && s.charCodeAt(end) >= 48 && s.charCodeAt(end) <= 57) end++;
            const frac = s.slice(start, end);
            const l = frac.length;
            if (l >= 1) {
                ms = l >= 3 ? ( (frac.charCodeAt(0)-48)*100 + (frac.charCodeAt(1)-48)*10 + (frac.charCodeAt(2)-48) )
                    : (l===2 ? (frac.charCodeAt(0)-48)*100 + (frac.charCodeAt(1)-48)*10
                        : (frac.charCodeAt(0)-48)*100 );
            }
            i = end;
        }
    }

    // TZ
    let tzSign = 0, tzH = 0, tzM = 0;
    const c = s.charAt(i);
    if (c === 'Z' || c === 'z') {
        tzSign = 0;
    } else if (c === '+' || c === '-') {
        tzSign = (c === '+') ? 1 : -1;
        tzH = (s.charCodeAt(i+1)-48)*10 + (s.charCodeAt(i+2)-48);
        i += 3;
        if (s.charAt(i) === ':' ) {
            tzM = (s.charCodeAt(i+1)-48)*10 + (s.charCodeAt(i+2)-48);
        } else {
            // "+hhmm"
            if (s.length >= i+2) tzM = (s.charCodeAt(i)-48)*10 + (s.charCodeAt(i+1)-48);
        }
    }

    // Build UTC ms
    const utc = Date.UTC(y, m-1, d, H, M, S, ms);
    if (utc !== utc) return NaN;
    const offset = tzSign !== 0 ? tzSign * (tzH*60 + tzM) * 60000 : 0;
    return utc - offset;
}

/* Parse "MM/DD/YYYY" or "DD/MM/YYYY" or "YYYY-MM-DD" and variants with time + optional AM/PM */
function _parseCommonDateTime(s, dayFirst){
    let str = _toLowerAsciiStr(s.trim());
    if (str.length < 6) return NaN;

    // Extract AM/PM
    let isPM = false, hasAmPm = false;
    if (str.endsWith(' am')) { hasAmPm = true; isPM = false; str = str.slice(0, -3); }
    else if (str.endsWith(' pm')) { hasAmPm = true; isPM = true; str = str.slice(0, -3); }

    // Find date/time split
    let datePart = str, timePart = '';
    const tIdx = str.indexOf(' ');
    if (tIdx !== -1) { datePart = str.slice(0, tIdx); timePart = str.slice(tIdx+1); }

    // Date separators
    let y=0,m=0,d=0;
    if (datePart.indexOf('-') !== -1 || datePart.indexOf('/') !== -1 || datePart.indexOf('.') !== -1) {
        const sep = datePart.indexOf('-')!==-1?'-':(datePart.indexOf('/')!==-1?'/':'.');
        const parts = datePart.split(sep);
        if (parts.length === 3) {
            if (parts[0].length === 4) {
                y = +parts[0]; m = +parts[1]; d = +parts[2];
            } else if (parts[2].length === 4) {
                if (dayFirst) { d = +parts[0]; m = +parts[1]; } else { m = +parts[0]; d = +parts[1]; }
                y = +parts[2];
            } else {
                return NaN;
            }
        } else return NaN;
    } else if (datePart.length === 8 && datePart >= '00000101') {
        // YYYYMMDD
        y = +(datePart.slice(0,4)); m = +(datePart.slice(4,6)); d = +(datePart.slice(6,8));
    } else {
        // Month name?
        let mon = -1;
        const lower = datePart;
        for (let i=0;i<12;i++){
            if (lower.indexOf(_MON_SHORT[i]) !== -1 || lower.indexOf(_MON_LONG[i]) !== -1){ mon = i+1; break; }
        }
        if (mon !== -1){
            // Pick numbers from string (day and year)
            let nums = [];
            let cur = '', c=0;
            for (let i=0,n=lower.length;i<n;i++){
                c = lower.charCodeAt(i);
                if (c>=48 && c<=57) { cur += lower[i]; }
                else { if (cur) { nums.push(+cur); cur=''; } }
            }
            if (cur) nums.push(+cur);
            if (nums.length === 2){ d = nums[0]; y = nums[1]; m = mon; }
            else if (nums.length === 3){
                // assume day, year among them
                nums.sort((a,b)=>a-b);
                y = nums[2]; d = nums[0]; m = mon;
            } else return NaN;
        } else return NaN;
    }

    // Time HH[:mm[:ss]][.SSS]
    let H=0, M=0, S=0, ms=0;
    if (timePart){
        const t = timePart;
        let i=0, n=t.length, buf=0, count=0;
        while (i<n && t.charCodeAt(i)>=48 && t.charCodeAt(i)<=57){ buf = buf*10 + (t.charCodeAt(i)-48); i++; }
        H = buf; count++;
        if (i<n && t.charAt(i) === ':'){
            i++; buf=0; while (i<n && t.charCodeAt(i)>=48 && t.charCodeAt(i)<=57){ buf = buf*10 + (t.charCodeAt(i)-48); i++; }
            M = buf; count++;
            if (i<n && t.charAt(i) === ':'){
                i++; buf=0; while (i<n && t.charCodeAt(i)>=48 && t.charCodeAt(i)<=57){ buf = buf*10 + (t.charCodeAt(i)-48); i++; }
                S = buf; count++;
            }
        }
        if (i<n && t.charAt(i) === '.'){
            i++; let start=i; while(i<n && t.charCodeAt(i)>=48 && t.charCodeAt(i)<=57) i++;
            const frac = t.slice(start,i);
            const l = frac.length;
            if (l>=1) ms = l>=3 ? ((frac.charCodeAt(0)-48)*100 + (frac.charCodeAt(1)-48)*10 + (frac.charCodeAt(2)-48))
                : (l===2 ? (frac.charCodeAt(0)-48)*100 + (frac.charCodeAt(1)-48)*10
                    : (frac.charCodeAt(0)-48)*100);
        }
        if (hasAmPm){
            if (H === 12) H = isPM ? 12 : 0;
            else if (isPM) H += 12;
        }
    }

    if (!(y>0 && m>=1 && m<=12 && d>=1 && d<=31)) return NaN;
    return { y, m, d, H, M, S, ms };
}

export function coerceToDate(value, opts){
    // opts: {
    //   nullishAsNow?: boolean,
    //   epochUnit?: 'ms'|'s'|'auto' (default 'auto'),
    //   dayFirst?: boolean (for dd/mm/yyyy),
    //   assumeUTC?: boolean (interpret naive as UTC; default false=local),
    //   twoDigitYearPivot?: number (base 2000 pivot; default 70 -> 00..69 => 2000..2069, 70..99 => 1970..1999),
    //   onInvalid?: 'null'|'now'|'NaN' (default 'null')
    // }
    const o = opts || Object.create(null);
    const nullishAsNow = !!o.nullishAsNow;
    const epochUnit = o.epochUnit || 'auto';
    const dayFirst = !!o.dayFirst;
    const assumeUTC = !!o.assumeUTC;
    const pivot = (o.twoDigitYearPivot|0) || 70;
    const onInvalid = o.onInvalid || 'null';

    if (value == null){
        if (nullishAsNow) return new Date();
        return onInvalid === 'NaN' ? new Date(NaN) : null;
    }
    if (value instanceof Date){
        return isNaN(value.getTime()) ? (onInvalid === 'NaN' ? new Date(NaN) : null) : value;
    }
    if (typeof value === 'number'){
        const n = +value;
        const ms = epochUnit === 'ms' ? n : (epochUnit === 's' ? n*1000 : _epochAutoToMs(n));
        const d = new Date(ms);
        return isNaN(d.getTime()) ? (onInvalid === 'NaN' ? new Date(NaN) : null) : d;
    }
    if (typeof value === 'bigint'){
        const n = Number(value);
        const ms = epochUnit === 'ms' ? n : (epochUnit === 's' ? n*1000 : _epochAutoToMs(n));
        const d = new Date(ms);
        return isNaN(d.getTime()) ? (onInvalid === 'NaN' ? new Date(NaN) : null) : d;
    }

    // Strings / others → string
    let s = (typeof value === 'string') ? value : String(value);
    s = s.trim();
    if (s.length === 0) return onInvalid === 'NaN' ? new Date(NaN) : null;

    if (s.length <= MAX_STR_KEY_LEN){
        const hit = _dateCacheGet(s);
        if (hit !== undefined) return hit;
    }

    // Pure digits => epoch (auto/forced)
    let isDigits = true;
    for (let i=0,n=s.length;i<n;i++){ const c=s.charCodeAt(i); if (c<48 || c>57){ isDigits=false; break; } }
    if (isDigits){
        const n = Number(s);
        const ms = epochUnit === 'ms' ? n : (epochUnit === 's' ? n*1000 : _epochAutoToMs(n));
        const d = new Date(ms);
        const out = isNaN(d.getTime()) ? (onInvalid === 'NaN' ? new Date(NaN) : null) : d;
        if (s.length <= MAX_STR_KEY_LEN) _dateCacheSet(s, out);
        return out;
    }

    // ISO fast
    let msIso = _parseIsoFast(s);
    if (msIso === msIso){
        const d = new Date(msIso);
        if (s.length <= MAX_STR_KEY_LEN) _dateCacheSet(s, d);
        return d;
    }

    // Common date/time
    const parts = _parseCommonDateTime(s, dayFirst);
    if (parts !== parts) {
        // invalid
        return onInvalid === 'NaN' ? new Date(NaN) : null;
    }
    // Handle two-digit year
    let y = parts.y;
    if (y < 100){
        y = y + (y <= pivot ? 2000 : 1900);
    }
    let dOut;
    if (assumeUTC){
        const ms = Date.UTC(y, parts.m-1, parts.d, parts.H, parts.M, parts.S, parts.ms);
        dOut = new Date(ms);
    } else {
        dOut = new Date(y, parts.m-1, parts.d, parts.H, parts.M, parts.S, parts.ms);
    }
    if (s.length <= MAX_STR_KEY_LEN) _dateCacheSet(s, dOut);
    return dOut;
}

/* ===========================
   formatDate
=========================== */

const DATE_FMT_CACHE_DEFAULT = 0; // per-config LRU entries
const _fmtDateCache = typeof WeakMap !== 'undefined' ? new WeakMap() : null; // cfg -> {map,cap}
const _intlCache = DATE_FMT_CACHE_DEFAULT > 0 ? new LRUCache(DATE_FMT_CACHE_DEFAULT) : null; // key -> Intl.DateTimeFormat

const _WD_SHORT = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const _WD_LONG  = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const _MN_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const _MN_LONG  = ['January','February','March','April','May','June','July','August','September','October','November','December'];

/* Per-config LRU for formatted strings */
function _getDateLRU(cfg){
    return null;
    if (!_fmtDateCache) return null;
    const cap = (cfg && typeof cfg.formatCacheSize === 'number') ? (cfg.formatCacheSize | 0) : DATE_FMT_CACHE_DEFAULT;
    if (cap <= 0) return null;
    let slot = _fmtDateCache.get(cfg);
    if (!slot || slot._cap !== cap){ slot = { map:new LRUCache(cap), _cap:cap }; _fmtDateCache.set(cfg, slot); }
    return slot;
}
function _dateFmtCacheGet(slot, key){
    if(!slot) return undefined;
    return slot.map.get(key);
}
function _dateFmtCacheSet(slot, key, value){
    if(!slot) return;
    const m = slot.map;
    m.set(key, value);
}

/* Round epoch to unit */
function _roundEpoch(ms, unit){
    switch(unit){
        case 's': return Math.floor(ms/1000)*1000;
        case 'm': return Math.floor(ms/60000)*60000;
        case 'h': return Math.floor(ms/3600000)*3600000;
        case 'd': return Math.floor(ms/86400000)*86400000;
        default: return ms; // 'ms'
    }
}

/* Compile date pattern into token list; supports literals via single quotes */
function _compileDatePattern(pat){
    // Supported tokens: YYYY YY, M MM MMM MMMM, D DD, H HH, h hh, m mm, s ss, SSS, A a, ddd dddd, Z ZZ, Q
    const parts = [];
    let i = 0, n = pat.length;
    while (i < n){
        const ch = pat.charAt(i);

        // Literals in single quotes
        if (ch === '\''){
            i++;
            let lit = '';
            while (i < n){
                const c = pat.charAt(i);
                if (c === '\''){ i++; if (i<n && pat.charAt(i)==='\''){ lit+='\''; i++; continue; } else break; }
                lit += c; i++;
            }
            if (lit) parts.push({t:'lit', v:lit});
            continue;
        }

        // Multi-char tokens longest-first
        if (ch === 'Y'){
            if (pat.substr(i,4) === 'YYYY'){ parts.push({t:'YYYY'}); i+=4; continue; }
            if (pat.substr(i,2) === 'YY'){ parts.push({t:'YY'}); i+=2; continue; }
        } else if (ch === 'M'){
            if (pat.substr(i,4) === 'MMMM'){ parts.push({t:'MMMM'}); i+=4; continue; }
            if (pat.substr(i,3) === 'MMM'){ parts.push({t:'MMM'}); i+=3; continue; }
            if (pat.substr(i,2) === 'MM'){ parts.push({t:'MM'}); i+=2; continue; }
            parts.push({t:'M'}); i++; continue;
        } else if (ch === 'D'){
            if (pat.substr(i,2) === 'DD'){ parts.push({t:'DD'}); i+=2; continue; }
            parts.push({t:'D'}); i++; continue;
        } else if (ch === 'H'){
            if (pat.substr(i,2) === 'HH'){ parts.push({t:'HH'}); i+=2; continue; }
            parts.push({t:'H'}); i++; continue;
        } else if (ch === 'h'){
            if (pat.substr(i,2) === 'hh'){ parts.push({t:'hh'}); i+=2; continue; }
            parts.push({t:'h'}); i++; continue;
        } else if (ch === 'm'){
            if (pat.substr(i,2) === 'mm'){ parts.push({t:'mm'}); i+=2; continue; }
            parts.push({t:'m'}); i++; continue;
        } else if (ch === 's'){
            if (pat.substr(i,2) === 'ss'){ parts.push({t:'ss'}); i+=2; continue; }
            parts.push({t:'s'}); i++; continue;
        } else if (ch === 'S'){
            if (pat.substr(i,3) === 'SSS'){ parts.push({t:'SSS'}); i+=3; continue; }
        } else if (ch === 'A' || ch === 'a'){
            parts.push({t: ch}); i++; continue;
        } else if (ch === 'd'){
            if (pat.substr(i,4) === 'dddd'){ parts.push({t:'dddd'}); i+=4; continue; }
            if (pat.substr(i,3) === 'ddd'){ parts.push({t:'ddd'}); i+=3; continue; }
        } else if (ch === 'Z'){
            if (pat.substr(i,2) === 'ZZ'){ parts.push({t:'ZZ'}); i+=2; continue; }
            parts.push({t:'Z'}); i++; continue;
        } else if (ch === 'Q'){
            parts.push({t:'Q'}); i++; continue;
        }

        // Fallback literal (single char)
        parts.push({t:'lit', v:ch}); i++;
    }
    return parts;
}

/* Format using compiled parts */
function _formatCompiled(parts, d, useUTC){
    const get = useUTC ? {
        y: d.getUTCFullYear(), m: d.getUTCMonth(), day: d.getUTCDate(), H: d.getUTCHours(), M: d.getUTCMinutes(), S: d.getUTCSeconds(), ms: d.getUTCMilliseconds(), w: d.getUTCDay()
    } : {
        y: d.getFullYear(), m: d.getMonth(), day: d.getDate(), H: d.getHours(), M: d.getMinutes(), S: d.getSeconds(), ms: d.getMilliseconds(), w: d.getDay()
    };
    const out = new Array(parts.length); let oi = 0;
    for (let i=0,n=parts.length;i<n;i++){
        const p = parts[i];
        switch(p.t){
            case 'lit': out[oi++] = p.v; break;
            case 'YYYY': out[oi++] = '' + get.y; break;
            case 'YY': out[oi++] = _pad2(get.y % 100); break;
            case 'M': out[oi++] = '' + (get.m+1); break;
            case 'MM': out[oi++] = _pad2(get.m+1); break;
            case 'MMM': out[oi++] = _MN_SHORT[get.m]; break;
            case 'MMMM': out[oi++] = _MN_LONG[get.m]; break;
            case 'D': out[oi++] = '' + get.day; break;
            case 'DD': out[oi++] = _pad2(get.day); break;
            case 'H': out[oi++] = '' + get.H; break;
            case 'HH': out[oi++] = _pad2(get.H); break;
            case 'h': {
                let h = get.H % 12; if (h === 0) h = 12; out[oi++] = '' + h; break;
            }
            case 'hh': {
                let h = get.H % 12; if (h === 0) h = 12; out[oi++] = _pad2(h); break;
            }
            case 'm': out[oi++] = '' + get.M; break;
            case 'mm': out[oi++] = _pad2(get.M); break;
            case 's': out[oi++] = '' + get.S; break;
            case 'ss': out[oi++] = _pad2(get.S); break;
            case 'SSS': out[oi++] = _pad3(get.ms); break;
            case 'A': out[oi++] = (get.H < 12 ? 'AM' : 'PM'); break;
            case 'a': out[oi++] = (get.H < 12 ? 'am' : 'pm'); break;
            case 'ddd': out[oi++] = _WD_SHORT[get.w]; break;
            case 'dddd': out[oi++] = _WD_LONG[get.w]; break;
            case 'Z': {
                const off = useUTC ? 0 : -d.getTimezoneOffset(); // minutes east of UTC
                const sign = off >= 0 ? '+' : '-';
                const abs = Math.abs(off);
                out[oi++] = sign + _pad2((abs/60)|0) + ':' + _pad2(abs%60);
                break;
            }
            case 'ZZ': {
                const off2 = useUTC ? 0 : -d.getTimezoneOffset();
                const sign2 = off2 >= 0 ? '+' : '-';
                const abs2 = Math.abs(off2);
                out[oi++] = sign2 + _pad2((abs2/60)|0) + _pad2(abs2%60);
                break;
            }
            case 'Q': {
                out[oi++] = '' + (((get.m / 3) | 0) + 1);
                break;
            }
            default: out[oi++] = ''; break;
        }
    }
    return out.join('');
}

/* Optional Intl formatter (cached) */
function _intlFormat(d, cfg){
    const key = (cfg.timezone==='utc'?'u':'l') + '|' + (cfg.intlOptionsKey || '');
    let fmt = _intlCache.get(key);
    if (!fmt){
        // Build options; default to medium date/time if not specified
        const opts = cfg.intlOptions || { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit' };
        fmt = new Intl.DateTimeFormat(undefined, Object.assign({ timeZone: cfg.timezone==='utc'?'UTC':undefined }, opts));
        _intlCache.set(key, fmt);
    }
    return fmt.format(d);
}

export function formatDate(value, cfg){
    // cfg: {
    //   pattern?: string (e.g., "YYYY-MM-DD HH:mm:ss.SSS"),
    //   preset?: 'iso'|'isoDate'|'isoTime' (ignored if pattern given),
    //   timezone?: 'utc'|'local' (default 'local'),
    //   roundTo?: 'ms'|'s'|'m'|'h'|'d' (default 'ms')  // for caching & display rounding
    //   prefix?: string, suffix?: string, returnParts?: boolean,
    //   onInvalid?: string|null (string override or null),
    //   useIntl?: boolean, intlOptions?: Intl.DateTimeFormatOptions, intlOptionsKey?: string,
    //   formatCacheSize?: number
    // }
    cfg = cfg || Object.create(null);
    const tz = cfg.timezone === 'utc' ? 'utc' : 'local';

    // Coerce date
    const d = (value instanceof Date) ? value : coerceToDate(value, { onInvalid: 'NaN' });
    if (!(d instanceof Date) || isNaN(d.getTime())){
        const onInv = cfg.onInvalid;
        if (onInv === null) return null;
        if (typeof onInv === 'string') return onInv;
        return '';
    }

    // Round (for stability + cache hits)
    const roundTo = cfg.roundTo || 'ms';
    const epochRounded = _roundEpoch(d.getTime(), roundTo);
    const dRounded = (epochRounded === d.getTime()) ? d : new Date(epochRounded);

    // Compose cache key
    const pattern = cfg.pattern ? cfg.pattern :
        (cfg.preset === 'iso' ? 'YYYY-MM-DDTHH:mm:ss.SSSZ' :
            cfg.preset === 'isoDate' ? 'YYYY-MM-DD' :
                cfg.preset === 'isoTime' ? 'HH:mm:ss' : 'YYYY-MM-DD HH:mm:ss');

    const lru = _getDateLRU(cfg);
    const cacheKey = (tz === 'utc' ? 'u' : 'l') + '|' + pattern + '|' + roundTo + '|' + epochRounded;
    const cached = _dateFmtCacheGet(lru, cacheKey);
    if (cached !== undefined){
        const pre = cfg.prefix || '';
        const suf = cfg.suffix || '';
        if (cfg.returnParts) return { prefix: pre, body: cached, suffix: suf };
        return pre + cached + suf;
    }

    // Format
    let body;
    if (cfg.useIntl){
        body = _intlFormat(dRounded, { timezone: tz, intlOptions: cfg.intlOptions, intlOptionsKey: cfg.intlOptionsKey });
    } else {
        let compiled = _patternCache.get(pattern);
        if (!compiled){ compiled = _compileDatePattern(pattern); _patternCache.set(pattern, compiled); }
        body = _formatCompiled(compiled, dRounded, tz === 'utc');
    }

    _dateFmtCacheSet(lru, cacheKey, body);
    const pre = cfg.prefix || '';
    const suf = cfg.suffix || '';
    if (cfg.returnParts) return { prefix: pre, body, suffix: suf };
    return pre + body + suf;
}

export function isNumber(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n;
}

export function isInteger(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && Math.floor(n) === n;
}

export function isFiniteNumber(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && n !== Infinity && n !== -Infinity;
}

export function isSafeInteger(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && Number.isSafeInteger(n);
}

export function isPositive(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && n > 0;
}

export function isNegative(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && n < 0;
}

export function isNonNegative(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && n >= 0;
}

export function isNonPositive(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && n <= 0;
}

export function isZero(x, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && n === 0;
}

export function isNearZero(x, eps=1e-9, opts){
    const n = coerceToNumber(x, opts);
    return typeof n === 'number' && n === n && (n <= eps && n >= -eps);
}

export function isApprox(x, y, eps=1e-9, opts){
    const a = coerceToNumber(x, opts);
    if (!(typeof a === 'number' && a === a)) return false;
    const b = coerceToNumber(y, opts);
    if (!(typeof b === 'number' && b === b)) return false;
    const d = a - b;
    return d <= eps && d >= -eps;
}

export function isInRange(x, min, max, opts){
    const n = coerceToNumber(x, opts);
    if (!(typeof n === 'number' && n === n)) return false;
    return n >= min && n <= max;
}

// ---------- Date predicates ----------

export function isDate(x, opts){
    const d = coerceToDate(x, opts);
    return d instanceof Date && !isNaN(d.getTime());
}

// ---------- String / boolean / nullish ----------

export function isString(x){
    return typeof x === 'string';
}

export function isNonEmptyString(x){
    return typeof x === 'string' && x.length > 0;
}

export function isBlank(x){
    if (typeof x !== 'string') return false;
    // Fast blank check: skip regex; accept common whitespace: space(32), tab(9), NBSP(160), CR(13), LF(10)
    for (let i=0, n=x.length; i<n; i++){
        const c = x.charCodeAt(i);
        if (!(c===32 || c===9 || c===160 || c===13 || c===10)) return false;
    }
    return true; // empty string also qualifies as blank
}

export function isBool(x){
    return typeof x === 'boolean';
}

export function isNully(x){
    return x == null; // null or undefined
}

export function isTruthy(x, opts){
    return coerceToBool(x, opts) === true;
}

export function isFalsey(x, opts){
    return coerceToBool(x, opts) === false;
}

// ---------- Structural helpers (no coercion) ----------

export function isArray(x){
    return Array.isArray(x);
}

export function isObject(x){
    return x !== null && typeof x === 'object';
}

export function isPlainObject(x){
    if (x === null || typeof x !== 'object') return false;
    const proto = Object.getPrototypeOf(x);
    return proto === Object.prototype || proto === null;
}

/* ===========================
   String Helpers
=========================== */

function _isAlphaNumCC(c){
    return (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122);
}
function _isUpperCC(c){ return c >= 65 && c <= 90; }
function _isLowerCC(c){ return c >= 97 && c <= 122; }
function _toLowerAsciiChar(c){ return (c >= 65 && c <= 90) ? String.fromCharCode(c + 32) : String.fromCharCode(c); }
function _toUpperAsciiChar(c){ return (c >= 97 && c <= 122) ? String.fromCharCode(c - 32) : String.fromCharCode(c); }
function _isAllUpperAsciiLetters(str){
    let seenLetter = false;
    for (let i=0, n=str.length; i<n; i++){
        const c = str.charCodeAt(i);
        if (c >= 65 && c <= 90){ seenLetter = true; continue; }
        if (c >= 97 && c <= 122) return false; // has lowercase
        // digits allowed, ignore
    }
    return seenLetter;
}
function _lowercaseFirstAsciiPreserveRest(s){
    if (s.length === 0) return s;
    const c0 = s.charCodeAt(0);
    if (c0 >= 65 && c0 <= 90){
        return String.fromCharCode(c0 + 32) + s.slice(1);
    }
    return s;
}
function _capitalizeFirstAscii(s, forceLowerRest){
    if (s.length === 0) return s;
    let out = '';
    // find first alpha to capitalize
    let i=0, n=s.length;
    while (i<n){
        const c = s.charCodeAt(i);
        if (_isLowerCC(c)){
            out += String.fromCharCode(c - 32);
            i++;
            break;
        } else if (_isUpperCC(c)){
            out += s.charAt(i);
            i++;
            break;
        } else {
            // digit or other (should be only alnum here), keep as-is
            out += s.charAt(i);
            i++;
            // if it's a digit, we just keep and continue; no "cap" yet
            if (c >= 48 && c <= 57) break;
        }
    }
    if (!forceLowerRest) return out + s.slice(i);
    // lower rest (ASCII)
    for (; i<n; i++){
        const c = s.charCodeAt(i);
        out += (c >= 65 && c <= 90) ? String.fromCharCode(c + 32) : s.charAt(i);
    }
    return out;
}

export function toCamel(){
    // Signature: toCamel(...args, opts?)
    // If last arg is a config object with any of these known fields, treat it as opts.
    const argc = arguments.length;
    let opts = null, last = argc ? arguments[argc-1] : null;
    if (last && typeof last === 'object' && (last.keepUppers !== undefined || last.force !== undefined)){
        opts = last;
    }
    const keepUppers = !!(opts && opts.keepUppers);
    const force = !!(opts && opts.force);

    // Collect tokens from all args; split on non-alphanumerics
    const tokens = [];
    const upto = opts ? argc-1 : argc;
    for (let ai=0; ai<upto; ai++){
        let s = arguments[ai];
        if (s == null) continue;
        s = (typeof s === 'string') ? s : String(s);
        let start = -1;
        for (let i=0, n=s.length; i<n; i++){
            const c = s.charCodeAt(i);
            if (_isAlphaNumCC(c)){
                if (start === -1) start = i;
            } else {
                if (start !== -1){
                    tokens.push(s.slice(start, i));
                    start = -1;
                }
            }
        }
        if (start !== -1) tokens.push(s.slice(start));
    }
    if (tokens.length === 0) return '';

    // Build camel
    let out = '';
    for (let ti=0; ti<tokens.length; ti++){
        const tok = tokens[ti];
        if (tok.length === 0) continue;
        const isAllUpper = _isAllUpperAsciiLetters(tok)
        if (ti === 0){
            if (keepUppers && isAllUpper){
                // keep as ALLCAPS for leading token (per spec)
                out += tok;
            } else if (force || isAllUpper){
                // lowercase everything, but keep digits
                let seg = '';
                for (let i=0, n=tok.length; i<n; i++){
                    const c = tok.charCodeAt(i);
                    seg += (c >= 65 && c <= 90) ? String.fromCharCode(c + 32) : tok.charAt(i);
                }
                out += seg;
            } else {
                // lower only the first letter, preserve rest (do NOT undo camel)
                out += _lowercaseFirstAsciiPreserveRest(tok);
            }
        } else {
            if (keepUppers && isAllUpper){
                // keep acronym as-is
                out += tok;
            } else {
                out += _capitalizeFirstAscii(tok, force || isAllUpper);
            }
        }
    }
    return out;
}

export function fromCamel(input, opts){
    // opts: { delimiter: '_', toList: false, formatter: (s)=>s.toLowerCase() }
    if (input == null) return opts && opts.toList ? [] : '';
    const s = typeof input === 'string' ? input : String(input);
    const delimiter = (opts && opts.delimiter != null) ? String(opts.delimiter) : '_';
    const toList = !!(opts && opts.toList);
    const formatter = (opts && typeof opts.formatter === 'function') ? opts.formatter : (t => {
        // fast ASCII lower
        let out = '';
        for (let i=0,n=t.length;i<n;i++){
            const c=t.charCodeAt(i);
            out += (c>=65&&c<=90) ? String.fromCharCode(c+32) : t.charAt(i);
        }
        return out;
    });

    if (s.length === 0) return toList ? [] : '';

    const parts = [];
    let start = 0;
    for (let i=1, n=s.length; i<n; i++){
        const prev = s.charCodeAt(i-1);
        const curr = s.charCodeAt(i);
        const next = (i+1<n) ? s.charCodeAt(i+1) : 0;

        const prevIsLower = _isLowerCC(prev);
        const prevIsUpper = _isUpperCC(prev);
        const currIsUpper = _isUpperCC(curr);
        const currIsLower = _isLowerCC(curr);
        const prevIsDigit = (prev>=48 && prev<=57);
        const currIsDigit = (curr>=48 && curr<=57);

        // Split rules:
        // 1) lower/digit -> Upper : split before Upper
        if ((prevIsLower || prevIsDigit) && currIsUpper){
            parts.push(s.slice(start, i));
            start = i;
            continue;
        }
        // 2) Upper followed by Upper then Lower: "HTMLParser" -> split before 'P' (i where next is lower)
        if (prevIsUpper && currIsUpper && (next && _isLowerCC(next))){
            parts.push(s.slice(start, i));
            start = i;
            continue;
        }
        // 3) Letter <-> Digit boundary splits
        if ((!prevIsDigit && currIsDigit) || (prevIsDigit && !currIsDigit)){
            parts.push(s.slice(start, i));
            start = i;
            continue;
        }
    }
    parts.push(s.slice(start));

    // Apply formatter
    for (let i=0; i<parts.length; i++){
        const p = parts[i];
        parts[i] = formatter ? formatter(p) : p;
    }

    return toList ? parts : parts.join(delimiter);
}

export function toTitleFirst(str){
    if (str == null) return '';
    const s = typeof str === 'string' ? str : String(str);
    if (s.length === 0) return s;
    const c0 = s.charCodeAt(0);
    if (c0 >= 97 && c0 <= 122){
        return String.fromCharCode(c0 - 32) + s.slice(1);
    }
    return s;
}

export function toTitleByDelim(str, delim){
    if (str == null) return '';
    const s = typeof str === 'string' ? str : String(str);
    if (!delim || delim === '') return toTitleFirst(s);

    const d = String(delim);
    const dn = d.length;
    if (dn === 1){
        const dc = d.charCodeAt(0);
        let out = '';
        let capNext = true;
        for (let i=0, n=s.length; i<n; i++){
            const c = s.charCodeAt(i);
            if (c === dc){ out += s.charAt(i); capNext = true; continue; }
            if (capNext && c >= 97 && c <= 122){ out += String.fromCharCode(c - 32); capNext = false; continue; }
            capNext = false;
            out += s.charAt(i);
        }
        return out;
    } else {
        // multi-char delimiter
        let out = '';
        let i = 0, n = s.length, capNext = true;
        while (i < n){
            if (i + dn <= n && s.substr(i, dn) === d){
                out += d; i += dn; capNext = true; continue;
            }
            const c = s.charCodeAt(i);
            if (capNext && c >= 97 && c <= 122){ out += String.fromCharCode(c - 32); capNext = false; i++; continue; }
            capNext = false; out += s.charAt(i); i++;
        }
        return out;
    }
}

export function toTitle(str, delim){
    return toTitleByDelim(str, delim);
}

export function truncText(str, n, opts){
    // opts: { mode: 'replaceTail'|'truncate'|'ignore', overflowText: '…' }
    if (str == null) return '';
    const s = typeof str === 'string' ? str : String(str);
    const len = s.length;
    if (len <= n) return s;

    const mode = (opts && opts.mode) || 'replaceTail';
    if (mode === 'ignore') return s;
    if (mode === 'truncate') return s.slice(0, n);

    const overflowText = (opts && opts.overflowText != null) ? String(opts.overflowText) : '…';
    const m = overflowText.length;
    if (n <= m) return overflowText.slice(0, n); // edge: not enough room
    return s.slice(0, n - m) + overflowText;
}

export function padText(str, n, opts){
    // opts: { padChar: ' ', side: 'right'|'left', overflow: 'truncate'|'ignore'|'replaceTail', overflowText?: string }
    if (str == null) str = '';
    const s = typeof str === 'string' ? str : String(str);
    const len = s.length;
    if (len === n) return s;
    if (len > n){
        const mode = (opts && opts.overflow) || 'truncate';
        if (mode === 'ignore') return s;
        if (mode === 'truncate') return s.slice(0, n);
        // replaceTail via truncText
        return truncText(s, n, { mode: 'replaceTail', overflowText: opts && opts.overflowText });
    }
    const padChar = (opts && opts.padChar != null) ? String(opts.padChar) : ' ';
    const side = (opts && opts.side) === 'left' ? 'left' : 'right';
    const need = n - len;

    // Fast repeat for small padChar (use first char if longer)
    const ch = padChar.length === 1 ? padChar : padChar.charAt(0);
    let pad = '';
    // Exponentiation by squaring style build
    let block = ch;
    let k = need;
    while (k > 0){
        if (k & 1) pad += block;
        if (k === 1) break;
        block = block + block;
        k >>= 1;
    }
    return side === 'left' ? (pad + s) : (s + pad);
}

export function contains(haystack, needle, opts){
    if (haystack == null || needle == null) return false;
    const hs = typeof haystack === 'string' ? haystack : String(haystack);
    const nd = typeof needle === 'string' ? needle : String(needle);
    if (nd.length === 0) return true;
    const cs = !(opts && opts.caseSensitive === false);
    if (cs) return hs.indexOf(nd) !== -1;

    // case-insensitive
    // Use built-ins once; engines optimize .toLowerCase heavily
    return hs.toLowerCase().indexOf(nd.toLowerCase()) !== -1;
}

export function replace(str, search, replacement, opts){
    if (str == null) return '';
    const s = typeof str === 'string' ? str : String(str);
    const src = typeof search === 'string' ? search : String(search);
    const rep = typeof replacement === 'string' ? replacement : String(replacement);
    if (src.length === 0) return s; // no-op

    const cs = !(opts && opts.caseSensitive === false);
    if (cs){
        const idx = s.indexOf(src);
        if (idx === -1) return s;
        return s.slice(0, idx) + rep + s.slice(idx + src.length);
    } else {
        const lowerS = s.toLowerCase();
        const lowerSrc = src.toLowerCase();
        const idx = lowerS.indexOf(lowerSrc);
        if (idx === -1) return s;
        return s.slice(0, idx) + rep + s.slice(idx + src.length);
    }
}

export function replaceAll(str, search, replacement, opts){
    if (str == null) return '';
    const s = typeof str === 'string' ? str : String(str);
    const src = typeof search === 'string' ? search : String(search);
    const rep = typeof replacement === 'string' ? replacement : String(replacement);
    if (src.length === 0) return s;

    const cs = !(opts && opts.caseSensitive === false);
    if (cs){
        // Fast path: split/join
        // Avoids regex and scanning char-by-char
        const parts = s.split(src);
        if (parts.length === 1) return s;
        return parts.join(rep);
    } else {
        // Case-insensitive manual scan with single pass
        const lowerS = s.toLowerCase();
        const lowerSrc = src.toLowerCase();
        const out = [];
        let i = 0, n = s.length, m = src.length;
        while (i <= n - m){
            if (lowerS.substr(i, m) === lowerSrc){
                out.push(rep);
                i += m;
            } else {
                out.push(s.charAt(i));
                i++;
            }
        }
        if (i < n) out.push(s.slice(i));
        return out.join('');
    }
}
