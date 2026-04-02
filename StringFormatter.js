
import {formatNumber, coerceToNumber} from './typeHelpers.js';

import capitalize from 'lodash/capitalize';
import camelCase from 'lodash/camelCase';
import snakeCase from 'lodash/snakeCase';
import startCase from 'lodash/startCase';
import kebabCase from 'lodash/kebabCase';

export class StringFormatter {
    static toCapitalize(text) {
        return capitalize(text);
    }
    static fromCamelCase(text, delim=" ") {
        return text.replace(/([A-Z])/g, `${delim}$1`)
    }
    static toCamelCase(text) {
        return camelCase(text);
    }
    static toSnakeCase(text) {
        return snakeCase(text);
    }
    static fromSnakeCase(text, delim=" ") {
        return text.replace(/(_)/g, `${delim}$1`)
    }
    static toKebabCase(text) {
        return kebabCase(text)
    }
    static fromKebabCase(text, delim=" ") {
        return text.replace(/(-)/g, `${delim}$1`)
    }
    static toStartCase(text) {
        return startCase(text);
    }
    static toCapitalizeEachWord(text) {
        return startCase(text.toLowerCase())
    }
    static toUppercaseFirstWord(text, delim=" ") {
        const p = text.split(delim);
        if (p.length > 0) {
            p[0] = p[0].toUpperCase();
        }
        return p.join(" ")
    }
    static stripNonAlphaNumeric(text) {
        return text.replace(/[^a-zA-Z0-9]/g, '');
    }
    static stripNonAlphaNumericWildcards(text) {
        return text.replace(/[^a-zA-Z0-9?*%#.]/g, '');
    }
    static truncate(text, n, trail="...") {
        return text.length > n ? text.slice(0, n) + trail : text;
    }
    static clean_camel(text) {
        if (text === null || text === undefined) return;
        if (typeof text !== "string") text = String(text);
        if (text === "") return "";

        let result = text.split(/[^a-zA-Z0-9]+/).filter(Boolean);

        let pattern = /[A-Z]+(?=[A-Z][a-z]|\d|\s|$)|[A-Z](?![a-z])|[A-Z][a-z]+|[a-z]+|\d+/g;

        let chunks = [];
        for (let t of result) {
            let matches = t.match(pattern);
            if (!matches) continue;
            for (let item of matches) {
                chunks.push(StringFormatter.camelize(item, true)); // Always Pascal-case sub-words
            }
        }
        let merged = chunks.filter(Boolean).join('');
        return StringFormatter.camelize(merged, false);
    }
    static camelize(text, first) {
        if (text === null || text === undefined) return;
        if (typeof text !== "string") text = String(text);
        if (text === "") return "";
        if (/^[A-Z0-9]+$/.test(text)) text = text.toLowerCase(); // "ALLCAPS" special case

        // Remove all non-alphanumeric, then break on word boundaries or digits
        let words = text
            .replace(/[_\-\s]+/g, ' ')
            .split(' ')
            .map(w => w.trim())
            .filter(Boolean)
            .flatMap(w => w.match(/[A-Z]+(?=[A-Z][a-z]|\d|\s|$)|[A-Z](?![a-z])|[A-Z][a-z]+|[a-z]+|\d+/g) || []);

        if (!words.length) return "";

        words = words.map((w, i) => {
            // First word may be lower-cased (camelCase)
            if (i === 0 && !first) {
                return w.charAt(0).toLowerCase() + w.slice(1);
            }
            return w.charAt(0).toUpperCase() + w.slice(1);
        });
        return words.join('');
    }
}
// Utility: split a string by multiple delimiters (as in split_by_multiple)
function splitByMultiple(text, delimiters) {
    if (typeof text !== "string" || !text) return [];
    let pattern = new RegExp(`[${delimiters.map(d => d.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('')}]`, 'g');
    return text.split(pattern).filter(Boolean);
}



