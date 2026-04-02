

import {
    eachMinuteOfInterval, eachHourOfInterval, eachDayOfInterval, eachWeekOfInterval,
    eachMonthOfInterval, eachQuarterOfInterval, eachYearOfInterval,
    isWeekend, addMinutes, addHours, addDays, addWeeks, addMonths, addQuarters, addYears,
    subDays, subWeeks, subMonths, subYears,
    startOfMinute, startOfHour, startOfDay, startOfWeek, startOfMonth, startOfQuarter, startOfYear,
    endOfMinute, endOfHour, endOfDay, endOfWeek, endOfMonth, endOfQuarter, endOfYear,
    differenceInMinutes, differenceInHours, differenceInDays, differenceInWeeks,
    differenceInMonths, differenceInQuarters, differenceInYears,
    isSameDay, isSameWeek, isSameMonth, isSameYear,
    parseISO, format, getDate, getDaysInMonth, isLeapYear, getDayOfYear,
    isValid, isBefore, isAfter, isEqual, getYear, getMonth, getQuarter,
    isWithinInterval, differenceInCalendarDays, differenceInBusinessDays,
    setDate, getWeek, getISOWeek
} from 'date-fns';

const _DTF_ET_YMD = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
});

/**
 * Normalize any date-like input to a plain Date object.
 * Accepts: Date, ISO string, "YYYY-MM-DD" string, or number (timestamp).
 */
function toDate(input) {
    if (input instanceof Date) return input;
    if (typeof input === 'number') return new Date(input);
    if (typeof input === 'string') {
        // Try ISO parse first (handles "YYYY-MM-DD" and full ISO strings)
        const parsed = parseISO(input);
        if (isValid(parsed)) return parsed;
        // Fallback
        const fallback = new Date(input);
        if (isValid(fallback)) return fallback;
    }
    throw new TypeError(`Cannot convert to Date: ${input}`);
}

function subtractBusinessDays(date, days) {
    let d = toDate(date);
    let remaining = days;
    while (remaining > 0) {
        d = subDays(d, 1);
        if (!isWeekend(d)) {
            remaining--;
        }
    }
    return d;
}


function getLastBusinessDayOfMonth(date) {
    let d = endOfMonth(toDate(date));
    d = startOfDay(d);
    while (isWeekend(d)) {
        d = subDays(d, 1);
    }
    return d;
}

function isLastBusinessDayOfMonth(date) {
    const d = startOfDay(toDate(date));
    return isSameDay(d, getLastBusinessDayOfMonth(d));
}

export class DateHelpers {

    static millisecondsUntilEndOfDay() {
        const now = new Date();
        return endOfDay(now).getTime() - now.getTime();
    }

    static isBetweenDates(dateToCheck, startDate, endDate) {
        const check = toDate(dateToCheck);
        const start = toDate(startDate);
        const end = toDate(endDate);
        return check >= start && check <= end;
    }

    static getTodaysDate() {
        return format(new Date(), 'yyyy-MM-dd');
    }

    static isLastBusinessDayOfMonth(date) {
        return isLastBusinessDayOfMonth(date);
    }

    static randomDateBetween(startDate, endDate) {
        const start = toDate(startDate).getTime();
        const end = toDate(endDate).getTime();
        return new Date(start + Math.random() * (end - start));
    }

    static dateToMoment(date, _fmt = 'yyyy-MM-dd') {
        if (typeof date === 'string') {
            return toDate(date);
        }
        if (date instanceof Date) return date;
        return toDate(date);
    }

    static shouldAlignToPeriod(rangeLabel) {
        const dayBasedRanges = [
            'Today', 'Yesterday', 'Last 7 Days', 'Last 30 Days', 'Custom Range'
        ];
        return !dayBasedRanges.includes(rangeLabel);
    }

    static determinePeriodType(rangeLabel) {
        if (rangeLabel.includes('Week')) return 'week';
        if (rangeLabel.includes('Month')) return 'month';
        if (rangeLabel.includes('Quarter')) return 'quarter';
        if (rangeLabel.includes('Year')) return 'year';
        if (rangeLabel === 'WTD') return 'week';
        if (rangeLabel === 'MTD') return 'month';
        if (rangeLabel === 'YTD') return 'year';
        return 'day';
    }

    /**
     * @param {Date} startDate
     * @param {Date} endDate
     * @param {string|null} rangeLabel
     * @returns {Date[]} [historicalStart, historicalEnd] or []
     */
    static computeHistoricalPeriod(startDate, endDate, rangeLabel = null) {
        try {
            if (!startDate || !endDate) return [];

            const start = toDate(startDate);
            const end = toDate(endDate);

            // Single-date selection on the last business day of a month:
            // compare vs the last business day of the previous month.
            if (isSameDay(start, end) && isLastBusinessDayOfMonth(start)) {
                const prevMonthLastBizDay = getLastBusinessDayOfMonth(subMonths(start, 1));
                return [prevMonthLastBizDay, prevMonthLastBizDay];
            }

            if (rangeLabel && DateHelpers.shouldAlignToPeriod(rangeLabel)) {
                const period = DateHelpers.determinePeriodType(rangeLabel);
                return DateHelpers.computeAlignedHistoricalPeriod(start, end, period);
            }

            const durationDays = differenceInCalendarDays(end, start) + 1;
            let historicalStart, historicalEnd;

            if (durationDays <= 7) {
                historicalEnd = subtractBusinessDays(start, 1);
                historicalStart = subtractBusinessDays(start, durationDays);
                // Ensure start <= end after business-day math
                if (isAfter(historicalStart, historicalEnd)) {
                    [historicalStart, historicalEnd] = [historicalEnd, historicalStart];
                }
            } else {
                historicalEnd = subDays(start, 1);
                historicalStart = subDays(historicalEnd, durationDays - 1);
            }

            return [historicalStart, historicalEnd];
        } catch (e) {
            console.error("Error while updating historical data", e);
            return [];
        }
    }

    /**
     * Computes a historical period aligned to complete time periods (months, quarters, years).
     * @param {Date} startDate
     * @param {Date} endDate
     * @param {string} periodType
     * @returns {Date[]}
     */
    static computeAlignedHistoricalPeriod(startDate, endDate, periodType) {
        const start = toDate(startDate);
        const end = toDate(endDate);

        if (DateHelpers.isPartialPeriod(start, end, periodType)) {
            return DateHelpers.handlePartialPeriod(start, end, periodType);
        }

        const periodDuration = DateHelpers.getPeriodDuration(start, end, periodType);
        const historicalEnd = subDays(start, 1);
        let historicalStart;

        switch (periodType) {
            case 'day':
                historicalStart = subDays(historicalEnd, periodDuration - 1);
                break;
            case 'week':
                historicalStart = addDays(subWeeks(historicalEnd, periodDuration), 1);
                break;
            case 'month':
                historicalStart = addDays(subMonths(historicalEnd, periodDuration), 1);
                break;
            case 'quarter':
                historicalStart = addDays(subMonths(historicalEnd, periodDuration * 3), 1);
                break;
            case 'year':
                historicalStart = addDays(subYears(historicalEnd, periodDuration), 1);
                break;
            default:
                historicalStart = subDays(historicalEnd, periodDuration - 1);
        }

        return [historicalStart, historicalEnd];
    }

    static isDateAlignedToStartOfPeriod(date, periodType) {
        const d = startOfDay(toDate(date));
        switch (periodType) {
            case 'month':   return isEqual(d, startOfMonth(d));
            case 'quarter': return isEqual(d, startOfQuarter(d));
            case 'year':    return isEqual(d, startOfYear(d));
            default:        return false;
        }
    }

    static isDateAlignedToEndOfPeriod(date, periodType) {
        const d = endOfDay(toDate(date));
        switch (periodType) {
            case 'month':   return isEqual(d, endOfMonth(d));
            case 'quarter': return isEqual(d, endOfQuarter(d));
            case 'year':    return isEqual(d, endOfYear(d));
            default:        return false;
        }
    }

    static isPartialPeriod(startDate, endDate, periodType) {
        const start = toDate(startDate);
        const end = toDate(endDate);

        const startAligned = (fn) => isSameDay(start, fn(start));
        const endAligned = (fn) => isSameDay(end, fn(end));

        switch (periodType) {
            case 'week':
                return startAligned(d => startOfWeek(d, { weekStartsOn: 1 }))
                    && !endAligned(d => endOfWeek(d, { weekStartsOn: 1 }));
            case 'month':
                return startAligned(startOfMonth) && !endAligned(endOfMonth);
            case 'quarter':
                return startAligned(startOfQuarter) && !endAligned(endOfQuarter);
            case 'year':
                return startAligned(startOfYear) && !endAligned(endOfYear);
            default:
                return false;
        }
    }

    /**
     * @param {Date} startDate
     * @param {Date} endDate
     * @param {string} periodType
     * @returns {Date[]}
     */
    static handlePartialPeriod(startDate, endDate, periodType) {
        const start = toDate(startDate);
        const end = toDate(endDate);
        let historicalStart, historicalEnd;

        switch (periodType) {
            case 'week': {
                historicalStart = subWeeks(start, 1);
                historicalEnd = addDays(historicalStart, differenceInCalendarDays(end, start));
                break;
            }
            case 'month': {
                historicalStart = subMonths(start, 1);
                const currentDaySpan = getDate(end) - getDate(start);
                const maxDayInPrevMonth = getDaysInMonth(historicalStart);
                const lastDay = Math.min(getDate(start) + currentDaySpan, maxDayInPrevMonth);
                historicalEnd = setDate(historicalStart, lastDay);
                break;
            }
            case 'quarter': {
                historicalStart = subMonths(start, 3);
                const currentOffset = differenceInCalendarDays(end, start);
                const daysInHistQuarter = DateHelpers.getDaysInQuarter(historicalStart);
                const dayOfHistQuarter = DateHelpers.getDayOfQuarter(historicalStart);
                const cappedOffset = Math.min(currentOffset, daysInHistQuarter - dayOfHistQuarter);
                historicalEnd = addDays(historicalStart, cappedOffset);
                break;
            }
            case 'year': {
                historicalStart = subYears(start, 1);
                // Handle leap year: if end is Feb 29 but prev year isn't leap
                if (getMonth(end) === 1 && getDate(end) === 29 && !isLeapYear(historicalStart)) {
                    historicalEnd = new Date(getYear(historicalStart), 1, 28);
                } else {
                    historicalEnd = subYears(end, 1);
                }
                break;
            }
            default:
                return [];
        }

        return [historicalStart, historicalEnd];
    }

    static getPeriodDuration(startDate, endDate, periodType) {
        const start = toDate(startDate);
        const end = toDate(endDate);

        switch (periodType) {
            case 'day':     return differenceInCalendarDays(end, start) + 1;
            case 'week':    return Math.ceil(differenceInWeeks(end, start, { roundingMethod: 'ceil' }) || 1);
            case 'month':   return Math.max(1, differenceInMonths(end, start) || 1);
            case 'quarter': return Math.max(1, differenceInQuarters(end, start) || 1);
            case 'year':    return Math.max(1, differenceInYears(end, start) || 1);
            default:        return 1;
        }
    }

    static getDayOfQuarter(date) {
        const d = toDate(date);
        return differenceInCalendarDays(d, startOfQuarter(d)) + 1;
    }

    static getDaysInQuarter(date) {
        const d = toDate(date);
        return differenceInCalendarDays(endOfQuarter(d), startOfQuarter(d)) + 1;
    }

    static getDayOfYear(date) {
        return getDayOfYear(toDate(date));
    }

    static getLastDayOfYear(date) {
        return isLeapYear(toDate(date)) ? 366 : 365;
    }


    static getDynamicBucket(startDate, endDate) {
        const defaultBucket = { bucket: "yyyy-MM-dd", format: "yyyy-MM-dd" };

        if (!startDate || !endDate) {
            console.warn("DateHelpers.getDynamicBucket: Invalid start or end date. Falling back to daily bucket.");
            return defaultBucket;
        }

        const start = toDate(startDate);
        const end = toDate(endDate);

        if (!isValid(start) || !isValid(end) || isAfter(start, end)) {
            console.warn("DateHelpers.getDynamicBucket: Invalid start or end date. Falling back to daily bucket.");
            return defaultBucket;
        }

        try {
            const durationDays = differenceInCalendarDays(end, start);
            const now = new Date();
            const sameYear = isSameYear(start, end);
            const currentYear = isSameYear(start, now);

            // 1. Intra-day
            if (durationDays === 0) {
                const durationMins = differenceInMinutes(end, start);
                if (durationMins <= 12 * 60) {
                    return { bucket: "yyyy-MM-dd HH:mm", format: 'h:mmaaa' };
                }
                return { bucket: "yyyy-MM-dd HH:00", format: 'h:mmaaa' };
            }

            // 2. Up to 3 days
            if (durationDays <= 3) {
                return { bucket: "yyyy-MM-dd HH:00", format: "EEE h:mmaaa" };
            }

            // 3. Up to 1 week
            if (durationDays <= 7) {
                return { bucket: "yyyy-MM-dd", format: sameYear ? 'EEE, MMM d' : 'EEE, MMM d yyyy' };
            }

            // 4. Up to 1 month
            if (durationDays <= 31) {
                return { bucket: "yyyy-MM-dd", format: sameYear ? 'MMM d' : 'MMM d yyyy' };
            }

            // 5. Up to 3 months
            if (durationDays <= 92) {
                return { bucket: "yyyy-'W'II", format: currentYear ? "'Week' II" : "yyyy-'W'II" };
            }

            // 6. Up to 1 year
            if (durationDays <= 366) {
                return { bucket: 'yyyy-MM', format: 'MMM yyyy' };
            }

            // 7. More than 1 year
            return { bucket: "yyyy", format: 'yyyy' };

        } catch (e) {
            console.error("Error in dynamic bucket resolution", e);
            return defaultBucket;
        }
    }

    static summarizeDateRange(a, b) {
        const msInDay = 86_400_000;
        const days = Math.round(Math.abs(toDate(b) - toDate(a)) / msInDay) + 1;

        if (days <= 6)   return days + 'D';
        if (days === 7)  return '1W';
        if (days <= 13)  return days + 'D';
        if (days < 28)   return Math.round(days / 7) + 'W';
        if (days < 365)  return Math.round(days / 30) + 'M';
        return Math.round(days / 365) + 'Y';
    }

    static getDateForEt(input = new Date()) {
        const d = input instanceof Date
            ? input
            : (typeof input === "number" && Number.isFinite(input))
                ? new Date(input)
                : new Date(String(input));

        if (Number.isNaN(d.getTime())) throw new TypeError("getDateForEt: invalid date input");

        const parts = _DTF_ET_YMD.formatToParts(d);
        let y, m, day;
        for (const part of parts) {
            if (part.type === "year") y = part.value;
            else if (part.type === "month") m = part.value;
            else if (part.type === "day") day = part.value;
        }
        return new Date(`${y}-${m}-${day}T00:00:00Z`);
    }
}


export class SmartTimeBucketer {
    constructor({
        marketOpen = '09:30',
        marketClose = '16:00',
        skipWeekends = true,
        useUTC = false,
        maxPoints = 5000
    } = {}) {
        this.marketOpen = marketOpen;
        this.marketClose = marketClose;
        this.skipWeekends = skipWeekends;
        this.useUTC = useUTC;
        this.maxPoints = maxPoints;
        this.originalData = null;
        this.filteredData = null;
        this.zoomLevel = 4;
        this.minZoomLevel = 0;
        this.lockZoom = false;
        this.prepReset = false;
        this._lastBucketKeysString = null;
        this.lastBucketMapSize = null;

        this.bucketLevels = [
            { label: '1m', ms: 60_000,         type: 'minute',  gaps: true },
            { label: '1m', ms: 60_000,         type: 'minute',  gaps: false },
            { label: '1h', ms: 3_600_000,      type: 'hour',    gaps: true },
            { label: '1h', ms: 3_600_000,      type: 'hour',    gaps: false },
            { label: '6h', ms: 3_600_000 * 6,  type: 'hour',    gaps: false },
            { label: '1d', ms: 86_400_000,     type: 'day',     gaps: false },
            { label: '1w', ms: 604_800_000,    type: 'week',    gaps: false },
            { label: '1M', ms: 2_592_000_000,  type: 'month',   gaps: false },
            { label: '1Q', ms: 7_776_000_000,  type: 'quarter', gaps: false },
            { label: '1Y', ms: 31_536_000_000, type: 'year',    gaps: false }
        ];
        this.maxZoomLevel = this.bucketLevels.length - 1;
    }

    setData(data, col = "datetime") {
        this.originalData = data.map(d => ({
            t: new Date(d[col]),
            v: d
        }));
        this.filteredData = [...this.originalData];
        this.lockZoom = false;
        this._lastBucketKeysString = null;
    }

    zoomIn() {
        const proposedLevel = this.zoomLevel - 1;
        if (this._canZoom(proposedLevel)) {
            this.zoomLevel = proposedLevel;
            this.lockZoom = true;
            return true;
        }
        return false;
    }

    zoomOut() {
        const proposedLevel = this.zoomLevel + 1;
        if (this._canZoom(proposedLevel)) {
            this.zoomLevel = proposedLevel;
            this.lockZoom = true;
            return true;
        }
        return false;
    }

    setViewRange(startTs, endTs) {
        const start = new Date(startTs);
        const end = new Date(endTs);
        this.filteredData = this.originalData
            ? this.originalData.filter(d => d.t >= start && d.t <= end)
            : [];
        this.lockZoom = false;
        this._lastBucketKeysString = null;
    }

    _canZoom(newLevel, maxPoints = null) {
        maxPoints = maxPoints ?? this.maxPoints;
        if (newLevel < this.minZoomLevel || newLevel > this.maxZoomLevel || !this.filteredData?.length) {
            return false;
        }

        const bucketDef = this.bucketLevels[newLevel];
        if (!bucketDef) return false;

        const bucketMs = bucketDef.ms;
        const dataSpan = this.filteredData[this.filteredData.length - 1].t - this.filteredData[0].t;
        const estimatedPoints = dataSpan > 0 ? Math.ceil(dataSpan / bucketMs) : 1;

        // Don't allow zooming in if it would produce too many points
        return !(estimatedPoints > maxPoints && newLevel < this.zoomLevel);
    }

    getBucketsForWidth(pixelWidth, reset = false) {
        if (!this.filteredData || this.filteredData.length === 0) {
            this._lastBucketKeysString = JSON.stringify([]);
            this.lastBucketMapSize = null;
            return { x: [], yGroups: [] };
        }

        const minPixelsPerPoint = 6;
        const targetPoints = Math.max(10, Math.floor(pixelWidth / minPixelsPerPoint));

        // Sort once
        this.filteredData.sort((a, b) => a.t - b.t);

        const minDate = this.filteredData[0].t;
        const maxDate = this.filteredData[this.filteredData.length - 1].t;

        // 1. Get bucket level
        let bucketLevel;
        if (reset || this.prepReset) {
            this.lastBucketMapSize = null;
            bucketLevel = this._chooseBestBucketType(minDate, maxDate, targetPoints);
            this.zoomLevel = this.bucketLevels.indexOf(bucketLevel);
        } else {
            const idx = Math.min(Math.max(this.zoomLevel, 0), this.bucketLevels.length - 1);
            bucketLevel = this.bucketLevels[idx];
        }
        const { type: granularity, gaps: showGaps } = bucketLevel;
        this.prepReset = false;

        // 2. Generate intervals
        const hasWeekendData = this._dataIncludesWeekend();
        const intervals = this._generateBucketIntervals(
            minDate, maxDate, granularity, this.skipWeekends, hasWeekendData
        );

        // 3. Map data into buckets
        const bucketMap = new Map();
        for (const { t, v } of this.filteredData) {
            try {
                const bucketStart = this._floorToBucketStart(t, granularity);
                const key = bucketStart.getTime(); // Use numeric key for performance
                if (!bucketMap.has(key)) bucketMap.set(key, []);
                bucketMap.get(key).push(v);
            } catch (e) {
                console.error('Bucket mapping error:', e);
            }
        }

        // 4. Build output
        const x = [];
        const yGroups = [];
        for (const dt of intervals) {
            const key = dt.getTime();
            const dataInBucket = bucketMap.get(key) || [];
            if (showGaps || dataInBucket.length > 0) {
                x.push(this._formatDateForXAxis(dt, minDate, maxDate, granularity));
                yGroups.push(dataInBucket);
            }
        }

        // 5. Cache
        this.lastBucketMapSize = bucketMap.size;
        this._lastBucketKeysString = JSON.stringify(intervals.map(dt => dt.getTime()));
        return { x, yGroups };
    }

    _chooseBestBucketType(minDate, maxDate, targetPoints = 100) {
        if (!minDate || !maxDate || this.bucketLevels.length === 0 || minDate > maxDate) {
            return this.bucketLevels.find(b => b.type === 'day') || this.bucketLevels[5];
        }

        const spanMs = maxDate - minDate;
        let best = this.bucketLevels[this.bucketLevels.length - 1];
        let bestDiff = Infinity;

        for (const bucket of this.bucketLevels) {
            if (bucket.gaps) continue; // skip gap-showing levels for auto-selection

            let count;
            switch (bucket.type) {
                case 'minute':
                case 'hour':
                case 'day':
                case 'week':
                    count = Math.max(1, Math.round(spanMs / bucket.ms) + 1);
                    break;
                case 'month':
                    count = Math.max(1,
                        (getYear(maxDate) - getYear(minDate)) * 12
                        + (getMonth(maxDate) - getMonth(minDate)) + 1
                    );
                    break;
                case 'quarter':
                    count = Math.max(1,
                        (getYear(maxDate) - getYear(minDate)) * 4
                        + (getQuarter(maxDate) - getQuarter(minDate)) + 1
                    );
                    break;
                case 'year':
                    count = Math.max(1, getYear(maxDate) - getYear(minDate) + 1);
                    break;
                default:
                    count = Math.max(1, Math.round(spanMs / bucket.ms) + 1);
            }

            if (count > targetPoints * 1.5) continue;

            const diff = Math.abs(count - targetPoints);
            if (diff < bestDiff || (diff === bestDiff && bucket.ms < best.ms)) {
                bestDiff = diff;
                best = bucket;
            }
        }
        return best;
    }

    _dataIncludesWeekend() {
        if (!this.filteredData) return false;
        for (const { t } of this.filteredData) {
            if (isWeekend(t)) return true;
        }
        return false;
    }

    _floorToBucketStart(date, granularity) {
        const d = new Date(date);
        switch (granularity) {
            case 'minute':  return startOfMinute(d);
            case 'hour':    return startOfHour(d);
            case 'day':     return startOfDay(d);
            case 'week':    return startOfWeek(d, { weekStartsOn: 0 });
            case 'month':   return startOfMonth(d);
            case 'quarter': return startOfQuarter(d);
            case 'year':    return startOfYear(d);
            default:        return startOfDay(d);
        }
    }

    _generateBucketIntervals(minDate, maxDate, granularity, skipWeekends, hasWeekendData) {
        const config = {
            minute:  { interval: eachMinuteOfInterval,  add: addMinutes,  start: startOfMinute,  end: endOfMinute },
            hour:    { interval: eachHourOfInterval,    add: addHours,    start: startOfHour,    end: endOfHour },
            day:     { interval: eachDayOfInterval,     add: addDays,     start: startOfDay,     end: endOfDay },
            week:    { interval: eachWeekOfInterval,    add: addWeeks,    start: startOfWeek,    end: endOfWeek },
            month:   { interval: eachMonthOfInterval,   add: addMonths,   start: startOfMonth,   end: endOfMonth },
            quarter: { interval: eachQuarterOfInterval, add: addQuarters, start: startOfQuarter, end: endOfQuarter },
            year:    { interval: eachYearOfInterval,    add: addYears,    start: startOfYear,    end: endOfYear },
        };

        const { interval: intervalFunc, add: addFunc, start: startFunc, end: endFunc } =
        config[granularity] || config.day;

        // For fine granularities, optionally skip weekends
        if (['minute', 'hour', 'day'].includes(granularity) && skipWeekends && !hasWeekendData) {
            const out = [];
            let t = startFunc(minDate);
            const last = endFunc(maxDate);
            while (t <= last) {
                if (!isWeekend(t)) out.push(new Date(t));
                t = addFunc(t, 1);
            }
            return out;
        }

        return intervalFunc({ start: startFunc(minDate), end: endFunc(maxDate) });
    }

    _formatDateForXAxis(date, minDate, maxDate, granularity) {
        const sameYr = isSameYear(minDate, maxDate);
        const sameMo = isSameMonth(minDate, maxDate);
        const sameWk = isSameWeek(minDate, maxDate, { weekStartsOn: 1 });
        const sameDy = isSameDay(minDate, maxDate);

        switch (granularity) {
            case 'minute':
                if (sameDy) return format(date, 'hh:mmaaa');
                if (sameYr && sameMo && sameWk) return format(date, 'EEE, hh:mmaaa');
                if (sameYr) return format(date, "MMM do hh:mmaaa");
                return format(date, 'MM/dd/yy hh:mmaaa');

            case 'hour':
                if (sameDy) return format(date, 'hhaaa');
                if (sameYr && sameMo && sameWk) return format(date, 'EEE, hhaaa');
                if (sameYr) return format(date, "MMM do hhaaa");
                return format(date, 'MM/dd/yy hhaaa');

            case 'day':
                if (sameDy) return format(date, "MMM do");
                if (sameYr && sameMo && sameWk) return format(date, 'EEE');
                if (sameYr) return format(date, "MMM do");
                return format(date, 'MM/dd/yy');

            case 'day-compressed':
                if (sameDy) return format(date, 'hh:mmaaa');
                if (sameYr && sameMo && sameWk) return format(date, 'EEE, hh:mmaaa');
                if (sameYr) return format(date, "MMM do hh:mmaaa");
                return format(date, 'MM/dd/yy hh:mmaaa');

            case 'week': {
                const w = getISOWeek(date);
                return sameYr ? `W${w}` : `W${w}-${format(date, 'yy')}`;
            }

            case 'month':
                return sameYr ? format(date, 'MMM') : format(date, 'MMM yy');

            case 'quarter':
                return sameYr ? format(date, 'QQQ') : format(date, "QQQ yy");

            case 'year':
                return format(date, 'yyyy');

            default:
                return format(date, "MMM do yyyy HH:mm");
        }
    }

    formatLabel(date, bucketType) {
        const d = this.useUTC
            ? {
                Y: date.getUTCFullYear(),
                M: date.getUTCMonth() + 1,
                D: date.getUTCDate(),
                h: date.getUTCHours(),
                m: date.getUTCMinutes()
            }
            : {
                Y: date.getFullYear(),
                M: date.getMonth() + 1,
                D: date.getDate(),
                h: date.getHours(),
                m: date.getMinutes()
            };

        const pad = (n) => String(n).padStart(2, '0');

        if (bucketType === '1M') return `${d.Y}-${pad(d.M)}`;
        if (bucketType === '1w' || bucketType === '1d')
            return `${d.Y}-${pad(d.M)}-${pad(d.D)}`;
        return `${d.Y}-${pad(d.M)}-${pad(d.D)} ${pad(d.h)}:${pad(d.m)}`;
    }
}

