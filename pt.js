
import "../css/pt.css";

import {OverlayScrollbars} from 'overlayscrollbars';

import { ENumberFlow } from '@/utils/eNumberFlow.js';
import { addMinutes, format, isFuture } from "date-fns";
import { formatInTimeZone, toZonedTime } from 'date-fns-tz';
import interact from "interactjs";
import {writeObjectToClipboard, writeStringToClipboard} from "@/utils/clipboardHelpers.js";
import {generateAllMarketColumns} from '@/pt/js/grids/portfolio/marketColumnFactory.js';

// Managers
import { PageBase } from '@/pageBase/js/pageBaseV2.js';
import { WidgetManager } from '@/pt/js/widgets/widgetManager.js';
import { GridSettings } from '@/pt/js/grids/portfolioSettings.js';

// Widgets
import { OverviewWidget } from "@/pt/js/widgets/overview.js";
import { PivotWidget } from "@/pt/js/widgets/pivotArrowWidget.js";
import { ScriptWidget } from '@/pt/js/widgets/scriptWidget.js';
import { RefreshWidget } from '@/pt/js/widgets/refreshWidget.js';

import { ConfigurableDropdown } from '@/pt/js/grids/radioDropdown.js';

import {roundTo, roundToNumeric} from "@/utils/NumberFormatter.js";
import {debounce, asArray} from '@/utils/helpers.js';
import {coerceToNumber, coerceToBool, formatNumber} from '@/utils/typeHelpers.js';
import {getElementCoordinatesAsPercentage} from '@/utils/AnimationHelper.js';
import confetti from 'canvas-confetti';
import {StringFormatter} from '@/utils/StringFormatter.js';

import {EnhancedPayloadBatcher} from "@/global/enhancedPayloadBatcher.js";
import {ArrowEngine, ArrowAgGridAdapter} from '@/grids/js/arrow/arrowEngine.js';
import {virtualPortfolioColumns, realPortfolioColumns, generalPortfolioColumns} from '@/pt/js/grids/portfolio/portfolioColumns.js';
import {SIG_FIGS_BY_QUOTE_TYPE} from '@/pt/js/grids/portfolioSigFigs.js';
import PillManager from "@/global/js/pillManager.js";
// import { ClientSummaryModal } from '@/grids/js/arrow/clientModal.js';

import {ACTION_MAP} from "@/global/actionMap.js";
import {MetaEditorModal} from "@/grids/js/metaEditorModal.js";
import {MicroGridManager} from "@/global/js/microGridManager.js";
import {MICRO_GRID_GROUPS} from "@/global/js/microGridConfigs.js";

const VENUE_MEDIUM = {
    'BBG': 'BBG',
    'MX': 'MarketAxess',
    'TW': 'Tradeweb',
    'TR': 'Trumid',
    'TRUMID': 'Trumid',
    'MANUAL': 'MANUAL',
    'OTHER': null
}

const MARKET_PRIORITY = {
    "BVAL (Latest)": 0,
    "MACP": 1,
    "AM": 2,
    "TRACE": 3,
    "CBBT": 4,
    "ALLQ": 5,
    "MLCR": 6,
    "MARK": 7,
    "MARKIT (Hourly)": 8,
    "MARKIT (Live)": 9,
    "IDC": 10,
}

function _isNullyString(v) {
    if (v == null) return true
    if (v.trim() === '') return true
    return false
}

function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

const PORTFOLIO_KEY_REGEX = /([a-f0-9]{32})/;
export class PortfolioPage extends PageBase {
    constructor(name = "pt", { url_context = null, context = {}, config = {}, container=null} = {}) {
        super(name, {url_context, context, config, container});
        this.context.portfolio_key = (url_context ?? this.context.url_context)?.toLowerCase()?.match(PORTFOLIO_KEY_REGEX)?.[0];

        if (!this.context.portfolio_key) {
            this.context.portfolio_key = (url_context ?? this.context.url_context)
        }

        if (!this.context.portfolio_key) {
            console.error("Portfolio Key missing from URL context. Cannot load page.");
            window.request_page('homepage', false);
            throw new Error("Portfolio Key is required.");
        }

        this.context.portfolioRoom = `${this.context.portfolio_key.toUpperCase()}.PORTFOLIO`;
        this.context.metaRoom = `${this.context.portfolio_key.toUpperCase()}.META`;

        this.grid_id = 'portfolio';
        this.room = this.context.portfolioRoom;
        this.grid_filters = {key:'portfolioKey', op:'eq', value:this.context.portfolio_key}

        this.selector = '#portfolio-grid'
        this.ptGrid = null;

        this.osInstance = null;
        this.resizeObserver = null;
        this.isMinimized = false;

        this.theme = 'cool';
        this.defaultQuoteType = 'client';
        this.defaultMarket = 'bval';
        this.defaultSide = 'Mid';
        this.defaultWaterfall = true;
        this.defaultWidget = 'overviewWidget';

        this.metaSizeState = 'grossSize';
        this.metaElements = new Map();
        this.metaElemEvents = new Map();
        this._timerMap = new Map();

        // --- Managers & Widgets ---
        this.widgetManager = null;
        this.emit_tokens = new Map();
        this.marketDataMap = new Map();

        // --- Subscription bookkeeping  ---
        this._subs = [];           // unsubscribe functions for store listeners
        this._dueTimeUnsub = null; // tracks the single dueTime subscription

        // -- Auto switch --
        this.yet_to_auto = true;
        this.autoTabController = new AbortController();
        this.buttonController = new AbortController()
        this.bidSkew = null;
        this.askSkew = null;
        this.skew_initialized = false;

        this._toolPanelIdx = 0;
        this._buttonPosition = 0;

        // Micro-grid manager (initialized after page ready)
        this._microGridManager = null;
    }

    async onBeforeInit() {
        this.setupDom();
        this.setupStores();
    }


    setupDom() {
        this.refMktDrop = document.getElementById('portfolio-refmkt-wrapper');
        this.qtDrop = document.getElementById('portfolio-quote-type');
        this.raiseWidgetElement = this._raiseWidgetElement.bind(this)
        this.lowerWidgetElement = this._lowerWidgetElement.bind(this)
    }

    setupEmitter() {
        const adapter = this.ptGrid;
        const engine = this.engine;
        const _em = this.engine.emitter;
        const sm = this.subscriptionManager();
        const metaData = this?._ptRaw?.context;
        const room = metaData.room;

        this._cellEditToken = _em.on('cell-edit', async (payloads) => {

            const rows = payloads.map((payload) => {
                const { rowIndex, changes } = payload;
                if (engine) {
                    const pks = engine.getRowObject(rowIndex, metaData.primary_keys);
                    return { ...pks, ...changes };
                }
                throw Error("Engine not setup.");
            });

            const meta = sm.buildPayloadContext(room, metaData);
            const data = sm.buildPayloadData(room, metaData, rows);
            await sm.publishMessage(room, meta, data);
        });
    }

    setupStores() {

        // Defaults
        this.page$.set('activeQuoteType', this.defaultQuoteType);
        this.page$.set('activeMarket', this.defaultMarket);
        this.page$.set('activeSide', this.defaultSide);
        this.page$.set('waterfallRef', this.defaultWaterfall);
        this.page$.set('quickSearch', '');
        this.page$.set('linkedPivotFilters', false);
        this.page$.set('lockPivotTotals', false);
        this.page$.set('colorizePivot', false);
        this.page$.set('lastAgFilterModel', {});
        this.page$.set('bulkSkewMode', false);

        // Meta
        this._metaRaw = null;
        this._metaHyper = null;
        this._metaStore = this.createStore('portfolioMeta', {});
        this.portfolioMeta$ = this._metaStore;
        this.page$.set('portfolioMeta', this._metaStore);

        // Skew
        this.overallSkew$ = this.createStore('overallSkew', {
            bid: {value: null, unit: null},
            ask: {value: null, unit: null},
        });

        // Market
        this.activeQuoteType$ = this.page$.pick('activeQuoteType');
        this.activeMarket$ = this.page$.pick('activeMarket');
        this.activeSide$ = this.page$.pick('activeSide');
        this.activeRefMarket$ = this.page$.selectKeys(['activeMarket', 'activeSide']);
        this.activeRefSettings$ = this.page$.selectKeys(['activeMarket', 'activeSide', 'activeQuoteType']);
        this.waterfallRef$ = this.page$.pick('waterfallRef');
        this.activeRefSettingsWaterfall$ = this.page$.selectKeys(['activeMarket', 'activeSide', 'activeQuoteType', 'waterfallRef']);

        // Search
        this.quickSearch$ = this.page$.pick('quickSearch');
        this.lastAgFilterModel$ = this.page$.pick('lastAgFilterModel');

        // Pivot
        this.linkedPivotFilters$ = this.page$.pick('linkedPivotFilters');
        this.lockPivotTotals$ = this.page$.pick('lockPivotTotals');
        this.colorizePivot$ = this.page$.pick('colorizePivot');

        // Username
        const up = this.userManager().userProfile$;
        this._subs.push(up.pick(['username', 'displayName']).onRawChanges((ch)=>{
            this.page$.set('username', up.get('username'));
            this.page$.set('displayName', up.get('displayName'));
        }));


        // Grid Settings
        this.qtSigFigs$ = this.createSharedStore('qtSigFigs', SIG_FIGS_BY_QUOTE_TYPE, {persist: 'local', storageKey: 'qtSigFigs'});
        this._subs.push(this.qtSigFigs$.onRawChanges((ch) => {this.page$.set('sigFigs', this.qtSigFigs$.asObject());}));

        this.settingsStorageKey = `${this.grid_id}.settings`;
        this.gridSettings$ = this.createSharedStore('gridSettings', {
            highlightMyBonds: true,
            showGridStats: true,
            autoNavigateToPivot: true,
            showPtStatus: true,
            sideText: {'bid': 'bid', 'offer': 'offer'},
            sideColor: {'bid': 'red', 'offer': 'green'}
        },{persist: 'local', storageKey: this.settingsStorageKey});
        this._subs.push(this.gridSettings$.onRawChanges((ch) => {
            this.page$.set('gridSettings', this.gridSettings$.asObject());
        }));
        this.sideText$ = this.gridSettings$.pick('sideText');
        this.sideColor$ = this.gridSettings$.pick('sideColor');

        this.setupMetaEvents();
        this.setupUiEvents();
    }

    setupUiEvents() {

        this._subs.push(this.activeQuoteType$.onChanges((ch)=> {
            const newType = ch.current.get('activeQuoteType');
            if (this.quoteTypeDisplay) this.quoteTypeDisplay.textContent = this._quoteTypeDisplay(newType || 'client');
            if (this.quoteTypeDisplayLower) this.quoteTypeDisplayLower.textContent = this._quoteTypeDisplay(newType || 'client');
        }));

        this._subs.push(this.sideColor$.onRawChanges((ch) => {
            const colors = this.sideColor$.get('sideColor');
            if (colors && Object.keys(colors).length === 2) {

                const bidColor = colors?.bid?.toLowerCase();
                const askColor = colors?.offer?.toLowerCase();
                if (!bidColor && !askColor) return;

                if (bidColor === 'green') {
                    document.documentElement.style.setProperty('--bid-background-color', `var(--green-side-cell)`);
                    document.documentElement.style.setProperty('--bid-border-color', `var(--green-border-color)`);
                    document.documentElement.style.setProperty('--ask-background-color', `var(--red-side-cell)`);
                    document.documentElement.style.setProperty('--ask-border-color', `var(--red-border-color)`);
                    document.documentElement.style.setProperty('--side-text-color', `white`);
                    document.documentElement.style.setProperty('--side-text-transform', `uppercase`);
                    document.documentElement.style.setProperty('--side-font-weight', `bold`);
                } else if (bidColor === 'red') {
                    document.documentElement.style.setProperty('--bid-background-color', `var(--red-side-cell)`);
                    document.documentElement.style.setProperty('--ask-background-color', `var(--green-side-cell)`);
                    document.documentElement.style.setProperty('--bid-border-color', `var(--red-border-color)`);
                    document.documentElement.style.setProperty('--ask-border-color', `var(--green-border-color)`);
                    document.documentElement.style.setProperty('--side-text-color', `white`);
                    document.documentElement.style.setProperty('--side-text-transform', `uppercase`);
                    document.documentElement.style.setProperty('--side-font-weight', `bold`);
                } else {
                    document.documentElement.style.setProperty('--bid-background-color', `transparent`);
                    document.documentElement.style.setProperty('--ask-background-color', `transparent`);
                    document.documentElement.style.setProperty('--bid-border-color', `transparent`);
                    document.documentElement.style.setProperty('--ask-border-color', `transparent`);
                    document.documentElement.style.setProperty('--side-text-color', `black`);
                    document.documentElement.style.setProperty('--side-text-transform', `none`);
                    document.documentElement.style.setProperty('--side-font-weight', `normal`);
                }
            }
        }));

        this._subs.push(this.gridSettings$.pick('showGridStats').onChanges((ch) => {
            if (this?.ptGrid?.api?.toggleAggStatusBar) {
                this.ptGrid.api.toggleAggStatusBar(!this.gridSettings$.get('showGridStats'));
            }
        }));

        this._subs.push(this.gridSettings$.pick('showPtStatus').onRawChanges((ch) => {
            this._updateStatusDisplay();
        }));

        // --- Impersonate Mode ---
        if (this.impersonateButton) {
            this.userManager().setImpersonate(false);
            //this.impersonateToggle.checked = !!this.gridSettings$.get('impersonateMode');
            this.addEventListener(this.impersonateButton, 'click', (e) => {
                const on = this.userManager().toggleImpersonate();
                this._applyImpersonateVisuals(on);
            });
        }
        const dev_pull_url = 'http://nykpwm20879513:8889/api/pt/portfolio/pull-to-dev?portfolioKey='
        const btn = document.querySelector('#frame-pull button')
        this.addEventListener(btn, 'click', async (event) => {
            await fetch(dev_pull_url + this.context.portfolio_key.toLowerCase());
            window.location.reload();
        })
    }

    _updateStatusDisplay({force=false}={}) {
        const show = force || this.gridSettings$.get('showPtStatus');
        this.headerStatusElem.style.display = show ? 'flex' : 'none';
    }

    _applyImpersonateVisuals(on) {
        const BORDER_ID = 'impersonate-border-overlay';
        let overlay = document.getElementById(BORDER_ID);

        if (on) {
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = BORDER_ID;
                overlay.className = 'impersonate-overlay-border'
                document.body.appendChild(overlay);
            }
            overlay.style.opacity = '1';
            if (this.impersonateButton) {
                this.impersonateButton.classList.toggle('impersonate-on', true);
                this.impersonateButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a9 9 0 0 0-9 9v11l3-3l3 3l3-3l3 3l3-3l3 3V11a9 9 0 0 0-9-9M9 8a2 2 0 0 1 2 2a2 2 0 0 1-2 2a2 2 0 0 1-2-2a2 2 0 0 1 2-2m6 0a2 2 0 0 1 2 2a2 2 0 0 1-2 2a2 2 0 0 1-2-2a2 2 0 0 1 2-2"/></svg>`;
            }
        } else {
            if (overlay) {
                overlay.style.opacity = '0';
                setTimeout(() => overlay?.remove(), 200);
            }
            if (this.impersonateButton) {
                this.impersonateButton.classList.toggle('impersonate-on', false);
                this.impersonateButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><path fill="currentColor" d="M8.29 5.09L6.84 3.64A8.93 8.93 0 0 1 12 2a9 9 0 0 1 9 9v6.8l-2-2V11c0-3.86-3.14-7-7-7c-1.37 0-2.64.4-3.71 1.09m13.82 16.37l-1.27 1.27l-3.28-3.28L15 22l-3-3l-3 3l-3-3l-3 3V11c0-1.74.5-3.37 1.36-4.75L1.11 3l1.28-1.27l4.5 4.5l1.8 1.8l2.28 2.28l6.44 6.45h.01L21 20.34v.01zm-5.97-3.43l-6.25-6.25c-.27.14-.57.22-.89.22a2 2 0 0 1-2-2c0-.32.08-.62.22-.89l-1.4-1.4C5.3 8.69 5 9.81 5 11v6.17l1-1l1.41 1.42L9 19.17l1.59-1.58L12 16.17l1.41 1.42L15 19.17zM15 8c-1.04 0-1.89.8-2 1.82L15.18 12c1.02-.11 1.82-.96 1.82-2a2 2 0 0 0-2-2"/></svg>`;
            }
        }
    }

    async onCacheDom() {
        this.clearFilterButton = document.querySelector('#pt-clearFilters');
        this.clearFilterIcon = document.querySelector('#filterStateWrapper');
        this.headerNameElem = document.querySelector("#pt-header-client-name");
        this.headerSizeElem = document.querySelector("#pt-header-size");
        this.headerNetElem = document.querySelector("#pt-header-net");
        this.headerCountElem = document.querySelector("#pt-header-count");
        this.headerDirectionElem = document.querySelector("#pt-header-direction");
        this.quoteTypeButton = document.querySelector("#portfolio-quote-type");
        this.quoteTypeButtonLower = document.querySelector("#portfolio-quote-type-lower");
        this.headerSubItemsElem = document.querySelector('.pt-header-sub-items');
        this.shadowContainer = document.querySelector('#portfolioBody');
        this.headerStatusElem = document.querySelector("#pt-header-status");
        this.stateDropdown = document.querySelector('.pt-status-dropdown');
        this.stateDisplay = document.querySelector("#pt-header-status");
        this.stateEditorContainer = document.querySelector("#pt-header-status-wrapper");
        this.progressBar = document.getElementById('pricing-progress-bar');
        this.progressBarPctDom = document.getElementById('pricing-progress-bar-pct');
        this.progressBarCountDom = document.getElementById('pricing-progress-bar-count');
        this.frameSkew = document.getElementById('frame-skew');
        this.scroller = document.querySelector('#auto-scroll-down');
        this.gridTools = document.querySelector('#portfolio-grid-tools');
        this.pricingGrid = document.getElementById('portfolio-pricing-grid');
        this.minimizeButton = document.querySelector('#toolbar-minimize');
        this.toolbarButtons = document.querySelectorAll('.toolbar-button.toolbar-widget:not(#toolbar-minimize)');
        this.pushBtn = document.getElementById('push-to-kdb-sidebar-btn');
        this.reloadBtn = document.getElementById('reload-sidebar-btn');
        this.metaEdit = document.getElementById('meta-sidebar-btn');
        this.settingsButton = document.getElementById("settingsBtn");
        this.pricingContainer = document.querySelector(".pricing-progress-container");
        this.toolbarTop = document.querySelector('.toolbar-top');
        this.push_pull_buttons = document.querySelector('#frame-test-button-wrapper');
        this.scrollContainer = document.querySelector('#portfolio-content-shadow');
        this.portfolioBody = document.getElementById("portfolioBody");
        this.quoteTypeDisplay = document.getElementById('portfolio-current-quote-type');
        this.quoteTypeDisplayLower = document.getElementById('portfolio-current-quote-type-lower');
        this.contentArea = document.querySelector('.content-area');
        this.refMktBtn = document.querySelector('#refmkt-btn');

        this.headerRow = document.querySelector('.header-row');
        this.contentRow = document.querySelector('#content-row');
        this.toolbar = document.querySelector('#toolbar');
        this.topTwo = document.getElementById('portfolio-top-two-rows');

        this.gridSearchBar = document.getElementById('grid-search');
        this.searchElement = document.getElementById('pt-quick-filter');
        this.searchInput = document.getElementById("pt-quick-filter-text");
        this.searchInput?.setAttribute("autocomplete", "off");
        this.searchInput?.setAttribute("autocorrect", "off");
        this.searchInput?.setAttribute("autoCapitalize", "off");
        this.searchInput?.setAttribute("spellCheck", "false");
        this.searchIcon = document.querySelector('.ag-search-bar-icon');
        this.searchInput.placeholder = '';

        this.impersonateButton = document.getElementById('pt-impersonate');

        this.copyBtnFull = document.getElementById('copy-summary-btn');
        this.copyBtnShort = document.getElementById('copy-summary-btn-short');
        this.copyBtnLink = document.getElementById('copy-link-btn');
        this.copyBtnKey = document.getElementById('copy-key-btn');

        this.weightGroup = document.getElementById('pivot-weight-group');
        this.sideGroup = document.getElementById('pivot-sides-group');
        this.weightRadios = this.weightGroup ? this.weightGroup.querySelectorAll('input[name="pivot-weight"]') : [];
        this.sideRadios = this.sideGroup ? this.sideGroup.querySelectorAll('input') : [];

        this._initProgressBarFlows();
    }

    _initProgressBarFlows() {
        if (this.progressBarPct) { try { this.progressBarPct.destroy(); } catch(_){} }
        if (this.progressBarCount) { try { this.progressBarCount.destroy(); } catch(_){} }
        this.progressBarPct = new ENumberFlow(this.progressBarPctDom, {
            displayMode:'percentage',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        });
        this.progressBarCount = new ENumberFlow(this.progressBarCountDom, {
            displayMode:'default',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        });
    }

    onBeforeBind() {
        this.theme$ = this.themeManager().theme$;
        this.theme = this.theme$.get('theme');
    }

    onBind() {
        this._setupScrollShadow();
        this._setupDropdown();
        this._setupResizer();
        this._setupListeners();
        // this._setupAutoSwitch();
        this._trackDom();
    }

    _setupCopyButtons() {
        // Copy Summary button
        const ov = this.getWidget('overviewWidget');

        const copyButtons = {
            'full': [this.copyBtnFull, ov.generateSummaryString.bind(ov)],
            'short': [this.copyBtnShort, ov.getShortString.bind(ov)],
            'link': [this.copyBtnLink, ov.getLinkString.bind(ov)],
            'key': [this.copyBtnKey, ov.getKeyString.bind(ov)]
        }

        const page = this.context.page;
        const timers = this._timerMap;
        Object.entries(copyButtons).forEach(([key, btn]) => {
            const btnDom = btn[0];
            const txtFn = btn[1];

            page.addEventListener(btnDom, 'click', async () => {
                if (timers.has(key)) return;

                const originalContent = btnDom.innerHTML;
                const summary = txtFn();

                try {
                    await writeStringToClipboard(summary);
                    btnDom.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="m9.55 18l-5.7-5.7l1.425-1.425L9.55 15.15l9.175-9.175L20.15 7.4L9.55 18Z"/></svg>`;
                    btnDom.classList.add('success');
                } catch (err) {
                    btnDom.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M12 17q.425 0 .713-.288T13 16q0-.425-.288-.713T12 15q-.425 0-.713.288T11 16q0 .425.288.713T12 17Zm0-4q.425 0 .713-.288T13 12V8q0-.425-.288-.713T12 7q-.425 0-.713.288T11 8v4q0 .425.288.713T12 13Z"/></svg>`;
                    btnDom.classList.add('error');
                }

                const t = setTimeout(() => {
                    btnDom.innerHTML = originalContent;
                    btnDom.classList.remove('success', 'error');
                    timers.delete(key);
                }, 1500);

                timers.set(key, t)
            });
        })
    }

    async _setupPills(){
        this.pillManager = new PillManager(this.context, {
            defaultContainer: this.headerSubItemsElem,
            metaStore: this.portfolioMeta$,
            engine: this.engine,
            createInfoModal: this.createInfoModal.bind(this),
            emitter: this.emitter
        });

        this.pillManager.createPill('meta', {
            id: 'venueShort', columns: 'venueShort', label: null, type: 'market',
            valueFormatter: (v) => VENUE_MEDIUM?.[v] ? VENUE_MEDIUM[v] : null,
            tooltip: null,
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        this.pillManager.createPill('meta', {
            id: 'numDealers', columns: 'numDealers', label: null, type: 'market',
            valueFormatter: (v) => v ? `${v} in Comp` : null,
            tooltip: null,
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        this.pillManager.createPill('meta', {
            id: 'isAon', columns: 'isAon', label: null, type: 'market',
            valueFormatter: (v) => v ? 'AON' : 'PARTIALS',
            tooltip: null,
            modal: {
                title: "What does 'All or None' (AON) mean?",
                info: `<p>AON: All lines MUST be quoted for submission</p><p>PARTIALS: Client (may) allow partial submissions</p>`
            }
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        window.wire = this.pillManager.createPill('meta', {
            id: 'wire',
            columns: 'wireTime',
            type: 'timing',
            valueFormatter: (v) => v ? `Wire: ${Math.round((Number(v)||0)/60)}min` : null,
            tooltip: () => this._getMaxTime(),
            tooltipConfig: {
                singleton: true,
            },
            modal: { title: 'Wire Time', info: '<p>The defined time window AFTER the dueTime during which the client can respond to dealer pricing. Dealers may still update their levels during this period. Not a strict window, often acted on well before this time expires.</p>' }
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        this.pillManager.createPill('meta', {
            id: 'isFwdStrike', columns: 'isFwdStrike', label: null, type: 'timing',
            valueFormatter: (v) => v ? 'Fwd Strike' : null,
            tooltip: null,
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        const meta = this._metaStore;
        this.pillManager.createPill('meta', {
            id: 'fwdStrikeTimeMnemonic',
            columns: 'fwdStrikeTimeMnemonic',
            label: null,
            type: 'timing',
            valueFormatter: (v) => {
                if (v == null) return;
                let mkt = meta.get('fwdStrikeMkt') ?? null;

                // Hacky, occasionally dbl LDN tag
                if (mkt && mkt.includes("LONDON")) {
                    mkt = mkt.replace("LONDON", "");
                }

                const side = meta.get('fwdStrikeSide') ?? null;
                if (mkt && side) return `${mkt} ${v} ${side}`;
                if (mkt) return `${mkt} ${v}`;
                if (side) return `${v} ${side}`;
                return v;
            },
            tooltip: null,
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        // this.pillManager.createPill('meta', {
        //     id: 'tier', columns: 'clientTier', label: null, type: 'client',
        //     valueFormatter: (v) => (v != null && v !== 'NT' && v?.toString()?.trim() !== '') ? `Tier ${v}` : null,
        // }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        this.pillManager.createPill('meta', {
            id: 'barcSalesName',
            columns: 'barcSalesName',
            label: null,
            type: 'client',
            valueFormatter: (v) => v ? StringFormatter.toCapitalizeEachWord(v) : null,
            tooltip: null,
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

        this.pillManager.createPill('meta', {
            id: 'clientTraderName',
            columns: 'clientTraderName',
            label: null,
            type: 'client',
            valueFormatter: (v) => v ? StringFormatter.toCapitalizeEachWord(v) : null,
            tooltip: null,
        }).then(async (p) => await p.mount((this.headerSubItemsElem)));

    }

    setupHotkeys() {

        const blocked = ['ArrowUp', 'ArrowDown'];
        const BLOCK_SCROLL_KEYS = new Set(blocked);

        this.addEventListener(document, 'keydown', async (event) => {
            if (BLOCK_SCROLL_KEYS.has(event.key)) {
                event.preventDefault();
                event.stopPropagation();
            }

            // Ctrl+K to focus search input
            if (event.ctrlKey && event.key.toLowerCase() === 'f') {
                event.preventDefault();
                this.searchInput.focus();
            }
            // Ctrl+s to save config
            if (event.ctrlKey && event.key.toLowerCase() === 's') {
                event.preventDefault();
                if (this?.ptGrid?.api?.treeService?.hasUnsavedChanges) {
                    await this?.ptGrid?.api?.treeService?._handleSave();
                }
                if (this.getWidget('pivotWidget')?.api?.treeService?.hasUnsavedChanges) {
                    await this.getWidget('pivotWidget')?.api?.treeService?._handleSave();
                }
            }
            if (event.ctrlKey && event.key.toLowerCase() === 'o') {
                event.preventDefault();
                await this?.ptGrid?.api?.treeService?.showLoadDialog();
            }
            // Ctrl+L to clear search input
            else if (event.ctrlKey && event.key.toLowerCase() === 'l') {
                event.preventDefault();
                if ((this.ptGrid.api.getOpenedToolPanel() === 'treeColumnChooser') &&
                    (this.ptGrid.api.treeService.searchTerm !== '')) {
                    this.ptGrid?.api?.treeService?.clearSearch();
                } else {
                    this.page$.set('quickSearch', null);
                    this.ptGrid.clearFilters();
                    this.clearPivotFilters();
                }
            }

            else if (event.ctrlKey && event.key.toLowerCase() === 'm') {
                event.preventDefault();
                if (this.ptGrid.api.getOpenedToolPanel()) {
                    if (document.activeElement) {
                        document.activeElement.blur();
                    }
                    this.ptGrid.api.closeToolPanel()

                } else {
                    const panels = this.ptGrid.api.getSideBar().toolPanels;
                    this.ptGrid.api.openToolPanel(panels[this._toolPanelIdx].id);
                    this._toolPanelIdx = (this._toolPanelIdx+1) % panels.length;
                }
            }

            else if (event.ctrlKey && event.key === '9') {
                event.preventDefault();
                this._toggleMinimize();
            }

            else if (event.ctrlKey && event.key === '1') {
                event.preventDefault();
                await this.switchTab("overviewWidget");
            }

            else if (event.ctrlKey && event.key === '2') {
                event.preventDefault();
                await this.switchTab("pivotWidget");
            }

                // else if (event.ctrlKey && event.key === '3') {
                //     event.preventDefault();
                //     await this.switchTab("heatmapWidget");
            // }

            else if (event.ctrlKey && event.key === '3') {
                event.preventDefault();
                await this.switchTab("scriptWidget");
            }

            else if (event.ctrlKey && event.key === '4') {
                event.preventDefault();
                await this.switchTab("refreshWidget");
            }

            // else if (event.ctrlKey && event.key === '6') {
            //     event.preventDefault();
            //     await this.switchTab("adminTools");
            // }

        });
    }

    _trackDom() {
        // this.addEventListener(this.contentRow, 'animationend', (e) => {
        //     this.page$.set('domAnimationFinished', true)
        // }, {once: true})
    }

    async onReady() {
        requestAnimationFrame(() => {
            const body = this.portfolioBody;
            if (body) body.style.display = '';
            requestAnimationFrame(() => {
                if (this.pricingContainer) this.pricingContainer.style.opacity = "100";
                if (this.frameSkew) this.frameSkew.style.opacity = "100";
            });
        });

        // setTimeout(() => {
        //     this.contentArea?.classList.add('initialized');
        // }, 1000);
    }

    async onInit() {
        this.context.settingsManager = new GridSettings(this.context);
        await this._setupWebSocketSubscriptions();
        this._microGridManager = new MicroGridManager(this);

        // Subscribe to micro-grids immediately so data is available for
        // derived flag columns — no need to wait for the modal to open.
        this._microGridManager.subscribeGroup(MICRO_GRID_GROUPS.pt_tools).catch(() => {});

        // Auto-resubscribe micro-grids on WebSocket reconnect
        this._microReconnectCb = async () => {
            if (this._microGridManager) {
                this._microGridManager.showReconnectIndicator();
                await this._microGridManager.resubscribeAll();
            }
        };
        this.subscriptionManager().onReconnect(this._microReconnectCb);
    }

    async setupDynamicTooltips() {
        const ov = this.getWidget('overviewWidget');
        const generate_summary_string = ov.generateSummaryString.bind(ov)

        const generate_fake_string = () => {
            return `P\u200b̲P\u200b̲M\u200b̲ \u200b̲A\u200b̲m\u200b̲e\u200b̲r\u200b̲i\u200b̲c\u200b̲a\u200b̲,\u200b̲ \u200b̲I\u200b̲n\u200b̲c\u200b̲.\u200b̲ \u200b̲1\u200b̲0\u200b̲2\u200b̲m\u200b̲m\u200b̲ \u200b̲B\u200b̲W\u200b̲I\u200b̲C\u200b̲ \u200b̲I\u200b̲G
-- TraderName: Mario Scrimenti
--102mm, 69 lines
-- 48k dv01, 38k cs01%, A/A-, 6.7 liq, 9.85 dur, Sprd 64.7, Px 95.14, Yld 5.04
-- Liquidity: Low 30%, Medium 11%, 𝗛𝗶𝗴𝗵 𝟱𝟵%
   - Suggest Extra Liq Charge: 1bp
-- Dur: NA: 2%;  MMY: 4%;  2: 0%;  3: 1%;  4: 9%;  5: 5%; 6: 3%;  𝟴: 𝟭𝟱%;  9: 8%;  10: 1%;  11-15: 7%;  >=15: 6%; 20: 2%;  𝟯𝟬: 𝟯𝟳%
-- BSR: 4%  8% (SU/All), BSIFR 2%, WW 6%, Blocks 0%
-- bvalBO 3.52bp, macpBO 1.78bp, bvalCOD 0.18bp
-- Signals: 0.45bp (𝗮𝗴𝗮𝗶𝗻𝘀𝘁): Stat 0.12bp, HF 0.33bp
   - ALL Stat: -13.4  -21.7  34.2  +17.5  +8.7  (SSU/SU/𝗡/SA/SSA%), NoS 4.5%
   - BSI Stat: -13.2  -21.7  33.8  +17.9  +8.9  , NoS 4.5%
-- Total in comp: Multi Dealer`
        }

        this.tooltipManager().add({
            id: 'summary-string-tooltip',
            title: 'Summary String',
            target: this.copyBtnFull,
            width: 605,
            height: 350,
            flash: { color: 'var(--teal-500)', duration: 3200, intensity: 0.25, spread: 10 },
            padding: 2,
            contextMenu: { enabled: true, items: ['copy', 'pop'] },
            content: async () => {
                const txt = generate_fake_string();
                return `<div style="color: var(--text-color); font-size: 12px;"><pre>${txt}</pre></div>`
            },
            html: true,
            interactive:false,
        });
    }

    _weightStorageKey() {
        const k = (this.context?.portfolio_key || this.context?.url_context || 'default').toUpperCase();
        return `pt:pivotWeight`;
    }

    _readWeightCache() {
        try {
            const raw = localStorage.getItem(this._weightStorageKey());
            const v = raw ? String(raw) : 'notional';
            if (v === 'auto') {
                return 'auto'
            } else if (v === 'dv01') {
                return 'dv01'
            }
            return 'notional';
        } catch { return 'notional'; }
    }

    _writeWeightCache(mode) {
        try { localStorage.setItem(this._weightStorageKey(), String(mode)); } catch {}
    }
    async _setupRefSettings() {
        const pivot = this.getWidget('pivotWidget');
        const initWeight = this._readWeightCache();
        if (this.weightRadios && this.weightRadios.length) {
            Array.from(this.weightRadios).forEach(r => {
                r.checked = (r.value === initWeight);
                this.context.page.addEventListener(r, 'change', async (e) => {
                    if (!e.target.checked) return;
                    const mode = e.target.value;
                    this._writeWeightCache(mode)
                    await pivot.ptPivot.setWavgWeightMode(mode);

                }, {passive:true});
            });
            // initialize engine on first render
            await pivot.ptPivot.setWavgWeightMode(initWeight);
        }

        if (this.sideRadios && this.sideRadios.length) {
            Array.from(this.sideRadios).forEach(r => {
                this.context.page.addEventListener(r, 'change', (e) => {
                    if (!e.target.checked) return;
                    this.refDropdown.setValue('side', e.target.value, true);
                    this.refDropdownLower.setValue('side', e.target.value, true);
                    // this.refDropdownLower.setValue('side', e.target.value, true);
                }, {passive:true});
            });
        }
    }

    async onCleanup() {
        this.resizeObserver?.disconnect();
        this.osInstance?.destroy();

        this.pricingContainer.style.opacity = "0";
        if (this.progressBarPct) {
            try { this.progressBarPct.destroy() } catch(_){}
            try { this.progressBarPct.destroy(); } catch(_){}
        }
        if (this.progressBarCount) {
            try { this.progressBarCount.destroy } catch(_){}
            try { this.progressBarCount.destroy(); } catch(_){}
        }
        if (this.bidSkew) this.bidSkew?.destroy();
        if (this.askSkew) this.askSkew?.destroy();

        this.autoTabController.abort();
        this.buttonController.abort();

        // Cleanup interact.js instance
        this._interactInstance?.unset();

        // Cleanup any open info modals
        document.querySelectorAll('dialog.modal').forEach(m => { try { m.close(); m.remove(); } catch(_){} });

        // Central teardown handles engine, grid, pill, widgets, subs, emitter tokens
        this._teardownGrid();

        // Cleanup micro-grid flag listener
        if (this._microFlagDisposer) {
            try { this._microFlagDisposer(); } catch {}
            this._microFlagDisposer = null;
        }

        // Cleanup micro-grid manager
        if (this._microReconnectCb) {
            try { this.subscriptionManager().offReconnect(this._microReconnectCb); } catch (e) { /* ignore */ }
            this._microReconnectCb = null;
        }
        if (this._microGridManager) {
            this._microGridManager.destroy().catch(() => {});
            this._microGridManager = null;
        }

        requestAnimationFrame(()=>{
            this.frameSkew.innerHTML = '';
            requestAnimationFrame(()=>{
                this.frameSkew.style.opacity = "0";
            })
        })

    }

    _setupResizer() {
        const target = this.contentRow;
        const page = this;
        this._interactInstance = interact(target)
        .resizable({
            edges: {top: false, left: false, bottom: true, right: false},
            listeners: {
                move: function (event) {
                    const target = event.target;
                    target.style.height = Math.max(100, event.rect.height) + 'px';
                }
            }
        }).on('doubletap', (event) => {
            const elem = event.target.closest('div');
            if (
                elem
                && elem?.className?.includes("ag-")
                && !elem?.className?.includes('ag-side-button')
                && !elem?.className?.includes('ag-header')
                && !elem?.className?.includes('editable-cell')
            ) {
                const w = page.getWidget(event.target.closest('.tab-content').id);
                if (w?.alignWidthOnTap) w.alignWidthOnTap();

                if (target.getBoundingClientRect().height > 401) {
                    target.style.height = '400px';
                } else {
                    if (w && w?.getMaxHeight && w.getMaxHeight()) {
                        if (w?.api?.getGridOption('onRowDoubleClicked') && elem.className.includes('ag-cell')) return;
                        target.style.height = `${Math.max(400, w.getMaxHeight())}px`;
                    } else {
                        target.style.height = '1000px';
                    }
                }

            }
        })
    }

    _setupDropdown() {
        if (this.stateDropdown) {
            this.stateDropdown.innerHTML = ''; // Clear existing
            [
                "LIVE", 'WON', 'COVERED', 'MISSED', 'DNT', 'PASSED', 'NETDOWN', 'INDICATIVE', 'ERROR', 'CANCELLED',
                "EXPIRED", 'NEW', 'PENDING', 'TEST', "TRANSFER", 'OTHER', 'UNKNOWN', 'TIED'
            ].forEach(stateValue => {
                this.stateDropdown.appendChild(this._renderStateEditorOption(stateValue));
            });
        }
    }

    _refreshDropdownColors() {
        if (!this.stateDropdown) return;
        const pills = this.stateDropdown.querySelectorAll('.pt-status-option');
        pills.forEach(pill => {
            const state = pill.closest('.pt-status-option-wrapper')?.dataset?.value;
            if (!state) return;
            const color = this.colorManager().getStateColor(state, this.theme);
            pill.style.background = color['background'];
            pill.style.color = color['text'];
            pill.style.borderColor = color['text'];
        });
    }

    _renderStateBadge(state) {
        if (!state) return '<div class="colored-pill gray-pill">N/A</div>'; // Fallback
        const state_pill = this.colorManager().getStateColor(state, this.theme);
        return `
            <div class="pt-status-option colored-pill" style="background:${state_pill['background']}; color:${state_pill['text']}; border:1px solid ${state_pill['text']};">
                ${state}
            </div>
        `;
    }

    _renderStateEditorOption(stateValue) {
        const option = document.createElement('div');
        option.className = 'pt-status-option-wrapper';
        option.dataset.value = stateValue;
        option.innerHTML = this._renderStateBadge(stateValue);
        return option;
    }

    async _setProgressBar(percentage, count, total) {
        percentage = Math.max(0, percentage);
        count = Math.max(0, count);
        if (this.progressBar) {
            requestAnimationFrame(() => {
                this.progressBar.style.transform = `translateX(-${100 - percentage}%)`;
                if (percentage >= 100) {
                    this.progressBar.classList.add('complete');
                } else {
                    this.progressBar.classList.remove('complete');
                }
            });
        }
        if (!this.progressBarPct || !this.progressBarCount) return;
        if (percentage > 25) {
            this.progressBarPct.setConfig({'suffix': 'Priced'});
        } else {
            this.progressBarPct.setConfig({'suffix': ''});
        }
        this.progressBarPct.update(percentage/100);
        const fake_suffix = `/${total}`;
        if (fake_suffix !== this.progressBarCount.config.suffix) {
            this.progressBarCount.setConfig({'suffix': fake_suffix});
        }
        this.progressBarCount.update(count);
    }

    // ===== Widgets =====
    getWidget(name) {return this.widgetManager.widgets.get(name)?.instance}

    async _initializeWidgets() {
        this.widgetManager = new WidgetManager(this.context);
        this.widgetManager.register(OverviewWidget, 'overviewWidget', this.ptGrid, '#overviewWidget');
        this.widgetManager.register(PivotWidget, 'pivotWidget', this.ptGrid, '#pivotWidget');
        // this.widgetManager.register(HeatmapWidget, 'heatmapWidget', this.ptGrid, '#heatmapWidget');
        this.widgetManager.register(ScriptWidget, 'scriptWidget', this.ptGrid, '#scriptWidget');
        // this.widgetManager.register(AdminToolsWidget, 'adminTools', this.ptGrid, '#adminToolsWidget');
        this.widgetManager.register(RefreshWidget, 'refreshWidget', this.ptGrid, '#refreshWidget');
        this.widgetManager.mount(this.ptGrid);

        this.widgetManager.widgets.get('pivotWidget').instance._init();
        await this.widgetManager.switch(this.defaultWidget);
        this._setupCopyButtons()
    }

    alignRadios() {
        const side = this.page$.get('activeSide');
        const new_side = this.sideGroup.querySelector(`input[value="${side}"]`);
        const old_side = this.sideGroup.querySelector("input:checked");
        if (new_side && old_side) {
            if (new_side.value !== old_side.value) {
                old_side.checked = false;
                new_side.checked = true;
            }
        }
    }

    _computeMarketAvailability(metricsData, state, adapter) {
        const engine = adapter.engine;
        const t = engine.numRows();
        if (!t) return metricsData;

        const mktsMap = new Map();
        const getRefMkt = engine._getValueGetter('refMktMkt');
        for (let i = 0; i < t; i++) {
            const mkt = getRefMkt(i);
            const m = mkt ? mkt.toUpperCase() : '_missing_';
            mktsMap.set(m, (mktsMap.get(m) || 0) + 1);
        }

        const w = state.waterfallEnabled;
        const m = mktsMap.get('_missing_') || 0;
        const r = { ...metricsData };

        r['Dynamic']['coverage'] = w ? 1 - m / t : null;
        r['Dynamic']['usage'] = w ? 1 - m / t : null;

        const fields = engine.fieldSet();
        const metrics = Object.keys(metricsData);
        for (let i = 0; i < metrics.length; i++) {
            const mkt = metrics[i];
            if (mkt === 'Dynamic') continue;
            const col = mkt + '_Level';
            if (fields.has(col) || engine._isDerived(col)) {
                r[mkt]['coverage'] = engine.computeStats(col).count / t;
                r[mkt]['usage'] = (mktsMap.get(mkt.toUpperCase()) || 0) / t;
            }
        }
        return r;
    }

    _handleRefMarketChange(newState, otherDropdown) {
        const mkt = newState.selected.market;
        const marketsToLoad = mkt === 'Dynamic'
            ? Array.from(this.marketDataMap.keys())
            : [mkt];
        this._ensureMarketColumns(marketsToLoad).catch(() => {});

        this.page$.set('activeMarket', mkt);
        this.page$.set('activeSide', newState.selected.side);
        this.alignRadios();

        if (otherDropdown) {
            otherDropdown.setValue('market', mkt, false, false);
            otherDropdown.setValue('side', newState.selected.side, false, false);
        }

        const t = document.querySelector('.refmkt-toggle');
        if (mkt === 'Dynamic') {
            if (t) {
                t.classList.toggle('toggle-disabled', true);
                const inp = t.querySelector('input');
                if (inp) inp.checked = true;
                newState.waterfallEnabled = true;
                this.page$.set('waterfallRef', true);
            }
        } else {
            if (t) t.classList.toggle('toggle-disabled', false);
            this.page$.set('waterfallRef', newState.waterfallEnabled);
        }
    }

    async _initializeManagersAndWidgets() {
        await this._setupPills();
        await this._initializeWidgets();

        const marketDataForDropdown = {"Dynamic": { coverage: 0, usage: 0, label: 'Asset Dynamic', value: 'Dynamic', abbr: 'Dynamic' }}

        const mkts = this.marketDataMap;
        Array.from(mkts.entries()).sort((a, b) => {
            const al = a[1]?.label || a[0];
            const bl = b[1]?.label || b[0];
            const pa = MARKET_PRIORITY[al];
            const pb = MARKET_PRIORITY[bl];
            if (pa !== undefined && pb !== undefined) return pa - pb;
            if (pa !== undefined && pb === undefined) return -1;
            if (pa === undefined && pb !== undefined) return 1;
            return al.localeCompare(bl);
        }).forEach(m => {
            const mm = m[1];
            const fld = m[0];
            marketDataForDropdown[fld] = {coverage: 0, usage: 0, value: fld, label: mm?.label || fld, abbr: mm?.abbr || mm?.label || fld}
        })

        const dropdownGroups = [
            {
                key: 'market',
                label: 'Market',
                info: 'Underlying source/provider of the base reference level. If Dynamic, reference market is determined based on the asset class of bond.',
                options: Object.entries(marketDataForDropdown).map(([field, expl])=>{return {value: field, label: expl.label, abbr: expl.abbr}}),
                showMetrics: true, // This group's options can show metrics
                metricKeySource: 'value' // Use option.value to lookup in metricsData
            },
            {
                key: 'side',
                label: 'Side',
                info: "Side of the market to use for the reference level. If 'By Side' then will use 'Bid' for Barclays Buys and 'Ask' for Barclays Sells.",
                options: [
                    { value: 'Auto', label: 'By Side' },
                    { value: 'Bid', label: 'Bid' },
                    { value: 'Mid', label: 'Mid' },
                    { value: 'Ask', label: 'Ask' }
                ]
            }
        ];

        const adapter = this.ptGrid;
        const page = this;
        const marketDataArr = Array.from(this.marketDataMap.keys());
        this._ensureMarketColumns(marketDataArr).catch(() => {});

        this.refDropdown = new ConfigurableDropdown(this.context, '#my-dropdown-container', '#refmkt-btn', {
            groups: dropdownGroups,
            metricsData: marketDataForDropdown,
            initialState: {
                selected: {
                    market: 'bval',
                    side: 'Mid'
                },
                waterfallEnabled: true,
                etfEnabled: false,
                displayMode: 'coverage'
            },
            placeholder: '',
            context: page.context,
            adapter: adapter,
            checkDataAvailability: async (selected, state, metricsData, adapter) => {
                return this._computeMarketAvailability(metricsData, state, adapter);
            },
            footer: {
                waterfallToggle: {
                    label: 'Waterfall missing data',
                    info: "If missing reference values in the requested market, fallback to other markets to fill the gaps. Recommended to keep on. MUST be enabled for 'Dynamic' market mode.",
                    stateKey: 'waterfallEnabled',
                },
                etfAdjustToggle: {
                    label: 'Adjust by ETF Premium? [BETA]',
                    info: "Attempt to automatically adjust levels by the current ETF Premium/Discount. Which ETF to use is based on the asset class of the underlying bond - or otherwise AGG. Only works during US trading hours.",
                    stateKey: 'etfEnabled', // Must match a key in initialState
                },
                displayModeToggle: {
                    label: 'Display mode',
                    info: "Controls the % numbers seen on this dropdown. Coverage shows the availability of data. Usage shows which markets are actually in use on the page based on waterfall logic.",
                    stateKey: 'displayMode', // Must match a key in initialState
                    modes: [
                        { value: 'coverage', label: 'Coverage %' }, // Custom label
                        { value: 'usage', label: 'Usage %' }    // Custom label
                    ]
                },
                warningMessageText: '⚠️ No data for this selection if waterfall is off.' // Custom text
            },
            onChange: (newState) => this._handleRefMarketChange(newState, this.refDropdownLower),
            onOpen: () => {},
            onClose: () => {},
        });

        await this.refDropdown.init();

        this.refDropdownLower = new ConfigurableDropdown(this.context, '#my-dropdown-container-lower', '#refmkt-btn-lower', {
            groups: dropdownGroups,
            metricsData: marketDataForDropdown,
            initialState: {
                selected: {
                    market: 'bval',
                    side: 'Mid'
                },
                waterfallEnabled: true,
                etfEnabled: false,
                displayMode: 'coverage'
            },
            placeholder: '',
            context: page.context,
            adapter: adapter,
            checkDataAvailability: async (selected, state, metricsData, adapter) => {
                return this._computeMarketAvailability(metricsData, state, adapter);
            },
            footer: {
                waterfallToggle: {
                    label: 'Waterfall missing data',
                    info: "If missing reference values in the requested market, fallback to other markets to fill the gaps. Recommended to keep on. MUST be enabled for 'Dynamic' market mode.",
                    stateKey: 'waterfallEnabled',
                },
                etfAdjustToggle: {
                    label: 'Adjust by ETF Premium? [BETA]',
                    info: "Attempt to automatically adjust levels by the current ETF Premium/Discount. Which ETF to use is based on the asset class of the underlying bond - or otherwise AGG. Only works during US trading hours.",
                    stateKey: 'etfEnabled', // Must match a key in initialState
                },
                displayModeToggle: {
                    label: 'Display mode',
                    info: "Controls the % numbers seen on this dropdown. Coverage shows the availability of data. Usage shows which markets are actually in use on the page based on waterfall logic.",
                    stateKey: 'displayMode', // Must match a key in initialState
                    modes: [
                        { value: 'coverage', label: 'Coverage %' }, // Custom label
                        { value: 'usage', label: 'Usage %' }    // Custom label
                    ]
                },
                warningMessageText: '⚠️ No data for this selection if waterfall is off.' // Custom text
            },
            onChange: (newState) => this._handleRefMarketChange(newState, this.refDropdown),
            onOpen: () => {},
            onClose: () => {},
        }, "-lower");

        await this.refDropdownLower.init();

        await this._setupRefSettings();
    }


    // ===== Socket Subscriptions =====
    async _setupWebSocketSubscriptions() {
        await this._subscribeToMeta();
        await this._subscribeToPortfolio();
    }

    // ===== Message Handlers =====
    async _subscribeToMeta() {
        const sm = this.context.page.subscriptionManager();
        await sm.subscribeToRoom(this.context.metaRoom, {
            grid_id: 'meta',
            grid_filters: {
                "field": "portfolioKey",
                "op": "eq",
                "value": this.context.portfolio_key
            },
            // Meta is a single row — always fetch all keys (no lazy loading).
            columns: null
        }).catch(console.error);

        const metaRoom = this.context.metaRoom;
        this.addMessageHandler(metaRoom, 'subscribe', this._handleMetaSubscription.bind(this));
        this.addMessageHandler(metaRoom, 'publish', this._parsePublishedMetaData.bind(this));
    }

    async _subscribeToPortfolio() {
        const sm = this.context.page.subscriptionManager();
        await sm.subscribeToRoom(this.context.portfolioRoom, {
            grid_id: 'portfolio',
            grid_filters: {
                "field": "portfolioKey",
                "op": "eq",
                "value": this.context.portfolio_key,
            },
            columns: null //this._getInitialColumns('portfolio') // added to pageBase
        }).catch(console.error);

        const ptRoom = this.context.portfolioRoom;
        this.addMessageHandler(ptRoom, 'subscribe', this._handlePortfolioSubscription.bind(this));
        this.addMessageHandler(ptRoom, 'publish', this._parsePublishedPortfolioData.bind(this));
        this.addMessageHandler(ptRoom, 'ack', this._parseAck.bind(this));

    }

    async _parseAck(message) {
        // console.log('ACK', message)
        if (message?.status?.code === 400) return
        if (message.status.reason === 'refresh') {
            if (this.getWidget('refreshWidget').isActive) {
                setTimeout(async ()=> await this.getWidget('refreshWidget').rebuildMarkets(), 3000);
            }
        }
    }

    async _handleMetaSubscription(message) {
        if (message?.status?.code === 400) {
            console.error('META subscription failed:', message?.error || message);
            this.portfolioMeta$.clear();
            return;
        }

        if (this.context.metaRoom !== message?.context?.room) {
            this.toastManager().error('Subscription', 'Failed to subscribe to portfolio room.');
            return;
        }

        if (!message?.data) {
            this.toastManager().error('Subscription', 'No META data returned for portfolio.');
            return;
        }

        this._metaRaw = message;
        this._metaHyper = message.data;
        this._metaStore.update(this._metaHyper.toObject()[0]);
        this._rooms.add(message?.context?.room);

        if (this._metaStateUnsub) {
            try {
                if (typeof this._metaStateUnsub === 'function') this._metaStateUnsub();
                else if (this._metaStateUnsub?.unsubscribe) this._metaStateUnsub.unsubscribe();
            } catch (_) {}
            this._metaStateUnsub = null;
        }

        this._metaStateUnsub = this._metaStore.pick(['client', 'state']).onChanges(() => {
            const client = this._metaStore.get('client');
            const state = this._metaStore.get('state');
            this.update_site_assets(client, state);
        });


    }

    update_site_title({client, state}) {
        if (client == null) client = this._metaStore.get('client');
        if (state == null) state = this._metaStore.get('state');
        return this.setWindowTitle(this._build_title(client, state));
    }

    update_site_icon(state) {
        switch (state) {
            case 'LIVE':
            case 'NEW':
            case 'PENDING':
                return this.setFavicon('blue')
            case 'WON':
                return this.setFavicon('green')
            case 'COVERED':
                return this.setFavicon('orange')
            case 'MISSED':
            case 'ERROR':
                return this.setFavicon('red')
            case 'DNT':
            case 'PASSED':
            case 'CANCELLED':
            case 'EXPIRED':
                return this.setFavicon('gray')
            case 'OTHER':
            case 'TEST':
            case 'NETDOWN':
            case 'TRANSFER':
            case 'INDICATIVE':
            case 'UNKNOWN':
                return this.setFavicon('purple');
            case 'TIED':
                return this.setFavicon('gold');
            default:
                return this.setFavicon()
        }
    }

    update_site_assets(client, state) {
        this.update_site_title({client, state})
        this.update_site_icon(state)
    }

    _build_title(client, state, l = 10) {
        let h = client;
        if (client.length > l) {
            h = h.substring(0, l) + '...';
        }
        const metaDesc = document.querySelector('meta[name="description"]');
        if (metaDesc) metaDesc.setAttribute('content', client);
        return `${h} [${state}]`;
    }

    _change_pill_color(element, color) {
        const newClass = color ? `${color}-pill` : null;
        Array.from(element.classList)
              .filter(c => !c.startsWith('pt-') && c !== 'colored-pill' && c !== newClass)
              .forEach(c => element.classList.remove(c));
        element.classList.toggle('colored-pill', true);
        if (color) element.classList.toggle(newClass, true);
    }

    getSizeAndCountPill() {
        const ntl = formatNumber(this.portfolioMeta$.get('grossSize')||0,
            {divisor: true, spacing: true, sigFigs: 1, trimSigFigs: 0}
        );
        const cnt = formatNumber(this.portfolioMeta$.get('count')||0,
            {divisor: false, spacing: false, sigFigs: 0, trimSigFigs: 0}
        );
        return `${ntl} (${cnt})`
    }

    setupMetaEvents() {
        this._subs.push(this.portfolioMeta$.onValueAddedOrChanged('state', (activeState)=>{
            this.activeState = activeState;
            if (this.timeoutId) clearTimeout(this.timeoutId);
            this._updateStatusDisplay();
            this._metaUpdateHeader(this.activeState);
        }));

        this._subs.push(this.page$.onValueConfirmed('domAnimationFinished', () => {
            this._setupGridSearch();
            this.contentArea?.classList.add('initialized');
            this.push_pull_buttons.classList.remove('hidden');
        }));

        // dueTime subscription is registered once (not nested inside rfqSide) to
        // prevent unbounded listener accumulation.  Uses this._dueTimeUnsub so
        // _teardownGrid() can clean it up on reconnect.
        this._dueTimeUnsub = this.portfolioMeta$.onValueAddedOrChanged('dueTime', (dueTime)=>{
            if (isFuture(dueTime)) {
                if (this.timeoutId) clearTimeout(this.timeoutId);
                this._refresh_status_color('LIVE');
                this.timedState(new Date(dueTime));
            } else {
                if (this.timeoutId) clearTimeout(this.timeoutId);
                this._metaUpdateHeader(this.portfolioMeta$.get('state'), false);
                this._refresh_status_color(this.portfolioMeta$.get('state'));
            }
        });

        this._subs.push(this.portfolioMeta$.onValueAddedOrChanged('rfqSide', (side)=>{
            this.side = side;
            this.headerDirectionElem.textContent = (this.side || "* UNKNOWN SIDE *");
            const directionColor = this.colorManager().getSideColor(this.side);

            this._change_pill_color(this.headerSizeElem, directionColor);
            this._change_pill_color(this.headerNetElem, directionColor);
            this._change_pill_color(this.headerCountElem, directionColor);
            this._change_pill_color(this.headerDirectionElem, directionColor);
        }));

        this._bindElementToMeta(this.headerNameElem, 'client', (client) => this._nameNormalizer(client), {setTitle: true});

        this._bindElementToMeta(this.headerSizeElem, 'grossSize', (size) => {
            return this.getSizeAndCountPill()
        });

        this._bindElementToMeta(this.headerSizeElem, 'count', (size) => {
            return this.getSizeAndCountPill()
        });

        // this._bindElementToMeta(this.headerNetElem, 'netSize', (size) => formatNumber(this.portfolioMeta$.get('netSize')||0,
        //     { divisor: true, spacing: true, sigFigs: 1, trimSigFigs: 0, suffix:' (net)'}
        // ));
        // this._bindElementToMeta(this.headerCountElem, 'count', (size) => formatNumber(this.portfolioMeta$.get('count')||0,
        //     { divisor: false, spacing: true, sigFigs: 1, trimSigFigs: 0, suffix:'bonds'}
        // ));


        this.addEventListener(this.headerSizeElem, 'click', () => {
            this.metaSizeState  = this.metaSizeState === 'grossSize' ? 'netSize' : (this.metaSizeState === 'netSize' ? 'count' : 'grossSize');
            let v = this.portfolioMeta$.get(this.metaSizeState);
            if (this.metaSizeState === 'grossSize') {
                v = this.getSizeAndCountPill();
            } else if (this.metaSizeState === 'netSize') {
                if (v !== this.portfolioMeta$.get('grossSize')) {
                    v = formatNumber(v || 0, { divisor: true, spacing: true, sigFigs: 1, trimSigFigs: 0, suffix:' (net)'})
                } else {
                    this.metaSizeState = 'count';
                    v = this.portfolioMeta$.get(this.metaSizeState);
                    v = formatNumber(v || 0, { divisor: false, spacing: true, sigFigs: 1, trimSigFigs: 0, suffix:'bonds'})
                }
            } else {
                v = formatNumber(v || 0, { divisor: false, spacing: true, sigFigs: 1, trimSigFigs: 0, suffix:'bonds'})
            }
            this.headerSizeElem.textContent = v;
        });

        // const strikeTime = meta.isFwdStrike ? `${meta.fwdStrikeMkt} ${meta.fwdStrikeTimeMnemonic} ${meta.fwdStrikeSide}` : null;
    }

    _getMaxTime(wire=null, due=null) {
        wire = wire != null ? wire : Number(this.portfolioMeta$.get('wireTime'));
        due = due != null ? due : this.portfolioMeta$.get('dueTime');
        if (wire && due) {
            const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
            due = toZonedTime(due, tz);
            const wireMins = roundTo(coerceToNumber(wire) / 60, 0);
            return 'At max: ' + format(addMinutes(due, wireMins), "p");
        }
    }

    _bindElementToMeta(element, key, fn, {setTitle=false}={}) {
        this._subs.push(this.portfolioMeta$.onValueAddedOrChanged(key, (v) => {
            if (fn) v = fn(v);
            element.textContent = v;
            if (setTitle) element.title = v;
        }));
    }

    _bindSubItemToMeta(key, {fn=null, color="green", modal=null, tooltip=null, tooltipFn=null}={}) {
        this._subs.push(this.portfolioMeta$.onValueAddedOrChanged(key, (v) => {
            let e = this.metaElements.get(key);
            v = fn ? fn(v) : v;

            if (v == null) {
                if (e) {
                    e.remove();
                    this.metaElements.delete(key);
                    const m = this.metaElemEvents.get(key);
                    if (m) this.removeEventListener(m);
                }
                return;
            }
            if (!e) {
                e = document.createElement("div");
                this.metaElements.set(key, e);
                this.headerSubItemsElem.append(e);
            }

            e.textContent = v;
            this._change_pill_color(e, color);

            if (tooltipFn) {
                tooltip = tooltipFn(tooltip);
            }
            if (tooltip) {
                e.classList.toggle('tooltip-top', true);
                e.setAttribute('data-tooltip', tooltip);
            } else {
                e.classList.toggle('tooltip-top', false);
                e.removeAttribute('data-tooltip');
            }
            const m = this.metaElemEvents.get(key);
            if (m) this.removeEventListener(m);
            if (modal) {
                e.classList.toggle('modal-info', true);
                const u = this.addEventListener(e, 'click', () => {
                    this.createInfoModal(modal?.title, modal?.info, 'info', modal?.subtitle);
                });
                this.metaElemEvents.set(key, u);
            } else {
                e.classList.toggle('modal-info', false);
            }
        }));
    }

    _refresh_status_color(state=null) {
        if (!state) state = this.portfolioMeta$.get('state');
        const newColor = this.colorManager().getStateColor(state, this.theme);
        this.headerStatusElem.style.setProperty('background', newColor['background']);
        this.headerStatusElem.style.setProperty('color', newColor['text']);
        this.headerStatusElem.style.setProperty('border-color', `${newColor['text']}`);
    }

    _metaUpdateHeader(newState, animate=true) {
        const previous = this.headerStatusElem.textContent;
        this.headerStatusElem.textContent = newState || 'UNKNOWN';
        this._refresh_status_color(newState);
        if (animate && this.headerStatusElem && previous && (previous !== "UNKNOWN") && this.gridSettings$.get('showPtStatus')) {
            const header = this.headerStatusElem;
            this.addEventListener(header, "animationend", () => {
                header.classList.remove("wobble-ver-right");
            }, {once: true});
            header.classList.add("wobble-ver-right");

            // A very important feature
            if (newState === 'WON') {
                confetti({
                    origin:getElementCoordinatesAsPercentage(this.headerStatusElem),
                    angle: 120,
                    zIndex: 1
                });
            }
        }
    }

    // columnar → row conversion
    _parseMessagePayloads(data) {
        const payloads = {};
        for (const action of ['add', 'update', 'remove']) {
            const p = data.payloads[action];
            if (p) payloads[action] = p._format ? EnhancedPayloadBatcher.fromColumnar(p) : p;
        }
        return payloads
    }

    _parsePublishedMetaData(message) {
        const data = message?.data;

        if (!data || !data.payloads) {
            this.toastManager().warning("Message", `Received a message with no data for ${this.context.metaRoom}`);
            return
        }

        const payloads = this._parseMessagePayloads(data);
        if (payloads?.add) asArray(payloads?.add).forEach(p => this.portfolioMeta$.merge(p));
        if (payloads?.update) asArray(payloads?.update).forEach(p => this.portfolioMeta$.merge(p));
        if (payloads?.remove && payloads?.remove.length > 0) {
            this.toastManager().warning('Unexpected Message', 'Received a message to remove meta data from context which is odd.')
        }

    }

    // ───────── Teardown (memory-leak fix) ─────────
    _teardownGrid() {
        // 1. Unsubscribe all store listeners accumulated in _subs
        for (let i = 0; i < this._subs.length; i++) {
            try {
                const sub = this._subs[i];
                if (typeof sub === 'function') sub();
                else if (sub && typeof sub.unsubscribe === 'function') sub.unsubscribe();
            } catch (_) {}
        }
        this._subs.length = 0;

        if (this._metaStateUnsub) {
            try { typeof this._metaStateUnsub === 'function' ? this._metaStateUnsub() : this._metaStateUnsub?.unsubscribe?.(); } catch (_) {}
            this._metaStateUnsub = null;
        }

        // 2. Unsubscribe dueTime listener if any
        if (this._dueTimeUnsub) {
            try { this._dueTimeUnsub(); } catch (_) {}
            this._dueTimeUnsub = null;
        }

        // 3. Shutdown pill manager (detaches epoch + emitter + destroys pills)
        try { this.pillManager?.shutdown?.(); } catch (_) {}
        this.pillManager = null;

        // 4. Dispose widget manager (pivot, overview, etc.)
        try { this.widgetManager?.cleanup?.(); } catch (_) {}
        this.widgetManager = null;

        // 5. Dispose old grid (also disposes its engine internally)
        try { this.ptGrid?.dispose(); } catch (_) {}
        this.ptGrid = null;

        // 6. If engine survived (e.g. grid.dispose didn't null it), dispose separately
        try { this.engine?.dispose(); } catch (_) {}
        this.engine = null;

        // 7. Clear emitter tokens accumulated from setupGridReactives / setupEmitter
        if (this.emitter && this.emitter.clear) {
            try {this.emitter.clear()} catch(_){}
        }

        // 8. Clear timer map
        this._timerMap.forEach((id) => { try { clearTimeout(id); } catch (_) {} });
        this._timerMap.clear();
        if (this.timeoutId) { clearTimeout(this.timeoutId); this.timeoutId = null; }

        // 9. Clear meta DOM elements and their event listeners
        if (this.metaElemEvents.size) {
            this.metaElemEvents.forEach((unsub) => { try { this.removeEventListener(unsub); } catch (_) {} });
            this.metaElemEvents.clear();
        }
        if (this.metaElements.size) {
            this.metaElements.forEach((el) => { try { el.remove(); } catch (_) {} });
            this.metaElements.clear();
        }

        if (this.refDropdown) {
            try { this.refDropdown.destroy?.(); } catch (_) {}
            this.refDropdown = null;
        }
        if (this.refDropdownLower) {
            try { this.refDropdownLower.destroy?.(); } catch (_) {}
            this.refDropdownLower = null;
        }

        // 11. Destroy meta editor
        if (this.metaEditor) {
            try { this.metaEditor.destroy?.(); } catch (_) {}
            this.metaEditor = null;
        }

        // 12. Destroy micro-grid manager (may be recreated in onInit)
        // Note: only if reconnect path recreates it
        // if (this._microGridManager) {
        //     try { this._microGridManager.destroy(); } catch (_) {}
        //     this._microGridManager = null;
        // }

        // 13. Clean up epoch reaction token
        if (this.pctPricedToken) {
            try { this.pctPricedToken(); } catch (_) {}
            this.pctPricedToken = null;
        }

        // 14. Clean up overallSkew flows
        if (this._bidSkewFlow) {
            try { this._bidSkewFlow.destroy(); } catch (_) {}
            this._bidSkewFlow = null;
            this.bidSkew = null;
        }
        if (this._askSkewFlow) {
            try { this._askSkewFlow.destroy(); } catch (_) {}
            this._askSkewFlow = null;
            this.askSkew = null;
        }

        // 15. Null large data references
        this._ptRaw = null;
        this._ptHyper = null;

    }

    // ───────── PORTFOLIO snapshot & updates ─────────
    async _handlePortfolioSubscription(message) {
        if (message?.status?.code === 400) {
            console.error('Portfolio subscription failed:', message?.error || message);
            return;
        }

        if (this.context.portfolioRoom !== message?.context?.room) {
            this.toastManager().error('Subscription', 'Failed to subscribe to portfolio room.');
            return;
        }

        if (!message?.data) {
            this.toastManager().error('Subscription', 'No PORTFOLIO data returned for portfolio.');
            return;
        }

        // Tear down previous engine/grid/subscriptions before rebuilding (reconnect-safe)
        if (this.ptGrid || this.engine) {
            this._teardownGrid();
        }

        this._ptRaw = message;
        this._ptHyper = message.data;
        const room = message?.context?.room;
        this._rooms.add(room);

        // this.gridManager.setupDataAdapter(this._ptHyper);
        this.rowSelectionKey = `${room}:rowSelections`;
        this.rowSelections$ = this.createSharedStore(this.rowSelectionKey, new Map(), {persist: true, storageKey:this.rowSelectionKey, storageType:'session'});

        const settingsDict = this.page$;

        const fullSchema = message.data?.schema();
        const allColumnNames = message.data?.metaData?.all_columns
            || message.context?.all_columns
            || null;

        this.engine = new ArrowEngine(
            this.context,
            this._ptHyper.toArrow(),
            {
                idProperty:'tnum',
                rowIdGetter: (ri, engine) => engine.getCell(ri, 'tnum'),
                master_schema: fullSchema,
                allColumnNames: allColumnNames
            }
        );

        const socketManager = this.context.page.socketManager();
        const page = this;
        const ctx = message.context;
        // this.engine.enableLazyColumns(async (columnNames) => {
        //
        //     const response = await socketManager._sendWebSocketMessage({
        //         action: ACTION_MAP.get("fetch_columns"),
        //         context: ctx,
        //         columns: columnNames
        //     }, {wait:true});
        //     const data = response.data;
        //
        //     return (data && typeof data.toArrow === 'function')
        //         ? data.toArrow()
        //         : data;
        //
        // }, { debounceMs: 16} )

        // Track lazily-fetched columns so subsequent visits include them upfront
        // this.trackEagerColumns(this.engine, 'portfolio');
        //
        // Pre-fetch essential columns
        // this.engine.ensureColumns([
        //     'newLevel',
        // ]).catch(() => {});

        let patternBidPx = /bidpx/i;
        let patternExclude = /(benchmark|skew)/i;
        let patternExtract = /^(.*?)BidPx/i;
        let isRegionalMark = /(^|\s)mark(\s|$)/i
        let isAllqMarket = /(^|\s)allq(\s|$)/i

        const flds = this.engine.fields()
        const mkts = this.marketDataMap;
        for (const fld of flds) {
            if (patternBidPx.test(fld) && !patternExclude.test(fld)) {
                const match = patternExtract.exec(fld);
                if (match && match.length) {
                    const market = match[1]
                    if (market !== 'newLevel') {
                        let _lbl = StringFormatter.fromCamelCase(market).toUpperCase();
                        let k = _lbl;
                        if (_lbl === "MARKIT RT") {
                            _lbl = "MARKIT (Live)"
                            k = "MARKIT RT"
                        } else if (_lbl === "MA LIVE") {
                            _lbl = "MA (Live)"
                            k = "MA LIVE"
                        } else if (_lbl === "MARKIT") {
                            _lbl = "MARKIT (Hourly)"
                            k = "MARKIT"
                        } else if (_lbl === "BVAL") {
                            _lbl = "BVAL (Latest)"
                            k = "BVAL"
                        } else if ((_lbl !== "MARK") && isRegionalMark.test(_lbl)) {
                            const p = _lbl.split(" ");
                            _lbl = `${p[1]} (${p[0]})`;
                            k = `${p[0]} ${p[1]}`;
                        } else if ((_lbl !== "ALLQ") && isAllqMarket.test(_lbl)) {
                            const p = _lbl.split(" ");
                            _lbl = `${p[0]} (${p[1]})`;
                            k = p[1]
                        }
                        mkts.set(market, {abbr: k, label: _lbl})
                    }
                }
            }
        }

        let columnDefs = [...virtualPortfolioColumns, ...realPortfolioColumns];
        const n = [];
        for (const x of this.marketDataMap.keys()) {
            n.push(...generateAllMarketColumns(x, message.data?.schema().fields.map(x=>x.name)));
        }
        columnDefs = [...columnDefs, ...n];
        const known_fields = new Set(columnDefs.map(col=>col.field));
        const all_fields = this.engine.table.schema.fields.map(f=>f.name);
        const missing_fields = all_fields.filter(x => !known_fields.has(x));
        const m = missing_fields.map(f=>this.create_generic_field(f, generalPortfolioColumns, this.engine));
        columnDefs = [...m, ...columnDefs];

        this.ptGrid = await ArrowAgGridAdapter.create(this.context, this.engine, columnDefs, {
            settingsDict
        }, {globalViews:this.globalPreset(), name:'portfolio', enableFilterMemory:false, enablePasteOverride:true});

        requestAnimationFrame(async ()=>{
            this.ptGrid.mount(this.selector);
            this.setupEmitter();
            this.setupMetaEditor();
            await this._initializeManagersAndWidgets();
            this.setupGridReactives();
            await this.setupDynamicTooltips();
            this._setupMicroGridFlags();
        })
    }

    setupMetaEditor() {
        this.metaEditor = new MetaEditorModal(this.engine, 0, this._metaStore, this.context, {asStore:true});
    }

    globalPreset() {
        return [
            {
                name: "Default View",
                version: '1.2.2', // Version for cache-busting
                columnState: [
                    {
                        "colId": "tnum",
                        "width": 75,
                        "hide": true,
                    },
                    {
                        "colId": "isin",
                        "width": 120,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "description",
                        "width": 197,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "assignedTrader",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "desigName",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "claimed",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "grossSize",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "grossDv01",
                        "width": 45,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "userSide",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "assetClass",
                        "width": 45,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "ratingCombined",
                        "width": 49,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "macpLiqScore",
                        "width": 44,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "bsrFlag",
                        "width": 44,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "firmAggPosition",
                        "width": 88,
                        "hide": false,
                        "pinned": null,

                    },
                    {
                        "colId": "isAxed",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "algoPosition",
                        "width": 67,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "signalTotal",
                        "width": 63,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "lastEditUser",
                        "width": 86,
                        "hide":false,
                        "pinned": null,
                    },
                    {
                        "colId": "comment",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                    },
                    {
                        "colId": "refLevel",
                        "width": 75,
                        "hide": false,
                        "pinned": 'right',
                    },
                    {
                        "colId": "refMktDisp",
                        "width": 75,
                        "hide": false,
                        "pinned": 'right',
                    },
                    {
                        "colId": "QT",
                        "width": 75,
                        "hide": false,
                        "pinned": 'right',

                    },
                    {
                        "colId": "refSkew",
                        "width": 75,
                        "hide": false,
                        "pinned": 'right',
                    },
                    {
                        "colId": "newLevelDisplay",
                        "width": 90,
                        "hide": false,
                        "pinned": 'right',
                    },
                ],
                metaData: {
                    isMutable: false,                          // Can the user save over this view?
                    isTemporary: false,                        // Will this view persist into local browser storage
                    isGlobal: true,                            // Set by the server or by user
                    isDefault: false,                          // the active default state
                    owner: 'Cade Zawacki',                     // Who created the view originally
                    lastModified: '2025-08-14T07:29:12.796Z',  // Last modified timestamp, Date().toISOString()
                }
            },
            {
                name: "Igor's View",
                version: '1.0.0',
                columnState: [
                    {
                        "colId": "tnum",
                        "width": 75,
                        "hide": true,
                        "pinned": "left",
                    },
                    {
                        "colId": "ticker",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "size",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "isin",
                        "width": 128,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "desigName",
                        "width": 143,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "QT",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "macpLiqScore",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "userSide",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "isAxed",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "description",
                        "width": 200,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "AM_Level",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "refSkew",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "refLevel",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "newLevelDisplay",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "comment",
                        "width": 124,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "algoPosition",
                        "width": 112,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "firmAggPosition",
                        "width": 99,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "amountOutstanding",
                        "width": 137,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "signalTotal",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "ratingCombined",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "lastEditUser",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "signalStats",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "refMktMkt",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "smadSector",
                        "width": 200,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "currency",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "grossDv01",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "bvalSnapshot",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "benchmarkName",
                        "width": 75,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    },
                    {
                        "colId": "benchmarkIsin",
                        "width": 125,
                        "hide": false,
                        "pinned": null,
                        "sort": null
                    }
                ],
                metaData: {
                    isMutable: false,                          // Can the user save over this view?
                    isTemporary: false,                        // Will this view persist into local browser storage
                    isGlobal: true,                            // Set by the server or by user
                    isDefault: false,                          // the active default state
                    owner: 'Igor Pavlov',                     // Who created the view originally
                    lastModified: '2025-09-12T07:29:12.796Z',  // Last modified timestamp, Date().toISOString()
                }
            }
        ];
    }

    async _parsePublishedPortfolioData(message) {
        const data = message?.data;
        if (!data || !data.payloads) return;

        const payloads = {};
        let _apply = false;
        for (const action of ['add', 'update', 'remove']) {
            const p = data.payloads[action];
            if (p) {
                payloads[action] = p._format ? EnhancedPayloadBatcher.fromColumnar(p) : p;
                _apply = true;
            }
        }

        if (_apply) {
            // Don't await — let the transaction process asynchronously.
            // applyServerTransaction internally coalesces and debounces.
            this.ptGrid.applyServerTransaction(payloads, { emitAsEdit: false })
            .catch(e => console.error('[pt] applyServerTransaction failed:', e));
        }

        if (message?.options?.full_reset === true) {
            // Full reset DOES need to wait for the transaction to complete.
            // Queue it after the transaction settles.
            const doReset = () => {
                this.engine?.resetOverlays?.();
                this.ptGrid?._hardRefresh?.();
                this.getWidget('pivotWidget')?.refresh?.();
            };

            if (_apply) {
                // Wait for the fire-and-forget to settle first
                setTimeout(doReset, 100);
            } else {
                doReset();
            }
        }
    }

    _nameNormalizer(name) {
        if (!name) return "* UNKNOWN NAME *";
        name = name.toString();
        if (name.split(" ").length === 1) return name;
        if (name === name.toUpperCase()) return StringFormatter.toCapitalizeEachWord(name);
        return name;
    }

    timedState(dueTime) {
        const updateTime = () => {
            const currentTime = new Date();
            let remainingMs = dueTime.getTime() - currentTime.getTime();

            if (remainingMs > 0) {
                const remainingHours = Math.floor(remainingMs / 3600000);
                const remainingMinutes = Math.floor((remainingMs % 3600000) / 60000);
                const remainingSeconds = Math.floor((remainingMs % 60000) / 1000);

                let newContent;
                if (remainingHours > 0) {
                    newContent = `${remainingHours}:${String(remainingMinutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
                } else {
                    newContent = `${remainingMinutes}:${String(remainingSeconds).padStart(2, '0')}`;
                }

                this.headerStatusElem.innerText = newContent;
                this.update_site_title({state:newContent});
                const nextUpdateIn = 1000 - (Date.now() % 1000);
                this.timeoutId = setTimeout(updateTime, nextUpdateIn);
            } else {
                this._metaUpdateHeader('PENDING', false);
            }
        };

        updateTime();
    }

    getModalIcon(type) {
        if (type === 'info') {
            return `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><path fill="currentColor" d="M12 17q.425 0 .713-.288T13 16v-4q0-.425-.288-.712T12 11t-.712.288T11 12v4q0 .425.288.713T12 17m0-8q.425 0 .713-.288T13 8t-.288-.712T12 7t-.712.288T11 8t.288.713T12 9m0 13q-2.075 0-3.9-.788t-3.175-2.137T2.788 15.9T2 12t.788-3.9t2.137-3.175T8.1 2.788T12 2t3.9.788t3.175 2.137T21.213 8.1T22 12t-.788 3.9t-2.137 3.175t-3.175 2.138T12 22m0-2q3.35 0 5.675-2.325T20 12t-2.325-5.675T12 4T6.325 6.325T4 12t2.325 5.675T12 20m0-8"/></svg>`;
        }
        return null;
    }

    createInfoModal(title, content, type = "info", subtitle=null) {
        const modal = document.createElement('dialog');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-box pt-info-box">
                <div class="modal-header-wrapper">
                    ${this.getModalIcon(type)}
                    <h3 class="font-bold">${title || 'Information'}</h3>
                </div>
                <div class="modal-header-subtitle">${subtitle ? subtitle : ''}</div>
                <div class="modal-body-wrapper">
                    ${content || '<p>No details provided.</p>'}
                </div>
            </div>
            <form method="dialog" class="modal-backdrop">
                <button class="modal-closer">close</button>
            </form>
        `;
        document.body.appendChild(modal);
        this.addEventListener(modal.querySelector('.modal-closer'), 'click', () => modal.close(), { once: true });
        this.addEventListener(modal, 'close', () => modal.remove(), { once: true });
        modal.showModal();
    }

    _setupScrollShadow() {
        if (!this.scrollContainer) return;

        this.osInstance = OverlayScrollbars(this.scrollContainer, {
            overflow: { x: 'hidden' },
            update: {
                elementEvents: [[]],
                debounce: {
                    mutation: [0, 33],
                    resize: null,
                    event: [33, 99],
                    env: [222, 666, true],
                },
                ignoreMutation: true,
            },
            scrollbars: {
                visibility: 'visible',
                theme: 'os-theme-light',
                clickScroll: false,
                autoHide: 'never',
                pointers: ['mouse']
            }
        });
    }

    _skewTypeForQt(qt) {
        qt = qt.toString().toUpperCase();
        if (
            qt === 'PX' ||
            qt === 'BPS' ||
            qt === 'BPS MID' ||
            qt === 'CASH' ||
            qt === 'DIRTY'
        ) return 'Px'
        return 'Spd'
    }

    _determineSkewSide(side, unit_hint=null) {
        if (!side || !side.length) return {};
        const qtMap = {};
        side.forEach(s => {
            const qt = this._skewTypeForQt(s.QT);
            if (!qtMap[qt]) qtMap[qt] = 0;
            qtMap[qt] += s.Gross;
        });

        let b, bs=0;
        Object.entries(qtMap).forEach(([q, s]) => {
            if (s > bs) {b = q; bs = s;}
        });

        const skewCol = `refSkew${b}`;
        const w = side.reduce((acc, current) => {
            acc.prod += (current.Gross * current?.[skewCol]);
            acc.weight += current.Gross;
            return acc
        }, {prod: 0, weight: 0});

        const result = w.weight !== 0 ? (w.prod / w.weight) : null;
        if (result == null) return {};
        return {value:result, unit:b}
    }

    _determineSkew(totals, unit_hint=null) {
        if (!totals || !totals.length) return 0;
        const bids = totals.filter(t=>t?.userSide === 'BID');
        const asks = totals.filter(t=>t?.userSide === 'OFFER');
        const bidObj = this._determineSkewSide(bids, unit_hint);
        const askObj = this._determineSkewSide(asks, unit_hint);
        const bidStr = bidObj && bidObj?.value != null ? bidObj.value : 0;
        const askStr = askObj && askObj?.value != null ? askObj.value : 0;
        const bidUnit = (bidObj?.unit?.toUpperCase() === 'PX' ? "pts" : "bps");
        const askUnit = (askObj?.unit?.toUpperCase() === 'PX' ? "pts" : "bps");
        return [{value:bidStr, unit:bidUnit}, {value:askStr, unit:askUnit}]
    }

    setupUser() {
        const grid = this.ptGrid;
            this._subs.push(
                this.userManager().userProfile$.selectKeys(['displayName','username']).onChanges( (ch) => {
                grid.api.redrawRows()
            })
        );
    }

    setupGridReactives() {

        const engine = this.engine;
        const filterToken = this.emitter.on(ArrowAgGridAdapter.FILTER_EVENT, this.checkFilterIcon.bind(this));
        this.addEventListener(this.clearFilterButton, 'click', (e) => {
            this.ptGrid.clearFilters();
            this.clearPivotFilters();
            this.page$.set('quickSearch', null);
        });

        const pt_row_event = ArrowAgGridAdapter.ROW_COUNT_EVENT + ":portfolio";
        const rowCountToken = this.emitter.on(pt_row_event, async () => {
            const n = this._getCountOfPriced();
            if (this.progressBarCount) this.progressBarCount.setConfig({'suffix': `/${n.total}`});
            await this.updatePctPriced(n);
            await this.updateSkew();
        });
        this.setupGridEpochReactions();
        this.setupUser();

        const qt_cols = this.ptGrid?.getAllColumnDefs()?.filter(x=> x.context?.deps?.includes('QT')).map(x=>x.field);
        const api = this.ptGrid?.api;
        const grid = this.ptGrid;

        // Refresh QT-dependent cells when the active quote type changes.
        this._subs.push(this.activeQuoteType$.onChanges(() => {
            const v = new Set(grid._projection);
            if (!v) return;
            const vc = qt_cols.filter(x => v.has(x));
            //api.refreshCells({columns: vc, force: true});
            api.refreshServerSide?.({ purge: false });
        }));

        this._subs.push(this.qtSigFigs$.onRawChanges((ch) => {
            const v = new Set(grid._projection);
            if (!v) return
            const vc = qt_cols.filter(x=>v.has(x));
            //api.refreshCells({columns:vc});
            api.refreshServerSide?.({ purge: false });
        }));

        const d = this._determineSkew.bind(this);
        const overallSkew$ = this.overallSkew$;
        this._subs.push(this.getWidget('pivotWidget').ptPivot.onComputed((totals) => {
            const f = d(totals);
            overallSkew$.update({
                bid: f[0],
                ask: f[1]
            });
        }));
        this._subs.push(this.getWidget('pivotWidget').ptPivot.onComputedFull((totals) => {
            const f = d(totals);
            overallSkew$.update({
                bid: f[0],
                ask: f[1]
            });
        }));

        this._subs.push(this.portfolioMeta$.onValueAddedOrChanged('assetClass', (assetClass)=>{
            if (assetClass === 'EM') {
                this.refDropdown.setValue('market', 'cbbt', true);
            }
        }));
    }

    setupSkewReactions() {
        // Cleanup previous instances
        this.skew_initialized = true;
        if (this.frameSkew) this.frameSkew.innerHTML = '';
        if (this._bidSkewFlow) { try { this._bidSkewFlow.destroy(); } catch(_){} }
        if (this._askSkewFlow) { try { this._askSkewFlow.destroy(); } catch(_){} }

        requestAnimationFrame(() => {
            this.frameSkew.style.opacity = '100';
        })
        const side = this._metaStore.get('rfqSide');
        if (side === 'BOWIC' || side === 'BWIC') {
            this.bidSkew = new ENumberFlow(this.frameSkew, {
                displayMode:'number',
                showSign: 'always',
                duration: 200,
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });
            this._bidSkewFlow = this.bidSkew;
        }
        const s = document.createElement('span')
        s.innerText = '/'
        this.frameSkew.appendChild(s);

        if (side === 'BOWIC' || side === 'OWIC') {
            this.askSkew = new ENumberFlow(this.frameSkew, {
                displayMode: 'number',
                showSign: 'always',
                duration: 200,
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });
            this._askSkewFlow = this.askSkew;
        }

        const unsub = this.overallSkew$.onChanges(() => {
            this.updateSkew();
        });

        // Also listen for ref market/side/quote type changes
        const refUnsub = this.activeRefSettingsWaterfall$.onChanges(() => {
            this.updateSkew();
        });

        this.updateSkew();
    }

    updateSkew(bid=null, ask=null) {
        const u = this._updateSkew.bind(this)
        return debounce(u, 200)(bid, ask)
    }

    _updateSkew(bid=null, ask=null) {
        if (!this.skew_initialized) {
            this.setupSkewReactions();
        }

        const skews = this.overallSkew$.asObject();
        const bid_value = bid ?? skews?.bid?.value;
        const ask_value = ask ?? skews?.ask?.value;

        if ((this?.askSkew != null) && (ask_value != null)) {
            const unit = skews.ask.unit;
            if (this.askSkew?.config.suffix !== unit) {
                this.askSkew.setConfig({'suffix': unit});
            }
            const v = roundToNumeric(ask_value, 2);
            if (this.askSkew?.currentValue !== v) {
                this.askSkew.update(v);
            }
        } else if ((this?.bidSkew != null) && (bid_value != null)){
            const unit = skews.bid?.unit
            if (this.bidSkew?.config.suffix !== unit) {
                this.bidSkew.setConfig({'suffix': unit});
            }
        }
        if ((this?.bidSkew != null) && (bid_value != null)) {
            const v = roundToNumeric(bid_value, 2);
            if (this.bidSkew && this.bidSkew.update && (this.bidSkew?.currentValue !== v)) {
                this.bidSkew.update(v);
            }
        }
    }

    setupGridEpochReactions() {
        this.pctPricedToken = this.engine.onColumnEpochChange(
            ['newLevel', 'isReal'],
            () => {
                this.updatePctPriced();
                this.updateSkew();
            },
            {debounce: 200}
        )
    }

    _getCountOfPriced() {
        const engine = this.engine;
        if (!engine) return { priced: 0, total: 0, pct: 0 };

        const n = engine.numRows() | 0;
        const getPriced = engine._getValueGetter('isPriced');
        const getReal = engine._getValueGetter('isReal');

        let priced = 0;
        let total = 0;

        for (let i = 0; i < n; i++) {
            const isReal = Number(getReal(i));
            if (!isReal) continue;
            total += isReal;
            priced += Number(getPriced(i)) * isReal;
        }

        return { priced, total, pct: total ? priced / total : 0 };
    }

    async updatePctPriced(pctPriced = null) {

        if (pctPriced == null) {
            pctPriced = this._getCountOfPriced();
        }

        const {priced, total, pct} = pctPriced;

        if ((pct == null)) {
            console.error("Invalid progress:", pctPriced)
            return;
        }
        await this._setProgressBar(Number(pct)*100, Number(priced), Number(total));
    }

    checkFilterIcon(force=null) {
        if (force !== null) {
            force = coerceToBool(force, {nullishAsFalse: false})
            this.clearFilterIcon.classList.toggle('has-grid-filters', force);
        }
        const k = this.ptGrid.grid$.get("filterModel");
        const fmCheck = k && (Object.keys(k)?.length);
        const qsCheck = ! _isNullyString(this.page$.get("quickSearch"));

        const hasFilters =  fmCheck || qsCheck;
        this.clearFilterIcon.classList.toggle('has-grid-filters', hasFilters);
    }


    clearPivotFilters() {
        try { this?.getWidget('pivotWidget')?.api?.filterManager?.setFilterModel({}) } catch {}
    }

    _setupAutoSwitch() {
        // this._killAutoToggle = () => {
        //     this.yet_to_auto = false;
        //     this.autoTabController.abort();
        //     this.osInstance.elements().viewport.removeEventListener('scroll', autoToggle, {signal: this.buttonController.signal});
        // }
        // const autoToggle = async (e) => {
        //     const vp = this.osInstance.elements().viewport;
        //     const pos = vp.scrollTop >= (0.5 * (vp.scrollHeight - vp.clientHeight));
        //     if (pos) {
        //         this._killAutoToggle()
        //         if (this.gridSettings$.get('autoNavigateToPivot')) {
        //             await this._clickToolbar('pivotWidget');
        //         }
        //     }
        // };
        // this.osInstance.elements().viewport.addEventListener('scroll', autoToggle, { signal: this.autoTabController.signal });

        const autoButtons = (e) => {
            const vp = this.osInstance.elements().viewport;
            const pos = vp.scrollTop >= (0.5 * (vp.scrollHeight - vp.clientHeight));
            if (pos && (this._buttonPosition === 0)) {
                this._buttonPosition = 1;
                // console.log('moving buttons DOWN')
            } else if (!pos && (this._buttonPosition)) {
                this._buttonPosition = 0;
                // console.log('moving buttons UP')
            }

        };

        this.osInstance.elements().viewport.addEventListener('scroll', autoButtons, { signal: this.buttonController.signal });

    }

    setupJumpScrollerButton(){
        if (this.scroller) {
            this.addEventListener(this.osInstance.elements().viewport, 'scrollend', (e) => {
                const vp = this.osInstance.elements().viewport;
                vp.scrollTop >= (0.9 * (vp.scrollHeight - vp.clientHeight))
                    ?  this.scroller.classList.add('atBottom') : this.scroller.classList.remove('atBottom');
            });

            this.addEventListener(this.scroller, 'click', () => {

                const vp = this.osInstance.elements().viewport;
                const atBottom = vp.scrollTop >= (0.9 * (vp.scrollHeight - vp.clientHeight));
                const pos = atBottom ? 0 : this.osInstance.elements().viewport.scrollHeight;

                if (this.osInstance) {
                    this.osInstance?.elements().viewport.scrollTo({
                        top: pos,
                        behavior: 'smooth'
                    });
                } else {
                    window.scrollTo({
                        top: !pos ? pos : document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                }

            }, { passive: true });
        }
    }
    async _clickToolbar(tabName) {
        if (!this.widgetManager) return
        // if (this.yet_to_auto && this._killAutoToggle) this._killAutoToggle();

        try {
            await this.switchTab(tabName);
        } catch (error) {
            console.error(error);
            this.widgetManager.isSwitching = false;
        }
    }

    async send_full_push(event="", silent=false) {
        const roomContext = this._ptRaw.context;
        await this.socketManager()._sendWebSocketMessage({
            action: ACTION_MAP.get("push"),
            context: roomContext,
            options: { broadcast: false, log: false, silent:silent},
            data: {event: event}
        });
    }

    setupToolbarButtons() {
        if (this.minimizeButton) {
            this.addEventListener(this.minimizeButton, 'click', () => this._toggleMinimize());
        }
        if (this.toolbarButtons) {
            this.toolbarButtons.forEach(button => {
                this.addEventListener(button, 'click', async () => {
                    if (button.classList.contains('active')) return;
                    const tabName = button.dataset.name;
                    if (!tabName) return;
                    await this._clickToolbar(tabName);
                });
            });
        }
        const self = this;
        const sm = this.socketManager();
        this.addEventListener(this.pushBtn, 'click', async (e) => {
            await self.send_full_push("Manual Push", false)
        });

        const mm = this.modalManager();
        this.addEventListener(this.reloadBtn, 'click', async (e) => {

            const result = await mm.show({
                title: null,
                body: `<div class="refresh-modal-body">Refresh All Reference Markets?</div>`,
                fields: null,
                buttons: [
                    { text: 'Cancel', value: 'cancel' },
                    { text: 'Refresh', value: 'refresh', class: 'btn-primary'}
                ],
                modalBoxClass:'pt-refresh-btn-modal'
            });

            if (result !== 'refresh') return;

            const roomContext = self._ptRaw.context;
            await sm._sendWebSocketMessage({
                action: ACTION_MAP.get("refresh"),
                context: roomContext,
                options: { broadcast: false, log: false }
            });
        });
        this.addEventListener(this.metaEdit, 'click', () => {
            if (this.metaEditor) this.metaEditor.open()
        })

    }

    // =========================================================================
    // Micro-grid → derived flag columns
    // =========================================================================

    _setupMicroGridFlags() {
        const mgr = this._microGridManager;
        if (!mgr) return;

        /** Set of derived column names we've already registered. */
        this._microFlagCols = this._microFlagCols || new Set();

        const deps = ['ticker', 'isin', 'cusip', 'description'];

        /** Helper: register one derived flag column + AG Grid colDef (idempotent). */
        const ensureCol = (colName, headerName, group, infoText) => {
            if (this._microFlagCols.has(colName)) return;

            this.engine.registerDerivedColumn(colName, (ri, engine) => {
                const tests = this._microFlagIndex?.[colName];
                if (!tests || !tests.length) return false;
                for (const { col, test } of tests) {
                    const val = engine.getCell(ri, col);
                    if (val != null && test(val)) return true;
                }
                return false;
            }, { deps });

            this.ptGrid.addColumns([{
                field: colName,
                colId: colName,
                headerName,
                hide: true,
                context: {
                    dataType: 'flag',
                    abbreviation: colName.replace(/^is/, '').replace(/^flagged/, 'fl:'),
                    customColumnGroup: group,
                    isSearchable: false,
                    availableInApi: true,
                    allowAggregation: false,
                    allowGrouping: true,
                    hideFromMenu: false,
                    suppressColumnMenu: [],
                    showPin: false,
                    showLock: false,
                    isWarningFlag: true,
                    isReverseFlag: false,
                    showInfo: true,
                    infoText,
                },
            }]);

            this._microFlagCols.add(colName);
        };

        /** Convert a tag string to a safe column-name suffix: "AI" → "Ai", "foo:bar" → "FooBar" */
        const tagToColSuffix = (tag) => {
            return tag.replace(/[^a-zA-Z0-9]+/g, ' ').trim().split(/\s+/)
            .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join('');
        };

        /** Build a matcher function from a hot_tickers row. */
        const buildTest = (row) => {
            const mode = row.match_mode || 'literal';
            const pat = row.pattern;
            if (mode === 'regex') {
                try { const re = new RegExp(pat, 'i'); return (v) => re.test(v); }
                catch { return (v) => String(v).toUpperCase().includes(pat.toUpperCase()); }
            }
            const upper = pat.toUpperCase();
            return (v) => String(v).toUpperCase() === upper;
        };

        // Called on every snapshot/delta so new rules / tags appear dynamically.
        // Debounced via rAF so rapid-fire deltas (optimistic publish + server
        // echo, tag edits, bulk operations) coalesce into a single rebuild per
        // frame instead of freezing the UI.
        let rebuildRafId = 0;
        let pendingRows = null;

        const rebuildNow = (rows) => {
            if (!this.engine || !this.ptGrid) return;

            // ── Build per-column index: { colName → [{ col, test }] } ───────
            const index = {};
            const severities = ['low', 'med', 'high', 'other'];
            const seenTags = new Set();

            // Pre-init fixed columns
            index['isFlagged'] = [];
            for (const s of severities) index[`isFlagged${s.charAt(0).toUpperCase() + s.slice(1)}Severity`] = [];

            for (const row of rows) {
                const col = row.column;
                const pat = row.pattern;
                if (!col || !pat) continue;

                const test = buildTest(row);
                const entry = { col, test };

                // isFlagged — any match
                index['isFlagged'].push(entry);

                // severity bucket
                const sev = (row.severity || 'other').toLowerCase();
                const sevCol = `isFlagged${sev.charAt(0).toUpperCase() + sev.slice(1)}Severity`;
                if (!index[sevCol]) index[sevCol] = [];
                index[sevCol].push(entry);

                // per-tag columns
                const tags = row.tags;
                if (tags && typeof tags === 'string') {
                    for (let t of tags.split(',')) {
                        t = t.trim();
                        if (!t) continue;
                        seenTags.add(t);
                        const tagCol = `isFlagged${tagToColSuffix(t)}`;
                        if (!index[tagCol]) index[tagCol] = [];
                        index[tagCol].push(entry);
                    }
                }
            }

            this._microFlagIndex = index;

            // ── Ensure derived columns exist ────────────────────────────────
            ensureCol('isFlagged', 'Flagged', 'Bond/Flags/User', 'Bond matches any active rule.');

            for (const s of severities) {
                const colName = `isFlagged${s.charAt(0).toUpperCase() + s.slice(1)}Severity`;
                const label = s.charAt(0).toUpperCase() + s.slice(1);
                ensureCol(colName, `Flagged (${label})`, 'Bond/Flags/User', `Bond matches a ${s}-severity rule.`);
            }

            for (const tag of seenTags) {
                const suffix = tagToColSuffix(tag);
                const colName = `isFlagged${suffix}`;
                ensureCol(colName, `Flagged: ${tag}`, 'Bond/Flags/User/Tags', `Bond matches an rule tagged "${tag}".`);
            }

            // ── Invalidate caches & refresh ─────────────────────────────────
            const allCols = [...this._microFlagCols];
            for (const c of allCols) {
                if (this.engine._getterCache) this.engine._getterCache.delete(c);
            }
            try { this.ptGrid.api?.refreshCells?.({ columns: allCols, force: true }); } catch {}
        };

        const rebuild = (rows) => {
            pendingRows = rows;
            if (!rebuildRafId) {
                rebuildRafId = requestAnimationFrame(() => {
                    rebuildRafId = 0;
                    if (pendingRows) rebuildNow(pendingRows);
                    pendingRows = null;
                });
            }
        };

        // Fire on initial data (if already received) and on every future change
        const initial = mgr.getData('hot_tickers');
        if (initial.length) rebuildNow(initial);  // synchronous for first paint
        const unsub = mgr.onDataChange('hot_tickers', rebuild);
        this._microFlagDisposer = () => {
            unsub();
            if (rebuildRafId) { cancelAnimationFrame(rebuildRafId); rebuildRafId = 0; }
        };
    }

    setupGridTools(){
        if (this.quoteTypeButton) {
            this.quoteTypeButton.querySelectorAll('.quote-types').forEach(item => {
                this.addEventListener(item, 'mousedown', (e) => {
                    const newType = e.currentTarget.dataset.type;
                    if (newType) {
                        this.page$.set('activeQuoteType', newType);
                    }
                });
            });
        }
        if (this.quoteTypeButtonLower) {
            this.quoteTypeButtonLower.querySelectorAll('.quote-types').forEach(item => {
                this.addEventListener(item, 'mousedown', (e) => {
                    const newType = e.currentTarget.dataset.type;
                    if (newType) {
                        this.page$.set('activeQuoteType', newType);
                    }
                });
            });
        }
    }

    _setupListeners() {
        this.setupToolbarButtons();
        this.setupJumpScrollerButton();
        this.setupGridTools();
        this.setupStateBadgeDropdown();
        this.addEventListener(this.settingsButton, 'click', () => {
            this.context.settingsManager.showSettingsModal();
        })
        this.setupZindex()
        // this._setupClientModal();
    }

    // _setupClientModal() {
    //     if (!this._clientModal) {
    //         this._clientModal = ClientSummaryModal.spawn({
    //             useMockData: false,
    //             onFetchHistorical: async (clientData, activePortfolio) => {
    //                 // TODO: Replace with REST endpoint: GET /api/client/{clientName}/portfolios
    //                 return [];
    //             },
    //             onFetchSimilarity: async (activePortfolio) => {
    //                 // TODO: Replace with REST endpoint: GET /api/portfolio/{key}/similarity
    //                 return null;
    //             },
    //             onPortfolioSelect: (key) => {
    //                 console.log('[pt.js] ClientModal portfolio selected:', key);
    //             },
    //         });
    //     }
    //     if (this.headerNameElem) {
    //         this.headerNameElem.style.cursor = 'pointer';
    //         this.addEventListener(this.headerNameElem, 'click', () => {
    //             const meta = this.portfolioMeta$;
    //             const clientName = meta.get('client');
    //             const portfolioKey = this.context.portfolio_key;
    //             const metaObj = meta.asObject ? meta.asObject() : {};
    //             this._clientModal.openForClient(clientName, metaObj, portfolioKey);
    //         });
    //     }
    // }

    _raiseWidgetElement() {
        const top_two = this.topTwo;
        const grid = this.pricingGrid;
        if (top_two) top_two.style.zIndex = '2';
        if (grid) grid.style.zIndex = '0';
    }

    _lowerWidgetElement() {
        const top_two = this.topTwo;
        const grid = this.pricingGrid;
        if (top_two) top_two.style.zIndex = '0';
        if (grid) grid.style.zIndex = 'auto';
    }

    setupZindex() {
        const vc = document.getElementById('portfolio-quote-type');
        const vd = document.getElementById('portfolio-refmkt-wrapper');
        const raiseElem = this.raiseWidgetElement;
        const lowerElem = this.lowerWidgetElement;

        if (vc && this.context?.page?.addEventListener) {
            this.context.page.addEventListener(vc, 'mousedown', () => {
                raiseElem();
            });
            const vcb = vc.querySelector('*');
            if (vcb) this.context.page.addEventListener(vcb, 'blur', () => {
                lowerElem();
            });
        }
        if (vd && this.context?.page?.addEventListener) {
            this.context.page.addEventListener(vd, 'mousedown', () => {
                raiseElem();
            });
            const vdb = vd.querySelector('*');
            if (vdb) this.context.page.addEventListener(vdb, 'blur', () => {
                lowerElem();
            });
        }
    }

    setupStateBadgeDropdown() {
        if (this.stateEditorContainer && this.stateDisplay && this.stateDropdown) {

            const container = this.stateEditorContainer;
            const display = this.stateDisplay;
            const dropdown = this.stateDropdown;
            const toggleDropdown = (event) => { event.stopPropagation(); container.classList.toggle('open'); };
            this.context.page.addEventListener(display, 'click', toggleDropdown);

            this.context.page.addEventListener(display, 'keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleDropdown(e); } else if (e.key === 'Escape') { container.classList.remove('open'); } });

            this.context.page.addEventListener(dropdown, 'click', async (event) => {
                let meta, data;
                const targetOption = event.target.closest('.pt-status-option-wrapper');
                if (targetOption) {
                    const newState = targetOption.dataset.value;
                    if (newState !== this.portfolioMeta$.get('state')) {

                        this.portfolioMeta$.set('state', newState);
                        const metaData = this?._metaRaw?.context || {};
                        const pks = this.portfolioMeta$.pick(this._metaHyper.getPrimaryKey()).asObject()
                        const d =  {...pks, 'state': newState};
                        try {
                            meta = this.subscriptionManager().buildPayloadContext(this.context.metaRoom, metaData);
                            data = this.subscriptionManager().buildPayloadData(this.context.metaRoom, metaData, d);
                            await this.subscriptionManager().publishMessage(this.context.metaRoom, meta, data);
                        } catch(e) {
                            console.error('error publishing meta message', e, this.context.metaRoom, meta, data)
                        }
                    }
                    container.classList.remove('open');
                }
            });

            // Close dropdown on outside click (handler defined in constructor/show)
            this._clickOutsideHandler = (event) => {
                if (container && !container.contains(event.target)) {
                    container.classList.remove('open');
                }
            };
            this.context.page.addEventListener(document, 'click', this._clickOutsideHandler);
        }

        this._subs.push(this.theme$.onChanges(ch => {
            this.theme = this.theme$.get('theme');
            this._refreshDropdownColors();
            const state = this.portfolioMeta$.get('state');
            if (state) {
                this._metaUpdateHeader(state, false);
            }
        }))
    }

    /**
     * Return the physical data column names required for a given market.
     * These are the columns that derived market columns (BidPx/AskPx/MidPx/Spd/Yield/Level)
     * read from the engine via getCell.
     */
    _marketDataColumns(market) {
        const suffixes = [
            'BidPx', 'AskPx', 'MidPx',
            'BidSpd', 'AskSpd', 'MidSpd',
            'BidYld', 'AskYld', 'MidYld',
            'BidMmy', 'AskMmy', 'MidMmy',
            'BidDm', 'AskDm', 'MidDm'
        ];
        return suffixes.map(s => `${market}${s}`);
    }

    /**
     * Ensure that the data columns for the given market(s) are loaded.
     * No-op when lazy loading is disabled.
     */
    async _ensureMarketColumns(markets) {
        // if (!this.engine?.ensureColumns) return;
        // const cols = [];
        // const needed = Array.isArray(markets) ? markets : [markets];
        // const allFields = this.engine.allFieldSet?.() || new Set(this.engine.allFields());
        // for (const mkt of needed) {
        //     for (const c of this._marketDataColumns(mkt)) {
        //         if (allFields.has(c)) cols.push(c);
        //     }
        // }
        // if (cols.length) await this.engine.ensureColumns(cols);
    }

    _quoteTypeDisplay(val) {
        const map = {
            client: 'Client',
            convention: 'Convention',
            price: 'Price',
            spread: 'Spread',
            mmy: 'MMY',
            dm: 'DM',
            cash: 'Clean',
            dirty_cash: 'Dirty'
        };
        return map[val] || val; // Return input value if not found in map
    }

    _toggleMinimize(force=null) {
        try {
            this.isMinimized = force || !this.isMinimized;
            const headerHeight = this.headerRow.offsetHeight;

            if (!this.isMinimized) {
                requestAnimationFrame(() => {
                    document.documentElement.style.setProperty('--header-height', `-${headerHeight}px`);
                    this.toolbarTop.style.overflow = 'hidden';
                    this.toolbar.classList.toggle('minimized', this.isMinimized);
                    this.minimizeButton.classList.toggle('active', this.isMinimized);
                    this.topTwo.classList.toggle('minimized', this.isMinimized);

                    requestAnimationFrame(() => {
                        this.contentRow.classList.toggle('minimized', this.isMinimized);
                    });
                    // setTimeout(()=> {
                    //     window.dispatchEvent(new Event('resize'));
                    //     if (this.toolbarTop) this.toolbarTop.style.overflow = '';
                    // }, 0)
                });
            } else {
                requestAnimationFrame(() => {
                    this.contentRow.classList.toggle('minimized', this.isMinimized);
                    requestAnimationFrame(() => {
                        document.documentElement.style.setProperty('--header-height', `-${headerHeight}px`);
                        this.toolbarTop.style.overflow = 'hidden';
                        this.toolbar.classList.toggle('minimized', this.isMinimized);
                        this.minimizeButton.classList.toggle('active', this.isMinimized);
                        this.topTwo.classList.toggle('minimized', this.isMinimized);
                    });
                });
            }

            // setTimeout(() => {
            //     window.dispatchEvent(new Event('resize'));
            //     if (this.toolbarTop) this.toolbarTop.style.overflow = '';
            // }, 500);

        } catch (error) {
            console.error('Minimize transition error:', error);
        }
    }

    async switchTab(targetWidgetKey) {

        const button = document.querySelector(`[data-name='${targetWidgetKey}']`);
        if (button) {
            document.querySelector('.toolbar-button.active')?.classList.remove('active');
            button.classList.add('active');
        }

        if (this.widgetManager.isSwitching) {
            console.error('Cannot swtich, is switching')
            return;
        }

        const switchResult = await this.widgetManager.switch(targetWidgetKey);
        if (!switchResult) {
            console.error('Switch was aborted or failed')
            return;
        } // Switch was aborted or failed

        const { currentContainer, targetContainer } = switchResult;
        const duration = 200;

        if (currentContainer) {
            currentContainer.style.transition = `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`;
            currentContainer.classList.add('slide-out');
            currentContainer.classList.remove('active');
        }

        if (targetContainer) {
            targetContainer.style.transition = `opacity ${duration}ms ease-in, transform ${duration}ms ease-in`;
            targetContainer.classList.add('active', 'slide-in');
        }

        setTimeout(() => {
            if (currentContainer) {
                currentContainer.classList.remove('slide-out');
                currentContainer.style.transition = '';
            }
            if (targetContainer) {
                targetContainer.classList.remove('slide-in');
                targetContainer.style.transition = '';
            }
            window.dispatchEvent(new Event('resize'));
            this.widgetManager.setSwitching(false);
        }, duration);

    }

    _setupGridSearch() {
        // let index = 0;
        // this.searchInput.placeholder = '';
        // const si = this.searchInput;
        // const textToType = 'Search this portfolio...';
        // function typeNextCharacter() {
        //     if (index < textToType.length) {
        //         si.placeholder += textToType[index];
        //         index++;
        //         setTimeout(typeNextCharacter, 75)
        //     }
        // }
        // setTimeout(()=> typeNextCharacter(), 750); // Delayed start

        this.searchInput.placeholder = 'Search this portfolio...'

        this.addEventListener(this.gridSearchBar, 'click', () => {
            this.searchInput.focus();
        });

        this._subs.push(this.page$.onValueChanged("quickSearch", (cur, prev) => {
            if (cur) {
                this.searchElement.classList.add('active');
            } else {
                this.searchElement.classList.remove('active');
                this.searchInput.value = '';
            }
        }));

        this.addEventListener(this.searchInput, 'input', () => {
            this.page$.set('quickSearch', this.searchInput?.value?.trim() || '');
        })

        this.addEventListener(this.searchElement, 'click', (e) => {
            this.searchInput.focus();
        });

        this.addEventListener(this.searchIcon, 'click', (e) => {
            this.page$.set('quickSearch', null);
        })
    }

    setQuickSearch(v) {
        const vv = v || '';
        this.page$.set('quickSearch', vv);
        this.searchInput.value = vv;
    }

}

export default () => PortfolioPage;

