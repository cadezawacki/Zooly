import Fuse from 'fuse.js';
import {debounce} from '@/utils/helpers.js';
import {writeObjectToClipboard, writeStringToClipboard} from "@/utils/clipboardHelpers.js";

import 'jstree/dist/themes/default/style.min.css';
import '../css/jstreeTheme.css';
import {ACTION_MAP} from "@/global/actionMap.js";

const GRID_EVENT_DEBOUNCE_MS = 200;

export class TreeColumnChooser {

    constructor(config = {}) {

        // Core properties
        this.context = null;
        this.adapter = null;
        this.api = null;
        this.gridName = null;
        this.globalPresets = [];

        this.columnDefs = [];
        this.originalTreeData = [];
        this.flattenedNodes = [];

        // Search properties
        this.searchTerm = '';
        this.searchCache = new Map();
        this.nodeTextCache = new Map();
        this.currentSearchPromise = null;
        this.fuse = null;
        this.searchMatches = new Map(); // Store match positions for highlighting

        // UI state
        this.virtualizer = null;
        this.expandedNodes = new Set();
        this.selectedNodes = new Set();
        this.indeterminateNodes = new Set();
        this.hoveredNodeId = null;
        this.focusedNodeIndex = -1;
        this.modalOpen = false;

        this._delegatedHandlers = {};
        this._gridListeners = new Map();
        this._timeouts = new Set();
        this._debouncers = new Set();
        this._onSearchInput = null;
        this._onSearchToggleClearBtn = null;
        this._onClearClick = null;

        this._lastFuseResults = null;
        this._fuseRank = new Map();
        this._is_applying = false;
        this._filterSelected = false;
        this._sortedColumns = new Map();

        // Configuration
        this.config = {
            autoFocusSearch: false,
            enableKeyboardNav: false,
            enableDragDrop: false,
            enableCustomNames: true,
            animationDuration: 200,
            debounceDelay: 200,
            gridEvents: [
                'columnVisible',
                'columnPinned',
                'columnMoved',
                //'sortChanged',
                'filterChanged',
                'columnResized',
            ],
            enableFilterMemory:true,
            enableSortMemory: true,
            ...config
        };

        this.controller = new AbortController();
        const { signal } = this.controller;
        this.signal = signal;

        // Custom names storage
        this.customNames = {};

        // Column state tracking
        this.pinnedColumns = new Set();
        this.lockedColumns = new Set();

        // State Management
        this.presets = new Map();
        this.customColumns = new Map();
        this.activePresetName = null;
        this.hasUnsavedChanges = false;
        this.isLoadingState = false;

        // Initialization flags
        this.initialized = false;
        this.domReady = false;
        this.pendingInit = null;

        // Performance optimizations
        this.rafId = null;
        this.renderQueue = [];

        // Modal and utilities
        this.modalManager = null;
        // this.bindCoreEvents();

        this.simpleUndo = null;

        // Undo stack for reverting column selection actions
        this._undoStack = [];
        this._undoMaxSize = 50;
    }

    // ==================== Initialization ====================

    async init(params) {
        this.context = params.context;
        this.adapter = params.adapter;
        this.engine = this.adapter.engine;
        this.api = params.api;
        this.outsideSaveBtnSelector = params.outsideSaveBtnSelector;
        this.outsideLoadBtnSelector = params.outsideLoadBtnSelector;
        this.outsideCreateBtnSelector = params.outsideCreateBtnSelector;
        this.outsideReloadBtnSelector = params.outsideReloadBtnSelector;
        this.config = params?.config ? {...this.config, ...params?.config} : this.config;

        this.grid$ = this.adapter.grid$;
        this.gridName = params.gridName || 'grid';
        this.toolbarId = params?.toolbarId || 'treeColumnChooser';
        this.globalPresets = params.globalPresets || [];

        this.modalManager = this.context.page.modalManager();

        this.showPin = params?.config?.showPin ?? true;
        this.showLock = false; //params?.config?.showLock ?? true;
        this.showInfo = true; //false; //params?.config?.showInfo ?? true;

        this.customNames = this.loadCustomNames();
        if (Object.keys(this.customNames).length > 0) {
            this.applyCustomNamesToGrid();
        }

        // Store initialization params for deferred setup
        this.initParams = params;
        this.api.treeService = this;

        // Create DOM elements
        this.createElements();
        return this.mainContainer;
    }

    _getGlobalState() {
        return 'Default View'
    }

    isDomReady() {
        if (!this.mainContainer) return false;
        const rect = this.mainContainer.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    completeInitialization() {
        if (this.initialized) return;

        this.columnDefs = this.getColumnDefs();
        this.fields = this.getFields();

        if (!this.columnDefs || this.columnDefs.length === 0) {
            this.showEmptyState();
            return;
        }

        // Setup tree (this will apply custom names to tree nodes)
        this.setupTree();
        this.attachEventListeners();
        this.setupGridChangeListeners();
        this.initializeSelectionStates();

        if (this.config.autoFocusSearch && this.searchInput) {
            this.searchInput.focus();
        }

        this.initialized = true;
        this.domReady = true;
        this.buildFuseIndex();
        this._bindDelegatedEvents();
        this.renderAllNodes();
    }

    // ==================== DOM Creation ====================

    createElements() {
        this.mainContainer = document.createElement('div');
        this.mainContainer.className = 'tree-column-chooser';
        this.mainContainer.style.cssText = `
            height: 100%;
            width: 100%;
            display: flex;
            flex-direction: column;
            position: relative;
        `;

        // Inject drag indicator styles
        if (!document.getElementById('tree-drag-styles')) {
            const style = document.createElement('style');
            style.id = 'tree-drag-styles';
            style.textContent = `
                .tree-node.drag-over-top, .virtual-tree-node.drag-over-top {
                    border-top: 2px solid var(--ag-range-selection-border-color, #2196F3) !important;
                }
                .tree-node.drag-over-bottom, .virtual-tree-node.drag-over-bottom {
                    border-bottom: 2px solid var(--ag-range-selection-border-color, #2196F3) !important;
                }
                .drag-handle:hover { opacity: 0.8 !important; }
                .tree-node[draggable="true"] { transition: opacity 0.15s; }
            `;
            document.head.appendChild(style);
        }

        this.createHeader();
        this.createScrollContainer();
        // this.createFooter();
    }


    createHeader() {
        this.headerElement = document.createElement('div');
        this.headerElement.className = 'tree-header';

        const searchContainer = document.createElement('div');
        searchContainer.className = 'search-container';
        searchContainer.style.cssText = `position: relative; margin-bottom: 8px;`;

        this.searchInput = document.createElement('input');
        this.searchInput.type = 'text';
        this.searchInput.placeholder = 'Search columns...';
        this.searchInput.className = 'tree-search-input';
        this.searchInput.style.cssText = `
            width: 100%;
            padding: 6px 30px 6px 8px;
            border: 1px solid var(--ag-border-color, #ddd);
            border-radius: 4px;
            outline: none;
            font-size: 14px;
        `;
        this.searchInput.setAttribute('autocomplete', 'off');

        this.clearButton = document.createElement('button');
        this.clearButton.className = 'clear-search-btn';
        this.clearButton.innerHTML = '×';
        this.clearButton.style.cssText = `
            position: absolute; right: 8px; top: 50%;
            transform: translateY(-50%); background: none; border: none;
            font-size: 20px; cursor: pointer; color: var(--ag-secondary-foreground-color, #666);
            display: none; padding: 0 4px; line-height: 1;
        `;
        this.clearButton.title = 'Clear search';

        searchContainer.appendChild(this.searchInput);
        searchContainer.appendChild(this.clearButton);

        const controlWrapper = document.createElement('div');
        controlWrapper.className = 'tree-controls-wrapper';
        const controls = document.createElement('div');
        controls.className = 'tree-controls';
        controls.style.cssText = `display: flex; gap: 4px; flex-wrap: wrap;`;

        const saveIcon = `<div id="saveStateWrapper"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M5 21q-.825 0-1.412-.587T3 19V5q0-.825.588-1.412T5 3h11.175q.4 0 .763.15t.637.425l2.85 2.85q.275.275.425.638t.15.762V19q0 .825-.587 1.413T19 21zM19 7.85L16.15 5H5v14h14zM12 18q1.25 0 2.125-.875T15 15t-.875-2.125T12 12t-2.125.875T9 15t.875 2.125T12 18m-5-8h7q.425 0 .713-.288T15 9V7q0-.425-.288-.712T14 6H7q-.425 0-.712.288T6 7v2q0 .425.288.713T7 10M5 7.85V19V5z"/></svg></div>`;

        const createIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 32 32"><path fill="currentColor" d="M30 24h-4v-4h-2v4h-4v2h4v4h2v-4h4z"/><path fill="currentColor" d="M16 28H8V4h8v6a2.006 2.006 0 0 0 2 2h6v4h2v-6a.91.91 0 0 0-.3-.7l-7-7A.9.9 0 0 0 18 2H8a2.006 2.006 0 0 0-2 2v24a2.006 2.006 0 0 0 2 2h8Zm2-23.6l5.6 5.6H18Z"/></svg>`
        const loadIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V6h5.17l2 2H20v10z"/></svg>`;
        const reloadIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 32 32"><path fill="currentColor" d="M18 28A12 12 0 1 0 6 16v6.2l-3.6-3.6L1 20l6 6l6-6l-1.4-1.4L8 22.2V16a10 10 0 1 1 10 10Z"/></svg>`;
        // const addColumnIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>`;
        const deselectIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="m19.775 22.6l-5.6-5.6H7V9.825l-5.6-5.6L2.8 2.8l18.4 18.4zM9 15h3.175L9 11.825zm8-.825l-2-2V9h-3.175l-2-2H17zM5 19v2q-.825 0-1.412-.587T3 19zm-2-2v-2h2v2zm0-4v-2h2v2zm0-4V7h2v2zm4 12v-2h2v2zM7 5V3h2v2zm4 16v-2h2v2zm0-16V3h2v2zm4 16v-2h2v2zm0-16V3h2v2zm4 12v-2h2v2zm0-4v-2h2v2zm0-4V7h2v2zm0-4V3q.825 0 1.413.588T21 5z"/></svg>`;

        this.nonSelectedIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M8 6a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-8a2 2 0 0 1-2-2z"/><path d="M16 16v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2"/></g></svg>`

        this.selectedIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="m8 10.5l6.492-6.492M13.496 16L20 9.496zm-4.91-.586L19.413 4.587M8 6a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-8a2 2 0 0 1-2-2z"/><path d="M16 16v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2"/></g></svg>`

        const signal = this.signal;
        this.saveBtn = this.createControlButton(saveIcon, 'Save View', () => this._handleSave());
        this.outsideSaveBtn = this.outsideSaveBtnSelector ? document.querySelector(this.outsideSaveBtnSelector) : null;
        if (this.outsideSaveBtn) this.outsideSaveBtn.addEventListener('click', ()=> this.saveBtn.click(), {signal});

        this.createBtn = this.createControlButton(createIcon, 'Create View', () => this._handleCreate());
        this.outsideCreateBtn = this.outsideCreateBtnSelector ? document.querySelector(this.outsideCreateBtnSelector) : null;
        if (this.outsideCreateBtn) this.outsideCreateBtn.addEventListener('click', ()=> this.createBtn.click(), {signal});

        this.loadBtn = this.createControlButton(loadIcon, 'Load View', () => this.showLoadDialog());
        this.outsideLoadBtn = this.outsideLoadBtnSelector ? document.querySelector(this.outsideLoadBtnSelector) : null;
        if (this.outsideLoadBtn) this.outsideLoadBtn.addEventListener('click', ()=> this.loadBtn.click(), {signal});

        this.outsidReloadBtn = this.outsideReloadBtnSelector ? document.querySelector(this.outsideReloadBtnSelector) : null;
        if (this.outsidReloadBtn) this.outsidReloadBtn.addEventListener('click', ()=> this._handleReload(), {signal});

        this.reloadBtn = this.createControlButton(reloadIcon, 'Reload View', () => this._handleReload());
        this.filterSelectedBtn = this.createControlButton(this.nonSelectedIcon, 'Filter Selected', () => this._handleFilterSelected());

        const undoIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M7 19v-2h7.1q1.575 0 2.737-1T18 13.5T16.838 11T14.1 10H7.8l2.6 2.6L9 14L4 9l5-5l1.4 1.4L7.8 8h6.3q2.425 0 4.163 1.575T20 13.5t-1.737 3.925T14.1 19z"/></svg>`;
        this.undoBtn = this.createControlButton(undoIcon, 'Undo Last Action', () => this._handleUndo());
        this.undoBtn.disabled = true;

        const exportIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M6 20q-.825 0-1.412-.587T4 18v-3h2v3h12v-3h2v3q0 .825-.587 1.413T18 20zm6-4l-5-5l1.4-1.4l2.6 2.6V4h2v8.2l2.6-2.6L17 11z"/></svg>`;
        this.exportBtn = this.createControlButton(exportIcon, 'Export / Import Config', () => this._showExportImportModal());

        // this.deselectBtn = this.createControlButton(deselectIcon, 'Deselect All', () => this.deselectAll());
        // this.addColumnBtn = this.createControlButton(addColumnIcon, 'Create Column', () => this._handleNewColumn(), 'create-tree-column');

        controls.appendChild(this.saveBtn);
        controls.appendChild(this.createBtn);
        controls.appendChild(this.loadBtn);
        controls.appendChild(this.reloadBtn);
        searchContainer.appendChild(this.filterSelectedBtn);
        controls.appendChild(this.undoBtn);
        controls.appendChild(this.exportBtn);
        // controls.appendChild(this.deselectBtn);
        // controls.appendChild(this.addColumnBtn);

        controlWrapper.appendChild(controls);
        this.headerElement.appendChild(searchContainer);
        this.headerElement.appendChild(controlWrapper);
        this.mainContainer.appendChild(this.headerElement);

        this._updateButtonStates();
    }

    createControlButton(icon, title, onClick, id=null) {
        const btn = document.createElement('button');
        btn.className = 'tree-control-btn tooltip tooltip-top';
        btn.innerHTML = icon;
        btn.setAttribute('data-tooltip', title);
        if (id) btn.id = id;
        btn.style.cssText = `
            padding: 4px 8px;
            border: 1px solid var(--ag-border-color, #ddd);
            background: var(--ag-background-color, #fff);
            cursor: pointer;
            border-radius: 3px;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: visible;
        `;
        btn.addEventListener('click', onClick);
        return btn;
    }

    createScrollContainer() {
        // this.scrollContainer = document.createElement('div');
        // this.scrollContainer.className = 'tree-scroll-container';
        // this.scrollContainer.style.cssText = `
        //     flex: 1;
        //     overflow: auto;
        //     min-height: 0;
        //     position: relative;
        // `;
        // this.scrollContainer.setAttribute('tabindex', '0');

        this.virtualContainer = document.createElement('div');
        this.virtualContainer.className = 'tree-virtual-container';
        this.virtualContainer.style.cssText = `
            position: relative;
            width: 100%;
        `;

        // this.scrollContainer.appendChild(this.virtualContainer);
        this.mainContainer.appendChild(this.virtualContainer);
    }

    createFooter() {
        this.footerElement = document.createElement('div');
        this.footerElement.className = 'tree-footer';

        this.statusText = document.createElement('span');
        this.updateStatus();

        this.footerElement.appendChild(this.statusText);
        this.mainContainer.appendChild(this.footerElement);
    }

    // ==================== Event Handling ====================

    /**
     * Bind core event handlers
     */


    attachEventListeners() {
        // Search input → debounced search
        this._onSearchInput = debounce(async () => {
            await this.handleSearch();
            this.focusedNodeIndex = -1;
        }, this.config.debounceDelay);
        this.searchInput.addEventListener('input', this._onSearchInput);

        // Show/hide clear button
        this._onSearchToggleClearBtn = () => {
            this.clearButton.style.display = this.searchInput.value ? 'block' : 'none';
        };
        this.searchInput.addEventListener('input', this._onSearchToggleClearBtn);

        // Clear button
        this._onClearClick = () => this.clearSearch();
        this.clearButton.addEventListener('click', this._onClearClick);
    }


    // ==================== Keyboard Navigation ====================

    navigateUp() {
        if (this.focusedNodeIndex > 0) {
            this.focusedNodeIndex--;
            this.focusNode(this.focusedNodeIndex);
        }
    }

    /**
     * Navigate to next node
     */
    navigateDown() {
        if (this.focusedNodeIndex < this.flattenedNodes.length - 1) {
            this.focusedNodeIndex++;
            this.focusNode(this.focusedNodeIndex);
        }
    }

    /**
     * Navigate to first node
     */
    navigateToFirst() {
        this.focusedNodeIndex = 0;
        this.focusNode(this.focusedNodeIndex);
    }

    /**
     * Navigate to last node
     */
    navigateToLast() {
        this.focusedNodeIndex = this.flattenedNodes.length - 1;
        this.focusNode(this.focusedNodeIndex);
    }

    /**
     * Focus a node by index
     */
    focusNode(index) {
        if (index < 0 || index >= this.flattenedNodes.length) return;

        const node = this.flattenedNodes[index];
        const nodeElements = this.virtualContainer.querySelectorAll('.tree-node');
        const element = nodeElements[index];

        if (element) {
            // Remove previous focus
            nodeElements.forEach(el => el.classList.remove('focused'));

            // Add focus to current
            element.classList.add('focused');

            // Scroll into view
            element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    expandCurrentNode() {
        const idx = this.focusedNodeIndex === -1 ? 0 : this.focusedNodeIndex
        const node = this.flattenedNodes[idx];
        if (node && node.isGroup && !this.expandedNodes.has(node.id)) {
            this.toggleNode(node.id);
        }
    }

    collapseCurrentNode() {
        const idx = this.focusedNodeIndex === -1 ? 0 : this.focusedNodeIndex
        const node = this.flattenedNodes[idx];
        if (node && node.isGroup && this.expandedNodes.has(node.id)) {
            this.toggleNode(node.id);
        }
    }

    toggleCurrentNode() {
        const idx = this.focusedNodeIndex === -1 ? 0 : this.focusedNodeIndex
        const node = this.flattenedNodes[idx];
        if (!node) return;

        if (node.isGroup) {
            this.toggleNode(node.id);
        } else {
            this.toggleCurrentNodeSelection();
        }
    }

    toggleCurrentNodeSelection() {
        const idx = this.focusedNodeIndex === -1 ? 0 : this.focusedNodeIndex
        const node = this.flattenedNodes[idx];
        if (!node) return;

        const isSelected = this.selectedNodes.has(node.id);
        this.handleNodeSelection(node, !isSelected);
    }

    selectNodeAndChildren() {
        const node = this.flattenedNodes[this.focusedNodeIndex];
        if (!node) return;

        this.handleNodeSelection(node, true);
        if (node.isGroup && node.children) {
            this.selectAllChildren(node);
        }
    }

    setHoveredNode(nodeId) {
        this.hoveredNodeId = nodeId;

        // Find index for keyboard navigation sync
        if (nodeId) {
            const index = this.flattenedNodes.findIndex(n => n.id === nodeId);
            if (index !== -1) {
                this.focusedNodeIndex = index;
            }
        }

        // Update visual state
        this.renderVirtualItems();
    }
    // ==================== Col Management ====================
    _getRequiredSet() {
        if (!this._requiredColumns) {
            const key = `treeColumnChooser_required_${this.gridName || 'grid'}`;
            let saved = [];
            try { saved = JSON.parse(localStorage.getItem(key) || '[]'); } catch {}
            this._requiredColumns = new Set(Array.isArray(saved) ? saved : []);
        }
        return this._requiredColumns;
    }

    _persistRequiredSet() {
        try {
            localStorage.setItem(
                `treeColumnChooser_required_${this.gridName || 'grid'}`,
                JSON.stringify(Array.from(this._getRequiredSet()))
            );
        } catch {}
    }

    getRequiredColumns() {
        return Array.from(this._getRequiredSet());
    }

    setRequiredColumns(cols = []) {
        const req = this._getRequiredSet();
        const toAdd = Array.isArray(cols) ? cols : [cols];
        let changed = false;

        toAdd.forEach(id => {
            if (id && !req.has(id)) { req.add(id); changed = true; }
        });
        if (!changed) return;

        this._persistRequiredSet();

        // Ensure they are visible in grid and selected in tree
        const showable = toAdd.filter(id => this.api?.getColumn(id) !== null);
        if (showable.length && this.api) this.api.setColumnsVisible(showable, true);

        showable.forEach(id => {
            this.selectedNodes.add(id);
            this.indeterminateNodes.delete(id);
        });

        this.updateAllParentStates?.();
        this.renderAllNodes?.();
    }

    removeRequiredColumns(cols = []) {
        const req = this._getRequiredSet();
        const toRemove = Array.isArray(cols) ? cols : [cols];
        let changed = false;

        toRemove.forEach(id => { if (req.delete(id)) changed = true; });
        if (!changed) return;

        this._persistRequiredSet();

        // Do not auto-hide removed columns; just allow toggling again.
        this.updateAllParentStates?.();
        this.renderAllNodes?.();
    }

    /* Enforce required columns inside tree selection */
    _enforceRequiredSelections({ updateGrid = false } = {}) {
        const req = this._getRequiredSet();
        if (!req || req.size === 0) return;

        // Only leaf columns, skip groups
        const mustHave = Array.from(req).filter(id => {
            const n = this.findNodeById?.(id);
            return n && !n.isGroup;
        });

        mustHave.forEach(id => {
            this.selectedNodes.add(id);
            this.indeterminateNodes.delete(id);
        });

        if (updateGrid && this.api) {
            const present = mustHave.filter(id => this.api.getColumn(id) !== null);
            if (present.length) this.api.setColumnsVisible(present, true);
        }

        this.updateAllParentStates?.();
    }

    // ==================== State Management ====================

    _getGridState() {
        // Prefer adapter if present, but fall back to stable direct read from grid.
        if (this.adapter?.getGridState) {
            try { return this.adapter.getGridState(); } catch (_) {}
        }
        if (!this.api) return null;

        // Build normalized state with explicit pin info and stable order
        const raw = this.api.getColumnState ? this.api.getColumnState() : [];
        const state = [];
        const pinnedColumns = { left: [], right: [] };

        // Map visible order: the order of raw is the current order when applyOrder is supported
        for (let i = 0, n = raw.length; i < n; i++) {
            const c = raw[i];
            const colId = c.colId;
            if (!colId) continue;

            // Normalize booleans and pin
            const hide = !!c.hide;
            const pinned = c.pinned === 'left' ? 'left' : (c.pinned === 'right' ? 'right' : null);
            if (pinned === 'left') pinnedColumns.left.push(colId);
            else if (pinned === 'right') pinnedColumns.right.push(colId);

            state.push({
                colId,
                hide,
                pinned,
                sort: c.sort || null,
                sortIndex: typeof c.sortIndex === 'number' ? c.sortIndex : null,
                width: typeof c.width === 'number' ? c.width : null,
                aggFunc: c.aggFunc || null,
                rowGroup: !!c.rowGroup,
                rowGroupIndex: typeof c.rowGroupIndex === 'number' ? c.rowGroupIndex : null,
                pivot: !!c.pivot,
                pivotIndex: typeof c.pivotIndex === 'number' ? c.pivotIndex : null
            });
        }

        // Include models if available
        let sortModel = null, filterModel = null;
        try { sortModel = this.api.getSortModel ? this.api.getSortModel() : null; } catch (_) {}
        try { filterModel = this.api.getFilterModel ? this.api.getFilterModel() : null; } catch (_) {}

        return {
            columnState: state,
            pinnedColumns,
            sortModel,
            filterModel,
            version: 1
        };
    }

    async _applyGridState(viewState) {
        if (!this.api || !viewState) return;
        if (this._is_applying) return
        this._is_applying = true;

        // Guard: avoid re-entrancy and event storms
        this.isLoadingState = true;
        this._removeGridChangeListeners();

        // Suppress auto-size during state application so AG Grid doesn't
        // clobber the restored widths with its fitCellContents / auto-size pass.
        const prevAutoSize = this.api.getGridOption?.('autoSizeStrategy');
        const prevSuppressAuto = this.api.getGridOption?.('suppressAutoSize');
        try { this.api.setGridOption('suppressAutoSize', true); } catch {}
        try { this.api.setGridOption('autoSizeStrategy', undefined); } catch {}

        const allColIds = (() => {
            // Prefer columnDefs known to tree; fall back to grid if needed
            const out = [];
            if (this.columnDefs && this.columnDefs.length) {
                for (let i = 0; i < this.columnDefs.length; i++) {
                    const def = this.columnDefs[i];
                    const id = def?.colId || def?.field;
                    if (id) out.push(id);
                }
                return out;
            }
            try {
                const cols = this.api.getColumns ? this.api.getColumns() : (this.api.getAllGridColumns ? this.api.getAllGridColumns() : null);
                if (!cols) return out;
                for (let i = 0; i < cols.length; i++) {
                    const id = cols[i]?.getId ? cols[i].getId() : (cols[i]?.colId || cols[i]?.id);
                    if (id) out.push(id);
                }
            } catch (_) {}
            return out;
        })();

        const validSet = new Set(allColIds);

        // Normalize incoming state
        const pinnedLeftSet  = new Set((viewState.pinnedColumns && Array.isArray(viewState.pinnedColumns.left))  ? viewState.pinnedColumns.left  : []);
        const pinnedRightSet = new Set((viewState.pinnedColumns && Array.isArray(viewState.pinnedColumns.right)) ? viewState.pinnedColumns.right : []);
        const lockedRightSet = new Set(Array.isArray(viewState.lockedColumns) ? viewState.lockedColumns : []);

        // Sanitize columnState (drop unknown columns; coerce hide/pin)
        const rawArray = Array.isArray(viewState.columnState) ? viewState.columnState : [];
        const sanitized = [];
        for (let i = 0, n = rawArray.length; i < n; i++) {
            const s = rawArray[i] || {};
            const colId = s.colId;
            if (!colId || !validSet.has(colId)) continue;

            const isPinnedLeft  = pinnedLeftSet.has(colId);
            const isPinnedRight = pinnedRightSet.has(colId) || lockedRightSet.has(colId);
            const pinned = isPinnedLeft ? 'left' : (isPinnedRight ? 'right' : (s.pinned === 'left' || s.pinned === 'right' ? s.pinned : null));

            sanitized.push({
                colId,
                hide: !!s.hide,
                pinned,
                sort: s.sort || null,
                sortIndex: (typeof s.sortIndex === 'number' ? s.sortIndex : null),
                width: (typeof s.width === 'number' ? s.width : null),
                minWidth: (typeof s.minWidth === 'number' ? s.minWidth : null),
                maxWidth: (typeof s.maxWidth === 'number' ? s.maxWidth : null),
                aggFunc: s.aggFunc || null,
                rowGroup: !!s.rowGroup,
                rowGroupIndex: (typeof s.rowGroupIndex === 'number' ? s.rowGroupIndex : null),
                pivot: !!s.pivot,
                pivotIndex: (typeof s.pivotIndex === 'number' ? s.pivotIndex : null)
            });
        }

        // Partition into left / center / right for deterministic ordering
        const left = []; const center = []; const right = [];
        for (let i = 0, n = sanitized.length; i < n; i++) {
            const s = sanitized[i];
            if (s.hide === true) continue; // hidden columns do not participate in display order
            if (s.pinned === 'left') left.push(s);
            else if (s.pinned === 'right') right.push(s);
            else center.push(s);
        }

        // Order as left -> center -> right in the sequence of incoming list
        const orderedVisible = left.concat(center, right);

        // We apply in one shot with a strict defaultState to purge stale pins/visibility.
        const defaultState = {
            hide: true,
            pinned: null,
            sort: null,
            sortIndex: null,
            rowGroup: false,
            rowGroupIndex: null,
            pivot: false,
            pivotIndex: null,
            aggFunc: null
        };

        // Build final state array: visible (ordered) first, then any hidden with explicit hide flags
        const stateArray = orderedVisible.slice(); // visible first, maintains order

        // Append hidden entries as needed to carry width/sort/etc if provided while keeping them hidden.
        for (let i = 0, n = sanitized.length; i < n; i++) {
            const s = sanitized[i];
            if (s.hide === true) stateArray.push(s);
        }

        // If the view carries explicit widths, permanently disable the
        // autoSizeStrategy (e.g. fitCellContents) so that subsequent data
        // refreshes do not shrink columns back to their min-width.
        const hasExplicitWidths = sanitized.some(s => typeof s.width === 'number' && s.width > 0);

        // Apply column state deterministically
        try {
            this.api.applyColumnState({
                state: stateArray,
                defaultState,
                applyOrder: true
            });

            // Reinforce explicit widths in a single batch so they beat any
            // residual auto-size pass from AG Grid.
            if (hasExplicitWidths) {
                const widthBatch = [];
                for (let i = 0; i < sanitized.length; i++) {
                    const s = sanitized[i];
                    if (typeof s.width === 'number' && s.width > 0) {
                        widthBatch.push({ key: s.colId, newWidth: s.width | 0 });
                    }
                }
                if (widthBatch.length) {
                    try { this.api.setColumnWidths(widthBatch, false); } catch {}
                }
            }

            // Apply "locked right" semantics if provided (pin+lock). Do this in a second pass.
            if (lockedRightSet.size > 0) {
                const locked = [];
                lockedRightSet.forEach(colId => {
                    if (!validSet.has(colId)) return;
                    locked.push({
                        colId,
                        pinned: 'right',
                        lockPosition: true,
                        suppressMovable: true
                    });
                });
                if (locked.length) {
                    this.api.applyColumnState({ state: locked, applyOrder: false });
                }
            }

            // Optional: apply models AFTER column state to avoid churn
            if (this.config.enableSortMemory && viewState.sortModel && this.api.setSortModel) {
                this.api.setSortModel(viewState.sortModel);
            }
            if (this.config.enableFilterMemory && viewState.filterModel && this.api.setFilterModel) {
                this.api.setFilterModel(viewState.filterModel);
            }

            // Wait one frame for AG Grid to finish its synchronous DOM update,
            // then sync tree UI from the actual applied column state.
            requestAnimationFrame(() => {
                // Sync selection/pin state in tree from actual grid, not the payload
                this.initializeSelectionStates();
                this.renderVirtualItems?.();
                // Custom names, if present
                if (this.customNames && Object.keys(this.customNames).length > 0) {
                    this.applyCustomNamesToGrid?.();
                }
                this.api.refreshHeader?.();
            });

            // Clear dirty bit for a fresh load
            this.hasUnsavedChanges = false;
            this._updateButtonStates?.();
        } finally {
            this.isLoadingState = false;
            this._rebindGridChangeListenersSafe();

            // Restore auto-size only if the view had no explicit widths;
            // otherwise keep it suppressed so future data refreshes don't clobber widths.
            if (!hasExplicitWidths) {
                try { this.api.setGridOption('autoSizeStrategy', prevAutoSize); } catch {}
                try { this.api.setGridOption('suppressAutoSize', prevSuppressAuto); } catch {}
            }

            // Clear re-entrancy guard after one frame so the grid has
            // fully processed the column state change.
            requestAnimationFrame(() => { this._is_applying = false; });
        }
    }

    _rebindGridChangeListenersSafe() {
        // Ensures listeners are reattached exactly once after apply
        try {
            this._removeGridChangeListeners();
            this.setupGridChangeListeners();
        } catch (_) {}
    }

    _initializeStateManagement() {
        // 1. Load custom columns first
        this.loadCustomColumnsFromCache();
        this.registerCachedCustomColumns();

        // 2. Load user presets from cache
        this.loadPresetsFromCache();

        // 3. Merge global presets (without overriding user presets)
        this.mergeGlobalPresets(this.globalPresets);

        // 4. Determine and apply the default preset.
        // Called from onFirstDataRendered so columns are available — no delay needed.
        this.applyDefaultPreset();
    }

    setupGridChangeListeners() {
        if (!this.api) return;

        // Debounced unified handler to keep tree in sync and mark dirty once.
        const debounceFn = this.debounce;
        const handler = (e) => this._handleGridChange(e);
        this._gridDebouncedChangeHandler = debounceFn ? debounceFn(handler, GRID_EVENT_DEBOUNCE_MS) : handler;

        this._removeGridChangeListeners?.();
        if (!this._gridListeners) this._gridListeners = new Map();

        // Events that change visibility/order/pin/sort/filter => must sync tree immediately
        const gridEvents = this.config.gridEvents;
        for (let i = 0; i < gridEvents.length; i++) {
            const type = gridEvents[i];
            this.api.addEventListener(type, this._gridDebouncedChangeHandler);
            this._gridListeners.set(type, this._gridDebouncedChangeHandler);
        }

        // When new columns are added dynamically, rebuild the tree to include them
        const onNewCols = () => {
            const fresh = this.getColumnDefs();
            if (!fresh || fresh.length === this.columnDefs?.length) return;
            this.columnDefs = fresh;
            this.fields = this.getFields();
            this.setupTree();
            this.initializeSelectionStates();
            this.buildFuseIndex();
            this.renderAllNodes();
        };
        this.api.addEventListener('newColumnsLoaded', onNewCols);
        this._gridListeners.set('newColumnsLoaded', onNewCols);
    }

    _removeGridChangeListeners() {
        if (!this.api || !this._gridListeners || this._gridListeners.size === 0) return;
        this._gridListeners.forEach((handler, type) => {
            this.api.removeEventListener(type, handler);
        });
        this._gridListeners.clear();
        if (this._gridDebouncedChangeHandler?.cancel) {
            this._gridDebouncedChangeHandler.cancel();
        }
    }

    _handleGridChange(e) {
        // Avoid feedback during programmatic apply, but always refresh tree snapshot
        if (!this.isLoadingState) {
            // Keep tree selection/pin indicators authoritative to the real grid
            // Fast path uses a single sweep over defs + grid columns
            this.initializeSelectionStates();
            this.renderVirtualItems?.();

            // Set dirty once; do not churn button state on every event
            if (!this.hasUnsavedChanges) {
                this.hasUnsavedChanges = true;
                this._updateButtonStates?.();
            }
        }
        if (this.engine) {
            this.engine._notifyColumnEvent(e.type)
        }

    }

    _syncTreeFromGrid() {
        if (this.isLoadingState) return;
        this.initializeSelectionStates();
        this.renderVirtualItems?.();
    }

    _setTimeout(fn, ms) {
        const id = setTimeout(() => {
            this._timeouts.delete(id);
            fn();
        }, ms);
        this._timeouts.add(id);
        return id;
    }

    _clearAllTimeouts() {
        for (const id of this._timeouts) clearTimeout(id);
        this._timeouts.clear();
    }

    _updateButtonStates() {
        const activePreset = this.presets.get(this.activePresetName);
        const isGlobal = activePreset?.metaData?.isGlobal;
        const isMutable = activePreset?.metaData?.isMutable !== false;
        const canSave = activePreset && !isGlobal && isMutable;

        if (this.saveBtn) {
            // Update tooltip to indicate behavior
            if (isGlobal) {
                this.saveBtn.setAttribute('data-tooltip', 'Save as New View (Global views are read-only)');
            } else if (!isMutable) {
                this.saveBtn.setAttribute('data-tooltip', 'Save as New View (This view is read-only)');
            } else {
                this.saveBtn.setAttribute('data-tooltip', 'Save View');
            }

            // Always enable save button if there are changes, but change its behavior
            // this.saveBtn.disabled = !this.hasUnsavedChanges;

            this.saveBtn.classList.toggle('unsaved-changes', this.hasUnsavedChanges);
            if (this.outsideSaveBtn) {
                this.outsideSaveBtn.classList.toggle('unsaved-changes', this.hasUnsavedChanges);
            }

            // Visual indicator for different save modes
            if (isGlobal || !isMutable) {
                this.saveBtn.classList.add('btn-warning'); // Different color for "save as new"
                if (this.outsideSaveBtn) this.outsideSaveBtn.classList.add('btn-warning');
            } else {
                this.saveBtn.classList.remove('btn-warning');
                if (this.outsideSaveBtn) this.outsideSaveBtn.classList.remove('btn-warning');
            }
        }

        if (this.reloadBtn) {
            this.reloadBtn.disabled = !activePreset;
        }
    }

    loadCustomColumnsFromCache() {
        try {
            const stored = localStorage.getItem(`treeColumnChooser_customColumns_${this.gridName}`);
            if (stored) {
                this.customColumns = new Map(JSON.parse(stored));
            }
        } catch (e) {
            console.error("Failed to load custom columns from cache", e);
            this.customColumns = new Map();
        }
    }

    saveCustomColumnsToCache() {
        try {
            localStorage.setItem(
                `treeColumnChooser_customColumns_${this.gridName}`,
                JSON.stringify(Array.from(this.customColumns.entries()))
            );
        } catch (e) {
            console.error("Failed to save custom columns to cache", e);
        }
    }

    registerCachedCustomColumns() {
        return

    }

    loadPresetsFromCache() {
        try {
            const stored = localStorage.getItem(`treeColumnChooser_presets_${this.gridName}`);
            if (stored) {
                const parsed = JSON.parse(stored);
                const userPresets = parsed.filter(p => !p.metaData?.isGlobal);
                userPresets.forEach(preset => {
                    this.presets.set(preset.name, preset);
                });
            }
        } catch (e) {
            console.error("Failed to load presets from cache", e);
        }
    }

    savePresetsToCache() {
        try {
            // Only save user-defined presets (not global ones)
            const presetsArray = Array.from(this.presets.values())
                                       .filter(p => !p.metaData?.isGlobal);

            // Strip default/falsy values from columnState to reduce size
            const compact = presetsArray.map(p => {
                if (!Array.isArray(p.columnState)) return p;
                const slimColumns = p.columnState.map(c => {
                    const s = { colId: c.colId };
                    if (c.hide) s.hide = true;
                    if (c.pinned) s.pinned = c.pinned;
                    if (c.sort) s.sort = c.sort;
                    if (typeof c.sortIndex === 'number') s.sortIndex = c.sortIndex;
                    if (typeof c.width === 'number' && c.width > 0) s.width = c.width;
                    if (c.aggFunc) s.aggFunc = c.aggFunc;
                    if (c.rowGroup) s.rowGroup = true;
                    if (typeof c.rowGroupIndex === 'number') s.rowGroupIndex = c.rowGroupIndex;
                    if (c.pivot) s.pivot = true;
                    if (typeof c.pivotIndex === 'number') s.pivotIndex = c.pivotIndex;
                    return s;
                });
                // Shallow copy, drop transient import metadata
                const { metadata, isTemporary, ...rest } = p;
                return { ...rest, columnState: slimColumns };
            });

            localStorage.setItem(
                `treeColumnChooser_presets_${this.gridName}`,
                JSON.stringify(compact)
            );
        } catch (e) {
            console.error("Failed to save presets to cache", e);
            if (e?.name === 'QuotaExceededError') {
                this.context.page.toastManager?.().error(
                    'Storage is full. Try deleting unused views to free up space.',
                    this.gridName.toUpperCase()
                );
            }
        }
    }

    mergeGlobalPresets(globalPresets = []) {
        globalPresets.forEach(globalPreset => {
            const existing = this.presets.get(globalPreset.name);

            // Only update if doesn't exist or version is newer
            if (!existing || (existing.metaData?.isGlobal && existing.version < globalPreset.version)) {
                this.presets.set(globalPreset.name, {
                    ...globalPreset,
                    metaData: {
                        ...globalPreset.metaData,
                        isGlobal: true
                    }
                });
            }
        });
    }

    async applyDefaultPreset() {
        // First check if there's already a user-defined default
        let defaultPreset = Array.from(this.presets.values()).find(p =>
            p?.metaData?.isDefault && !p?.metaData?.isGlobal
        );

        if (!defaultPreset) {
            if (this.context.page.userManager().userProfile$.get("firstName").toLowerCase() === 'igor') {
                if (this.presets.has("Igor's View")) {
                    defaultPreset = this.presets.get("Igor's View");
                }
            }
        }

        // If no user default exists, check for a global default
        if (!defaultPreset) {
            defaultPreset = Array.from(this.presets.values()).find(p =>
                p?.metaData?.isDefault && p?.metaData?.isGlobal
            );
        }

        // If still no default, use the first global preset as default
        if (!defaultPreset) {
            defaultPreset = Array.from(this.presets.values()).find(p =>
                p?.metaData?.isGlobal
            );

            // Mark it as default if found
            if (defaultPreset) {
                defaultPreset.metaData.isDefault = true;
            }
        }

        // If still no default (shouldn't happen), use first available preset
        if (!defaultPreset && this.presets.size > 0) {
            defaultPreset = Array.from(this.presets.values())[0];
            if (defaultPreset.metaData) {
                defaultPreset.metaData.isDefault = true;
            }
        }

        this.default_preset = defaultPreset;

        // Apply the default preset if found
        if (defaultPreset) {
            await this._loadPreset(defaultPreset.name);
        } else {
            this.grid$.set('presetLoaded', true);
        }
    }

    async _handleSave() {
        const activePreset = this.presets.get(this.activePresetName);

        // If no active preset or it's a global preset, create new instead
        if (!activePreset || activePreset.metaData?.isGlobal) {
            // Show create dialog with a message about why
            const message = activePreset?.metaData?.isGlobal ?
                `Cannot modify global view "${this.activePresetName}". Create a new view instead.` :
                'No active view to save. Create a new view.';

            this.context.page.toastManager().info(message, this.gridName.toUpperCase());

            // Create new view
            const result = await this.modalManager.show({
                title: 'Save as New View',
                body: activePreset?.metaData?.isGlobal ?
                    `<p class="text-info mb-3">Global views cannot be modified. Your changes will be saved as a new default view.</p>` : '',
                fields: [
                    {
                        type: 'text',
                        id: 'viewName',
                        label: 'View Name',
                        required: true,
                        value: activePreset ? `${activePreset.name} (copy)` : 'My View',
                        placeholder: activePreset ? `${activePreset.name} (Modified)` : 'My View'
                    },
                    {
                        type: 'textarea',
                        id: 'description',
                        label: 'Description (optional)',
                        rows: 2
                    }
                ],
                buttons: [
                    { text: 'Cancel', value: 'cancel' },
                    { text: 'Save as Default', value: 'save', class: 'btn-primary', isSubmit: true }
                ]
            });

            if (result && result.viewName) {
                const newName = result.viewName.trim();
                const description = result.description?.trim() || '';

                if (this.presets.has(newName)) {
                    this.context.page.toastManager().error(`A view named "${newName}" already exists.`, this.gridName.toUpperCase());
                    return;
                }

                const currentState = this._getGridState();
                if (!currentState) return;

                const owner = this.context.page.userManager().displayName ||
                    this.context.page.userManager?.getCurrentUserName?.() ||
                    'Unknown User';

                // Create new preset with default flag set to true
                const newPreset = {
                    name: newName,
                    ...currentState,
                    metaData: {
                        isMutable: true,
                        isTemporary: false,
                        isGlobal: false,
                        isDefault: true, // Set as default
                        owner: owner,
                        description: description,
                        lastModified: new Date().toISOString(),
                        created: new Date().toISOString(),
                        basedOn: activePreset?.name // Track what it was based on
                    },
                    version: '1.0.0',
                    timestamp: Date.now()
                };

                // Clear default flag from all other presets
                this.presets.forEach(preset => {
                    if (preset.metaData && preset.metaData.isDefault) {
                        preset.metaData.isDefault = false;
                    }
                });

                // Add the new preset
                this.presets.set(newName, newPreset);
                this.savePresetsToCache();

                // Load the new preset
                await this._loadPreset(newName);

                this.context.page.toastManager().success(
                    `${this.gridName.toUpperCase()} - Success!`,
                    `New default view "${newName}" created and applied.`
                );
            }
            return;
        }

        // For mutable presets, save normally
        if (activePreset.metaData?.isMutable !== false) {
            const currentState = this._getGridState();
            if (!currentState) return;

            // Preserve existing metadata but update last modified
            const updatedPreset = {
                ...activePreset,
                ...currentState,
                metaData: {
                    ...activePreset.metaData,
                    lastModified: new Date().toISOString()
                },
                version: activePreset.version || '1.0.0',
                timestamp: Date.now()
            };

            this.presets.set(this.activePresetName, updatedPreset);
            this.savePresetsToCache();
            this.hasUnsavedChanges = false;
            this._updateButtonStates();
            // this.updateStatus();
            this.context.page.toastManager().success(
                `${this.gridName.toUpperCase()} - Success!`,
                `View "${this.activePresetName}" updated.`
            );
        } else {
            // Preset is not mutable
            this.context.page.toastManager().warning(
                `View "${this.activePresetName}" is read-only. Create a new view to save changes.`
            );
            await this._handleCreate();
        }
    }

    async _handleFilterSelected() {
        if (!this._filterSelected) {
            this._filterSelected = true;
            this.filterSelectedBtn.innerHTML = this.selectedIcon;
            this.filterSelectedBtn.classList.toggle('highlight', true);
            this._updateSortColumns();
        } else {
            this._filterSelected = false;
            this.filterSelectedBtn.innerHTML = this.nonSelectedIcon;
            this.filterSelectedBtn.classList.toggle('highlight', false);
            this.expandedNodes.clear();
        }
        this.updateFlattenedNodes();
        this.renderAllNodes();
    }

    _applyDragReorder(orderedNodes) {
        if (!this.api) return;

        // Build desired column order from the drag result.
        // orderedNodes only contains visible/selected columns.
        // We need to produce a full applyColumnState that preserves
        // hidden columns in their relative positions while reordering
        // the visible ones according to the drag.

        const desiredOrder = orderedNodes.map(n => n.id);

        // Get the current full column state from AG-Grid (includes hidden cols)
        const currentState = this.api.getColumnState();
        if (!currentState) return;

        // Separate into the visible columns we're reordering vs everything else
        const desiredSet = new Set(desiredOrder);
        const otherCols = []; // hidden columns not in our reorder set
        const reorderedCols = new Map(); // colId -> state entry

        for (const cs of currentState) {
            if (desiredSet.has(cs.colId)) {
                reorderedCols.set(cs.colId, cs);
            } else {
                otherCols.push(cs);
            }
        }

        // Build final state: place reordered visible columns in the desired
        // order, preserving hidden columns in their original relative slots.
        // Strategy: walk the original state; replace visible-reordered cols
        // with the next item from desiredOrder.
        let desiredIdx = 0;
        const newState = [];
        for (const cs of currentState) {
            if (desiredSet.has(cs.colId)) {
                // Substitute with the next column from the drag order
                const nextId = desiredOrder[desiredIdx++];
                newState.push(reorderedCols.get(nextId));
            } else {
                newState.push(cs);
            }
        }

        this.api.applyColumnState({ state: newState, applyOrder: true });

        // Update sort column cache to match the new grid order
        this._updateSortColumns();
        this.hasUnsavedChanges = true;
        this._updateButtonStates();
    }

    _updateSortColumns() {
        const api = this.api;
        const _sc = this._sortedColumns;

        let i = 0;
        this.api.getDisplayedLeftColumns().forEach(c => {
            _sc.set(c.colId, i);
            i += 1;
        })
        this.api.getDisplayedCenterColumns().forEach(c => {
            _sc.set(c.colId, i);
            i += 1;
        })
        this.api.getDisplayedRightColumns().forEach(c => {
            _sc.set(c.colId, i);
            i += 1;
        });
    }


    async _handleCreate() {
        const defaultName = this.activePresetName
            ? this.activePresetName + ' (copy)'
            : 'My View';
        const result = await this.modalManager.show({
            title: 'Save New View',
            fields: [
                { type: 'text', id: 'viewName', label: 'View Name', required: true, value: defaultName },
                { type: 'textarea', id: 'description', label: 'Description (optional)', rows: 2 }
            ],
            buttons: [
                { text: 'Cancel', value: 'cancel' },
                { text: 'Save', value: 'save', class: 'btn-primary', isSubmit: true }
            ]
        });

        if (result && result.viewName) {
            const newName = result.viewName.trim();
            const description = result.description?.trim() || '';

            if (this.presets.has(newName)) {
                this.context.page.toastManager().error(`A view named "${newName}" already exists.`);
                return;
            }

            const currentState = this._getGridState();
            if (!currentState) return;

            const owner = this.context.page.userManager().displayName ||
                this.context.page.userManager?.getCurrentUserName?.() ||
                'Unknown User';

            const newPreset = {
                name: newName,
                ...currentState,
                metaData: {
                    isMutable: true,
                    isTemporary: false,
                    isGlobal: false,
                    isDefault: false,
                    owner: owner,
                    description: description,
                    lastModified: new Date().toISOString(),
                    created: new Date().toISOString()
                },
                version: '1.0.0',
                timestamp: Date.now()
            };

            this.presets.set(newName, newPreset);
            this.savePresetsToCache();
            await this._loadPreset(newName);
            this.context.page.toastManager().success(
                `${this.gridName.toUpperCase()} - Success!`,
                `View "${newName}" saved.`
            );
        }
    }

    async showLoadDialog() {
        const pageContext = this; // Capture context

        await this.modalManager.showCustom({
            title: "Manage Views",
            modalClass: "state-manager-modal",
            modalBoxClass: "w-full max-w-3xl h-[70vh] flex flex-col", // Wider, height, flex
            includeDefaultActions: true,
            setupContent: (contentArea, dialog) => {
                // Style content area for scrolling list
                contentArea.className = 'modal-custom-content flex-grow overflow-y-auto border-t border-b border-base-300 mb-4'; // Added borders
                contentArea.innerHTML = `<div class="state-list-container"><span class="loading loading-spinner loading-md p-4"> Loading views...</span></div>`;

                const listContainer = contentArea.querySelector('.state-list-container');
                const actionBar = dialog.querySelector('[data-role="modal-actions"]');

                // --- Setup Action Bar Buttons ---
                if (actionBar) {
                    actionBar.innerHTML = '';

                    // Import Button
                    const importButton = document.createElement('button');
                    importButton.className = 'coming-soon-btn btn btn-sm btn-outline btn-disabled tooltip-right';
                    importButton.innerHTML = `Import View`;
                    // importButton.addEventListener('click', async () => {
                    //     dialog.close('cancel');
                    //     await pageContext.importState();
                    // });

                    //importButton.disabled = true; // TODO
                    importButton.setAttribute('data-tooltip', 'Coming Soon');

                    actionBar.appendChild(importButton);

                    // Spacer
                    const spacer = document.createElement('div');
                    spacer.className = 'flex-grow';
                    actionBar.appendChild(spacer);

                    // Close Button
                    const closeButton = document.createElement('button');
                    closeButton.className = 'btn btn-sm';
                    closeButton.textContent = 'Close';
                    closeButton.addEventListener('click', () => dialog.close('cancel'));
                    actionBar.appendChild(closeButton);
                }

                // Cleanup is handled by _attachStateListListeners via { once: true }

                // --- Initial Render & Attach Listeners ---
                try {
                    pageContext._renderStateList(listContainer, pageContext);
                    pageContext._attachStateListListeners(listContainer, dialog, pageContext);
                } catch (err) {
                    console.error("Error setting up state list:", err);
                    listContainer.innerHTML = `<div class="text-error p-4">Failed to load views. Check console.</div>`;
                }
            } // End setupContent
        });
    }

    _renderStateItemHTML(state, isCurrent, hasUnsavedChanges) {
        const isCurrentState = isCurrent === true ||
            (typeof isCurrent === 'object' && isCurrent?.name === state.name) ||
            state?.metaData?.isTemporary;

        const displayName = state?.metaData?.isTemporary ?
            `${state.name} (Temporary)` : state.name;
        const canModify = !state?.metaData?.isTemporary &&
            !state?.metaData?.isGlobal &&
            state?.metaData?.isMutable !== false;
        const canDelete = canModify;

        // Format metadata for display
        const owner = state?.metaData?.owner || '';
        const lastModified = state?.metaData?.lastModified ?
            new Date(state.metaData.lastModified).toLocaleDateString() : '';
        const description = state?.metaData?.description || '';

        const escHtml = (s) => s ? s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : '';
        let metadataHtml = '';
        if (owner || lastModified || description) {
            metadataHtml = '<div class="state-item-meta" style="font-size: 0.75rem; color: #666; margin-top: 2px;">';
            if (description) {
                metadataHtml += `<div style="font-style: italic;">${escHtml(description)}</div>`;
            }
            if (owner || lastModified) {
                metadataHtml += '<div>';
                if (owner) metadataHtml += `By ${escHtml(owner)}`;
                if (owner && lastModified) metadataHtml += ' • ';
                if (lastModified) metadataHtml += `${escHtml(lastModified)}`;
                metadataHtml += '</div>';
            }
            metadataHtml += '</div>';
        }

        return `
        <div class="state-item flex items-center justify-between p-2 border-b border-base-300 hover:bg-base-200 
             ${state?.metaData?.isDefault ? 'default-state' : ''} 
             ${isCurrentState ? 'active bg-base-300' : ''}" 
             data-state-name="${escHtml(state.name)}">
            <div class="state-item-info flex-grow mr-2 overflow-hidden">
                <div class="state-item-name block truncate 
                     ${state?.metaData?.isTemporary ? 'italic' : ''} 
                     ${state?.metaData?.isDefault ? 'font-semibold' : ''}
                     ${isCurrentState && hasUnsavedChanges && !state?.metaData?.isTemporary ? 'italic text-error' : ''}">
                    ${displayName}
                    ${state?.metaData?.isDefault ?
            ' <span class="badge badge-primary badge-sm ml-1">Default</span>' : ''}
                    ${isCurrentState && hasUnsavedChanges && !state?.metaData?.isTemporary ?
            ' <span class="badge badge-warning badge-sm ml-1">Unsaved</span>' : ''}
                    ${state?.metaData?.isGlobal ?
            ' <span class="badge badge-info badge-sm ml-1">Global</span>' : ''}
                </div>
                ${metadataHtml}
            </div>
            <div class="state-item-actions flex items-center space-x-1 shrink-0">
                <button class="btn btn-xs btn-ghost tooltip-top" data-tooltip="Load View" data-action="load"
                        ${state?.metaData?.isTemporary ? 'disabled' : ''}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                </button>
                <button class="btn btn-xs btn-ghost bookmark-btn tooltip-top ${state?.metaData?.isDefault ? 'active' : ''}"
                        data-action="default" 
                        data-tooltip="${state?.metaData?.isDefault ? 'Remove as default' : 'Set as default'}"
                        ${state?.metaData?.isTemporary ? 'disabled' : ''}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" 
                         fill="${state?.metaData?.isDefault ? 'currentColor' : 'none'}" 
                         stroke="currentColor" stroke-width="2">
                        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                    </svg>
                </button>
                <button class="btn btn-xs btn-ghost tooltip-top" data-action="rename" 
                        data-tooltip="Rename"
                        ${!canModify ? 'disabled' : ''}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"></path>
                        <path d="m15 5 4 4"></path>
                    </svg>
                </button>
                <button class="btn btn-xs btn-ghost tooltip-top" data-action="delete" 
                        data-tooltip="Delete"
                        ${!canDelete ? 'disabled' : ''}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            </div>
        </div>
    `;
    }

    _renderStateList(listContainer, context) {
        const states = context.presets;
        let viewStates = Array.from(states.values());

        // Add global presets if not already in the list
        context.globalPresets.forEach(globalPreset => {
            const exists = viewStates.some(s =>
                s.name === globalPreset.name && s.metaData?.isGlobal
            );
            if (!exists) {
                // Make sure global presets have their metadata
                viewStates.push({
                    ...globalPreset,
                    metaData: {
                        ...globalPreset.metaData,
                        isGlobal: true
                    }
                });
            }
        });

        // Sort states with proper priority
        viewStates.sort((a, b) => {
            // 1. Temporary states always first
            if (a.metaData?.isTemporary !== b.metaData?.isTemporary) {
                return a.metaData?.isTemporary ? -1 : 1;
            }

            // 2. Default state should be second (after temporary)
            // This should check the current default state, not the active state
            const aIsDefault = a.metaData?.isDefault === true;
            const bIsDefault = b.metaData?.isDefault === true;
            if (aIsDefault !== bIsDefault) {
                return aIsDefault ? -1 : 1;
            }

            // 3. Currently active/loaded state (if different from default)
            const isACurrent = a.name === context.activePresetName;
            const isBCurrent = b.name === context.activePresetName;
            if (isACurrent !== isBCurrent) {
                return isACurrent ? -1 : 1;
            }

            // 4. Global states come before user states
            const aIsGlobal = a.metaData?.isGlobal === true;
            const bIsGlobal = b.metaData?.isGlobal === true;
            if (aIsGlobal !== bIsGlobal) {
                return aIsGlobal ? -1 : 1;
            }

            // 5. Sort by last modified date (newest first)
            const dateA = new Date(a.metaData?.lastModified || a.timestamp || 0);
            const dateB = new Date(b.metaData?.lastModified || b.timestamp || 0);
            if (dateA.getTime() !== dateB.getTime()) {
                return dateB.getTime() - dateA.getTime();
            }

            // 6. Finally sort alphabetically by name
            return a.name.localeCompare(b.name);
        });

        if (viewStates.length === 0) {
            listContainer.innerHTML = `
            <div class="text-center p-8 text-base-content/60">
                No views found. Create a new view to get started.
            </div>`;
        } else {
            // Pass the current state correctly for rendering
            listContainer.innerHTML = viewStates.map(state => {
                // Find if this is the current state
                const isCurrent = state.name === context.activePresetName;
                return context._renderStateItemHTML(state, isCurrent ? state : null, context.hasUnsavedChanges);
            }).join('');
        }
    }

    _attachStateListListeners(listContainer, dialog, context) {
        // Remove any existing listener before adding a new one
        if (context._stateListClickHandler) {
            listContainer.removeEventListener('click', context._stateListClickHandler);
            context._stateListClickHandler = null;
        }

        const handleListClick = async (event) => {
            const button = event.target.closest('button[data-action]');
            if (!button || button.disabled) return;

            const stateItem = button.closest('.state-item');
            const stateName = stateItem?.dataset?.stateName;
            const action = button.dataset.action;

            if (!stateName) return;

            // Prevent double-clicking
            if (button.dataset.processing === 'true') return;
            button.dataset.processing = 'true';

            const originalButtonContent = button.innerHTML;
            const showLoading = !['rename', 'delete', 'default'].includes(action);

            try {
                if (showLoading) {
                    button.innerHTML = '<span class="loading loading-spinner loading-xs"></span>';
                    button.disabled = true;
                }

                switch (action) {
                    case 'load':
                        await context._loadPreset(stateName);
                        dialog.close('loaded');
                        context.context.page.toastManager().success(
                            `View loaded: '${stateName}'`,
                            'Successfully loaded grid view'
                        );
                        break;

                    case 'default': {
                        const currentState = context.presets.get(stateName);
                        if (!currentState) break;

                        const currentIsDefault = currentState.metaData?.isDefault;

                        // Check if trying to remove default from the only view
                        if (currentIsDefault) {
                            const nonTempViews = Array.from(context.presets.values())
                                                       .filter(p => !p.metaData?.isTemporary);

                            if (nonTempViews.length === 1) {
                                context.context.page.toastManager().warning(
                                    'Cannot remove default status - at least one view must be default.',
                                    'Warning'
                                );
                                break;
                            }
                        }

                        const success = await context.setDefaultState(stateName, !currentIsDefault);

                        if (success) {
                            // Re-render the list to show updated default status
                            context._renderStateList(listContainer, context);
                            // Re-attach listeners - but the old one is removed first
                            context._attachStateListListeners(listContainer, dialog, context);
                        }
                        break;
                    }

                    case 'rename': {
                        const stateToEdit = context.presets.get(stateName);
                        if (!stateToEdit) break;

                        const editResult = await context.modalManager.show({
                            title: `Edit View: ${stateName}`,
                            fields: [
                                {
                                    id: 'newName',
                                    label: 'View Name',
                                    type: 'text',
                                    value: stateName,
                                    required: true
                                },
                                {
                                    id: 'newDescription',
                                    label: 'Description',
                                    type: 'textarea',
                                    value: stateToEdit.metaData?.description || '',
                                    rows: 3
                                }
                            ],
                            buttons: [
                                { text: 'Cancel', value: 'cancel' },
                                {
                                    text: 'Save Changes',
                                    value: 'save',
                                    class: 'btn-primary',
                                    isSubmit: true
                                }
                            ]
                        });

                        if (editResult && editResult.newName) {
                            const success = await context.renameState(
                                stateName,
                                editResult.newName.trim(),
                                editResult.newDescription?.trim()
                            );

                            if (success) {
                                // Re-render the list with updated names
                                context._renderStateList(listContainer, context);
                                // Re-attach listeners - but the old one is removed first
                                context._attachStateListListeners(listContainer, dialog, context);
                            }
                        }
                        break;
                    }

                    case 'delete': {
                        const stateToDelete = context.presets.get(stateName);
                        if (!stateToDelete) break;

                        const confirmResult = await context.modalManager.show({
                            title: `Delete View "${stateName}"?`,
                            body: `
                            <p class="text-error">
                                This action cannot be undone. Are you sure you want to delete this view?
                            </p>
                            ${stateToDelete.metaData?.description ?
                                `<p class="text-sm mt-2">
                                   Description: ${stateToDelete.metaData.description}
                               </p>` : ''}
                        `,
                            buttons: [
                                { text: 'Cancel', value: 'cancel' },
                                { text: 'Delete', value: 'delete', class: 'btn-error' }
                            ]
                        });

                        if (confirmResult === 'delete') {
                            const success = await context.deleteState(stateName);

                            if (success) {
                                // Re-render the list without the deleted item
                                context._renderStateList(listContainer, context);
                                // Re-attach listeners - but the old one is removed first
                                context._attachStateListListeners(listContainer, dialog, context);
                            }
                        }
                        break;
                    }

                    case 'share':
                        // Share functionality not implemented yet
                        context.context.page.toastManager().info(
                            'Share functionality coming soon',
                            'Info'
                        );
                        break;

                    default:
                        console.warn(`Unhandled action: ${action}`);
                }
            } catch (err) {
                console.error(`Error performing action ${action} on ${stateName}:`, err);
                context.context.page.toastManager().error(
                    `Failed to ${action} view: ${err.message || 'Unknown error'}`,
                    'Error'
                );
            } finally {
                // Reset button state
                delete button.dataset.processing;
                if (showLoading && button.isConnected) {
                    button.innerHTML = originalButtonContent;
                    button.disabled = false;
                }
            }
        };

        // Store the handler reference so we can remove it later
        context._stateListClickHandler = handleListClick;
        listContainer.addEventListener('click', handleListClick);

        dialog.addEventListener('close', () => {
            listContainer.removeEventListener('click', handleListClick);
            context._stateListClickHandler = null;
        }, { once: true });
    }

    async _handleReload() {
        // Reapply active preset or current stored state snapshot
        const name = this.activePresetName;
        if (name && this.presets?.get && this.presets.has(name)) {
            await this._applyGridState(this.presets.get(name));
            this.initParams?.context?.page?.toastManager?.().info?.(`View "${name}" reloaded.`);
            return;
        }
        // Fallback: reload last known grid state
        const snap = this._getGridState();
        if (snap) {
            await this._applyGridState(snap);
            this.initParams?.context?.page?.toastManager?.().info?.(`Current view reapplied.`);
        }
    }

    async _loadPreset(name) {
        const preset = this.presets.get(name);
        if (!preset) {
            console.error(`Preset "${name}" not found.`);
            return;
        }

        this.isLoadingState = true;
        await this._applyGridState(preset);
        this.activePresetName = name;

        this.hasUnsavedChanges = false;
        // Wait two frames: first for AG Grid's synchronous column processing,
        // second for the resulting DOM paint to complete — then unlock state.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                this.isLoadingState = false;
                this._updateButtonStates();
                this.grid$?.set('presetLoaded', true);
            });
        });
    }

    async _handleNewColumn() {
        // todo
    }



    // ==================== Tree Management ====================

    /**
     * Setup tree structure from column definitions
     */
    setupTree() {
        if (!this.columnDefs || this.columnDefs.length === 0) return;

        // Clear caches when tree structure changes
        this.invalidateCaches();

        const filteredDefs = this.filterColumnDefs(this.columnDefs);
        this.originalTreeData = this.buildGroupHierarchy(filteredDefs);

        // Build node cache after tree is created
        this.nodeCache = new Map();
        this.buildNodeCache(this.originalTreeData);

        // Apply any custom names
        this.applyCustomNames(this.originalTreeData);

        // Initialize expanded state
        //this.initializeExpandedState();

        // Flatten nodes for virtualization
        this.updateFlattenedNodes();
    }

    /**
     * Invalidate all caches
     */
    invalidateCaches() {
        if (this.nodeCache) {
            this.nodeCache.clear();
        }
        if (this.fieldCache) {
            this.fieldCache.clear();
        }
        this.searchCache.clear();
        this.nodeTextCache.clear();
    }

    /**
     * Filter column definitions based on configuration
     */
    filterColumnDefs(columnDefs) {
        return columnDefs.filter(col => {
            const context = col?.context || {};

            // Check if column should be hidden from menu
            if (context.hideFromMenu) return false;

            // Check if suppressed for this specific grid
            if (context.suppressColumnMenu && this.gridName) {
                const suppressList = Array.isArray(context.suppressColumnMenu)
                    ? context.suppressColumnMenu
                    : [context.suppressColumnMenu];
                if (suppressList.includes(this.gridName)) return false;
            }

            return true;
        });
    }

    /**
     * Build group hierarchy from column definitions
     */
    buildGroupHierarchy(colDefs) {
        const nodes = [];
        const groupMap = new Map();

        // Process each column definition
        colDefs.forEach(colDef => {
            const colId = colDef.colId || colDef.field;
            if (!colId) return;

            const context = colDef.context || {};
            const headerName = context.treeHeader || colDef?.context?.menuNameOverride || colDef.headerName || colId;

            const columnNode = {
                id: colId,
                text: headerName,
                originalText: headerName,
                isGroup: false,
                children: null,
                childNames: new Set(),
                colDef: colDef,
                level: 0
            };

            if (context.customColumnGroup) {
                // Handle grouped column
                const groupPath = context.customColumnGroup.split('/');
                const parentGroup = this.ensureGroupPath(groupPath, nodes, groupMap);

                if (parentGroup) {
                    if (!parentGroup.children) parentGroup.children = [];
                    if (!parentGroup.childNames) parentGroup.childNames = new Set();
                    parentGroup.children.push(columnNode);
                    columnNode.parent = parentGroup.id;
                    columnNode.level = parentGroup.level + 1;
                } else {
                    nodes.push(columnNode);
                }
            } else {
                // Ungrouped column
                nodes.push(columnNode);
            }
        });

        nodes.forEach(node => {
            this._recursiveChildren(node)
        });

        // Sort nodes
        this.sortTreeNodes(nodes);
        return nodes;
    }

    _recursiveChildren(node, children) {
        if (children == null) children = new Set();
        if (!node.children) {
            children.add(node.id);
            return children;
        }
        node.children.forEach(child=>{
            children = this._recursiveChildren(child, children);
        });
        node.childNames = new Set([...children]);
        return children;
    }

    /**
     * Ensure group path exists and return the leaf group
     */
    ensureGroupPath(pathArray, rootNodes, groupMap) {
        let currentLevel = rootNodes;
        let parentGroup = null;
        let fullPath = '';
        let level = 0;

        for (const groupName of pathArray) {
            fullPath = fullPath ? `${fullPath}/${groupName}` : groupName;

            if (!groupMap.has(fullPath)) {
                const groupNode = {
                    id: `group_${fullPath.replace(/[^a-zA-Z0-9_]/g, '_')}`,
                    text: groupName,
                    originalText: groupName,
                    isGroup: true,
                    children: [],
                    childNames: new Set(),
                    level: level,
                };

                if (parentGroup) {
                    parentGroup.children.push(groupNode);
                    groupNode.parent = parentGroup.id;
                } else {
                    currentLevel.push(groupNode);
                }
                groupMap.set(fullPath, groupNode);
                parentGroup = groupNode;
            } else {
                parentGroup = groupMap.get(fullPath);
            }

            level++;
        }

        return parentGroup;
    }

    /**
     * Sort tree nodes recursively
     */
    sortTreeNodes(nodes) {
        // Separate groups and columns
        const groups = nodes.filter(n => n.isGroup);
        const columns = nodes.filter(n => !n.isGroup);

        // Sort each category
        groups.sort((a, b) => a.text.localeCompare(b.text));
        columns.sort((a, b) => a.text.localeCompare(b.text));

        // Sort children recursively
        groups.forEach(group => {
            if (group.children && group.children.length > 0) {
                this.sortTreeNodes(group.children);
            }
        });

        // Rebuild array with columns first, then groups
        nodes.length = 0;
        nodes.push(...columns, ...groups);

        return nodes;
    }

    /**
     * Apply custom names to nodes
     */
    applyCustomNames(nodes) {
        nodes.forEach(node => {
            const customName = this.customNames[node.id];
            if (customName) {
                node.text = customName;
                node.hasCustomName = true;

                // Also apply to the grid column header
                if (this.api) {
                    const column = this.api.getColumn(node.id);
                    if (column) {
                        const colDef = column.getColDef();
                        // Store original header name if not already stored
                        if (!colDef.originalHeaderName) {
                            colDef.originalHeaderName = colDef.headerName || colDef.field;
                        }
                        colDef.headerName = customName;
                    }
                }
            }

            if (node.children) {
                this.applyCustomNames(node.children);
            }
        });

        // Refresh grid headers if we made any changes
        if (this.api && Object.keys(this.customNames).length > 0) {
            this.api.refreshHeader();
        }
    }

    /**
     * Initialize expanded state
     */
    initializeExpandedState() {
        // Load saved expanded state or use defaults
        const savedState = this.loadExpandedState();
        if (savedState) {
            this.expandedNodes = new Set(savedState);
        } else {
            // Expand first level by default
            this.expandFirstLevel();
        }
    }

    /**
     * Expand first level of groups
     */
    expandFirstLevel() {
        this.originalTreeData.forEach(node => {
            if (node.isGroup) {
                this.expandedNodes.add(node.id);
            }
        });
    }

    /**
     * Update flattened nodes for virtualization
     */
    updateFlattenedNodes() {

        let flattenedNodes = [];
        const traverse = (nodes, level = 0, parentExpanded = true) => {
            nodes.forEach(node => {
                if (parentExpanded || (this._filterSelected)) {
                    flattenedNodes.push({...node, level});
                }
                if (node.children && ((this.expandedNodes.has(node.id) && parentExpanded) || (this._filterSelected))) {
                    traverse(node.children, level + 1, true);
                }
            });
        };

        const nodesToTraverse = (this.searchTerm != null) && (this.searchTerm.toString().trim() !== '')
            ? this.getFilteredNodes()
            : this.originalTreeData;
        traverse(nodesToTraverse);

        if (this._filterSelected) {
            flattenedNodes = flattenedNodes.filter(n=>!n.isGroup && this.selectedNodes.has(n.id)).map(n=>{
                return {...n, level: 0, parent: null};
            });
            flattenedNodes = flattenedNodes.toSorted((a,b) => {
                return (this._sortedColumns.get(a.id) ?? 0) - (this._sortedColumns.get(b.id) ?? 0);
            })
        }
        this.flattenedNodes = flattenedNodes;
    }

    // ==================== Search Functionality ====================

    /**
     * Build Fuse search index
     */
    buildFuseIndex() {
        if (!this.columnDefs || this.columnDefs.length === 0) return;

        const fuseOptions = {
            keys: [
                { name: 'field', weight: 1 },
                { name: 'headerName', weight: 1.25 },
                { name: 'pivotName', weight: 0.25 },
                { name: 'metaTags', weight: 0.5 },
                { name: 'groupNames', weight: 0.25 },
                { name: 'autoTags', weight: 0.02 },
                { name: 'customName', weight: 0.9 }
            ],
            includeScore: true,
            includeMatches: false,
            isCaseSensitive: false,
            ignoreDiacritics: true,
            shouldSort: true,
            threshold: 0.02,
            ignoreLocation: true,
            useExtendedSearch: true,
            minMatchCharLength: 2,
            findAllMatches: false
        };

        const fuseData = this.columnDefs.map(col => {
            const colId = col.colId || col.field;
            const customName = this.customNames[colId];

            return {
                id: colId,
                field: col.field || colId,
                headerName: col.headerName,
                pivotName: col?.context?.pivotName,
                metaTags: col?.context?.metaTags,
                groupNames: col?.context?.customColumnGroup?.split("/"),
                autoTags: this.generateAutoTags(col),
                customName: customName || null,
                _searchImportance: col?.context?.search_importance ?? 1
            };
        });

        this.fuse = new Fuse(fuseData, fuseOptions);
    }

    /**
     * Generate auto tags for improved search
     */
    generateAutoTags(col) {
        const tags = new Set();
        const sources = [
            col.field,
            col.headerName,
            col?.context?.pivotName,
            col?.context?.treeHeader
        ].filter(Boolean);

        sources.forEach(source => {
            // Original
            tags.add(source.toLowerCase());

            // Without spaces
            tags.add(source.toLowerCase().replace(/\s+/g, ''));

            // Camel case split
            const camelSplit = source.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();
            tags.add(camelSplit);

            // Common abbreviations
            const abbreviations = {
                'spread': 'spd',
                'spd': 'spread',
                'price': 'px',
                'px': 'price',
                'yld': 'yield',
                'yield': 'mmy',
                'average': 'avg',
                'avg': 'average'
            };

            Object.entries(abbreviations).forEach(([from, to]) => {
                if (source.toLowerCase().includes(from)) {
                    tags.add(source.toLowerCase().replace(from, to));
                }
            });
        });

        return Array.from(tags);
    }

    async handleSearch() {
        this.searchTerm = this.searchInput.value.toLowerCase().trim();

        if (this.searchTerm) {
            this.searchInput.classList.add('active');
            await this.performSearch();
        } else {
            this.searchInput.classList.remove('active');
            this.searchMatches.clear();
            this.updateFlattenedNodes();
            this.renderAllNodes();
        }
    }

    async performSearch() {
        if (!this.fuse) return;

        const results = this.fuse.search(this.searchTerm);

        // Re-rank results using Fuse score combined with search_importance.
        // Fuse score: 0 = perfect match, 1 = worst; importance: higher = better (default 1).
        // Combined: lower score means the item should appear first.
        results.sort((a, b) => {
            const impA = a.item._searchImportance ?? 1;
            const impB = b.item._searchImportance ?? 1;
            // Divide fuse score by importance so higher importance lowers the effective score
            const scoreA = (a.score ?? 0) / Math.max(impA, 0.01);
            const scoreB = (b.score ?? 0) / Math.max(impB, 0.01);
            return scoreA - scoreB;
        });

        this._lastFuseResults = results;

        this.searchMatches.clear();
        this._fuseRank.clear();

        for (let i = 0; i < results.length; i++) {
            const r = results[i];
            if (r?.item?.id != null) {
                this._fuseRank.set(r.item.id, i); // lower = better
            }
            if (r.matches) this.searchMatches.set(r.item.id, r.matches);
        }

        this.updateFlattenedNodes();
        this.renderAllNodes();
    }


    /**
     * Get filtered nodes based on search
     */
    getFilteredNodes() {
        if (!this.searchTerm || !this.fuse) return this.originalTreeData;

        const results = this.fuse.search(this.searchTerm);
        if (!results || results.length === 0) return [];

        // Re-rank with search_importance (same logic as performSearch)
        results.sort((a, b) => {
            const impA = a.item._searchImportance ?? 1;
            const impB = b.item._searchImportance ?? 1;
            const scoreA = (a.score ?? 0) / Math.max(impA, 0.01);
            const scoreB = (b.score ?? 0) / Math.max(impB, 0.01);
            return scoreA - scoreB;
        });

        const rank = new Map();
        for (let i = 0; i < results.length; i++) rank.set(results[i].item.id, i);
        const matchedIds = new Set(rank.keys());

        const dfs = (nodes) => {
            const out = [];
            let minRank = Number.POSITIVE_INFINITY;

            for (const node of nodes) {
                let include = false;
                let child = null;
                let selfMin = Number.POSITIVE_INFINITY;

                if (node.children && node.children.length) {
                    const r = dfs(node.children);
                    if (r.nodes.length) {
                        include = true;
                        child = r.nodes;
                        selfMin = Math.min(selfMin, r.minRank);
                        this.expandedNodes.add(node.id);
                    }
                }

                if (!node.isGroup && matchedIds.has(node.id)) {
                    include = true;
                    selfMin = Math.min(selfMin, rank.get(node.id));
                }

                if (include) {
                    const projected = { ...node, children: child && child.length ? child : null };
                    projected.__minRank = selfMin;
                    out.push(projected);
                    if (selfMin < minRank) minRank = selfMin;
                }
            }

            out.sort((a, b) => {
                const ar = a.__minRank;
                const br = b.__minRank;
                if (ar !== br) return ar - br;
                const at = a.text || a.originalText || '';
                const bt = b.text || b.originalText || '';
                return at.localeCompare(bt);
            });

            for (const n of out) delete n.__minRank;
            if (minRank === Number.POSITIVE_INFINITY) minRank = Number.MAX_SAFE_INTEGER;
            return { nodes: out, minRank };
        };

        return dfs(this.originalTreeData).nodes;
    }

    /**
     * Clear search
     */
    clearSearch() {
        this.searchTerm = '';
        this.searchInput.value = '';
        this.searchInput.classList.remove('active');
        this.clearButton.style.display = 'none';
        this.expandedNodes.clear();
        this.searchMatches.clear();
        this.updateFlattenedNodes();
        this.renderAllNodes();
        this.searchInput.focus();
    }

    // ==================== Virtualization ====================

    setupVirtualizer() {
        // Skip virtualizer completely - just render all nodes
        this.renderAllNodes();
    }


    updateVirtualizer() {
        this.renderAllNodes();
    }

    forceRender() {
        this.renderAllNodes();
    }

    renderVirtualItems() {
        this.renderAllNodes();
    }

    renderAllNodes() {
        if (this.flattenedNodes.length === 0) {
            this.virtualContainer.innerHTML = '<div class="empty-state">No items to display</div>';
            return;
        }

        const itemHeight = 28;
        const totalNodes = this.flattenedNodes.length;
        const VIRTUALIZE_THRESHOLD = 200;

        // Clear container
        this.virtualContainer.innerHTML = '';
        this.virtualContainer.style.height = `${totalNodes * itemHeight}px`;
        this.virtualContainer.style.position = 'relative';

        if (totalNodes <= VIRTUALIZE_THRESHOLD) {
            // Small list: render all nodes directly
            this.flattenedNodes.forEach((node, index) => {
                const element = this.createSimpleNodeElement(node, index);
                this.virtualContainer.appendChild(element);
            });
        } else {
            // Large list: only render visible window + buffer
            const scrollParent = this.virtualContainer.parentElement;
            const viewportH = scrollParent ? scrollParent.clientHeight : 600;
            const scrollTop = scrollParent ? scrollParent.scrollTop : 0;
            const buffer = 20;
            const startIdx = Math.max(0, Math.floor(scrollTop / itemHeight) - buffer);
            const endIdx = Math.min(totalNodes, Math.ceil((scrollTop + viewportH) / itemHeight) + buffer);

            for (let i = startIdx; i < endIdx; i++) {
                const element = this.createSimpleNodeElement(this.flattenedNodes[i], i);
                element.style.position = 'absolute';
                element.style.top = `${i * itemHeight}px`;
                element.style.width = '100%';
                this.virtualContainer.appendChild(element);
            }

            if (scrollParent && !this._virtualScrollBound) {
                this._virtualScrollBound = true;
                this._scrollParent = scrollParent;
                let rafPending = false;
                const onScroll = () => {
                    if (rafPending) return;
                    rafPending = true;
                    this._scrollRafId = requestAnimationFrame(() => {
                        rafPending = false;
                        this.renderAllNodes();
                    });
                };
                this._virtualScrollHandler = onScroll;
                scrollParent.addEventListener('scroll', onScroll, { passive: true, signal: this.signal || undefined });
            }
        }
    }

    createSimpleNodeElement(node, index) {
        const element = document.createElement('div');
        element.className = 'tree-node';
        element.dataset.nodeId = node.id;
        element.dataset.index = index;

        const inFilterSelected = this._filterSelected;

        element.style.cssText = `
        height: 28px;
        display: flex;
        align-items: center;
        padding: 4px 8px;
        border-bottom: 1px solid var(--ag-border-color, #eee);
        cursor: pointer;
        user-select: none;
        box-sizing: border-box;
    `;

        // Drag handle for filter-selected mode
        if (inFilterSelected && !node.isGroup) {
            element.draggable = true;
            const dragHandle = document.createElement('span');
            dragHandle.className = 'drag-handle';
            dragHandle.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24"><path fill="currentColor" d="M9 3h2v2H9zm4 0h2v2h-2zM9 7h2v2H9zm4 0h2v2h-2zM9 11h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9zm4 0h2v2h-2z"/></svg>';
            dragHandle.style.cssText = 'cursor:grab; opacity:0.4; margin-right:4px; flex-shrink:0; display:flex; align-items:center;';
            dragHandle.title = 'Drag to reorder';
            element.appendChild(dragHandle);
        }

        const indent = document.createElement('div');
        indent.style.width = `${node.level * 20}px`;
        indent.style.flexShrink = '0';
        element.appendChild(indent);

        // Expand/collapse icon (no handler here)
        if (node.isGroup && node.children && node.children.length > 0) {
            const expandBtn = document.createElement('button');
            expandBtn.className = 'expand-btn';
            expandBtn.innerHTML = this.expandedNodes.has(node.id) ? '▼' : '▶';
            expandBtn.style.cssText = `
            width: 20px; height: 20px; display:flex; align-items:center; justify-content:center;
            font-size:10px; cursor:pointer; border:none; background:none; padding:0; margin-right:4px;
        `;
            element.appendChild(expandBtn);
        } else {
            const spacer = document.createElement('div');
            spacer.style.width = '24px';
            spacer.style.flexShrink = '0';
            element.appendChild(spacer);
        }

        // Checkbox (no handler here)
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'tree-checkbox';
        checkbox.checked = this.selectedNodes.has(node.id);
        checkbox.indeterminate = this.indeterminateNodes.has(node.id);
        checkbox.style.cssText = 'margin-right: 8px; cursor: pointer;';
        const isRequired = !node.isGroup && this._getRequiredSet().has(node.id);
        if (isRequired) {
            checkbox.disabled = true;
            checkbox.title = 'Required column';
        }
        element.appendChild(checkbox);

        // Text/content
        const textContent = document.createElement('div');
        textContent.className = 'tree-text';
        textContent.style.cssText = 'flex: 1; display: flex; align-items: center;';
        const displayName = this.getNodeDisplayName(node);
        if (node.isGroup) {
            const strong = document.createElement('strong');
            strong.textContent = displayName;
            textContent.appendChild(strong);
        } else {
            textContent.textContent = displayName;
        }
        if (node.hasCustomName) {
            const customIndicator = document.createElement('span');
            customIndicator.innerHTML = '(©)';
            customIndicator.style.cssText = 'color: var(--amber-500); font-size: 10px; margin-left:5px;';
            customIndicator.title = 'Custom name';
            textContent.appendChild(customIndicator);
        }
        element.appendChild(textContent);

        // Right-side icons (no handlers here)
        if (!node.isGroup) {
            const iconsContainer = document.createElement('div');
            iconsContainer.style.cssText = 'display: flex; gap: 4px; margin-left: auto;';
            const colDef = node.colDef;
            const context = colDef?.context || {};
            const column = this.api?.getColumn(node.id);

            if (this.showPin && context.showPin !== false) {
                const isPinnedLeft = column?.pinned === 'left';
                const pinBtn = document.createElement('span');
                pinBtn.className = `col-action-icon pin-icon ${isPinnedLeft ? 'active' : ''}`;
                pinBtn.dataset.colId = node.id;
                pinBtn.dataset.iconType = 'pin';
                pinBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 14 14"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" d="M9.73 7.65L13 5.54A1 1 0 0 0 13.21 4L10 .79A1 1 0 0 0 8.46 1L6.3 4.23l-4.49 1a.6.6 0 0 0-.29 1l6.15 6.16a.61.61 0 0 0 1-.3ZM4.59 9.38L.5 13.5"/></svg>';
                pinBtn.style.cssText = `cursor:pointer; padding:2px; opacity:${isPinnedLeft?'1':'0.5'}; transition:opacity 0.2s;`;
                pinBtn.title = 'Pin Left';
                iconsContainer.appendChild(pinBtn);

                const isPinnedRight = column?.pinned === 'right';
                const pinRightBtn = document.createElement('span');
                pinRightBtn.className = `col-action-icon pin-right-icon ${isPinnedRight ? 'active' : ''}`;
                pinRightBtn.dataset.colId = node.id;
                pinRightBtn.dataset.iconType = 'pin-right';
                pinRightBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 14 14" style="transform:scaleX(-1)"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" d="M9.73 7.65L13 5.54A1 1 0 0 0 13.21 4L10 .79A1 1 0 0 0 8.46 1L6.3 4.23l-4.49 1a.6.6 0 0 0-.29 1l6.15 6.16a.61.61 0 0 0 1-.3ZM4.59 9.38L.5 13.5"/></svg>';
                pinRightBtn.style.cssText = `cursor:pointer; padding:2px; opacity:${isPinnedRight?'1':'0.5'}; transition:opacity 0.2s;`;
                pinRightBtn.title = 'Pin Right';
                iconsContainer.appendChild(pinRightBtn);
            }

            if (this.showLock && context.showLock !== false) {
                const isLocked = column?.pinned === 'right' && column?.getColDef && column.getColDef().suppressMovable;
                const lockBtn = document.createElement('span');
                lockBtn.className = `col-action-icon lock-icon ${isLocked ? 'active' : ''}`;
                lockBtn.dataset.colId = node.id;
                lockBtn.dataset.iconType = 'lock';
                lockBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24"><path fill="currentColor" d="M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h1V6q0-2.075 1.463-3.537T12 1t3.538 1.463T17 6v2h1q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22zm6-5q.825 0 1.413-.587T14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17M9 8h6V6q0-1.25-.875-2.125T12 3t-2.125.875T9 6z"/></svg>';
                lockBtn.style.cssText = `cursor:pointer; padding:2px; opacity:${isLocked?'1':'0.5'}; transition:opacity 0.2s;`;
                iconsContainer.appendChild(lockBtn);
            }

            if (this.showInfo && context.showInfo) {
                const infoBtn = document.createElement('span');
                infoBtn.className = 'col-action-icon info-icon';
                infoBtn.dataset.colId = node.id;
                infoBtn.dataset.iconType = 'info';
                infoBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24"><path fill="currentColor" d="M12 17q.425 0 .713-.288T13 16v-4q0-.425-.288-.712T12 11t-.712.288T11 12v4q0 .425.288.713T12 17m0-8q.425 0 .713-.288T13 8t-.288-.712T12 7t-.712.288T11 8t.288.713T12 9m0 13q-2.075 0-3.9-.788t-3.175-2.137T2.788 15.9T2 12t.788-3.9t2.137-3.175T8.1 2.788T12 2t3.9.788t3.175 2.137T21.213 8.1T22 12t-.788 3.9t-2.137 3.175t-3.175 2.138T12 22"/></svg>';
                infoBtn.style.cssText = 'cursor:pointer; padding:2px; opacity:0.7; transition:opacity 0.2s;';
                iconsContainer.appendChild(infoBtn);
            }

            element.appendChild(iconsContainer);
        }

        return element;
    }

    _bindDelegatedEvents() {
        if (!this.virtualContainer) return;

        const clickHandler = async (e) => {
            const expand = e.target.closest('.expand-btn');
            if (expand) {
                const row = expand.closest('.tree-node, .virtual-tree-node');
                if (!row) return;
                const nodeId = row.dataset.nodeId;
                this.toggleNode(nodeId);
                return;
            }

            const icon = e.target.closest('.col-action-icon');
            if (icon) {
                const colId = icon.dataset.colId;
                const resNode = this.getNodeFromCache(colId) || this.findNodeById(colId);
                if (resNode) {
                    await this.handleIconClick(icon, resNode);
                }
                return;
            }

            const row = e.target.closest('.tree-node, .virtual-tree-node');
            if (row) {
                if (e.target.closest('input.tree-checkbox')) return;
                const nodeId = row.dataset.nodeId;
                const node = this.getNodeFromCache(nodeId);
                if (!node) return;

                const checkbox = row.querySelector('input.tree-checkbox');
                const isRequired = !node.isGroup && this._getRequiredSet().has(node.id);
                if (!checkbox || checkbox.disabled || isRequired) return;

                const next = !checkbox.checked;
                checkbox.checked = next;
                this.handleNodeSelection(node, next);
                return;
            }
        };

        const changeHandler = (e) => {
            const checkbox = e.target.closest('input.tree-checkbox');
            if (!checkbox) return;
            const row = checkbox.closest('.tree-node, .virtual-tree-node');
            if (!row) return;
            const nodeId = row.dataset.nodeId;
            const node = this.getNodeFromCache(nodeId);
            if (!node) return;

            const isRequired = !node.isGroup && this._getRequiredSet().has(node.id);
            if (checkbox.disabled || isRequired) return;

            this.handleNodeSelection(node, checkbox.checked);
        };

        // Drag-to-reorder in filter selected mode
        let dragSrcIndex = null;

        const dragStartHandler = (e) => {
            if (!this._filterSelected) return;
            const row = e.target.closest('.tree-node, .virtual-tree-node');
            if (!row) return;
            dragSrcIndex = parseInt(row.dataset.index, 10);
            row.style.opacity = '0.4';
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', row.dataset.nodeId);
        };

        const dragOverHandler = (e) => {
            if (!this._filterSelected || dragSrcIndex == null) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            const row = e.target.closest('.tree-node, .virtual-tree-node');
            if (row) {
                // Clear previous indicators
                this.virtualContainer.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(el => {
                    el.classList.remove('drag-over-top', 'drag-over-bottom');
                });
                const rect = row.getBoundingClientRect();
                const midY = rect.top + rect.height / 2;
                if (e.clientY < midY) {
                    row.classList.add('drag-over-top');
                } else {
                    row.classList.add('drag-over-bottom');
                }
            }
        };

        const dragLeaveHandler = (e) => {
            const row = e.target.closest('.tree-node, .virtual-tree-node');
            if (row) {
                row.classList.remove('drag-over-top', 'drag-over-bottom');
            }
        };

        const dropHandler = (e) => {
            e.preventDefault();
            if (!this._filterSelected || dragSrcIndex == null) return;

            // Clear all drag indicators
            this.virtualContainer.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(el => {
                el.style.opacity = '';
                el.classList.remove('drag-over-top', 'drag-over-bottom');
            });

            const row = e.target.closest('.tree-node, .virtual-tree-node');
            if (!row) { dragSrcIndex = null; return; }

            let dropIndex = parseInt(row.dataset.index, 10);
            if (isNaN(dropIndex) || dropIndex === dragSrcIndex) { dragSrcIndex = null; return; }

            // Determine if dropping above or below based on mouse position
            const rect = row.getBoundingClientRect();
            const midY = rect.top + rect.height / 2;
            if (e.clientY > midY) dropIndex += 1;

            // Reorder flattenedNodes
            const nodes = this.flattenedNodes;
            const [moved] = nodes.splice(dragSrcIndex, 1);
            const insertAt = dropIndex > dragSrcIndex ? dropIndex - 1 : dropIndex;
            nodes.splice(insertAt, 0, moved);

            // Apply the new order to AG-Grid
            this._applyDragReorder(nodes);

            dragSrcIndex = null;
            this.renderAllNodes();
        };

        const dragEndHandler = (e) => {
            // Reset opacity on the dragged element
            const row = e.target.closest('.tree-node, .virtual-tree-node');
            if (row) row.style.opacity = '';
            this.virtualContainer.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(el => {
                el.classList.remove('drag-over-top', 'drag-over-bottom');
            });
            dragSrcIndex = null;
        };

        this._delegatedHandlers.click = clickHandler;
        this._delegatedHandlers.change = changeHandler;
        this._delegatedHandlers.dragstart = dragStartHandler;
        this._delegatedHandlers.dragover = dragOverHandler;
        this._delegatedHandlers.dragleave = dragLeaveHandler;
        this._delegatedHandlers.drop = dropHandler;
        this._delegatedHandlers.dragend = dragEndHandler;

        this.virtualContainer.addEventListener('click', clickHandler);
        this.virtualContainer.addEventListener('change', changeHandler);
        this.virtualContainer.addEventListener('dragstart', dragStartHandler);
        this.virtualContainer.addEventListener('dragover', dragOverHandler);
        this.virtualContainer.addEventListener('dragleave', dragLeaveHandler);
        this.virtualContainer.addEventListener('drop', dropHandler);
        this.virtualContainer.addEventListener('dragend', dragEndHandler);
    }

    _unbindDelegatedEvents() {
        if (!this.virtualContainer) return;
        const events = ['click', 'change', 'dragstart', 'dragover', 'dragleave', 'drop', 'dragend'];
        events.forEach(evt => {
            if (this._delegatedHandlers[evt]) {
                this.virtualContainer.removeEventListener(evt, this._delegatedHandlers[evt]);
            }
        });
        this._delegatedHandlers = {};
    }

    /**
     * Cleanup element and its listeners
     */
    cleanupElement(element) {
        // Remove all event listeners using stored references
        const listeners = element._eventListeners;
        if (listeners) {
            listeners.forEach(({ target, type, handler }) => {
                target.removeEventListener(type, handler);
            });
            delete element._eventListeners;
        }
    }

    /**
     * Cleanup entire element cache
     */
    cleanupElementCache() {
        if (this.elementCache) {
            this.elementCache.forEach(element => {
                this.cleanupElement(element);
            });
            this.elementCache.clear();
        }
    }

    /**
     * Create node element
     */
    createNodeElement(node, virtualItem) {
        const element = document.createElement('div');
        element.className = 'virtual-tree-node';
        element.dataset.nodeId = node.id;
        element.dataset.index = virtualItem.index;

        // Add draggable for non-group nodes
        if (!node.isGroup && this.config.enableDragDrop) {
            element.draggable = true;
        }

        // Add hover state
        if (node.id === this.hoveredNodeId) {
            element.classList.add('hovered');
        }

        // Add custom name indicator
        if (node.hasCustomName) {
            element.classList.add('has-custom-name');
        }

        element.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: ${virtualItem.size}px;
        transform: translateY(${virtualItem.start}px);
        display: flex;
        align-items: center;
        padding: 4px 8px;
        border-bottom: 1px solid var(--ag-border-color, #eee);
        cursor: pointer;
        user-select: none;
        box-sizing: border-box;
        transition: background-color 0.2s;
    `;

        // Hover effect
        if (node.id === this.hoveredNodeId) {
            element.style.backgroundColor = 'var(--ag-row-hover-color, rgba(0,0,0,0.05))';
        }

        // Create content
        this.populateNodeElement(element, node);

        // No individual event handlers - using delegation instead

        return element;
    }

    /**
     * Update existing node element
     */
    updateNodeElement(element, node, virtualItem) {
        // Update hover state
        if (node.id === this.hoveredNodeId) {
            element.classList.add('hovered');
            element.style.backgroundColor = 'var(--ag-row-hover-color, rgba(0,0,0,0.05))';
        } else {
            element.classList.remove('hovered');
            element.style.backgroundColor = '';
        }

        // Update checkbox state
        const checkbox = element.querySelector('.tree-checkbox');
        if (checkbox) {
            checkbox.checked = this.selectedNodes.has(node.id);
            checkbox.indeterminate = this.indeterminateNodes.has(node.id);
        }

        // Update expand/collapse icon
        const expandBtn = element.querySelector('.expand-btn');
        if (expandBtn && node.isGroup && node.children) {
            expandBtn.innerHTML = this.expandedNodes.has(node.id) ? '▼' : '▶';
        }

        // Update icon states
        this.updateIconStates(element, node);
    }

    async setDefaultState(name, makeDefault = true) {
        const state = this.presets.get(name);
        if (!state) { this.context?.page?.toastManager?.().error?.(`View "${name}" not found.`); return false; }
        if (state.metaData?.isTemporary) { this.context?.page?.toastManager?.().warning?.(`Cannot set temporary view "${name}" as default.`); return false; }

        if (!makeDefault) {
            const nonTemp = Array.from(this.presets.values()).filter(p => !p.metaData?.isTemporary);
            if (nonTemp.length === 1 && nonTemp[0].name === name) {
                this.context?.page?.toastManager?.().warning?.(`Cannot remove default from the only view.`); return false;
            }
            state.metaData.isDefault = false;
            const globalPreset = Array.from(this.presets.values()).find(p => p.metaData?.isGlobal);
            if (globalPreset) {
                globalPreset.metaData.isDefault = true;
                await this._loadPreset(globalPreset.name);
                this.context?.page?.toastManager?.().success?.(`"${name}" is no longer default. "${globalPreset.name}" is now default.`);
            }
        } else {
            this.presets.forEach(p => { if (p.metaData) p.metaData.isDefault = false; });
            state.metaData.isDefault = true;
            await this._loadPreset(name);
            this.context?.page?.toastManager?.().success?.(`"${name}" is now the default view.`);
        }
        this.savePresetsToCache();
        this._updateButtonStates();
        return true;
    }

    async renameState(oldName, newName, newDescription = '') {
        const state = this.presets.get(oldName);

        if (!state) {
            this.context.page.toastManager().error(`State "${oldName}" not found.`, 'Error');
            return false;
        }

        if (state.metaData?.isGlobal || !state.metaData?.isMutable) {
            this.context.page.toastManager().warning(`View "${oldName}" cannot be modified.`, 'Warning');
            return false;
        }

        // Check if new name already exists
        if (oldName !== newName && this.presets.has(newName)) {
            this.context.page.toastManager().error(`A view named "${newName}" already exists.`, 'Error');
            return false;
        }

        // Update the state
        state.name = newName;
        state.metaData.lastModified = new Date().toISOString();
        if (newDescription !== undefined) {
            state.metaData.description = newDescription;
        }

        // If name changed, update map key
        if (oldName !== newName) {
            this.presets.delete(oldName);
            this.presets.set(newName, state);

            // Update active preset name if necessary
            if (this.activePresetName === oldName) {
                this.activePresetName = newName;
            }
        } else {
            this.presets.set(oldName, state);
        }

        this.savePresetsToCache();
        this.context.page.toastManager().success(`View renamed successfully.`, 'Success');
        return true;
    }

    async deleteState(name) {
        const state = this.presets.get(name);

        if (!state) {
            this.context.page.toastManager().error(`State "${name}" not found.`, 'Error');
            return false;
        }

        if (state.metaData?.isGlobal || !state.metaData?.isMutable) {
            this.context.page.toastManager().warning(`View "${name}" cannot be deleted.`, 'Warning');
            return false;
        }

        const wasDefault = state.metaData?.isDefault;

        // Delete the state
        this.presets.delete(name);

        // If it was the default, set global as default
        if (wasDefault) {
            const globalDefault = this.globalPresets.find(p => p.metaData?.isDefault);
            if (globalDefault) {
                // Re-add global default to presets
                this.presets.set(globalDefault.name, globalDefault);
                await this._loadPreset(globalDefault.name);
            }
        }

        // If it was the active preset, switch to default
        if (this.activePresetName === name) {
            const defaultPreset = Array.from(this.presets.values())
                                        .find(p => p.metaData?.isDefault);
            if (defaultPreset) {
                await this._loadPreset(defaultPreset.name);
            }
        }

        this.savePresetsToCache();
        this.context.page.toastManager().success(`View "${name}" deleted.`, 'Success');
        return true;
    }

    /**
     * Populate node element with content
     */
    populateNodeElement(element, node) {
        // Indentation
        const indent = document.createElement('div');
        indent.style.width = `${node.level * 20}px`;
        indent.style.flexShrink = '0';

        // Expand/collapse button
        const expandBtn = document.createElement('div');
        expandBtn.className = 'expand-btn';
        if (node.isGroup && node.children && node.children.length > 0) {
            expandBtn.innerHTML = this.expandedNodes.has(node.id) ? '▼' : '▶';
            expandBtn.style.cssText = `
            width: 16px;
            height: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            cursor: pointer;
            flex-shrink: 0;
            margin-right: 4px;
            pointer-events: auto;
        `;
        } else {
            expandBtn.style.cssText = 'width: 16px; flex-shrink: 0; margin-right: 4px;';
        }

        // Checkbox - make it more clickable
        const checkboxWrapper = document.createElement('div');
        checkboxWrapper.style.cssText = `
        display: flex;
        align-items: center;
        margin-right: 8px;
        flex-shrink: 0;
        pointer-events: auto;
    `;

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'tree-checkbox';
        checkbox.checked = this.selectedNodes.has(node.id);
        checkbox.indeterminate = this.indeterminateNodes.has(node.id);
        checkbox.style.cssText = `
        cursor: pointer;
        margin: 0;
        pointer-events: auto;
    `;
        const isRequired = !node.isGroup && this._getRequiredSet().has(node.id);
        if (isRequired) {
            checkbox.disabled = true;
            checkbox.title = 'Required column';
        }
        checkboxWrapper.appendChild(checkbox);

        // Text content with highlighting
        const textContent = document.createElement('div');
        textContent.className = 'tree-text';
        textContent.style.cssText = 'flex: 1; display: flex; align-items: center; pointer-events: none;';

        const displayName = this.getNodeDisplayName(node);

        // Apply search highlighting if needed
        if (this.searchMatches.has(node.id)) {
            textContent.innerHTML = this.highlightSearchMatches(displayName, node.id);
        } else {
            textContent.textContent = displayName;
        }

        // Add group indicator
        if (node.isGroup) {
            textContent.innerHTML = `<span class="tree-parent">[${textContent.innerHTML}]</span>`;
        }

        // Add custom name indicator
        if (node.hasCustomName) {
            const customIndicator = document.createElement('span');
            customIndicator.className = 'custom-name-indicator';
            customIndicator.innerHTML = '✏';
            customIndicator.style.cssText = 'margin-left: 4px; color: var(--ag-secondary-foreground-color, #666); font-size: 10px;';
            customIndicator.title = 'Custom name';
            textContent.appendChild(customIndicator);
        }

        // Action icons for columns
        if (!node.isGroup) {
            const iconsContainer = this.createActionIcons(node);
            iconsContainer.style.pointerEvents = 'auto';
            textContent.appendChild(iconsContainer);
        }

        // Assemble element
        element.appendChild(indent);
        element.appendChild(expandBtn);
        element.appendChild(checkboxWrapper);
        element.appendChild(textContent);
    }

    getNodeDisplayName(node) {
        return node.text || node.originalText || node.id;
    }

    /**
     * Highlight search matches in text
     */
    _escapeHtml(str) {
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    highlightSearchMatches(text, nodeId) {
        const matches = this.searchMatches.get(nodeId);
        if (!matches || matches.length === 0) return this._escapeHtml(text);

        // Find the best match (usually headerName or customName)
        const bestMatch = matches.find(m =>
            m.key === 'customName' || m.key === 'headerName'
        ) || matches[0];

        if (!bestMatch || !bestMatch.indices) return this._escapeHtml(text);

        let result = '';
        let lastIndex = 0;

        bestMatch.indices.forEach(([start, end]) => {
            result += this._escapeHtml(text.substring(lastIndex, start));
            result += `<mark class="search-highlight">${this._escapeHtml(text.substring(start, end + 1))}</mark>`;
            lastIndex = end + 1;
        });

        result += this._escapeHtml(text.substring(lastIndex));
        return result;
    }

    /**
     * Create action icons for column nodes
     */
    createActionIcons(node) {
        const container = document.createElement('span');
        container.className = 'col-actions';
        container.style.cssText = 'margin-left: auto; display: flex; gap: 4px;';

        const colDef = node.colDef;
        const context = colDef?.context || {};
        const column = this.api?.getColumn(node.id);

        // Pin left icon
        if (context.showPin !== false) {
            const isPinnedLeft = column?.pinned === 'left';
            const pinIcon = this.createIcon('pin', isPinnedLeft, node.id);
            pinIcon.title = 'Pin Left';
            container.appendChild(pinIcon);

            const isPinnedRight = column?.pinned === 'right';
            const pinRightIcon = this.createIcon('pin-right', isPinnedRight, node.id);
            pinRightIcon.title = 'Pin Right';
            container.appendChild(pinRightIcon);
        }

        // Lock icon
        if (context.showLock !== false) {
            const isLocked = column?.pinned === 'right' && column?.getColDef().suppressMovable;
            const lockIcon = this.createIcon('lock', isLocked, node.id);
            container.appendChild(lockIcon);
        }

        // Info icon
        if (context.showInfo && context.infoText) {
            const infoIcon = this.createIcon('info', false, node.id, context.infoText);
            container.appendChild(infoIcon);
        }

        return container;
    }

    /**
     * Create an action icon
     */
    createIcon(type, isActive, nodeId, extraData) {
        const icon = document.createElement('span');
        icon.className = `col-action-icon ${type}-icon ${isActive ? 'active' : ''}`;
        icon.dataset.colId = nodeId;
        icon.dataset.iconType = type;
        if (extraData) {
            icon.dataset.extraData = extraData;
        }

        icon.style.cssText = `
            cursor: pointer;
            padding: 2px;
            opacity: ${isActive ? '1' : '0.5'};
            transition: opacity 0.2s;
        `;

        // Icon SVGs
        const icons = {
            pin: '<svg width="13" height="13" viewBox="0 0 14 14"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" d="M9.73 7.65L13 5.54A1 1 0 0 0 13.21 4L10 .79A1 1 0 0 0 8.46 1L6.3 4.23l-4.49 1a.6.6 0 0 0-.29 1l6.15 6.16a.61.61 0 0 0 1-.3ZM4.59 9.38L.5 13.5"/></svg>',
            'pin-right': '<svg width="13" height="13" viewBox="0 0 14 14" style="transform:scaleX(-1)"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" d="M9.73 7.65L13 5.54A1 1 0 0 0 13.21 4L10 .79A1 1 0 0 0 8.46 1L6.3 4.23l-4.49 1a.6.6 0 0 0-.29 1l6.15 6.16a.61.61 0 0 0 1-.3ZM4.59 9.38L.5 13.5"/></svg>',
            lock: '<svg width="13" height="13" viewBox="0 0 24 24"><path fill="currentColor" d="M6 22q-.825 0-1.412-.587T4 20V10q0-.825.588-1.412T6 8h1V6q0-2.075 1.463-3.537T12 1t3.538 1.463T17 6v2h1q.825 0 1.413.588T20 10v10q0 .825-.587 1.413T18 22zm6-5q.825 0 1.413-.587T14 15t-.587-1.412T12 13t-1.412.588T10 15t.588 1.413T12 17M9 8h6V6q0-1.25-.875-2.125T12 3t-2.125.875T9 6z"/></svg>',
            info: '<svg width="13" height="13" viewBox="0 0 24 24"><path fill="currentColor" d="M12 17q.425 0 .713-.288T13 16v-4q0-.425-.288-.712T12 11t-.712.288T11 12v4q0 .425.288.713T12 17m0-8q.425 0 .713-.288T13 8t-.288-.712T12 7t-.712.288T11 8t.288.713T12 9m0 13q-2.075 0-3.9-.788t-3.175-2.137T2.788 15.9T2 12t.788-3.9t2.137-3.175T8.1 2.788T12 2t3.9.788t3.175 2.137T21.213 8.1T22 12t-.788 3.9t-2.137 3.175t-3.175 2.138T12 22"/></svg>'
        };

        icon.innerHTML = icons[type] || '';
        icon.title = type.charAt(0).toUpperCase() + type.slice(1);

        return icon;
    }

    /**
     * Update icon states based on column state
     */
    updateIconStates(element, node) {
        if (node.isGroup || !this.api) return;

        try {
            const column = this.api.getColumn(node.id);
            if (!column) return;

            // Update pin left icon
            const pinIcon = element.querySelector('.pin-icon');
            if (pinIcon) {
                const isPinnedLeft = column?.pinned === 'left';
                pinIcon.classList.toggle('active', isPinnedLeft);
                pinIcon.style.opacity = isPinnedLeft ? '1' : '0.5';
            }

            // Update pin right icon
            const pinRightIcon = element.querySelector('.pin-right-icon');
            if (pinRightIcon) {
                const isPinnedRight = column?.pinned === 'right';
                pinRightIcon.classList.toggle('active', isPinnedRight);
                pinRightIcon.style.opacity = isPinnedRight ? '1' : '0.5';
            }

            // Update lock icon
            const lockIcon = element.querySelector('.lock-icon');
            if (lockIcon && column?.pinned && column.getColDef) {
                const isLocked = column?.pinned === 'right' && column.getColDef().suppressMovable;
                lockIcon.classList.toggle('active', isLocked);
                lockIcon.style.opacity = isLocked ? '1' : '0.5';
            }
        } catch (error) {
            console.warn(`Error updating icon states for ${node.id}:`, error);
        }
    }


    // ==================== Node Actions ====================

    toggleNode(nodeId) {
        if (this.expandedNodes.has(nodeId)) {
            this.expandedNodes.delete(nodeId);
        } else {
            this.expandedNodes.add(nodeId);
        }

        this.saveExpandedState();
        this.updateFlattenedNodes();
        this.renderAllNodes(); // Re-render all nodes

    }

    _collectLeafIds(groupNode) {
        const out = [];
        const dfs = (n) => {
            if (!n) return;
            if (n.isGroup && n.children) { for (let i = 0; i < n.children.length; i++) dfs(n.children[i]); }
            else if (!n.isGroup) out.push(n.id);
        };
        dfs(groupNode);
        return out;
    }

    _filterPresentColumns(ids) {
        if (!this.api) return ids.slice();
        const out = [];
        for (let i = 0; i < ids.length; i++) if (this.api.getColumn(ids[i]) !== null) out.push(ids[i]);
        return out;
    }

    handleNodeSelection(node, wantSelected) {
        if (!node) return;
        this._pushUndo();
        if (node.isGroup) {
            // Fix [1]: batch apply visibility for descendants and actually append to grid
            const leaves = this._collectLeafIds(node);
            const valid = this._filterPresentColumns(leaves);
            if (!valid.length) {
                // Try to be forgiving: AG Grid may not have defs yet; still update sets
            }
            if (wantSelected) {
                for (let i = 0; i < leaves.length; i++) { this.selectedNodes.add(leaves[i]); this.indeterminateNodes.delete(leaves[i]); }
            } else {
                for (let i = 0; i < leaves.length; i++) { this.selectedNodes.delete(leaves[i]); this.indeterminateNodes.delete(leaves[i]); }
            }

            if (this.api && valid.length) {
                this.api.setColumnsVisible(valid, wantSelected === true);
                if (wantSelected) this._appendColumnsToEnd(valid);
            }
        } else {
            if (wantSelected) { this.selectedNodes.add(node.id); this.indeterminateNodes.delete(node.id); }
            else { this.selectedNodes.delete(node.id); this.indeterminateNodes.delete(node.id); }

            if (this.api && this.api.getColumn(node.id) !== null) {
                this.api.setColumnsVisible([node.id], !!wantSelected);
                if (wantSelected) this._appendColumnsToEnd([node.id]);
            }
        }

        this._enforceRequiredSelections({ updateGrid: true });
        this.updateAllParentStates();
        this.renderAllNodes();
        this.hasUnsavedChanges = true;
        this._updateButtonStates();
    }

    _pushUndo() {
        const state = this._getGridState();
        if (!state) return;
        this._undoStack.push(state);
        if (this._undoStack.length > this._undoMaxSize) {
            this._undoStack.shift();
        }
        this._updateUndoBtn();
    }

    async _handleUndo() {
        if (this._undoStack.length === 0) return;
        const prev = this._undoStack.pop();
        await this._applyGridState(prev);
        this.initializeSelectionStates();
        this.renderAllNodes();
        this._updateUndoBtn();
    }

    _updateUndoBtn() {
        if (this.undoBtn) {
            this.undoBtn.disabled = this._undoStack.length === 0;
            this.undoBtn.setAttribute('data-tooltip',
                this._undoStack.length > 0
                    ? `Undo Last Action (${this._undoStack.length})`
                    : 'Undo Last Action'
            );
        }
    }

    _appendColumnsToEnd(ids) {
        // todo
    }

    getDescendantColumns(node) {
        const columns = [];
        const traverse = (currentNode) => {
            if (!currentNode.isGroup) {
                columns.push(currentNode.id);
            } else if (currentNode.children) {
                currentNode.children.forEach(child => traverse(child));
            }
        };
        traverse(node);
        return columns;
    }

    updateSelectionStates(node, isSelected) {
        // Batch state updates
        const statesToUpdate = {
            selected: [],
            deselected: [],
            indeterminate: []
        };

        // Collect all state changes in single traversal
        this.collectSelectionStates(node, isSelected, statesToUpdate);

        // Apply all state changes at once
        statesToUpdate.selected.forEach(id => {
            this.selectedNodes.add(id);
            this.indeterminateNodes.delete(id);
        });

        statesToUpdate.deselected.forEach(id => {
            this.selectedNodes.delete(id);
            this.indeterminateNodes.delete(id);
        });

        statesToUpdate.indeterminate.forEach(id => {
            this.selectedNodes.delete(id);
            this.indeterminateNodes.add(id);
        });

        if (!isSelected) {
            this.indeterminateNodes.delete(node.id);
            this.selectedNodes.delete(node.id);
        }
    }

    isRequired(node) {
        return !node.isGroup && this._getRequiredSet().has(node.id);
    }

    /**
     * Collect selection states in single traversal
     */
    collectSelectionStates(node, isSelected, statesToUpdate, _processQueue) {
        // Process current node and children
        const isRequired = this.isRequired(node)

        if (isRequired) {
            statesToUpdate.selected.push(node.id);
        } else {
            if (isSelected) {
                statesToUpdate.selected.push(node.id);
            } else {
                statesToUpdate.deselected.push(node.id);
            }
        }

        // Record process
        if (_processQueue == null) _processQueue = new Set();
        // Process children
        if (node.children) {
            // Mutate in-place instead of O(n) copy-on-recurse
            for (let i = 0; i < node.children.length; i++) _processQueue.add(node.children[i].id);
            node.children.forEach(child => {
                this.collectSelectionStates(child, isSelected, statesToUpdate, _processQueue);
            });
        }

        // Process parent (only once per update cycle)
        if (node.parent && !statesToUpdate._parentProcessed) {
            statesToUpdate._parentProcessed = new Set();
        }

        if (node.parent && !statesToUpdate._parentProcessed.has(node.parent)) {
            const parent = this.getNodeFromCache(node.parent);
            const childNames = parent?.childNames;
            const hasOverlap = childNames && _processQueue.size > 0 && (function() {
                for (const id of _processQueue) { if (childNames.has(id)) return true; }
                return false;
            })();
            if (parent && !hasOverlap) {
                statesToUpdate._parentProcessed.add(node.parent);
                this.updateParentStateEfficient(parent, statesToUpdate);
            }
        }
    }

    /**
     * Update parent state efficiently
     */
    updateParentStateEfficient(parent, statesToUpdate) {
        if (!parent.children) return;

        this.indeterminateNodes.delete(parent.id);
        this.selectedNodes.delete(parent.id);

        // Build O(1) lookup Sets from accumulated arrays to avoid O(n*m) .includes()
        if (!statesToUpdate._selectedSet) statesToUpdate._selectedSet = new Set(statesToUpdate.selected);
        if (!statesToUpdate._indeterminateSet) statesToUpdate._indeterminateSet = new Set(statesToUpdate.indeterminate);
        const selectedSet = statesToUpdate._selectedSet;
        const indeterminateSet = statesToUpdate._indeterminateSet;

        let selectedCount = 0;
        let indeterminateCount = 0;

        parent.children.forEach(child => {
            const isRequired = !child.isGroup && this._getRequiredSet().has(child.id);

            if (isRequired || (selectedSet.has(child.id) || this.selectedNodes.has(child.id))) {
                selectedCount++;
            } else if (indeterminateSet.has(child.id) ||
                this.indeterminateNodes.has(child.id)) {
                indeterminateCount++;
            }
        });

        const isRequired = !parent.isGroup && this._getRequiredSet().has(parent.id);

        if (selectedCount === parent.children.length) {
            statesToUpdate.selected.push(parent.id);
            if (statesToUpdate._selectedSet) statesToUpdate._selectedSet.add(parent.id);
        } else if (selectedCount > 0 || indeterminateCount > 0) {
            statesToUpdate.indeterminate.push(parent.id);
            if (statesToUpdate._indeterminateSet) statesToUpdate._indeterminateSet.add(parent.id);
        } else {
            if (isRequired) {
                statesToUpdate.indeterminate.push(parent.id);
                if (statesToUpdate._indeterminateSet) statesToUpdate._indeterminateSet.add(parent.id);
            } else {
                statesToUpdate.deselected.push(parent.id);
            }
        }

        // Recursively update grandparent
        if (parent.parent && !statesToUpdate._parentProcessed.has(parent.parent)) {
            statesToUpdate._parentProcessed.add(parent.parent);
            const grandparent = this.getNodeFromCache(parent.parent);
            if (grandparent) {
                this.updateParentStateEfficient(grandparent, statesToUpdate);
            }
        }
    }

    /**
     * Get node from cache or find it
     */
    getNodeFromCache(nodeId) {
        // Initialize node cache if needed
        if (!this.nodeCache) {
            this.nodeCache = new Map();
            this.buildNodeCache(this.originalTreeData);
        }

        return this.nodeCache.get(nodeId);
    }

    /**
     * Build node cache for fast lookups
     */
    buildNodeCache(nodes) {
        nodes.forEach(node => {
            this.nodeCache.set(node.id, node);
            if (node.children) {
                this.buildNodeCache(node.children);
            }
        });
    }

    /**
     * Handle icon clicks
     */
    async handleIconClick(iconElement, node) {
        const iconType = iconElement.dataset.iconType;
        const colId = iconElement.dataset.colId;
        const column = this.api?.getColumn(colId);

        if (!column) return;

        switch (iconType) {
            case 'pin':
                this.togglePin(column, iconElement);
                break;
            case 'pin-right':
                this.togglePinRight(column, iconElement);
                break;
            case 'lock':
                this.toggleLock(column, iconElement);
                break;
            case 'info':
                await this.showInfoModal(node, column);
                break;
        }
    }

    togglePin(column, iconElement) {
        if (!column) return;

        const currentPin = column?.pinned;
        const newPin = currentPin === 'left' ? null : 'left';

        this.api.setColumnsPinned([column.getColId()], newPin);

        if (newPin) {
            this.pinnedColumns.add(column.getColId());
            iconElement.style.opacity = '1';
        } else {
            this.pinnedColumns.delete(column.getColId());
            iconElement.style.opacity = '0.5';
        }

        // Update sibling pin-right icon if present
        const row = iconElement.closest('.tree-node, .virtual-tree-node');
        const pinRightIcon = row?.querySelector('.pin-right-icon');
        if (pinRightIcon && newPin === 'left') {
            pinRightIcon.classList.remove('active');
            pinRightIcon.style.opacity = '0.5';
        }
    }

    togglePinRight(column, iconElement) {
        if (!column) return;

        const currentPin = column?.pinned;
        const newPin = currentPin === 'right' ? null : 'right';

        this.api.setColumnsPinned([column.getColId()], newPin);

        if (newPin) {
            iconElement.classList.add('active');
            iconElement.style.opacity = '1';
        } else {
            iconElement.classList.remove('active');
            iconElement.style.opacity = '0.5';
        }

        // Update sibling pin-left icon if present
        const row = iconElement.closest('.tree-node, .virtual-tree-node');
        const pinIcon = row?.querySelector('.pin-icon');
        if (pinIcon && newPin === 'right') {
            pinIcon.classList.remove('active');
            pinIcon.style.opacity = '0.5';
            this.pinnedColumns.delete(column.getColId());
        }
    }

    /**
     * Toggle lock with icon update
     */
    toggleLock(column, iconElement) {
        if (!column) return;

        const colId = column.getColId();
        const isLocked = column?.pinned === 'right' && column.getColDef().suppressMovable;

        if (isLocked) {
            this.api.applyColumnState({
                state: [{
                    colId: colId,
                    pinned: null,
                    lockPosition: false,
                    suppressMovable: false
                }]
            });
            this.lockedColumns.delete(colId);
            iconElement.style.opacity = '0.5';
        } else {
            this.api.applyColumnState({
                state: [{
                    colId: colId,
                    pinned: 'right',
                    lockPosition: true,
                    suppressMovable: true
                }]
            });
            this.lockedColumns.add(colId);
            iconElement.style.opacity = '1';
        }

        this.api.refreshHeader();
    }

    /**
     * Show info modal with custom name option
     */
    async showInfoModal(node, column) {
        const colDef = column?.getColDef();
        const context = colDef?.context || {};
        const headerName = colDef?.context?.menuNameOverride || colDef?.headerName || node.id;
        const fieldId = colDef?.field || 'N/A';
        const customName = this.customNames[node.id] || '';
        this.modalOpen = true;

        try {
            const result = await this.modalManager.showCustom({
                id: 'column-info-modal',
                title: `Column Info: ${headerName}`,
                modalClass: 'modal-md',
                setupContent: (contentArea, dialog, closeDialog) => {
                    // API Key
                    const apiKeyDiv = document.createElement('div');
                    apiKeyDiv.className = 'column-modal-subheader';
                    apiKeyDiv.textContent = `API Key: ${fieldId}`;
                    apiKeyDiv.style.cssText = 'margin-bottom: 16px; font-weight: bold;';

                    // Info text
                    const infoDiv = document.createElement('div');
                    infoDiv.style.cssText = 'margin-bottom: 16px; line-height: 1.6;';
                    infoDiv.textContent = context.infoText || 'No additional information available.';

                    // Custom name section
                    const customNameSection = document.createElement('div');
                    customNameSection.style.cssText = 'margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--ag-border-color, #ddd);';

                    const customNameLabel = document.createElement('label');
                    customNameLabel.textContent = 'Custom Name:';
                    customNameLabel.style.cssText = 'display: block; margin-bottom: 8px; font-weight: bold;';

                    const customNameInput = document.createElement('input');
                    customNameInput.type = 'text';
                    customNameInput.value = customName;
                    customNameInput.placeholder = 'Enter custom name...';
                    customNameInput.style.cssText = `
                        width: 100%;
                        padding: 6px;
                        border: 1px solid var(--ag-border-color, #ddd);
                        border-radius: 4px;
                        margin-bottom: 8px;
                    `;

                    const customNameHelp = document.createElement('div');
                    customNameHelp.style.cssText = 'font-size: 12px; color: var(--ag-secondary-foreground-color, #666);';
                    customNameHelp.textContent = 'This custom name will be shown in the tree and column header.';

                    customNameSection.appendChild(customNameLabel);
                    customNameSection.appendChild(customNameHelp);
                    customNameSection.appendChild(customNameInput);

                    // Buttons
                    const buttonContainer = document.createElement('div');
                    buttonContainer.className = 'modal-action';
                    buttonContainer.style.cssText = 'display: flex; justify-content: flex-end; gap: 8px; margin-top: 20px;';



                    const clearButton = document.createElement('button');
                    clearButton.className = 'btn btn-sm btn-warning';
                    clearButton.textContent = 'Clear Custom Name';
                    clearButton.addEventListener('click', () => {
                        customNameInput.value = '';
                    });

                    const cancelButton = document.createElement('button');
                    cancelButton.className = 'btn btn-sm';
                    cancelButton.textContent = 'Cancel';
                    cancelButton.addEventListener('click', () => {
                        closeDialog('cancel'); // Use closeDialog instead of dialog.close
                    });

                    const saveButton = document.createElement('button');
                    saveButton.className = 'btn btn-sm btn-primary';
                    saveButton.textContent = 'Save';
                    saveButton.addEventListener('click', () => {
                        const newCustomName = customNameInput.value.trim();
                        closeDialog({ action: 'save', customName: newCustomName }); // Pass object
                    });

                    buttonContainer.appendChild(cancelButton);
                    if (customName) {
                        buttonContainer.appendChild(clearButton);
                    }
                    buttonContainer.appendChild(saveButton);

                    // Assemble content
                    contentArea.appendChild(apiKeyDiv);
                    contentArea.appendChild(infoDiv);
                    if (this.config.enableCustomNames) {
                        contentArea.appendChild(customNameSection);
                    }
                    contentArea.appendChild(buttonContainer);
                }
            });

            // Handle result
            if (result && result.action === 'save') {
                this.setCustomName(node.id, result.customName);
            }
        } catch (error) {
            console.error('Error showing info modal:', error);
        } finally {
            this.modalOpen = false;
        }
    }

    // ==================== Custom Names Management ====================

    applyCustomNamesToGrid() {
        if (!this.api || !this.customNames) return;

        let hasChanges = false;

        Object.entries(this.customNames).forEach(([colId, customName]) => {
            try {
                const column = this.api.getColumn(colId);
                if (column) {
                    const colDef = column.getColDef();
                    // Store original header name if not already stored
                    if (!colDef.originalHeaderName) {
                        colDef.originalHeaderName = colDef.headerName || colDef.field;
                    }
                    // Only update if different
                    if (colDef.headerName !== customName) {
                        colDef.headerName = customName;
                        hasChanges = true;
                    }
                }
            } catch (error) {
                console.warn(`Could not apply custom name to column ${colId}:`, error);
            }
        });

        // Only refresh if we made changes
        if (hasChanges) {
            this.api.refreshHeader();
        }
    }

    setCustomName(nodeId, customName) {
        if (customName) {
            this.customNames[nodeId] = customName;
        } else {
            delete this.customNames[nodeId];
        }

        // Save to localStorage
        this.saveCustomNames();

        // Update node in tree
        const node = this.findNodeById(nodeId);
        if (node) {
            if (customName) {
                node.text = customName;
                node.hasCustomName = true;
            } else {
                node.text = node.originalText;
                node.hasCustomName = false;
            }
        }

        // Update column header in grid
        const column = this.api?.getColumn(nodeId);
        if (column) {
            const colDef = column.getColDef();
            if (customName) {
                // Store original if not already stored
                if (!colDef.originalHeaderName) {
                    colDef.originalHeaderName = colDef.headerName || colDef.field;
                }
                colDef.headerName = customName;
            } else {
                // Restore original name
                colDef.headerName = colDef.originalHeaderName || colDef.field;
            }
            this.api.refreshHeader();
        }

        // Rebuild search index
        this.buildFuseIndex();

        // Re-render tree
        this.updateFlattenedNodes();
        this.renderAllNodes();
    }

    /**
     * Load custom names from localStorage
     */
    loadCustomNames() {
        try {
            const stored = localStorage.getItem(`treeColumnChooser_customNames_${this.gridName}`);
            return stored ? JSON.parse(stored) : {};
        } catch (error) {
            console.error('Error loading custom names:', error);
            return {};
        }
    }


    saveCustomNames() {
        try {
            localStorage.setItem(
                `treeColumnChooser_customNames_${this.gridName}`,
                JSON.stringify(this.customNames)
            );
        } catch (error) {
            console.error('Error saving custom names:', error);
        }
    }

    // ==================== Preset Management ====================

    /**
     * Load column presets
     */
    async loadPresets() {
        try {
            const stored = localStorage.getItem(`treeColumnChooser_presets_${this.gridName}`);
            this.presets = stored ? JSON.parse(stored) : this.getDefaultPresets();
        } catch (error) {
            console.error('Error loading presets:', error);
            this.presets = this.getDefaultPresets();
        }
    }

    /**
     * Get default presets
     */
    getDefaultPresets() {
        return [
        ];
    }

    /**
     * Apply preset configuration
     */
    applyPreset(preset) {

        // Clear current selection
        this.selectedNodes.clear();
        this.indeterminateNodes.clear();

        // Apply preset columns
        if (preset?.metaData?.isDefault) {
            // Select all columns
            this.selectAll();
        } else if (preset.isMinimal) {
            // Select only essential columns
            this.selectEssentialColumns();
        } else if (preset.columns && preset.columns.length > 0) {
            // Select specific columns
            preset.columns.forEach(colId => {
                this.selectedNodes.add(colId);
            });

            // Update grid
            const allColumns = this.columnDefs.map(c => c.colId || c.field);
            allColumns.forEach(colId => {
                const isVisible = this.selectedNodes.has(colId);
                this.api?.setColumnsVisible([colId], isVisible);
            });
        }

        // Update UI
        this.initializeSelectionStates();
        this.renderVirtualItems();
        // this.updateStatus();
    }

    /**
     * Save current configuration as preset
     */
    saveAsPreset(name) {
        const preset = {
            name: name,
            columns: Array.from(this.selectedNodes).filter(id => !id.startsWith('group_')),
            timestamp: Date.now()
        };

        // Add or update preset
        const existingIndex = this.presets.findIndex(p => p.name === name && !p?.metaData?.isDefault && !p.isMinimal);
        if (existingIndex !== -1) {
            this.presets[existingIndex] = preset;
        } else {
            this.presets.push(preset);
        }

        // Save to localStorage
        this.savePresets();

        return preset;
    }

    /**
     * Save presets to localStorage
     */
    savePresets() {
        try {
            const toSave = this.presets.filter(p => !p?.metaData?.isDefault && !p.isMinimal);
            localStorage.setItem(
                `treeColumnChooser_presets_${this.gridName}`,
                JSON.stringify(toSave)
            );
        } catch (error) {
            console.error('Error saving presets:', error);
        }

        // this.context.page.socketManager().dumpStorage();
    }

    /**
     * Apply columns programmatically
     */
    setColumns(columnIds, exclusive = false) {
        if (exclusive) {
            this.selectedNodes.clear();
            this.indeterminateNodes.clear();
        }

        columnIds.forEach(colId => {
            const node = this.findNodeById(colId);
            if (node && !node.isGroup) this.selectedNodes.add(colId);
        });

        if (exclusive) {
            const allColumns = this.columnDefs.map(c => c.colId || c.field);
            allColumns.forEach(colId => {
                const isVisible = this.selectedNodes.has(colId);
                this.api?.setColumnsVisible([colId], isVisible);
            });
        } else {
            this.api?.setColumnsVisible(columnIds, true);
        }

        // Always enforce required ON
        this._enforceRequiredSelections({ updateGrid: true });

        this.initializeSelectionStates?.();
        this.renderVirtualItems?.();
    }

    /**
     * Select all columns
     */
    selectAll() {
        const selectNodes = (nodes) => {
            nodes.forEach(node => {
                this.selectedNodes.add(node.id);
                this.indeterminateNodes.delete(node.id);
                if (node.children) {
                    selectNodes(node.children);
                }
            });
        };

        selectNodes(this.originalTreeData);

        // Update grid
        if (this.api) {
            const allColumns = this.columnDefs
                                    .map(c => c.colId || c.field)
                                    .filter(colId => {
                if (!colId) return false;
                const column = this.api.getColumn(colId);
                return column !== null;
            });

            if (allColumns.length > 0) {
                this.api.setColumnsVisible(allColumns, true);
            }
        }

        this.renderAllNodes();
        // this.updateStatus();
    }

    /**
     * Deselect all columns
     */
    async deselectAll() {
        if ((this.selectedNodes.size === 0) && (this.simpleUndo)) {
            await this._applyGridState(this.simpleUndo);
            this.simpleUndo = null;
            return;
        }

        this._pushUndo();
        this.simpleUndo = this.adapter.getGridState();
        this.selectedNodes.clear();
        this.indeterminateNodes.clear();

        const required = Array.from(this._getRequiredSet());

        if (this.api) {
            const allColumns = this.columnDefs
                                    .map(c => c.colId || c.field)
                                    .filter(colId => {
                if (!colId) return false;
                const column = this.api.getColumn(colId);
                return column !== null;
            });

            if (allColumns.length > 0) {
                this.api.setColumnsVisible(allColumns, false);
            }
            if (required.length > 0) {
                const showable = required.filter(id => this.api.getColumn(id) !== null);
                if (showable.length) this.api.setColumnsVisible(showable, true);
            }
        }

        required.forEach(id => this.selectedNodes.add(id));
        this.updateAllParentStates?.();
        this.renderAllNodes?.();
    }

    /**
     * Select essential columns only
     */
    selectEssentialColumns() {
        // This would be customized based on your needs
        const essentialFields = ['id', 'name', 'status', 'date'];

        this.selectedNodes.clear();
        this.indeterminateNodes.clear();

        essentialFields.forEach(field => {
            const node = this.findNodeByField(field);
            if (node) {
                this.selectedNodes.add(node.id);
            }
        });

        // Update grid
        const allColumns = this.columnDefs.map(c => c.colId || c.field);
        allColumns.forEach(colId => {
            const isVisible = this.selectedNodes.has(colId);
            this.api?.setColumnsVisible([colId], isVisible);
        });

        this.initializeSelectionStates();
        this.renderVirtualItems();
        // this.updateStatus();
    }

    /**
     * Select all children of a node
     */
    selectAllChildren(node) {
        if (!node.children) return;

        node.children.forEach(child => {
            this.selectedNodes.add(child.id);
            this.indeterminateNodes.delete(child.id);

            if (!child.isGroup) {
                this.api?.setColumnsVisible([child.id], true);
            }

            if (child.children) {
                this.selectAllChildren(child);
            }
        });
    }

    /**
     * Toggle sidebar visibility
     */
    toggleSidebar() {
        if (this.api?.isToolPanelShowing()) {
            this.api.closeToolPanel();
            this.clearSearch();
        } else {
            this.api?.openToolPanel(this.toolbarId);
            if (this.config.autoFocusSearch) {
                this._setTimeout(() => this.searchInput?.focus(), 100);
            }
        }
    }

    /**
     * Update status text
     */
    updateStatus() {
        if (!this.statusText) return;

        const totalColumns = this.columnDefs.filter(c => {
            const colId = c.colId || c.field;
            return colId && !this.findNodeById(colId)?.isGroup;
        }).length;

        const selectedColumns = Array.from(this.selectedNodes).filter(id =>
            !id.startsWith('group_')
        ).length;

        let status = `${selectedColumns} of ${totalColumns} columns selected.`;
        let stemp;
        if (this.activePresetName) {
            const safeName = String(this.activePresetName).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
            stemp = `<div class="footer-view">View: <strong>${safeName}</strong>`;
            if (this.hasUnsavedChanges) {
                stemp += ` <span class="unsaved-indicator">*</span>`;
                this.saveBtn.classList.add('unsaved-changes');
            } else {
                this.saveBtn.classList.remove('unsaved-changes');
            }
            stemp += '</div>' + status;
            status = stemp;
        }
        this.statusText.innerHTML = status;
    }

    /**
     * Track column states
     */
    trackColumnStates() {
        if (!this.api) return;

        if (this._onColumnPinned) {
            this.api?.removeEventListener('columnPinned', this._onColumnPinned);
        }
        if (this._onColumnVisible) {
            this.api?.removeEventListener('columnVisible', this._onColumnVisible);
        }

        this._onColumnPinned = (e) => {
            const columns = e.columns || [e.column];
            columns.forEach((column) => {
                if (column?.pinned === 'left') {
                    this.pinnedColumns.add(column.getColId());
                } else {
                    this.pinnedColumns.delete(column.getColId());
                }
            });
            this.renderVirtualItems();
        };

        this._onColumnVisible = (e) => {
            const columns = e.columns || [e.column];
            columns.forEach((column) => {
                if (column.visible) {
                    this.selectedNodes.add(column.getColId());
                } else {
                    this.selectedNodes.delete(column.getColId());
                }
            });

            this.renderVirtualItems();
        };

        this.api.addEventListener('columnPinned', this._onColumnPinned);
        this.api.addEventListener('columnVisible', this._onColumnVisible);
    }

    // ==================== State Persistence ====================
    saveExpandedState() {
        try {
            localStorage.setItem(
                `treeColumnChooser_expanded_${this.gridName}`,
                JSON.stringify(Array.from(this.expandedNodes))
            );
        } catch (error) {
            console.error('Error saving expanded state:', error);
        }
    }

    loadExpandedState() {
        try {
            const stored = localStorage.getItem(`treeColumnChooser_expanded_${this.gridName}`);
            return stored ? JSON.parse(stored) : null;
        } catch (error) {
            console.error('Error loading expanded state:', error);
            return null;
        }
    }


    loadSavedState() {
        try {
            const stored = localStorage.getItem(`treeColumnChooser_state_${this.gridName}`);
            if (stored) {
                const state = JSON.parse(stored);

                // Apply state
                if (state.expanded) {
                    this.expandedNodes = new Set(state.expanded);
                }

                if (state.customNames) {
                    this.customNames = state.customNames;
                    this.applyCustomNames(this.originalTreeData);
                }

                // Note: Column selection is initialized from grid state

                this.updateFlattenedNodes();
                this.updateVirtualizer();
            }
        } catch (error) {
            console.error('Error loading saved state:', error);
        }
    }

    async _showExportImportModal() {
        const treeCtx = this;
        const currentState = this._getGridState();
        const metaPayload = {gridName: this.gridName, exported: true, exportedAt: new Date().toISOString()};
        const exportPayload = {
            name: this.activePresetName || 'Untitled View',
            metadata: metaPayload,
            ...currentState,
        };
        // Include custom column names so they survive export/import
        if (this.customNames && Object.keys(this.customNames).length > 0) {
            exportPayload.customNames = { ...this.customNames };
        }
        const exportJson = JSON.stringify(exportPayload, null, 2);
        const pageContext = this; // Capture context
        await this.modalManager.showCustom({
            title: 'Export / Import Configuration',
            modalBoxClass: 'w-full max-w-2xl',
            includeDefaultActions: false,
            setupContent: (contentArea, dialog) => {
                contentArea.innerHTML = `
                            <div style="display:flex;flex-direction:column;gap:16px;padding:8px 0;">
                                <div>
                                    <h3 style="font-weight:600;margin-bottom:8px;font-size:14px;">Export Current Config</h3>
                                    <div style="display:flex;gap:8px;">
                                        <button class="btn btn-sm btn-outline" data-action="copy-clipboard">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                            Copy to Clipboard
                                        </button>
                                        <button class="btn btn-sm btn-outline" data-action="save-file">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                            Save to File
                                        </button>
                                    </div>
                                </div>
                                <div style="border-top:1px solid var(--ag-border-color, #ddd);padding-top:12px;">
                                    <h3 style="font-weight:600;margin-bottom:8px;font-size:14px;">Import Config</h3>
                                    <div style="display:flex;gap:8px;align-items:center;">
                                        <button class="btn btn-sm btn-primary" data-action="import-clipboard">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>
                                            Read from Clipboard
                                        </button>
                                        <span style="color:#999;font-size:12px;">or</span>
                                        <label class="btn btn-sm btn-outline" style="cursor:pointer;">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                            Upload File
                                            <input type="file" accept=".json" data-action="import-file" style="display:none;" />
                                        </label>
                                    </div>
                                </div>
                                <div data-role="status-msg" style="display:none;padding:8px;border-radius:4px;font-size:13px;"></div>
                            </div>
                        `;

                const statusEl = contentArea.querySelector('[data-role="status-msg"]');
                const showStatus = (msg, type = 'info') => {
                    statusEl.style.display = 'block';
                    statusEl.textContent = msg;
                    statusEl.style.background = type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#d1ecf1';
                    statusEl.style.color = type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#0c5460';
                };

                // Copy to clipboard
                contentArea.querySelector('[data-action="copy-clipboard"]').addEventListener('click', async () => {
                    try {
                        await writeStringToClipboard(exportJson);
                        showStatus('Config copied to clipboard!', 'success');
                    } catch (err) {
                        showStatus('Failed to copy: ' + err.message, 'error');
                    }
                });

                // Save to file
                contentArea.querySelector('[data-action="save-file"]').addEventListener('click', () => {
                    try {
                        const safeName = (treeCtx.activePresetName || treeCtx.gridName || 'grid-config').replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
                        const blob = new Blob([exportJson], {type: 'application/json'});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${safeName}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                        showStatus('File download started!', 'success');
                    } catch (err) {
                        showStatus('Failed to save: ' + err.message, 'error');
                    }
                });

                // Import from clipboard
                const clipboardBtn = contentArea.querySelector('[data-action="import-clipboard"]');
                clipboardBtn.addEventListener('click', async () => {
                    try {
                        clipboardBtn.disabled = true;
                        clipboardBtn.textContent = 'Reading Clipboard...';
                        const text = await navigator.clipboard.readText();
                        if (!text || !text.trim()) {
                            showStatus('Clipboard is empty.', 'error');
                            clipboardBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg> Read from Clipboard';
                            clipboardBtn.disabled = false;
                            return;
                        }
                        treeCtx.handleImport(text.trim());
                        dialog.close('imported_clipboard');
                    } catch (err) {
                        showStatus('Failed to read clipboard: ' + err.message, 'error');
                        clipboardBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px;"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg> Read from Clipboard';
                        clipboardBtn.disabled = false;
                    }
                });

                // Import from file upload
                contentArea.querySelector('[data-action="import-file"]').addEventListener('change', (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = (evt) => {
                        try {
                            treeCtx.handleImport(evt.target.result);
                            dialog.close('imported_file');
                        } catch (err) {
                            showStatus('Failed to import file: ' + err.message, 'error');
                        }
                    };
                    reader.onerror = () => showStatus('Failed to read file.', 'error');
                    reader.readAsText(file);
                });

                // Close button in action bar
                const actionBar = dialog.querySelector('[data-role="modal-actions"]');
                if (actionBar) {
                    const closeButton = document.createElement('button');
                    closeButton.className = 'btn btn-sm';
                    closeButton.textContent = 'Close';
                    closeButton.addEventListener('click', () => dialog.close('cancel'));
                    actionBar.appendChild(closeButton);
                }
            }
        });
    }

    async importState() {
        const pageContext = this;
        const result = await this.modalManager.showCustom({
            title: 'Import View',
            modalBoxClass: 'import-modal-box',
            includeDefaultActions: true,
            setupContent: (contentArea, dialog) => {
                contentArea.innerHTML = `
                    <div class="space-y-4">
                        <button class="btn btn-block" data-action="clipboard">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-2"> <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path> <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect> </svg>
                            Import from Clipboard
                        </button>
                        <div class="form-control w-full">
                            <label class="label"> <span class="label-text">Import from File</span> </label>
                            <input type="file" accept=".json" class="file-input file-input-bordered w-full" data-action="file" />
                        </div>
                    </div>
                `;

                // --- Add Buttons to Action Bar ---
                const actionBar = dialog.querySelector('[data-role="modal-actions"]');
                if (actionBar) {
                    const cancelButton = document.createElement('button');
                    cancelButton.className = 'btn btn-sm';
                    cancelButton.textContent = 'Cancel';
                    cancelButton.addEventListener('click', () => {
                        dialog.close('cancel'); // Explicitly close with 'cancel'
                    });
                    actionBar.appendChild(cancelButton);
                }

                // --- Setup Event Listeners ---
                const clipboardBtn = contentArea.querySelector('button[data-action="clipboard"]');
                const fileInput = contentArea.querySelector('input[data-action="file"]');

                clipboardBtn.addEventListener('click', async () => {
                    try {
                        clipboardBtn.disabled = true; // Prevent double clicks
                        clipboardBtn.textContent = 'Reading Clipboard...';
                        const text = await navigator.clipboard.readText();
                        if (text) {
                            await pageContext.handleImport(text); // Call external handler
                            dialog.close('imported_clipboard'); // Close indicating success type
                        } else {
                            console.warn('Clipboard was empty or permission denied.');
                            clipboardBtn.textContent = 'Import from Clipboard';
                            clipboardBtn.disabled = false;
                        }
                    } catch (err) {
                        console.error('Failed to read from clipboard:', err);
                        clipboardBtn.textContent = 'Error Reading Clipboard';
                        clipboardBtn.disabled = false;
                    }
                });

                fileInput.addEventListener('change', (e) => {
                    const file = e.target.files[0];
                    if (file) {
                        fileInput.disabled = true; // Disable while reading
                        const reader = new FileReader();
                        reader.onload = async (loadEvent) => {
                            try {
                                pageContext.handleImport(loadEvent.target.result);
                                dialog.close('imported_file'); // Close indicating success type
                            } catch (importError) {
                                console.error('Error handling imported file content:', importError);
                                fileInput.disabled = false; // Re-enable
                                fileInput.value = ''; // Reset file input
                            }
                        };
                        reader.onerror = () => {
                            console.error('Failed to read file');
                            // Show error feedback
                            fileInput.disabled = false;
                            fileInput.value = '';
                        }
                        reader.readAsText(file);
                    }
                });
            }
        });

        if (result === 'cancel_esc') {
            await this.showLoadDialog()
        }
    }

    setUnsavedChanges(){
        this.hasUnsavedChanges = false;
        this._updateButtonStates?.();
    }

    async saveImport(name, state) {

            const newPreset = {
                name: name,
                ...state,
                metaData: {
                    isMutable: true,
                    isTemporary: false,
                    isGlobal: false,
                    isDefault: false,
                    owner: '',
                    description: '',
                    lastModified: new Date().toISOString(),
                    created: new Date().toISOString(),
                    basedOn: name
                },
                version: '1.0.0',
                timestamp: Date.now()
            };

            // Add the new preset
            this.presets.set(name, newPreset);
            this.savePresetsToCache();

            // Load the new preset
            await this._loadPreset(name);

            this.context.page.toastManager().success(
                `${this.gridName.toUpperCase()} - Success!`,
                `New default view "${name}" created and applied.`
            );
        }

    /** Handles the actual import logic after getting data (string or object). */
    async handleImport(data) {
        try {
            const stateData = typeof data === 'string' ? JSON.parse(data) : data;

            // Basic validation of the imported structure
            if (!stateData?.name || !stateData.columnState) {
                throw new Error('Invalid view data format. Missing essential properties (name, columnState).');
            }

            // Show confirmation modal (View Only / Import)
            const safeImportName = String(stateData.name).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
            const actionResult = await this.modalManager.show({
                title: 'Import View Confirmation',
                body: `<p>You are about to import the view "<strong>${safeImportName}</strong>". How would you like to proceed?</p>`,
                preventBackdropClick: true,
                buttons: [
                    { text: 'Cancel', value: 'cancel' },
                    { text: 'View Only (Temporary)', value: 'view', class: 'btn-outline' },
                    { text: 'Import and Save', value: 'import', class: 'btn-primary' }
                ]
            });

            if (actionResult === 'cancel' || actionResult === 'cancel_esc' || actionResult === 'cancel_backdrop') {
                this.context.page.toastManager().info('Import cancelled.', 'Info');
                return false; // Indicate import was cancelled
            }

            // Reset unsaved changes flag *before* applying potentially temporary state
            this.setUnsavedChanges(false);

            // Prepare the state object (normalize, add/update metadata)
            const currentUser = window.frame.context.userManager.getCurrentUserName();
            const cleanState = {
                // Base structure from imported data
                name: stateData.name,
                columnState: stateData.columnState,
                filterModel: this.config.enableFilterMemory ? (stateData.filterModel || null) : null,
                sortModel: this.config.enableSortMemory ? (stateData.sortModel || null) : null,
                pinnedColumns: stateData.pinnedColumns || { left: [], right: [] }, // Ensure pinned exists

                metadata: {
                    ...(stateData.metadata || {}), // Keep existing metadata if present
                    mutable: true, // Imported states are mutable
                    owner: stateData.metadata?.owner || currentUser, // Keep original owner or set current user
                    created: stateData.metadata?.created || new Date().toISOString(),
                    lastModified: new Date().toISOString(), // Set import time as last modified
                    imported: true,
                    importedAt: new Date().toISOString(),
                    importedBy: currentUser
                },
                isGlobal: false, // Imported states are never global
                isDefault: false, // Imported states are never default initially
                isTemporary: (actionResult === 'view') // Set temporary flag for 'View Only'
            };

            // Apply imported custom column names if present
            if (stateData.customNames && typeof stateData.customNames === 'object') {
                Object.assign(this.customNames, stateData.customNames);
                this.saveCustomNames();
            }

            if (actionResult === 'import') {
                // Check if a preset with this name already exists
                let saveName = cleanState.name;
                if (this.presets.has(saveName)) {
                    const renameResult = await this.modalManager.show({
                        title: 'View Name Conflict',
                        body: `<p>A view named "<strong>${safeImportName}</strong>" already exists. Please choose a new name.</p>`,
                        fields: [
                            { type: 'text', id: 'viewName', label: 'View Name', required: true, value: saveName + ' (imported)' }
                        ],
                        buttons: [
                            { text: 'Cancel', value: 'cancel' },
                            { text: 'Save', value: 'save', class: 'btn-primary', isSubmit: true }
                        ]
                    });
                    if (!renameResult || renameResult === 'cancel' || renameResult === 'cancel_esc' || renameResult === 'cancel_backdrop' || !renameResult.viewName?.trim()) {
                        this.context.page.toastManager().info('Import cancelled.', 'Info');
                        return false;
                    }
                    saveName = renameResult.viewName.trim();
                    cleanState.name = saveName;
                }
                const success = this.saveImport(saveName, cleanState);
            } else { // actionResult === 'view'
                this.currentState = cleanState;
                this.pendingStateName = cleanState.name;
                await this._applyGridState(cleanState);
                this._setTimeout(() => {
                    this.setUnsavedChanges(true); this.context.page.toastManager()?.info(`Viewing "${cleanState.name}" temporarily.`, 'Info');
                }, 100);
            }

            return true; // Indicate import process was initiated (saved or viewed)

        } catch (err) {
            console.error('Import failed:', err);
            this.context.page.toastManager().error(`Import failed: ${err.message}`, 'Error');
            return false; // Indicate import failed
        }
    }

    // --- URL Parameter Handling ---

    checkUrlParams() {
        const params = new URLSearchParams(window.location.search);
        const viewData = params.get('view');

        if (viewData) {
            try {
                // Decode Base64 and parse JSON
                const decodedData = JSON.parse(atob(viewData));

                // Schedule the import handling after the main thread frees up
                // and potentially after user profile is loaded
                this.pendingImport = {
                    data: decodedData,
                    timestamp: Date.now()
                };
                // Remove the parameter from the URL to prevent re-import on refresh
                window.history.replaceState({}, document.title, window.location.pathname);

                // Attempt immediate processing if profile ready, otherwise wait
                if (window.frame?.context?.userProfile) {
                    this._setTimeout(() => this.processPendingImport(), 0);
                } else {
                    // Need an event listener or callback for when profile is ready
                    // window.addEventListener('profileLoaded', () => this.processPendingImport());
                }

                return true; // Indicate URL state was found
            } catch (err) {
                console.error('Failed to parse view data from URL:', err);
                this.context.page.toastManager().error('Failed to import view from URL parameter.', 'Error');
                // Remove invalid parameter
                window.history.replaceState({}, document.title, window.location.pathname);
            }
        }
        return false; // No valid URL state found
    }

    async processPendingImport() {
        if (this.pendingImport && (Date.now() - this.pendingImport.timestamp) < 30000) {
            await this.handleImport(this.pendingImport.data);
            this.pendingImport = null; // Clear pending data
            this._setupGridListeners()
        } else if (this.pendingImport) {
            console.warn("Pending import data expired or invalid.");
            this.pendingImport = null;
        }
    }

    // ==================== Utility Methods ====================

    findNodeById(id) {
        // Use cache for O(1) lookup
        return this.getNodeFromCache(id);
    }

    findNodeByField(field) {
        // Build field cache if needed
        if (!this.fieldCache) {
            this.fieldCache = new Map();
            const buildFieldCache = (nodes) => {
                nodes.forEach(node => {
                    if (node.colDef?.field) {
                        this.fieldCache.set(node.colDef.field, node);
                    }
                    if (node.children) {
                        buildFieldCache(node.children);
                    }
                });
            };
            buildFieldCache(this.originalTreeData);
        }

        return this.fieldCache.get(field);
    }

    getColumnDefs() {
        return this.adapter.getAllColumnDefs();
    }

    getFields() {
        return this.adapter.getAllFields();
    }


    initializeSelectionStates() {
        // Clear and rebuild in single pass
        const visibleColumns = new Set();
        const pinnedLeft = new Set();
        const lockedRight = new Set();

        // Collect all states from grid with safety checks
        if (this.api) {
            this.columnDefs.forEach(colDef => {
                const colId = colDef.colId || colDef.field;
                if (!colId) return;

                try {
                    const column = this.api.getColumn(colId);
                    if (column) {

                        if (column.visible) {
                            visibleColumns.add(colId);
                        }

                        if (column?.pinned === 'left') {
                            pinnedLeft.add(colId);
                        }
                        if (column?.pinned === 'right' &&
                            column.getColDef && column.getColDef().suppressMovable) {
                            lockedRight.add(colId);
                        }
                    }
                } catch (error) {
                    console.warn(`Error checking column state for ${colId}:`, error);
                }
            });
        }

        // Apply states to nodes efficiently
        this.selectedNodes.clear();
        this.indeterminateNodes.clear();
        this.pinnedColumns = pinnedLeft;
        this.lockedColumns = lockedRight;

        // Use node cache for efficient updates
        if (this.nodeCache) {
            this.nodeCache.forEach((node, nodeId) => {
                if (!node.isGroup && visibleColumns.has(nodeId)) {
                    this.selectedNodes.add(nodeId);
                }
            });
        }

        // Update parent states in single bottom-up pass
        this.updateAllParentStates();
    }

    /**
     * Update all parent states efficiently
     */
    updateAllParentStates() {
        // Build parent-child relationships if not cached
        if (!this.parentChildMap) {
            this.parentChildMap = new Map();
            this.nodeCache.forEach(node => {
                if (node.parent) {
                    if (!this.parentChildMap.has(node.parent)) {
                        this.parentChildMap.set(node.parent, []);
                    }
                    this.parentChildMap.get(node.parent).push(node.id);
                }
            });
        }

        // Process from leaves to root (bottom-up)
        const processedNodes = new Set();

        const processNode = (nodeId) => {
            if (processedNodes.has(nodeId)) return;
            processedNodes.add(nodeId);

            const node = this.nodeCache.get(nodeId);
            if (!node || !node.isGroup) return;

            const children = this.parentChildMap.get(nodeId) || [];
            if (children.length === 0) return;

            let selectedCount = 0;
            let indeterminateCount = 0;

            children.forEach(childId => {
                // Process child first (ensuring bottom-up)
                processNode(childId);

                if (this.selectedNodes.has(childId)) {
                    selectedCount++;
                } else if (this.indeterminateNodes.has(childId)) {
                    indeterminateCount++;
                }
            });

            // Update this node's state
            if (selectedCount === children.length) {
                this.selectedNodes.add(nodeId);
                this.indeterminateNodes.delete(nodeId);
            } else if (selectedCount > 0 || indeterminateCount > 0) {
                this.selectedNodes.delete(nodeId);
                this.indeterminateNodes.add(nodeId);
            } else {
                this.selectedNodes.delete(nodeId);
                this.indeterminateNodes.delete(nodeId);
            }
        };

        // Start from all nodes
        this.nodeCache.forEach((node, nodeId) => {
            processNode(nodeId);
        });
    }

    /**
     * Show empty state
     */
    showEmptyState() {
        this.virtualContainer.innerHTML = `
            <div style="padding: 20px; text-align: center; color: var(--ag-secondary-foreground-color, #666);">
                <p>No column definitions found.</p>
                <p style="font-size: 12px; margin-top: 10px;">
                    Column definitions will appear here once the grid is configured.
                </p>
            </div>
        `;
    }

    /**
     * Create cancellable promise
     */
    createCancellablePromise(asyncFn) {
        let cancelled = false;

        const checkCancelled = () => {
            if (cancelled) {
                throw new Error('Operation cancelled');
            }
        };

        const promise = asyncFn(checkCancelled);

        return {
            promise,
            cancel: () => { cancelled = true; }
        };
    }

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeoutId = null;
        const debounced = (...args) => {
            if (timeoutId) clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                timeoutId = null;
                func(...args);
            }, wait);
        };
        debounced.cancel = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
        };
        // this._debouncers.add(debounced);
        return debounced;
    }

    // ==================== AG Grid Interface Methods ====================

    /**
     * Get GUI element
     */
    getGui() {
        return this.mainContainer;
    }

    isToolPanelShowing() {
        return this.mainContainer && this.mainContainer.style.display !== 'none';
    }

    destroy() {
        // Unbind delegated container events
        this._unbindDelegatedEvents();

        if (this.controller) {
            this.controller.abort();
        }

        // Remove search listeners
        if (this._onSearchInput) {
            this.searchInput?.removeEventListener('input', this._onSearchInput);
            this._onSearchInput.cancel?.();
            this._onSearchInput = null;
        }
        if (this._onSearchToggleClearBtn) {
            this.searchInput?.removeEventListener('input', this._onSearchToggleClearBtn);
            this._onSearchToggleClearBtn = null;
        }
        if (this._onClearClick) {
            this.clearButton?.removeEventListener('click', this._onClearClick);
            this._onClearClick = null;
        }

        // Unregister AG Grid listeners
        this._removeGridChangeListeners();

        // Remove trackColumnStates listeners
        if (this.api) {
            if (this._onColumnPinned) { try { this.api.removeEventListener('columnPinned', this._onColumnPinned); } catch {} }
            if (this._onColumnVisible) { try { this.api.removeEventListener('columnVisible', this._onColumnVisible); } catch {} }
        }
        this._onColumnPinned = null;
        this._onColumnVisible = null;

        // Cancel all outstanding debouncers
        for (const d of this._debouncers) d.cancel?.();
        this._debouncers.clear();

        // Clear all outstanding timeouts
        this._clearAllTimeouts();

        cancelAnimationFrame(this._scrollRafId);

        if (this._virtualScrollHandler && this._scrollParent) {
            this._scrollParent.removeEventListener('scroll', this._virtualScrollHandler);
            this._virtualScrollHandler = null;
            this._virtualScrollBound = false;
            this._scrollParent = null;
        }

        // Sever references
        if (this.api && this.api.treeService === this) {
            this.api.treeService = null;
        }
        this.api = null;
        this.adapter = null;
        this.engine = null;
        this.modalManager = null;

        this.pendingImport = null;
        this._undoStack = [];

        this.fieldCache?.clear();
        this.fieldCache = null;
        this.parentChildMap?.clear();
        this.parentChildMap = null;

        this.nodeCache?.clear?.();
        this.searchCache.clear();
        this.nodeTextCache.clear();
        this.columnDefs = [];
        this.originalTreeData = [];
        this.flattenedNodes = [];
        this.expandedNodes.clear();
        this.selectedNodes.clear();
        this.indeterminateNodes.clear();
        this.searchMatches.clear();
        this.pinnedColumns.clear();
        this.lockedColumns.clear();

        // Remove DOM
        if (this.mainContainer?.parentNode) {
            this.mainContainer.parentNode.removeChild(this.mainContainer);
        }

        this.initialized = false;
        this.domReady = false;
    }
}

export class PivotColumnChooser extends TreeColumnChooser {
    constructor(config = {}) {
        config['enableFilterMemory'] = false;
        config['enableSortMemory'] = false;
        super(config);
    }

    getColumnDefs() {
        return this.api.getColumnDefs()
    }

    getFields() {
        return this.api.getColumnDefs().map(x=>x.field)
    }

}
