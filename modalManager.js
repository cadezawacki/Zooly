


export class ModalManager {
    constructor(context) {
        this.context = context;
    }

    async show(config) {
        return new Promise((resolve) => {
            // --- 1. Create Modal Elements ---
            const dialog = document.createElement('dialog');
            dialog.className = `modal ${config.modalClass || ''}`.trim();
            if (config.id) {
                dialog.id = config.id;
            }

            const modalBox = document.createElement('div');
            modalBox.className = `modal-box ${config.modalBoxClass || ''}`.trim();

            // Title
            const titleElement = document.createElement('h3');
            titleElement.className = 'font-bold text-lg mb-4';
            titleElement.textContent = config.title;
            modalBox.appendChild(titleElement);

            // Body (before fields)
            const bodyElement = document.createElement('div');
            bodyElement.className = 'modal-wrapper'; // Add margin below body, before fields
            if (typeof config.body === 'string') {
                bodyElement.innerHTML = config.body;
            } else if (config.body instanceof HTMLElement) {
                bodyElement.appendChild(config.body);
            } else if (config.body) { // Check if body is defined but not string/element
                console.warn('Modal body content type not supported, omitting.');
            }
            // Only append body if it has content or children
            if (bodyElement.innerHTML || bodyElement.hasChildNodes()) {
                modalBox.appendChild(bodyElement);
            }


            // --- 2. Create Form and Fields ---
            const form = document.createElement('form');
            // Prevent default browser submission; we handle it manually
            form.addEventListener('submit', (e) => e.preventDefault());
            form.className = config.formClass || '';

            const fieldElements = {}; // Store references to input elements { id: element }
            let firstFieldElement = null; // To focus later

            if (config.fields && config.fields.length > 0) {
                config.fields.forEach(fieldConfig => {
                    if (!fieldConfig.id) {
                        console.warn('Modal field is missing an ID, skipping:', fieldConfig);
                        return;
                    }

                    const wrapper = document.createElement('div');
                    // Example: using 'form-control' similar to DaisyUI/your example
                    wrapper.className = `form-control w-full mb-3 ${fieldConfig.wrapperClass || ''}`.trim();

                    if (fieldConfig.label) {
                        const label = document.createElement('label');
                        label.htmlFor = fieldConfig.id;
                        label.textContent = fieldConfig.label;
                        label.className = `label ${fieldConfig.labelClass || ''}`.trim();
                        wrapper.appendChild(label);
                    }

                    let inputElement;
                    if (fieldConfig.type === 'textarea') {
                        inputElement = document.createElement('textarea');
                        inputElement.rows = fieldConfig.rows || 3;
                    } else {
                        inputElement = document.createElement('input');
                        inputElement.type = fieldConfig.type || 'text'; // Default to text
                    }

                    inputElement.id = fieldConfig.id;
                    inputElement.name = fieldConfig.id; // Useful for form data if needed elsewhere
                    inputElement.placeholder = fieldConfig.placeholder || '';
                    inputElement.value = fieldConfig.value || '';
                    // Example: using 'input input-bordered'
                    inputElement.className = `input input-bordered w-full ${fieldConfig.inputClass || ''}`.trim();
                    if(fieldConfig.type === 'textarea') {
                        inputElement.classList.remove('input'); // Textarea might use different base class
                        inputElement.classList.add('textarea', 'textarea-bordered'); // Example textarea classes
                    }

                    if (fieldConfig.required) {
                        inputElement.required = true; // Use native 'required' attribute
                    }

                    wrapper.appendChild(inputElement);
                    form.appendChild(wrapper);
                    fieldElements[fieldConfig.id] = inputElement;

                    if (!firstFieldElement) {
                        firstFieldElement = inputElement;
                    }
                });
                modalBox.appendChild(form);
            }


            // --- 3. Actions / Buttons ---
            const modalAction = document.createElement('div');
            modalAction.className = 'modal-action flex justify-end gap-2 mt-4'; // Added top margin

            const buttonElements = {}; // Store references { value: element }

            const validateAndGetSubmitButton = () => {
                let submitButton = null;
                for (const btnValue in buttonElements) {
                    const btnConfig = config.buttons.find(b => b.value === btnValue);
                    if (btnConfig?.isSubmit) {
                        submitButton = buttonElements[btnValue];
                        break; // Assume only one primary submit button for validation enabling/disabling
                    }
                }
                return submitButton;
            };

            const checkFormValidity = () => {
                if (!config.fields || config.fields.length === 0) return true; // No fields, always valid

                for (const fieldConfig of config.fields) {
                    if (fieldConfig.required) {
                        const element = fieldElements[fieldConfig.id];
                        if (!element || !element.value.trim()) {
                            return false; // Found a required field that's empty
                        }
                    }
                    // Add more complex validation checks here if needed
                }
                return true; // All required fields are filled
            };

            config.buttons.forEach(buttonConfig => {
                const button = document.createElement('button');
                button.type = 'button'; // Explicitly set type
                button.textContent = buttonConfig.text;
                button.className = `btn btn-sm ${buttonConfig.class || ''}`.trim();
                button.dataset.value = buttonConfig.value; // Store value for retrieval

                if (buttonConfig.isDisabled || (buttonConfig.isSubmit && !checkFormValidity())) {
                    button.disabled = true;
                }


                button.addEventListener('click', async (event) => {
                    event.preventDefault();
                    button.disabled = true; // Prevent double clicks

                    try {
                        // 1. Run pre-close action if defined
                        if (buttonConfig.action && typeof buttonConfig.action === 'function') {
                            await buttonConfig.action();
                        }

                        // 2. Handle submission or regular close
                        if (buttonConfig.isSubmit) {
                            if (checkFormValidity()) {
                                const results = {};
                                config.fields.forEach(fConfig => {
                                    if (fieldElements[fConfig.id]) {
                                        results[fConfig.id] = fieldElements[fConfig.id].value;
                                    }
                                });
                                dialog.close(JSON.stringify(results)); // Stringify object for returnValue
                            } else {
                                console.warn("Submit validation failed.");
                                // Re-enable button if validation fails? Or keep disabled until fixed?
                                button.disabled = false; // Re-enable for user to fix input
                                // Do NOT close the dialog here
                            }
                        } else if (buttonConfig?.dontClose !== true) {
                            // Regular button click, close with its value
                            dialog.close(buttonConfig.value);
                        }
                    } catch (error) {
                        console.error("Error during button action or closing:", error);
                        // Decide on recovery: maybe close with an error value?
                        button.disabled = false; // Re-enable button on error
                        // Don't close automatically on error within action? Optional.
                    }
                });

                modalAction.appendChild(button);
                buttonElements[buttonConfig.value] = button;
            });

            modalBox.appendChild(modalAction);
            dialog.appendChild(modalBox);

            // --- 4. Add Validation Listeners ---
            if (config.fields && config.fields.length > 0) {
                const submitButton = validateAndGetSubmitButton();
                if (submitButton) {
                    Object.values(fieldElements).forEach(inputEl => {
                        inputEl.addEventListener('input', () => {
                            submitButton.disabled = !checkFormValidity();
                        });

                        // Add Enter key listener (only for single-line inputs)
                        if (inputEl.tagName.toLowerCase() === 'input' && inputEl.type !== 'textarea') {
                            inputEl.addEventListener('keyup', (e) => {
                                if (e.key === 'Enter' && checkFormValidity()) {
                                    e.preventDefault(); // Prevent potential form submission
                                    submitButton.click(); // Simulate click on the valid submit button
                                }
                            });
                        }
                    });
                }
            }

            // --- 5. Backdrop & Native Closing ---
            const backdropForm = document.createElement('form');
            backdropForm.method = 'dialog';
            backdropForm.className = 'modal-backdrop';
            const closeButton = document.createElement('button');
            // Style same as before
            closeButton.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; cursor:default; background:transparent; border:none;';
            closeButton.setAttribute('aria-label', 'Close modal');
            closeButton.value = config.preventBackdropClick ? '_ignore_backdrop' : 'cancel_backdrop';
            backdropForm.appendChild(closeButton);
            dialog.appendChild(backdropForm);

            // --- 6. Event Listener for Closing ---
            const onClose = () => {
                const rawValue = dialog.returnValue;
                dialog.removeEventListener('close', onClose);
                dialog.remove(); // Cleanup from DOM

                let result;
                if (rawValue === 'cancel_backdrop' || rawValue === '_ignore_backdrop') {
                    result = 'cancel_backdrop';
                } else if (rawValue === '' || rawValue === undefined) {
                    result = 'cancel_esc';
                } else {
                    // Try to parse as JSON (for submitted data)
                    try {
                        result = JSON.parse(rawValue); // This will be the {id: value} object
                    } catch (e) {
                        result = rawValue; // Otherwise, it's a normal button value string
                    }
                }
                resolve(result);
            };
            dialog.addEventListener('close', onClose);

            // --- 7. Show Modal & Focus ---
            document.body.appendChild(dialog);
            dialog.showModal();

            // Focus the first input field after modal is shown
            if (firstFieldElement) {
                // Use requestAnimationFrame to ensure focus works after rendering/transitions
                requestAnimationFrame(() => {
                    firstFieldElement.focus();
                });
            }
        });
    }

    async showCustom(config) {
        return new Promise((resolve) => {
            let dialog = null;
            let isResolved = false; // Prevent double resolution

            // --- Centralized Cleanup ---
            const cleanup = (reason = 'unknown') => {
                if (dialog && dialog.isConnected) {
                    dialog.removeEventListener('close', onClose);
                    try {
                        if (dialog.open) {
                            dialog.close('cleanup_initiated');
                        }
                        dialog.remove();
                    } catch (e) {
                        console.error('[ModalManager.showCustom] Error during dialog removal:', e);
                    }
                }
                dialog = null;
            };

            // --- Safe Resolution ---
            const safeResolve = (value) => {
                if (!isResolved) {
                    isResolved = true;
                    resolve(value);
                }
            };

            // --- Close Event Handler ---
            const onClose = () => {
                const rawValue = dialog?.returnValue;
                cleanup('native_close_event');

                // Determine resolution value
                let result;
                if (rawValue === 'cancel_backdrop' || rawValue === '_ignore_backdrop') {
                    result = 'cancel_backdrop';
                } else if (rawValue === '' || rawValue === undefined || rawValue === 'cancel') {
                    result = 'cancel';
                } else if (rawValue === 'cleanup_initiated') {
                    result = 'cancel';
                } else {
                    // Try to parse as JSON if it looks like JSON
                    if (typeof rawValue === 'string' &&
                        (rawValue.startsWith('{') || rawValue.startsWith('['))) {
                        try {
                            result = JSON.parse(rawValue);
                        } catch (e) {
                            result = rawValue;
                        }
                    } else {
                        result = rawValue;
                    }
                }

                safeResolve(result);
            };

            try {
                // --- 1. Create Modal Shell ---
                dialog = document.createElement('dialog');
                dialog.className = `modal ${config.modalClass || ''}`.trim();
                if (config.id) dialog.id = config.id;

                const modalBox = document.createElement('div');
                modalBox.className = `modal-box ${config.modalBoxClass || ''}`.trim();

                if (config.title) {
                    const titleWrapper = document.createElement('div');
                    titleWrapper.className = 'modal-title';
                    const titleElement = document.createElement('h3');
                    titleElement.className = 'font-bold text-lg mb-4';
                    titleElement.textContent = config.title;
                    titleWrapper.appendChild(titleElement);
                    modalBox.appendChild(titleWrapper);
                }

                const contentArea = document.createElement('div');
                contentArea.className = 'modal-custom-content';
                modalBox.appendChild(contentArea);

                if (config.includeDefaultActions) {
                    const modalAction = document.createElement('div');
                    modalAction.className = 'modal-action flex justify-end gap-2 mt-4';
                    modalAction.dataset.role = 'modal-actions';
                    modalBox.appendChild(modalAction);
                }
                dialog.appendChild(modalBox);

                // --- 2. Backdrop & Native Closing ---
                const backdropForm = document.createElement('form');
                backdropForm.method = 'dialog';
                backdropForm.className = 'modal-backdrop';
                const closeButton = document.createElement('button');
                closeButton.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; cursor:default; background:transparent; border:none;';
                closeButton.setAttribute('aria-label', 'Close modal');
                closeButton.value = config.preventBackdropClick ? '_ignore_backdrop' : 'cancel_backdrop';
                backdropForm.appendChild(closeButton);
                dialog.appendChild(backdropForm);

                // --- 3. Attach Close Listener EARLY ---
                dialog.addEventListener('close', onClose);

                // --- 4. Setup Custom Content ---
                // DON'T pass resolve directly! Pass a wrapper that closes the dialog properly
                const closeWithValue = (value) => {
                    if (dialog && dialog.open) {
                        // Convert objects to JSON string for dialog.returnValue
                        const returnValue = typeof value === 'object'
                            ? JSON.stringify(value)
                            : (value || 'cancel');
                        dialog.close(returnValue);
                        // The onClose handler will resolve the promise
                    } else {
                        // Dialog already closed or not open, resolve directly
                        safeResolve(value || 'cancel');
                    }
                };

                // Call setupContent with the safe close function instead of resolve
                config.setupContent(contentArea, dialog, closeWithValue);

                // --- 5. Append to DOM and Show ---
                document.body.appendChild(dialog);

                if (!dialog.isConnected) {
                    console.error("[ModalManager.showCustom] Failed to append dialog to body!");
                    cleanup('append_failed');
                    safeResolve('error_dom');
                    return;
                }

                dialog.showModal();

                if (!dialog.open) {
                    console.warn("[ModalManager.showCustom] Dialog did not open after showModal() call.");
                }

            } catch (error) {
                console.error('[ModalManager.showCustom] Error during modal setup:', error);
                cleanup('setup_error');
                safeResolve('error_setup');
            }
        });
    }
}

