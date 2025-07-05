import { html, css, LitElement } from '../assets/lit-core-2.7.4.min.js';

export class ApiKeyHeader extends LitElement {
    static properties = {
        apiKey: { type: String },
        isLoading: { type: Boolean },
        errorMessage: { type: String },
        selectedProvider: { type: String },
        ollamaModels: { type: Array },
        selectedOllamaModel: { type: String },
        isLoadingModels: { type: Boolean },
    };

    static styles = css`
        :host {
            display: block;
            transform: translate3d(0, 0, 0);
            backface-visibility: hidden;
            transition: opacity 0.25s ease-out;
        }

        :host(.sliding-out) {
            animation: slideOutUp 0.3s ease-in forwards;
            will-change: opacity, transform;
        }

        :host(.hidden) {
            opacity: 0;
            pointer-events: none;
        }

        @keyframes slideOutUp {
            from {
                opacity: 1;
                transform: translateY(0);
            }
            to {
                opacity: 0;
                transform: translateY(-20px);
            }
        }

        * {
            font-family: 'Helvetica Neue', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            cursor: default;
            user-select: none;
            box-sizing: border-box;
        }

        .container {
            width: 285px;
            min-height: 260px;
            padding: 18px 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            overflow: visible;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 16px;
            padding: 1px;
            background: linear-gradient(169deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 50%, rgba(255, 255, 255, 0.5) 100%);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: destination-out;
            mask-composite: exclude;
            pointer-events: none;
        }

        .close-button {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 14px;
            height: 14px;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            border-radius: 3px;
            color: rgba(255, 255, 255, 0.7);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            z-index: 10;
            font-size: 14px;
            line-height: 1;
            padding: 0;
        }

        .close-button:hover {
            background: rgba(255, 255, 255, 0.2);
            color: rgba(255, 255, 255, 0.9);
        }

        .close-button:active {
            transform: scale(0.95);
        }

        .title {
            color: white;
            font-size: 16px;
            font-weight: 500; /* Medium */
            margin: 0;
            text-align: center;
            flex-shrink: 0;
        }

        .form-content {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .error-message {
            color: #ff6b6b;
            font-size: 12px;
            text-align: center;
            min-height: 16px;
            transition: opacity 0.2s ease;
        }

        .api-input {
            width: 100%;
            height: 34px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border: none;
            padding: 0 10px;
            color: white;
            font-size: 12px;
            font-weight: 400; /* Regular */
            margin-bottom: 6px;
            text-align: center;
            user-select: text;
            cursor: text;
        }

        .api-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }

        .api-input:focus {
            outline: none;
        }

        .provider-label {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 5px;
        }

        .provider-select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            font-size: 13px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            outline: none;
            transition: border-color 0.2s ease;
        }

        .provider-select:hover {
            background-color: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .provider-select:focus {
            border-color: #007AFF;
            box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
        }

        .provider-select option {
            background: #1a1a1a;
            color: white;
            padding: 5px;
        }

        .action-button {
            width: 100%;
            height: 34px;
            background: rgba(255, 255, 255, 0.2);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 12px;
            font-weight: 500; /* Medium */
            cursor: pointer;
            transition: background 0.15s ease;
            position: relative;
            overflow: visible;
        }

        .action-button::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 10px;
            padding: 1px;
            background: linear-gradient(169deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 50%, rgba(255, 255, 255, 0.5) 100%);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: destination-out;
            mask-composite: exclude;
            pointer-events: none;
        }

        .action-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .action-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .or-text {
            color: rgba(255, 255, 255, 0.5);
            font-size: 12px;
            font-weight: 500; /* Medium */
            margin: 10px 0;
        }
        
        .ollama-info {
            padding: 12px;
            background: rgba(0, 120, 255, 0.1);
            border: 1px solid rgba(0, 120, 255, 0.3);
            border-radius: 6px;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.9);
            line-height: 1.4;
        }

        .ollama-info p {
            margin: 0 0 8px 0;
        }

        .ollama-info p:last-child {
            margin-bottom: 0;
        }

        .model-select-container {
            margin-top: 10px;
        }

        .model-label {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 5px;
            display: block;
        }

        .model-select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            font-size: 13px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            outline: none;
            transition: border-color 0.2s ease;
        }

        .model-select:hover {
            background-color: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .model-select:focus {
            border-color: #007AFF;
            box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
        }

        .model-select option {
            background: #1a1a1a;
            color: white;
            padding: 5px;
        }

        .model-loading {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            padding: 10px;
        }

        .model-error {
            font-size: 12px;
            color: #ff6b6b;
            text-align: center;
            padding: 10px;
        }
    `;

    constructor() {
        super();
        this.dragState = null;
        this.wasJustDragged = false;
        this.apiKey = '';
        this.isLoading = false;
        this.errorMessage = '';
        this.validatedApiKey = null;
        this.selectedProvider = 'openai';
        this.ollamaModels = [];
        this.selectedOllamaModel = '';
        this.isLoadingModels = false;

        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleKeyPress = this.handleKeyPress.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleInput = this.handleInput.bind(this);
        this.handleAnimationEnd = this.handleAnimationEnd.bind(this);
        this.handleUsePicklesKey = this.handleUsePicklesKey.bind(this);
        this.handleProviderChange = this.handleProviderChange.bind(this);
        this.handleModelChange = this.handleModelChange.bind(this);
        this.checkAndRequestPermissions = this.checkAndRequestPermissions.bind(this);
    }

    reset() {
        this.apiKey = '';
        this.isLoading = false;
        this.errorMessage = '';
        this.validatedApiKey = null;
        this.selectedProvider = 'openai';
        this.ollamaModels = [];
        this.selectedOllamaModel = '';
        this.isLoadingModels = false;
        this.requestUpdate();
    }

    async handleMouseDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON' || e.target.tagName === 'SELECT') {
            return;
        }

        e.preventDefault();

        const { ipcRenderer } = window.require('electron');
        const initialPosition = await ipcRenderer.invoke('get-header-position');

        this.dragState = {
            initialMouseX: e.screenX,
            initialMouseY: e.screenY,
            initialWindowX: initialPosition.x,
            initialWindowY: initialPosition.y,
            moved: false,
        };

        window.addEventListener('mousemove', this.handleMouseMove);
        window.addEventListener('mouseup', this.handleMouseUp, { once: true });
    }

    handleMouseMove(e) {
        if (!this.dragState) return;

        const deltaX = Math.abs(e.screenX - this.dragState.initialMouseX);
        const deltaY = Math.abs(e.screenY - this.dragState.initialMouseY);

        if (deltaX > 3 || deltaY > 3) {
            this.dragState.moved = true;
        }

        const newWindowX = this.dragState.initialWindowX + (e.screenX - this.dragState.initialMouseX);
        const newWindowY = this.dragState.initialWindowY + (e.screenY - this.dragState.initialMouseY);

        const { ipcRenderer } = window.require('electron');
        ipcRenderer.invoke('move-header-to', newWindowX, newWindowY);
    }

    handleMouseUp(e) {
        if (!this.dragState) return;

        const wasDragged = this.dragState.moved;

        window.removeEventListener('mousemove', this.handleMouseMove);
        this.dragState = null;

        if (wasDragged) {
            this.wasJustDragged = true;
            setTimeout(() => {
                this.wasJustDragged = false;
            }, 200);
        }
    }

    handleInput(e) {
        this.apiKey = e.target.value;
        this.errorMessage = '';
        console.log('Input changed:', this.apiKey?.length || 0, 'chars');

        this.requestUpdate();
        this.updateComplete.then(() => {
            const inputField = this.shadowRoot?.querySelector('.apikey-input');
            if (inputField && this.isInputFocused) {
                inputField.focus();
            }
        });
    }

    handleProviderChange(e) {
        this.selectedProvider = e.target.value;
        this.errorMessage = '';
        console.log('Provider changed to:', this.selectedProvider);
        
        // Clear API key when switching to Ollama since it doesn't need one
        if (this.selectedProvider === 'ollama') {
            this.apiKey = 'ollama-local'; // Placeholder value for Ollama
            this.loadOllamaModels();
        } else if (this.apiKey === 'ollama-local') {
            this.apiKey = '';
            this.ollamaModels = [];
            this.selectedOllamaModel = '';
        }
        
        this.requestUpdate();
    }

    handleModelChange(e) {
        this.selectedOllamaModel = e.target.value;
        console.log('Ollama model changed to:', this.selectedOllamaModel);
        this.requestUpdate();
    }

    async loadOllamaModels() {
        this.isLoadingModels = true;
        this.ollamaModels = [];
        this.selectedOllamaModel = '';
        this.requestUpdate();

        try {
            console.log('Loading Ollama models...');
            const response = await fetch('http://localhost:11434/api/tags');
            
            if (response.ok) {
                const data = await response.json();
                this.ollamaModels = data.models || [];
                
                // Auto-select the first model if available
                if (this.ollamaModels.length > 0) {
                    this.selectedOllamaModel = this.ollamaModels[0].name;
                }
                
                console.log('Loaded Ollama models:', this.ollamaModels.map(m => m.name));
            } else {
                console.error('Failed to load Ollama models:', response.status);
                this.ollamaModels = [];
            }
        } catch (error) {
            console.error('Error loading Ollama models:', error);
            this.ollamaModels = [];
        } finally {
            this.isLoadingModels = false;
            this.requestUpdate();
        }
    }

    handlePaste(e) {
        e.preventDefault();
        this.errorMessage = '';
        const clipboardText = (e.clipboardData || window.clipboardData).getData('text');
        console.log('Paste event detected:', clipboardText?.substring(0, 10) + '...');

        if (clipboardText) {
            this.apiKey = clipboardText.trim();

            const inputElement = e.target;
            inputElement.value = this.apiKey;
        }

        this.requestUpdate();
        this.updateComplete.then(() => {
            const inputField = this.shadowRoot?.querySelector('.apikey-input');
            if (inputField) {
                inputField.focus();
                inputField.setSelectionRange(inputField.value.length, inputField.value.length);
            }
        });
    }

    handleKeyPress(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            this.handleSubmit();
        }
    }

    async handleSubmit() {
        if (this.wasJustDragged || this.isLoading) {
            console.log('Submit blocked:', {
                wasJustDragged: this.wasJustDragged,
                isLoading: this.isLoading,
            });
            return;
        }

        // For Ollama, check if model is selected
        if (this.selectedProvider === 'ollama') {
            if (!this.selectedOllamaModel) {
                this.errorMessage = 'Please select an Ollama model';
                this.requestUpdate();
                return;
            }
        } else if (!this.apiKey.trim()) {
            console.log('Submit blocked: no API key');
            return;
        }

        console.log('Starting validation...');
        this.isLoading = true;
        this.errorMessage = '';
        this.requestUpdate();

        const apiKey = this.apiKey.trim();
        let isValid = false;
        try {
            const isValid = await this.validateApiKey(this.apiKey.trim(), this.selectedProvider);
            
            if (isValid) {
                console.log('API key valid – checking system permissions…');
                const permissionResult = await this.checkAndRequestPermissions();

                if (permissionResult.success) {
                    console.log('All permissions granted – starting slide-out animation');
                    this.startSlideOutAnimation();
                    this.validatedApiKey = this.selectedProvider === 'ollama' ? 'ollama-local' : this.apiKey.trim();
                    this.validatedProvider = this.selectedProvider;
                    this.validatedModel = this.selectedProvider === 'ollama' ? this.selectedOllamaModel : null;
                } else {
                    this.errorMessage = permissionResult.error || 'Permission setup required';
                    console.log('Permission setup incomplete:', permissionResult);
                }
            } else {
                if (this.selectedProvider === 'ollama') {
                    this.errorMessage = 'Unable to connect to Ollama - is it running on port 11434?';
                } else {
                    this.errorMessage = 'Invalid API key - please check and try again';
                }
                console.log('Validation failed for provider:', this.selectedProvider);
            }
        } catch (error) {
            console.error('API key validation error:', error);
            this.errorMessage = 'Validation error - please try again';
        } finally {
            this.isLoading = false;
            this.requestUpdate();
        }
    }

    async validateApiKey(apiKey, provider = 'openai') {
        if (provider === 'ollama') {
            // For Ollama, check if the service is running on port 11434
            try {
                console.log('Validating Ollama connection...');
                const response = await fetch('http://localhost:11434/api/tags');
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('Ollama validation successful, available models:', data.models?.length || 0);
                    return true;
                } else {
                    console.log('Ollama validation failed:', response.status);
                    return false;
                }
            } catch (error) {
                console.error('Ollama validation network error:', error);
                return false;
            }
        }
        
        if (!apiKey || apiKey.length < 15) return false;
        
        if (provider === 'openai') {
            if (!apiKey.match(/^[A-Za-z0-9_-]+$/)) return false;
            
            try {
                console.log('Validating OpenAI API key...');

                const response = await fetch('https://api.openai.com/v1/models', {
                    headers: {
                        'Content-Type': 'application/json',
                        Authorization: `Bearer ${apiKey}`,
                    },
                });

                if (response.ok) {
                    const data = await response.json();

                    const hasGPTModels = data.data && data.data.some(m => m.id.startsWith('gpt-'));
                    if (hasGPTModels) {
                        console.log('OpenAI API key validation successful');
                        return true;
                    } else {
                        console.log('API key valid but no GPT models available');
                        return false;
                    }
                } else {
                    const errorData = await response.json().catch(() => ({}));
                    console.log('API key validation failed:', response.status, errorData.error?.message || 'Unknown error');
                    return false;
                }
            } catch (error) {
                console.error('API key validation network error:', error);
                return apiKey.length >= 20; // Fallback for network issues
            }
        } else if (provider === 'gemini') {
            // Gemini API keys typically start with 'AIza'
            if (!apiKey.match(/^[A-Za-z0-9_-]+$/)) return false;
            
            try {
                console.log('Validating Gemini API key...');
                
                // Test the API key with a simple models list request
                const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);
                
                if (response.ok) {
                    const data = await response.json();
                    if (data.models && data.models.length > 0) {
                        console.log('Gemini API key validation successful');
                        return true;
                    }
                }
                
                console.log('Gemini API key validation failed');
                return false;
            } catch (error) {
                console.error('Gemini API key validation network error:', error);
                return apiKey.length >= 20; // Fallback
            }
        }
        
        return false;
    }

    async checkAndRequestPermissions() {
        if (!window.require) return { success: true };
    
        const { ipcRenderer } = window.require('electron');
    
        try {
            const permissions = await ipcRenderer.invoke('check-system-permissions');
            console.log('[Permissions] Current status:', permissions);
    
            if (!permissions.needsSetup) return { success: true };
    
            if (!permissions.microphone) {
                console.log('[Permissions] Requesting microphone permission…');
                const micResult = await ipcRenderer.invoke('request-microphone-permission');
                if (!micResult.success) {
                    await ipcRenderer.invoke('open-system-preferences', 'microphone');
                    return {
                        success: false,
                        error: 'Please grant microphone access in System Preferences',
                    };
                }
            }
    
            if (!permissions.screen) {
                console.log('[Permissions] Screen-recording permission needed');
                await ipcRenderer.invoke('open-system-preferences', 'screen-recording');
                return {
                    success: false,
                    error: 'Please grant screen recording access in System Preferences',
                };
            }
    
            return { success: true };
        } catch (err) {
            console.error('[Permissions] Error checking/requesting permissions:', err);
            return { success: false, error: 'Failed to check permissions' };
        }
    }

    startSlideOutAnimation() {
        this.classList.add('sliding-out');
    }

    handleUsePicklesKey(e) {
        e.preventDefault();
        if (this.wasJustDragged) return;

        console.log('Requesting Firebase authentication from main process...');
        if (window.require) {
            window.require('electron').ipcRenderer.invoke('start-firebase-auth');
        }
    }

    handleClose() {
        console.log('Close button clicked');
        if (window.require) {
            window.require('electron').ipcRenderer.invoke('quit-application');
        }
    }

    handleAnimationEnd(e) {
        if (e.target !== this) return;

        if (this.classList.contains('sliding-out')) {
            this.classList.remove('sliding-out');
            this.classList.add('hidden');

            if (this.validatedApiKey) {
                if (window.require) {
                    const payload = {
                        apiKey: this.validatedApiKey,
                        provider: this.validatedProvider || 'openai'
                    };
                    
                    // Add model info for Ollama
                    if (this.validatedProvider === 'ollama' && this.validatedModel) {
                        payload.model = this.validatedModel;
                    }
                    
                    window.require('electron').ipcRenderer.invoke('api-key-validated', payload);
                }
                this.validatedApiKey = null;
                this.validatedProvider = null;
                this.validatedModel = null;
            }
        }
    }

    connectedCallback() {
        super.connectedCallback();
        this.addEventListener('animationend', this.handleAnimationEnd);

    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeEventListener('animationend', this.handleAnimationEnd);

    }

    getPlaceholderText() {
        switch (this.selectedProvider) {
            case 'openai':
                return "Enter your OpenAI API key";
            case 'gemini':
                return "Enter your Gemini API key";
            case 'ollama':
                return "No API key needed for Ollama";
            default:
                return "Enter your API key";
        }
    }

    render() {
        const isButtonDisabled = this.isLoading || 
            (this.selectedProvider === 'ollama' ? !this.selectedOllamaModel : (!this.apiKey || !this.apiKey.trim()));
        console.log('Rendering with provider:', this.selectedProvider);

        return html`
            <div class="container" @mousedown=${this.handleMouseDown}>
                <button class="close-button" @click=${this.handleClose} title="Close application">
                    <svg width="8" height="8" viewBox="0 0 10 10" fill="currentColor">
                        <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" stroke-width="1.2" />
                    </svg>
                </button>
                <h1 class="title">Choose how to power your AI</h1>

                <div class="form-content">
                    <div class="error-message">${this.errorMessage}</div>
                    <div class="provider-label">Select AI Provider:</div>
                    <select
                        class="provider-select"
                        .value=${this.selectedProvider || 'openai'}
                        @change=${this.handleProviderChange}
                        ?disabled=${this.isLoading}
                        tabindex="0"
                    >
                        <option value="openai" ?selected=${this.selectedProvider === 'openai'}>OpenAI</option>
                        <option value="gemini" ?selected=${this.selectedProvider === 'gemini'}>Google Gemini</option>
                        <option value="ollama" ?selected=${this.selectedProvider === 'ollama'}>Ollama (Local)</option>
                    </select>
                    <input
                        type="password"
                        class="api-input"
                        placeholder=${this.getPlaceholderText()}
                        .value=${this.apiKey || ''}
                        @input=${this.handleInput}
                        @keypress=${this.handleKeyPress}
                        @paste=${this.handlePaste}
                        @focus=${() => (this.errorMessage = '')}
                        ?disabled=${this.isLoading || this.selectedProvider === 'ollama'}
                        ?hidden=${this.selectedProvider === 'ollama'}
                        autocomplete="off"
                        spellcheck="false"
                        tabindex="0"
                    />
                    ${this.selectedProvider === 'ollama' ? html`
                        <div class="ollama-info">
                            <p>Ollama runs locally on your machine. Make sure Ollama is installed and running on port 11434.</p>
                            <p>No API key required for local models.</p>
                        </div>
                        <div class="model-select-container">
                            <label class="model-label">Select Model:</label>
                            ${this.isLoadingModels ? html`
                                <div class="model-loading">Loading models...</div>
                            ` : this.ollamaModels.length === 0 ? html`
                                <div class="model-error">No models found. Run 'ollama pull <model>' to download models.</div>
                            ` : html`
                                <select
                                    class="model-select"
                                    .value=${this.selectedOllamaModel || ''}
                                    @change=${this.handleModelChange}
                                    ?disabled=${this.isLoading}
                                    tabindex="0"
                                >
                                    <option value="">Select a model...</option>
                                    ${this.ollamaModels.map(model => html`
                                        <option value="${model.name}" ?selected=${this.selectedOllamaModel === model.name}>
                                            ${model.name}
                                        </option>
                                    `)}
                                </select>
                            `}
                        </div>
                    ` : ''}

                    <button class="action-button" @click=${this.handleSubmit} ?disabled=${isButtonDisabled} tabindex="0">
                        ${this.isLoading ? 'Validating...' : 'Confirm'}
                    </button>

                    <div class="or-text">or</div>

                    <button class="action-button" @click=${this.handleUsePicklesKey}>Use Pickle's API Key</button>
                </div>
            </div>
        `;
    }
}

customElements.define('apikey-header', ApiKeyHeader);
