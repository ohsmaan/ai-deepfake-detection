class DeepfakeDetectorPopup {
    constructor() {
        this.apiUrl = 'http://localhost:8001';
        this.results = [];
        this.init();
    }

    async init() {
        this.bindEvents();
        await this.checkApiStatus();
        await this.loadResults();
        await this.loadFilterState();
    }

    bindEvents() {
        document.getElementById('scanBtn').addEventListener('click', () => this.scanPage());
        document.getElementById('scanVideosBtn').addEventListener('click', () => this.scanVideos());
        document.getElementById('scanAudioBtn').addEventListener('click', () => this.scanAudio());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearResults());
        document.getElementById('filterToggleBtn').addEventListener('click', () => this.toggleFilter());
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const status = response.ok ? 'online' : 'offline';
            this.updateApiStatus(status);
        } catch (error) {
            this.updateApiStatus('offline');
        }
    }

    updateApiStatus(status) {
        const indicator = document.getElementById('apiStatus');
        const text = document.getElementById('apiStatusText');
        
        indicator.className = `status-indicator ${status}`;
        text.textContent = `API: ${status === 'online' ? 'Connected' : 'Disconnected'}`;
    }

    async scanPage() {
        this.setStatus('Scanning page images...', 'loading');
        this.disableButton('scanBtn');

        try {
            // Send message to content script to scan images
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, { action: 'scanImages' });
            
            if (response && response.images) {
                await this.processImages(response.images);
            } else {
                this.setStatus('No images found on page', 'error');
            }
        } catch (error) {
            console.error('Scan error:', error);
            this.setStatus('Error scanning page', 'error');
        }

        this.enableButton('scanBtn');
    }

    async scanVideos() {
        this.setStatus('Scanning page videos...', 'loading');
        this.disableButton('scanVideosBtn');

        try {
            // Send message to content script to scan videos
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, { action: 'scanVideos' });
            
            if (response && response.videos) {
                this.setStatus(`Found ${response.videos.length} videos on page`, 'success');
                // Note: Video analysis is done directly in content script for speed
            } else {
                this.setStatus('No videos found on page', 'error');
            }
        } catch (error) {
            console.error('Video scan error:', error);
            this.setStatus('Error scanning videos', 'error');
        }

        this.enableButton('scanVideosBtn');
    }

    async scanAudio() {
        this.setStatus('Scanning page audio...', 'loading');
        this.disableButton('scanAudioBtn');

        try {
            // Send message to content script to scan audio
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, { action: 'scanAudio' });
            
            if (response && response.audio) {
                this.setStatus(`Found ${response.audio.length} audio sources on page`, 'success');
                // Note: Audio analysis is done directly in content script for speed
            } else {
                this.setStatus('No audio sources found on page', 'error');
            }
        } catch (error) {
            console.error('Audio scan error:', error);
            this.setStatus('Error scanning audio', 'error');
        }

        this.enableButton('scanAudioBtn');
    }

    async processImages(images) {
        this.setStatus(`Processing ${images.length} images...`, 'loading');
        
        const results = [];
        for (let i = 0; i < images.length; i++) {
            const image = images[i];
            this.setStatus(`Processing image ${i + 1}/${images.length}...`, 'loading');
            
            try {
                const result = await this.detectDeepfake(image);
                results.push(result);
            } catch (error) {
                console.error(`Error processing image ${image.src}:`, error);
                results.push({
                    url: image.src,
                    is_deepfake: false,
                    confidence: 0,
                    error: error.message
                });
            }
        }

        this.results = results;
        await this.saveResults();
        await this.displayResults();
        this.setStatus(`Found ${results.length} images`, 'success');
    }

    async detectDeepfake(image) {
        // Convert image to blob for upload
        const response = await fetch(image.src);
        const blob = await response.blob();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', blob, 'image.jpg');

        // Send to API
        const apiResponse = await fetch(`${this.apiUrl}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!apiResponse.ok) {
            throw new Error(`API error: ${apiResponse.status}`);
        }

        const result = await apiResponse.json();
        return {
            url: image.src,
            is_deepfake: result.is_deepfake,
            confidence: result.confidence,
            processing_time: result.processing_time,
            model_used: result.model_used,
            claude_analysis: result.claude_analysis
        };
    }

    async displayResults() {
        const resultsContainer = document.getElementById('results');
        
        if (this.results.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results">No content scanned yet</div>';
            return;
        }

        const resultsHtml = this.results.map(result => {
            const status = result.is_deepfake ? 'fake' : 'real';
            const detectionType = result.detection_type || 'unknown';
            
            // Determine label and icon based on detection type
            let label, icon;
            if (detectionType === 'deepfake') {
                label = '🤖 Deepfake';
                icon = '🤖';
            } else if (detectionType === 'possible_ai') {
                label = '⚠️ Possible AI';
                icon = '⚠️';
            } else if (detectionType === 'real') {
                label = '✅ Real';
                icon = '✅';
            } else if (detectionType === 'uncertain') {
                label = '❓ Uncertain';
                icon = '❓';
            } else {
                label = result.is_deepfake ? '🤖 AI Generated' : '✅ Real';
                icon = result.is_deepfake ? '🤖' : '✅';
            }
            
            const confidence = Math.round(result.confidence * 100);
            
            return `
                <div class="result-item ${status}">
                    <div class="result-header">
                        <span class="result-label">${label}</span>
                        <span class="result-confidence">${confidence}%</span>
                    </div>
                    <div class="result-url">${this.truncateUrl(result.url)}</div>
                    ${result.analysis ? `<div class="result-analysis">${result.analysis}</div>` : ''}
                </div>
            `;
        }).join('');

        resultsContainer.innerHTML = resultsHtml;
    }

    truncateUrl(url) {
        try {
            const urlObj = new URL(url);
            return urlObj.hostname + urlObj.pathname.substring(0, 30) + '...';
        } catch {
            return url.substring(0, 50) + '...';
        }
    }

    async clearResults() {
        this.results = [];
        await this.saveResults();
        await this.displayResults();
        this.setStatus('Results cleared', 'success');
    }

    setStatus(text, type = 'info') {
        const statusIcon = document.querySelector('.status-icon');
        const statusText = document.querySelector('.status-text');
        
        const icons = {
            info: '🔍',
            loading: '⏳',
            success: '✅',
            error: '❌'
        };

        statusIcon.textContent = icons[type] || icons.info;
        statusText.textContent = text;
        
        if (type === 'loading') {
            statusIcon.classList.add('loading');
        } else {
            statusIcon.classList.remove('loading');
        }
    }

    disableButton(buttonId) {
        document.getElementById(buttonId).disabled = true;
    }

    enableButton(buttonId) {
        document.getElementById(buttonId).disabled = false;
    }

    async saveResults() {
        await chrome.storage.local.set({ results: this.results });
    }

    async loadResults() {
        const data = await chrome.storage.local.get(['results']);
        this.results = data.results || [];
        await this.displayResults();
    }

    async loadFilterState() {
        const data = await chrome.storage.sync.get(['filterMode']);
        const filterMode = data.filterMode || false;
        
        const toggleBtn = document.getElementById('filterToggleBtn');
        const toggleText = document.getElementById('filterToggleText');
        
        if (filterMode) {
            toggleBtn.classList.add('active');
            toggleText.textContent = 'Disable Filter';
            toggleBtn.style.background = '#ff4444';
        } else {
            toggleBtn.classList.remove('active');
            toggleText.textContent = 'Enable Filter';
            toggleBtn.style.background = '#666';
        }
    }

    async toggleFilter() {
        const data = await chrome.storage.sync.get(['filterMode']);
        const currentMode = data.filterMode || false;
        const newMode = !currentMode;
        
        await chrome.storage.sync.set({ filterMode: newMode });
        
        // Update popup UI
        const toggleBtn = document.getElementById('filterToggleBtn');
        const toggleText = document.getElementById('filterToggleText');
        
        if (newMode) {
            toggleBtn.classList.add('active');
            toggleText.textContent = 'Disable Filter';
            toggleBtn.style.background = '#ff4444';
            this.setStatus('AI Content Filter: ENABLED', 'success');
        } else {
            toggleBtn.classList.remove('active');
            toggleText.textContent = 'Enable Filter';
            toggleBtn.style.background = '#666';
            this.setStatus('AI Content Filter: DISABLED', 'info');
        }
        
        // Send message to content script to update filter state
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, { 
                action: 'updateFilterMode', 
                filterMode: newMode 
            });
        } catch (error) {
            console.error('Error updating filter mode:', error);
        }
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DeepfakeDetectorPopup();
});
