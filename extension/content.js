class DeepfakeDetectorContent {
    constructor() {
        this.settings = {};
        this.init();
    }

    async init() {
        this.loadSettings();
        this.bindEvents();
    }

    async loadSettings() {
        const data = await chrome.storage.sync.get(['autoScan', 'showOverlay']);
        this.settings = data;
    }

    bindEvents() {
        // Listen for messages from popup and background
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'scanImages') {
                this.scanImages().then(sendResponse);
                return true; // Keep message channel open for async response
            } else if (request.action === 'showNotification') {
                this.showNotification(request.message, request.type);
            }
        });
    }

    async scanImages() {
        const images = Array.from(document.querySelectorAll('img')).filter(img => 
            this.isValidImage(img)
        );

        console.log(`Found ${images.length} valid images on page`);
        
        return { images: images.map(img => ({ src: img.src, alt: img.alt })) };
    }

    isValidImage(img) {
        // Check if image is valid and visible
        if (!img.src || img.src.startsWith('data:') || img.src.startsWith('blob:')) {
            return false;
        }

        // Check minimum size (avoid tiny icons/avatars)
        const rect = img.getBoundingClientRect();
        if (rect.width < 100 || rect.height < 100) {
            return false;
        }

        // Check if image is visible
        const style = window.getComputedStyle(img);
        if (style.display === 'none' || style.visibility === 'hidden') {
            return false;
        }

        return true;
    }

    showNotification(message, type = 'info') {
        // Remove any existing notifications
        const existingNotification = document.getElementById('deepfake-notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Create notification element
        const notification = document.createElement('div');
        notification.id = 'deepfake-notification';
        notification.className = `deepfake-notification ${type}`;
        
        // Set icon based on type
        const icons = {
            info: '🔍',
            success: '✅',
            warning: '🤖',
            error: '❌'
        };
        
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icons[type] || icons.info}</span>
                <span class="notification-text">${message}</span>
                <button class="notification-close">×</button>
            </div>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${this.getNotificationColor(type)};
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            max-width: 300px;
            animation: slideIn 0.3s ease-out;
        `;

        // Add close button functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            margin-left: 8px;
            padding: 0;
            line-height: 1;
        `;
        
        closeBtn.addEventListener('click', () => {
            notification.remove();
        });

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }

    getNotificationColor(type) {
        const colors = {
            info: '#667eea',
            success: '#28a745',
            warning: '#ffc107',
            error: '#dc3545'
        };
        return colors[type] || colors.info;
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize content script
new DeepfakeDetectorContent();
