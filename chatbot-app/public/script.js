class GeminiChatApp {
    constructor() {
        this.currentConversationId = null;
        this.selectedImage = null;
        this.initializeElements();
        this.attachEventListeners();
        this.autoResizeTextarea();
        this.startNewConversation();
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.imageInput = document.getElementById('imageInput');
        this.imageUploadBtn = document.getElementById('imageUploadBtn');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImage = document.getElementById('previewImage');
        this.removeImageBtn = document.getElementById('removeImage');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.loadingIndicator = document.getElementById('loadingIndicator');
    }

    attachEventListeners() {
        // Send message events
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (e.shiftKey) {
                    // Allow Shift+Enter for line breaks - don't prevent default
                    return;
                } else {
                    // Enter without Shift sends the message
                    e.preventDefault();
                    this.sendMessage();
                }
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
        this.messageInput.addEventListener('paste', () => {
            setTimeout(() => this.autoResizeTextarea(), 0);
        });

        // Image upload events
        this.imageUploadBtn.addEventListener('click', () => this.imageInput.click());
        this.imageInput.addEventListener('change', (e) => this.handleImageSelect(e));
        this.removeImageBtn.addEventListener('click', () => this.removeSelectedImage());

        // New chat event
        this.newChatBtn.addEventListener('click', () => this.startNewConversation());

        // Image click to view full size
        this.chatMessages.addEventListener('click', (e) => {
            if (e.target.classList.contains('message-image')) {
                this.viewImageFullSize(e.target.src);
            }
        });
    }

    async startNewConversation() {
        try {
            const response = await fetch('/api/conversation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error('Failed to start new conversation');
            }

            const data = await response.json();
            this.currentConversationId = data.conversationId;
            this.clearChat();

        } catch (error) {
            console.error('Error starting new conversation:', error);
            this.showError('Failed to start new conversation. Please refresh the page.');
        }
    }

    clearChat() {
        this.chatMessages.innerHTML = '';
        this.removeSelectedImage();
        this.messageInput.value = '';
        this.autoResizeTextarea();
    }

    handleImageSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.selectedImage = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                this.previewImage.src = e.target.result;
                this.imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }

    removeSelectedImage() {
        this.selectedImage = null;
        this.imagePreview.style.display = 'none';
        this.previewImage.src = '';
        this.imageInput.value = '';
    }

    autoResizeTextarea() {
        // Reset height to auto to get the actual content height
        this.messageInput.style.height = 'auto';
        
        // Calculate the height needed, with a maximum limit
        const maxHeight = 120; // About 5-6 lines
        const minHeight = 24;  // Single line height
        const contentHeight = this.messageInput.scrollHeight;
        
        // Set the height, clamped between min and max
        const newHeight = Math.max(minHeight, Math.min(contentHeight, maxHeight));
        this.messageInput.style.height = newHeight + 'px';
        
        // Show/hide scrollbar based on whether we hit the max height
        if (contentHeight > maxHeight) {
            this.messageInput.style.overflowY = 'auto';
        } else {
            this.messageInput.style.overflowY = 'hidden';
        }
    }

    async sendMessage() {
        const messageText = this.messageInput.value.trim();

        if (!messageText && !this.selectedImage) {
            return;
        }

        if (!this.currentConversationId) {
            this.showError('No active conversation. Please start a new chat.');
            return;
        }

        // Disable input while processing
        this.setInputEnabled(false);
        this.showLoading(true);

        try {
            // Store reference to selected image before clearing
            const selectedImageFile = this.selectedImage;

            // Create user message and show it immediately
            const userMessage = {
                id: Date.now().toString(), // Temporary ID
                role: 'user',
                content: messageText,
                timestamp: new Date()
            };

            // Show user message immediately (without image preview for now)
            this.addMessageToChat(userMessage);

            // Clear input immediately after showing message
            this.messageInput.value = '';
            this.autoResizeTextarea();
            this.removeSelectedImage();

            // Create FormData for the request
            const formData = new FormData();
            formData.append('message', messageText);

            if (selectedImageFile) {
                formData.append('image', selectedImageFile);
            }

            // Send message to server
            const response = await fetch(`/api/conversation/${this.currentConversationId}/message`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to send message');
            }

            const data = await response.json();

            // Only add assistant message (user message already shown)
            this.addMessageToChat(data.assistantMessage);

        } catch (error) {
            console.error('Error sending message:', error);
            this.showError(error.message || 'Failed to send message. Please try again.');
        } finally {
            this.setInputEnabled(true);
            this.showLoading(false);
            this.messageInput.focus();
        }
    }


    addMessageToChat(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message.content;

        messageDiv.appendChild(contentDiv);

        // Add image if present (single image - backward compatibility)
        if (message.image) {
            const imageElement = document.createElement('img');
            imageElement.src = `/uploads/${message.image}`;
            imageElement.className = 'message-image';
            imageElement.alt = 'Shared image';
            messageDiv.appendChild(imageElement);
        }

        // Add multiple images if present
        if (message.images && message.images.length > 0) {
            const imagesContainer = document.createElement('div');
            imagesContainer.className = 'message-images-container';

            message.images.forEach((imageName, index) => {
                const imageElement = document.createElement('img');
                imageElement.src = `/uploads/${imageName}`;
                imageElement.className = 'message-image';
                imageElement.alt = `Generated image ${index + 1}`;
                imagesContainer.appendChild(imageElement);
            });

            messageDiv.appendChild(imagesContainer);
        }

        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = this.formatTimestamp(new Date(message.timestamp));
        messageDiv.appendChild(timestampDiv);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatTimestamp(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    setInputEnabled(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendBtn.disabled = !enabled;
        this.imageUploadBtn.disabled = !enabled;
        this.newChatBtn.disabled = !enabled;
    }

    showLoading(show) {
        this.loadingIndicator.style.display = show ? 'flex' : 'none';
    }

    showError(message) {
        const errorMessage = {
            role: 'assistant',
            content: `❌ Error: ${message}`,
            timestamp: new Date()
        };
        this.addMessageToChat(errorMessage);
    }

    viewImageFullSize(src) {
        // Create a modal to view image in full size
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
            cursor: pointer;
        `;

        const img = document.createElement('img');
        img.src = src;
        img.style.cssText = `
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
        `;

        modal.appendChild(img);
        document.body.appendChild(modal);

        // Close modal when clicked
        modal.addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        // Close modal with Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(modal);
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GeminiChatApp();
});