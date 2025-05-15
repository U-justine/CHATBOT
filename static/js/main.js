class ChatBot {
    constructor() {
        // Initialize DOM elements
        this.chatBox = document.getElementById('chatBox');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.fileAttachment = document.getElementById('fileAttachment');

        // Set up event listeners
        this.setupEventListeners();

        // Show welcome message
        this.addMessage("Hello! How can I help you today?", false);
    }

    setupEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        this.fileAttachment.addEventListener('change', (e) => this.handleFileAttachment(e));
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    formatTimestamp() {
        return new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    async handleFileAttachment(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                this.addMessage(`File uploaded: ${file.name}`, true);
                this.addMessage(`File received. You can access it here: ${data.url}`, false);
            }
        } catch (error) {
            console.error('File upload failed:', error);
            alert('File upload failed. Please try again.');
        }
    }

    addMessage(message, isUser, fileUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', isUser ? 'user-message' : 'bot-message');

        const messageText = document.createElement('p');
        messageText.textContent = message;
        messageDiv.appendChild(messageText);

        const timestamp = document.createElement('div');
        timestamp.classList.add('timestamp');
        timestamp.textContent = this.formatTimestamp();
        messageDiv.appendChild(timestamp);

        if (fileUrl) {
            const fileAttachment = document.createElement('div');
            fileAttachment.classList.add('file-attachment');
            fileAttachment.innerHTML = `
                <i class="fas fa-file"></i>
                <a href="${fileUrl}" target="_blank">Download Attachment</a>
            `;
            messageDiv.appendChild(fileAttachment);
        }

        this.chatBox.appendChild(messageDiv);
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
    }

    async sendMessage() {
        const message = this.userInput.value.trim();
        if (message === '') return;

        this.addMessage(message, true);
        this.userInput.value = '';
        this.showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();

            setTimeout(() => {
                this.hideTypingIndicator();
                this.addMessage(data.response, false);
            }, 1000);
        } catch (error) {
            console.error('Error:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, there was an error processing your request.', false);
        }
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});