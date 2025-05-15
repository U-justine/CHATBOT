class ChatBot {
    constructor() {
        this.chatBox = document.getElementById('chatBox');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.fileAttachment = document.getElementById('fileAttachment');
        
        this.setupEventListeners();
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

    addMessage(message, isUser, fileUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        messageDiv.textContent = message;

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

document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});