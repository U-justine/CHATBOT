// static/js/main.js
class ChatBot {
    constructor() {
        this.chatBox = document.getElementById('chatBox');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.fileAttachment = document.getElementById('fileAttachment');
        this.loginForm = document.getElementById('loginForm');
        this.logoutBtn = document.getElementById('logoutBtn');
        this.authContainer = document.getElementById('authContainer');
        this.mainContainer = document.getElementById('mainContainer');
        this.userEmail = document.getElementById('userEmail');
        
        this.setupEventListeners();
        this.checkAuth();
    }

    setupEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        this.fileAttachment.addEventListener('change', (e) => this.handleFileAttachment(e));
        this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        this.logoutBtn.addEventListener('click', () => this.handleLogout());
    }

    checkAuth() {
        const token = localStorage.getItem('chatbot_token');
        if (token) {
            this.showMainInterface(localStorage.getItem('user_email'));
        }
    }

    async handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();
            if (data.token) {
                localStorage.setItem('chatbot_token', data.token);
                localStorage.setItem('user_email', email);
                this.showMainInterface(email);
            }
        } catch (error) {
            console.error('Login failed:', error);
            alert('Login failed. Please try again.');
        }
    }

    handleLogout() {
        localStorage.removeItem('chatbot_token');
        localStorage.removeItem('user_email');
        this.authContainer.style.display = 'flex';
        this.mainContainer.style.display = 'none';
    }

    showMainInterface(email) {
        this.authContainer.style.display = 'none';
        this.mainContainer.style.display = 'block';
        this.userEmail.textContent = email;
        this.addMessage("Hello! How can I help you today?", false);
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    formatTimestamp() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    async handleFileAttachment(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('chatbot_token')}`
                }
            });

            const data = await response.json();
            if (data.url) {
                this.addMessage(`File uploaded: ${file.name}`, true, data.url);
            }
        } catch (error) {
            console.error('File upload failed:', error);
            alert('File upload failed. Please try again.');
        }
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
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('chatbot_token')}`
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            setTimeout(() => {
                this.hideTypingIndicator();
                this.addMessage(data.response, false);
            }, 1000); // Simulated delay for typing effect
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