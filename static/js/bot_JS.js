let chatBox = document.getElementById('chat-box');
let userInput = document.getElementById('user-input');

function sendMessage() {
    let message = userInput.value;
    if (message.trim() === '') return;

    // Display user message
    displayMessage('user', message);

    // Clear input field
    userInput.value = '';

    // Send message to server
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response
        displayMessage('bot', data.response);
    })
    .catch((error) => {
        console.error('Error:', error);
        displayMessage('bot', 'Sorry, there was an error processing your request.');
    });
}

function displayMessage(sender, message) {
    let messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.innerHTML = message;
    chatBox.appendChild(messageElement);
    
    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}

function toggleChat() {
    let chatContainer = document.getElementById('chat-container');
    chatContainer.style.display = chatContainer.style.display === 'none' ? 'block' : 'none';
}

function newChat() {
    chatBox.innerHTML = '';
    displayMessage('bot', 'How can I assist you today?');
}

// Event listener for the Enter key in the input field
userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Initialize chat when the page loads
document.addEventListener('DOMContentLoaded', function() {
    newChat();
});
