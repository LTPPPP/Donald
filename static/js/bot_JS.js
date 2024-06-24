function toggleChat() {
    const chatContainer = document.getElementById("chat-container");
    chatContainer.style.display = chatContainer.style.display === "none" || chatContainer.style.display === "" ? "block" : "none";
}

document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    userInput.value = "";

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });
        const data = await response.json();
        if (data.response) {
            appendMessage("bot", data.response);
        } else if (data.error) {
            appendMessage("bot", `Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
        appendMessage("bot", "Sorry, there was an error processing your request.");
    }
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.className = `message ${sender}`;
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}