let chatOpenedBefore = false;

function toggleChat() {
    const chatContainer = document.getElementById("chat-container");
    if (chatContainer.style.display === "none" || chatContainer.style.display === "") {
        chatContainer.style.display = "block";
        if (!chatOpenedBefore) {
            appendMessage("bot", "Xin chào, tôi là chatbot hỗ trợ và tư vấn về tự kỷ ở trẻ em, tôi có thể giúp gì cho bạn?");
            chatOpenedBefore = true;
        }
    } else {
        chatContainer.style.display = "none";
    }
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
        appendMessage("bot", "Xin lỗi. Hệ thống đang bận, vui lòng thử lại sau!");
    }
}

let conversationHistory = [];

function appendMessage(sender, message) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.className = `message ${sender}`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;

    typeWriterEffect(message, messageElement);

    chatBox.scrollTop = chatBox.scrollHeight;

    conversationHistory.push({ sender: sender, message: message });
}

function typeWriterEffect(message, element) {
    let i = 0;
    function type() {
        if (i < message.length) {
            element.textContent += message.charAt(i);
            i++;
            setTimeout(type, 30); // Adjust the speed here (milliseconds)
        }
    }
    type();
}

function newChat() {
    document.getElementById("chat-box").innerHTML = '';
    appendMessage("bot", "Xin chào, tôi là chatbot hỗ trợ và tư vấn về tự kỷ ở trẻ em, tôi có thể giúp gì cho bạn?");
}
