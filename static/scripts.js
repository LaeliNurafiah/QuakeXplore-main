document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    // Tambah pesan pengguna ke chat-box
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML += `<div>User: ${userInput}</div>`;

    // Kirim pesan ke server
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Tambah balasan chatbot ke chat-box
        chatBox.innerHTML += `<div>Bot: ${data.message}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll ke bawah
    });

    // Kosongkan input pengguna
    document.getElementById('user-input').value = '';
});

async function sendMessage() {
    const userMessage = document.getElementById('userInput').value;
    const chatlogs = document.querySelector('.chatlogs');
    if (userMessage.trim() === "") {
        return;
    }
    
    // Add user message
    addMessageToChat('user', userMessage);
    
    // Clear input
    document.getElementById('userInput').value = '';
    
    // Show typing indicator
    const typingIndicator = addTypingIndicator();
    
    // Simulate a delay before the bot responds (you can adjust this or remove it if using a real API)
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userMessage })
        });
        const result = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        // Add bot response
        if (result.response) {
            addMessageToChat('bot', result.response);
        } else {
            addMessageToChat('bot', 'Maaf, terjadi kesalahan. Silakan coba lagi.');
        }
    } catch (error) {
        console.error('Error:', error);
        
        // Remove typing indicator
        typingIndicator.remove();
        
        addMessageToChat('bot', 'Maaf, terjadi kesalahan. Silakan coba lagi.');
    }
    
    chatlogs.scrollTop = chatlogs.scrollHeight;
}

function addMessageToChat(sender, message) {
    const chatlogs = document.querySelector('.chatlogs');
    const messageContainer = document.createElement('div');
    messageContainer.className = 'message-container';
    
    if (sender === 'bot') {
        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender}-avatar`;
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        messageContainer.appendChild(avatar);
    }
    
    const messageElement = document.createElement('div');
    messageElement.className = sender === 'user' ? 'user-message' : 'bot-response';
    
    // Split the message into paragraphs
    const paragraphs = message.split('\n').filter(p => p.trim() !== '');
    paragraphs.forEach(p => {
        const paragraph = document.createElement('p');
        paragraph.textContent = p;
        messageElement.appendChild(paragraph);
    });
    
    messageContainer.appendChild(messageElement);
    chatlogs.appendChild(messageContainer);
}


function addTypingIndicator() {
    const chatlogs = document.querySelector('.chatlogs');
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';
    chatlogs.appendChild(typingIndicator);
    return typingIndicator;
}

// Add event listener for Enter key press
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
