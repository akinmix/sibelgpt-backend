// Sohbet geçmişini Local Storage'da tutmak için anahtar
const HISTORY_STORAGE_KEY = 'sibelgpt_conversations';

let currentConversation = [];
let chatBox, userInput, newChatButton, historyList, splashScreen, mainInterface;

// ✅ Görsel üretim kontrolü ve işleyici
async function istekGorselIseYonet(input) {
  const lower = input.toLowerCase();
  const anahtarKelimeler = [
    "çiz", "görsel", "resim", "fotoğraf", "bir şey çiz", "görsel üret", "resmini yap", 
    "çizimini yap", "şunun görselini", "şunu çiz", "görselini oluştur"
  ];
  const istekGorselMi = anahtarKelimeler.some(kelime => lower.includes(kelime));
  if (!istekGorselMi) return null;

  try {
    const res = await fetch("https://sibelgpt-backend.onrender.com/image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: input })
    });
    const data = await res.json();

    if (data.image_url) {
      return `
        <div style="display: flex; flex-direction: column; align-items: flex-start;">
          <img src="${data.image_url}" alt="Üretilen Görsel" style="max-width: 100%; border-radius: 8px; margin-bottom: 8px;" />
          <button onclick="indirGorsel('${data.image_url}')" style="padding: 6px 12px; font-size: 14px; border: none; border-radius: 4px; background-color: #6a5acd; color: white; cursor: pointer;">
            📥 İndir
          </button>
        </div>
      `;
    } else {
      return "❗ Görsel üretilemedi, lütfen tekrar deneyin.";
    }
  } catch (e) {
    console.error("Görsel üretim hatası:", e);
    return "⚠️ Görsel üretim sırasında bir hata oluştu.";
  }
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  appendMessage("Sen", message, "user", true);
  userInput.value = "";

  const gorselHTML = await istekGorselIseYonet(message);
  if (gorselHTML !== null) {
    appendMessage("SibelGPT", gorselHTML, "bot", true);
    return;
  }

  try {
    const response = await fetch("https://sibelgpt-backend.onrender.com/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: message }),
    });

    const data = await response.json();
    const reply = data.reply || "❌ Bir hata oluştu. Lütfen tekrar deneyin.";
    appendMessage("SibelGPT", reply, "bot", true);

  } catch (error) {
    appendMessage("SibelGPT", "❌ Bir hata oluştu. Sunucuya ulaşılamıyor.", "bot", true);
    console.error("Mesaj gönderirken hata:", error);
  }
}

function appendMessage(sender, text, role, addToHistory = false) {
  const messageElem = document.createElement("div");
  messageElem.className = "message " + role;
  messageElem.innerHTML = `<strong>${sender}:</strong> ${text}`;
  chatBox.appendChild(messageElem);

  if (addToHistory) {
    currentConversation.push({ sender, text, role });
  }

  setTimeout(() => {
    chatBox.scrollTop = chatBox.scrollHeight;
  }, 100);
}

function indirGorsel(url) {
  const link = document.createElement('a');
  link.href = url;
  link.download = 'sibelgpt-image.jpg';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function handleInputKeyPress(event) {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendMessage();
  }
}

function loadConversations() {
  const conversationsJson = localStorage.getItem(HISTORY_STORAGE_KEY);
  try {
    return conversationsJson ? JSON.parse(conversationsJson) : [];
  } catch (e) {
    console.error("Sohbet geçmişi yüklenirken hata:", e);
    return [];
  }
}

function saveConversations(conversations) {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(conversations));
  } catch (e) {
    console.error("Sohbet geçmişi kaydedilirken hata:", e);
  }
}

function saveCurrentConversation() {
  if (currentConversation.length <= 1 && currentConversation[0] && currentConversation[0].role === 'bot' && currentConversation[0].text.includes('Merhaba!')) return;
  if (currentConversation.length === 0) return;
  const chatId = Date.now();
  const title = generateConversationTitle(currentConversation);
  const conversations = loadConversations();
  conversations.unshift({ id: chatId, title: title, messages: currentConversation });
  saveConversations(conversations);
  displayHistory();
}

function generateConversationTitle(conversation) {
  const firstUserMessage = conversation.find(msg => msg.role === 'user');
  if (firstUserMessage?.text) {
    const text = firstUserMessage.text.trim();
    return text.length > 30 ? text.substring(0, text.lastIndexOf(' ', 30)) + '...' : text;
  }
  return "Yeni Sohbet";
}

function clearChat() {
  chatBox.innerHTML = '';
  currentConversation = [];
  highlightSelectedChat(null);
}

function displayHistory() {
  const conversations = loadConversations();
  historyList.innerHTML = '';
  if (conversations.length === 0) {
    const placeholder = document.createElement('li');
    placeholder.textContent = 'Henüz kaydedilmiş sohbet yok.';
    placeholder.style.cursor = 'default';
    placeholder.style.opacity = '0.7';
    historyList.appendChild(placeholder);
    return;
  }
  conversations.forEach(conv => {
    const listItem = document.createElement('li');
    listItem.textContent = conv.title;
    listItem.setAttribute('data-chat-id', conv.id);
    historyList.appendChild(listItem);
  });
}

function loadConversation(chatId) {
  saveCurrentConversation();
  const conversations = loadConversations();
  const conversationToLoad = conversations.find(conv => conv.id == chatId);
  if (conversationToLoad) {
    chatBox.innerHTML = '';
    currentConversation = [];
    conversationToLoad.messages.forEach((msg) => {
      appendMessage(msg.sender, msg.text, msg.role, false);
    });
    currentConversation = JSON.parse(JSON.stringify(conversationToLoad.messages));
    highlightSelectedChat(chatId);
    userInput.focus();
  }
}

function highlightSelectedChat(chatId) {
  historyList.querySelectorAll('li').forEach(li => li.classList.remove('selected'));
  if (chatId !== null) {
    const selectedItem = historyList.querySelector(`li[data-chat-id="${chatId}"]`);
    if (selectedItem) selectedItem.classList.add('selected');
  }
}

window.addEventListener("load", () => {
  chatBox = document.getElementById("chat-box");
  userInput = document.getElementById("user-input");
  newChatButton = document.querySelector(".new-chat-button button");
  historyList = document.getElementById("history-list");
  splashScreen = document.getElementById("splash-screen");
  mainInterface = document.getElementById("main-interface");

  splashScreen.addEventListener('animationend', () => {
    splashScreen.style.opacity = 0;
    setTimeout(() => {
      splashScreen.style.display = "none";
      mainInterface.style.display = "flex";
      initializeChatInterface();
      const wrapper = document.getElementById("video-wrapper");
      if (wrapper) wrapper.style.display = "flex";
    }, 300);
  });

  userInput.addEventListener("keypress", handleInputKeyPress);
  newChatButton.addEventListener("click", handleNewChat);
  historyList.addEventListener("click", handleHistoryClick);

  const initialBotMessageElement = chatBox.querySelector('.bot-message');
  if (initialBotMessageElement) {
    currentConversation.push({
      sender: 'SibelGPT',
      text: initialBotMessageElement.textContent.replace('SibelGPT:', '').trim(),
      role: 'bot'
    });
  }

  setTimeout(() => { userInput.focus(); }, 100);
});

function initializeChatInterface() {
  displayHistory();
}

function playIntroVideo() {
  const video = document.getElementById("intro-video");
  const wrapper = document.getElementById("video-wrapper");
  if (video && wrapper) {
    video.muted = false;
    video.currentTime = 0;
    wrapper.classList.remove("fade-out");
    wrapper.style.display = "flex";
    video.play().catch(e => console.warn("Video oynatılamadı:", e));
    video.onended = () => {
      wrapper.classList.add("fade-out");
      setTimeout(() => {
        wrapper.style.display = "none";
        wrapper.classList.remove("fade-out");
      }, 1500);
    };
  }
}

function handleNewChat() {
  saveCurrentConversation();
  clearChat();
  userInput.focus();
}

function handleHistoryClick(event) {
  const clickedElement = event.target;
  if (clickedElement.tagName === 'LI' && clickedElement.hasAttribute('data-chat-id')) {
    const chatId = clickedElement.getAttribute('data-chat-id');
    loadConversation(chatId);
    userInput.focus();
  }
}

window.addEventListener('beforeunload', () => {
  saveCurrentConversation();
});
