import streamlit as st

from apiClient import ApiClient

client = ApiClient('http://backend:8001')

# Настройка страницы
st.set_page_config(page_title="Чат-бот", page_icon="🦊", layout="wide")

# Инициализация session state
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if "chats" not in st.session_state:
    st.session_state.chats = ["Chat 1"]

# Боковая панель
with st.sidebar:
    st.title("💭 Чаты")
    
    if st.button("✨ Новый чат"):
        new_chat = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats.append(new_chat)
        st.session_state.messages[new_chat] = []
        st.session_state.current_chat = new_chat
    
    for chat in st.session_state.chats:
        if st.sidebar.button(chat, key=chat, use_container_width=True):
            st.session_state.current_chat = chat

# Инициализация сообщений для текущего чата
if st.session_state.current_chat not in st.session_state.messages:
    st.session_state.messages[st.session_state.current_chat] = []

# Основная область чата
st.title(f"🦊 Чат-бот - {st.session_state.current_chat}")

# Контейнер для сообщений
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages[st.session_state.current_chat]:
        with st.chat_message(message["role"]):
            st.write(message["content"])



def clear_text():
    st.session_state["user_input"] = ""

def on_click():
    st.session_state.messages[st.session_state.current_chat].append(
        {"role": "user", "content": user_input}
    )
    
    # Ответ бота
    bot_response = client.send_message(user_input)
    clear_text()
    # Добавляем ответ бота
    st.session_state.messages[st.session_state.current_chat].append(
        {"role": "assistant", "content": bot_response.message}
    )

# Форма ввода
with st.container():
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("Сообщение:", key="user_input", label_visibility="collapsed")
    with col2:
        # Теперь кнопка будет вызывать callback перед обновлением
        send_button = st.button("Отправить", on_click=on_click, use_container_width=True)


