import streamlit as st
from apiClient import ApiClient

# Инициализация клиента API
client = ApiClient(base_url='http://127.0.0.1:8001')

# Настройка страницы
st.set_page_config(page_title="Финансовый Ассистент", page_icon="💰", layout="wide")

# Инициализация session state
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Основной чат"
if "chats" not in st.session_state:
    st.session_state.chats = ["Основной чат"]

# Стилизация CSS
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #f7f8fa;
        }
        .main-container {
            background-color: #ffffff;
            border: 1px solid #e3e6eb;
            border-radius: 10px;
            padding: 15px;
        }
        .chat-message {
            margin-bottom: 10px;
        }
        .chat-user {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 10px;
            text-align: left;
        }
        .chat-assistant {
            background-color: #fff7e6;
            border-radius: 10px;
            padding: 10px;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Боковая панель
with st.sidebar:
    st.title("📈 Ваш Финансовый Ассистент")
    st.write("Добро пожаловать! Управляйте своими финансами эффективно.")

    if st.button("✨ Новый чат"):
        new_chat = f"Чат {len(st.session_state.chats) + 1}"
        st.session_state.chats.append(new_chat)
        st.session_state.messages[new_chat] = []
        st.session_state.current_chat = new_chat

    st.write("### Ваши чаты:")
    for chat in st.session_state.chats:
        if st.sidebar.button(chat, key=chat, use_container_width=True):
            st.session_state.current_chat = chat

# Инициализация сообщений для текущего чата
if st.session_state.current_chat not in st.session_state.messages:
    st.session_state.messages[st.session_state.current_chat] = []

# Основная область
st.title(f"💳 {st.session_state.current_chat}")

# Контейнер для отображения сообщений
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages[st.session_state.current_chat]:
        css_class = "chat-user" if message["role"] == "user" else "chat-assistant"
        with st.container():
            st.markdown(
                f'<div class="{css_class} chat-message">{message["content"]}</div>',
                unsafe_allow_html=True
            )

# Очистка текстового поля
def clear_text():
    st.session_state["user_input"] = ""

# Формирование объединённого текста для API
def create_prompt():
    dialog = st.session_state.messages[st.session_state.current_chat]
    prompt = """"""
    for message in dialog:
        if message["role"] == "user":
            prompt += f"Пользователь: {message['content']}\n"
        else:
            prompt += f"Ассистент: {message['content']}\n"
    return prompt.strip()

# Обработка нажатия кнопки отправки сообщения
def on_click():
    st.session_state.messages[st.session_state.current_chat].append(
        {"role": "user", "content": st.session_state.user_input}
    )

    # Формируем полный диалог для отправки в API
    prompt = create_prompt()

    # Получение ответа от бота
    bot_response = client.send_message(prompt)
    if bot_response.message in ["Error", "None"]:
        bot_response.message = bot_response.error

    # Добавляем ответ бота
    st.session_state.messages[st.session_state.current_chat].append(
        {"role": "assistant", "content": bot_response.message}
    )
    clear_text()

# Поле ввода сообщений и кнопка отправки
with st.container():
    st.divider()
    st.write("### Введите ваше сообщение:")
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Сообщение:", key="user_input", label_visibility="collapsed"
        )
    with col2:
        st.button("Отправить", on_click=on_click, use_container_width=True)
