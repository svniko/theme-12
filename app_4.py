# app.py — Крок 4: Інтеграція з Google Gemini 3 Pro

import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================
st.set_page_config(
    page_title="AI Чатбот з Gemini 3.1 Flash Lite",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# КОНСТАНТИ
# ============================================================
MODEL_NAME = "gemini-3.1-flash-lite"

# ============================================================
# ІНІЦІАЛІЗАЦІЯ КЛІЄНТА GEMINI
# ============================================================
@st.cache_resource
def get_gemini_client():
    """
    Створює та кешує клієнт Gemini.
    Клієнт створюється один раз і перевикористовується.
    """
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Не знайдено GOOGLE_API_KEY у secrets.toml")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_gemini_client()

# ============================================================
# ЗАГОЛОВОК
# ============================================================
st.title("🤖 AI Чатбот з Gemini 3 Pro")

# ============================================================
# БІЧНА ПАНЕЛЬ
# ============================================================
with st.sidebar:
    st.header("⚙️ Налаштування")

    st.info(f"Модель: **{MODEL_NAME}**")

    temperature = st.slider(
        "Температура",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="temperature_slider"
    )

    st.divider()

    if st.button("🗑️ Очистити історію", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Інформація про застосунок
    with st.expander("ℹ️ Про застосунок"):
        st.markdown("""
        Цей чатбот використовує модель **Google Gemini 3 Pro**
        для генерації відповідей.

        **Можливості:**
        - Діалог з контекстом
        - Налаштування параметрів генерації
        - Стрімінг відповідей
        """)

# ============================================================
# ІНІЦІАЛІЗАЦІЯ СТАНУ
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================
def convert_to_gemini_history(messages: list) -> list[types.Content]:
    """
    Конвертує історію Streamlit у формат Gemini API.

    Streamlit: [{"role": "user/assistant", "content": "..."}]
    Gemini: [Content(role="user/model", parts=[Part(text="...")])]
    """
    contents = []
    for msg in messages:
        # Gemini використовує "model" замість "assistant"
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    return contents

def stream_gemini_response(prompt: str, history: list):
    """
    Генератор для стрімінгу відповіді від Gemini 3 Pro.

    Yields:
        Частини тексту відповіді по мірі їх генерації.
    """
    # Конвертуємо історію та додаємо новий запит
    contents = convert_to_gemini_history(history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    )

    # Створюємо стрім
    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=2048,
        )
    )

    # Yield'имо частини відповіді
    for chunk in stream:
        if chunk.text:
            yield chunk.text

# ============================================================
# ВІДОБРАЖЕННЯ ІСТОРІЇ ЧАТУ
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# ОБРОБКА НОВОГО ПОВІДОМЛЕННЯ
# ============================================================
if prompt := st.chat_input("Введіть ваше повідомлення..."):
    # Додаємо повідомлення користувача
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Відображаємо повідомлення користувача
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генеруємо та відображаємо відповідь зі стрімінгом
    with st.chat_message("assistant"):
        # st.write_stream повертає повний текст після завершення
        response = st.write_stream(
            stream_gemini_response(prompt, st.session_state.messages[:-1])
        )

    # Зберігаємо відповідь в історії
    st.session_state.messages.append({"role": "assistant", "content": response})