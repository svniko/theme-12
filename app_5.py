# app.py — Крок 5: Покращення UX

import streamlit as st
from google import genai
from google.genai import types
import time

# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================
st.set_page_config(
    page_title="AI Чатбот з Gemini 3 Pro",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# КОНСТАНТИ
# ============================================================
MODEL_NAME = "gemini-3.1-flash-lite"

# ============================================================
# CSS ДЛЯ КАСТОМІЗАЦІЇ (опціонально)
# ============================================================
st.markdown("""
<style>
    /* Зменшуємо відступ зверху */
    .block-container {
        padding-top: 2rem;
    }

    /* Стиль для повідомлень про помилку */
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ІНІЦІАЛІЗАЦІЯ КЛІЄНТА GEMINI
# ============================================================
@st.cache_resource
def get_gemini_client():
    """Створює та кешує клієнт Gemini."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

client = get_gemini_client()

# ============================================================
# ПЕРЕВІРКА КЛІЄНТА
# ============================================================
if client is None:
    st.error("❌ Не знайдено GOOGLE_API_KEY")
    st.info("""
    **Як налаштувати:**
    1. Створіть файл `.streamlit/secrets.toml`
    2. Додайте рядок: `GOOGLE_API_KEY = "ваш_ключ"`
    3. Перезапустіть застосунок

    Ключ можна отримати на [Google AI Studio](https://aistudio.google.com/apikey)
    """)
    st.stop()

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
        key="temperature_slider",
        help="0 = детерміновано, 1 = максимально креативно"
    )

    max_tokens = st.number_input(
        "Макс. токенів",
        min_value=100,
        max_value=8192,
        value=2048,
        step=100,
        key="max_tokens_input",
        help="Максимальна довжина відповіді"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистити", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📋 Копіювати", use_container_width=True):
            # Формуємо текст для копіювання
            chat_text = "\n\n".join([
                f"{'Користувач' if m['role'] == 'user' else 'Асистент'}: {m['content']}"
                for m in st.session_state.get("messages", [])
            ])
            st.session_state.copy_text = chat_text
            st.toast("Історію скопійовано!")

    # Статистика
    st.divider()
    msg_count = len(st.session_state.get("messages", []))
    st.metric("Повідомлень", msg_count)

    with st.expander("ℹ️ Про застосунок"):
        st.markdown(f"""
        **Модель:** {MODEL_NAME}

        **Можливості:**
        - 💬 Діалог із збереженням контексту
        - ⚡ Стрімінг відповідей
        - ⚙️ Налаштування параметрів
        - 🔄 Очищення історії
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
    """Конвертує історію Streamlit у формат Gemini API."""
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    return contents

def stream_gemini_response(prompt: str, history: list):
    """Генератор для стрімінгу відповіді від Gemini 3 Pro."""
    contents = convert_to_gemini_history(history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    )

    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )

        for chunk in stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n\n❌ **Помилка:** {str(e)}"

# ============================================================
# СИСТЕМНЕ ПРИВІТАННЯ
# ============================================================
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(f"""
        👋 Вітаю! Я AI-асистент на базі **Google Gemini 3 Pro**.

        Я можу допомогти з:
        - 📝 Написанням та редагуванням тексту
        - 💡 Генерацією ідей
        - 🔍 Відповідями на запитання
        - 💻 Поясненням коду

        Просто напишіть своє повідомлення нижче!
        """)

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

    # Генеруємо відповідь зі стрімінгом
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            # Невелика затримка для відображення спінера
            time.sleep(2)

        response = st.write_stream(
            stream_gemini_response(prompt, st.session_state.messages[:-1])
        )

    # Зберігаємо відповідь
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})