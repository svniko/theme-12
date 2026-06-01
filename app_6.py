# app.py — Крок 6: Інтеграція з LangGraph-агентом (Gemini)

import uuid

import streamlit as st
from google import genai
from google.genai import types

from langchain_core.messages import HumanMessage

# Імпортуємо агента
from agent import create_agent, extract_response_text, extract_tools_debug, MODEL_NAME

# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================
st.set_page_config(
    page_title="AI Чатбот з Gemini",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# ІНІЦІАЛІЗАЦІЯ
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_langgraph_agent(api_key: str, model_name: str):
    return create_agent(api_key, model_name)

api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Не знайдено GOOGLE_API_KEY у secrets.toml")
    st.stop()

# ============================================================
# ЗАГОЛОВОК
# ============================================================
st.title("🤖 AI Чатбот з Gemini")

# ============================================================
# БІЧНА ПАНЕЛЬ
# ============================================================
with st.sidebar:
    st.header("⚙️ Налаштування")

    mode = st.radio(
        "Режим",
        ["💬 Звичайний чат", "🛠️ Агент з інструментами"],
        index=0,
        key="mode_radio",
        help="Агент може використовувати калькулятор, Wikipedia та інші інструменти",
    )

    st.divider()
    st.info(f"Модель: **{MODEL_NAME}**")

    temperature = st.slider(
        "Температура (для звичайного чату)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="temperature_slider",
    )

    show_agent_debug = False
    if "Агент" in mode:
        show_agent_debug = st.checkbox("Показувати debug агента (tool calls)", value=False)

    st.divider()

    if st.button("🗑️ Очистити історію", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    if "Агент" in mode:
        st.divider()
        with st.expander("🛠️ Доступні інструменти"):
            st.markdown(
                """
- **calculator** — математичні обчислення
- **current_datetime** — поточна дата/час
- **wikipedia_search** — пошук у Wikipedia
"""
            )

# ============================================================
# ІНІЦІАЛІЗАЦІЯ СТАНУ
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ============================================================
# ФУНКЦІЇ ДЛЯ ЗВИЧАЙНОГО РЕЖИМУ (стрімінг)
# ============================================================
def convert_to_gemini_history(messages: list) -> list[types.Content]:
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    return contents

def stream_gemini_response(prompt: str, history: list):
    client = get_gemini_client(api_key)
    contents = convert_to_gemini_history(history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    )

    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n❌ **Помилка:** {str(e)}"

# ============================================================
# ФУНКЦІЯ ДЛЯ АГЕНТА
# ============================================================
def get_agent_response(prompt: str):
    agent = get_langgraph_agent(api_key, MODEL_NAME)

    # ВАЖЛИВО: thread_id потрібен для збереження/відновлення state між викликами
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config,
        )
        final_message = result["messages"][-1]
        text = extract_response_text(final_message)
        debug = extract_tools_debug(result["messages"])
        return text, debug
    except Exception as e:
        return f"❌ **Помилка агента:** {str(e)}", []

# ============================================================
# ПРИВІТАННЯ
# ============================================================
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if "Агент" in mode:
            st.markdown(
                f"""
👋 Вітаю! Я AI-агент на базі **{MODEL_NAME}** з інструментами.

Я можу:
- 🧮 Обчислювати математичні вирази (через tool)
- 📅 Повідомляти поточну дату та час (через tool)
- 📚 Шукати інформацію у Wikipedia (через tool)

Спробуйте: *"100/52"* або *"Хто такий Тарас Шевченко?"*
"""
            )
        else:
            st.markdown(
                f"""
👋 Вітаю! Я AI-асистент на базі **{MODEL_NAME}**.

Просто напишіть повідомлення нижче!
"""
            )

# ============================================================
# ВІДОБРАЖЕННЯ ІСТОРІЇ
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# ОБРОБКА ПОВІДОМЛЕННЯ
# ============================================================
if prompt := st.chat_input("Введіть ваше повідомлення..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "Агент" in mode:
            with st.spinner("🤔 Агент думає..."):
                response_text, debug = get_agent_response(prompt)
            st.markdown(response_text)

            if show_agent_debug and debug:
                with st.expander("Debug: tool calls / results", expanded=False):
                    st.json(debug)

            response = response_text
        else:
            response = st.write_stream(
                stream_gemini_response(prompt, st.session_state.messages[:-1])
            )

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})