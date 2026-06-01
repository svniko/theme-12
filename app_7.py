# app.py — Крок 7: Додаткові функції (Gemini + LangGraph)

import json
import uuid
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types

from langchain_core.messages import HumanMessage, SystemMessage

# Імпортуємо агента
from agent import (
    create_agent,
    extract_response_text,
    extract_tools_debug,   # якщо не потрібно — можна прибрати
    MODEL_NAME,
)

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
    # Агент створюється один раз на сесію (checkpointer всередині агента)
    return create_agent(api_key, model_name)

api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Не знайдено GOOGLE_API_KEY у secrets.toml")
    st.stop()

# ============================================================
# СТАН STREAMLIT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Ти корисний AI-асистент. Відповідай українською мовою."

# ============================================================
# UI: ЗАГОЛОВОК
# ============================================================
st.title("🤖 AI Чатбот з Gemini")

# ============================================================
# UI: БІЧНА ПАНЕЛЬ
# ============================================================
with st.sidebar:
    st.header("⚙️ Налаштування")

    mode = st.radio(
        "Режим",
        ["💬 Звичайний чат", "🛠️ Агент з інструментами"],
        index=0,
        key="mode_radio",
    )

    st.divider()
    st.info(f"Модель: **{MODEL_NAME}**")

    # Температура актуальна для "звичайного" режиму
    temperature = st.slider(
        "Температура (звичайний чат)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="temperature_slider",
    )

    st.divider()
    with st.expander("📝 Системний промпт", expanded=False):
        system_prompt = st.text_area(
            "Інструкції для моделі",
            value=st.session_state.system_prompt,
            height=110,
            key="system_prompt_input",
        )
        if st.button("💾 Зберегти промпт", use_container_width=True):
            st.session_state.system_prompt = system_prompt.strip() or st.session_state.system_prompt
            st.toast("Системний промпт збережено!")

    st.divider()

    # Кнопки дій
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистити", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.rerun()

    with col2:
        if st.session_state.get("messages"):
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "mode": mode,
                "thread_id": st.session_state.thread_id,
                "system_prompt": st.session_state.system_prompt,
                "messages": st.session_state.messages,
            }
            st.download_button(
                "📥 Експорт",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()

    # Статистика
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Повідомлень", len(st.session_state.get("messages", [])))
    with col2:
        total_chars = sum(len(m.get("content", "")) for m in st.session_state.get("messages", []))
        st.metric("~Токенів", total_chars // 4)

    # Debug + список інструментів
    show_agent_debug = False
    if "Агент" in mode:
        st.divider()
        show_agent_debug = st.checkbox("Показувати debug (tool calls)", value=False)
        with st.expander("🛠️ Інструменти"):
            st.markdown(
                """
- **calculator** — математика  
- **current_datetime** — дата/час  
- **wikipedia_search** — Wikipedia
"""
            )

# ============================================================
# ФУНКЦІЇ ДЛЯ ЗВИЧАЙНОГО РЕЖИМУ
# ============================================================
def convert_to_gemini_history(messages: list) -> list[types.Content]:
    """Конвертує історію у формат Gemini."""
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

def stream_gemini_response(prompt: str, history: list, system_prompt: str):
    """
    Стрімінг відповіді Gemini.
    1) Першою спробою використовуємо system_instruction (якщо підтримується бекендом/SDK).
    2) Якщо API поверне помилку про unknown systemInstruction — fallback на "вставку" інструкцій у messages.
    """
    client = get_gemini_client(api_key)
    contents = convert_to_gemini_history(history)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    # Спроба №1: system_instruction
    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
        return
    except Exception as e:
        err = str(e)
        # Якщо system_instruction не “пройшов” — робимо fallback
        if "systemInstruction" not in err and "Unknown name" not in err and "INVALID_ARGUMENT" not in err:
            yield f"\n\n❌ **Помилка:** {err}"
            return

    # Спроба №2 (fallback): додаємо системні інструкції як перші повідомлення
    try:
        contents2 = []
        if system_prompt:
            contents2.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"[Системні інструкції]: {system_prompt}")],
                )
            )
            contents2.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Зрозуміло, буду дотримуватися інструкцій.")],
                )
            )
        contents2.extend(convert_to_gemini_history(history))
        contents2.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        stream2 = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents2,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        for chunk in stream2:
            if chunk.text:
                yield chunk.text
    except Exception as e2:
        yield f"\n\n❌ **Помилка:** {str(e2)}"
# ============================================================
# ФУНКЦІЯ ДЛЯ АГЕНТА
# ============================================================
def get_agent_response(prompt: str):
    agent = get_langgraph_agent(api_key, MODEL_NAME)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # SystemMessage з фіксованим id -> add_messages зможе "оновлювати", а не плодити дублікати
    sys = SystemMessage(content=st.session_state.system_prompt, id="user_system_prompt")

    try:
        result = agent.invoke(
            {"messages": [sys, HumanMessage(content=prompt)]},
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
        st.markdown(
            f"""
👋 Вітаю! Модель: **{MODEL_NAME}**

📝 Поточний системний промпт:
> {st.session_state.system_prompt[:160]}{"..." if len(st.session_state.system_prompt) > 160 else ""}

Обери режим зліва і напиши повідомлення нижче.
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
                stream_gemini_response(
                    prompt,
                    st.session_state.messages[:-1],
                    st.session_state.system_prompt,
                )
            )

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})