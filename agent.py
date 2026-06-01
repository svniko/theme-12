# agent.py — LangGraph-агент з інструментами (Gemini)

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

import numexpr as ne
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================
# КОНСТАНТИ
# ============================================================
MODEL_NAME = "gemini-3.1-flash-lite"

# Системна інструкція, щоб модель ДІЙСНО користувалась інструментами
SYSTEM_PROMPT = SystemMessage(
    content=(
        "Ти — AI-агент з інструментами. "
        "Правила:\n"
        "1) Для БУДЬ-ЯКИХ арифметичних обчислень (дроби, відсотки, ділення, множення тощо) "
        "ОБОВ'ЯЗКОВО викликай tool `calculator` і НЕ рахуй у голові.\n"
        "2) Для запитів про поточну дату/час/день тижня викликай `current_datetime`.\n"
        "3) Для фактів/довідок/біографій спочатку викликай `wikipedia_search`, "
        "а потім коротко відповідай, спираючись на результат.\n"
        "4) Якщо інструмент повернув помилку — поясни помилку та попроси уточнення.\n"
    )
)

# ============================================================
# СХЕМА СТАНУ
# ============================================================
class AgentState(TypedDict):
    # add_messages робить messages append-only (merge повідомлень), потрібен для multi-turn
    messages: Annotated[list[Any], add_messages]


# ============================================================
# ІНСТРУМЕНТИ
# ============================================================
@tool
def calculator(expression: str) -> str:
    """
    Обчислює математичний вираз.
    Використовуй для будь-яких арифметичних обчислень.

    Args:
        expression: Математичний вираз (наприклад, "2 + 2 * 3")
    """
    expr = (expression or "").strip()
    if not expr:
        return "Помилка: порожній вираз."

    # Дозволяємо лише безпечні символи
    allowed = set("0123456789+-*/()., eE")
    cleaned = expr.replace(" ", "")
    if not all(ch in allowed for ch in cleaned):
        return "Помилка: недопустимі символи у виразі."

    # Кома -> крапка (часта помилка вводу)
    cleaned = cleaned.replace(",", ".")

    try:
        result = ne.evaluate(cleaned)
        value = result.item() if hasattr(result, "item") else result
        return f"Результат: {expression} = {value}"
    except Exception as e:
        return f"Помилка обчислення: {str(e)}"


@tool
def current_datetime(query: str) -> str:
    """
    Повертає поточну дату, час або день тижня.

    Args:
        query: Тип запиту (date/time/day/datetime)
    """
    now = datetime.now()
    days_ua = {
        "Monday": "понеділок",
        "Tuesday": "вівторок",
        "Wednesday": "середа",
        "Thursday": "четвер",
        "Friday": "п'ятниця",
        "Saturday": "субота",
        "Sunday": "неділя",
    }
    q = (query or "").lower()

    if "day" in q:
        return f"Сьогодні {days_ua.get(now.strftime('%A'), now.strftime('%A'))}"
    if "time" in q:
        return f"Поточний час: {now.strftime('%H:%M:%S')}"
    if "date" in q:
        return f"Поточна дата: {now.strftime('%d.%m.%Y')}"

    day_en = now.strftime("%A")
    return f"Зараз {days_ua.get(day_en, day_en)}, {now.strftime('%d.%m.%Y %H:%M:%S')}"


wiki_client = WikipediaAPIWrapper(lang="uk", top_k_results=1, doc_content_chars_max=1000)


@tool
def wikipedia_search(query: str) -> str:
    """
    Шукає інформацію в українській Wikipedia.

    Args:
        query: Пошуковий запит
    """
    q = (query or "").strip()
    if not q:
        return "Помилка: порожній запит."

    try:
        result = wiki_client.run(q)
        return result if result else f"Не знайдено: {q}"
    except Exception as e:
        return f"Помилка пошуку: {str(e)}"


TOOLS = [calculator, current_datetime, wikipedia_search]


# ============================================================
# ПОБУДОВА ГРАФА
# ============================================================
def create_agent(api_key: str, model_name: str = MODEL_NAME):
    """
    Створює скомпільований LangGraph-агент.

    Args:
        api_key: Google API key
        model_name: Назва моделі Gemini

    Returns:
        Скомпільований граф агента
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,  # нижче = краще для tool-calling
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        # Системне повідомлення додаємо "епізодично", не пишемо його в state
        msgs = [SYSTEM_PROMPT] + state["messages"]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")

    # tools_condition -> "tools" або END
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def extract_response_text(message) -> str:
    """Витягує текст з повідомлення LangChain."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def extract_tools_debug(messages: list[Any]) -> list[dict]:
    """Повертає короткий список викликів інструментів/результатів (для debug UI)."""
    debug = []
    for m in messages:
        if isinstance(m, ToolMessage):
            debug.append({"type": "tool_result", "content": m.content, "tool_call_id": m.tool_call_id})
        else:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    debug.append(
                        {
                            "type": "tool_call",
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                            "id": tc.get("id"),
                        }
                    )
    return debug