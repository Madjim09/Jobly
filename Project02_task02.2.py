from pydantic import BaseModel, Field
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from vk_api.longpoll import VkLongPoll, VkEventType

import json
import trafilatura
import os
import vk_api
import random

load_dotenv()
llm = GigaChat(credentials=os.getenv("API_KEY"), verify_ssl_certs=False, model="GigaChat-2-Max")

VK_TOKEN = os.getenv("BOT_KEY") 

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)

STORAGE = "materials.json"

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Material(BaseModel):
    """Учебный материал, извлечённый со страницы"""
    url: str = Field(description="Ссылка на материал")
    title: str = Field(description="Заголовок материала")
    subject: str = Field(
        description="Предмет: численные методы, компьютерные сети, "
                    "программирование на Python или физика"
    )
    date_published: str = Field(
        description="Дата публикации в формате YYYY-MM-DD. "
                    "Если не нашёл — верни пустую строку."
    )

# второй LLM — умеет возвращать объект Material
extractor = GigaChat(
    credentials=os.getenv("API_KEY"),
    verify_ssl_certs=False,
    model="GigaChat-2-Max",
).with_structured_output(Material)


def load_json():
    if not os.path.exists(STORAGE):
        return []
    with open(STORAGE, encoding="utf-8") as f:
        return json.load(f)


def save_json(items):
    with open(STORAGE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


@tool
def get_text_from_link(url: str) -> str:
    """Скачать страницу и вернуть её текст."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return f"Не удалось скачать {url}"
    text = trafilatura.extract(downloaded)
    return text[:12000] if text else f"Нет текста на {url}"


@tool
def add_material(url: str) -> str:
    """Разобрать ссылку, извлечь данные и сохранить материал в JSON."""
    text = get_text_from_link.invoke({"url": url})
    if text.startswith("Не удалось") or text.startswith("Нет текста"):
        return text

    prompt = f"URL: {url}\n\nТекст страницы:\n{text}"
    material = extractor.invoke(prompt)

    # нормализуем предмет
    subj = material.subject.lower()
    for s in ["численные методы", "компьютерные сети",
              "программирование на python", "физика"]:
        if s in subj:
            material.subject = s
            break

    items = load_json()
    items.append(material.model_dump())
    save_json(items)
    return f"Сохранено: {material.title} → {material.subject}"


@tool
def load_from_json(subject: str) -> str:
    """Вернуть JSON-список материалов по предмету."""
    items = load_json()
    result = [it for it in items
              if it["subject"] == subject.lower()]
    return json.dumps(result, ensure_ascii=False, indent=2)


TOOLS = [add_material, load_from_json]
llm = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(content="""
Ты агент-библиотекарь учебных материалов.

Инструменты:
- add_material(url) — разобрать ссылку и сохранить материал.
- load_from_json(subject) — вернуть список материалов по предмету.

Предметы: численные методы, компьютерные сети,
программирование на Python, физика.

Если дали ссылку → add_material.
Если просят список → load_from_json.

Когда вызываешь load_from_json, верни ВСЕ поля которые там есть,
но в более удобном и читаемом виде.

НЕ ИСПОЛЬЗУЙ ОБЕРТКИ ДЛЯ ТЕКСТА НАПРИМЕР: **привет**, лучше просто - привет
Также убирай ковычки из строк:
"Изучаем Python: теория, практика, настройка инструментов" -> Изучаем Python: теория, практика, настройка инструментов
""")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
tool_node = ToolNode(tools=TOOLS)


def chatBot(state: State) -> dict:
    return {"messages": [llm.invoke([SYSTEM_PROMPT] + state["messages"])]}


graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatBot)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for value in event.values():
            msg = value["messages"][-1]
            if isinstance(msg, AIMessage) and msg.content:
                print("ИИ-агент:", msg.content)

def send_message(user_id, text):
    """Отправить сообщение пользователю."""
    vk.messages.send(
        user_id=user_id,
        message=text,
        random_id=random.randint(0, 2**31),
    )

def get_graph_response(user_input: str) -> str:
    """Прогнать сообщение через граф и вернуть финальный ответ."""
    final = ""
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for value in event.values():
            msg = value["messages"][-1]
            if isinstance(msg, AIMessage) and msg.content:
                final = msg.content
    return final

def main():
    print("VK-бот запущен. Ожидаю сообщения...")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_text = event.text
            user_id = event.user_id

            print(f"[{user_id}]: {user_text}")

            try:
                response = get_graph_response(user_text)
            except Exception as e:
                response = f"Ошибка: {e}"

            send_message(user_id, response)


if __name__ == "__main__":
    main()