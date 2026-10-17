from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, AIMessage

import os
import requests
import lxml

load_dotenv()
LLM_API = os.getenv("LLM_KEY")

llm = GigaChat(credentials=LLM_API, verify_ssl_certs=False, model="GigaChat-2")