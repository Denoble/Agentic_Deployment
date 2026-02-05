import os
# Securely prompt for your API key in an interactive environment
import getpass
from dotenv import load_dotenv

# Web scraping utilities
import requests
from bs4 import BeautifulSoup

# LangChain imports for agent creation and chat models
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core import prompts
from langchain_classic.memory import ConversationBufferMemory
from langchain.agents import create_agent



load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY") or getpass.getpass("Enter your OpenAI API Key: ")
print(f"API Key loaded successfully. {'Key Length: ' + str(len(OPEN_AI_API_KEY)) if OPEN_AI_API_KEY else 'No Key Found'}")

def fetch_webpage_content(url: str) -> str:
    """Fetches and returns the text content of a webpage given its URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        return f"Error fetching the webpage: {e}"
webpage = fetch_webpage_content("https://linkedin.com")
print(webpage)
