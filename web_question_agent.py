import os
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import getpass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Annotated
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 2 – Load the OpenAI API key
load_dotenv()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY ") or getpass.getpass("Enter your OpenAI API Key: ")
print(f"API Key loaded successfully. {'Key Length: ' + str(len(OPENAI_API_KEY )) if OPENAI_API_KEY  else 'No Key Found'}")
"""
An agent that fetches website content, answers questions about it,
    and maintains conversation history for follow-up questions.
"""

# Step 3.1 – Define a function to extract text from a URL
def extract_website_text(url: str, timeout: int = 10) -> Optional[str]:
    """
    Extract visible text from a website with proper error handling.
    
    Args:
        url: Website URL to extract text from
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract and clean text
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    except requests.exceptions.RequestException as e:
       logger.error(f"Request failed for {url}: {e}")
       return None
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return None


# Step 4.3 – Create the memory to store chat history
# Use ChatMessageHistory for in-memory storage (replaces ConversationBufferMemory)
store = {}

def get_session_history(session_id: str = "default") -> BaseChatMessageHistory:
    """Retrieve or create a chat message history for a given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Step 4.2 – Create the tool using the @tool decorator
@tool
def website_qa_tool(input_text: str) -> str:
    """Answer a user's question using website content or the chat history.

    The input format can be either:
    - "<url> | <question>" to fetch content from the URL and answer the question.
    - "<question>" to ask a follow‑up question based on the chat history.

    Args:
        input_text: A string containing either a URL and a question separated by '|' or just a question.

    Returns:
        The model's answer as a string.
    """
    # Check whether input contains a URL and a question, also handle the case where the agent includes a placeholder URL
    if '|' in input_text and "<website URL>" not in input_text:
        url, question = input_text.split('|', 1)
        # Fetch website content and build a prompt for the LLM
        content = extract_website_text(url.strip())
        prompt = f"Answer this question using the content from the website:\n\n{question.strip()}\n\nContent:\n{content[:6000]}"

    else:
        # If no valid URL is provided or if a placeholder URL is present, load the chat history from memory
        history_obj = get_session_history()
        msgs = history_obj.messages
        # Convert the messages into a simple string format
        history = "\n".join([f"{m.type}: {m.content}" for m in msgs])
        # If the input contains the placeholder and other text, extract the question part
        if "<website URL>" in input_text and "|" in input_text:
            _, question = input_text.split('|', 1)
            input_text = question.strip()

        prompt = f"Using the chat history below, answer the user's question.\n\nChat history:\n{history}\n\nQuestion:\n{input_text}"

    # Use an OpenAI chat model to generate the answer
    llm = ChatOpenAI()
    response = llm.invoke(prompt)
    return response.content 

print("=" * 80)
result1 = website_qa_tool.invoke(
    "https://us.etrade.com | Can you tell me what this website offers to beginner investors?")
