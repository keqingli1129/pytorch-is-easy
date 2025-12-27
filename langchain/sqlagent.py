import re

# Core imports
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

# Connect to employees database
db = SQLDatabase.from_uri("sqlite:///db/employees.db")

# Check connection and get basic info
try:
    # Test connection by getting table names
    tables = db.get_usable_table_names()
    print(f"✓ Database connected successfully")
    print(f"✓ Found {len(tables)} tables: {', '.join(tables)}")
    
except Exception as e:
    print(f"✗ Database connection failed: {e}")

# Get schema information
SCHEMA = db.get_table_info()
# print("\nDatabase Schema:", SCHEMA)
print("✓ Connected to employees database")

llm = ChatOllama(
    model="qwen3:0.6b", 
    base_url="http://localhost:11434", 
    temprature = 0.1)

response = llm.invoke("Hello, how are you?")
response.pretty_print()
print("✓ Initialized Ollama chat model")