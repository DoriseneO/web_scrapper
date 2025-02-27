
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Create LLM Model
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Define Tools (Web Search)
search = TavilySearchResults(max_results=2)
tools = [search]

# Set up Memory for Multi-turn Conversations
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the Agent Executor
agent_executor = initialize_agent(
    tools=tools,            
    llm=model,              
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct reasoning type
    memory=memory,          
    verbose=True           
)

# Interactive Loop
while True:
    query = input("\n Ask me a question (or type 'exit' to quit): ")
    
    if query.lower() == "exit":
        print("👋 Exiting. Have a great day!")
        break
    
    # Run the agent
    response = agent_executor.invoke({"input": query})
    
    # Print the response
    print("\n🔹 Agent Response:", response["output"])
