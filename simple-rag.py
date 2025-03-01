
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser

# Load .env file
load_dotenv()

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Target file path
file_path = "./health_retail.pdf"

# Load PDF content
loader = PyPDFLoader(file_path)
load = loader.load()

# Split the loaded content into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index=True)
text_split = splitter.split_documents(load)

# Embed the content
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Store in the vector store
vector_store = Chroma.from_documents(text_split, embeddings)
print("Vector store created successfully.")

# Initialize the LLM model
llm_model = ChatOpenAI(model="gpt-4")  # Use "gpt-4" or "gpt-3.5-turbo"

# State class definition
class State(TypedDict):
    query: str
    retrieved_doc: list
    response: str
    
query = input("Ask me ANYTHING: ")

# Define the initial state
initial_state = State(query=query, retrieved_doc=[], response="")

# Retrieval node function
def retrieval_node(state: State):
    query = state["query"]
    retrieved_doc = vector_store.similarity_search(query=query, k=2)
    return {"retrieved_doc": retrieved_doc}

# Generate node function
def generate_node(state: State):
    retrieved_doc = state["retrieved_doc"]
    docs_content = "\n".join(doc.page_content for doc in retrieved_doc)
    
    # Define the conversational prompt
    conversational_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "{docs_content}\n"
             "You are an AI assistant. Retrieve the relevant documents based on the query and use them to answer the question. "
             "If the information is insufficient, respond with 'Can you be more specific and provide more details?'"),
            ("human", "{query}")
        ]
    )
    
    # Format the prompt with the retrieved content and query
    formatted_prompt = conversational_prompt.format(docs_content=docs_content, query=state["query"])
    
    # Generate the response using the LLM
    response = llm_model.invoke(formatted_prompt)
    return {"response": response}


# Build the state graph
builder = StateGraph(State)
builder.add_node("retrieval_node", retrieval_node)
builder.add_node("generate_node", generate_node)
builder.add_edge("retrieval_node", "generate_node")
builder.set_entry_point("retrieval_node")
builder.set_finish_point("generate_node")
graph = builder.compile()

# Execute the graph
result = graph.invoke(initial_state)
print("Response:", result)