import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

# Load .env file
load_dotenv()

# Retrieve environment variables
USER_AGENT = os.getenv("USER_AGENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Set headers
headers = {"User-Agent": USER_AGENT}

# Target URL
web_url = "https://www.cvshealth.com/"

# Load web content using WebBaseLoader
loader = WebBaseLoader(web_url)
loader.requests_kwargs = {
    "headers": headers,
    "verify": True
}
load = loader.load()

# Split the loaded content into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index=True)
text_split = splitter.split_documents(load)

# Embed the content
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Store in the vector store
vector_store = Chroma.from_documents(text_split, embeddings)

llm_model = ChatOpenAI(model="gpt-4o-mini")

# Create a conversational prompt template with the query and relevant context
conversational_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "{context}"
         "You are an AI assistant. Retrieve the relevant documents based on the query and use them to answer the question. "
         "If the information is insufficient, respond with 'can you be more specific and provide more details.'"),
        ("human", "{user_query}")
    ]
)

class State(TypedDict):
    question: str
    context: List[Document]

def retrieve(state: State):
    # Correct the method for similarity search
    retrieved_doc = vector_store.similarity_search(state["question"], k=2)
    return {"context": retrieved_doc}

def generate(state: State):
    # Join document content into a string
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])  
    # Format the conversational prompt
    formatted_prompt = conversational_prompt.format(user_query=state["question"], context=docs_content)

  
    response = llm_model.invoke(formatted_prompt)
    output = StrOutputParser()
    response_output_parser = output.invoke(response)
    print(response_output_parser)

    # Define the StateGraph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])

    # Add edge transitions in the graph
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

 
while True:
    query = input("ask me anyhting or quit\n")
    if query == 'quit':
      break
    print("Goodbye!")
    
# Execute the graph with a test query
    result = graph.invoke({"question": query,"context": "context"})


