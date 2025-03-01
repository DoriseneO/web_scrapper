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
from langgraph.graph import StateGraph, START,END
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import tool_node
from IPython.display import Image, display

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

# this is an inbuilt pydantic schema
custom_graph_query = StateGraph(MessagesState)

# this is the retrieve node
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

#this is the query node
def query_node(state :MessagesState):
    """Generate tool call for retrieval or respond."""
    query = state["messages"]
    response = retrieve(query)
    return "answered_content",response


#this is the generate node 
def generate_node(state: MessagesState):   
    """generate tool message""" 
    tool_msg = []
    for message in state["messages"]:
      if message.type == "tool":
         new_tool_msg = tool_msg.append(message)
      else:
           break
      
    #format the generated response
    generated_doc = "\n\n".join(doc.content for doc in new_tool_msg)  
            
    #create prompt 
    chat_prompt= ChatPromptTemplate([
    ("system",f"provide response based on the retrived documents {generated_doc}"),
    ("human","{user_query}")
    ])
    rt= chat_prompt.invoke(user_query= state["messages"][-1]["content"])
    invoke_llm = llm_model.invoke(rt)
    print(invoke_llm)
    
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("node_1",query_node)
graph_builder.add_node("node_2",retrieve)
graph_builder.add_node("node_3",generate_node) 

#logic 
graph_builder.add_edge(START,"node_1")
graph_builder.add_edge("node_2","node_3")
graph_builder.add_edge("node_3",END)
        
#add graph
graph = graph_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# while True:
#     query = input("ask me anyhting or quit\n")
#     if query == 'quit':
#       break
#     print("Goodbye!")
    
# Execute the graph with a test query
    # result = graph.invoke({"question": query,"context": "context"})
