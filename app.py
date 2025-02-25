import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

# Load .env file
load_dotenv()

# Retrieve environment variables
USER_AGENT = os.getenv("USER_AGENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Set headers
headers = {"User-Agent": USER_AGENT}

# Target URL
# web_url = "https://python.langchain.com/docs/concepts/"
web_url ="https://iamdoris.com/"

# Load web content using WebBaseLoader
loader = WebBaseLoader(web_url)
loader.requests_kwargs = {
    "headers": headers,
    "verify": True
}
load = loader.load()

# Split the loaded content into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
text_split = splitter.split_documents(load)

# Embed the content
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Store in the vector store
vector_store = Chroma.from_documents(text_split, embeddings)

# Custom Retriever Class
class MyCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=5)

# Instantiate your custom retriever
custom_retriever = MyCustomRetriever()

while True:
    query = input("Ask me a question on langchain (or type 'quit' to quit): \n USER: ")

    if query.lower() == 'quit':
        print("Goodbye!")
        break

    # Get relevant documents
    relevant_documents = custom_retriever._get_relevant_documents(query)

# # # Debugging: Print retrieved documents and their count
#     print(f"Number of retrieved documents: {len(relevant_documents)}")
#     for i, doc in enumerate(relevant_documents):
#         print(f"Document {i + 1}: {doc.page_content}")

    
    # Create a conversational prompt template
    conversational_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"You are an AI assistant. Based on the following context, provide a clear and concise response to the user's query: '{query}'. If the information is insufficient, respond with 'I don't know.'"),
        HumanMessage(content=query),
        MessagesPlaceholder(variable_name="msg")
    ])
    
  # Format the retrieved documents as a single string with context
    retrieved_context = [doc.page_content for doc in relevant_documents]

    # Format chat prompt by including the retrieved context
    chat_prompt = conversational_prompt.format(user_query=query,msg=retrieved_context)

    # Create LLM and get response
    try:
        llm_model = init_chat_model("gpt-4o-mini", model_provider="openai")
        response = llm_model.invoke(chat_prompt)
        print("AI:", response.content)
    except Exception as e:
        print(f"Error while getting response: {e}")
