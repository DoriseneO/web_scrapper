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


# Load .env file
load_dotenv()

# Retrieve environment variables
USER_AGENT = os.getenv("USER_AGENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Set headers
headers = {"User-Agent": USER_AGENT}

# Target URL
web_url = "https://www.walgreensbootsalliance.com/"

# Send request with User-Agent
response = requests.get(web_url, headers=headers)

# Load web content using WebBaseLoader
loader = WebBaseLoader(web_url)
loader.requests_kwargs = {
    "headers": headers,
    "verify": True
}
load = loader.load()

# Split the loaded content into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
text_split = splitter.split_documents(load)
print(f"Number of chunks created: {len(text_split)}")

# Embed the content
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Store in the vector store
vector_store = Chroma.from_documents(text_split, embeddings)

# Custom Retriever Class
class MyCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return vector_store.similarity_search(query)

# Instantiate your custom retriever
custom_retriever = MyCustomRetriever()

# Query the retriever
query = "regulation year?"
relevant_documents = custom_retriever._get_relevant_documents(query)
 #ftyujkl;
# Print the retrieved documents
for doc in relevant_documents:
    print(doc.page_content)