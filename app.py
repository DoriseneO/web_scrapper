import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load .env file
load_dotenv()

# Retrieve environment variables
USER_AGENT = os.getenv("USER_AGENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Set headers
headers = {"User-Agent": USER_AGENT}

# Target URL
web_url = "https://iamdoris.com/","https://www.walgreensbootsalliance.com/"

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

# Custom Retriever Class
class MyCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=2)

# Instantiate your custom retriever
custom_retriever = MyCustomRetriever()

llm_model = ChatOpenAI(model="gpt-4o-mini")   

while True:
    query = input("Ask me a question on langchain (or type 'quit' to quit): \n USER: ")

    if query.lower() == 'quit':
        print("Goodbye!")
        break
    
    # Create a conversational prompt template with the query and relevant context
    conversational_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are an AI assistant. Retrieve the relevant documents based on the query and use them to answer the question. "
             "get answers from the vector store and provide response"
             "you are to provide response from the scrapped website"
             "{context}"
             "If the information is insufficient, respond with 'can you be more specific and provide more details.'"),
            ("human", "{user_query}")
        ]
    )
    relevant_documents = custom_retriever._get_relevant_documents(query)
    docs_content = "\n\n".join(doc.page_content for doc in relevant_documents)
    prompt = conversational_prompt.invoke({"user_query": query, "context": docs_content})
    answer = llm_model.invoke(prompt)
    print(answer.content)
    # Retrieve the relevant documents based on the query
    # relevant_documents = custom_retriever._get_relevant_documents(query)
    # # Execute chain directly within the prompt
    # chain = conversational_prompt | llm_model | StrOutputParser()
    # result = chain.invoke({"user_query": query,"context":relevant_documents})

    # print(result)
