import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define classification model with proper default values
class Classification(BaseModel):
    positive: Optional[str] = Field(default=None, description="Classify this review as positive or negative")
    language: Optional[str] = Field(default=None, description="Detect the language used in the review")

# Get user input
query = input("\nAI: Ask me anything \nUSER: ")

# Define the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze and classify the review"),
    ("human", "{user_query}")
])

# Initialize the model correctly
llm = ChatOpenAI(model_name="gpt-4o-mini").with_structured_output(Classification)

# Format the prompt correctly
chat_prompt_format = chat_prompt.format(user_query=query)

# Invoke the model and get the response
result = llm.invoke(chat_prompt_format)

# Print the structured response
print(result)
