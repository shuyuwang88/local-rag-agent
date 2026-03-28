# RAG pipeline

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever # import retriever from vector store module

# Initialize the local LLM via ollama
model = OllamaLLM(model="llama3.1")

# Define the prompt template
# This controls how the LLM receives context + question
template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

# Create a prompt object
prompt = ChatPromptTemplate.from_template(template)
# Create a chain: prompt -> model
chain = prompt | model

while True:
    print("\n\n--------------------------")
    question = input("Ask your question(q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    # Retrieve top-k relevant documents from vector store
    reviews = retriever.invoke(question)
    
    # Invoke the chain with context + question
    result = chain.invoke({
        "reviews":reviews, 
        "question": question
    })
    print(result)
