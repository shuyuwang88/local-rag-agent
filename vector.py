from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load dataset (restaurant reviews)
df = pd.read_csv("realistic_restaurant_reviews.csv")
# Initialize embedding model (convert text -> vectors)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define persistent storage location
db_location = "./chrome_langchain_db"
# If database does not exist, create and populate the database
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    # Convert each row into a LongChain Document
    for i, row in df.iterrows():
        document = Document(
            # Text used for embedding and retrieval
            page_content=row["Title"] + " " + row["Review"],
            
            # Metadata (not embedded, but useful for filtering or display)
            metadata={
                "rating": row["Rating"], 
                "date": row["Date"]
            },
            # Unique identifier
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="restaurant_reviews", # name of the collection
    persist_directory=db_location, # storage path
    embedding_function=embeddings # embedding function
)

# Add documents only if database is newly created
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever interface
# Allowing semantic search over the vector database
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}   
)