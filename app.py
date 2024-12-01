import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load model and ChromaDB client
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client(Settings(persist_directory="./vector_db"))
collection = client.get_or_create_collection("pdf_embeddings")

# Function to query vector store
def query_vector_store(query, top_k=3):
    query_embedding = model.encode([query])  # Generate query embedding
    
    # Perform the search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    documents = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    return documents, distances, metadatas

# Streamlit Interface
st.title("Content Engine - Query System")
st.write("Welcome to the PDF Query Engine. Please enter a query below:")

query = st.text_input("Enter your query:")

if query:
    st.write("Searching...")
    documents, distances, metadatas = query_vector_store(query)
    
    for idx, (doc, distance, metadata) in enumerate(zip(documents, distances, metadatas)):
        st.subheader(f"Result {idx+1} (Distance: {distance:.4f})")
        st.write(f"Sentence: {doc}")
        st.write(f"Metadata: {metadata}")
        st.write("")
