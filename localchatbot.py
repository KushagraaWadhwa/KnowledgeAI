import streamlit as st
from llama_index.core import SimpleDirectoryReader
import re
import os
import faiss
import numpy as np
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.corpus import stopwords
import nltk
import gc

nltk.download('stopwords')

# Clear session state if needed
if st.button('Clear Session State'):
    st.session_state.clear()

# Streamlit UI setup
st.title("Indian Constitution Chatbot")
st.write("Ask me questions about the Indian Constitution!")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if 'documents' not in st.session_state:
    with st.spinner("Reading and processing documents..."):
        try:
            documents = SimpleDirectoryReader('data').load_data()  # Loading the documents
            for doc in documents:
                doc.text = preprocess_text(doc.text)  # Preprocess each document
            st.session_state['documents'] = documents
            st.write("Documents loaded and preprocessed.")
        except Exception as e:
            st.error(f"Error during document loading: {e}")
else:
    documents = st.session_state['documents']

# Function to retrieve relevant documents from FAISS
def search_faiss(query, faiss_index, documents):
    query_embedding = st.session_state['embed_model'].encode([query])  # Use embed_model from session state
    distances, indices = faiss_index.search(query_embedding, k=5)  # Retrieve top-5 docs
    st.write(f"Retrieved indices: {indices}")  # Log retrieved indices
    results = [documents[idx] for idx in indices[0] if idx < len(documents)]  # Ensure idx is valid
    return results

# Function to generate answer using retrieved documents and LLM
def generate_answer(question, retrieved_docs):
    context = " ".join([doc.text for doc in retrieved_docs])
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    st.write(f"Generated Prompt: {prompt}")  # Log the prompt
    result = generator(prompt, max_length=200)
    return result[0]['generated_text']

# Function for batched embedding generation to reduce memory issues
def batch_encode(texts, model, batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_embeddings = model.encode(texts[i:i + batch_size])
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Generate embeddings (without loader)
if 'embeddings' not in st.session_state:
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Local embedding model
        documents_text = [doc.text for doc in documents]
        embeddings = batch_encode(documents_text, embed_model, batch_size=10)  # Batched encoding
        st.session_state['embeddings'] = np.array(embeddings)
        st.write(f"Embeddings generated with shape: {embeddings.shape}.")
    except Exception as e:
        st.error(f"Error during embedding generation: {e}")
else:
    embeddings = st.session_state['embeddings']

# Initialize FAISS index (without loader)
if 'faiss_index' not in st.session_state:
    try:
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for FAISS
        faiss_index.add(embeddings)
        st.session_state['faiss_index'] = faiss_index
        st.write(f"FAISS index created with {faiss_index.ntotal} entries.")
    except Exception as e:
        st.error(f"Error during FAISS index creation: {e}")
else:
    faiss_index = st.session_state['faiss_index']

# Load local LLM for text generation (without loader)
if 'generator' not in st.session_state:
    try:
        st.write("Loading LLM Model...")
        st.session_state['generator'] = pipeline('text-generation', model='gpt2')
        st.write("LLM Model Loaded.")
    except Exception as e:
        st.error(f"Error during LLM loading: {str(e)}")
        st.stop()  # Stop further execution if model loading fails
generator = st.session_state['generator']

# Mark everything is ready
if 'ready' not in st.session_state:
    st.session_state['ready'] = True
    st.success("Ready for Q&A! You can ask your questions now.")

# Search and generate answers
query = st.text_input("Enter your question:")
# Main logic to process query and respond
if query:
    with st.spinner("Searching for the answer..."):
        try:
            retrieved_docs = search_faiss(query, faiss_index, documents, embed_model)  # embed_model should be accessible here
            if not retrieved_docs:
                st.error("No relevant documents found for the query.")
            else:
                answer = generate_answer(query, retrieved_docs)
                st.write("Response:")
                st.write(answer)
        except Exception as e:
            st.error(f"Error during query processing: {e}")




