import streamlit as st
from gtts import gTTS
import os
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from dotenv import load_dotenv
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
load_dotenv()

st.title("The Indian Constitution Chatbot")
st.write("Ask me questions about the Indian Constitution")

# Preprocess text for indexing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()

# Function to load documents and initialize query engine
def initialize_query_engine():
    if 'documents' not in st.session_state:
        documents = SimpleDirectoryReader('data').load_data()
        for doc in documents:
            doc.text = preprocess_text(doc.text)
        st.session_state['documents'] = documents
    else:
        documents = st.session_state['documents']
    
    if 'nodes' not in st.session_state:
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        st.session_state['nodes'] = nodes
    else:
        nodes = st.session_state['nodes']

    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    # Initialize embeddings and LLM
    if 'embed_model' not in st.session_state:
        st.session_state['embed_model'] = GeminiEmbedding(model_name="models/embedding-001", api_key=GEMINI_API_KEY)
        Settings.embed_model = st.session_state['embed_model']
        
    if 'llm' not in st.session_state:
        st.session_state['llm'] = Gemini(api_key=GEMINI_API_KEY)
        Settings.llm = st.session_state['llm']

    # Initialize storage context and create indexes
    if 'storage_context' not in st.session_state:
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        st.session_state['storage_context'] = storage_context
    else:
        storage_context = st.session_state['storage_context']

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, top_k=5)

    # Custom retriever to combine vector and keyword retrievers
    class CustomRetriever(BaseRetriever):
        def __init__(self, vector_retriever, keyword_retriever, mode="AND"):
            self._vector_retriever = vector_retriever
            self._keyword_retriever = keyword_retriever
            self._mode = mode if mode in ("AND", "OR") else "AND"
            super().__init__()

        def _retrieve(self, query_bundle):
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
            vector_ids = {n.node.node_id for n in vector_nodes}
            keyword_ids = {n.node.node_id for n in keyword_nodes}

            combined_dict = {n.node.node_id: n for n in vector_nodes}
            combined_dict.update({n.node.node_id: n for n in keyword_nodes})

            retrieve_ids = vector_ids.intersection(keyword_ids) if self._mode == "AND" else vector_ids.union(keyword_ids)
            return [combined_dict[r_id] for r_id in retrieve_ids]

    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
    response_synthesizer = get_response_synthesizer()

    return RetrieverQueryEngine(retriever=custom_retriever, response_synthesizer=response_synthesizer)

# Initialize query engine
if 'custom_query_engine' not in st.session_state:
    with st.spinner("Loading data, please wait..."):
        st.session_state['custom_query_engine'] = initialize_query_engine()
    st.success("Ready! You can now ask questions.")

# Input for user query
query = st.text_input("Enter your question here:")

# Process the query if provided
if query:
    with st.spinner("Searching for an answer..."):
        custom_query_engine = st.session_state['custom_query_engine']
        result = custom_query_engine.query(query)
        response = result.response
        st.write("Response:")
        st.write(response)

        # Optional: Play response with TTS
        tts = gTTS(response, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            tts.save(audio_file.name)
            st.audio(audio_file.name)
        os.remove(audio_file.name)  # Remove temporary audio file
