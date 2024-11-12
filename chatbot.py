import streamlit as st
from scipy.io.wavfile import write
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import os
import re
import tempfile
import numpy as np
import whisper
from gtts import gTTS
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex, get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv

nltk.download('stopwords')
load_dotenv()

# Whisper model for transcription
whisper_model = whisper.load_model("base")

# Title
st.title("The Indian Constitution Chatbot")

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

# Load and initialize query engine
def initialize_query_engine():
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    if 'documents' not in st.session_state:
        documents = SimpleDirectoryReader('data').load_data()
        st.session_state['documents'] = [preprocess_text(doc.text) for doc in documents]

    embed_model = GeminiEmbedding(api_key=GEMINI_API_KEY)
    llm = Gemini(api_key=GEMINI_API_KEY)
    
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(st.session_state['documents'])
    
    vector_index = VectorStoreIndex(storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(storage_context=storage_context)
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, top_k=5)

    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
    response_synthesizer = get_response_synthesizer()

    return RetrieverQueryEngine(retriever=custom_retriever, response_synthesizer=response_synthesizer)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def start_recording(self):
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame.to_ndarray().flatten())
        return frame

# Transcription
def transcribe_audio(frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        audio_data = (np.concatenate(frames) * 32767).astype(np.int16)
        write(temp_wav.name, 16000, audio_data)
    return whisper_model.transcribe(temp_wav.name)["text"]

# TTS
def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(text=text, lang="en")
        tts.save(temp_audio.name)
        return temp_audio.name

# Initialize Audio Processor
if 'audio_processor' not in st.session_state:
    st.session_state['audio_processor'] = AudioProcessor()

# Start/Stop Buttons
if st.button("Start Recording"):
    st.session_state['webrtc_ctx'] = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: st.session_state['audio_processor'],
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )
    st.session_state['audio_processor'].start_recording()
    st.write("Recording started...")

if st.button("Stop Recording"):
    frames = st.session_state['audio_processor'].frames
    if frames:
        st.write("Recording stopped.")
        transcription = transcribe_audio(frames)
        st.session_state['transcribed_text'] = transcription
        st.write(f"Transcribed Text: {transcription}")
    else:
        st.write("No audio frames captured.")

# Query Engine Initialization
if 'query_engine' not in st.session_state:
    with st.spinner("Loading data..."):
        st.session_state['query_engine'] = initialize_query_engine()
    st.success("Ready!")

# Text or Voice Input
input_method = st.radio("Select input method:", ("Text", "Voice"))
query = st.text_input("Enter your question:") if input_method == "Text" else st.session_state.get('transcribed_text')

# Process Query
if query:
    with st.spinner("Fetching response..."):
        response = st.session_state['query_engine'].query(query).response
        st.write("Response:")
        st.write(response)
        
        if input_method == "Voice":
            audio_file = text_to_speech(response)
            with open(audio_file, "rb") as f:
                st.audio(f.read(), format='audio/mp3')
            os.remove(audio_file)
