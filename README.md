
---

# Constitution Chatbot

This is a Streamlit-based chatbot that allows users to interact with PDF documents through text and voice inputs. Using **Natural Language Processing (NLP)**, **Whisper** for Speech-to-Text (STT), **gTTS** for Text-to-Speech (TTS), and **Gemini** embeddings, this chatbot can process user queries and respond with information extracted from the loaded PDFs.

View the app here
https://indconstitutionchatbot.streamlit.app/

## Features

- **Voice and Text Input**: Users can type queries or ask questions verbally.
- **Speech Recognition**: Utilizes Whisper to transcribe voice inputs into text.
- **Custom Document Indexing**: Processes and indexes PDF text for efficient querying.
- **Hybrid Retrieval System**: Combines keyword and vector-based search to retrieve relevant information.
- **Text-to-Speech Responses**: Responds to queries with audio using gTTS.

## Requirements

- **Python 3.8+**
- **Streamlit**
- **Streamlit WebRTC**
- **Whisper** (for speech recognition)
- **Gemini API** (for embeddings and LLM)
- **nltk**
- **gTTS**
- **scipy**
- **dotenv** (for environment variables)

## Installation

1. Clone the repository.
    ```bash
    git clone https://github.com/your-repo/constitution-chatbot.git
    cd constitution-chatbot
    ```

2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3. Download NLTK stopwords.
    ```python
    import nltk
    nltk.download('stopwords')
    ```

4. Set up your `.env` file to include your Gemini API key:
    ```plaintext
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Usage

1. Start the Streamlit app.
    ```bash
    streamlit run app.py
    ```

2. Open the app in your browser at `http://localhost:8501`.

3. Load your PDF documents into the `data` directory.

4. Select an input method (Text or Voice) to ask questions about the content of your PDF.

5. If using **Voice** input, click **Record Question** to begin recording. Your question will be submitted automatically after it has been recorded.

6. The chatbot will respond with an answer based on the content of your PDF. Voice responses can be played.

## How It Works

1. **Document Loaded**: The knowledge base has been preloaded in the app.
2. **Embedding & Retrieval**: Uses Gemini embeddings to create vector and keyword indexes.
3. **Custom Query Engine**: Combines vector and keyword retrieval to handle user queries.
4. **Audio Processing**: Records, transcribes, and responds to audio inputs.

## File Structure

- `app.py`: Main Streamlit app code.
- `data/`: Directory to store PDF documents.
- `requirements.txt`: List of dependencies.
- `.env`: Environment variables for API keys.

