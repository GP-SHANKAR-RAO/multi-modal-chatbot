import streamlit as st
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np
from faiss import IndexFlatL2

# ------------------------
# Gemini API Configuration
# ------------------------
GENI_API_KEY = ""  # Replace with your key
genai.configure(api_key=GENI_API_KEY)

# ------------------------
# Initialize Voice Components
# ------------------------streamlit run APP-TEST.py
engine = pyttsx3.init()
recognizer = sr.Recognizer()


# ------------------------
# RAG-based Chatbot Functions
# ------------------------
def load_and_preprocess_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Chunk size for splitting text
        chunk_overlap=200  # Overlap for context
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_index(doc_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.array([model.encode(chunk.page_content) for chunk in doc_chunks])
    texts = [chunk.page_content for chunk in doc_chunks]

    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    docstore = InMemoryDocstore({str(i): texts[i] for i in range(len(texts))})
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    vector_store = FAISS(
        embedding_function=model.encode,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vector_store


def query_gemini_model(query, retriever):
    try:
        # Retrieve similar chunks from the PDF document
        results = retriever.vectorstore.similarity_search(query, k=3)
        context_chunks = []
        for doc, _ in results:
            try:
                doc_id = int(doc.metadata['doc_id'])
                # Get the corresponding docstore id safely
                docstore_id = retriever.vectorstore.index_to_docstore_id.get(doc_id)
                if not docstore_id:
                    continue
                chunk = retriever.vectorstore.docstore._dict.get(docstore_id)
                if chunk:
                    context_chunks.append(chunk)
            except Exception:
                # Optionally log the error internally without showing it to the user
                continue
        context = "\n".join(context_chunks)
        genai_model = genai.GenerativeModel("gemini-1.5-pro")
        response = genai_model.generate_content(f"Context: {context}\nQuestion: {query}")
        return response.text
    except Exception:
        # Return a generic error message without exposing internal error details
        return "Answer: An error occurred while processing your request"



# ------------------------
# Voice-based Functions
# ------------------------
def process_voice_input():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.write(f"You said: {user_input}")
            return user_input
        except Exception as e:
            st.write(f"Voice recognition error: {e}")
            return None


def speak_text(text):
    # Save the TTS output to a file and return its path.
    output_file = "response.mp3"
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    return output_file


# ------------------------
# Streamlit UI
# ------------------------
st.title("Multi-Modal Chatbot Application")

# Let the user choose the mode
mode = st.radio("Select Chat Mode", ("Text Chat", "Voice Chat"))

# For text chat, we load the PDF vector index.
if mode == "Text Chat":
    @st.cache(allow_output_mutation=True)
    def load_vector_index():
        file_path = "BUSINESS DOCUMENTS.pdf"  # Path to your PDF document
        doc_chunks = load_and_preprocess_pdf(file_path)
        vector_index = create_vector_index(doc_chunks)
        return vector_index


    vector_index = load_vector_index()

if mode == "Text Chat":
    user_query = st.text_input("Enter your message:")
    if st.button("Send"):
        if user_query:
            response_text = query_gemini_model(user_query, vector_index.as_retriever())
            st.write("Bot:", response_text)
elif mode == "Voice Chat":
    if st.button("Record"):
        user_input = process_voice_input()
        if user_input:
            # For voice mode, send the transcript directly to Gemini API
            genai_model = genai.GenerativeModel("gemini-1.5-pro")
            response = genai_model.generate_content(user_input)
            response_text = response.text
            st.write("Bot:", response_text)

            # Convert response text to speech and play the audio
            audio_file = speak_text(response_text)
            st.audio(audio_file)

