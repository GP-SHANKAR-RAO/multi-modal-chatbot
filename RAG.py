import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np


# Step 1: Load and preprocess the PDF
def load_and_preprocess_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split text into 1000-character chunks
        chunk_overlap=200  # Allow some overlap for better context
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# Step 2: Create a vector index
def create_vector_index(doc_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use MiniLM for embeddings
    embeddings = np.array([model.encode(chunk.page_content) for chunk in doc_chunks])
    texts = [chunk.page_content for chunk in doc_chunks]

    # Create FAISS index and document store
    from faiss import IndexFlatL2
    index = IndexFlatL2(embeddings.shape[1])  # Initialize FAISS index
    index.add(embeddings)  # Add embeddings to the FAISS index

    docstore = InMemoryDocstore({str(i): texts[i] for i in range(len(texts))})
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    vector_store = FAISS(embedding_function=model.encode, index=index, docstore=docstore,
                         index_to_docstore_id=index_to_docstore_id)
    return vector_store


# Step 3: Query the Gemini Generative Model
def query_gemini_model(query, retriever):
    try:
        # Perform similarity search
        results = retriever.vectorstore.similarity_search(query, k=3)
        # Retrieve document contents
        context = "\n".join([retriever.vectorstore.docstore._dict[
                                 retriever.vectorstore.index_to_docstore_id[int(doc.metadata['doc_id'])]] for doc, _ in
                             results])
        # Configure Gemini API
        genai.configure(api_key="")
        genai_model = genai.GenerativeModel("gemini-1.5-pro")
        # Generate response
        response = genai_model.generate_content(f"Context: {context}\nQuestion: {query}")
        return response.text
    except ValueError as e:
        return f"Error during query processing: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# Main Function
if __name__ == "__main__":
    # Path to the business document
    file_path = "BUSINESS DOCUMENTS.pdf"

    # Load and preprocess the PDF
    doc_chunks = load_and_preprocess_pdf(file_path)

    # Create vector index
    vector_index = create_vector_index(doc_chunks)

    print("Chatbot is ready. Type 'exit' to end the chat.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        bot_response = query_gemini_model(user_query, vector_index.as_retriever())
        print("Bot:", bot_response)
