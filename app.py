import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load .env variables
load_dotenv()
ENV_PATH = Path(".env")

# Check or request API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key_input = st.text_input("🔑 Enter your OpenAI API key:", type="password")
    if api_key_input:
        ENV_PATH.write_text(f"OPENAI_API_KEY={api_key_input}")
        st.success("API key saved. Reloading...")
        st.rerun()
    else:
        st.stop()

# Page setup
st.set_page_config(page_title="Chat with Your PDFs")
st.title("📚 Chat with Your PDFs")

# FAISS persistent directory
FAISS_DIR = "storage/faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

# Load or create FAISS vectorstore
def load_vectorstore():
    if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        return FAISS.load_local(FAISS_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return None

vectorstore = load_vectorstore()

# Only allow upload if vectorstore doesn't exist yet
if not vectorstore:
    uploaded_files = st.file_uploader("Upload one or more PDF files (only required once)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        raw_text = ""
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embedding=embeddings)
        vectorstore.save_local(FAISS_DIR)
        st.success("PDFs processed and saved successfully! You can now ask questions.")
else:
    st.info("📦 Previously uploaded PDFs loaded from storage. You can now ask questions.")

# Ask questions
query = st.text_input("💬 Ask a question about your PDFs")
if query and vectorstore:
    docs = vectorstore.similarity_search(query)
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.write(response)
elif query:
    st.warning("Please upload and process PDFs first.")