import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import json

# Load environment variables
load_dotenv()
ENV_PATH = Path(".env")

# Check or request API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key_input = st.text_input("🔑 Enter your OpenAI API key:", type="password")
    if api_key_input:
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key_input}")
        st.success("API key saved. Please reload the page.")
        st.stop()
    else:
        st.stop()

# Page setup
st.set_page_config(page_title="Chat with Your PDFs")
st.title("📚 Chat with Your PDFs")

# FAISS persistent directory
FAISS_DIR = "faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

# Load or create FAISS vectorstore
def load_vectorstore():
    if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        return FAISS.load_local(FAISS_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return None

vectorstore = load_vectorstore()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            # Įkeliam seną duomenų bazę ir papildom
            existing_vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            existing_vectorstore.add_texts(texts)
            existing_vectorstore.save_local(FAISS_DIR)
            vectorstore = existing_vectorstore
        else:
            vectorstore = FAISS.from_texts(texts, embedding=embeddings)
            vectorstore.save_local(FAISS_DIR)

        st.success("PDFs processed and saved successfully! You can now ask questions.")
else:
    st.info("📦 Previously uploaded PDFs loaded from storage. You can now ask questions.")

# Ask questions
query = st.text_input("💬 Ask a question about your PDFs", key="user_input")
if query and vectorstore:
    docs = vectorstore.similarity_search(query)
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)

    # Save to session state
    st.session_state.chat_history.append({"question": query, "answer": response})

# Export history
if st.session_state.chat_history:
    export = st.button("⬇️ Export Chat History")
    if export:
        json_str = json.dumps(st.session_state.chat_history, indent=2, ensure_ascii=False)
        st.download_button("Download History as JSON", data=json_str, file_name="chat_history.json", mime="application/json")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🕘 Chat History")
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**Assistant:** {entry['answer']}")
        st.markdown("---")

elif query:
    st.warning("Please upload and process PDFs first.")