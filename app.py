import streamlit as st
import os
import fitz  # PyMuPDF
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# === NUSTATYMAI ===
STORAGE_DIR = "storage"
PASSWORD = "milijonas"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === Slaptažodis ===
st.set_page_config(page_title="📚 Pokalbis su knygomis", layout="wide")
st.title("📚 Pokalbis su tavo PDF knygomis")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Įveskite slaptažodį:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.experimental_rerun()
    else:
        st.stop()

# === API raktas ===
api_key = st.text_input("🗝️ Įveskite OpenAI API raktą:", type="password")
if not api_key:
    st.warning("Reikia įvesti API raktą.")
    st.stop()

# === Knygų įkėlimas ===
st.markdown("---")
st.subheader("📤 Įkelkite PDF knygą (-as)")
uploaded_files = st.file_uploader("Pasirinkite PDF failus", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_id = Path(file.name).stem
        save_path = os.path.join(STORAGE_DIR, file_id)
        if os.path.exists(save_path):
            st.info(f"📘 '{file.name}' jau yra įkelta.")
            continue

        st.write(f"⏳ Apdorojama: {file.name}")
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            full_text = "\n".join([page.get_text() for page in doc])

        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = splitter.split_text(full_text)
        documents = [Document(page_content=t) for t in texts]

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(save_path)
        st.success(f"✅ Įkelta: {file.name}")

# === Galimų knygų sąrašas ===
st.markdown("---")
st.subheader("📚 Pasirinkite knygą arba visas")
book_dirs = [d for d in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, d))]
book_choice = st.selectbox("Pasirinkite knygą", options=["Visos"] + book_dirs)

# === Klausimas ===
st.markdown("---")
st.subheader("💬 Užduokite klausimą")
question = st.text_input("Klausimas apie PDF turinį")

if question and book_choice:
    with st.spinner("🧠 Mąstoma..."):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        if book_choice == "Visos":
            dbs = []
            for b in book_dirs:
                db = FAISS.load_local(os.path.join(STORAGE_DIR, b), embeddings, allow_dangerous_deserialization=True)
                dbs.append(db)
            merged_db = dbs[0]
            for db in dbs[1:]:
                merged_db.merge_from(db)
            retriever = merged_db.as_retriever()
        else:
            db = FAISS.load_local(os.path.join(STORAGE_DIR, book_choice), embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever()

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa.run(question)

        st.success(answer)