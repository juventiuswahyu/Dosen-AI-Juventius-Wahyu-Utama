import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --------------------------------------------------
# KONFIGURASI JUNIA
# --------------------------------------------------
AI_NAME = "JUNIA"
AI_FULL_NAME = "Juven's University & Intelligent Academic Assistant"
AI_AVATAR_URL = "https://raw.githubusercontent.com/juventiuswahyu/Dosen-AI-Juventius-Wahyu-Utama/main/Karakter%20Asisten%20AI.png"

SYS_PROMPT = (
    "Kamu adalah JUNIA (Juven's University & Intelligent Academic Assistant). "
    "Kamu asisten akademik yang ramah, sopan, dan profesional. "
    "Jawab pertanyaan HANYA berdasarkan materi referensi berikut.\n\n"
)

# 1. Page Config
st.set_page_config(
    page_title="JUNIA Academic Assistant",
    page_icon="🤖",
    layout="wide"
)

# CSS Kustom untuk Memperdekat Jarak Header & Merapikan Tombol Reset
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }
    h1 {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Ambil Kunci API
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")

if not GROQ_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    st.error("⚠️ Silakan atur Secrets pada Streamlit Cloud!")
    st.stop()

# 3. Model & Database
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "materi_library"

# Fungsi Reset Chat
def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya " + AI_NAME + " (" + AI_FULL_NAME + "). Ada yang ingin Anda tanyakan hari ini?"
        }
    ]

# 4. Sidebar
with st.sidebar:
    st.image(AI_AVATAR_URL, use_container_width=True)
    st.title("Panel JUNIA")
    st.caption(AI_FULL_NAME)
    st.markdown("---")
    st.header("📤 Unggah Materi PDF")
    
    uploaded_file = st.file_uploader("Pilih file PDF", type=["pdf"])
    
    if uploaded_file and st.button("Simpan ke Library", type="primary", use_container_width=True):
        with st.spinner("Memproses dokumen..."):
            temp_filename = "temp_" + uploaded_file.name
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_filename)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            splits = text_splitter.split_documents(docs)
            
            QdrantVectorStore.from_documents(
                documents=splits,
                embedding=embeddings,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=collection_name
            )
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            st.success("✅ Berhasil disimpan!")

    if st.button("🔄 Reset / Chat Baru", use_container_width=True):
