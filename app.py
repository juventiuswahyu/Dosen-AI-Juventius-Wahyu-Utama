import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ------------------------------------------------------------------
# KONFIGURASI KARAKTER ASISTEN AI (JUNIA)
# ------------------------------------------------------------------
AI_NAME = "JUNIA"
AI_FULL_NAME = "Juven's University & Intelligent Academic Assistant"

# URL Gambar dari GitHub Repository kamu (Raw URL)
AI_AVATAR_URL = "https://raw.githubusercontent.com/juventiuswahyu/Dosen-AI-Juventius-Wahyu-Utama/main/Karakter%20Asisten%20AI.png"

AI_SYSTEM_PROMPT = f"""
Kamu adalah '{AI_NAME}' ({AI_FULL_NAME}), asisten virtual akademik yang sangat profesional, ramah, dan pintar.
Peranmu adalah membantu mahasiswa/pengguna memahami materi perkuliahan yang sudah diunggah ke perpustakaan data.

Jawablah pertanyaan pengguna secara terstruktur dan jelas HANYA berdasarkan materi referensi di bawah ini:

Aturan Jawaban:
1. Sapalah pengguna dengan ramah dan profesional sebagai JUNIA.
2. Jika jawaban ditemukan di materi referensi, jelaskan secara mendalam, rapi, dan komunikatif dengan format poin-poin/penomoran.
3. Gunakan bahasa Indonesia yang baik, benar, dan santun.
4. Jika jawaban sama sekali tidak ada di dalam materi referensi, katakan dengan sopan: "Maaf, topik tersebut belum tersedia di dalam perpustakaan materi yang diunggah saat ini. Silakan tanyakan hal lain terkait materi yang tersedia."
5. Jangan mengarang jawaban di luar materi referensi.
"""

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title=f"{AI_NAME} - {AI_FULL_NAME}", 
    page_icon="🤖", 
    layout="wide"
)

# 2. Ambil Kunci API dari Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")

if not GROQ_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    st.error("⚠️ Silakan atur GROQ_API_KEY, QDRANT_URL, dan QDRANT_API_KEY pada menu Secrets di Streamlit Cloud!")
    st.stop()

# 3. Inisialisasi Embeddings & Model Groq
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

llm = ChatGroq(
    model_name="openai/gpt-oss-120b", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)

# 4. Koneksi ke Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "materi_library"

# 5. Sidebar
with st.sidebar:
    st.image(AI_AVATAR_URL, use_container_width=True)
    st.title(f"Panel {AI_NAME}")
    st.caption(AI_FULL_NAME)
    st.markdown("---")
    st.header("📤 Unggah Materi PDF")
    
    uploaded_file = st.file_uploader("Pilih file PDF materi", type=["pdf"])
    
    if uploaded_file and st.button("Simpan ke Library", type="primary", use_container_width=True):
        with st.spinner("Sedang membaca dan memproses dokumen..."):
            temp_filename = f"temp_{uploaded_file.name}"
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
                
            st.success(f"✅ '{uploaded_file.name}' berhasil ditambahkan!")

    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header Utama Aplikasi
st.title(f"🤖 {AI_NAME}")
st.subheader(f"*{AI_FULL_NAME}*")
st.caption("Selamat datang! Silakan tanyakan apa saja terkait materi perkuliahan
