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

# DAFTAR MATERI PERTEMUAN
MODUL_PERTEMUAN = [
    {
        "pertemuan": "Pertemuan 1",
        "judul": "Pengantar Pemasaran & Positioning",
        "cover": "https://raw.githubusercontent.com/juventiuswahyu/Dosen-AI-Juventius-Wahyu-Utama/main/Karakter%20Asisten%20AI.png",
        "drive_link": "https://drive.google.com"
    },
    {
        "pertemuan": "Pertemuan 2",
        "judul": "Analisis Pasar & Perilaku Konsumen",
        "cover": "https://raw.githubusercontent.com/juventiuswahyu/Dosen-AI-Juventius-Wahyu-Utama/main/Karakter%20Asisten%20AI.png",
        "drive_link": "https://drive.google.com"
    }
]

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

# CSS Kustom
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

# 2. Ambil Kunci API & PIN Admin
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")
ADMIN_PIN = st.secrets.get("ADMIN_PIN", "1234")

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
    
    st.header("🔒 Akses Dosen")
    input_pin = st.text_input("Masukan PIN Khusus Dosen:", type="password")
    
    if input_pin == ADMIN_PIN:
