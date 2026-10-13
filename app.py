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

# 2. Ambil Kunci API & PIN Admin dari Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")
ADMIN_PIN = st.secrets.get("ADMIN_PIN", "1234")  # Default PIN: 1234 jika belum diatur di secrets

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

# 4. Sidebar (Diakses oleh Dosen & Mahasiswa)
with st.sidebar:
    st.image(AI_AVATAR_URL, use_container_width=True)
    st.title("Panel JUNIA")
    st.caption(AI_FULL_NAME)
    st.markdown("---")
    
    # Mode Akses Dosen / Admin
    st.header("🔒 Akses Dosen")
    input_pin = st.text_input("Masukan PIN Khusus Dosen:", type="password")
    
    if input_pin == ADMIN_PIN:
        st.success("🔓 Akses Dosen Aktif")
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
    elif input_pin != "":
        st.error("❌ PIN Salah!")
        
    st.markdown("---")
    if st.button("🔄 Reset Chat", key="btn_sidebar_reset", use_container_width=True):
        reset_chat()
        st.rerun()

# 5. Header Utama
st.title("🤖 " + AI_NAME)

# 6. Chat History Init
if "messages" not in st.session_state:
    reset_chat()

# 7. Layout Utama (Karakter & Chat)
col1, col2 = st.columns([1, 3])

with col1:
    st.image(AI_AVATAR_URL, width=250)

with col2:
    st.subheader("💬 Diskusi dengan " + AI_NAME)
    
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown("**🤖 " + AI_NAME + ":**")
            st.info(message["content"])
        else:
            st.markdown("**👤 Anda:**")
            st.success(message["content"])
            
    # Input Teks Pertanyaan
    prompt = st.text_input(
        label="Ketik pertanyaan Anda di sini:",
        placeholder="Contoh: Apa itu positioning?"
    )
    
    # Tombol Berdampingan Kiri-Kanan
    c1, c2, c3 = st.columns([1.5, 1, 2])
    submit_button = c1.button("Kirim Pertanyaan 🚀", type="primary", use_container_width=True)
    reset_button = c2.button("🔄 Reset Chat", use_container_width=True)

# 8. Proses Aksi (Kirim / Reset)
if reset_button:
    reset_chat()
    st.rerun()

if submit_button and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    docs = vector_store.similarity_search(prompt, k=4)
    
    if not docs:
        response_text = "Maaf, perpustakaan materi Anda masih kosong. Silakan minta Dosen untuk mengunggah PDF."
    else:
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        full_prompt = (
            SYS_PROMPT + 
            "[REFERENSI]\n" + context_text + 
            "\n\n[PERTANYAAN]\n" + prompt
        )
        res = llm.invoke(full_prompt)
        response_text = res.content

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()
