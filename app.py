import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Set Halaman Streamlit
st.set_page_config(
    page_title="AI Library Materi Khusus", 
    page_icon="📚", 
    layout="wide"
)

st.title("📚 AI Assistant Library Materi Khusus (Groq)")
st.write("Sistem Tanya Jawab Materi Khusus dengan Kapasitas Database Besar (Qdrant Cloud).")

# 1. Ambil Kunci API dari Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")

# Cek Ketersediaan Secrets
if not GROQ_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    st.error("⚠️ Silakan atur GROQ_API_KEY, QDRANT_URL, dan QDRANT_API_KEY pada menu Secrets di Streamlit Cloud!")
    st.stop()

# 2. Inisialisasi Model Embedding dan Model LLM Groq
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Menggunakan ID model gpt-oss-120b yang valid pada API Groq
llm = ChatGroq(
    model_name="openai/gpt-oss-120b", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)

# 3. Koneksi ke Qdrant Vector Database
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "materi_library"

# 4. Sidebar: Upload PDF & Unggah ke Database
with st.sidebar:
    st.header("📤 Upload Dokumen Materi")
    st.info("💡 Upload PDF materi kamu satu per satu di sini.")
    
    uploaded_file = st.file_uploader("Upload File PDF Materi", type=["pdf"])
    
    if uploaded_file and st.button("Proses & Simpan ke Database", type="primary"):
        with st.spinner("Membaca, memecah teks, dan mengirim ke Qdrant Vector Database..."):
            temp_filename = f"temp_{uploaded_file.name}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Load PDF & Pecah Teks
                loader = PyPDFLoader(temp_filename)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=150
                )
                splits = text_splitter.split_documents(docs)
                
                # Simpan Dokumen ke Qdrant Vector Database
                QdrantVectorStore.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name=collection_name
                )
                st.success(f"✅ File '{uploaded_file.name}' berhasil diproses dan disimpan ke AI Library!")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

# 5. Interface Main Area: Tanya Jawab
st.subheader("💬 Tanya Jawab Seputar Materi")

query = st.text_input("Ketik pertanyaan kamu tentang materi di sini:")

if query:
    with st.spinner("Mencari jawaban terbaik dari materi..."):
        try:
            # Hubungkan ke Vector Store Qdrant
            vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=collection_name,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            
            # Cari 4 dokumen terdekat/paling relevan
            docs = vector_store.similarity_search(query, k=4)
            
            if not docs:
                st.warning("Belum ada dokumen materi yang tersimpan di database.")
            else:
                context = "\n\n---\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"""Kamu adalah asisten instruktur yang ramah dan ahli. 
Jawab pertanyaan pengguna secara akurat dan jelas HANYA berdasarkan materi referensi di bawah ini:

[MATERI REFERENSI]
{context}

[PERTANYAAN]
{query}

Catatan Aturan Jawab:
1. Jika jawaban ditemukan di materi referensi, jelaskan secara mendalam.
2. Jika jawaban sama sekali tidak ada di dalam materi referensi, katakan: "Maaf, materi tersebut belum ada di perpustakaan data." Jangan mengarang jawaban di luar materi.
"""
                response = llm.invoke(prompt)
                
                st.markdown("### 🤖 **Jawaban AI:**")
                st.write(response.content)
                
                with st.expander("📄 Lihat Potongan Sumber Materi"):
                    st.write(context)
                    
        except Exception as e:
            st.error(f"Gagal melakukan pencarian di database (pastikan kamu sudah upload minimal 1 file PDF): {e}")
