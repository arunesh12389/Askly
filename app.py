import sentence_transformers
from sentence_transformers import SentenceTransformer 
import streamlit as st
import tempfile
import os
import time
import numpy as np
from dotenv import load_dotenv
from pydantic import SecretStr
from typing import List
from utils import extract_text_from_pdf, chunk_text, create_vector_store
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA  # Keep this import
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in your .env file.")
GROQ_API_KEY = SecretStr(api_key)

# ---------- Page config ----------
st.set_page_config(
    page_title="Askly – Knowledge-based Search Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Header ----------
st.title("🛰️ Askly – Knowledge-based Search Engine")
st.caption("Upload documents, build a vector store, and ask Askly questions.")

st.markdown("---")

# ---------- Sidebar: quick stats & controls ----------
with st.sidebar:
    st.markdown("### ⚙️ Askly Controls")
    st.write("Upload multiple PDFs, rebuild the index, and manage memory.")
    clear_sessions = st.button("🗑️ Clear session")

if clear_sessions:
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    st.rerun()

# ---------- Initialize session state ----------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "timings" not in st.session_state:
    st.session_state["timings"] = []
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# ---------- Upload & Processing Card ----------
with st.container(border=True):
    st.subheader("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files. Askly will merge the extracted text into one index."
    )

    if uploaded_files:
        st.write("Files to process:")
        for f in uploaded_files:
            st.caption(f"• {f.name} — {f.size/1024:.1f} KB")
    
    process_btn = st.button("Process Files and Build Index")

# ---------- Process uploaded files ----------
def process_pdf_files(file_objs: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """Save uploaded st files to temp files, extract text, chunk, and create vector store."""
    combined_texts = []
    file_count = len(file_objs)
    
    progress_bar = st.progress(0, text="Starting processing...")
    
    for idx, f in enumerate(file_objs):
        progress_text = f"Extracting text from {f.name} ({idx+1}/{file_count})..."
        progress_bar.progress((idx+1)/file_count, text=progress_text)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        
        extracted = extract_text_from_pdf(tmp_path)
        combined_texts.append((f.name, extracted))
        
        try:
            os.remove(tmp_path)
        except Exception:
            pass
            
    progress_bar.empty()
    st.success(f"Extracted text from {file_count} files.")
    
    merged_text = "\n\n".join([f"---\nSource: {name}\n---\n{text}" for name, text in combined_texts])
    
    st.info("Chunking combined text...")
    chunks = chunk_text(merged_text)
    st.success(f"Chunked into {len(chunks)} pieces.")
    
    st.info("Creating vector store (this may take a short while)...")
    vstore = create_vector_store(chunks)
    st.success("Vector store ready.")
    
    st.session_state["vector_store"] = vstore
    st.session_state["uploaded_files"] = [f.name for f in file_objs]
    return vstore

if process_btn and uploaded_files:
    try:
        with st.spinner("Processing files..."):
            process_pdf_files(uploaded_files)
    except Exception as e:
        st.error(f"Error while processing files: {e}")

def create_retrieval_chain(retriever, llm):
    """
    [FIXED] Creates a RetrievalQA chain using the correct 'from_chain_type' method.
    The complex QAChainAdapter is no longer needed.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff" 
    )
    return qa_chain

# ---------- Q&A / Interaction Card ----------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("💬 Ask Askly")

    if st.session_state.get("vector_store") is None:
        st.info("Upload PDFs and click 'Process Files' to start asking questions.")
    else:
        user_question = st.chat_input("Type your question about the uploaded documents")

        if st.session_state["is_generating"]:
            st.warning("⏳ Askly is generating an answer...")

        if user_question:
            st.session_state["is_generating"] = True
            with st.spinner("⏳ Generating answer..."):
                try:
                    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)
                    retriever = st.session_state["vector_store"].as_retriever()
                    
                    qa_chain = create_retrieval_chain(retriever, llm)

                    t0 = time.perf_counter()
                    response = qa_chain.invoke({"query": user_question})
                    t1 = time.perf_counter()

                    response_time = t1 - t0
                    answer_str = response.get("result") if isinstance(response, dict) else str(response)

                    st.session_state["chat_history"].append({
                        "question": user_question,
                        "answer": answer_str,
                        "time": response_time
                    })
                    st.session_state["timings"].append(response_time)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                finally:
                    st.session_state["is_generating"] = False
                    st.rerun() # Rerun to display the new chat message


# ---------- Display chat history ----------
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

if st.session_state["chat_history"]:
    st.subheader("🧾 Conversation")

    for chat in reversed(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.write(chat['question'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            st.caption(f"⏱ {chat['time']:.2f}s") # Cleaner time display

    if len(st.session_state["timings"]) >= 1:
        arr = np.array(st.session_state["timings"])
        
        with st.expander("📊 Performance Summary"):
            st.write(
                f"- Average response: {arr.mean():.2f}s  \n"
                f"- Median response: {np.median(arr):.2f}s  \n"
                f"- 95th percentile: {np.percentile(arr,95):.2f}s"
            )

# ---------- Footer ----------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("Askly · Built with Streamlit and Groq LLM")