import os
import streamlit as st
import warnings
import logging
from typing import Optional, List
from dotenv import load_dotenv

# ======================
# Load .env file (LOCAL)
# ======================
load_dotenv()

# ======================
# Groq API Key (SECURE)
# ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Check your .env or Streamlit Secrets.")

# ======================
# LangChain Imports
# ======================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.base import LLM

# ======================
# Groq SDK
# ======================
from groq import Groq

# ======================
# Suppress Warnings
# ======================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ======================
# Custom Groq LLM
# ======================
class GroqLLM(LLM):
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("📄 Chatbot 🤖")

PDF_PATH = "Iqra_University_Student_Handbook.pdf"

# ======================
# Session State
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ======================
# Load & Cache Vector Store
# ======================
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("❌ Iqra_University_Student_Handbook.pdf not found.")

    loader = PyPDFLoader(PDF_PATH)

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}  # <-- THIS IS THE FIX
        ),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    )

    return index.from_loaders([loader]).vectorstore

# ======================
# Chat Input
# ======================
prompt = st.chat_input("Ask anything about the student handbook...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    try:
        vectorstore = load_vectorstore()
        llm = GroqLLM()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = qa_chain({"query": prompt})
        answer = result["result"]

        # ======================
        # SMART GENERIC FALLBACK
        # ======================
        low_answer = answer.lower()

        if (
            "i don't know" in low_answer
            or "does not contain" in low_answer
            or "not mentioned" in low_answer
            or "no information" in low_answer
        ):
            answer = llm(
                f"Answer this question in a clear and general academic way: {prompt}"
            )

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error(f"🚫 Error: {str(e)}")
