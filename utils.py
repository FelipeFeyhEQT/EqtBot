from pathlib import Path
import os

import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
from configs import *

# Carrega .env local (em prod o Streamlit Secrets já injeta)
_ = load_dotenv(find_dotenv())

# =============================
# Configurações via env/secrets
# =============================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)
MODEL_NAME = os.getenv(
    "MODEL_NAME", get_config("model_name")
)

PASTA_ARQUIVOS = Path(__file__).parent / "arquivos"


# =============================
# Importação dos documentos
# =============================
def importacao_documentos():
    documentos = []
    for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
        loader = PyPDFLoader(str(arquivo))
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos


# =============================
# Split dos documentos
# =============================
def split_de_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    documentos = splitter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata["source"] = doc.metadata.get("source", "").split("/")[-1]
        doc.metadata["doc_id"] = i

    return documentos


# =============================
# Vector Store (FAISS)
# =============================
def cria_vector_store(documentos):
    if not documentos:
        st.error("Nenhum documento PDF encontrado na pasta 'arquivos'.")
        st.stop()

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )

    return vector_store


# =============================
# Chain Conversacional
# =============================
def cria_chain_conversa():
    documentos = importacao_documentos()
    documentos = split_de_documentos(documentos)
    vector_store = cria_vector_store(documentos)

    chat = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0
    )

    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

    retriever = vector_store.as_retriever(
        search_type=get_config("retrieval_search_type"),
        search_kwargs=get_config("retrieval_kwargs")
    )

    prompt = PromptTemplate.from_template(
        get_config("prompt")
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    st.session_state["chain"] = chat_chain
