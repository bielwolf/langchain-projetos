 # Importações necessárias
import os 
import streamlit as st
from dotenv import load_dotenv

# Variáveis de ambiente
load_dotenv()

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings e LLMs
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Vectorstores
from langchain_community.vectorstores import Chroma

# Prompts
from langchain_core.prompts import PromptTemplate


# Diretório do banco vetorial
PERSIST_DIRECTORY = "../chroma_rh"

# Modelo de embedding
EMBEDDING_MODEL = "text-embedding-3-small"

# Modelo de linguagem
LLM_MODEL = "gpt-3.5-turbo"


@st.cache_data
def carregar_documentos():
    """
    Carrega os documentos PDF e os divide em chunks.
    """

    base_path = os.path.join(os.getcwd(), "data")
    caminhos = [
        os.path.join(base_path, "codigo_conduta.pdf"),
        os.path.join(base_path, "politica_ferias.pdf"),
        os.path.join(base_path, "politica_home_office.pdf")
    ]

    documentos = []

    for caminho in caminhos:
        loader = PyPDFLoader(caminho)
        docs = loader.load()

        for doc in docs: 
            doc.metadata["documento"] = caminho

        documentos.extend(docs)

    return documentos

def gerar_chunks(documentos):
    """
    Divide os documento em chunks semanticos.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    return splitter.split_documents(documentos)

def enriquecer_chunks(chunks):
    """
    Classifica os chunks por categoria semântica
    """
    for chunk in chunks:
        texto = chunk.page_content.lower()

        if "férias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "etica" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"
    return chunks

@st.cache_resource
def criar_vectorstore(_chunks):
    """
    Cria ou carrega o banco vetorial a partir dos chunks.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore

def rerank_documentos(pergunta, documentos, llm):
    """
    Reordena os documentos recuperados com base na relevância
    usando o próprio LLM (reranking semântico)
    """
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""..."""
    )
    
    documentos_com_score = []
    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0
        
        documentos_com_score.append((score, doc))

    # Ordena do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )
    # Retorna apenas os documentos
    return [doc for _, doc in documentos_ordenados]

def responder_pergunta(pergunta, vectorstore):
    """
    Pipeline completo:
    - Recuperação
    - Reranking
    - Geração de resposta
    """
    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # Recuperação inicial (top-k mais alto)
    documentos_recuperados = vectorstore.similarity_search(
        pergunta,
        k=8
    )

    # Reranking
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )
    
    # Seleciona os melhores
    contexto_final = documentos_rerankeados[:4]

    # Prompt final
    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    prompt_final = f"""..."""

    resposta = llm.invoke(prompt_final)

    return resposta.content, contexto_final

st.set_page_config(page_title="Agente de RH com RAG", layout="wide")
st.title("🤖 Agente de RH — Políticas Internas")

pergunta = st.text_input("Digite sua pergunta sobre políticas internas de RH:")

if pergunta:
    with st.spinner("Consultando políticas internas..."):
        documentos = carregar_documentos()
        chunks = gerar_chunks(documentos)
        chunks = enriquecer_chunks(chunks)
        vectorstore = criar_vectorstore(chunks)

        resposta, fontes = responder_pergunta(pergunta, vectorstore)

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Fontes utilizadas")
    for i, doc in enumerate(fontes, start=1):
        st.markdown(f"**Trecho {i}**")
        st.write(f"Documento: {doc.metadata.get('documento')}")
        st.write(f"Categoria: {doc.metadata.get('categoria')}")
        st.write(doc.page_content)
        st.divider()