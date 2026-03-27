# Importa o módulo para manipulação de variáveis de ambiente
import os

from dotenv import load_dotenv

# Loader responsável por ler arquivos PDF
from langchain_community.document_loaders import PyPDFLoader

# Responsável por dividir textos grandes em chunks menores
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Classe que transforma texto em embeddings vetoriais
from langchain_openai import OpenAIEmbeddings

# Banco vetorial para armazenamento e busca semântica
from langchain_community.vectorstores import Chroma

# Modelo de linguagem conversacional
from langchain_openai import ChatOpenAI

# Cadeia pronta de Perguntas e Respostas com RAG
from langchain_openai import ChatOpenAI

# Carrega as variáveis de ambiente
load_dotenv()
chaveApi = os.getenv("CHAVE_API")

# Caminho para os arquivos
base_path = os.path.join(os.getcwd(), "data")
caminhos_bules = [
    os.path.join(base_path, "dipirona.pdf"),
    os.path.join(base_path, "paracetamol.pdf")
]

# Lista que armazena os documentos
documentos = []

# Percorre os arquivos PDF
for caminho in caminhos_bules:
    # Cria o loader
    loader = PyPDFLoader(caminho)

    # Carrega os documentos
    docs = loader.load()

    # Adiciona o nome do medicamento como metadata
    for doc in docs:
        doc.metadata["medicamento"] = caminho.split("/")[-1].replace(".pdf", "")

    # Adiciona os documentos a lista
        documentos.extend(docs)

print(len(documentos))



