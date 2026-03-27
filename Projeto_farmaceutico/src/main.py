# Importa o módulo para manipulação de variáveis de ambiente
import os

from dotenv import load_dotenv

# Loader responsável por ler arquivos PDF
from langchain.document_loaders import PyPDFLoader

# Responsável por dividir textos grandes em chunks menores
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Classe que transforma texto em embeddings vetoriais
from langchain.embeddings import OpenAIEmbeddings

# Banco vetorial para armazenamento e busca semântica
from langchain.vectorstores import Chroma

# Modelo de linguagem conversacional
from langchain.chat_models import ChatOpenAI

# Cadeia pronta de Perguntas e Respostas com RAG
from langchain.chains import RetrievalQA

# Carrega as variáveis de ambiente
load_dotenv()
chaveApi = os.getenv("CHAVE_API")

