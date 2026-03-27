# Importa o módulo para manipulação de variáveis de ambiente
import os

import random

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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega as variáveis de ambiente
load_dotenv()

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

# Cria o splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, # Tamanho dos chunks
    chunk_overlap=150 # Overlap entre os chunks
    )

# Divide os documentos em chunks
chunks = text_splitter.split_documents(documentos)
print(len(chunks))

# Percorre cada chunk para classificar semanticamente seu conteúdo
for chunk in chunks:

    # Normaliza o texto para facilitar as verificações
    texto = chunk.page_content.lower()

    # Identificação do medicamento
    if "identificação do medicamento" in texto or "composição" in texto:
        chunk.metadata["categoria"] = "identificacao"

    # Indicações terapêuticas
    elif "indicação" in texto or "para que este medicamento é indicado" in texto:
        chunk.metadata["categoria"] = "indicacao"

    # Funcionamento do medicamento
    elif "como este medicamento funciona" in texto or "ação" in texto:
        chunk.metadata["categoria"] = "como_funciona"

    # Contraindicações
    elif "contraindicação" in texto or "quando não devo usar" in texto:
        chunk.metadata["categoria"] = "contraindicacao"

    # Advertências e precauções
    elif "advertência" in texto or "precaução" in texto or "o que devo saber antes de usar" in texto:
        chunk.metadata["categoria"] = "advertencias_precaucoes"

    # Interações medicamentosas
    elif "interação" in texto or "interações medicamentosas" in texto:
        chunk.metadata["categoria"] = "interacoes"

    # Posologia e modo de uso
    elif "dose" in texto or "posologia" in texto or "como devo usar" in texto:
        chunk.metadata["categoria"] = "posologia_modo_uso"

    # Reações adversas
    elif "reações adversas" in texto or "quais os males" in texto:
        chunk.metadata["categoria"] = "reacoes_adversas"

    # Armazenamento
    elif "onde, como e por quanto tempo posso guardar" in texto or "armazenar" in texto:
        chunk.metadata["categoria"] = "armazenamento"

    # Superdosagem
    elif "quantidade maior do que a indicada" in texto or "superdosagem" in texto:
        chunk.metadata["categoria"] = "superdosagem"

    # Conteúdo geral / administrativo
    else:
        chunk.metadata["categoria"] = "geral"

# Seleciona dois chunks aleatórios
chunks_aleatorios = random.sample(chunks, 2)

for i, chunk in enumerate(chunks_aleatorios, start=1):
    print(f"\n--- Chunk Aleatório {i} ---")
    print(f"Metadados: {chunk.metadata}")
    print("\nConteúdo (início):")
    print(chunk.page_content[:300])

# Inicializa o modelo de embeddings (forma atual)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small" # opcional, mas recomendado
)

# Cria o banco vetorial com os chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_bulas"
)

# Cria o retriever para busca semântica
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4} # Número de chunks retornados
)

# Inicializa o modelo de linguagem
llm = ChatOpenAI(
    model="gpt-4o-mini",
)

# Cria o prompt
prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado em bulas de medicamentos.

Use apenas o contexto abaixo para responder à pergunta.
Se não souber a resposta, diga claramente que não encontrou a informação.

Contexto:
{context}

Pergunta:
{input}
""")

# Cria a cadeia RAG
question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

qa_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Pergunta de teste 1
pergunta = "Quais são as contraindicações da dipirona?"

# Executa a pergunta no agente RAG (forma atual)
resposta = qa_chain.invoke({"input": pergunta})
print(f"Pergunta: {pergunta}")
print(f"\nResposta do Agente: \n{resposta['answer']}")

print("\nTrechos utilizados como contexto:\n")
for i, doc in enumerate(resposta["context"], start=1):
    print(f"--- Trecho {i} ---")

    # Metadados principais
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    
    # Conteúdo recuperado
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print("\n" + "="*50)

# Pergunta de teste 2
pergunta = "Qual é a posologia recomendada do paracetamol para adultos?"

# Executa a pergunta no agente RAG
resposta = qa_chain.invoke({"input": pergunta})

print(f"Pergunta: {pergunta}")
print(f"\nResposta do Agente: \n{resposta['answer']}")

print("\nTrechos utilizados como contexto:\n")
for i, doc in enumerate(resposta["context"], start=1):
    print(f"--- Trecho {i} ---")
    
    # Metadados principais
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    
    # Conteúdo recuperado
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print("\n" + "="*50)