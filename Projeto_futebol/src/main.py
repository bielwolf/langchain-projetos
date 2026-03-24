# Importações básicas
import os

# Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# Divisão de texto
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Banco vetorial
from langchain_community.vectorstores import Chroma

# LLM
from langchain_openai import ChatOpenAI

# Cadeia RAG
from langchain.chains import RetrievalQA

# Caminho do PDF
CAMINHO_PDF = os.path.join(os.getcwd(), "data", "regras_futebol.pdf")

# Carregamento do PDF
loader = PyPDFLoader(CAMINHO_PDF)
documentos = loader.load()

# Quantidade de páginas
print(len(documentos))

# Divisão de texto - Chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chuncks = text_splitter.split_documents(documentos)
print(len(chuncks))

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="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"
)

# Banco vetorial
vectorStore = Chroma.from_documents(
    documents=chuncks,
    embedding=embeddings,
    persist_directory="chroma_regras"
)

# Cadeia RAG
retriver = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

# Inicialização da LLM
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    openai_api_key="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"
)

# Cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriver,
    return_source_documents=True
)

pergunta = "Um jogador pode usar as mãos em uma partida de futebol?"
resposta = qa_chain(pergunta)

print("Pergunta: ", pergunta)
print("Resposta: ", resposta["result"])

print("\nTrechos utilizado com contexto:\n")
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f'Fonte: {doc.metada.get("source", "Desconhecido")}')
    print(f'Página: {doc.metadata.get("page", "N/A")}')
    print("Conteúdo:")
    print(doc.page_content)
    print("\n")

