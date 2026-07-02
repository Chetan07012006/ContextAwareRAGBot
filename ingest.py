import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

FAQ_FILE = "company_faq.txt"
CHROMA_DIR = "chroma_db"

# Check if FAQ exists
if not os.path.exists(FAQ_FILE):
    print("Error: company_faq.txt not found.")
    exit()

print("Loading FAQ document...")

loader = TextLoader(FAQ_FILE, encoding="utf-8")
documents = loader.load()

print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("Vector database created successfully!")