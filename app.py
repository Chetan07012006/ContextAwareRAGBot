import os
import sqlite3

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Error: GROQ_API_KEY not found.")
    print("Please add it inside your .env file.")
    exit()

# -----------------------------
# Load Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load Chroma Vector Database
# -----------------------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------
# Load Llama 3
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# -----------------------------
# Prompt Template
# -----------------------------
template = """
You are an AI customer support assistant.

You are speaking with:

Name: {name}

Membership Tier: {membership_tier}

Answer the user's question using ONLY the context below.

If the answer is not available in the context, reply exactly:

"I do not have enough information in the provided knowledge base to answer this."

Context:
{context}

User Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=[
        "name",
        "membership_tier",
        "context",
        "question"
    ],
    template=template
)

# -----------------------------
# Connect SQLite
# -----------------------------
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

print("=" * 60)
print("Context-Aware Customer Support RAG Bot")
print("=" * 60)

while True:

    user_input = input("\nEnter User ID (or type exit): ")

    if user_input.lower() == "exit":
        break

    if not user_input.isdigit():
        print("Please enter a valid numeric user ID.")
        continue

    user_id = int(user_input)

    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id=?",
        (user_id,)
    )

    user = cursor.fetchone()

    if not user:
        print("User not found. Please enter a valid user_id.")
        continue

    name, membership = user

    question = input("Ask your question: ")

    docs = retriever.invoke(question)

    if len(docs) == 0:
        print(
            "I do not have enough information in the provided knowledge base to answer this."
        )
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        name=name,
        membership_tier=membership,
        context=context,
        question=question
    )

    try:

        response = llm.invoke(final_prompt)

        print("\n" + "=" * 60)
        print(response.content)
        print("=" * 60)

    except Exception as e:

        print("Groq API Error")
        print(e)

conn.close()