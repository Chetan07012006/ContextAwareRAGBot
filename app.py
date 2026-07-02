from flask import Flask, render_template, request
import sqlite3
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

def get_membership(user_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT membership_tier FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

@app.route("/", methods=["GET","POST"])
def home():
    answer = ""

    if request.method == "POST":
        user_id = request.form["user_id"]
        question = request.form["question"]

        membership = get_membership(user_id)

        if membership:
            prompt = f"""
User Membership: {membership}

Question:
{question}
"""
            answer = qa.run(prompt)
        else:
            answer = "User ID not found."

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)