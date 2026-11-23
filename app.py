from flask import Flask, render_template, request
import sqlite3
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)

# Uses Hugging Face Inference API (uses ~0 MB RAM instead of loading PyTorch locally)
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

PROMPT_TEMPLATE = """You are an AI customer support assistant.

You are speaking with:
Name: {name}
Membership Tier: {membership_tier}

Answer the user's question using only the context provided below.

If the answer is not available in the context, say:
"I do not have enough information in the provided knowledge base to answer this."

Context:
{retrieved_chunks}

User Question:
{user_query}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["name", "membership_tier", "retrieved_chunks", "user_query"],
    template=PROMPT_TEMPLATE
)

def get_user_details(user_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT name, membership_tier FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row if row else None

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""

    if request.method == "POST":
        user_id = request.form.get("user_id", "").strip()
        user_query = request.form.get("question", "").strip()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            return render_template(
                "index.html",
                answer="Error: GROQ_API_KEY is missing. Please set GROQ_API_KEY in your environment variables."
            )

        user = get_user_details(user_id)
        if not user:
            return render_template(
                "index.html",
                answer="User not found. Please enter a valid user_id."
            )

        name, membership_tier = user

        # Retrieve relevant chunks with safe fallback to company_faq.txt
        retrieved_chunks = ""
        try:
            docs = retriever.invoke(user_query)
            if docs:
                retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            if os.path.exists("company_faq.txt"):
                with open("company_faq.txt", "r", encoding="utf-8") as f:
                    retrieved_chunks = f.read()[:4000]

        if not retrieved_chunks.strip():
            return render_template(
                "index.html",
                answer="I do not have enough information in the provided knowledge base to answer this."
            )

        formatted_prompt = prompt_template.format(
            name=name,
            membership_tier=membership_tier,
            retrieved_chunks=retrieved_chunks,
            user_query=user_query
        )

        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="openai/gpt-oss-120b"
            )
            response = llm.invoke(formatted_prompt)
            answer = response.content
        except Exception as e:
            answer = f"Service temporarily unavailable due to an API error or rate limit. Please try again shortly. ({str(e)})"

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
