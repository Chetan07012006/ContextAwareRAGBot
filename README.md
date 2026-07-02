# Context-Aware Customer Support RAG Bot

## Tech Stack
- Python 3.11
- LangChain
- Groq API
- ChromaDB
- SQLite
- HuggingFace Embeddings

## Setup

1. Clone the repository

2. Install dependencies

pip install -r requirements.txt

3. Add your API key

Create a .env file

GROQ_API_KEY=your_api_key

4. Run

python create_db.py

python ingest.py

python app.py

## Sample Queries

101
What is the refund policy?

103
Do I get premium customer support?

102
Can I cancel my account?

999
What are my benefits?
