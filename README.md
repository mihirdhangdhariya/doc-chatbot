
# AI Document Chatbot

A Streamlit-powered chatbot that lets you upload documents (PDF), ask questions, and get accurate answers using Cohere's large language models. The app also logs queries and shows analytical insights using charts and tables.

---

## Features

- Upload and index multiple PDF documents
- Ask natural language questions and receive context-aware responses
- Logs queries and responses for future analysis
- Displays top 5 most frequently asked queries in a data table and bar chart
- Uses Cohere API for response generation
- Vector-based document search with `sentence-transformers`
- Intuitive UI with Streamlit

---

## Setup Instructions

### 1. Clone the Repository
bash
git clone https://github.com/your-username/ai-doc-chatbot.git
cd ai-doc-chatbot


### 2. Install Dependencies
bash
pip install -r requirements.txt


requirements.txt

streamlit
pandas
matplotlib
plotly
sentence-transformers
cohere
PyMuPDF


### 3. Set Up API Key
Create a `.env` file in the root directory and add your Cohere API key:

COHERE_API_KEY=your-cohere-api-key


Or, if deploying on Streamlit Cloud, add it in the Secrets Manager.

### 4. Run the App Locally
bash
streamlit run app.py


---

## Architecture Overview

text
        ┌────────────┐
        │  User UI   │
        └─────┬──────┘
              │
              ▼
      ┌─────────────────┐
      │ Upload Document │◄──── PDF
      └────────┬────────┘
               │
               ▼
      ┌────────────────────┐
      │   PDF Text Parser  │
      └────────┬───────────┘
               │
        ┌──────▼─────┐
        │ Vectorizer │  ← SentenceTransformer
        └──────┬─────┘
               │
        ┌──────▼────────┐
        │ Vector Search │ ← Cosine Similarity
        └──────┬────────┘
               │
        ┌──────▼───────┐
        │  Cohere API  │ ← Generates response
        └──────┬───────┘
               │
      ┌────────▼───────────┐
      │ Response + Logging │ → CSV
      └────────┬───────────┘
               │
        ┌──────▼─────────┐
        │ Analytics View │ ← Top 5 queries, pie + bar chart
        └────────────────┘


---

## API Usage Guidelines

### Using Cohere’s Chat API

In `app.py`, the chatbot sends a user query and a document-based context to Cohere's chat model:
python
response = co.chat(
    model="command-r",
    message=user_query,
    documents=[{"text": context}]
)


> Note: `command-nightly` and `command-r` require the `chat()` method, not `generate()`.

### Getting Document Embeddings

We use `sentence-transformers` to create semantic embeddings from document text:
python
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(doc_chunks)


### Logging Queries and Responses

Each user query and response is saved to `query_logs/query_log.csv`:
python
with open("query_logs/query_log.csv", "a", encoding="utf-8") as f:
    f.write(f"{query}|{response}\n")


---

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Deploy via [streamlit.io/cloud](https://streamlit.io/cloud)
3. Add your Cohere API key in the Secrets Manager
4. Done!

---

## Folder Structure


├── app.py
├── requirements.txt
├── query_logs/
│   └── query_log.csv
├── documents/
│   └── uploaded PDFs
└── README.md


---

## Future Enhancements

- Add authentication for different users
- Use FAISS or other vector databases for scalable search
- Add support for other document formats (DOCX, TXT)
- Export analytics as PDF/Excel
