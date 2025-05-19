import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import cohere
import fitz  # PyMuPDF for PDF parsing

# Initialize paths and Cohere
DOCUMENTS_DIR = "documents"
QUERY_LOG_FILE = "query_logs/query_log.csv"
COHERE_API_KEY = "LUGPhUpYNaKb9h3RyeeGGZr1h4Lm63Rnubr8chJl"  # Replace with your actual API key

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(QUERY_LOG_FILE), exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
co = cohere.Client(COHERE_API_KEY)

# Utility Functions
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def load_documents():
    texts = []
    filenames = []
    for fname in os.listdir(DOCUMENTS_DIR):
        if fname.endswith(".pdf"):
            fpath = os.path.join(DOCUMENTS_DIR, fname)
            text = extract_text_from_pdf(fpath)
            texts.append(text)
            filenames.append(fname)
    return texts, filenames

def retrieve_relevant_context(query, docs, embeddings):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_idx = scores.argmax().item()
    return docs[top_idx]

def get_response(query, context):
    response = co.chat(
        message=query,
        documents=[{"text": context}],
        model="command-r-plus",
        temperature=0.4
    )
    return response.text

def log_query(query, response):
    with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{query}|{response}\n")

def get_top_queries():
    if not os.path.exists(QUERY_LOG_FILE):
        return pd.DataFrame(columns=["query", "count"])
    
    df = pd.read_csv(QUERY_LOG_FILE, sep="|", names=["query", "response"])
    top_queries = df["query"].value_counts().head(5).reset_index()
    top_queries.columns = ["query", "count"]
    return top_queries

# Streamlit UI
st.set_page_config("AI Doc Assistant", layout="wide")
st.title("AI Document Assistant")
st.markdown("Upload PDFs, ask questions, and analyze the most asked queries.")

# Upload Section
st.sidebar.header("Upload PDF")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        save_uploaded_file(file)
    st.sidebar.success("Files uploaded and saved.")

# Ask a Question Section
st.subheader("Ask a Question")
user_query = st.text_input("Type your question here")

if st.button("Get Answer") and user_query:
    with st.spinner("Processing..."):
        docs, filenames = load_documents()
        if not docs:
            st.warning("No documents found. Please upload PDFs.")
        else:
            doc_embeddings = model.encode(docs, convert_to_tensor=True)
            context = retrieve_relevant_context(user_query, docs, doc_embeddings)
            response = get_response(user_query, context)
            log_query(user_query, response)
            st.markdown("**Answer:**")
            st.success(response)

# Query Analytics Section
st.subheader("Query Analytics")
top_queries_df = get_top_queries()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top 5 Asked Queries (Table)**")
    st.dataframe(top_queries_df)

with col2:
    if not top_queries_df.empty:
        fig = px.bar(top_queries_df, x='query', y='count', title="Top 5 Queries - Bar Chart")
        st.plotly_chart(fig, use_container_width=True)

# Optional Pie Chart
if not top_queries_df.empty:
    st.markdown("**Pie Chart of Top Queries**")
    fig_pie = px.pie(top_queries_df, values='count', names='query', title='Top 5 Queries Distribution')
    st.plotly_chart(fig_pie)
