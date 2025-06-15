import streamlit as st
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.base import BaseLLM
import os

# --- Page Config ---
st.set_page_config(page_title="Ask Your CSV 📊", layout="wide")
st.title("💡 Ask Questions on Your CSV using GPT + RAG")

st.markdown("""
Upload a CSV file, ask questions in natural language, and get accurate, GPT-powered answers with Retrieval-Augmented Generation (RAG).

This app uses your uploaded data to generate context-specific responses. Powered by OpenAI and LlamaIndex.
""")

# --- Load OpenAI API Key securely ---
openai_api_key = st.secrets["openai_api_key"]  # Set this in Streamlit Cloud

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# --- Ask a Question ---
user_query = st.text_input("💬 Enter your question (e.g. 'What is the total revenue in Q1?')")

if uploaded_file and user_query:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)

        # Convert each row into a Document object
        docs = [Document(text=row.to_string()) for _, row in df.iterrows()]

        # Set up OpenAI LLM
        llm: BaseLLM = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        # Create index and query engine
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(llm=llm)

        # Run query
        response = query_engine.query(user_query)

        # Display result
        st.success(f"✅ Answer: {response}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
