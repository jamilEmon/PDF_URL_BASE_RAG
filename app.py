import streamlit as st
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# Streamlit UI Setup
st.set_page_config(page_title="PDF & URL RAG Chatbot", layout="wide")
st.title(" PDF & URL RAG Chatbot")
st.write("Upload a PDF or enter a URL, ask a question, and get answers based on the content.")

# Sidebar Inputs
st.sidebar.header("Upload or Enter Source")
uploaded_file = st.sidebar.file_uploader("Upload PDF file:", type=["pdf"])
url_input = st.sidebar.text_input("Or enter URL:")

user_question = st.sidebar.text_area("Ask a question:")

# Text Extraction Functions
@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Extract visible text
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator="\n")

@st.cache_data
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

@st.cache_resource
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore


# LLM Setup
@st.cache_resource
def get_qa_chain():
    # Using a local HuggingFace pipeline
    pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# Main Logic
if st.sidebar.button("Get Answer") and user_question:
    source_text = ""
    if uploaded_file:
        source_text = extract_text_from_pdf(uploaded_file)
    elif url_input:
        source_text = extract_text_from_url(url_input)
    else:
        st.warning("Please upload a PDF or enter a URL.")
    
    if source_text:
        chunks = get_text_chunks(source_text)
        vectorstore = get_vectorstore(chunks)
        docs = vectorstore.similarity_search(user_question, k=3)
        chain = get_qa_chain()
        response = chain.run(input_documents=docs, question=user_question)
        
        st.subheader("Your Question:")
        st.write(user_question)
        st.subheader("Answer:")
        st.write(response)
