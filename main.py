import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


st.set_page_config(
    page_title="PDF QA App",
    page_icon="📄"
)

st.title("📄 PDF Question Answering App")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully!")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever()

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    prompt = PromptTemplate(
        template="""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    query = st.text_input(
        "Ask your question"
    )

    if query:

        with st.spinner("Generating Answer..."):

            response = retrieval_chain.invoke({
                "input": query
            })

        st.subheader("Answer")

        st.write(response["answer"])

    os.remove(pdf_path)    st.success("PDF Uploaded Successfully!")

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    st.write(f"Loaded {len(documents)} pages")

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    st.write(f"Created {len(docs)} chunks")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever()

    st.success("Vector Database Ready!")

    # LLM
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    # Prompt
    prompt = PromptTemplate(
        template="""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """,
        input_variables=["context", "input"]
    )

    # Create Chains
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # User Question
    query = st.text_input(
        "Ask a question about the PDF"
    )

    if query:

        with st.spinner("Generating Answer..."):

            response = retrieval_chain.invoke({
                "input": query
            })

        st.subheader("Answer")

        st.write(response["answer"])

    # Cleanup
    os.remove(pdf_path)
