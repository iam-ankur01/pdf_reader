import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

st.set_page_config(page_title="PDF QA WebApp", page_icon="📄")

st.title("📄 PDF Question Answering WebApp")
st.write("Upload a PDF and ask questions based on its content!")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:

    # Save PDF temporarily
    pdf_path = f"temp_{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.write(f"Loaded {len(documents)} pages")

    # Split text
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

    # Vector DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    st.success("Embeddings created successfully!")

    # LLM
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Prompt
    prompt = PromptTemplate(
        template="""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """,
        input_variables=["context", "input"]
    )

    # Chains
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # User Query
    query = st.text_input("Ask a question about the PDF")

    if query:

        with st.spinner("Generating answer..."):

            response = retrieval_chain.invoke({
                "input": query
            })

        st.markdown("### Answer")
        st.write(response["answer"])

    # Cleanup
    os.remove(pdf_path)
