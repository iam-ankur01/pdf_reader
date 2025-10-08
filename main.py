import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline
import os

st.set_page_config(page_title="PDF QA WebApp", page_icon="📄")

st.title("📄 PDF Question Answering WebApp")
st.write("Upload a PDF and ask questions based on its content!")

# -------------------------------
# Step 1: Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file temporarily
    pdf_path = f"./temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded {uploaded_file.name}")

    # -------------------------------
    # Step 2: Load PDF & Split
    # -------------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.write(f"✅ Loaded {len(documents)} page(s) from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    st.write(f"✅ Split into {len(docs)} chunks.")

    # -------------------------------
    # Step 3: Create Embeddings & VectorStore
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./vectorstore"
    )
    st.success("✅ Embeddings created and stored successfully!")

    # -------------------------------
    # Step 4: Load LLM (Flan-T5)
    # -------------------------------
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # -------------------------------
    # Step 5: Setup Retrieval QA
    # -------------------------------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # -------------------------------
    # Step 6: Ask Questions
    # -------------------------------
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa.invoke({"query": query})
        st.markdown(f"**Answer:** {answer['result']}")

    # Clean up temporary PDF
    os.remove(pdf_path)
