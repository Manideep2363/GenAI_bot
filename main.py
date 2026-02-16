import os
import streamlit as st
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.title("GenAI-Bot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini"

main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

if process_url_clicked and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading Data...✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(file_path)

    main_placeholder.text("Processing Complete ✅")

query = st.text_input("Ask a Question:")

if query and os.path.exists(file_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        file_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context provided below.
        Use the context to provide a comprehensive and accurate answer.
        If the context contains relevant information, use it to answer the question.
        If the context doesn't contain enough information to fully answer the question, 
        provide the best answer you can based on the available context and indicate what information might be missing.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Modern LCEL Chain
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)

    st.header("Answer")
    st.write(response)