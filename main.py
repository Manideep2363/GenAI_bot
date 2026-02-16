import os
import streamlit as st

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
file_path = "faiss_store_hf"

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

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask your question...")

if query and os.path.exists(file_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        file_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Retrieve relevant docs
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build conversation history text
    history_text = ""
    for chat in st.session_state.chat_history:
        history_text += f"{chat['role']}: {chat['content']}\n"

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful news research assistant.

        Conversation History:
        {history}

        Context:
        {context}

        User Question:
        {question}

        Answer clearly and concisely.
        """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "history": history_text,
        "context": context,
        "question": query
    })

    # Store conversation
    st.session_state.chat_history.append({"role": "User", "content": query})
    st.session_state.chat_history.append({"role": "Assistant", "content": response})

    # Display conversation
for chat in st.session_state.chat_history:
    if chat["role"] == "User":
        with st.chat_message("user"):
            st.write(chat["content"])
    else:
        with st.chat_message("assistant"):
            st.write(chat["content"])