import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from torch import embedding


load_dotenv()
working_dir = os.path.dirname(os.path.abspath((__file__))) # it is for the current file path

embeddings = HuggingFaceEmbeddings()
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0
)
def process_document_to_chroma_db(file_name):
    # Load the PDF document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    # Store the document chunks in a Chroma vector database
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

#to retreview from the vector database and answer the user question
def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embeddings
    )
    # Create a retriever for document search
    retriever = vectordb.as_retriever()

    # Create a RetrievalQA chain to answer user questions using Llama-3.3-70B
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                                                                                                                                                                              
        chain_type="stuff",   # stuff meands all chunks combined as one prompt                                                  
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer