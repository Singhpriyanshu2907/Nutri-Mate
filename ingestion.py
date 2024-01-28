import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

DATA = 'data/'
DB_PATH = 'vectorstore/db_faiss'

# Create vector database
def vector_db():
    loader = DirectoryLoader(DATA,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=250)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})


    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_PATH)

if __name__ == "__main__":
        vector_db()