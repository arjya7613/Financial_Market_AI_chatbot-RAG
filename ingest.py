# 1. Import Library

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def build_index():

        # 2. A) Load PDF from Directory & create document
        loader = DirectoryLoader('./data', glob='**/*.pdf', loader_cls=PyPDFLoader)
        pdf_docs = loader.load()

        # 2. B) Load CSV from Directory & create document
        csv_loader = CSVLoader(file_path="./data/Apple_Dataset.csv", encoding="utf-8")
        csv_docs = csv_loader.load()

        documents = csv_docs + pdf_docs

        for doc in documents:
                source = doc.metadata.get("source", "unknown")
                if source.endswith(".pdf"):
                        doc.metadata["document_type"] = "financial_report"
                elif source.endswith(".csv"):
                        doc.metadata["document_type"] = "structured_financial_dataset"
                else:
                        doc.metadata["document_type"] = "unknown"
                
        # 3. RecursiveCharacter Text Splitter
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100,
                                                            separators=["\n\n", "\n", " ", "", ".",",", ";"])

        recursive_tokens = recursive_splitter.split_documents(documents)

        # 4. Create embeddings using HFEmbeddings
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 5. Create Vector Store
        faiss_store = FAISS.from_documents(documents = recursive_tokens, embedding=hf_embeddings)

        # persist the vector store
        faiss_store.save_local("faiss_index")

        print("FAISS faiss_index created successfully!")

if __name__ == "__main__":
        build_index()

