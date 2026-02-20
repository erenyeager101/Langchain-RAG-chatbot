import os
import shutil
import pickle
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.graph import GraphRAG

load_dotenv()

CHROMA_PATH = "chroma_db"
BM25_PATH = "bm25_retriever.pkl"
GRAPH_PATH = "graph_data.pkl"

class IngestionPipeline:
    def __init__(self):
        # We assume OPENAI_API_KEY is in env
        self.embeddings = OpenAIEmbeddings()
        # Semantic Chunking is better than recursive character splitter
        self.text_splitter = SemanticChunker(self.embeddings)
        self.graph_rag = GraphRAG()

        # Load existing graph if available
        if os.path.exists(GRAPH_PATH):
            try:
                self.graph_rag.load(GRAPH_PATH)
            except Exception as e:
                print(f"Failed to load graph: {e}")

    def load_file(self, file_path: str) -> List[Document]:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                return []
            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def ingest(self, file_paths: List[str]):
        all_new_docs = []
        for path in file_paths:
            print(f"Loading {path}...")
            docs = self.load_file(path)
            all_new_docs.extend(docs)

        if not all_new_docs:
            print("No documents to ingest.")
            return

        print("Splitting documents...")
        chunks = self.text_splitter.split_documents(all_new_docs)
        print(f"Created {len(chunks)} chunks.")

        # 1. Store in Vector DB (Chroma)
        print("Updating Vector Store...")
        if os.path.exists(CHROMA_PATH):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
            db.add_documents(chunks)
        else:
            db = Chroma.from_documents(chunks, self.embeddings, persist_directory=CHROMA_PATH)

        # 2. Update BM25 Retriever
        # Fetch all docs from Chroma to rebuild BM25 index correctly
        print("Rebuilding BM25 Retriever...")
        try:
            # Getting all documents from Chroma can be tricky with large datasets
            # For this MVP, getting all is acceptable.
            # db.get() returns a dict with 'ids', 'embeddings', 'documents', 'metadatas'
            collection_data = db.get()
            all_texts = collection_data['documents']
            all_metadatas = collection_data['metadatas']

            reconstructed_docs = []
            if all_texts:
                for t, m in zip(all_texts, all_metadatas):
                    reconstructed_docs.append(Document(page_content=t, metadata=m))
            else:
                reconstructed_docs = chunks # Fallback if empty db query result

            if reconstructed_docs:
                bm25_retriever = BM25Retriever.from_documents(reconstructed_docs)
                with open(BM25_PATH, 'wb') as f:
                    pickle.dump(bm25_retriever, f)
            else:
                print("Warning: No documents for BM25.")

        except Exception as e:
            print(f"Error updating BM25: {e}")
            # Fallback to just current chunks
            bm25_retriever = BM25Retriever.from_documents(chunks)
            with open(BM25_PATH, 'wb') as f:
                pickle.dump(bm25_retriever, f)

        # 3. Update Graph
        print("Updating Knowledge Graph...")
        for chunk in chunks:
            self.graph_rag.add_document(chunk.page_content)

        self.graph_rag.save(GRAPH_PATH)
        print("Ingestion Complete.")

if __name__ == "__main__":
    pass
