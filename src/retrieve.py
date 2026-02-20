import os
import pickle
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from src.graph import GraphRAG
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
BM25_PATH = "bm25_retriever.pkl"
GRAPH_PATH = "graph_data.pkl"

class HybridRetriever:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

        # Load Vector Store
        if os.path.exists(CHROMA_PATH):
            self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
            self.vector_store = None

        # Load BM25 Retriever
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
        else:
            self.bm25_retriever = None

        # Load GraphRAG
        self.graph_rag = GraphRAG()
        if os.path.exists(GRAPH_PATH):
            try:
                self.graph_rag.load(GRAPH_PATH)
            except Exception as e:
                print(f"Failed to load graph: {e}")

        # Reranker
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranker: {e}")
            self.reranker = None

    def get_relevant_documents(self, query: str, k=5) -> List[Document]:
        results = []

        # 1. Vector Search
        if self.vector_store:
            vector_results = self.vector_store.similarity_search(query, k=k)
            for doc in vector_results:
                doc.metadata['source_type'] = 'vector'
            results.extend(vector_results)

        # 2. Keyword Search (BM25)
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(query)
            # Take top k from BM25
            bm25_results = bm25_results[:k]
            for doc in bm25_results:
                doc.metadata['source_type'] = 'keyword'
            results.extend(bm25_results)

        # 3. Graph Search
        try:
            graph_context = self.graph_rag.query(query)
            if graph_context:
                graph_doc = Document(page_content=graph_context, metadata={"source": "knowledge_graph", "source_type": "graph"})
                results.append(graph_doc)
        except Exception as e:
            print(f"Graph search error: {e}")

        # Deduplicate by content
        unique_results = {}
        for doc in results:
            if doc.page_content not in unique_results:
                unique_results[doc.page_content] = doc

        final_results = list(unique_results.values())

        # 4. Rerank
        if self.reranker and final_results:
            pairs = [[query, doc.page_content] for doc in final_results]
            scores = self.reranker.predict(pairs)

            # Attach scores to docs for debugging/display
            for doc, score in zip(final_results, scores):
                doc.metadata['relevance_score'] = float(score)

            # Sort by score descending
            final_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)

        return final_results[:k]

if __name__ == "__main__":
    pass
