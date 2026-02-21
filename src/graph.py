import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Tuple
import pickle
import os

# Define the structure for triplets
class Triplet(BaseModel):
    subject: str = Field(description="The subject of the relationship")
    predicate: str = Field(description="The relationship or action")
    object: str = Field(description="The object of the relationship")

class TripletsList(BaseModel):
    triplets: List[Triplet]

class GraphRAG:
    def __init__(self, model_name="gpt-4o-mini"):
        self.graph = nx.Graph()
        self.llm = ChatOpenAI(temperature=0, model=model_name)

        # Define extraction chain
        structured_llm = self.llm.with_structured_output(TripletsList)
        system_prompt = (
            "You are a knowledge graph extractor. "
            "Extract subject-predicate-object triplets from the text. "
            "Keep entities concise and consistent."
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{text}")]
        )
        self.extraction_chain = prompt | structured_llm

    def extract_triplets(self, text: str) -> List[Triplet]:
        try:
            result = self.extraction_chain.invoke({"text": text})
            return result.triplets
        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return []

    def add_document(self, content: str):
        triplets = self.extract_triplets(content)
        for triplet in triplets:
            self.graph.add_edge(triplet.subject, triplet.object, relation=triplet.predicate)

    def query(self, query_text: str, depth=1) -> str:
        """
        Extracts entities from the query and retrieves neighbors from the graph.
        """
        # Extract entities from query to find start nodes
        # We use the same extraction logic but for the query
        query_triplets = self.extract_triplets(query_text)
        start_nodes = set()
        for t in query_triplets:
            start_nodes.add(t.subject)
            start_nodes.add(t.object)

        # Also try to match words in query to nodes directly (simple keyword match)
        for node in self.graph.nodes():
            if node.lower() in query_text.lower():
                start_nodes.add(node)

        related_info = []
        for node in start_nodes:
            if self.graph.has_node(node):
                # Get neighbors
                try:
                    # In networkx, ego_graph returns the subgraph of neighbors within radius
                    subgraph = nx.ego_graph(self.graph, node, radius=depth)
                    for u, v, data in subgraph.edges(data=True):
                        relation = data.get('relation', 'related to')
                        related_info.append(f"{u} {relation} {v}")
                except Exception as e:
                    print(f"Error traversing graph for node {node}: {e}")
                    continue

        return "\n".join(list(set(related_info)))

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
