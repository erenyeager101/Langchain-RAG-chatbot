import argparse
import logging
import sys

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
CHROMA_PATH = "chroma"
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 3

# Prompt template for question-answering
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def initialize_vector_db():
    """
    Initializes the Chroma vector DB with OpenAI embeddings.
    """
    try:
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        return db
    except Exception as e:
        logging.error(f"Failed to load Chroma DB: {e}")
        sys.exit(1)

def search_relevant_documents(db, query_text):
    """
    Perform similarity search for a given query.
    """
    try:
        results = db.similarity_search_with_relevance_scores(query_text, k=TOP_K_RESULTS)
        return results
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        sys.exit(1)

def generate_prompt(context_text, query_text):
    """
    Fills the prompt template with context and user question.
    """
    try:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return prompt_template.format(context=context_text, question=query_text)
    except Exception as e:
        logging.error(f"Failed to generate prompt: {e}")
        sys.exit(1)

def main():
    # === Step 1: Parse CLI arguments ===
    parser = argparse.ArgumentParser(description="RAG-based question answering from Chroma DB")
    parser.add_argument("query_text", type=str, help="The query you want to ask.")
    args = parser.parse_args()
    query_text = args.query_text

    # === Step 2: Load Vector DB ===
    db = initialize_vector_db()

    # === Step 3: Similarity Search ===
    logging.info("Searching for relevant documents...")
    results = search_relevant_documents(db, query_text)

    if not results:
        logging.warning("No results found.")
        print("❌ Unable to find matching results.")
        return

    # Filter by similarity score threshold
    filtered_results = [(doc, score) for doc, score in results if score >= SIMILARITY_THRESHOLD]
    if not filtered_results:
        logging.warning("No results above similarity threshold.")
        print("❌ Found results, but none were relevant enough.")
        return

    # === Step 4: Prepare Prompt ===
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])
    prompt = generate_prompt(context_text, query_text)
    logging.debug(f"Prompt:\n{prompt}")

    # === Step 5: Call LLM for Answer ===
    try:
        model = ChatOpenAI()
        response_text = model.predict(prompt)
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        print("❌ Failed to get a response from the model.")
        return

    # === Step 6: Display Results ===
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in filtered_results]
    formatted_response = f"\n✅ Response:\n{response_text}\n\n📚 Sources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
