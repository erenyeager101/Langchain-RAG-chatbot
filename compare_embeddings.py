from langchain_openai import OpenAIEmbeddings  
from langchain.evaluation import load_evaluator  
from dotenv import load_dotenv  
import openai  
import os  

# Load environment variables from the .env file
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def main():
    # === Step 1: Generate Embedding for a Word ===
    embedding_function = OpenAIEmbeddings()  # Initialize the embedding generator
    word = "apple"
    
   
    try:
        vector = embedding_function.embed_query(word)
        print(f"Vector for '{word}': {vector}")
        print(f"Vector length: {len(vector)}")  # Should typically be 1536 for `text-embedding-ada-002`
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return

    # === Step 2: Compare Two Word Embeddings ===
    evaluator = load_evaluator("pairwise_embedding_distance")  # Loads evaluator for pairwise distance
    word1 = "apple"
    word2 = "iphone"

    try:
        # Evaluate the distance between two word embeddings
        result = evaluator.evaluate_string_pairs(prediction=word1, prediction_b=word2)
        print(f"Similarity comparison between '{word1}' and '{word2}': {result}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
