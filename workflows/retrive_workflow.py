import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from src.vector_store import VectorStore
from src.embeddings_manager import EmbeddingsManager
from utils.config_file import RAGSystemConfig, LLMConfig

load_dotenv(find_dotenv())

class RAGRetriever:
    def __init__(self):
        """Initialize the RAG retriever with vector store and OpenAI client."""
        self.vector_store = VectorStore()
        self.embeddings_manager = EmbeddingsManager()
        self.n_results = RAGSystemConfig.N_RESULTS
        self.max_tokens = RAGSystemConfig.MAX_TOKENS
        self.temperature = RAGSystemConfig.TEMPERATURE
        self.model_name = LLMConfig.DEFAULT_MODEL
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB based on the query."""
        try:
            # Search for similar documents
            results = self.vector_store.search(query, n_results=n_results)
            
            # Extract and format the results
            retrieved_docs = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]  # First batch of results
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, doc in enumerate(documents):
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'distance': distances[i] if i < len(distances) else 0.0,
                        'relevance_score': 1 / (1 + distances[i]) if i < len(distances) else 0.0
                    })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate a prompt for OpenAI using the query and retrieved documents."""
        if not retrieved_docs:
            return f"""
            I don't have any relevant information in my knowledge base to answer your question: "{query}"
            
            Please provide a helpful response based on your general knowledge, but mention that this information is not from the specific documents in the system.
            """
        
        # Format retrieved documents
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"Document {i}:\n{doc['content']}\n\n"
        
        prompt = f"""
        You are a helpful AI assistant. Use the following documents to answer the user's question. 
        If the answer cannot be found in the provided documents, say so clearly.
        
        Context Documents:
        {context}
        
        User Question: {query}
        
        Instructions:
        - Answer based primarily on the provided documents
        - If information is missing, acknowledge it
        - Be concise but comprehensive
        - Cite which document(s) you're referencing when possible
        
        Answer:
        """
        
        return prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate a response using OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response.
        """
        print(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant documents
        print("Retrieving relevant documents...")
        retrieved_docs = self.retrieve_documents(query, self.n_results)
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Generate prompt
        prompt = self.generate_prompt(query, retrieved_docs)
        
        # Step 3: Generate response using OpenAI
        print("Generating response with OpenAI...")
        response = self.generate_response(prompt, self.max_tokens, self.temperature)
        
        # Return comprehensive result
        return {
            "query": query,
            "response": response,
            "retrieved_documents": retrieved_docs,
            "num_retrieved": len(retrieved_docs),
            "model_used": self.model_name  # This should work now
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the retrieval system."""
        try:
            collection_info = self.vector_store.get_collection_info()
            embeddings_info = self.embeddings_manager.get_model_info()
            
            return {
                "vector_store": collection_info,
                "embeddings": embeddings_info,
                "openai_model": self.model_name,
                "status": "ready"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize retriever
        retriever = RAGRetriever()
        
        # Get system info
        info = retriever.get_system_info()
        print("System Info:", info)
        
        # Example query
        query = "What is the Dynamic Decoding?"
        result = retriever.ask(query)
        
        print(f"\nQuery: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Documents retrieved: {result['num_retrieved']}")
        print(f"Model used: {result['model_used']}")
        
        # Show retrieved documents
        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"\nDocument {i} (Score: {doc['relevance_score']:.3f}):")
            print(f"Content: {doc['content'][:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")