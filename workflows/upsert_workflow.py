from src.document_loader import DocumentLoader
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStore

class RAGSystem:
    def __init__(self, file_directory: str):
        self.loader = DocumentLoader(file_directory)
        self.embeddings_manager = EmbeddingsManager()
        self.vector_store = VectorStore()

    def workflow(self):
        documents = self.loader.load_documents()
        embeddings = self.embeddings_manager.encode(documents)
        self.vector_store.update_collection(documents=documents, embeddings=embeddings)
        print(f"Loaded {len(documents)} documents into the vector store.")