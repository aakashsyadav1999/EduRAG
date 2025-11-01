import chromadb
import os
from utils.config_file import VectorStoreConfig
from typing import List, Dict, Union
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class VectorStore:
    def __init__(self):
        # Collection name is separate from database connection
        self.collection_name = getattr(VectorStoreConfig, 'COLLECTION_NAME', 'documents')  # Default collection name
        self.api_key = os.getenv("CHROMADB_API_KEY")
        self.tenant = getattr(VectorStoreConfig, 'TENANT', 'd8058fce-0e9b-431f-9577-bb1a4de75a4f')
        self.database = "data-gathering"
        self.client = None
        self.collection = None
        self.max_chunk_size = 15000

    def chunk_text(self, text: str, max_size: int | None = None) -> List[str]:
        """Split text into chunks that fit within ChromaDB's size limits."""
        if max_size is None:
            max_size = self.max_chunk_size
        
        words = text.split()
        chunks: List[str] = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word.encode('utf-8')) + 1
            
            if current_size + word_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def create_client(self):
        """Create or recreate the ChromaDB client and validate access."""
        try:
            if not self.api_key:
                raise ValueError("ChromaDB API key not set.")
            
            self.client = chromadb.CloudClient(
                api_key=self.api_key,
                tenant=self.tenant,
                database=self.database
            )
            
            try:
                collections = self.client.list_collections()
                print(f"Connected to ChromaDB. Available collections: {[c.name for c in collections]}")
            except Exception as e:
                raise PermissionError(f"ChromaDB authentication error: {e}")
            
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"Using existing collection: '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(name=self.collection_name)
                print(f"Created new collection: '{self.collection_name}'")
            
            return self.client
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")

    def update_collection(self, documents: List[str], embeddings: np.ndarray | None = None):
        """Update collection with documents, chunking large documents."""
        if self.client is None:
            self.create_client()
        
        try:
            all_chunks: List[str] = []
            all_metadatas: List[Dict[str, Union[str, int]]] = []
            all_ids: List[str] = []

            
            for i, doc in enumerate(documents):
                # Check document size
                doc_size = len(doc.encode('utf-8'))
                print(f"Document {i} size: {doc_size} bytes")
                
                if doc_size > self.max_chunk_size:
                    # Chunk large documents
                    chunks = self.chunk_text(doc)
                    print(f"Document {i} chunked into {len(chunks)} pieces")
                    
                    for j, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "source": "upload",
                            "document_id": i,
                            "chunk_id": j,
                            "total_chunks": len(chunks),
                            "original_size": doc_size
                        })
                        all_ids.append(f"doc_{i}_chunk_{j}")
                else:
                    all_chunks.append(doc)
                    all_metadatas.append({
                        "source": "upload",
                        "document_id": i,
                        "chunk_id": 0,
                        "total_chunks": 1,
                        "original_size": doc_size
                    })
                    all_ids.append(f"doc_{i}_chunk_0")
            
            # Add chunks in batches to avoid overwhelming the API
            batch_size = 10
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i+batch_size]
                batch_metadatas = all_metadatas[i:i+batch_size]
                batch_ids = all_ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"Added batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
            
            print(f"Successfully added {len(all_chunks)} chunks from {len(documents)} documents")
            
        except Exception as e:
            if "Quota exceeded" in str(e):
                print(f"ChromaDB quota exceeded. Consider:")
                print("1. Using smaller documents")
                print("2. Upgrading to a paid plan")
                print("3. Using a local vector store like FAISS")
                raise ConnectionError(f"ChromaDB quota exceeded: {e}")
            raise ConnectionError(f"Failed to update ChromaDB collection: {e}")

    def search(self, query: str, n_results: int = 5):
        """Search for similar documents in the collection."""
        if self.client is None:
            self.create_client()
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            raise ConnectionError(f"Failed to search in ChromaDB: {e}")

    def get_collection_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the current collection."""
        if self.client is None or self.collection is None:
            self.create_client()
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "database": self.database,
                "tenant": self.tenant
            }
        except Exception as e:
            raise ConnectionError(f"Failed to get collection info: {e}")

    def delete_collection(self):
        """Delete the current collection."""
        if self.client is None:
            self.create_client()
        
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully from database '{self.database}'")
        except Exception as e:
            raise ConnectionError(f"Failed to delete collection: {e}")

    def list_collections(self):
        """List all collections in the database."""
        if self.client is None:
            self.create_client()
        
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            raise ConnectionError(f"Failed to list collections: {e}")
