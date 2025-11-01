import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict
import numpy as np
from utils.config_file import EmbeddingsConfig

@dataclass
class EmbeddingsManager:
    # Use the best free model for ChromaDB
    model_name: str = field(default=getattr(EmbeddingsConfig, 'DEFAULT_MODEL', 'all-MiniLM-L6-v2'))
    data: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self):
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def set_data(self, data: List[str]):
        """Store data for embedding and downstream tasks."""
        self.data = data

    def encode(self, texts: Union[str, List[str], None] = None) -> np.ndarray:
        """Encode stored data or provided texts into embeddings."""
        if texts is None:
            if self.data is None:
                raise ValueError("No data provided or set in the manager.")
            texts = self.data
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, convert_to_tensor=False)
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to encode texts: {e}")

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Calculate similarity between embeddings."""
        try:
            similarity_result = self.model.similarity(emb1, emb2)
            if hasattr(similarity_result, 'cpu'):
                return similarity_result.cpu().numpy()
            return np.array(similarity_result)
        except Exception as e:
            raise ValueError(f"Failed to calculate similarity: {e}")

    def calculate_similarities_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate similarity matrix for a list of texts."""
        try:
            embeddings = self.encode(texts)
            return self.similarity(embeddings, embeddings)
        except Exception as e:
            raise ValueError(f"Failed to calculate similarities matrix: {e}")

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the current model."""
        max_seq_length = getattr(self.model, 'max_seq_length', None)
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": max_seq_length if max_seq_length is not None else 'Unknown'
        }