from dataclasses import dataclass

@dataclass
class EmbeddingsConfig:
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

@dataclass
class LLMConfig:
    DEFAULT_MODEL = "gpt-4.1-mini"
    MAX_TOKENS = 4096

@dataclass
class VectorStoreConfig:
    TENANT = "d8058fce-0e9b-431f-9577-bb1a4de75a4f"
    DISTANCE_METRIC = "cosine"
    COLLECTION_NAME = "documents"
    VECTOR_SIZE = 384

@dataclass
class RetrieverConfig:
    TOP_K = 5
    SIMILARITY_METRIC = "cosine"

@dataclass
class RAGSystemConfig:
    N_RESULTS: int = 1
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7