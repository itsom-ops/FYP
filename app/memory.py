"""
Perspecta - Long-Horizon Tiered Memory Module
Combines Fast Tier (Redis / Dict fallback) with Vector Tier (ChromaDB persistent vector store).
Measures exact retrieval latency in milliseconds.
"""

import time
import os
from typing import Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

try:
    import redis
    _HAS_REDIS_LIB = True
except ImportError:
    _HAS_REDIS_LIB = False


class FastMemoryStore:
    """
    Fast tier key-value memory store using Redis if reachable, falling back to Python dictionary cache.
    """
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.backend = "redis"
        self._dict_store: Dict[str, str] = {}
        self.redis_client = None

        if _HAS_REDIS_LIB:
            try:
                r = redis.Redis(host=host, port=port, socket_timeout=0.5)
                r.ping()
                self.redis_client = r
                self.backend = "redis"
            except Exception:
                self.redis_client = None
                self.backend = "in_memory_dict"
        else:
            self.backend = "in_memory_dict"

    def set(self, key: str, value: str) -> float:
        """Stores key-value pair and returns latency in ms."""
        start = time.perf_counter()
        if self.redis_client:
            self.redis_client.set(key, value)
        else:
            self._dict_store[key] = value
        end = time.perf_counter()
        return (end - start) * 1000.0

    def get(self, key: str) -> Tuple[Optional[str], float]:
        """Retrieves value by key and returns (value, latency_ms)."""
        start = time.perf_counter()
        val = None
        if self.redis_client:
            raw = self.redis_client.get(key)
            if raw:
                val = raw.decode("utf-8")
        else:
            val = self._dict_store.get(key)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        return val, latency_ms


class VectorMemoryStore:
    """
    ChromaDB persistent vector database for semantic memory retrieval across long horizons.
    """
    def __init__(self, db_dir: str = "./data/chroma_db"):
        os.makedirs(db_dir, exist_ok=True)
        self.db_dir = db_dir
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="long_horizon_constraints",
            metadata={"description": "Perspecta system constraints vector store"}
        )

    def add_constraint(self, constraint_id: str, text: str, metadata: Dict[str, Any]) -> float:
        """Adds a constraint document into Chroma and returns insertion duration in ms."""
        start = time.perf_counter()
        self.collection.upsert(
            ids=[constraint_id],
            documents=[text],
            metadatas=[metadata]
        )
        end = time.perf_counter()
        return (end - start) * 1000.0

    def search_similar(self, query: str, n_results: int = 1) -> Tuple[list, float]:
        """Queries Chroma vector index by text similarity and returns (results, latency_ms)."""
        start = time.perf_counter()
        res = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0] if "distances" in res else []

        formatted = []
        for i in range(len(documents)):
            formatted.append({
                "document": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 0.0
            })
        return formatted, latency_ms


class TieredMemoryManager:
    """
    Orchestrates both Fast KV Memory and Vector Memory tiers.
    """
    def __init__(self):
        self.fast_memory = FastMemoryStore()
        self.vector_memory = VectorMemoryStore()

    def store(self, key: str, text: str, tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tags = tags or {}
        fast_ms = self.fast_memory.set(key, text)
        chroma_ms = self.vector_memory.add_constraint(key, text, tags)
        return {
            "key": key,
            "fast_backend": self.fast_memory.backend,
            "fast_write_ms": round(fast_ms, 3),
            "chroma_write_ms": round(chroma_ms, 3)
        }

    def retrieve(self, query_text: str, exact_key: Optional[str] = None) -> Dict[str, Any]:
        val, fast_ms = self.fast_memory.get(exact_key or query_text)
        vector_results, chroma_ms = self.vector_memory.search_similar(query_text, n_results=1)

        return {
            "fast_backend": self.fast_memory.backend,
            "fast_val": val,
            "fast_latency_ms": round(fast_ms, 3),
            "chroma_backend": "ChromaDB (Persistent)",
            "vector_results": vector_results,
            "chroma_latency_ms": round(chroma_ms, 3)
        }
