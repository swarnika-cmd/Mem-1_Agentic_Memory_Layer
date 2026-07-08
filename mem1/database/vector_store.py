import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from mem1.database.base import BaseVectorStore

class SimpleVectorStore(BaseVectorStore):
    """Zero-dependency vector store using Bag-of-Words & Cosine Similarity in NumPy.
    
    Can save/load index from disk, and handles text tokenization internally.
    """

    def __init__(self, index_path: str = "mem1_vector_index.json"):
        self.index_path = index_path
        # node_id -> text
        self.texts: Dict[str, str] = {}
        # Vocabulary mapping for local vectorization
        self.vocab: Dict[str, int] = {}
        # node_id -> list of float weights (raw vector)
        self.vectors: Dict[str, List[float]] = {}
        
        self.load_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer to convert text to lowercased words."""
        # Remove common punctuation and split
        clean_text = text.lower()
        for char in ".,!?;:()[]{}'\"`":
            clean_text = clean_text.replace(char, " ")
        return [w for w in clean_text.split() if len(w) > 2]

    def _update_vocab_and_vectors(self) -> None:
        """Re-calculate BoW vectors for all stored texts based on unified vocabulary."""
        # 1. Rebuild vocabulary
        all_tokens = []
        for text in self.texts.values():
            all_tokens.extend(self._tokenize(text))
        
        unique_tokens = sorted(list(set(all_tokens)))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        
        # 2. Build term frequency vectors
        self.vectors = {}
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            return

        for node_id, text in self.texts.items():
            tokens = self._tokenize(text)
            vector = np.zeros(vocab_size)
            for t in tokens:
                if t in self.vocab:
                    vector[self.vocab[t]] += 1.0
            
            # L2 normalization to make cosine similarity a simple dot product
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            self.vectors[node_id] = vector.tolist()

    def add_vector(self, node_id: str, text: str, embedding: Optional[List[float]] = None) -> None:
        """Add a text entry and compile its vector representation."""
        self.texts[node_id] = text
        self._update_vocab_and_vectors()
        self.save_index()

    def remove_vector(self, node_id: str) -> None:
        """Remove a vector representation by node ID."""
        if node_id in self.texts:
            del self.texts[node_id]
        if node_id in self.vectors:
            del self.vectors[node_id]
        self._update_vocab_and_vectors()
        self.save_index()

    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Search similar items using cosine similarity of the local BoW representations."""
        query_tokens = self._tokenize(query)
        if not self.vocab or not query_tokens:
            return []

        # Build query vector using the same vocabulary
        vocab_size = len(self.vocab)
        query_vec = np.zeros(vocab_size)
        for t in query_tokens:
            if t in self.vocab:
                query_vec[self.vocab[t]] += 1.0
        
        # L2 Normalize
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        else:
            return []

        results = []
        for node_id, vec_list in self.vectors.items():
            vec = np.array(vec_list)
            # Since both vectors are L2-normalized, cosine similarity is the dot product
            sim = float(np.dot(query_vec, vec))
            if sim > 0:
                results.append((node_id, sim))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def save_index(self) -> None:
        """Save vector database index to disk."""
        # Ensure parent directories exist
        index_dir = os.path.dirname(self.index_path)
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir)
            
        data = {
            "texts": self.texts,
            "vocab": self.vocab,
            "vectors": self.vectors
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        """Load vector database index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.texts = data.get("texts", {})
                self.vocab = data.get("vocab", {})
                self.vectors = data.get("vectors", {})
            except Exception:
                # Fallback to empty if load fails
                self.texts = {}
                self.vocab = {}
                self.vectors = {}
