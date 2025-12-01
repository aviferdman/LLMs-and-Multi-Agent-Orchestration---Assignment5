"""
RAG Pipeline Module

Implements Retrieval-Augmented Generation pipeline using vector databases
for Experiment 3: RAG Impact Analysis.

Classes:
    RAGPipeline: Main RAG implementation
    VectorStore: Simple in-memory vector database

Example:
    >>> pipeline = RAGPipeline(chunk_size=500, top_k=3)
    >>> pipeline.index_documents(documents)
    >>> results = pipeline.retrieve("What is the main topic?")
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available, using fallback similarity")

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Manages document chunking, embedding, indexing, and retrieval
    for RAG-based question answering.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize RAG pipeline.

        Args:
            chunk_size: Size of document chunks (words)
            chunk_overlap: Overlap between chunks (words)
            top_k: Number of chunks to retrieve
            embedding_model: Model for generating embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = []

        logger.info(f"RAGPipeline initialized: chunk_size={chunk_size}, top_k={top_k}, model={embedding_model}")

    def chunk_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of document texts

        Returns:
            List of chunks with metadata
        """
        chunks = []

        for doc_id, doc in enumerate(documents):
            # Word-based chunking
            words = doc.split()

            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": len(chunks),
                    "start_pos": i,
                    "end_pos": i + len(chunk_words)
                })

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if not OLLAMA_AVAILABLE:
            # Fallback: simple word-based hash embedding
            words = text.lower().split()
            embedding = np.zeros(384)  # Standard embedding dimension
            for word in words[:100]:  # Limit to first 100 words
                hash_val = hash(word) % 384
                embedding[hash_val] += 1
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

        try:
            # Use Ollama's embedding API
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return np.array(response['embedding'])
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}, using fallback")
            # Fallback
            words = text.lower().split()
            embedding = np.zeros(384)
            for word in words[:100]:
                hash_val = hash(word) % 384
                embedding[hash_val] += 1
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

    def index_documents(self, documents: List[str]):
        """
        Index documents in vector store.

        Args:
            documents: List of document texts
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Create chunks
        self.chunks = self.chunk_documents(documents)
        
        # Generate embeddings for all chunks
        self.embeddings = []
        for i, chunk in enumerate(self.chunks):
            if i % 10 == 0:
                logger.debug(f"Generating embeddings: {i}/{len(self.chunks)}")
            embedding = self._get_embedding(chunk['text'])
            self.embeddings.append(embedding)
        
        self.embeddings = np.array(self.embeddings)
        logger.info(f"Indexed {len(documents)} documents into {len(self.chunks)} chunks")

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for query using cosine similarity.

        Args:
            query: Search query
            k: Number of chunks to retrieve (uses self.top_k if None)

        Returns:
            List of retrieved chunks with relevance scores
        """
        k = k if k is not None else self.top_k

        if not self.chunks or len(self.embeddings) == 0:
            logger.warning("No documents indexed")
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Calculate cosine similarities
        similarities = []
        for chunk_embedding in self.embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-10
            )
            similarities.append(similarity)

        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Return top k chunks with scores
        retrieved = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(similarities[idx])
            retrieved.append(chunk)

        scores_str = ", ".join([f"{c['score']:.3f}" for c in retrieved])
        logger.debug(f"Retrieved {len(retrieved)} chunks for query (scores: [{scores_str}])")
        return retrieved

    def query(self, query: str, llm_interface=None) -> Dict[str, Any]:
        """
        Perform RAG query.

        Args:
            query: Question to answer
            llm_interface: LLM interface for generation

        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query)

        # Combine into context
        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

        # Generate answer if LLM interface provided
        answer = "[NO LLM PROVIDED]"
        if llm_interface:
            try:
                response = llm_interface.query(context=context, query=query)
                answer = response.text
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                answer = f"[ERROR: {e}]"

        result = {
            "query": query,
            "context": context,
            "answer": answer,
            "retrieved_chunks": len(retrieved_chunks),
            "metadata": {
                "chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
                "scores": [c.get("score", 0.0) for c in retrieved_chunks]
            }
        }

        return result


class VectorStore:
    """
    Simple in-memory vector store.

    Provides interface for vector similarity search.
    """

    def __init__(self, collection_name: str = "context_windows"):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection
        """
        self.collection_name = collection_name
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        logger.info(f"VectorStore initialized: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents to vector store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
        """
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        logger.info(f"Added {len(documents)} documents to vector store (total: {len(self.documents)})")

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query vector store for similar documents.

        Args:
            query_embedding: Embedding vector for query
            k: Number of results to return

        Returns:
            List of similar documents with distances
        """
        if not self.embeddings:
            return []

        # Calculate cosine similarities
        similarities = []
        for emb in self.embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10
            )
            similarities.append(similarity)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(similarities[idx]),
                "index": int(idx)
            })

        return results
