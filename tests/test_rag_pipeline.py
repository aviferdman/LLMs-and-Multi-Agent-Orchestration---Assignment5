"""
Tests for RAG pipeline module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.rag_pipeline import RAGPipeline, VectorStore


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    def test_initialization_defaults(self):
        """Test RAGPipeline initialization with defaults."""
        pipeline = RAGPipeline()

        assert pipeline.chunk_size == 500
        assert pipeline.chunk_overlap == 50
        assert pipeline.top_k == 3
        assert pipeline.embedding_model == "nomic-embed-text"
        assert pipeline.chunks == []
        assert pipeline.embeddings == []

    def test_initialization_custom(self):
        """Test RAGPipeline initialization with custom parameters."""
        pipeline = RAGPipeline(
            chunk_size=300,
            chunk_overlap=30,
            top_k=5,
            embedding_model="custom-model"
        )

        assert pipeline.chunk_size == 300
        assert pipeline.chunk_overlap == 30
        assert pipeline.top_k == 5
        assert pipeline.embedding_model == "custom-model"

    def test_chunk_documents_single(self):
        """Test chunking a single document."""
        pipeline = RAGPipeline(chunk_size=10, chunk_overlap=2)
        documents = ["This is a test document with more than ten words to test chunking properly."]

        chunks = pipeline.chunk_documents(documents)

        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('doc_id' in chunk for chunk in chunks)
        assert all('chunk_id' in chunk for chunk in chunks)
        assert all(chunk['doc_id'] == 0 for chunk in chunks)

    def test_chunk_documents_multiple(self):
        """Test chunking multiple documents."""
        pipeline = RAGPipeline(chunk_size=10, chunk_overlap=2)
        documents = [
            "First document with some text.",
            "Second document with more text."
        ]

        chunks = pipeline.chunk_documents(documents)

        assert len(chunks) >= 2
        doc_ids = [chunk['doc_id'] for chunk in chunks]
        assert 0 in doc_ids
        assert 1 in doc_ids

    def test_chunk_documents_overlap(self):
        """Test that chunking produces overlapping chunks."""
        pipeline = RAGPipeline(chunk_size=5, chunk_overlap=2)
        documents = ["one two three four five six seven eight nine ten"]

        chunks = pipeline.chunk_documents(documents)

        # Should have overlapping chunks
        assert len(chunks) >= 2

    def test_get_embedding_fallback(self):
        """Test embedding generation with fallback method."""
        pipeline = RAGPipeline()
        text = "This is a test sentence."

        embedding = pipeline._get_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384
        assert np.sum(embedding) > 0  # Should not be all zeros

    def test_get_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        pipeline = RAGPipeline()

        emb1 = pipeline._get_embedding("This is text one.")
        emb2 = pipeline._get_embedding("Completely different text.")

        assert not np.array_equal(emb1, emb2)

    def test_get_embedding_normalization(self):
        """Test that embeddings are normalized."""
        pipeline = RAGPipeline()
        text = "Test normalization."

        embedding = pipeline._get_embedding(text)
        norm = np.linalg.norm(embedding)

        assert 0.99 <= norm <= 1.01  # Should be close to 1

    def test_index_documents(self):
        """Test indexing documents."""
        pipeline = RAGPipeline(chunk_size=5, chunk_overlap=0)
        # Create longer documents that will definitely create chunks
        documents = [
            " ".join(["word" + str(i) for i in range(20)]),
            " ".join(["test" + str(i) for i in range(20)])
        ]

        pipeline.index_documents(documents)

        assert len(pipeline.chunks) > 0
        assert len(pipeline.embeddings) > 0
        assert len(pipeline.chunks) == len(pipeline.embeddings)

    def test_retrieve_without_indexing(self):
        """Test retrieval without indexing returns empty."""
        pipeline = RAGPipeline()
        results = pipeline.retrieve("test query")

        assert results == []

    def test_retrieve_basic(self):
        """Test basic retrieval."""
        pipeline = RAGPipeline(chunk_size=10, top_k=2)
        documents = [
            "Information about artificial intelligence and machine learning.",
            "Details about natural language processing systems.",
            "Data about computer vision and image recognition."
        ]

        pipeline.index_documents(documents)
        results = pipeline.retrieve("machine learning")

        assert len(results) <= 2  # Should respect top_k
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)

    def test_retrieve_custom_k(self):
        """Test retrieval with custom k parameter."""
        pipeline = RAGPipeline(chunk_size=10, top_k=3)
        documents = ["Doc one.", "Doc two.", "Doc three.", "Doc four."]

        pipeline.index_documents(documents)
        results = pipeline.retrieve("query", k=2)

        assert len(results) <= 2

    def test_retrieve_scores_ordered(self):
        """Test that retrieval results are ordered by score."""
        pipeline = RAGPipeline(chunk_size=10, top_k=5)
        documents = ["Text about cats.", "Text about dogs.", "Text about birds."]

        pipeline.index_documents(documents)
        results = pipeline.retrieve("animals")

        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_without_llm(self):
        """Test query without LLM interface."""
        pipeline = RAGPipeline(chunk_size=10)
        documents = ["Sample document with information."]

        pipeline.index_documents(documents)
        result = pipeline.query("What is this about?")

        assert 'query' in result
        assert 'context' in result
        assert 'answer' in result
        assert result['answer'] == "[NO LLM PROVIDED]"
        assert 'retrieved_chunks' in result

    def test_query_with_mock_llm(self):
        """Test query with mocked LLM interface."""
        pipeline = RAGPipeline(chunk_size=10)
        documents = ["Document about testing."]
        pipeline.index_documents(documents)

        # Mock LLM interface
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "This is about testing."
        mock_llm.query.return_value = mock_response

        result = pipeline.query("What is this about?", llm_interface=mock_llm)

        assert result['answer'] == "This is about testing."
        assert mock_llm.query.called

    def test_query_llm_error_handling(self):
        """Test query handles LLM errors gracefully."""
        pipeline = RAGPipeline(chunk_size=10)
        documents = ["Test document."]
        pipeline.index_documents(documents)

        # Mock LLM that raises error
        mock_llm = Mock()
        mock_llm.query.side_effect = Exception("LLM error")

        result = pipeline.query("query", llm_interface=mock_llm)

        assert "[ERROR" in result['answer']

    def test_query_metadata(self):
        """Test that query returns proper metadata."""
        pipeline = RAGPipeline(chunk_size=10, top_k=2)
        documents = ["First doc.", "Second doc."]
        pipeline.index_documents(documents)

        result = pipeline.query("test")

        assert 'metadata' in result
        assert 'chunk_ids' in result['metadata']
        assert 'scores' in result['metadata']
        assert len(result['metadata']['chunk_ids']) == result['retrieved_chunks']


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_initialization_default(self):
        """Test VectorStore initialization with default name."""
        store = VectorStore()

        assert store.collection_name == "context_windows"
        assert store.documents == []
        assert store.embeddings == []
        assert store.metadata == []

    def test_initialization_custom_name(self):
        """Test VectorStore initialization with custom name."""
        store = VectorStore(collection_name="test_collection")

        assert store.collection_name == "test_collection"

    def test_add_documents_basic(self):
        """Test adding documents to vector store."""
        store = VectorStore()
        documents = ["doc1", "doc2"]
        embeddings = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        store.add_documents(documents, embeddings)

        assert len(store.documents) == 2
        assert len(store.embeddings) == 2
        assert len(store.metadata) == 2

    def test_add_documents_with_metadata(self):
        """Test adding documents with metadata."""
        store = VectorStore()
        documents = ["doc1"]
        embeddings = [np.array([1, 2, 3])]
        metadata = [{"source": "test"}]

        store.add_documents(documents, embeddings, metadata)

        assert store.metadata[0] == {"source": "test"}

    def test_add_documents_incremental(self):
        """Test adding documents incrementally."""
        store = VectorStore()

        # First batch
        store.add_documents(["doc1"], [np.array([1, 2, 3])])
        assert len(store.documents) == 1

        # Second batch
        store.add_documents(["doc2"], [np.array([4, 5, 6])])
        assert len(store.documents) == 2

    def test_query_empty_store(self):
        """Test querying empty vector store."""
        store = VectorStore()
        query_emb = np.array([1, 2, 3])

        results = store.query(query_emb, k=2)

        assert results == []

    def test_query_basic(self):
        """Test basic vector store query."""
        store = VectorStore()
        documents = ["doc1", "doc2", "doc3"]
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]

        store.add_documents(documents, embeddings)

        # Query similar to first document
        query_emb = np.array([0.9, 0.1, 0.0])
        query_emb = query_emb / np.linalg.norm(query_emb)
        results = store.query(query_emb, k=2)

        assert len(results) <= 2
        assert all('document' in r for r in results)
        assert all('score' in r for r in results)
        assert all('index' in r for r in results)

    def test_query_respects_k(self):
        """Test that query respects k parameter."""
        store = VectorStore()
        documents = ["doc1", "doc2", "doc3", "doc4"]
        embeddings = [np.random.rand(10) for _ in range(4)]

        store.add_documents(documents, embeddings)

        query_emb = np.random.rand(10)
        results = store.query(query_emb, k=2)

        assert len(results) == 2

    def test_query_scores_ordered(self):
        """Test that query results are ordered by score."""
        store = VectorStore()
        documents = ["doc1", "doc2", "doc3"]
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0])
        ]

        store.add_documents(documents, embeddings)

        query_emb = np.array([1.0, 0.0])
        results = store.query(query_emb, k=3)

        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_returns_metadata(self):
        """Test that query returns metadata."""
        store = VectorStore()
        documents = ["doc1"]
        embeddings = [np.array([1.0, 0.0])]
        metadata = [{"test": "value"}]

        store.add_documents(documents, embeddings, metadata)

        query_emb = np.array([1.0, 0.0])
        results = store.query(query_emb, k=1)

        assert results[0]['metadata'] == {"test": "value"}


class TestRAGPipelineEdgeCases:
    """Tests for RAG pipeline edge cases and error handling."""

    def test_get_embedding_with_ollama_error(self):
        """Test embedding generation with Ollama error fallback."""
        pipeline = RAGPipeline()

        with patch('src.rag_pipeline.OLLAMA_AVAILABLE', True):
            with patch('src.rag_pipeline.ollama.embeddings', side_effect=Exception("API Error")):
                # Should fall back to simple embedding
                embedding = pipeline._get_embedding("test text")
                assert isinstance(embedding, np.ndarray)
                assert len(embedding) == 384

    def test_get_embedding_no_ollama(self):
        """Test embedding when Ollama is not available."""
        pipeline = RAGPipeline()

        with patch('src.rag_pipeline.OLLAMA_AVAILABLE', False):
            embedding = pipeline._get_embedding("test text")
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 384

    def test_retrieve_with_scores(self):
        """Test that retrieve returns proper scores."""
        pipeline = RAGPipeline(chunk_size=5, top_k=2)
        documents = [" ".join([f"word{i}" for i in range(10)])]

        pipeline.index_documents(documents)
        results = pipeline.retrieve("word5 word6", k=2)

        # Check that scores are present and valid
        for result in results:
            assert 'score' in result
            assert 0 <= result['score'] <= 1

    def test_query_context_building(self):
        """Test that query properly builds context from chunks."""
        pipeline = RAGPipeline(chunk_size=3, top_k=2)
        documents = ["one two three four five six"]

        pipeline.index_documents(documents)
        result = pipeline.query("test query")

        assert 'context' in result
        assert 'query' in result
        assert result['query'] == "test query"
        assert isinstance(result['context'], str)

    def test_chunk_documents_empty_list(self):
        """Test chunking with empty document list."""
        pipeline = RAGPipeline()
        chunks = pipeline.chunk_documents([])

        assert chunks == []

    def test_chunk_documents_metadata(self):
        """Test that chunks contain proper metadata."""
        pipeline = RAGPipeline(chunk_size=5, chunk_overlap=1)
        documents = ["one two three four five six seven eight"]

        chunks = pipeline.chunk_documents(documents)

        for chunk in chunks:
            assert 'doc_id' in chunk
            assert 'chunk_id' in chunk
            assert 'start_pos' in chunk
            assert 'end_pos' in chunk
            assert 'text' in chunk
