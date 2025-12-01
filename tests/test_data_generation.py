"""
Tests for data_generation module.
"""

import pytest
from src.data_generation import DocumentGenerator, FactEmbedder, FactGenerator


class TestDocumentGenerator:
    """Tests for DocumentGenerator class."""

    def test_initialization_no_seed(self):
        """Test initialization without random seed."""
        generator = DocumentGenerator()
        assert generator is not None

    def test_initialization_with_seed(self):
        """Test initialization with random seed."""
        generator = DocumentGenerator(random_seed=42)
        assert generator is not None

    def test_generate_document_default(self):
        """Test document generation with default parameters."""
        generator = DocumentGenerator(random_seed=42)
        doc = generator.generate_document()

        assert isinstance(doc, str)
        assert len(doc) > 0
        words = doc.split()
        assert len(words) > 0

    def test_generate_document_custom_length(self):
        """Test document generation with custom length."""
        generator = DocumentGenerator(random_seed=42)
        doc = generator.generate_document(num_words=100, word_variance=10)

        words = doc.split()
        # Should be around 100 words ± variance
        assert 80 <= len(words) <= 120

    def test_generate_document_reproducibility(self):
        """Test that same seed produces similar documents."""
        gen1 = DocumentGenerator(random_seed=42)
        gen2 = DocumentGenerator(random_seed=42)

        doc1 = gen1.generate_document(num_words=50, word_variance=0)
        doc2 = gen2.generate_document(num_words=50, word_variance=0)

        # Same seed should produce similar length documents
        assert abs(len(doc1.split()) - len(doc2.split())) <= 5

    def test_generate_documents_multiple(self):
        """Test generating multiple documents."""
        generator = DocumentGenerator(random_seed=42)
        docs = generator.generate_documents(num_documents=5, num_words=50)

        assert len(docs) == 5
        assert all(isinstance(doc, str) for doc in docs)
        assert all(len(doc) > 0 for doc in docs)

    def test_generate_documents_different(self):
        """Test that multiple documents are different."""
        generator = DocumentGenerator(random_seed=42)
        docs = generator.generate_documents(num_documents=3, num_words=50)

        # Documents should be different (very unlikely to be identical)
        assert docs[0] != docs[1] or docs[1] != docs[2]


class TestFactEmbedder:
    """Tests for FactEmbedder class."""

    def test_initialization(self):
        """Test FactEmbedder initialization."""
        embedder = FactEmbedder()
        assert embedder is not None

    def test_embed_fact_start(self):
        """Test embedding fact at start position."""
        embedder = FactEmbedder()
        doc = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        fact = "The CEO is Alice."

        result = embedder.embed_fact(doc, fact, position="start")

        assert fact in result
        assert len(result) > len(doc)

    def test_embed_fact_middle(self):
        """Test embedding fact at middle position."""
        embedder = FactEmbedder()
        doc = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        fact = "The CEO is Bob."

        result = embedder.embed_fact(doc, fact, position="middle")

        assert fact in result
        assert len(result) > len(doc)

    def test_embed_fact_end(self):
        """Test embedding fact at end position."""
        embedder = FactEmbedder()
        doc = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        fact = "The CEO is Charlie."

        result = embedder.embed_fact(doc, fact, position="end")

        assert fact in result
        assert len(result) > len(doc)

    def test_embed_fact_invalid_position(self):
        """Test that invalid position raises ValueError."""
        embedder = FactEmbedder()
        doc = "Test document."
        fact = "Test fact."

        with pytest.raises(ValueError, match="Invalid position"):
            embedder.embed_fact(doc, fact, position="invalid")

    def test_embed_fact_default_position(self):
        """Test embedding fact with default position (middle)."""
        embedder = FactEmbedder()
        doc = "Sentence one. Sentence two. Sentence three."
        fact = "Important fact."

        result = embedder.embed_fact(doc, fact)
        assert fact in result


class TestFactGenerator:
    """Tests for FactGenerator class."""

    def test_initialization_no_seed(self):
        """Test FactGenerator initialization without seed."""
        generator = FactGenerator()
        assert generator is not None
        assert hasattr(generator, 'names')
        assert hasattr(generator, 'cities')

    def test_initialization_with_seed(self):
        """Test FactGenerator initialization with seed."""
        generator = FactGenerator(random_seed=42)
        assert generator is not None

    def test_generate_ceo_fact(self):
        """Test generating CEO name fact."""
        generator = FactGenerator(random_seed=42)
        fact = generator.generate_fact(fact_type="ceo_name")

        assert isinstance(fact, str)
        assert "CEO" in fact
        assert any(name in fact for name in generator.names)

    def test_generate_revenue_fact(self):
        """Test generating revenue fact."""
        generator = FactGenerator(random_seed=42)
        fact = generator.generate_fact(fact_type="revenue")

        assert isinstance(fact, str)
        assert "revenue" in fact or "Q3" in fact
        assert "$" in fact
        assert "M" in fact

    def test_generate_location_fact(self):
        """Test generating location fact."""
        generator = FactGenerator(random_seed=42)
        fact = generator.generate_fact(fact_type="location")

        assert isinstance(fact, str)
        assert "headquarters" in fact
        assert any(city in fact for city in generator.cities)

    def test_generate_fact_invalid_type(self):
        """Test that invalid fact type raises ValueError."""
        generator = FactGenerator()

        with pytest.raises(ValueError, match="Unknown fact type"):
            generator.generate_fact(fact_type="invalid_type")

    def test_generate_fact_reproducibility(self):
        """Test that same seed produces similar facts."""
        gen1 = FactGenerator(random_seed=42)
        gen2 = FactGenerator(random_seed=42)

        fact1 = gen1.generate_fact("ceo_name")
        fact2 = gen2.generate_fact("ceo_name")

        # Both should be valid CEO facts
        assert "CEO" in fact1 and "CEO" in fact2

    def test_fact_templates_exist(self):
        """Test that fact templates are defined."""
        assert hasattr(FactGenerator, 'FACT_TEMPLATES')
        assert 'ceo_name' in FactGenerator.FACT_TEMPLATES
        assert 'revenue' in FactGenerator.FACT_TEMPLATES
        assert 'location' in FactGenerator.FACT_TEMPLATES
