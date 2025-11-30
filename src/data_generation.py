"""
Data Generation Module

Provides utilities for generating synthetic documents and test data
for Context Windows experiments.

Classes:
    DocumentGenerator: Generate synthetic documents
    FactEmbedder: Embed facts at specific positions
    FactGenerator: Generate test facts

Example:
    >>> generator = DocumentGenerator(random_seed=42)
    >>> doc = generator.generate_document(num_words=200)
    >>> embedder = FactEmbedder()
    >>> doc_with_fact = embedder.embed_fact(doc, "The CEO is Alice.", position="middle")
"""

import random
from typing import List, Optional
from loguru import logger


class DocumentGenerator:
    """
    Generate synthetic documents with filler text.

    Creates realistic-looking documents with configurable length
    for controlled experiments.
    """

    # Filler sentences
    FILLER_SENTENCES = [
        "The company has been operating in the market for many years.",
        "Our products are designed with customer satisfaction in mind.",
        "The team consists of experienced professionals from various fields.",
        "We strive to maintain high standards in all our operations.",
        "Quality and innovation are at the core of our business strategy.",
        "The organization has expanded its presence across multiple regions.",
        "We continue to invest in research and development initiatives.",
        "Customer feedback is highly valued and regularly incorporated.",
        "The management team brings decades of industry experience.",
        "Sustainability and corporate responsibility guide our decisions.",
    ]

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize document generator.

        Args:
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        logger.info(f"DocumentGenerator initialized with seed: {random_seed}")

    def generate_document(
        self,
        num_words: int = 200,
        word_variance: int = 20
    ) -> str:
        """
        Generate a synthetic document.

        Args:
            num_words: Target number of words
            word_variance: Allowed variance in word count

        Returns:
            Generated document text
        """
        target_words = num_words + random.randint(-word_variance, word_variance)

        sentences = []
        current_words = 0

        while current_words < target_words:
            sentence = random.choice(self.FILLER_SENTENCES)
            sentences.append(sentence)
            current_words += len(sentence.split())

        document = " ".join(sentences)
        logger.debug(f"Generated document: {len(document.split())} words")

        return document

    def generate_documents(
        self,
        num_documents: int,
        num_words: int = 200,
        word_variance: int = 20
    ) -> List[str]:
        """
        Generate multiple documents.

        Args:
            num_documents: Number of documents to generate
            num_words: Target words per document
            word_variance: Allowed variance

        Returns:
            List of generated documents
        """
        documents = [
            self.generate_document(num_words, word_variance)
            for _ in range(num_documents)
        ]

        logger.info(f"Generated {num_documents} documents")
        return documents


class FactEmbedder:
    """
    Embed facts at specific positions in documents.
    """

    def __init__(self):
        """Initialize fact embedder."""
        logger.info("FactEmbedder initialized")

    def embed_fact(
        self,
        document: str,
        fact: str,
        position: str = "middle"
    ) -> str:
        """
        Embed a fact at specified position.

        Args:
            document: Base document text
            fact: Fact to embed
            position: 'start', 'middle', or 'end'

        Returns:
            Document with embedded fact

        Raises:
            ValueError: If position is invalid
        """
        if position not in ["start", "middle", "end"]:
            raise ValueError(f"Invalid position: {position}")

        # Split into sentences
        sentences = document.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        # Determine insertion position
        if position == "start":
            insert_idx = len(sentences) // 5  # 0-20%
        elif position == "middle":
            insert_idx = len(sentences) // 2  # 40-60%
        else:  # end
            insert_idx = int(len(sentences) * 0.8)  # 80-100%

        # Insert fact
        sentences.insert(insert_idx, fact)

        result = " ".join(sentences)
        logger.debug(f"Embedded fact at position '{position}'")

        return result


class FactGenerator:
    """
    Generate test facts for experiments.
    """

    FACT_TEMPLATES = {
        "ceo_name": "The CEO of the company is {name}.",
        "revenue": "The Q3 revenue was ${amount}M.",
        "location": "The headquarters is located in {city}.",
    }

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize fact generator.

        Args:
            random_seed: Optional random seed
        """
        if random_seed is not None:
            random.seed(random_seed)

        self.names = ["David Cohen", "Sarah Williams", "Michael Brown"]
        self.cities = ["New York", "London", "Tokyo"]

        logger.info("FactGenerator initialized")

    def generate_fact(self, fact_type: str = "ceo_name") -> str:
        """
        Generate a fact of specified type.

        Args:
            fact_type: Type of fact

        Returns:
            Generated fact string
        """
        if fact_type not in self.FACT_TEMPLATES:
            raise ValueError(f"Unknown fact type: {fact_type}")

        template = self.FACT_TEMPLATES[fact_type]

        if fact_type == "ceo_name":
            return template.format(name=random.choice(self.names))
        elif fact_type == "revenue":
            return template.format(amount=random.randint(10, 100))
        elif fact_type == "location":
            return template.format(city=random.choice(self.cities))
