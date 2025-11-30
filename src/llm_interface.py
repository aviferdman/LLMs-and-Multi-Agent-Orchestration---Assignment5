"""
LLM Interface Module

Provides abstraction layer for interacting with different LLM providers.
Supports Ollama, OpenAI, and other providers through a common interface.

Classes:
    LLMInterface: Abstract base class for LLM providers
    OllamaInterface: Ollama implementation
    Response: Structured response object

Example:
    >>> llm = OllamaInterface(model="llama2:13b")
    >>> response = llm.query(context="Document text", query="What is the main topic?")
    >>> print(response.text)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

from loguru import logger


@dataclass
class Response:
    """
    Structured response from LLM query.
    
    Attributes:
        text: The generated text response
        latency: Time taken to generate response (seconds)
        tokens: Number of tokens in response
        confidence: Model confidence score (if available)
        metadata: Additional provider-specific metadata
    """
    text: str
    latency: float
    tokens: int
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMInterface(ABC):
    """
    Abstract base class for LLM provider interfaces.
    
    All LLM providers must implement this interface to ensure
    consistent behavior across experiments.
    """
    
    @abstractmethod
    def query(self, context: str, query: str, **kwargs) -> Response:
        """
        Query LLM with context and question.
        
        Args:
            context: Contextual information to provide
            query: The question or prompt
            **kwargs: Provider-specific options
            
        Returns:
            Response object with generated text and metadata
        """
        pass
    
    @abstractmethod
    def embed(self, text: str) -> Any:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (typically numpy array)
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass


class OllamaInterface(LLMInterface):
    """
    Ollama LLM interface implementation.
    
    Provides access to locally-hosted Ollama models.
    
    Attributes:
        model: Name of the Ollama model
        base_url: Ollama server URL
        timeout: Request timeout in seconds
    
    Example:
        >>> llm = OllamaInterface(model="llama2:13b")
        >>> response = llm.query("Context here", "What is X?")
    """
    
    def __init__(
        self, 
        model: str = "llama2:13b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        """
        Initialize Ollama interface.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        
        logger.info(f"Initialized OllamaInterface with model: {model}")
    
    def query(self, context: str, query: str, **kwargs) -> Response:
        """
        Query Ollama model.
        
        Args:
            context: Context to provide
            query: Question to ask
            **kwargs: Additional Ollama options
            
        Returns:
            Response object
        """
        # TODO: Implement Ollama API call
        # This is a placeholder implementation
        
        start_time = time.time()
        
        # Combine context and query
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # TODO: Replace with actual Ollama API call
        # response = ollama.generate(model=self.model, prompt=full_prompt)
        
        latency = time.time() - start_time
        
        # Placeholder response
        return Response(
            text="[PLACEHOLDER] This will be implemented with actual Ollama API call",
            latency=latency,
            tokens=0,
            metadata={"model": self.model}
        )
    
    def embed(self, text: str) -> Any:
        """
        Generate embedding using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # TODO: Implement Ollama embedding
        logger.warning("Embedding not yet implemented")
        return None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4
