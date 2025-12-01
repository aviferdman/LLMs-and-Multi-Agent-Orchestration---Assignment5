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
from typing import Any, Dict, Optional, List
import time
import requests
import numpy as np

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
    
    Provides access to locally-hosted Ollama models with automatic retry,
    error handling, and comprehensive metrics tracking.
    
    Attributes:
        model: Name of the Ollama model
        base_url: Ollama server URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Example:
        >>> llm = OllamaInterface(model="llama2:latest")
        >>> response = llm.query("Context here", "What is X?")
        >>> print(f"Response: {response.text}")
        >>> print(f"Latency: {response.latency:.2f}s")
    """
    
    def __init__(
        self, 
        model: str = "llama2:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize Ollama interface.
        
        Args:
            model: Ollama model name (e.g., "llama2:latest", "llama3:latest")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            retry_delay: Seconds to wait between retries
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # API endpoints
        self.generate_url = f"{base_url}/api/generate"
        self.embed_url = f"{base_url}/api/embeddings"
        
        logger.info(f"Initialized OllamaInterface with model: {model}")
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """
        Verify Ollama server is accessible.
        
        Returns:
            True if connection successful
            
        Raises:
            ConnectionError: If server is not accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama server connection verified")
                return True
            else:
                logger.warning(f"⚠️ Ollama server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama server: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
    
    def query(self, context: str, query: str, **kwargs) -> Response:
        """
        Query Ollama model with automatic retry on failure.
        
        Args:
            context: Context to provide to the model
            query: Question to ask
            **kwargs: Additional Ollama options:
                - temperature: Randomness (0.0-1.0, default 0.1)
                - top_p: Nucleus sampling (0.0-1.0, default 0.9)
                - top_k: Top-k sampling (default 40)
                - num_ctx: Context window size (default 2048)
            
        Returns:
            Response object with text, latency, tokens, and metadata
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Combine context and query into prompt
        full_prompt = self._format_prompt(context, query)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "num_ctx": kwargs.get("num_ctx", 2048),
            }
        }
        
        # Attempt query with retry logic
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response text
                    text = result.get("response", "").strip()
                    
                    # Calculate tokens (approximate from timing data)
                    tokens = result.get("eval_count", self.count_tokens(text))
                    
                    # Log success
                    logger.debug(
                        f"Query successful: {len(text)} chars, "
                        f"{tokens} tokens, {latency:.2f}s"
                    )
                    
                    return Response(
                        text=text,
                        latency=latency,
                        tokens=tokens,
                        metadata={
                            "model": self.model,
                            "total_duration": result.get("total_duration"),
                            "load_duration": result.get("load_duration"),
                            "prompt_eval_count": result.get("prompt_eval_count"),
                            "eval_count": result.get("eval_count"),
                        }
                    )
                else:
                    error_msg = f"Ollama returned status {response.status_code}: {response.text}"
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise RuntimeError(error_msg)
                        
            except requests.Timeout:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: Request timeout")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Query timeout after {self.max_retries} attempts")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries}: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Query failed after {self.max_retries} attempts: {e}")
        
        raise RuntimeError("Unexpected error: retry loop completed without returning")
    
    def _format_prompt(self, context: str, query: str) -> str:
        """
        Format context and query into a prompt.
        
        Args:
            context: Context information
            query: Question to ask
            
        Returns:
            Formatted prompt string
        """
        if context and context.strip():
            return (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer based strictly on the context provided:"
            )
        else:
            return query
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding vector using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            RuntimeError: If embedding fails
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(
                self.embed_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                
                if embedding:
                    return np.array(embedding)
                else:
                    logger.warning("Empty embedding returned, using fallback")
                    # Fallback: simple word-based embedding
                    return self._simple_embedding(text)
            else:
                logger.warning(f"Embedding failed (status {response.status_code}), using fallback")
                return self._simple_embedding(text)
                
        except Exception as e:
            logger.error(f"Embedding error: {e}, using fallback")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Create simple word-based embedding as fallback.
        
        Args:
            text: Text to embed
            dim: Embedding dimension
            
        Returns:
            Simple embedding vector
        """
        # Simple hash-based embedding for fallback
        words = text.lower().split()
        embedding = np.zeros(dim)
        
        for i, word in enumerate(words[:dim]):
            # Simple hash to embedding dimension
            idx = hash(word) % dim
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def count_tokens(self, text: str) -> int:
        """
        Count approximate tokens in text.
        
        Uses heuristic: ~4 characters per token for English text.
        This is an approximation; actual tokenization may differ.
        
        Args:
            text: Text to count
            
        Returns:
            Approximate token count
        """
        # Improved approximation considering whitespace and punctuation
        # Average ~4 chars per token for English
        char_count = len(text)
        word_count = len(text.split())
        
        # Use word count if available, otherwise char-based estimate
        if word_count > 0:
            # Average English word is ~5 chars, tokens ~1.3x words
            return int(word_count * 1.3)
        else:
            return max(1, char_count // 4)
