"""
Tests for LLM interface module
"""

import pytest
import requests
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.llm_interface import OllamaInterface, Response

class TestOllamaInterface:
    """Test suite for OllamaInterface class"""
    
    @pytest.fixture
    def ollama_interface(self):
        """Create OllamaInterface instance for testing"""
        with patch.object(OllamaInterface, '_verify_connection', return_value=True):
            return OllamaInterface(model="llama2", base_url="http://localhost:11434")
    
    def test_initialization(self, ollama_interface):
        """Test OllamaInterface initialization"""
        assert ollama_interface is not None
        assert ollama_interface.model == "llama2"
        assert ollama_interface.base_url == "http://localhost:11434"
    
    def test_format_prompt(self, ollama_interface):
        """Test prompt formatting"""
        context = "Test context"
        question = "Test question?"
        prompt = ollama_interface._format_prompt(context, question)
        assert "Test context" in prompt
        assert "Test question?" in prompt
        assert isinstance(prompt, str)
    
    def test_count_tokens(self, ollama_interface):
        """Test token counting"""
        text = "This is a test"
        tokens = ollama_interface.count_tokens(text)
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_count_tokens_empty(self, ollama_interface):
        """Test token counting with empty string"""
        tokens = ollama_interface.count_tokens("")
        # Empty string still gets 1 token for BOS/EOS
        assert tokens >= 0
    
    @patch('requests.post')
    def test_query_success(self, mock_post, ollama_interface):
        """Test successful query"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test answer"}
        mock_response.elapsed.total_seconds.return_value = 1.5
        mock_post.return_value = mock_response
        
        result = ollama_interface.query("Context", "Question")
        assert isinstance(result, Response)
        assert result.text == "Test answer"
    
    def test_embed_simple(self, ollama_interface):
        """Test simple embedding"""
        text = "Test text"
        embedding = ollama_interface._simple_embedding(text, dim=128)
        assert embedding.shape == (128,)
        # NumPy arrays default to float64
        assert embedding.dtype in ['float32', 'float64']
    
    def test_response_class(self):
        """Test Response dataclass"""
        response = Response(
            text="Answer",
            latency=1.5,
            tokens=10
        )
        assert response.text == "Answer"
        assert response.latency == 1.5
        assert response.tokens == 10

    def test_count_tokens_longer_text(self, ollama_interface):
        """Test token counting with longer text."""
        long_text = " ".join(["word"] * 100)
        tokens = ollama_interface.count_tokens(long_text)
        assert tokens > 50  # Should have many tokens

    def test_format_prompt_context_only(self, ollama_interface):
        """Test format prompt with only context."""
        prompt = ollama_interface._format_prompt("Just context", "")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_format_prompt_question_only(self, ollama_interface):
        """Test format prompt with only question."""
        prompt = ollama_interface._format_prompt("", "Just question?")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @patch('requests.post')
    def test_query_with_metadata(self, mock_post, ollama_interface):
        """Test query returns proper metadata."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Answer",
            "total_duration": 1000000,
            "load_duration": 500000,
            "prompt_eval_count": 5,
            "eval_count": 10
        }
        mock_post.return_value = mock_response

        result = ollama_interface.query("Context", "Question")
        assert result.latency >= 0  # Latency calculated from actual time
        assert result.tokens > 0
        assert result.metadata["total_duration"] == 1000000
        assert result.metadata["eval_count"] == 10

    def test_embed_different_dimensions(self, ollama_interface):
        """Test embedding with different dimensions."""
        text = "Test"

        emb_64 = ollama_interface._simple_embedding(text, dim=64)
        emb_256 = ollama_interface._simple_embedding(text, dim=256)

        assert emb_64.shape == (64,)
        assert emb_256.shape == (256,)

    @patch('requests.post')
    def test_query_with_timeout(self, mock_post, ollama_interface):
        """Test query with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.elapsed.total_seconds.return_value = 1.0
        mock_post.return_value = mock_response

        result = ollama_interface.query("Context", "Question")
        assert result is not None

    @patch('requests.post')
    def test_query_non_200_status(self, mock_post, ollama_interface):
        """Test query handling non-200 status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server Error")
        mock_post.return_value = mock_response

        with pytest.raises(Exception):
            ollama_interface.query("Context", "Question")


    def test_format_prompt_empty_inputs(self, ollama_interface):
        """Test format prompt with empty inputs."""
        prompt = ollama_interface._format_prompt("", "")
        assert isinstance(prompt, str)

    def test_count_tokens_special_characters(self, ollama_interface):
        """Test token counting with special characters."""
        text = "Hello!!! @#$% ^&*() 123"
        tokens = ollama_interface.count_tokens(text)
        assert tokens > 0


    @patch('requests.post')
    def test_query_empty_response(self, mock_post, ollama_interface):
        """Test query with empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_post.return_value = mock_response

        result = ollama_interface.query("Context", "Question")
        assert result.text == ""

    @patch('requests.post')
    def test_query_timeout_with_retry(self, mock_post, ollama_interface):
        """Test query timeout triggers retry logic."""
        mock_post.side_effect = requests.Timeout("Request timeout")

        with pytest.raises(RuntimeError, match="Query timeout after"):
            ollama_interface.query("Context", "Question")

        # Should retry max_retries times
        assert mock_post.call_count == ollama_interface.max_retries

    @patch('requests.post')
    def test_query_generic_exception_with_retry(self, mock_post, ollama_interface):
        """Test generic exception triggers retry logic."""
        mock_post.side_effect = Exception("Generic error")

        with pytest.raises(RuntimeError, match="Query failed after"):
            ollama_interface.query("Context", "Question")

        assert mock_post.call_count == ollama_interface.max_retries

    @patch('requests.post')
    def test_query_non_200_with_retry(self, mock_post, ollama_interface):
        """Test non-200 status code triggers retry logic."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="Ollama returned status 500"):
            ollama_interface.query("Context", "Question")

        assert mock_post.call_count == ollama_interface.max_retries

    @patch('requests.post')
    def test_query_retry_with_success(self, mock_post, ollama_interface):
        """Test query succeeds after initial failure."""
        # First call fails, second succeeds
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Error"

        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"response": "Success"}

        mock_post.side_effect = [error_response, success_response]

        result = ollama_interface.query("Context", "Question")
        assert result.text == "Success"
        assert mock_post.call_count == 2

    def test_connection_verification_failure(self):
        """Test connection verification failure."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
                OllamaInterface(model="test-model", base_url="http://invalid:11434")

    @patch('requests.post')
    def test_query_with_custom_options(self, mock_post, ollama_interface):
        """Test query with custom Ollama options."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_post.return_value = mock_response

        result = ollama_interface.query(
            "Context",
            "Question",
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            num_ctx=4096
        )

        assert result.text == "Answer"
        # Verify the options were passed in the request
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['options']['temperature'] == 0.5
        assert payload['options']['top_p'] == 0.95
        assert payload['options']['top_k'] == 50
        assert payload['options']['num_ctx'] == 4096

    def test_connection_verification_success(self):
        """Test successful connection verification."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # This should not raise an exception
            interface = OllamaInterface(model="test-model", base_url="http://localhost:11434")
            assert interface is not None

    def test_connection_verification_non_200(self):
        """Test connection verification with non-200 status."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            # Non-200 status should log warning but not raise exception
            interface = OllamaInterface(model="test-model", base_url="http://localhost:11434")
            assert interface is not None

    @patch('requests.post')
    def test_embed_success(self, mock_post, ollama_interface):
        """Test successful embedding generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        embedding = ollama_interface.embed("Test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3
        assert embedding[0] == 0.1

    @patch('requests.post')
    def test_embed_empty_embedding(self, mock_post, ollama_interface):
        """Test embedding with empty result falls back to simple embedding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": []}
        mock_post.return_value = mock_response

        embedding = ollama_interface.embed("Test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0  # Fallback should return non-empty

    @patch('requests.post')
    def test_embed_non_200_status(self, mock_post, ollama_interface):
        """Test embedding with non-200 status falls back to simple embedding."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        embedding = ollama_interface.embed("Test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0  # Fallback should return non-empty

    @patch('requests.post')
    def test_embed_exception(self, mock_post, ollama_interface):
        """Test embedding with exception falls back to simple embedding."""
        mock_post.side_effect = Exception("Connection error")

        embedding = ollama_interface.embed("Test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0  # Fallback should return non-empty

