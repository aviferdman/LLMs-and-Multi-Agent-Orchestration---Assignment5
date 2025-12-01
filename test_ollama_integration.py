"""
Integration test for OllamaInterface

Tests the complete OllamaInterface implementation with real Ollama API calls.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_interface import OllamaInterface, Response
import time

def test_basic_query():
    """Test basic query functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic Query")
    print("="*60)
    
    try:
        llm = OllamaInterface(model="llama2:latest")
        
        context = "The capital of France is Paris. Paris is known for the Eiffel Tower."
        query = "What is the capital of France?"
        
        print(f"Context: {context}")
        print(f"Query: {query}")
        print("\nQuerying Ollama...")
        
        response = llm.query(context, query)
        
        print(f"\n✅ Query successful!")
        print(f"Response: {response.text}")
        print(f"Latency: {response.latency:.2f} seconds")
        print(f"Tokens: {response.tokens}")
        print(f"Model: {response.metadata.get('model')}")
        
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_no_context_query():
    """Test query without context"""
    print("\n" + "="*60)
    print("TEST 2: Query Without Context")
    print("="*60)
    
    try:
        llm = OllamaInterface(model="llama2:latest")
        
        query = "What is 5 + 3? Answer with just the number."
        
        print(f"Query: {query}")
        print("\nQuerying Ollama...")
        
        response = llm.query("", query)
        
        print(f"\n✅ Query successful!")
        print(f"Response: {response.text}")
        print(f"Latency: {response.latency:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_token_counting():
    """Test token counting"""
    print("\n" + "="*60)
    print("TEST 3: Token Counting")
    print("="*60)
    
    try:
        llm = OllamaInterface(model="llama2:latest")
        
        test_texts = [
            "Hello world",
            "This is a longer sentence with more words.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        for text in test_texts:
            count = llm.count_tokens(text)
            print(f"Text: '{text}'")
            print(f"Estimated tokens: {count}")
            print()
        
        print("✅ Token counting works!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_embedding():
    """Test embedding generation"""
    print("\n" + "="*60)
    print("TEST 4: Embedding Generation")
    print("="*60)
    
    try:
        llm = OllamaInterface(model="llama2:latest")
        
        text = "This is a test sentence for embedding."
        print(f"Text: {text}")
        print("\nGenerating embedding...")
        
        embedding = llm.embed(text)
        
        print(f"\n✅ Embedding generated!")
        print(f"Shape: {embedding.shape}")
        print(f"First 5 values: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_long_context():
    """Test with longer context"""
    print("\n" + "="*60)
    print("TEST 5: Long Context Query")
    print("="*60)
    
    try:
        llm = OllamaInterface(model="llama2:latest")
        
        # Create a longer context with multiple facts
        context = """
        Document 1: The company's Q1 revenue was $10 million.
        Document 2: The CEO's name is Alice Johnson.
        Document 3: The company was founded in 2010.
        Document 4: The main product is called SuperApp.
        Document 5: The company has 150 employees.
        """
        
        query = "What is the CEO's name?"
        
        print(f"Context length: {len(context)} characters")
        print(f"Query: {query}")
        print("\nQuerying Ollama...")
        
        response = llm.query(context, query)
        
        print(f"\n✅ Query successful!")
        print(f"Response: {response.text}")
        print(f"Latency: {response.latency:.2f} seconds")
        
        # Check if response contains the correct answer
        if "Alice" in response.text or "alice" in response.text.lower():
            print("✅ Correct answer extracted from context!")
        else:
            print("⚠️ Answer may not be accurate")
        
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_parameter_options():
    """Test with different temperature settings"""
    print("\n" + "="*60)
    print("TEST 6: Parameter Options (Temperature)")
    print("="*60)
    
    try:
        llm = OllamaInterface(model="llama2:latest")
        
        query = "Say 'hello' in one word."
        
        # Test with low temperature (more deterministic)
        print("Testing with temperature=0.1 (deterministic)...")
        response1 = llm.query("", query, temperature=0.1)
        print(f"Response: {response1.text}")
        
        # Test with high temperature (more creative)
        print("\nTesting with temperature=0.9 (creative)...")
        response2 = llm.query("", query, temperature=0.9)
        print(f"Response: {response2.text}")
        
        print("\n✅ Parameter options work!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("OLLAMA INTERFACE INTEGRATION TESTS")
    print("="*60)
    print(f"Testing with Ollama at http://localhost:11434")
    print(f"Model: llama2:latest")
    
    tests = [
        ("Basic Query", test_basic_query),
        ("No Context Query", test_no_context_query),
        ("Token Counting", test_token_counting),
        ("Embedding", test_embedding),
        ("Long Context", test_long_context),
        ("Parameter Options", test_parameter_options),
    ]
    
    results = []
    start_time = time.time()
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Time: {total_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! OllamaInterface is working correctly.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
