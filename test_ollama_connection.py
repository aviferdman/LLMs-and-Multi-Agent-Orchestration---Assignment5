"""Test Ollama API connectivity"""
import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama2:latest",
        "prompt": "What is 2+2? Answer in one word.",
        "stream": False
    }
    
    try:
        print("Testing Ollama connection...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ollama is working!")
            print(f"Response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_ollama()
