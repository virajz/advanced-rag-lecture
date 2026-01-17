"""
Step 1: Setup and Verify Mistral AI Connection
==============================================

This script verifies that your Mistral AI setup is working correctly.
We'll test both the chat completion and embeddings APIs.

Prerequisites:
1. Install dependencies: pip install -r requirements.txt
2. Set your API key: export MISTRAL_API_KEY="your-key-here"
   Or create a .env file with: MISTRAL_API_KEY=your-key-here

Run: python lecture/01_setup.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def verify_api_key():
    """Check if the API key is set."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY not found!")
        print("\nTo fix this, either:")
        print("  1. Export it: export MISTRAL_API_KEY='your-key-here'")
        print("  2. Create a .env file with: MISTRAL_API_KEY=your-key-here")
        return None
    print("API key found!")
    return api_key


def test_chat_completion(client):
    """Test the chat completion API."""
    print("\n--- Testing Chat Completion ---")

    response = client.chat.complete(
        model="mistral-small-latest",  # Cost-effective model for testing
        messages=[
            {"role": "user", "content": "Describe 3 random pantone of the year colors with only their names and year."}
        ]
    )

    answer = response.choices[0].message.content
    print(f"Response: {answer}")
    return True


def test_embeddings(client):
    """Test the embeddings API."""
    print("\n--- Testing Embeddings ---")

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=["Hello, this is a test sentence for embeddings."]
    )

    embedding = response.data[0].embedding
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    return True


def main():
    print("=" * 50)
    print("Mistral AI Setup Verification")
    print("=" * 50)

    # Step 1: Check API key
    api_key = verify_api_key()
    if not api_key:
        return

    # Step 2: Initialize client
    print("\nInitializing Mistral client...")
    try:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        print("Client initialized!")
    except ImportError:
        print("ERROR: mistralai package not installed!")
        print("Run: pip install -r requirements.txt")
        return

    # Step 3: Test chat completion
    try:
        test_chat_completion(client)
        print("Chat completion working!")
    except Exception as e:
        print(f"Chat completion failed: {e}")
        return

    # Step 4: Test embeddings
    try:
        test_embeddings(client)
        print("Embeddings working!")
    except Exception as e:
        print(f"Embeddings failed: {e}")
        return

    print("\n" + "=" * 50)
    print("SUCCESS! All systems operational.")
    print("You're ready for the Advanced RAG lecture!")
    print("=" * 50)


if __name__ == "__main__":
    main()
