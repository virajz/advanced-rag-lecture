# 01_setup.py - Setup and Verify Mistral AI Connection

## What This File Does

This is the starting point of the lecture series. It checks that your Mistral AI setup is working correctly before you begin building RAG systems.

## How It Works (Step by Step)

### 1. Check for API Key
```python
api_key = os.environ.get("MISTRAL_API_KEY")
```
The script first looks for your Mistral API key in your environment variables. Think of this like checking that you have the key to a building before trying to enter.

### 2. Initialize the Mistral Client
```python
client = Mistral(api_key=api_key)
```
This creates a connection to Mistral's servers. It's like dialing a phone number - you're establishing a communication line with their AI services.

### 3. Test Chat Completion
```python
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "..."}]
)
```
This tests if you can have a conversation with the AI. It sends a simple question and checks if you get a response back. Like testing a microphone before a presentation - "Can you hear me?"

### 4. Test Embeddings
```python
response = client.embeddings.create(
    model="mistral-embed",
    inputs=["Hello, this is a test sentence."]
)
```
This tests the embedding feature - turning text into numbers (vectors). Embeddings are crucial for RAG because they let us find similar pieces of text. This test confirms that feature works.

## Key Concepts Explained

### What are Embeddings?
Imagine each sentence as a point in space. Similar sentences are close together, different ones are far apart. Embeddings are the coordinates of those points - a list of numbers that represent the meaning of text.

### Why Two Tests?
RAG needs both abilities:
- **Chat Completion**: To generate answers based on context
- **Embeddings**: To find relevant documents for a question

## What Success Looks Like

When everything works, you'll see:
```
SUCCESS! All systems operational.
You're ready for the Advanced RAG lecture!
```

## Common Issues

1. **"API key not found"** - You need to set the MISTRAL_API_KEY environment variable
2. **"mistralai package not installed"** - Run `pip install -r requirements.txt`
3. **Connection errors** - Check your internet connection
