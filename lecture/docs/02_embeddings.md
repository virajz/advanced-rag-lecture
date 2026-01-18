# 02_embeddings.py - Understanding Embeddings

## What This File Does

This file teaches you how embeddings work - the foundation of all semantic search and RAG systems. By the end, you'll understand how computers can find text with similar meaning, not just matching words.

## How It Works (Step by Step)

### Part 1: Creating an Embedding

```python
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding
```

**What's happening:**
- You send a sentence to Mistral
- Mistral returns a list of 1024 numbers
- These numbers capture the "meaning" of your text

**Think of it like this:** Imagine describing a color using RGB values (red, green, blue). Instead of saying "sky blue", you say [135, 206, 235]. Embeddings do the same for meaning - they describe text using numbers.

### Part 2: Measuring Similarity

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(a, b)
    return dot_product / (np.linalg.norm(a) * np.linalg.norm(b))
```

**What's happening:**
- Takes two embeddings (lists of numbers)
- Calculates how similar they are
- Returns a score from -1 to 1

**The scores mean:**
- **1.0** = Identical meaning
- **0.7+** = Very similar
- **0.5** = Somewhat related
- **0.0** = Completely unrelated
- **-1.0** = Opposite meaning

**Think of it like this:** Two arrows pointing the same direction have similarity 1.0. Two arrows at right angles have similarity 0.0. Cosine similarity measures the angle between vectors.

### Part 3: Semantic Search

```python
# Get embeddings for query and all documents
query_embedding = embeddings[0]
doc_embeddings = embeddings[1:]

# Find most similar documents
for doc, embedding in zip(documents, doc_embeddings):
    similarity = cosine_similarity(query_embedding, embedding)
    results.append({"text": doc, "similarity": similarity})

# Sort by similarity
results.sort(key=lambda x: x["similarity"], reverse=True)
```

**What's happening:**
1. Convert the search query to an embedding
2. Compare it against all document embeddings
3. Rank documents by similarity score
4. Return the most similar ones

**This is powerful because:**
- "SSL configuration" matches "HTTPS setup" even though they use different words
- The computer understands meaning, not just keywords

## Key Concepts Explained

### Why Batch Embedding?
```python
def get_embeddings_batch(texts: list[str]):
    response = client.embeddings.create(inputs=texts)
```
Instead of calling the API once per document, you send many at once. This is faster and cheaper - like buying in bulk.

### The 1024 Dimensions
Mistral embeddings have 1024 numbers. Each dimension captures some aspect of meaning. You don't need to understand what each dimension means - just know that together they represent the text's meaning.

## Real-World Example

**Query:** "How do I set up SSL?"

**Documents ranked by similarity:**
1. "To configure HTTPS, set the certificate path..." (0.85 - HIGH)
2. "SSL certificates use PEM or DER formats..." (0.78 - HIGH)
3. "Rotate certificates every 90 days..." (0.65 - MEDIUM)
4. "Python is a programming language..." (0.32 - LOW)
5. "The weather forecast shows rain..." (0.15 - LOW)

Notice how SSL/HTTPS/certificates all rank high even though they use different words. That's the power of semantic search.

## Why This Matters for RAG

RAG (Retrieval-Augmented Generation) works like this:
1. User asks a question
2. Find relevant documents using embeddings (what you learned here)
3. Give those documents to an LLM to generate an answer

Embeddings are step 2 - they're how we find the right information to give the AI.
