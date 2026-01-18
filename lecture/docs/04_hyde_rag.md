# 04_hyde_rag.py - HyDE (Hypothetical Document Embeddings)

## What This File Does

This file solves the "vocabulary mismatch" problem in RAG. When users ask questions using different words than what's in your documents, baseline RAG fails. HyDE fixes this.

## The Problem

**User asks:** "Why is my app crashing?"
**Documents say:** "Application failures occur due to missing SSL configuration..."

These mean the same thing, but baseline RAG can't find the connection because:
- "crashing" ≠ "failures"
- "app" ≠ "application"
- User doesn't mention SSL at all

The embeddings are too different, so the right documents don't get retrieved.

## The HyDE Solution

**Key Insight:** Instead of embedding the question, embed a hypothetical answer.

### How It Works

```python
def hyde_retrieve(query: str, vector_store, top_k: int = 3) -> dict:
    # Step 1: Generate a hypothetical answer (BEFORE searching)
    hypothetical_doc = generate_hypothetical_document(query)

    # Step 2: Embed the hypothetical (NOT the question)
    hyde_embedding = get_mistral_embeddings([hypothetical_doc])[0]

    # Step 3: Search using the hypothetical embedding
    results = vector_store.search_with_embedding(hyde_embedding, top_k)

    return {"hypothetical_doc": hypothetical_doc, "results": results}
```

### Step 1: Generate Hypothetical Document

```python
def generate_hypothetical_document(query: str) -> str:
    prompt = """Given a user's question, write a short technical document
    that would answer this question. Write in documentation style."""

    response = client.chat.complete(messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content
```

**What's happening:**
- Ask the LLM: "If you were going to answer this question, what would the answer document look like?"
- The LLM generates a fake answer using technical, document-style language

**Example:**
- **Query:** "Why is my app crashing?"
- **Hypothetical:** "Application startup failures commonly occur due to missing SSL certificates or incorrect TLS configuration. Ensure certificate paths are correctly specified..."

### Step 2: Embed the Hypothetical

```python
hyde_embedding = get_mistral_embeddings([hypothetical_doc])[0]
```

Now we have an embedding that:
- Uses document-style vocabulary ("SSL," "certificates," "configuration")
- Matches the language of our actual documents
- Bridges the gap between user-speak and doc-speak

### Step 3: Search with Hypothetical Embedding

```python
results = vector_store.search_with_embedding(hyde_embedding, top_k)
```

The hypothetical document is similar to real documents, so we find better matches.

## Why This Works

**Think of it like this:**

Baseline RAG is like searching a library in French when all the books are in English. You won't find anything.

HyDE is like asking a translator: "What would the English version of my French question look like?" Then searching with that English translation.

The hypothetical answer uses the same language as your documents.

## Comparison: Baseline vs HyDE

```
Query: "Why is my app crashing on startup?"

BASELINE (embed the question):
  [ssl_basics] sim=0.42     <- Low similarity
  [troubleshooting] sim=0.38
  [ports] sim=0.35

HYDE (embed the hypothetical):
  Hypothetical: "Application failures during initialization often relate
                 to SSL/TLS configuration issues..."

  [troubleshooting] sim=0.78  <- Much higher!
  [ssl_basics] sim=0.75
  [production] sim=0.72
```

## Trade-offs

### Pros
- Fixes vocabulary mismatch
- Works great for vague or short queries
- Helps when users don't know technical terms

### Cons
- **Extra LLM call** = more latency and cost
- **Hypothetical might be wrong** - but the retrieval usually corrects it
- Not needed for detailed, technical queries

## When to Use HyDE

**Use HyDE when:**
- Queries are short and vague
- Users don't know technical terminology
- There's vocabulary mismatch between users and docs
- Building FAQ/support systems

**Skip HyDE when:**
- Queries already use technical terms
- Users are experts who know the vocabulary
- Cost/latency is critical and queries are detailed

## The Full HyDE RAG Pipeline

```python
def hyde_rag(query: str, vector_store, top_k: int = 3) -> dict:
    # 1. Generate hypothetical
    hyde_result = hyde_retrieve(query, vector_store, top_k)

    # 2. Generate answer using retrieved docs
    answer = generate_answer(query, hyde_result["results"])

    return {
        "query": query,
        "hypothetical_doc": hyde_result["hypothetical_doc"],
        "contexts": hyde_result["results"],
        "answer": answer
    }
```

This is baseline RAG + one extra step at the beginning to generate a hypothetical.
