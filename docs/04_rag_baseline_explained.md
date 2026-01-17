# rag_baseline.py - The Simple RAG System

## What is this file?

This is the **most basic RAG system** - the "vanilla" version that other files improve upon. It's intentionally simple (even "dumb") to show you the basics.

## What is RAG?

**RAG = Retrieval-Augmented Generation**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. SEARCH  │───▶│  2. RETRIEVE│───▶│  3. ANSWER  │
│  "Find docs │    │  "Here are  │    │  "Based on  │
│   about X"  │    │   3 docs"   │    │   docs..."  │
└─────────────┘    └─────────────┘    └─────────────┘
```

Instead of the AI making stuff up, it:
1. **Searches** a knowledge base
2. **Retrieves** relevant documents
3. **Generates** an answer based on those documents

## The Two Functions

### 1. naive_vector_search() - The Searcher 🔍

```python
def naive_vector_search(query: str, top_k: int = 3):
```

**What it does:** Finds the most relevant documents for your question.

**How it works (super simple version):**
1. Take your question, split into words: `"How configure SSL"` → `["how", "configure", "ssl"]`
2. For each document, count matching words
3. Return the top 3 documents with most matches

```
Question: "How do I configure SSL?"
         ↓
   Words: [how, do, i, configure, ssl]
         ↓
┌─────────────────────────────────────────┐
│ Doc: "SSL certificates are PEM format"  │ → 1 match (ssl)
│ Doc: "Production SSL setup steps..."    │ → 2 matches (ssl, setup)  ✓
│ Doc: "Configure port 443..."            │ → 1 match (configure)
└─────────────────────────────────────────┘
```

### 2. naive_generate_answer() - The Answerer 💬

```python
def naive_generate_answer(query: str, contexts):
```

**What it does:** Creates an answer using the retrieved documents.

**In this demo:** It's FAKE! It just shows the retrieved docs and gives a hardcoded answer. In a real system, this would call an LLM (like ChatGPT or Claude).

## Explained Like You're 5

**Searching:** Imagine you have a box of cards. You want cards about "dogs". You go through each card and count how many times it says "dog". The cards that say "dog" the most go to the top!

**Answering:** After finding the best cards, you read them out loud and try to answer the question based on what the cards say.

## The Code Flow

```python
if __name__ == "__main__":
    # 1. The question
    query = "How do I configure SSL certificates for production deployment?"

    # 2. Search for relevant documents
    ctx = naive_vector_search(query, top_k=3)

    # 3. Generate and print the answer
    print(naive_generate_answer(query, ctx))
```

## Why Is It Called "Naive"?

Because it's **oversimplified**! Real RAG systems use:

| This Demo | Real Systems |
|-----------|--------------|
| Word counting | Vector embeddings (AI understands meaning) |
| Exact matches | Semantic similarity (understands synonyms) |
| Fake answers | Real LLM calls |

**Example of the problem:**
- Query: "How to set up HTTPS?"
- Doc: "Configure SSL certificates..."
- This system finds **0 matches** because "HTTPS" ≠ "SSL" (even though they're related!)

## Running It

```bash
python rag_baseline.py
```

**Output:**
```
Q: How do I configure SSL certificates for production deployment?

Retrieved:
- (ssl_formats) SSL certificates are commonly encoded...
- (prod_ssl_steps) Production SSL setup steps...
- (rotation_policy) Certificate rotation policy...

Answer:
Use SSL certificates (PEM/DER) and rotate them regularly.
```

---
*This is the "before" picture - the other RAG files show how to make it better!*
