# 05_self_corrective_rag.py - Self-Corrective RAG

## What This File Does

This file implements a RAG system that checks its own work. Instead of blindly generating an answer from whatever documents it retrieves, it first asks: "Are these documents good enough to answer the question?"

If not, it tries again with a better query.

## The Problem

Sometimes the first retrieval misses the mark:
- Retrieved documents are somewhat related but don't actually answer the question
- Important keywords were missed
- The query was too vague

Baseline RAG just uses whatever it gets. Self-Corrective RAG is smarter.

## The Self-Corrective Pattern

### The Loop

```python
def self_corrective_rag(query, vector_store, max_iterations=3):
    current_query = query

    for iteration in range(max_iterations):
        # Step 1: Retrieve documents
        contexts = vector_store.search(current_query, top_k=3)

        # Step 2: Reflect - Are these good enough?
        reflection = reflect(query, contexts)

        # Step 3: Decide
        if reflection["decision"] == "YES":
            # Good enough! Generate answer
            answer = generate_answer(query, contexts)
            return answer

        # Not good enough - reformulate and try again
        current_query = reformulate_query(current_query, reflection["reason"])

    # Max iterations reached - use best available
    return generate_answer(query, contexts)
```

### Step 1: Retrieve

Same as baseline RAG - get documents based on query similarity.

### Step 2: Reflect (The Key Step)

```python
def reflect(query: str, contexts: list[dict]) -> dict:
    prompt = """Can these documents answer the question completely?
    QUESTION: {query}
    DOCUMENTS: {contexts}

    DECISION: YES or NO
    REASON: <explanation>"""

    response = client.chat.complete(...)
    return {"decision": "YES/NO", "reason": "..."}
```

**What's happening:**
- Show an LLM the question and retrieved documents
- Ask it to judge: "Is this enough to give a good answer?"
- Get a YES/NO decision and a reason

**Think of it like:** A teacher reviewing a student's research before they write an essay. "Did you find the right sources? No? Here's what's missing..."

### Step 3: Reformulate (When Needed)

```python
def reformulate_query(original_query: str, reflection_reason: str) -> str:
    prompt = """The search didn't find good documents.
    ORIGINAL QUERY: {original_query}
    PROBLEM: {reflection_reason}

    Create a better search query with more specific terms."""

    return client.chat.complete(...).content
```

**What's happening:**
- Take the original query and the reason it failed
- Generate a new, more specific query
- Use this for the next retrieval attempt

**Example:**
- Original: "How do I fix errors?"
- Problem: "Too vague, doesn't specify what kind of errors"
- Reformulated: "How do I fix SSL certificate configuration errors?"

## Real Example

```
Query: "What exact port and TLS version should I use for a compliant
        production container deployment?"

--- Iteration 1 ---
Query: "What exact port and TLS version should I use for..."
Retrieved: [ssl_basics, production, ports]
Reflection: NO - "Missing specific compliance requirements for TLS versions"
Reformulated: "TLS 1.2 version compliance requirements container port 8443"

--- Iteration 2 ---
Query: "TLS 1.2 version compliance requirements container port 8443"
Retrieved: [compliance, ports, production]
Reflection: YES - "Documents contain port 8443 and TLS 1.2+ requirements"

Final Answer: "Use port 8443 internally with TLS 1.2 or higher..."
```

## Key Concepts

### Why Reflect on Original Query?

```python
reflection = reflect(query, contexts)  # Always use ORIGINAL query
```

We always check against the original question, not the reformulated one. The goal is to answer what the user actually asked.

### Max Iterations

```python
max_iterations = 3
```

We cap the attempts to avoid infinite loops. If we can't find good context in 3 tries, we generate the best answer we can with what we have.

### The Trace

The function returns a detailed trace showing:
- Each iteration's query
- What was retrieved
- The reflection decision
- How queries were reformulated

This is valuable for debugging and understanding why answers are good or bad.

## Trade-offs

### Pros
- Catches bad retrievals before generating
- Improves answer quality for complex questions
- Self-healing - automatically fixes vague queries

### Cons
- **More API calls** - reflection + reformulation costs extra
- **Higher latency** - multiple retrieval rounds
- **Reflection can be wrong** - false positives/negatives

## Cost Analysis

```
Baseline RAG: 1 embedding + 1 generation = 2 API calls
Self-Corrective (2 iterations): 2 embeddings + 2 reflections + 1 generation = 5 API calls
```

That's 2.5x the API cost. Worth it for complex queries, overkill for simple ones.

## When to Use Self-Corrective RAG

**Use when:**
- Queries are complex or multi-part
- You need high accuracy
- Users ask vague questions
- Troubleshooting/error-fixing queries

**Skip when:**
- Queries are straightforward
- Cost/latency is critical
- Queries already use specific terminology

## Combining with HyDE

You can use both HyDE and Self-Correction:
1. HyDE generates a hypothetical to bridge vocabulary gaps
2. Self-Correction checks if the retrieval was good enough
3. If not, reformulate and try again

This "hybrid" approach is covered in the production RAG lecture.
