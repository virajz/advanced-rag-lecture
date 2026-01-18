# 06_evaluation.py - RAG Evaluation Metrics

## What This File Does

This file teaches you how to measure if your RAG system is actually good. Instead of "it feels right," you get actual numbers that tell you what's working and what's not.

## The Three Core Metrics

### 1. Context Relevance

**Question:** Did we retrieve the RIGHT documents?

```python
def evaluate_context_relevance(question: str, contexts: list[dict]) -> float:
    for ctx in contexts:
        prompt = f"""Is this document useful for answering the question?
        QUESTION: {question}
        DOCUMENT: {ctx['text']}
        Answer YES or NO."""

        is_relevant = "YES" in response.upper()

    score = relevant_count / total_contexts
    return score
```

**What's happening:**
- For each retrieved document, ask: "Is this useful for answering the question?"
- Count how many are relevant
- Score = relevant documents / total documents

**Scores mean:**
- **1.0** = All retrieved documents are relevant (perfect retrieval)
- **0.67** = 2 of 3 documents are useful (one wasted slot)
- **0.33** = Only 1 of 3 is useful (poor retrieval)

**If this score is low:** Your retrieval is the problem. Try HyDE, better chunking, or different embedding models.

### 2. Faithfulness

**Question:** Is the answer grounded in the documents? (No hallucination?)

```python
def evaluate_faithfulness(answer: str, contexts: list[dict]) -> float:
    # Step 1: Extract claims from the answer
    claims = extract_claims(answer)

    # Step 2: Check each claim against context
    for claim in claims:
        is_supported = check_if_supported(claim, contexts)

    score = supported_claims / total_claims
    return score
```

**What's happening:**
- Break the answer into individual factual claims
- For each claim, check: "Can this be found in or inferred from the context?"
- Score = supported claims / total claims

**Scores mean:**
- **1.0** = Every claim in the answer is backed by the documents
- **0.75** = 3 of 4 claims are supported, 1 is made up
- **0.5** = Half the answer is hallucinated

**If this score is low:** The LLM is making things up. Use stricter prompts ("ONLY use the context") or self-corrective RAG.

### 3. Answer Relevance

**Question:** Does the answer actually address what was asked?

```python
def evaluate_answer_relevance(question: str, answer: str) -> float:
    prompt = f"""Does the answer address the question?
    QUESTION: {question}
    ANSWER: {answer}
    Rate: FULLY, PARTIALLY, or NOT"""

    # Also use embedding similarity as backup signal
    embeddings = get_mistral_embeddings([question, answer])
    similarity = cosine_similarity(embeddings[0], embeddings[1])

    combined = 0.7 * llm_score + 0.3 * similarity
    return combined
```

**What's happening:**
- Ask an LLM: "Does this answer the question?"
- Also check embedding similarity (question vs answer)
- Combine both signals

**Scores mean:**
- **1.0** = Answer directly and completely addresses the question
- **0.5** = Answer is partially helpful
- **0.0** = Answer doesn't address the question at all

**If this score is low but faithfulness is high:** You have a generation problem, not a retrieval problem. The right info was found, but the answer missed the point.

## Complete Evaluation

```python
def evaluate_rag_response(question, contexts, answer):
    return {
        "context_relevance": evaluate_context_relevance(question, contexts),
        "faithfulness": evaluate_faithfulness(answer, contexts),
        "answer_relevance": evaluate_answer_relevance(question, answer)
    }
```

## Diagnostic Framework

| Context Relevance | Faithfulness | Answer Relevance | Diagnosis |
|-------------------|--------------|------------------|-----------|
| LOW | Any | Any | Retrieval problem - try HyDE |
| HIGH | LOW | Any | Hallucination - stricter prompts |
| HIGH | HIGH | LOW | Generation problem - improve prompts |
| HIGH | HIGH | HIGH | System is working well! |

## Example Results

### Test 1: Good Query
```
Query: "How do I configure SSL certificates for production?"
Context Relevance: 1.00  (all 3 docs relevant)
Faithfulness:      1.00  (all claims supported)
Answer Relevance:  0.95  (directly answers question)
```
Everything is working!

### Test 2: Hallucination Test
```
Deliberate hallucination: "Use port 8080 for development (made up)"
Context Relevance: 1.00
Faithfulness:      0.50  <- Caught the hallucination!
Answer Relevance:  0.85
```
The faithfulness metric detected unsupported claims.

### Test 3: Off-Topic Query
```
Query: "How do I make coffee?"
Context Relevance: 0.00  <- No relevant docs found
Faithfulness:      1.00  (correctly said "no info")
Answer Relevance:  0.30  (can't answer what's not there)
```
System correctly identified it can't help with this.

## Why These Metrics Matter

**Without metrics:**
- "I think the RAG is working okay"
- "Users seem happy sometimes"
- Guessing at what to improve

**With metrics:**
- "Context relevance dropped 20% on troubleshooting queries"
- "HyDE improves faithfulness by 15% for vague queries"
- "Self-correction adds cost but increases answer relevance by 25%"

**You can now make data-driven decisions** about which techniques to use and when.

## Using Metrics to Improve

1. **Run evaluation on test queries**
2. **Identify the bottleneck** (retrieval, hallucination, or generation)
3. **Apply the right fix:**
   - Low context relevance → HyDE, better chunking
   - Low faithfulness → Self-correction, stricter prompts
   - Low answer relevance → Better generation prompts
4. **Re-evaluate to confirm improvement**
