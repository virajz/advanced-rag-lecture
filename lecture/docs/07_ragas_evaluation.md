# 07_ragas_evaluation.py - Automated Evaluation Pipeline

## What This File Does

This file shows how to build a production-ready evaluation pipeline inspired by RAGAS (Retrieval Augmented Generation Assessment). Instead of manually testing queries one at a time, you create a test suite and run automated evaluations.

## The Test Case Structure

```python
@dataclass
class RAGTestCase:
    query: str           # The question to ask
    expected_answer: str # What a good answer should contain
    category: str        # Type of question (for analysis)
```

**Example test cases:**
```python
TEST_CASES = [
    RAGTestCase(
        query="How do I configure SSL certificates for production?",
        expected_answer="Put fullchain.pem and privkey.pem in /etc/myapp/tls/...",
        category="configuration"
    ),
    RAGTestCase(
        query="What is the certificate rotation policy?",
        expected_answer="Rotate every 90 days. Automate renewal...",
        category="policy"
    ),
]
```

**Why test cases matter:**
- Represent real user questions
- Have known good answers
- Categorized for analysis

## The Evaluation Pipeline

```python
def evaluate_rag_pipeline(test_cases, vector_store):
    results = []

    for test_case in test_cases:
        # 1. Run RAG pipeline
        contexts = vector_store.search(test_case.query)
        answer = generate_answer(test_case.query, contexts)

        # 2. Compute all metrics
        ctx_rel = compute_context_relevance(test_case.query, contexts)
        faith = compute_faithfulness(answer, contexts)
        ans_rel = compute_answer_relevance(test_case.query, answer)
        ans_corr = compute_answer_correctness(answer, test_case.expected_answer)

        results.append(EvaluationResult(...))

    return results
```

**What's happening:**
1. Run your RAG system on each test question
2. Measure all quality metrics
3. Collect results for analysis

## The Four Metrics

### 1. Context Relevance
"Are the retrieved documents useful for answering?"
```python
for ctx in contexts:
    "Is this document useful for answering the question? YES or NO"
score = yes_count / total_count
```

### 2. Faithfulness
"Is the answer grounded in the context?"
```python
"Does the answer contain only information from the context?"
# Rate: HIGH, MEDIUM, LOW
```

### 3. Answer Relevance
"Does the answer address the question?"
```python
"Rate how well the answer addresses the question."
# Rate: FULLY, PARTIALLY, NOT
```

### 4. Answer Correctness (NEW)
"How close is the answer to the expected answer?"
```python
def compute_answer_correctness(answer, expected):
    embeddings = get_mistral_embeddings([answer, expected])
    return cosine_similarity(embeddings[0], embeddings[1])
```

This uses embedding similarity to compare against the expected answer.

## Aggregating Results

```python
def aggregate_results(results):
    # Overall statistics
    overall = {
        "context_relevance": {"mean": 0.85, "std": 0.12, ...},
        "faithfulness": {"mean": 0.92, "std": 0.08, ...},
        ...
    }

    # Breakdown by category
    by_category = {
        "configuration": {"count": 2, "context_relevance": 0.90, ...},
        "policy": {"count": 1, "context_relevance": 0.75, ...},
    }

    return aggregated
```

**What you learn:**
- Overall system performance
- Which question categories are weak
- Where to focus improvements

## Example Output

```
--- Overall Metrics ---
Metric                Mean     Std      Min      Max
----------------------------------------------------
context_relevance     0.85     0.12     0.67     1.00
faithfulness          0.92     0.08     0.80     1.00
answer_relevance      0.80     0.15     0.50     1.00
answer_correctness    0.78     0.10     0.65     0.90

--- By Category ---
Category        Count   Ctx Rel   Faith   Ans Rel
-------------------------------------------------
configuration       2     0.90     0.95     0.85
policy              1     0.75     0.90     0.70
compliance          1     0.80     0.88     0.82
```

## Decision Framework

The script provides recommendations based on scores:

```python
if ctx_rel_mean < 0.7:
    print("LOW Context Relevance: Try HyDE for vocabulary mismatch")

if faith_mean < 0.8:
    print("LOW Faithfulness: Implement Self-Corrective RAG")

if ans_rel_mean < 0.7 and faith_mean >= 0.8:
    print("LOW Answer Relevance but HIGH Faithfulness:")
    print("This is a GENERATION problem, not retrieval")
```

## Rate Limiting

The script handles API rate limits:

```python
def llm_call_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.complete(...)
        except Exception as e:
            if "429" in str(e):  # Rate limited
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                time.sleep(wait_time)
```

And adds delays between calls:
```python
time.sleep(0.5)  # Between API calls to avoid rate limits
```

## Building Your Test Suite

### Start Small
Create 5-10 test cases representing your typical use cases.

### Cover Categories
- Configuration questions
- Troubleshooting
- Policy/compliance
- Reference lookups

### Include Edge Cases
- Vague queries
- Technical jargon
- Multi-part questions

### Write Good Expected Answers
These don't need to be perfect, but should contain the key information.

## The Key Insight

From the lecture:
> "You can now make an informed decision: the improvement in retrieval quality is worth the additional API cost for your use case. This is evidence-based engineering instead of guessing."

Instead of debating whether HyDE or self-correction is worth the cost, you can measure it:
- Run baseline evaluation → get scores
- Add HyDE → run evaluation again
- Compare: "HyDE improved context relevance by 15% for vague queries"
- Decision: "Worth the extra API call for our use case"
