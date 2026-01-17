# evaluation.py - The Report Card

## What is this file?

This file **grades** how well the RAG system is doing. It's like a teacher checking if the AI's answers are good!

## The Big Picture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   QUESTION   │     │   CONTEXTS   │     │    ANSWER    │
│  "How do I   │     │  [Doc1,Doc2] │     │  "Use SSL    │
│   setup SSL?"│     │  Retrieved   │     │   certs..."  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                    ┌───────────────┐
                    │   EVALUATE    │
                    │   📊 📊 📊     │
                    └───────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Context    │     │ Faithfulness │     │   Answer     │
│  Relevance   │     │              │     │  Relevance   │
│    0.67      │     │    0.80      │     │    0.50      │
└──────────────┘     └──────────────┘     └──────────────┘
```

## The Three Grades (Metrics)

### 1. Context Relevance 📚
**Question:** "Did we find the RIGHT documents?"

```python
def context_relevance(question: str, contexts) -> float:
```

- Takes the question words and checks: do the retrieved documents contain these words?
- Score: 0.0 (terrible) to 1.0 (perfect)

**Example:**
- Question: "How do I configure SSL?"
- If 2 out of 3 retrieved docs mention "SSL" or "configure" → Score = 0.67

### 2. Faithfulness 🤝
**Question:** "Is the answer actually based on the documents?"

```python
def faithfulness(answer: str, contexts) -> float:
```

- Checks if the answer's claims can be found in the retrieved documents
- Prevents the AI from making stuff up!

**Example:**
- Answer: "Use PEM files. Rotate every 90 days."
- If both claims appear in the documents → High score!
- If the AI says something not in the docs → Lower score

### 3. Answer Relevance ✅
**Question:** "Does the answer actually address the question?"

```python
def answer_relevance(question: str, answer: str) -> float:
```

- Checks word overlap between question and answer
- Makes sure we're not answering a different question!

## Explained Like You're 5

Imagine you ask your friend: **"What's your favorite color?"**

| Metric | Good Example | Bad Example |
|--------|--------------|-------------|
| Context Relevance | Friend looks at a color chart | Friend looks at a food menu |
| Faithfulness | "Blue - it says here blue is calming" | "Blue - I just made that up" |
| Answer Relevance | "My favorite color is blue" | "I had pizza for lunch" |

## The Main Function

```python
def evaluate(question: str, contexts, answer: str):
    return {
        "context_relevance": context_relevance(question, contexts),
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevance": answer_relevance(question, answer),
    }
```

This bundles all three scores into one report card!

## Important Note ⚠️

This is a **simplified** evaluation! The comments say "rough" and "very rough" because:
- Real evaluation uses AI models to judge
- These just count word overlaps
- It's good for learning, not production!

## Example Output

```python
{
    "context_relevance": 0.67,  # 67% of docs were relevant
    "faithfulness": 0.80,       # 80% of claims were grounded
    "answer_relevance": 0.50    # 50% word overlap with question
}
```

---
*This file helps you understand: "Is my RAG system actually working well?"*
