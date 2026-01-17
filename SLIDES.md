---
marp: true
theme: default
paginate: true
size: 16:9
style: |
    section {
      justify-content: flex-start;
      padding-top: 50px;
    }
    section.title {
      justify-content: center;
      text-align: center;
    }
    h1, h2 {
      color: #2d3748;
    }
    blockquote {
      border-left: 4px solid #4a5568;
      background: #f7fafc;
      padding: 1em;
      font-style: normal;
      font-weight: bold;
    }
    table {
      font-size: 0.85em;
    }
    code {
      background: #edf2f7;
    }
---

<!-- _class: title -->

# Advanced RAG

## From Naive Retrieval to Production-Grade Systems

**Viraj Zaveri**

---

## THE RAG CHALLENGE

RAG lets LLMs answer questions using your private data. But naive implementations fail predictably:

**The Three Gaps:**

1. **Vocabulary Mismatch** - User: "HTTPS" vs Docs: "SSL/TLS"
2. **Incomplete Retrieval** - First search misses critical docs, no self-correction
3. **Multi-Document Reasoning** - Answer requires scattered info across multiple docs

---

## THE REAL COST OF FAILURE

**At 5K queries/day:**

- 20% failures = 1,000 bad answers daily
- User distrust → adoption stalls
- Support escalations increase
- Investment doesn't pay off

> THE RETRIEVAL WAS TECHNICALLY RELEVANT BUT PRACTICALLY USELESS

---

## THREE APPROACHES COMPARED

| Aspect       | Naive RAG           | HyDE                 | Self-Corrective       |
| ------------ | ------------------- | -------------------- | --------------------- |
| LLM Calls    | 1                   | 2                    | 2-7                   |
| Latency      | ~500ms              | ~1s                  | 1-5s                  |
| Best For     | Explicit queries    | Vague queries        | Complex queries       |
| Failure Mode | Vocabulary mismatch | Bad hypothetical     | Loop without progress |
| When to Use  | Exact terminology   | Problem descriptions | Troubleshooting       |

**Cost @ 1K queries/day:** Naive ($2-5) | HyDE ($4-10) | Self-Corrective ($8-20)

---

## 1. NAIVE RAG

**The Baseline:** Query → Embed → Search → Generate

```python
def naive_rag(query: str, top_k: int = 3) -> str:
    query_embedding = embed(query)
    results = vector_store.search(query_embedding, top_k)
    context = "\n\n".join([doc.text for doc in results])
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

**✅ Use When:** Direct questions with document vocabulary ("What is PEM format?")
**❌ Fails When:** Vocabulary mismatch ("Why won't my app start?")

---

## 2. HYDE: HYPOTHETICAL DOCUMENT EMBEDDINGS

**Core Insight:** Bridge the gap between question-space and document-space

```
User: "Why is my app crashing?"  →  LLM generates hypothetical answer
↓
"Apps crash due to missing configs, wrong DB credentials, or unset env vars..."
↓
Embed hypothetical (not original query)  →  Search  →  Find relevant docs
```

**✅ Use When:** Vague queries, vocabulary mismatch
**❌ Skip When:** Explicit queries (adds unnecessary latency)

---

## 3. SELF-CORRECTIVE RAG

**Core Insight:** Detect bad retrieval and retry with better query

```python
def self_corrective_rag(query: str, max_iters: int = 3) -> str:
    for i in range(max_iters):
        results = search(query)

        if reflect(query, results).sufficient:
            return generate(query, results)

        query = reformulate(query, reflection.missing)
```

---

## SELF-CORRECTIVE RAG: EXAMPLE

**Example:** SSL production query

1. **Iteration 1:** Finds dev docs
2. **Reflection:** "Missing prod steps"
3. **Reformulate:** Add production keywords
4. **Iteration 2:** Finds correct production doc

**✅ Use When:** Complex queries, troubleshooting
**❌ Skip When:** Simple queries (wastes money)

---

## PRODUCTION REALITY: SMART ROUTING

Real systems route dynamically:

```python
def route_query(question: str) -> str:
    if len(question.split()) < 10 and "why" in question.lower():
        return "hyde"
    elif any(word in question.lower() for word in
             ["configure", "setup", "troubleshoot"]):
        return "self_corrective"
    else:
        return "baseline"
```

**When to Route:**

- **HyDE:** Short, vague queries with problem words
- **Self-Corrective:** Complex multi-step queries
- **Baseline:** Explicit terminology

---

## EVALUATION: THE THREE METRICS

**1. Context Relevance** - Are retrieved docs useful?
**2. Faithfulness** - Is answer grounded in context?
**3. Answer Relevance** - Does answer address question?

```python
context_relevance = relevant_docs / total_docs
faithfulness = supported_claims / total_claims
answer_relevance = llm_score(question, answer)
```

> SHIP NOTHING WITHOUT EVALUATION INFRASTRUCTURE

---

## DECISION FRAMEWORK

```
                 START: Analyze Query
                         │
         ┌───────────────┴───────────────┐
         │                               │
    EXPLICIT?                         VAGUE?
    (doc vocab)                    (problem words)
         │                               │
         ▼                               ▼
    ┌─────────┐                    ┌─────────┐
    │  NAIVE  │                    │  HYDE   │
    └─────────┘                    └─────────┘

         COMPLEX + MULTI-STEP?
                 │
                 ▼
          ┌──────────────────┐
          │ SELF-CORRECTIVE  │
          └──────────────────┘
```

---

## CHOOSING YOUR METHOD

**Quick Guide:**

- Explicit terminology → **Naive RAG**
- Vague problem → **HyDE**
- Complex/troubleshooting → **Self-Corrective**

**Constraints:**

- Latency < 1s → **Naive only**
- Cost sensitive → **Route smartly (60/30/10)**

---

## COST & LATENCY AT SCALE

**10,000 Queries/Day - Monthly Cost:**

| Approach                 | Cost         | Avg Latency |
| ------------------------ | ------------ | ----------- |
| All Naive                | $600-1,500   | 500ms       |
| All HyDE                 | $1,200-3,000 | 1s          |
| All Self-Corrective      | $2,400-6,000 | 3.5s        |
| Smart Routing (60/30/10) | $1,000-2,400 | 1.2s        |

**Key Insight:** Smart routing improves quality while reducing cost

---

## PITFALLS & SOLUTIONS (1/2)

| Method | Common Pitfall      | Solution                          |
| ------ | ------------------- | --------------------------------- |
| Naive  | Vocabulary mismatch | Add HyDE or synonyms              |
| Naive  | Generic answers     | Improve chunking, increase top-K  |
| HyDE   | Bad hypotheticals   | Better prompts, few-shot examples |
| HyDE   | Inconsistent        | Set temperature=0                 |

---

## PITFALLS & SOLUTIONS (2/2)

| Method          | Common Pitfall         | Solution                          |
| --------------- | ---------------------- | --------------------------------- |
| Self-Corrective | Max iterations hit     | Adjust sufficiency criteria       |
| Self-Corrective | Loops without progress | Track and exclude seen docs       |
| All             | No quality tracking    | Automated evaluation from day one |
| All             | One-size-fits-all      | Implement smart routing           |

---

## CASE STUDY: SSL KNOWLEDGE BASE

**Context:** Internal docs for SSL/TLS configuration (7 documents)

**Problem:** Engineers ask in problem-space, docs written in solution-space

**Approach:** Implemented smart routing to choose the right RAG method per query

---

## CASE STUDY: RESULTS

**Impact with Smart Routing:**

| Metric            | Before | After |
| ----------------- | ------ | ----- |
| Context Relevance | 0.45   | 0.85  |
| Faithfulness      | 0.70   | 0.90  |
| Answer Relevance  | 0.55   | 0.88  |
| Daily Cost        | $10    | $4    |
| Avg Latency       | 3.5s   | 1.2s  |

> KEY INSIGHT: Routing improved quality AND reduced cost

---

## GETTING STARTED (1/2)

**Phase 1: Foundation**

- [ ] Implement naive RAG baseline
- [ ] Create 10-20 test cases
- [ ] Measure baseline metrics

**Phase 2: Advanced**

- [ ] Add HyDE for vague queries
- [ ] Add self-corrective for complex queries
- [ ] Build query router

---

## GETTING STARTED (2/2)

**Phase 3: Production**

- [ ] Cost/latency monitoring
- [ ] Quality dashboards with alerts
- [ ] A/B test routing decisions

**Phase 4: Iterate**

- [ ] Analyze routing decisions
- [ ] Review failure cases
- [ ] Tune based on metrics

---

## COMMON MISTAKES (1/2)

❌ **Skipping naive baseline** - Need comparison point

❌ **No evaluation** - Quality degrades silently

❌ **One-size-fits-all** - Wastes money

❌ **Ignoring costs** - Surprise bills at scale

---

## COMMON MISTAKES (2/2)

❌ **Complex before simple** - Debug nightmare

❌ **No failure analysis** - Edge cases erode trust

❌ **Trusting embeddings blindly** - Add reranking

❌ **Forgetting the user** - Optimize for outcomes

---

## KEY TAKEAWAYS

> RAG IS NOT A SINGLE TECHNIQUE. IT IS A SPECTRUM OF TRADE-OFFS.

**Success Formula:**

1. Implement naive baseline
2. Measure quality metrics
3. Add advanced techniques for specific failures
4. Route dynamically based on query type

---

## REMEMBER

**Core Principles:**

- 60% of queries work fine with naive RAG
- Start simple, add complexity when justified
- Ship evaluation first, features second

**Measure → Route → Iterate**

---

## RESOURCES

**GitHub Repository:**
https://github.com/virajz/advanced-rag-lecture

**What's Included:**

✅ Complete implementations (naive, HyDE, self-corrective)
✅ Evaluation framework with metrics
✅ Interactive Streamlit demo with debug mode
✅ Production routing patterns

---

## GET STARTED IN 5 MINUTES

```bash
# Clone the repository
git clone https://github.com/virajz/advanced-rag-lecture.git

# Install dependencies
cd advanced-rag-lecture && uv sync

# Set your API key
export MISTRAL_API_KEY="your-key"

# Run the first lesson
uv run python lecture/01_setup.py
```

**All code is production-ready and documented**

---

## Q&A

Questions?

---
