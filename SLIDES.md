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

## FOUNDATION: THE RAG PROMISE

RAG (Retrieval-Augmented Generation) lets LLMs answer questions using your private data.

**The Basic Idea:**

- User asks a question
- System retrieves relevant documents
- LLM generates answer grounded in those documents

---

## WHY RAG MATTERS

**Key Benefits:**

- No fine-tuning required
- Data stays current (no retraining)
- Answers are traceable to sources
- Reduces hallucination (in theory)

**Use Case:**

Answering questions about private docs without retraining models

---

## THE FUNDAMENTAL CHALLENGE (1/2)

Naive RAG fails in predictable, costly ways.

**Gap 1: Vocabulary Mismatch**

- User says "HTTPS" but docs say "SSL/TLS"
- Question words differ from document words
- Semantic meaning lost in keyword matching

**Gap 2: Incomplete Retrieval**

- First search misses critical documents
- No mechanism to self-correct
- System confidently returns wrong context

---

## THE FUNDAMENTAL CHALLENGE (2/2)

**Gap 3: Multi-Document Reasoning**

- Answer requires info scattered across docs
- Single query cannot surface all pieces
- User gets partial, unhelpful answers

**The Reality:**

These gaps cause expensive failures at scale

---

## THE FUNDAMENTAL CHALLENGE: REAL EXAMPLE

**Query:** "How do I configure SSL certificates for production deployment?"

**What Naive RAG Returns:**

- `ssl_formats.md` - PEM vs DER formats
- `rotation_policy.md` - 90-day certificate rotation
- `dev_setup.md` - Local development SSL

**What Was Needed:**

- `prod_ssl_steps.md` - Step-by-step production setup

**Result:** Generic answer about SSL formats instead of actionable production steps.

> THE RETRIEVAL WAS TECHNICALLY RELEVANT BUT PRACTICALLY USELESS.

---

## THE FUNDAMENTAL CHALLENGE: THE COST

**At Scale, Bad Retrieval Costs Real Money:**

- 5,000 queries/day
- 20% retrieval failures
- 1,000 bad answers daily
- Support escalations, user frustration, lost trust

**The Hidden Tax:**

- Engineers manually review edge cases
- Users learn to distrust the system
- Adoption stalls despite investment

---

## WHAT IS NAIVE RAG?

The simplest implementation: Query, Search, Answer.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Query   │ →  │  Embed   │ →  │  Search  │ →  │ Generate │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

**Step 1:** Convert query to embedding vector
**Step 2:** Find similar document embeddings
**Step 3:** Feed retrieved docs to LLM
**Step 4:** Return generated answer

---

## NAIVE RAG: THE IMPLEMENTATION

```python
def naive_rag(query: str, top_k: int = 3) -> str:
    # Step 1: Embed the query
    query_embedding = embed(query)

    # Step 2: Search for similar documents
    results = vector_store.search(query_embedding, top_k)

    # Step 3: Build context from results
    context = "\n\n".join([doc.text for doc in results])

    # Step 4: Generate answer
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return llm.generate(prompt)
```

---

## NAIVE RAG: ASSUMPTIONS

**Critical Assumptions:**

- Query embedding captures intent
- Similar embeddings = relevant documents
- Top-K is always sufficient
- Retrieved context is always useful

**When Assumptions Break:**

System fails silently with wrong answers

---

## NAIVE RAG: STRENGTHS

**Core Strengths:**

1. Simple to implement and debug
2. Fast (single embedding, single search)
3. Low cost (one LLM call)
4. Predictable behavior

---

## NAIVE RAG: WHEN TO USE

**Best For:**

- Direct factual questions
- Queries using same vocabulary as documents
- When user knows exact terminology
- High-volume, simple use cases

**Works Well When:**

Users speak the same language as your docs

---

## NAIVE RAG: WHEN IT EXCELS

| Query Type                                  | Success Rate | Why                           |
| ------------------------------------------- | ------------ | ----------------------------- |
| "What is the PEM certificate format?"       | High         | Exact terminology match       |
| "How many days until certificate rotation?" | High         | Specific, measurable question |
| "What port does the container use?"         | High         | Direct factual lookup         |

**The Pattern:**

- Explicit queries with document vocabulary
- Single-document answers
- No reasoning required across sources

---

## CRITICAL LIMITATION

> NAIVE RAG ASSUMES THE USER KNOWS HOW TO ASK.

**The Vocabulary Gap:**

- Users speak in problems: "My app won't start"
- Documents speak in solutions: "Missing configuration files cause startup failures"
- Cosine similarity cannot bridge this gap

---

## MORE LIMITATIONS

**The Single-Shot Problem:**

- One chance to get retrieval right
- No feedback loop
- No self-correction

**The Context Window Waste:**

- Retrieved docs may be tangentially related
- LLM context filled with noise
- Answer quality degrades

---

## WHAT IS HYDE?

**HyDE: Hypothetical Document Embeddings**

A technique that bridges the vocabulary gap between questions and documents.

**Core Insight:**
Questions and documents live in different semantic spaces. HyDE translates questions into document-space before searching.

**The Mental Model:**
Instead of searching with "Why won't my app start?", search with what the answer might look like.

---

## HYDE: HOW IT WORKS

```
┌──────────┐    ┌────────────────┐    ┌──────────┐    ┌──────────┐
│  Query   │ →  │ Generate Hypo  │ →  │  Embed   │ →  │  Search  │
└──────────┘    │   Document     │    │  Hypo    │    └──────────┘
                └────────────────┘    └──────────┘
```

**Step 1:** User asks question
**Step 2:** LLM generates hypothetical answer document
**Step 3:** Embed the hypothetical (not the question)
**Step 4:** Search with hypothetical embedding
**Step 5:** Retrieve and generate final answer

---

## HYDE: THE IMPLEMENTATION (1/2)

```python
def hyde_rag(query: str, top_k: int = 3) -> str:
    # Step 1: Generate hypothetical document
    hypo_prompt = f"""Write a short document that would answer
    this question. Use technical terminology.
    Question: {query}"""

    hypothetical = llm.generate(hypo_prompt)

    # Step 2: Embed the hypothetical (not the query!)
    hypo_embedding = embed(hypothetical)
```

---

## HYDE: THE IMPLEMENTATION (2/2)

```python
    # Step 3: Search with hypothetical embedding
    results = vector_store.search(hypo_embedding, top_k)

    # Step 4: Generate final answer with real docs
    context = "\n\n".join([doc.text for doc in results])
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

**Key Point:**

We search with the hypothetical embedding, NOT the original query

---

## HYDE: WHY IT WORKS (1/2)

**Original Query:**

> "Why is my app crashing on startup?"

**Hypothetical Document:**

> "Application startup failures commonly occur due to missing configuration files, incorrect database credentials, or unset environment variables. Check that all required .env values are present and that config paths are accessible."

---

## HYDE: WHY IT WORKS (2/2)

**The Magic:**

- Hypothetical uses document vocabulary
- "crashing" becomes "startup failures"
- "why" becomes specific causes
- Search now finds relevant docs

**Result:**

Retrieval matches document style, not user style

---

## HYDE: ADVANTAGES (1/2)

**Superpower 1: Vocabulary Bridge**

- Translates user language to document language
- Handles synonyms automatically
- Works across domains

**Superpower 2: Intent Expansion**

- Short queries get expanded
- Implicit context made explicit
- Better retrieval for vague questions

---

## HYDE: ADVANTAGES (2/2)

**Superpower 3: No Training Required**

- Works with any LLM
- No fine-tuning needed
- Immediate deployment

**Bottom Line:**

Bridges the gap between user questions and document vocabulary

---

## HYDE: TRADE-OFFS

| Benefit                            | Cost                          |
| ---------------------------------- | ----------------------------- |
| Better retrieval for vague queries | Extra LLM call                |
| Bridges vocabulary gap             | Added latency (200-500ms)     |
| Works without training             | Hypothetical might be wrong   |
| Handles short queries              | Overkill for explicit queries |

**When NOT to Use HyDE:**

- Detailed, explicit queries (embedding works fine)
- High-volume, latency-sensitive applications
- When hypothetical accuracy is unreliable

---

## WHAT IS SELF-CORRECTIVE RAG?

A technique that reflects on retrieval quality and retries if needed.

**Core Insight:**
Sometimes the first search fails. Self-corrective RAG asks: "Did I get good documents?" and tries again if not.

**The Mental Model:**
A research assistant who checks their sources before answering, and goes back to the library if something is missing.

---

## SELF-CORRECTIVE RAG: THE LOOP

```
┌─────────┐    ┌──────────┐    ┌──────────┐
│ QUERY   │ →  │  SEARCH  │ →  │ REFLECT  │
└─────────┘    └──────────┘    └────┬─────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
               ┌──────────┐                  ┌──────────┐
               │ SUFFICIENT│                  │ MISSING  │
               │  Answer   │                  │Reformulate│
               └──────────┘                  └────┬─────┘
                                                  │
                                            (loop back)
```

---

## SELF-CORRECTIVE RAG: FOUR COMPONENTS (1/2)

**1. Search:** Standard vector retrieval

**2. Reflect:** LLM judges retrieval quality

```python
reflection = llm.generate(f"""
Given question: {query}
Retrieved context: {context}
Is this context sufficient to answer? YES or NO.
If NO, what is missing?
""")
```

---

## SELF-CORRECTIVE RAG: FOUR COMPONENTS (2/2)

**3. Reformulate:** Improve query based on reflection

```python
new_query = llm.generate(f"""
Original: {query}
Missing: {reflection.missing}
Write an improved query to find the missing information.
""")
```

**4. Generate:** Create answer when sufficient

---

## SELF-CORRECTIVE RAG: IMPLEMENTATION (1/2)

```python
def self_corrective_rag(query: str, max_iters: int = 3) -> str:
    current_query = query

    for i in range(max_iters):
        # Search
        results = vector_store.search(embed(current_query), top_k=3)
        context = "\n\n".join([doc.text for doc in results])

        # Reflect
        reflection = reflect(query, context)
```

---

## SELF-CORRECTIVE RAG: IMPLEMENTATION (2/2)

```python
        if reflection.sufficient:
            return generate_answer(query, context)

        # Reformulate
        current_query = reformulate(query, reflection.missing)

    # Max iterations reached, answer with best effort
    return generate_answer(query, context)
```

**Loop Control:**

Max iterations prevents infinite loops

---

## SELF-CORRECTIVE RAG: REAL EXAMPLE (1/2)

**Query:** "How do I configure SSL certificates for production deployment?"

**Iteration 1:**

- Retrieved: `ssl_formats`, `rotation_policy`, `dev_setup`
- Reflection: "NO - Missing production-specific configuration steps"
- Reformulated: "SSL certificates production deployment fullchain.pem TLS_CERT_PATH environment variables"

---

## SELF-CORRECTIVE RAG: REAL EXAMPLE (2/2)

**Iteration 2:**

- Retrieved: `prod_ssl_steps`, `ssl_formats`, `container_port`
- Reflection: "YES - Contains step-by-step production setup"
- Answer: Detailed production SSL configuration

**Outcome:**

Self-correction found the right document on second try

---

## SELF-CORRECTIVE RAG: ADVANTAGES (1/2)

**Superpower 1: Self-Healing Retrieval**

- Detects retrieval failures
- Automatically retries with better query
- Converges on relevant documents

**Superpower 2: Explainable Process**

- Reflection provides reasoning trace
- Debug why retrieval succeeded/failed
- Understand system behavior

---

## SELF-CORRECTIVE RAG: ADVANTAGES (2/2)

**Superpower 3: Handles Complex Queries**

- Multi-faceted questions get multiple shots
- Specific requirements surface through iteration
- Better for troubleshooting queries

**Bottom Line:**

Self-corrects when first retrieval fails

---

## SELF-CORRECTIVE RAG: TRADE-OFFS

| Benefit                    | Cost                         |
| -------------------------- | ---------------------------- |
| Self-healing retrieval     | Multiple LLM calls per query |
| Better for complex queries | Higher latency (1-5 seconds) |
| Explainable process        | Can get stuck in loops       |
| Handles edge cases         | More expensive at scale      |

**When NOT to Use:**

- Simple factual queries
- High-volume, low-latency requirements
- When cost is primary constraint

---

## SIDE-BY-SIDE: THE THREE APPROACHES

| Aspect       | Naive RAG           | HyDE             | Self-Corrective       |
| ------------ | ------------------- | ---------------- | --------------------- |
| LLM Calls    | 1                   | 2                | 2-7                   |
| Latency      | ~500ms              | ~1s              | 1-5s                  |
| Best For     | Explicit queries    | Vague queries    | Complex queries       |
| Failure Mode | Vocabulary mismatch | Bad hypothetical | Loop without progress |

---

## SIDE-BY-SIDE: QUERY EXAMPLES

| Query                                       | Best Method                 | Why                 |
| ------------------------------------------- | --------------------------- | ------------------- |
| "What is PEM format?"                       | Naive                       | Exact terminology   |
| "Why won't my app start?"                   | HyDE                        | Vocabulary gap      |
| "Configure SSL for production"              | Self-Corrective             | Needs specific docs |
| "Given compliance AND infra constraints..." | Multi-hop + Self-Corrective | Multi-document      |

---

## SIDE-BY-SIDE: COST ANALYSIS

**Example (Illustrative): 1,000 Queries/Day**

| Method                       | Embedding Calls | LLM Calls | Est. Daily Cost |
| ---------------------------- | --------------- | --------- | --------------- |
| Naive                        | 1,000           | 1,000     | ~$2-5           |
| HyDE                         | 2,000           | 2,000     | ~$4-10          |
| Self-Corrective (avg 2 iter) | 2,000           | 4,000     | ~$8-20          |
| Hybrid (routed)              | 1,500           | 2,500     | ~$5-12          |

> AT 5,000 QUERIES/DAY, THE DIFFERENCE IS $300-1,000/MONTH.

---

## SIDE-BY-SIDE: QUALITY VS COST

```
Quality ▲
        │           ┌─────────────────┐
        │           │ Self-Corrective │
        │       ┌───┴─────────────────┘
        │       │
        │   ┌───┴────┐
        │   │  HyDE  │
        │   └────────┘
        │ ┌──────┐
        │ │Naive │
        │ └──────┘
        └────────────────────────────────► Cost
```

**The Trade-off:**
No free lunch. Better quality costs more.

---

## SIDE-BY-SIDE: LATENCY PROFILE

| Method          | P50   | P95   | P99  |
| --------------- | ----- | ----- | ---- |
| Naive           | 400ms | 800ms | 1.2s |
| HyDE            | 800ms | 1.5s  | 2.5s |
| Self-Corrective | 1.5s  | 4s    | 8s   |

**User Experience Impact:**

- <1s feels instant
- 1-3s feels responsive
- > 3s users notice waiting

---

## DECISION FRAMEWORK: QUESTIONS 1-2

**Q1: How explicit are user queries?**

- Explicit (exact terms) → Naive RAG
- Vague (problem descriptions) → HyDE

**Q2: Is first-retrieval accuracy critical?**

- Yes, must be right first time → Self-Corrective
- Good enough is acceptable → Naive or HyDE

---

## DECISION FRAMEWORK: QUESTION 3

**Q3: What is your latency budget?**

- < 1 second → Naive only
- < 3 seconds → Naive or HyDE
- Flexible → Any method

**User Experience Impact:**

- Sub-second: Feels instant, high satisfaction
- 1-3 seconds: Acceptable for complex queries
- 3+ seconds: Users notice, needs progress indicator

---

## DECISION FRAMEWORK: QUESTIONS 4-5

**Q4: What is your cost sensitivity?**

- Very sensitive → Naive with routing
- Moderate → Hybrid approach
- Quality over cost → Full self-corrective

**Q5: What are typical failure modes?**

- Vocabulary mismatch → HyDE
- Incomplete retrieval → Self-Corrective
- Both → Hybrid with routing

---

## DECISION FRAMEWORK: SUMMARY

```
                    ┌─────────────────┐
                    │ Is query explicit│
                    │ with doc vocab?  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
         ┌────┴────┐                   ┌────┴────┐
         │   YES   │                   │   NO    │
         └────┬────┘                   └────┬────┘
              │                             │
              ▼                             ▼
         ┌─────────┐              ┌─────────────────┐
         │ NAIVE   │              │ Is it complex/  │
         │  RAG    │              │ troubleshooting?│
         └─────────┘              └────────┬────────┘
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                       ┌────┴────┐                   ┌────┴────┐
                       │   YES   │                   │   NO    │
                       └────┬────┘                   └────┬────┘
                            │                             │
                            ▼                             ▼
                    ┌───────────────┐              ┌──────────┐
                    │SELF-CORRECTIVE│              │   HYDE   │
                    └───────────────┘              └──────────┘
```

---

## NEITHER IS UNIVERSALLY BETTER

**Naive RAG is not "bad":**

- Perfect for explicit queries
- Lowest cost and latency
- Simplest to maintain
- Right choice for many use cases

**Advanced techniques are not "always better":**

- Overkill for simple queries
- Add cost and latency
- More complexity to debug
- Can fail in their own ways

---

## PRODUCTION REALITY: HYBRID APPROACH

Real systems don't pick one method. They route dynamically.

```python
def route_query(question: str) -> str:
    """Determine which RAG method to use."""

    if is_short_and_vague(question):
        return "hyde"  # Bridge vocabulary gap

    elif is_troubleshooting(question):
        return "self_corrective"  # Need specific docs

    elif is_complex_multi_part(question):
        return "hybrid"  # Both techniques

    else:
        return "baseline"  # Direct embedding works
```

---

## PRODUCTION REALITY: ROUTING LOGIC (1/2)

**Signals for HyDE:**

- Query length < 10 words
- Contains problem words: "why", "not working", "help"
- No technical terms matching document vocabulary

**Signals for Self-Corrective:**

- Contains "how to configure", "setup", "troubleshoot"
- Multiple requirements in one query
- High-stakes queries (production, security)

---

## PRODUCTION REALITY: ROUTING LOGIC (2/2)

**Signals for Baseline:**

- Contains exact document terminology
- Direct factual questions
- Single-document answers likely

**Decision Priority:**

1. Check if high-stakes or troubleshooting → Self-Corrective
2. Check if vocabulary mismatch likely → HyDE
3. Default to Baseline for efficiency

---

## PRODUCTION REALITY: COST TRACKING

```python
@dataclass
class CostTracker:
    embedding_calls: int = 0
    chat_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def estimate_cost(self) -> float:
        embed_cost = self.embedding_calls * 0.0001
        input_cost = self.total_input_tokens * 0.001 / 1000
        output_cost = self.total_output_tokens * 0.003 / 1000
        return embed_cost + input_cost + output_cost
```

---

## PRODUCTION REALITY: WHAT TO TRACK

**Per Query Metrics:**

- Which method was used
- Number of iterations (self-corrective)
- Total tokens consumed
- Final quality score

**Aggregate Metrics:**

- Cost per method type
- Success rates by routing decision
- Quality trends over time

---

## PRODUCTION REALITY: KEY METRICS

**Track These Metrics:**

1. Queries by routing decision
2. Average iterations for self-corrective
3. Cost per query by method
4. Quality scores by method
5. Latency percentiles

---

## PRODUCTION REALITY: ALERTS

**Set Alerts For:**

- Self-corrective hitting max iterations > 20%
- Average cost per query spike
- Quality score drops below threshold

**Goal:**

Catch degradation before users notice

---

## PITFALLS: NAIVE RAG

| Symptom              | Cause               | Solution                    |
| -------------------- | ------------------- | --------------------------- |
| Wrong docs retrieved | Vocabulary mismatch | Add HyDE or synonyms        |
| Generic answers      | Context too broad   | Improve chunking strategy   |
| Missing key info     | Top-K too low       | Increase K or use reranking |
| Hallucinated facts   | Weak grounding      | Add faithfulness checks     |

---

## PITFALLS: NAIVE RAG (CONTINUED)

**Symptom:** High retrieval but low answer quality

**Causes:**

1. Retrieved docs tangentially related
2. LLM ignoring context
3. Context window overflow

**Solutions:**

1. Add relevance filtering after retrieval
2. Use stronger system prompts
3. Implement context compression

---

## PITFALLS: HYDE

| Symptom                 | Cause                 | Solution                             |
| ----------------------- | --------------------- | ------------------------------------ |
| Worse than baseline     | Bad hypotheticals     | Improve hypo prompt                  |
| High latency            | Long hypotheticals    | Limit generation length              |
| Wrong domain vocabulary | Generic LLM knowledge | Few-shot examples in prompt          |
| Inconsistent results    | Hypothetical variance | Temperature = 0, or average multiple |

---

## PITFALLS: HYDE (CONTINUED)

**Symptom:** Hypothetical confidently wrong

**Causes:**

1. LLM hallucinating domain knowledge
2. Question ambiguous, hypo goes wrong direction
3. No grounding in actual documents

**Solutions:**

1. Add domain context to hypo prompt
2. Generate multiple hypotheticals, average embeddings
3. Combine with baseline (search with both)

---

## PITFALLS: SELF-CORRECTIVE

| Symptom                    | Cause                  | Solution                                    |
| -------------------------- | ---------------------- | ------------------------------------------- |
| Always hits max iterations | Reflection too strict  | Adjust sufficiency criteria                 |
| Never self-corrects        | Reflection too lenient | Tighten requirements                        |
| Loops without progress     | Same docs retrieved    | Track and exclude seen docs                 |
| High cost                  | Too many iterations    | Lower max_iters, route simpler queries away |

---

## PITFALLS: SELF-CORRECTIVE (CONTINUED)

**Symptom:** Reformulated query worse than original

**Causes:**

1. Reflection identifies wrong gap
2. Reformulation adds irrelevant keywords
3. Query becomes too specific

**Solutions:**

1. Improve reflection prompt with examples
2. Keep original query terms, only add
3. Cap reformulation additions

---

## EVALUATION IS NON-NEGOTIABLE

You cannot improve what you cannot measure.

**Three Core Metrics:**

1. **Context Relevance:** Are retrieved docs useful?
2. **Faithfulness:** Is answer grounded in context?
3. **Answer Relevance:** Does answer address question?

> SHIP NOTHING WITHOUT EVALUATION INFRASTRUCTURE.

---

## EVALUATION: CONTEXT RELEVANCE

```python
def context_relevance(question: str, contexts: list) -> float:
    """Score: What fraction of retrieved docs are relevant?"""

    relevant_count = 0
    for ctx in contexts:
        prompt = f"""Question: {question}
        Document: {ctx}
        Is this document useful for answering the question?
        Answer YES or NO only."""

        if llm.generate(prompt).strip() == "YES":
            relevant_count += 1

    return relevant_count / len(contexts)
```

---

## EVALUATION: CONTEXT RELEVANCE INTERPRETATION

**High Score (>0.7):**

- Good retrieval
- Efficient context use
- System working as intended

**Low Score (<0.4):**

- Wasted context window
- Retrieval failing
- Consider HyDE or better embeddings

---

## EVALUATION: FAITHFULNESS

```python
def faithfulness(answer: str, contexts: list) -> float:
    """Score: What fraction of answer claims are supported?"""

    # Extract claims from answer
    claims = extract_claims(answer)

    supported = 0
    for claim in claims:
        prompt = f"""Claim: {claim}
        Context: {contexts}
        Is this claim supported by the context?
        Answer YES or NO only."""

        if llm.generate(prompt).strip() == "YES":
            supported += 1

    return supported / len(claims)
```

---

## EVALUATION: FAITHFULNESS INTERPRETATION

**High Score (>0.8):**

- Answer grounded in sources
- Low hallucination risk
- Safe for production

**Low Score (<0.6):**

- Hallucination detected
- Strengthen grounding prompts
- Filter unreliable answers

---

## EVALUATION: ANSWER RELEVANCE

```python
def answer_relevance(question: str, answer: str) -> float:
    """Score: Does the answer address the question?"""

    prompt = f"""Question: {question}
    Answer: {answer}

    Rate how well the answer addresses the question.
    Score from 0.0 (completely irrelevant) to 1.0 (perfect).
    Return only the number."""

    return float(llm.generate(prompt).strip())
```

---

## EVALUATION: ANSWER RELEVANCE INTERPRETATION

**High Score (>0.7):**

- Answer is on-topic and useful
- Addresses user's question directly

**Low Score (<0.5):**

- Answer misses the point
- May need better retrieval or generation prompts

---

## EVALUATION: THE FULL PIPELINE (1/2)

```python
@dataclass
class RAGTestCase:
    query: str
    expected_answer: str
    category: str  # "configuration", "troubleshooting", etc.

def evaluate_rag_method(method, test_cases):
    results = []
    for tc in test_cases:
        answer, context = method(tc.query)
```

---

## EVALUATION: THE FULL PIPELINE (2/2)

```python
        results.append({
            "query": tc.query,
            "context_relevance": context_relevance(tc.query, context),
            "faithfulness": faithfulness(answer, context),
            "answer_relevance": answer_relevance(tc.query, answer),
            "category": tc.category
        })

    return aggregate_results(results)
```

**Run This:**

On every code change, before deployment

---

## COST AND LATENCY REALITY

**Example (Illustrative): Mistral AI Pricing**

| Resource                    | Approximate Cost |
| --------------------------- | ---------------- |
| Embedding (per 1K tokens)   | $0.0001          |
| Chat Input (per 1K tokens)  | $0.001           |
| Chat Output (per 1K tokens) | $0.003           |

**Per Query Estimates:**

- Naive RAG: ~$0.002-0.005
- HyDE: ~$0.004-0.010
- Self-Corrective (2 iter): ~$0.008-0.020

---

## COST REALITY: AT SCALE

**Example (Illustrative): 10,000 Queries/Day**

| Approach                 | Daily Cost | Monthly Cost |
| ------------------------ | ---------- | ------------ |
| All Naive                | $20-50     | $600-1,500   |
| All HyDE                 | $40-100    | $1,200-3,000 |
| All Self-Corrective      | $80-200    | $2,400-6,000 |
| Smart Routing (60/30/10) | $35-80     | $1,000-2,400 |

**Smart Routing Distribution:**

- 60% Naive (simple queries)
- 30% HyDE (vague queries)
- 10% Self-Corrective (complex queries)

---

## LATENCY REALITY: COMPONENT BREAKDOWN

| Component              | Latency    |
| ---------------------- | ---------- |
| Embedding API call     | 50-150ms   |
| Vector search          | 10-50ms    |
| LLM generation (short) | 200-500ms  |
| LLM generation (long)  | 500-2000ms |

**Method Totals:**

- Naive: 300-700ms (1 embed + 1 search + 1 generate)
- HyDE: 600-1500ms (1 generate + 1 embed + 1 search + 1 generate)
- Self-Corrective (2 iter): 1-4 seconds

---

## SCALING REALITY: VECTOR STORE (1/2)

**ChromaDB (This Project):**

- Good for: Development, small-medium scale
- Persistent storage on disk
- Simple API, easy to debug

**Production Alternatives:**

- Pinecone: Managed, scales automatically
- Weaviate: Self-hosted or cloud, rich features
- Qdrant: High performance, filtering
- pgvector: If you already use Postgres

---

## SCALING REALITY: VECTOR STORE (2/2)

**Scaling Considerations:**

- Index size (documents in millions?)
- Query throughput (QPS requirements?)
- Filtering needs (metadata queries?)

**Migration Path:**

1. Start with ChromaDB for development
2. Benchmark with production query patterns
3. Evaluate managed vs self-hosted based on ops capacity
4. Plan migration with zero-downtime strategy

---

## CASE STUDY: SSL KNOWLEDGE BASE

**Context:**
Internal documentation system for SSL/TLS configuration.
7 documents covering formats, policies, and procedures.

**The Problem:**
Engineers ask questions in problem-space language.
Documents written in solution-space language.

---

## CASE STUDY: THE ARCHITECTURE

**System Layers:**

1. **Streamlit UI** - Debug mode with routing visualization
2. **Query Router** - Classifies into baseline/HyDE/self-corrective
3. **Three RAG Methods** - Baseline, HyDE, Self-Corrective
4. **ChromaDB Vector Store** - Mistral embeddings, persisted

**Flow:**

User Query → Router → Selected Method → Vector Store → Answer

---

## CASE STUDY: RESULTS

**Key Metrics Improvement:**
| Metric | Before | After |
|--------|--------|-------|
| Context Relevance | 0.45 | 0.85 |
| Faithfulness | 0.70 | 0.90 |
| Answer Relevance | 0.55 | 0.88 |

---

## CASE STUDY: COST IMPACT

**Example (Illustrative):**

- Query volume: 500/day
- Before: All self-corrective (overkill)
- After: Smart routing

| Metric        | Before | After |
| ------------- | ------ | ----- |
| Daily cost    | $10    | $4    |
| Avg latency   | 3.5s   | 1.2s  |
| Quality score | 0.82   | 0.85  |

**Key Insight:**
Quality improved while cost decreased through smart routing.

---

## CASE STUDY: KEY INSIGHT

> ROUTING IS THE MULTIPLIER.

The best technique for each query type outperforms any single technique for all queries.

**What We Learned:**

1. 60% of queries work fine with naive RAG
2. 30% benefit from HyDE (vocabulary gap)
3. 10% need self-corrective (complex requirements)
4. Forcing advanced techniques everywhere wastes money

---

## GETTING STARTED CHECKLIST

**Phase 1: Foundation**

- [ ] Set up vector store (ChromaDB for dev)
- [ ] Implement naive RAG baseline
- [ ] Create evaluation test suite (10-20 cases)
- [ ] Establish quality metrics baseline

---

## GETTING STARTED CHECKLIST (CONTINUED)

**Phase 2: Advanced Techniques**

- [ ] Implement HyDE
- [ ] Implement self-corrective RAG
- [ ] Build query router
- [ ] Compare quality metrics across methods

---

## GETTING STARTED CHECKLIST (CONTINUED)

**Phase 3: Production Hardening**

- [ ] Add cost tracking
- [ ] Implement latency monitoring
- [ ] Set up quality dashboards
- [ ] Create alerting for degradation

---

## GETTING STARTED CHECKLIST (CONTINUED)

**Phase 4: Iteration**

- [ ] Analyze routing decisions
- [ ] Tune routing thresholds
- [ ] Expand test suite with production queries
- [ ] A/B test routing changes

---

## COMMON MISTAKES TO AVOID (1/4)

**Mistake 1: Skipping Naive RAG**

- "Let's just use the advanced stuff"
- Reality: You need a baseline to compare against
- Fix: Always implement and measure naive first

**Mistake 2: No Evaluation Infrastructure**

- "We'll eyeball the results"
- Reality: Quality degrades silently
- Fix: Automated evaluation from day one

---

## COMMON MISTAKES TO AVOID (2/4)

**Mistake 3: One-Size-Fits-All**

- "Self-corrective for everything!"
- Reality: Overkill costs money, adds latency
- Fix: Route based on query characteristics

**Mistake 4: Ignoring Cost**

- "It's just API calls"
- Reality: At scale, costs compound fast
- Fix: Track cost per query, set budgets

---

## COMMON MISTAKES TO AVOID (3/4)

**Mistake 5: Complex Before Simple**

- "Let's add multi-hop, agents, and reranking"
- Reality: Debug nightmare, unclear improvement
- Fix: Add complexity incrementally, measure each addition

**Mistake 6: No Failure Analysis**

- "It works most of the time"
- Reality: Edge cases erode trust
- Fix: Log and review failures weekly

---

## COMMON MISTAKES TO AVOID (4/4)

**Mistake 7: Trusting Embeddings Blindly**

- Reality: Embeddings capture similarity, not relevance
- Fix: Add reranking or LLM-based filtering

**Mistake 8: Forgetting the User**

- Reality: Users want correct answers, fast
- Fix: Optimize for user outcomes, not architecture

---

## RESOURCES: THIS PROJECT

**GitHub Repository:**

https://github.com/virajz/advanced-rag-lecture

**What's Included:**

- Complete implementations (naive, HyDE, self-corrective)
- Evaluation framework with metrics
- Interactive Streamlit demo

---

## RESOURCES: PAPERS

**Key Papers:**

- HyDE: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al.)
- RAGAS: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"

**Tools:**

- ChromaDB: https://www.trychroma.com
- Mistral AI: https://mistral.ai
- LangChain: https://langchain.com

---

## RESOURCES: FURTHER READING

**Research Papers:**

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.)
- "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al.)
- "Lost in the Middle" paper on context window limitations

---

## RESOURCES: COMMUNITY

**Join the Discussion:**

- r/LocalLLaMA for open-source RAG discussions
- Discord communities for LangChain, LlamaIndex

**Learn More:**

Active communities with practical RAG implementations

---

## FINAL THOUGHT

> RAG IS NOT A SINGLE TECHNIQUE. IT IS A SPECTRUM OF TRADE-OFFS.

The right answer is rarely "always use X."

The right answer is usually "measure, route, iterate."

Start simple. Add complexity only when measurement justifies it.

Ship evaluation first. Ship features second.

---

## GITHUB REPOSITORY

**Full Code Available At:**

https://github.com/virajz/advanced-rag-lecture

**What You Will Find:**

- Complete naive, HyDE, and self-corrective implementations
- Evaluation framework with RAGAS-style metrics
- Production routing patterns
- Interactive Streamlit demo with debug mode
- Step-by-step lecture modules

---

## GET STARTED NOW

**Clone and Run:**

```bash
git clone https://github.com/virajz/advanced-rag-lecture.git
cd advanced-rag-lecture
uv sync
export MISTRAL_API_KEY="your-key"
uv run python lecture/01_setup.py
```

**Ready in 5 Minutes:**

All code is production-ready and documented

---

## QnA

Questions?

---
