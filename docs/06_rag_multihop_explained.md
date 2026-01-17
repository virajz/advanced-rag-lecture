# rag_multihop_demo.py - The "Connect the Dots" Approach

## What is this file?

This demonstrates **multi-hop retrieval** - when you need information from MULTIPLE documents to answer a complex question. One search isn't enough!

## The Problem It Solves

Some questions need info from several places:

```
❌ SIMPLE QUESTION (one document enough):
"What is SSL?"
→ Just find an SSL doc ✓

❌ COMPLEX QUESTION (needs multiple docs):
"Given our compliance requirements AND infrastructure constraints,
 what SSL configuration should we use?"
→ Need: compliance doc + infrastructure doc + SSL doc
```

## Visual: Why One Search Fails

```
Question: "Given compliance requirements and infrastructure
          constraints, what SSL config should we use?"

NAIVE SEARCH (top 3 by keyword match):
┌─────────────────┐
│ 1. ssl_formats  │ ← mentions "SSL"
│ 2. prod_ssl_steps│ ← mentions "SSL", "configuration"
│ 3. rotation_policy│ ← mentions "SSL"
└─────────────────┘

❌ MISSING: compliance_req, infra_constraint docs!
   The answer is INCOMPLETE!
```

## The Multi-Hop Solution

```
IMPROVED SEARCH (expand query with related terms):

Original: "Given compliance requirements and infrastructure constraints..."
     +
Added: "compliance requires infrastructure constraint reverse proxy privileged ports"
     =
┌─────────────────┐
│ 1. compliance_req│ ← "TLS 1.2+, modern cipher suites"
│ 2. infra_constraint│ ← "only proxy can bind 443"
│ 3. prod_ssl_steps│ ← actual setup steps
│ 4. container_port│ ← "expose 8443 internally"
└─────────────────┘

✅ NOW we have ALL the pieces!
```

## The Three Functions

### 1. naive_answer() - The Bad Answer 😕

```python
def naive_answer(question: str, ctx):
```

Shows what happens with basic search - you miss important documents and give an incomplete answer.

### 2. better_retrieve_for_constraints() - The Smart Search 🧠

```python
def better_retrieve_for_constraints(question: str):
    q = question + " compliance requires infrastructure constraint reverse proxy privileged ports"
    return naive_vector_search(q, top_k=4)
```

**The trick:** Manually add keywords that should pull in related documents!

### 3. grounded_answer() - The Good Answer ✅

```python
def grounded_answer(question: str, ctx):
```

Uses ALL the retrieved docs to give a complete, grounded answer.

## Explained Like You're 5

**Naive approach:** Teacher asks "What should we pack for a camping trip in the mountains during winter?"

You search your brain for "camping" → "tent, sleeping bag"

**Multi-hop approach:** You think about ALL the requirements:
- Camping → tent, sleeping bag
- Mountains → hiking boots, map
- Winter → warm clothes, hand warmers

NOW your answer covers everything!

## The Code Flow

```python
# BAD: Simple search misses constraint documents
ctx_bad = naive_vector_search(q, top_k=3)
print(naive_answer(q, ctx_bad))
# Output: "Use port 443 and enable TLS 1.3" ← Too simple!

# GOOD: Expanded search gets all relevant docs
ctx_good = better_retrieve_for_constraints(q)
print(grounded_answer(q, ctx_good))
# Output: Complete answer with compliance + infrastructure + SSL info!
```

## Running It

```bash
python rag_multihop_demo.py
```

**Output:**
```
NAIVE:
 Retrieved: ['ssl_formats', 'prod_ssl_steps', 'rotation_policy']
 Answer: Use port 443 and enable TLS 1.3.

IMPROVED RETRIEVAL IDs: ['compliance_req', 'infra_constraint', 'prod_ssl_steps', 'container_port_note']

BETTER:
 Grounded synthesis:
 - Terminate TLS at the reverse proxy (only it can bind 443).
 - Containers expose a high port (e.g., 8443) internally.
 - Ensure TLS 1.2+ and modern cipher suites to satisfy compliance.
```

## Real-World Examples

| Complex Question | Documents Needed |
|-----------------|------------------|
| "How do I deploy to AWS with our security policies?" | AWS docs + Security policy docs + Deployment docs |
| "What's the best database for our scale and budget?" | Database comparison + Pricing docs + Scale requirements |
| "How to migrate while maintaining uptime?" | Migration guide + Uptime requirements + Rollback procedures |

## Key Insight 💡

**Multi-hop retrieval = Recognizing that complex questions have multiple facets, and making sure you retrieve documents for EACH facet.**

In production, this is often done with:
- Query decomposition (break question into sub-questions)
- Iterative retrieval (search, read, search again)
- Knowledge graphs (follow relationships between topics)

---
*Complex questions need information from multiple sources - don't stop at one search!*
