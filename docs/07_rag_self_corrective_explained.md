# rag_self_corrective.py - The "Check Your Work" System

## What is this file?

This demonstrates **Self-Corrective RAG** - a system that **checks if its search results are good enough**, and if not, **tries again with a better query**. It's like a student who re-reads the question when their answer doesn't make sense!

## The Problem It Solves

Sometimes the first search doesn't get what you need:

```
❌ WITHOUT SELF-CORRECTION:
Question: "How do I configure SSL for production?"
Search → Get generic SSL docs
Answer → "SSL uses PEM files" ← Not helpful! Missing the actual steps!

✅ WITH SELF-CORRECTION:
Question: "How do I configure SSL for production?"
Search → Get generic SSL docs
Check → "Hmm, no step-by-step instructions..."
Retry → Search with better keywords
Search → Get production setup guide!
Answer → "1) Put files here 2) Set these env vars 3) Restart" ← Helpful!
```

## Visual: The Self-Correcting Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ QUESTION │───▶│  SEARCH  │───▶│ REFLECT  │              │
│  └──────────┘    └──────────┘    └────┬─────┘              │
│                                       │                     │
│                          ┌────────────┴────────────┐       │
│                          ▼                         ▼       │
│                    ┌──────────┐              ┌──────────┐  │
│                    │   YES    │              │    NO    │  │
│                    │ Good     │              │ Not good │  │
│                    │ enough!  │              │ enough!  │  │
│                    └────┬─────┘              └────┬─────┘  │
│                         │                         │        │
│                         ▼                         ▼        │
│                    ┌──────────┐              ┌──────────┐  │
│                    │  ANSWER  │              │REFORMULATE│  │
│                    │          │              │  QUERY    │──┘
│                    └──────────┘              └──────────┘
│                                              (loop back!)
└─────────────────────────────────────────────────────────────┘
```

## The Four Key Functions

### 1. reflect() - The Judge 🧑‍⚖️

```python
def reflect(question: str, contexts) -> dict:
```

**What it does:** Looks at the retrieved documents and asks: "Is this good enough to answer the question?"

**Returns:**
- `{"decision": "YES", "why": "..."}` - Good to go!
- `{"decision": "NO", "why": "..."}` - Need to search again

**Example checks:**
- Does it contain step-by-step instructions?
- Does it mention specific file paths?
- Does it have the actual config values?

### 2. reformulate() - The Query Improver ✨

```python
def reformulate(question: str, reflection_why: str) -> str:
```

**What it does:** When reflection says "NO", this creates a BETTER search query.

**Example:**
```
Original: "How do I configure SSL for production?"
     ↓
Reflection: "Missing step-by-step setup instructions"
     ↓
New query: "How do I configure SSL for production? step-by-step
           setup fullchain.pem privkey.pem TLS_CERT_PATH"
```

### 3. generate_answer_from_context() - The Answerer 💬

```python
def generate_answer_from_context(question: str, contexts) -> str:
```

**What it does:** Creates the final answer from the (hopefully good) retrieved documents.

### 4. self_corrective_rag() - The Main Loop 🔄

```python
def self_corrective_rag(question: str, max_iters: int = 2):
```

**What it does:** Orchestrates the whole process:
1. Search
2. Reflect - good enough?
3. If NO → reformulate and go back to step 1
4. If YES → generate answer

## Explained Like You're 5

Imagine you're looking for your toy car:

**Without self-correction:**
1. Look under the bed → find socks
2. Give up, say "I found socks"

**With self-correction:**
1. Look under the bed → find socks
2. Think: "Socks aren't a toy car... let me try somewhere else"
3. Look in the toy box → find the car! 🚗
4. "Found it!"

## Running It

```bash
python rag_self_corrective.py
```

**Output:**
```
Iteration 1 reflection: NO — Context talks about SSL generally but lacks
                           step-by-step production configuration (paths/env vars/ports).
Reformulated query: How do I configure SSL certificates for production deployment?
                    step-by-step setup fullchain.pem privkey.pem TLS_CERT_PATH TLS_KEY_PATH

Iteration 2 reflection: YES — Retrieved context contains explicit production setup steps.

Answer grounded in docs:
Production SSL setup steps: 1) Put fullchain.pem and privkey.pem in /etc/myapp/tls/.
2) Set TLS_CERT_PATH and TLS_KEY_PATH. 3) Set HTTPS_PORT=443. 4) Restart service.
```

## The Magic: It Fixed Itself! 🪄

| Iteration | What Happened |
|-----------|---------------|
| 1 | Found generic SSL docs → Reflected → "Not specific enough" |
| 2 | Searched with better keywords → Found step-by-step guide → "Perfect!" |

## Real-World Applications

| Scenario | How Self-Correction Helps |
|----------|---------------------------|
| Code search | "Found the function, but not the error handling part..." → retry |
| Customer support | "Found general FAQ, but customer asked about billing..." → retry |
| Research | "Found the paper, but not the methodology section..." → retry |

## Key Components Recap

```
┌─────────────────────────────────────────────────────────────┐
│                   SELF-CORRECTIVE RAG                       │
├─────────────────────────────────────────────────────────────┤
│  🔍 SEARCH      → Find documents                            │
│  🧑‍⚖️ REFLECT    → "Good enough?" (YES/NO)                  │
│  ✨ REFORMULATE → Make query better (if NO)                 │
│  💬 ANSWER      → Generate response (if YES)                │
│  🔄 LOOP        → Try up to max_iters times                 │
└─────────────────────────────────────────────────────────────┘
```

## Trade-offs ⚖️

| Pros | Cons |
|------|------|
| Better answers | More LLM calls (slower, costs more) |
| Handles vague queries | Can get stuck in loops |
| Self-improving | Needs good reflection logic |

---
*Self-Corrective RAG = "Search, check, improve, repeat until good!"*
