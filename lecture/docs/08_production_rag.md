# 08_production_rag.py - Production RAG Patterns

## What This File Does

This is the final lecture - putting everything together into a production-ready RAG system. It shows how to selectively use advanced techniques based on query characteristics, track costs, and make data-driven decisions.

## The Core Insight

**"You don't need all techniques all the time."**

- Simple, technical queries → baseline RAG is fine
- Vague, short queries → HyDE helps
- Troubleshooting queries → self-correction helps
- Complex queries → maybe both

Smart routing applies the right technique for each query.

## The Production RAG Class

```python
class ProductionRAG:
    def __init__(self, config):
        self.enable_hyde = config.get("enable_hyde", False)
        self.enable_self_correction = config.get("enable_self_correction", False)
        self.enable_smart_routing = config.get("enable_smart_routing", True)

        self.cost_tracker = CostTracker()
        self.metrics_tracker = MetricsTracker()
        self.vector_store = get_vector_store(...)
```

**What's happening:**
- Feature flags control which techniques are enabled
- Cost tracking monitors API usage
- Metrics tracking logs query performance
- ChromaDB provides persistent vector storage

## Smart Query Routing

```python
def _route_query(self, question: str) -> str:
    words = question.split()
    is_short = len(words) < 10
    is_question = "?" in question or question.startswith("how", "what", ...)
    has_technical_terms = any(term in question.lower()
                              for term in ["ssl", "tls", "certificate", ...])

    # Decision logic
    if is_short and is_question and not has_technical_terms:
        return "hyde"  # Vague question - HyDE helps

    elif "troubleshoot" in question.lower() or "fix" in question.lower():
        return "self_corrective"  # Troubleshooting - self-correction helps

    elif is_short and is_question:
        return "hybrid"  # Short with tech terms - try both

    else:
        return "baseline"  # Detailed query - direct embedding works
```

**The routing logic:**
1. **Short + vague + no tech terms** → HyDE (bridges vocabulary gap)
2. **Contains "fix," "error," "troubleshoot"** → Self-correction (finds specific docs)
3. **Short + has tech terms** → Hybrid (both techniques)
4. **Everything else** → Baseline (cheapest, often sufficient)

## Cost Tracking

```python
@dataclass
class CostTracker:
    embedding_calls: int = 0
    chat_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def estimate_cost(self) -> float:
        # Calculate based on API pricing
        return embed_cost + chat_cost
```

**Why track costs:**
- Know exactly what you're spending
- Compare cost of different configurations
- Make informed cost/quality tradeoffs

**API calls per query:**
```
Baseline:           2 calls (1 embed + 1 generate)
HyDE:               3 calls (1 hypo gen + 1 embed + 1 generate)
Self-Corrective:    5 calls (2 embed + 2 reflect + 1 generate)
Hybrid (worst):     7 calls
```

At 5,000 queries/day:
- Baseline: ~$300/month
- Full advanced: ~$1,000+/month

## Metrics Tracking

```python
@dataclass
class QueryMetrics:
    query: str
    method: str       # "baseline", "hyde", "self_corrective", "hybrid"
    latency_ms: float
    api_calls: int
    retrieved_ids: list
    answer_length: int
```

**What's tracked:**
- Which method was used
- How long it took
- How many API calls
- What documents were retrieved

**Why track metrics:**
- Understand which methods are used most
- Identify slow queries
- Debug retrieval issues

## The Four Query Methods

### 1. Baseline Query
```python
def _baseline_query(self, question):
    contexts = self.vector_store.search(question, top_k=3)
    answer = self._generate(question, contexts)
    return result, 2  # 2 API calls
```
Simple and cheap. Works for detailed, technical queries.

### 2. HyDE Query
```python
def _hyde_query(self, question):
    # Generate hypothetical
    hypothetical = llm_call("Write a doc answering: {question}")

    # Embed and search
    embedding = get_mistral_embeddings([hypothetical])[0]
    contexts = self.vector_store.search_with_embedding(embedding)

    answer = self._generate(question, contexts)
    return result, 3  # 3 API calls
```
Extra call to generate hypothetical, but finds better matches for vague queries.

### 3. Self-Corrective Query
```python
def _self_corrective_query(self, question):
    for iteration in range(max_iterations):
        contexts = self.vector_store.search(current_query)
        reflection = self._reflect(question, contexts)

        if reflection["decision"] == "YES":
            break

        current_query = self._reformulate(current_query, reflection["reason"])

    answer = self._generate(question, contexts)
    return result, api_calls  # 4-7 API calls
```
Iterates until good context is found. Best for troubleshooting.

### 4. Hybrid Query
```python
def _hybrid_query(self, question):
    # HyDE first
    hypothetical = generate_hypothetical(question)
    contexts = search_with_embedding(hypothetical)

    # Then reflect
    reflection = self._reflect(question, contexts)

    if reflection["decision"] == "NO":
        # Reformulate and try again
        reformulated = self._reformulate(question, reflection["reason"])
        contexts = self.vector_store.search(reformulated)

    answer = self._generate(question, contexts)
    return result, api_calls  # 5-8 API calls
```
Maximum quality but maximum cost. Use for critical queries.

## ChromaDB Integration

```python
self.vector_store = get_vector_store(f"production_rag_{config_name}")
```

**Benefits:**
- **Persistent storage** - Documents embedded once, cached on disk
- **No re-embedding on restart** - Saves API calls and time
- **Rate limit protection** - Fewer embedding calls needed
- **Fast search** - Optimized vector similarity

## Configuration Examples

### Baseline Only (Cheapest)
```python
config = {
    "name": "Baseline Only",
    "enable_hyde": False,
    "enable_self_correction": False,
    "enable_smart_routing": False
}
```

### Smart Routing (Recommended)
```python
config = {
    "name": "Smart Routing",
    "enable_hyde": True,
    "enable_self_correction": True,
    "enable_smart_routing": True
}
```
Uses cheap methods for simple queries, expensive methods only when needed.

### Full Advanced (Maximum Quality)
```python
config = {
    "name": "Full Advanced",
    "enable_hyde": True,
    "enable_self_correction": True,
    "enable_smart_routing": False  # Always use hybrid
}
```
Best answers, highest cost. Use for high-stakes applications.

## The Production Mindset

1. **Start with baseline** - It's simpler and cheaper
2. **Measure with evaluation pipeline** - Know your weak spots
3. **Add techniques selectively** - HyDE for vocabulary issues, self-correction for complex queries
4. **Use smart routing** - Apply expensive techniques only where they help
5. **Track costs and metrics** - Make data-driven decisions
6. **Use persistent storage** - ChromaDB saves time and API calls

## Key Takeaway

> "Use evaluation framework to determine which queries benefit most from which techniques, then route intelligently."

Instead of applying HyDE and self-correction to every query (expensive), or never using them (lower quality), smart routing gives you the best of both worlds.
