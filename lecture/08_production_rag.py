"""
Step 8: Production RAG Patterns
================================

This script implements the production architecture from the lecture:
"Composable RAG with selective use of advanced techniques."

Key insights from the lecture:
1. "You don't need all techniques all the time"
2. "Use HyDE selectively based on query characteristics"
3. "Use evaluation framework to determine which queries benefit most"
4. "Route intelligently based on query type"

This demonstrates:
- Feature flags for HyDE and Self-Correction
- Intelligent query routing
- Cost tracking
- Metrics logging

Run: uv run python lecture/08_production_rag.py
"""

import os
import time
import numpy as np
from dataclasses import dataclass, field
from dotenv import load_dotenv
from mistralai import Mistral

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS
from lecture.vector_store import get_vector_store, get_mistral_embeddings

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def llm_call_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Make LLM call with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise
    return ""


# =============================================================================
# COST TRACKING
# =============================================================================

@dataclass
class CostTracker:
    """
    Track API calls for cost analysis.

    From lecture: "At 5000 queries/day, that's the difference between
    $300/month and over $1000/month in API costs."
    """
    embedding_calls: int = 0
    chat_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Approximate costs (adjust based on actual pricing)
    EMBED_COST_PER_1K = 0.0001  # $0.0001 per 1K tokens
    CHAT_COST_PER_1K_INPUT = 0.001  # $0.001 per 1K input tokens
    CHAT_COST_PER_1K_OUTPUT = 0.003  # $0.003 per 1K output tokens

    def log_embedding(self, num_texts: int, avg_tokens: int = 50):
        self.embedding_calls += 1
        self.total_input_tokens += num_texts * avg_tokens

    def log_chat(self, input_tokens: int = 500, output_tokens: int = 200):
        self.chat_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def estimate_cost(self) -> float:
        embed_cost = (self.total_input_tokens / 1000) * self.EMBED_COST_PER_1K
        chat_input_cost = (self.total_input_tokens / 1000) * self.CHAT_COST_PER_1K_INPUT
        chat_output_cost = (self.total_output_tokens / 1000) * self.CHAT_COST_PER_1K_OUTPUT
        return embed_cost + chat_input_cost + chat_output_cost

    def summary(self) -> str:
        return f"Embedding calls: {self.embedding_calls}, Chat calls: {self.chat_calls}, Est. cost: ${self.estimate_cost():.4f}"


# =============================================================================
# METRICS TRACKER
# =============================================================================

@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    method: str  # "baseline", "hyde", "self_corrective", "hybrid"
    latency_ms: float
    api_calls: int
    retrieved_ids: list
    answer_length: int


@dataclass
class MetricsTracker:
    """
    From lecture: "Always track metrics in production"
    """
    queries: list = field(default_factory=list)

    def log(self, metrics: QueryMetrics):
        self.queries.append(metrics)

    def summary(self):
        if not self.queries:
            return "No queries logged"

        by_method = {}
        for q in self.queries:
            if q.method not in by_method:
                by_method[q.method] = []
            by_method[q.method].append(q)

        summary = []
        for method, queries in by_method.items():
            avg_latency = np.mean([q.latency_ms for q in queries])
            avg_calls = np.mean([q.api_calls for q in queries])
            summary.append(f"{method}: {len(queries)} queries, avg latency {avg_latency:.0f}ms, avg API calls {avg_calls:.1f}")

        return "\n".join(summary)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(vec1, vec2) -> float:
    a, b = np.array(vec1), np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================================================
# PRODUCTION RAG CLASS
# =============================================================================

class ProductionRAG:
    """
    From lecture: "Composable RAG architecture with selective use
    of advanced techniques. This is build-break-scale methodology."

    Uses ChromaDB for persistent vector storage to avoid re-embedding
    documents on every run and reduce API rate limit issues.
    """

    def __init__(self, config: dict, collection_name: str = "production_rag"):
        self.enable_hyde = config.get("enable_hyde", False)
        self.enable_self_correction = config.get("enable_self_correction", False)
        self.enable_smart_routing = config.get("enable_smart_routing", True)
        self.max_correction_iterations = config.get("max_iterations", 2)

        self.cost_tracker = CostTracker()
        self.metrics_tracker = MetricsTracker()

        # Use ChromaDB vector store
        config_name = config.get("name", "default").lower().replace(" ", "_")
        self.vector_store = get_vector_store(f"{collection_name}_{config_name}")

    def index_documents(self, docs: list[dict]):
        """Index documents into the vector store (only indexes new docs)."""
        self.vector_store.add_documents(docs)

    def query(self, user_question: str, verbose: bool = False) -> dict:
        """
        Main query method with intelligent routing.

        From lecture: "Use HyDE selectively based on query characteristics.
        Not every query needs the expensive approach."
        """
        start_time = time.time()
        api_calls = 0

        # Determine method based on config and query
        if self.enable_smart_routing:
            method = self._route_query(user_question)
        elif self.enable_hyde:
            method = "hyde"
        elif self.enable_self_correction:
            method = "self_corrective"
        else:
            method = "baseline"

        if verbose:
            print(f"Routing to: {method}")

        # Execute the appropriate method
        if method == "hyde":
            result, api_calls = self._hyde_query(user_question, verbose)
        elif method == "self_corrective":
            result, api_calls = self._self_corrective_query(user_question, verbose)
        elif method == "hybrid":
            result, api_calls = self._hybrid_query(user_question, verbose)
        else:
            result, api_calls = self._baseline_query(user_question, verbose)

        # Log metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics = QueryMetrics(
            query=user_question,
            method=method,
            latency_ms=latency_ms,
            api_calls=api_calls,
            retrieved_ids=[c["id"] for c in result["contexts"]],
            answer_length=len(result["answer"])
        )
        self.metrics_tracker.log(metrics)

        result["method"] = method
        result["latency_ms"] = latency_ms
        result["api_calls"] = api_calls

        return result

    def _route_query(self, question: str) -> str:
        """
        From lecture: "If you detect a short, vague question (under 10 words
        with question markers), route it through HyDE. If detailed query with
        specific technical terms, use direct embedding."
        """
        words = question.split()
        is_short = len(words) < 10
        is_question = "?" in question or any(
            question.lower().startswith(w) for w in ["how", "what", "why", "when", "where"]
        )
        has_technical_terms = any(
            term in question.lower()
            for term in ["ssl", "tls", "certificate", "port", "https", "pem", "config"]
        )

        # Decision logic
        if is_short and is_question and not has_technical_terms:
            # Vague question - HyDE helps bridge vocabulary gap
            return "hyde"
        elif "troubleshoot" in question.lower() or "fix" in question.lower() or "error" in question.lower():
            # Troubleshooting - self-correction helps find specific docs
            return "self_corrective"
        elif is_short and is_question:
            # Short question with technical terms - try hybrid
            return "hybrid" if self.enable_hyde and self.enable_self_correction else "baseline"
        else:
            # Detailed query - direct embedding works fine
            return "baseline"

    def _baseline_query(self, question: str, verbose: bool) -> tuple[dict, int]:
        """Standard RAG: embed -> retrieve -> generate"""
        self.cost_tracker.log_embedding(1)
        contexts = self.vector_store.search(question, top_k=3)
        time.sleep(0.3)  # Rate limit protection

        answer = self._generate(question, contexts)
        return {"query": question, "contexts": contexts, "answer": answer}, 2  # 1 embed + 1 generate

    def _hyde_query(self, question: str, verbose: bool) -> tuple[dict, int]:
        """HyDE: generate hypothetical -> embed -> retrieve -> generate"""
        # Generate hypothetical document
        hypo_prompt = f"""Write a short technical document (2-3 sentences) that answers:
{question}
Write in documentation style, not as a response."""

        self.cost_tracker.log_chat()
        hypothetical = llm_call_with_retry(hypo_prompt)
        time.sleep(0.3)

        if verbose:
            print(f"  Hypothetical: {hypothetical[:100]}...")

        # Embed hypothetical and search
        self.cost_tracker.log_embedding(1)
        hypo_embedding = get_mistral_embeddings([hypothetical])[0]
        contexts = self.vector_store.search_with_embedding(hypo_embedding, top_k=3)

        # Generate answer
        answer = self._generate(question, contexts)

        return {
            "query": question,
            "hypothetical": hypothetical,
            "contexts": contexts,
            "answer": answer
        }, 3  # 1 hypo gen + 1 embed + 1 answer gen

    def _self_corrective_query(self, question: str, verbose: bool) -> tuple[dict, int]:
        """Self-corrective: retrieve -> reflect -> (reformulate -> retrieve)* -> generate"""
        api_calls = 0
        current_query = question

        for iteration in range(self.max_correction_iterations):
            # Retrieve
            self.cost_tracker.log_embedding(1)
            contexts = self.vector_store.search(current_query, top_k=3)
            api_calls += 1
            time.sleep(0.3)

            # Reflect
            reflection = self._reflect(question, contexts)
            api_calls += 1
            time.sleep(0.3)

            if verbose:
                print(f"  Iteration {iteration + 1}: {reflection['decision']} - {reflection['reason'][:50]}...")

            if reflection["decision"] == "YES":
                break

            # Reformulate if not last iteration
            if iteration < self.max_correction_iterations - 1:
                current_query = self._reformulate(current_query, reflection["reason"])
                api_calls += 1
                time.sleep(0.3)
                if verbose:
                    print(f"  Reformulated: {current_query[:50]}...")

        # Generate answer
        answer = self._generate(question, contexts)
        api_calls += 1

        return {
            "query": question,
            "final_query": current_query,
            "iterations": iteration + 1,
            "contexts": contexts,
            "answer": answer
        }, api_calls

    def _hybrid_query(self, question: str, verbose: bool) -> tuple[dict, int]:
        """Hybrid: HyDE + Self-correction for maximum quality"""
        api_calls = 0

        # First, use HyDE for better initial retrieval
        hypo_prompt = f"Write a technical document answering: {question}"
        self.cost_tracker.log_chat()
        hypothetical = llm_call_with_retry(hypo_prompt)
        api_calls += 1
        time.sleep(0.3)

        self.cost_tracker.log_embedding(1)
        hypo_embedding = get_mistral_embeddings([hypothetical])[0]
        api_calls += 1

        contexts = self.vector_store.search_with_embedding(hypo_embedding, top_k=3)

        # Then, reflect on quality
        reflection = self._reflect(question, contexts)
        api_calls += 1
        time.sleep(0.3)

        if verbose:
            print(f"  HyDE + Reflect: {reflection['decision']}")

        # If still not good enough, do one more retrieval with reformulated query
        if reflection["decision"] == "NO":
            reformulated = self._reformulate(question, reflection["reason"])
            api_calls += 1
            time.sleep(0.3)

            self.cost_tracker.log_embedding(1)
            contexts = self.vector_store.search(reformulated, top_k=3)
            api_calls += 1

        answer = self._generate(question, contexts)
        api_calls += 1

        return {
            "query": question,
            "hypothetical": hypothetical,
            "contexts": contexts,
            "answer": answer
        }, api_calls

    def _generate(self, question: str, contexts: list[dict]) -> str:
        """Generate answer from context."""
        context_text = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in contexts])
        prompt = f"""Answer based ONLY on the context.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        self.cost_tracker.log_chat()
        return llm_call_with_retry(prompt)

    def _reflect(self, question: str, contexts: list[dict]) -> dict:
        """Reflect on retrieval quality."""
        context_text = "\n".join([f"- {c['text']}" for c in contexts])
        prompt = f"""Can these documents answer the question completely?
QUESTION: {question}
DOCUMENTS:
{context_text}

Respond: DECISION: YES or NO
REASON: <one sentence>"""

        self.cost_tracker.log_chat()
        text = llm_call_with_retry(prompt)
        decision = "YES" if "DECISION: YES" in text.upper() else "NO"
        reason = text.split("REASON:")[-1].strip() if "REASON:" in text.upper() else ""
        return {"decision": decision, "reason": reason}

    def _reformulate(self, question: str, reason: str) -> str:
        """Reformulate query based on reflection."""
        prompt = f"""Improve this search query.
ORIGINAL: {question}
PROBLEM: {reason}
Return only the improved query."""

        self.cost_tracker.log_chat()
        return llm_call_with_retry(prompt).strip()

    def get_stats(self) -> str:
        """Get usage statistics."""
        return f"""
=== Production RAG Stats ===
{self.metrics_tracker.summary()}

Cost Tracker:
{self.cost_tracker.summary()}
"""


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    print("=" * 70)
    print("Production RAG with Intelligent Routing")
    print("=" * 70)
    print("\nUsing ChromaDB for persistent vector storage.")
    print("(Documents are only embedded once - subsequent runs reuse cached embeddings)")

    # Test queries of different types
    test_queries = [
        "How do I configure SSL certificates for production?",  # Technical, direct
        "Why is my app not working?",  # Vague, needs HyDE
        "How do I fix certificate errors?",  # Troubleshooting, needs self-correction
    ]

    # Compare different configurations (reduced for faster demo)
    configs = [
        {"name": "Baseline Only", "enable_hyde": False, "enable_self_correction": False, "enable_smart_routing": False},
        {"name": "Smart Routing", "enable_hyde": True, "enable_self_correction": True, "enable_smart_routing": True},
    ]

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Configuration: {config['name']}")
        print("=" * 70)

        rag = ProductionRAG(config)
        print("\nIndexing documents...")
        rag.index_documents(DOCS)
        print(f"Documents in store: {rag.vector_store.count()}")

        for query in test_queries:
            print(f"\nQuery: {query}")
            result = rag.query(query, verbose=True)
            print(f"  Method: {result['method']}")
            print(f"  Latency: {result['latency_ms']:.0f}ms")
            print(f"  API Calls: {result['api_calls']}")
            print(f"  Answer: {result['answer'][:100]}...")
            time.sleep(0.5)  # Rate limit protection between queries

        print(rag.get_stats())

    # Cost comparison summary
    print("\n" + "=" * 70)
    print("COST-QUALITY TRADEOFF ANALYSIS (from lecture)")
    print("=" * 70)
    print("""
API Calls per Query:
- Baseline RAG:           2 calls (1 embed + 1 generate)
- HyDE:                   3 calls (1 hypo gen + 1 embed + 1 generate)
- Self-Corrective (2x):   5 calls (2 embed + 2 reflect + 1 generate)
- Hybrid (worst case):    7 calls

At 5,000 queries/day (from lecture):
- Baseline:       ~$300/month
- Full Advanced:  ~$1,000+/month

RECOMMENDATION:
"Use evaluation framework to determine which queries benefit most
from which techniques, then route intelligently."

Smart routing applies expensive techniques only where they help,
optimizing the cost-quality tradeoff automatically.

NOTE: With ChromaDB, document embeddings are cached on disk.
This significantly reduces embedding API calls during development
and protects against rate limits.
""")


if __name__ == "__main__":
    main()
