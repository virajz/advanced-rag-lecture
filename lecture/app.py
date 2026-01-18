"""
RAG Chat Interface with Debug Mode
===================================

A Streamlit chat interface for the Advanced RAG system.

Features:
- Normal mode: Clean chat with streamed answers
- Debug mode: Shows retrieval, routing decisions, metrics, and more
- Multiple document collections to choose from

Run: uv run streamlit run lecture/app.py
"""

import os
import sys
import time
from dataclasses import dataclass, field

import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_corpus import DOCS as SSL_DOCS
from story_corpus import DOCS as STORY_DOCS
from lecture.vector_store import get_vector_store, get_mistral_embeddings

load_dotenv()

# Initialize Mistral client
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))

# Document collections
DOCUMENT_COLLECTIONS = {
    "SSL/TLS Configuration": {
        "docs": SSL_DOCS,
        "collection_name": "chat_app_ssl",
        "description": "Technical documentation about SSL certificates, TLS configuration, and security compliance.",
        "placeholder": "Ask about SSL/TLS configuration...",
        "icon": "🔐"
    },
    "The Lost Algorithm (Story)": {
        "docs": STORY_DOCS,
        "collection_name": "chat_app_story",
        "description": "A narrative about Dr. Elena Vasquez discovering an emergent AI algorithm at the Prometheus Institute.",
        "placeholder": "Ask about the story...",
        "icon": "📖"
    }
}


# =============================================================================
# COST & METRICS TRACKING
# =============================================================================

@dataclass
class QueryDebugInfo:
    """Debug information for a single query."""
    routing_decision: str = ""
    routing_reason: str = ""
    hypothetical_doc: str = ""
    retrieved_contexts: list = field(default_factory=list)
    reflection_steps: list = field(default_factory=list)
    reformulated_queries: list = field(default_factory=list)
    api_calls: int = 0
    latency_ms: float = 0.0
    embedding_calls: int = 0
    chat_calls: int = 0


# =============================================================================
# RAG FUNCTIONS
# =============================================================================

def route_query(question: str, doc_type: str = "technical") -> tuple[str, str]:
    """
    Smart query router using LLM to analyze the query and pick the best RAG method.
    Returns (method, reason).
    """
    prompt = f"""Analyze this query and decide which RAG retrieval method would work best.

QUERY: {question}
DOCUMENT TYPE: {doc_type}

Available methods:
1. BASELINE - Direct embedding search. Best for: simple factual queries, queries using exact terminology from documents
2. HYDE - Generate hypothetical answer first, then search. Best for: vague queries, vocabulary mismatch, when user terms differ from document terms
3. SELF_CORRECTIVE - Search, reflect if results are sufficient, reformulate if needed. Best for: complex queries needing multiple documents, multi-step answers, queries where first search might miss context
4. HYBRID - Combine HyDE + self-correction. Best for: difficult queries needing both vocabulary bridging AND multiple retrieval attempts

Consider:
- Does the query use domain-specific terms that likely match documents? → BASELINE
- Is the query vague or uses different vocabulary than technical docs? → HYDE
- Does answering require information from multiple documents? → SELF_CORRECTIVE
- Is it both vague AND complex? → HYBRID

Respond in this exact format:
METHOD: <one of: BASELINE, HYDE, SELF_CORRECTIVE, HYBRID>
REASON: <brief explanation>"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content

    # Parse response
    method = "baseline"  # default
    reason = "Default fallback"

    for line in text.strip().split("\n"):
        if line.startswith("METHOD:"):
            method_str = line.replace("METHOD:", "").strip().lower()
            if method_str in ["baseline", "hyde", "self_corrective", "hybrid"]:
                method = method_str
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return method, reason


def generate_hypothetical(question: str, doc_type: str = "technical") -> str:
    """Generate a hypothetical document for HyDE."""
    if doc_type == "story":
        prompt = f"""Write a short passage (2-3 sentences) from a story that would answer this question:
{question}
Write in narrative style, as if from a novel."""
    else:
        prompt = f"""Write a short technical document (2-3 sentences) that answers:
{question}
Write in documentation style, not as a response."""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def reflect_on_retrieval(question: str, contexts: list[dict]) -> dict:
    """Reflect on whether retrieved documents are sufficient."""
    context_text = "\n".join([f"- {c['text']}" for c in contexts])
    prompt = f"""Can these documents answer the question completely?
QUESTION: {question}
DOCUMENTS:
{context_text}

Respond: DECISION: YES or NO
REASON: <one sentence>"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content
    decision = "YES" if "DECISION: YES" in text.upper() else "NO"
    reason = text.split("REASON:")[-1].strip() if "REASON:" in text.upper() else ""
    return {"decision": decision, "reason": reason}


def reformulate_query(question: str, reason: str) -> str:
    """Reformulate query based on reflection feedback."""
    prompt = f"""Improve this search query.
ORIGINAL: {question}
PROBLEM: {reason}
Return only the improved query."""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generate_answer_streaming(question: str, contexts: list[dict], doc_type: str = "technical"):
    """Generate answer with streaming."""
    context_text = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in contexts])

    if doc_type == "story":
        prompt = f"""Answer the question based ONLY on the story excerpts provided. Be helpful and reference specific details from the text.

STORY EXCERPTS:
{context_text}

QUESTION: {question}

ANSWER:"""
    else:
        prompt = f"""Answer based ONLY on the context. Be concise and helpful.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

    response = client.chat.stream(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    for chunk in response:
        if chunk.data.choices[0].delta.content:
            yield chunk.data.choices[0].delta.content


def run_rag_query(question: str, vector_store, method: str = "smart", doc_type: str = "technical") -> tuple[str, QueryDebugInfo]:
    """
    Run RAG query with the specified method.
    Returns (answer, debug_info).
    """
    debug = QueryDebugInfo()
    start_time = time.time()
    contexts = []

    # Determine routing
    if method == "smart":
        routed_method, reason = route_query(question, doc_type)
        debug.routing_decision = routed_method
        debug.routing_reason = reason
    else:
        routed_method = method
        debug.routing_decision = method
        debug.routing_reason = f"User selected {method}"

    # Execute the appropriate method
    # Baseline uses top_k=1 (naive approach: just return the best match)
    # This demonstrates why simple RAG fails for multi-part answers
    # Advanced methods use smart_search for adaptive retrieval
    if routed_method == "baseline":
        debug.embedding_calls += 1
        contexts = vector_store.search(question, top_k=1)  # Naive: only best match
        debug.api_calls += 1

    elif routed_method == "hyde":
        # Generate hypothetical document
        debug.chat_calls += 1
        hypothetical = generate_hypothetical(question, doc_type)
        debug.hypothetical_doc = hypothetical
        debug.api_calls += 1

        # Embed and search with smart retrieval
        debug.embedding_calls += 1
        hypo_embedding = get_mistral_embeddings([hypothetical])[0]
        contexts = vector_store.smart_search_with_embedding(hypo_embedding)
        debug.api_calls += 1

    elif routed_method == "self_corrective":
        current_query = question
        max_iterations = 2

        for iteration in range(max_iterations):
            debug.embedding_calls += 1
            contexts = vector_store.smart_search(current_query)
            debug.api_calls += 1

            debug.chat_calls += 1
            reflection = reflect_on_retrieval(question, contexts)
            debug.reflection_steps.append({
                "iteration": iteration + 1,
                "query": current_query,
                "decision": reflection["decision"],
                "reason": reflection["reason"]
            })
            debug.api_calls += 1

            if reflection["decision"] == "YES":
                break

            if iteration < max_iterations - 1:
                debug.chat_calls += 1
                current_query = reformulate_query(current_query, reflection["reason"])
                debug.reformulated_queries.append(current_query)
                debug.api_calls += 1

    elif routed_method == "hybrid":
        # HyDE first
        debug.chat_calls += 1
        hypothetical = generate_hypothetical(question, doc_type)
        debug.hypothetical_doc = hypothetical
        debug.api_calls += 1

        debug.embedding_calls += 1
        hypo_embedding = get_mistral_embeddings([hypothetical])[0]
        contexts = vector_store.smart_search_with_embedding(hypo_embedding)
        debug.api_calls += 1

        # Reflect
        debug.chat_calls += 1
        reflection = reflect_on_retrieval(question, contexts)
        debug.reflection_steps.append({
            "iteration": 1,
            "query": question,
            "decision": reflection["decision"],
            "reason": reflection["reason"]
        })
        debug.api_calls += 1

        # Reformulate if needed
        if reflection["decision"] == "NO":
            debug.chat_calls += 1
            reformulated = reformulate_query(question, reflection["reason"])
            debug.reformulated_queries.append(reformulated)
            debug.api_calls += 1

            debug.embedding_calls += 1
            contexts = vector_store.smart_search(reformulated)
            debug.api_calls += 1

    debug.retrieved_contexts = contexts
    debug.latency_ms = (time.time() - start_time) * 1000

    return contexts, debug


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Advanced RAG Chat",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Advanced RAG Chat")
    st.caption("Powered by Mistral AI with Smart Routing")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")

        # Document collection selector
        st.subheader("📚 Document Collection")
        selected_collection = st.selectbox(
            "Choose documents",
            list(DOCUMENT_COLLECTIONS.keys()),
            index=0,
            help="Select which document collection to query"
        )

        collection_info = DOCUMENT_COLLECTIONS[selected_collection]
        st.caption(f"{collection_info['icon']} {collection_info['description']}")

        st.divider()

        debug_mode = st.toggle("🐛 Debug Mode", value=False, help="Show detailed RAG pipeline information")

        st.divider()

        st.subheader("RAG Configuration")
        rag_method = st.selectbox(
            "Routing Mode",
            ["smart", "baseline", "hyde", "self_corrective", "hybrid"],
            index=0,
            help="Smart routing automatically picks the best method"
        )

        st.divider()

        st.subheader("About")
        st.markdown("""
        **Methods:**
        - **Baseline**: Direct embedding search
        - **HyDE**: Generate hypothetical doc first
        - **Self-Corrective**: Reflect and reformulate
        - **Hybrid**: HyDE + Self-Correction
        - **Smart**: Auto-route based on query
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_collection" not in st.session_state:
        st.session_state.current_collection = None

    # Check if collection changed
    if st.session_state.current_collection != selected_collection:
        st.session_state.current_collection = selected_collection
        st.session_state.messages = []  # Clear chat on collection change

        # Load the new vector store
        with st.spinner(f"Loading {selected_collection}..."):
            st.session_state.vector_store = get_vector_store(collection_info["collection_name"])
            st.session_state.vector_store.add_documents(collection_info["docs"])

        st.rerun()

    # Ensure vector store is loaded
    if "vector_store" not in st.session_state:
        with st.spinner(f"Loading {selected_collection}..."):
            st.session_state.vector_store = get_vector_store(collection_info["collection_name"])
            st.session_state.vector_store.add_documents(collection_info["docs"])

    # Determine doc type for routing
    doc_type = "story" if "Story" in selected_collection else "technical"

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show debug info if available and debug mode is on
            if debug_mode and "debug" in message and message["role"] == "assistant":
                render_debug_info(message["debug"])

    # Chat input
    if prompt := st.chat_input(collection_info["placeholder"]):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Run RAG pipeline
            with st.spinner("Thinking..." if not debug_mode else "Running RAG pipeline..."):
                contexts, debug_info = run_rag_query(
                    prompt,
                    st.session_state.vector_store,
                    method=rag_method,
                    doc_type=doc_type
                )

            # Show debug info before answer if enabled
            if debug_mode:
                render_debug_info(debug_info)
                st.divider()

            # Stream the answer
            debug_info.chat_calls += 1
            debug_info.api_calls += 1

            response_placeholder = st.empty()
            full_response = ""

            for chunk in generate_answer_streaming(prompt, contexts, doc_type):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "debug": debug_info
            })


def render_debug_info(debug: QueryDebugInfo):
    """Render debug information in an expandable section."""
    with st.expander("🔧 Debug Information", expanded=True):
        # Routing info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Method", debug.routing_decision.upper())
        with col2:
            st.metric("API Calls", debug.api_calls)
        with col3:
            st.metric("Latency", f"{debug.latency_ms:.0f}ms")

        st.caption(f"**Routing reason:** {debug.routing_reason}")

        # Hypothetical document (for HyDE)
        if debug.hypothetical_doc:
            st.subheader("📝 Hypothetical Document")
            st.info(debug.hypothetical_doc)

        # Reflection steps (for self-corrective)
        if debug.reflection_steps:
            st.subheader("🔄 Reflection Steps")
            for step in debug.reflection_steps:
                status = "✅" if step["decision"] == "YES" else "❌"
                st.markdown(f"**Iteration {step['iteration']}** {status}")
                st.markdown(f"- Query: *{step['query'][:80]}...*" if len(step['query']) > 80 else f"- Query: *{step['query']}*")
                st.markdown(f"- Decision: **{step['decision']}** - {step['reason']}")

        # Reformulated queries
        if debug.reformulated_queries:
            st.subheader("🔀 Reformulated Queries")
            for i, query in enumerate(debug.reformulated_queries, 1):
                st.markdown(f"{i}. {query}")

        # Retrieved contexts
        st.subheader("📚 Retrieved Documents")
        for ctx in debug.retrieved_contexts:
            with st.container():
                similarity = ctx.get('similarity', 0)
                st.markdown(f"**[{ctx['id']}]** (similarity: {similarity:.3f})")
                st.caption(ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text'])


if __name__ == "__main__":
    main()
