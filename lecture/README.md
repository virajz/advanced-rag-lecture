# Advanced RAG Tutorial with Mistral AI

A step-by-step tutorial for building production-grade RAG systems using Mistral AI.

## Project Structure

```
lecture/
├── README.md              # This file - system instructions
├── vector_store.py        # Shared ChromaDB vector store module
├── app.py                 # Streamlit chat UI with debug mode
├── 01_setup.py            # Verify Mistral AI connection
├── 02_embeddings.py       # Learn embeddings and similarity search
├── 03_baseline_rag.py     # Basic RAG implementation
├── 04_hyde_rag.py         # HyDE (Hypothetical Document Embeddings)
├── 05_self_corrective_rag.py  # Self-corrective RAG with reflection
├── 06_evaluation.py       # RAG evaluation metrics
├── 07_ragas_evaluation.py # RAGAS-style automated evaluation pipeline
└── 08_production_rag.py   # Production patterns with smart routing
```

## Prerequisites

- Python 3.12+
- Mistral AI API key
- uv package manager

## Setup

```bash
# Install dependencies
uv sync

# Set API key (choose one method)
export MISTRAL_API_KEY="your-key-here"
# OR create .env file with: MISTRAL_API_KEY=your-key-here

# Verify setup
uv run python lecture/01_setup.py
```

## Mistral AI Models Used

| Model | Purpose | Notes |
|-------|---------|-------|
| `mistral-embed` | Text embeddings | 1024-dimensional vectors |
| `mistral-small-latest` | Chat completion | Cost-effective for most tasks |

## Key Concepts Covered

1. **Embeddings**: Vector representations of text for semantic search
2. **Baseline RAG**: Retrieve documents, generate answers
3. **HyDE**: Generate hypothetical answer first, then search (bridges query-document vocabulary gap)
4. **Self-Corrective RAG**: Reflect on retrieval quality, reformulate query if needed
5. **Evaluation**: Context relevance, faithfulness, answer relevance

## Architecture Patterns

### Basic RAG Flow
```
Query → Embed → Search → Retrieve Top-K → Generate Answer
```

### HyDE Flow
```
Query → Generate Hypothetical Doc → Embed Hypothetical → Search → Retrieve → Generate Answer
```

### Self-Corrective RAG Flow
```
Query → Search → Reflect ("Good enough?")
  ├─ YES → Generate Answer
  └─ NO  → Reformulate Query → Loop back to Search
```

## Chat UI

Launch the interactive chat interface:

```bash
uv run streamlit run lecture/app.py
```

Features:
- **Normal mode**: Clean chat with streamed answers
- **Debug mode**: Toggle in sidebar to see routing decisions, retrieved documents, reflection steps, and metrics

## Running Each Step

```bash
uv run python lecture/01_setup.py            # Verify connection
uv run python lecture/02_embeddings.py       # Learn embeddings
uv run python lecture/03_baseline_rag.py     # Basic RAG
uv run python lecture/04_hyde_rag.py         # HyDE RAG
uv run python lecture/05_self_corrective_rag.py  # Self-corrective
uv run python lecture/06_evaluation.py       # Basic evaluation metrics
uv run python lecture/07_ragas_evaluation.py # RAGAS-style pipeline
uv run python lecture/08_production_rag.py   # Production patterns
```

## Reference Documents

The `docs/` folder contains detailed explanations of each concept:
- `docs/04_rag_baseline_explained.md` - Baseline RAG concepts
- `docs/05_rag_hyde_explained.md` - HyDE explanation
- `docs/07_rag_self_corrective_explained.md` - Self-corrective RAG

## Toy Corpus

All examples use `toy_corpus.py` - a small knowledge base about SSL/TLS configuration with 7 documents. This simulates real-world scenarios where:
- Some docs are relevant but incomplete
- Some docs require multi-hop reasoning
- Vocabulary mismatch exists between queries and documents
