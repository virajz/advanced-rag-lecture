# Advanced RAG: From Naive Retrieval to Production-Grade Systems

A comprehensive tutorial on Retrieval-Augmented Generation (RAG) techniques, covering naive implementations, advanced methods (HyDE, Self-Corrective RAG), and production patterns with evaluation frameworks.

## Overview

This repository provides hands-on implementations of:

- **Naive RAG**: Basic query-embed-search-generate pipeline
- **HyDE (Hypothetical Document Embeddings)**: Bridge vocabulary gaps between questions and documents
- **Self-Corrective RAG**: Reflect on retrieval quality and iteratively improve
- **Multi-Hop Retrieval**: Handle complex queries requiring multiple documents
- **Production Patterns**: Smart routing, cost tracking, and monitoring

## Prerequisites

- Python 3.12 or higher
- Mistral AI API key ([Get one here](https://console.mistral.ai/))
- `uv` package manager (recommended) or `pip`

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/virajz/advanced-rag-lecture.git
cd advanced-rag-lecture

# Install dependencies
uv sync
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/virajz/advanced-rag-lecture.git
cd advanced-rag-lecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your Mistral AI API key using one of these methods:

### Method 1: Environment Variable

```bash
export MISTRAL_API_KEY="your-api-key-here"
```

### Method 2: .env File

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your-api-key-here
```

## Project Structure

```
advanced-rag-lecture/
├── lecture/                    # Production-grade implementations
│   ├── 01_setup.py            # Verify Mistral AI connection
│   ├── 02_embeddings.py       # Learn embeddings & cosine similarity
│   ├── 03_baseline_rag.py     # Basic RAG implementation
│   ├── 04_hyde_rag.py         # HyDE technique
│   ├── 05_self_corrective_rag.py  # Self-corrective with reflection
│   ├── 06_evaluation.py       # Basic evaluation metrics
│   ├── 07_ragas_evaluation.py # RAGAS-style automated evaluation
│   ├── 08_production_rag.py   # Production patterns with routing
│   ├── app.py                 # Streamlit chat UI with debug mode
│   └── vector_store.py        # ChromaDB vector store
├── docs/                       # Detailed explanations
│   ├── 01_main_explained.md
│   ├── 02_toy_corpus_explained.md
│   ├── 03_evaluation_explained.md
│   ├── 04_rag_baseline_explained.md
│   ├── 05_rag_hyde_explained.md
│   ├── 06_rag_multihop_explained.md
│   └── 07_rag_self_corrective_explained.md
├── toy_corpus.py              # SSL/TLS knowledge base (7 docs)
├── story_corpus.py            # Narrative corpus for testing (14 docs)
├── main.py                    # Entry point demo
├── rag_baseline.py            # Simplified baseline demo
├── rag_hyde.py                # Simplified HyDE demo
├── rag_self_corrective.py     # Simplified self-corrective demo
├── rag_multihop_demo.py       # Multi-hop reasoning demo
├── evaluation.py              # Simplified evaluation demo
├── SLIDES.md                  # Lecture slide deck
├── requirements.txt           # Dependencies
└── pyproject.toml             # Project configuration
```

## Quick Start

### 1. Verify Setup

Test your Mistral AI connection:

```bash
uv run python lecture/01_setup.py
```

Expected output:
```
Testing Mistral AI connection...
Chat API: OK
Embeddings API: OK (dimension: 1024)
Setup complete!
```

### 2. Run the Demonstrations

**Simple demos (no real embeddings):**

```bash
# Basic entry point
python main.py

# Naive RAG demonstration
python rag_baseline.py

# HyDE demonstration
python rag_hyde.py

# Self-corrective RAG demonstration
python rag_self_corrective.py

# Multi-hop reasoning demonstration
python rag_multihop_demo.py
```

**Full implementations (with real embeddings):**

```bash
# Step through the lecture series
uv run python lecture/01_setup.py       # Connection verification
uv run python lecture/02_embeddings.py  # Embeddings basics
uv run python lecture/03_baseline_rag.py    # Full baseline RAG
uv run python lecture/04_hyde_rag.py        # Full HyDE implementation
uv run python lecture/05_self_corrective_rag.py  # Full self-corrective
uv run python lecture/06_evaluation.py      # Basic evaluation metrics
uv run python lecture/07_ragas_evaluation.py    # RAGAS evaluation pipeline
uv run python lecture/08_production_rag.py      # Production patterns
```

### 3. Interactive Chat Interface

Launch the Streamlit app with debug mode:

```bash
uv run streamlit run lecture/app.py
```

Features:
- Toggle between document collections (SSL docs or Story corpus)
- Debug mode showing routing decisions, retrieved documents, and metrics
- Real-time cost and latency tracking

## Learning Path

### Beginner: Understand the Basics

1. Read `docs/04_rag_baseline_explained.md`
2. Run `python rag_baseline.py`
3. Observe the failure modes

### Intermediate: Learn Advanced Techniques

1. Read `docs/05_rag_hyde_explained.md`
2. Run `python rag_hyde.py`
3. Understand how HyDE bridges vocabulary gaps

4. Read `docs/07_rag_self_corrective_explained.md`
5. Run `python rag_self_corrective.py`
6. See the reflection loop in action

### Advanced: Build Production Systems

1. Run through the full lecture series (`lecture/01-08`)
2. Study `lecture/08_production_rag.py` for routing patterns
3. Explore `lecture/07_ragas_evaluation.py` for evaluation frameworks
4. Launch the Streamlit app and experiment with different queries

## Key Concepts

### Naive RAG
Simple pipeline: Query → Embed → Search → Generate

**Strengths:** Fast, cheap, works for explicit queries
**Weaknesses:** Vocabulary mismatch, single-shot retrieval

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer document, then search with its embedding.

**Use when:** Queries are vague or use different vocabulary than documents
**Trade-off:** Extra LLM call adds latency and cost

### Self-Corrective RAG
Reflect on retrieval quality, reformulate query if insufficient, retry.

**Use when:** Complex queries, troubleshooting, high-stakes answers
**Trade-off:** Multiple iterations add significant latency and cost

### Production Routing
Dynamically select the best technique based on query characteristics.

```python
def route_query(question: str) -> str:
    if is_short_and_vague(question):
        return "hyde"
    elif is_troubleshooting(question):
        return "self_corrective"
    else:
        return "baseline"
```

## Evaluation Metrics

The project implements three core metrics from the RAGAS framework:

1. **Context Relevance**: Are retrieved documents useful for the question?
2. **Faithfulness**: Is the answer grounded in the retrieved context?
3. **Answer Relevance**: Does the answer actually address the question?

Run the evaluation pipeline:

```bash
uv run python lecture/07_ragas_evaluation.py
```

## Cost Considerations

**Example estimates (illustrative):**

| Method | LLM Calls | Approximate Cost/Query |
|--------|-----------|------------------------|
| Naive RAG | 1 | $0.002-0.005 |
| HyDE | 2 | $0.004-0.010 |
| Self-Corrective (2 iter) | 4 | $0.008-0.020 |

At scale (5,000 queries/day), smart routing can save $300-600/month compared to using advanced techniques for all queries.

## Dependencies

- `mistralai>=1.0.0` - Mistral AI SDK for embeddings and chat
- `numpy>=1.26.0` - Vector operations and cosine similarity
- `python-dotenv>=1.0.0` - Environment variable management
- `chromadb>=0.4.0` - Vector database for persistent storage
- `streamlit>=1.30.0` - Interactive chat interface
- `watchdog>=3.0.0` - File monitoring for Streamlit

## Troubleshooting

### API Key Issues

```
Error: MISTRAL_API_KEY not found
```

Ensure your API key is set via environment variable or `.env` file.

### Rate Limiting

The vector store implementation includes exponential backoff for rate limits. If you encounter persistent rate limiting, reduce the batch size in `lecture/vector_store.py`.

### ChromaDB Persistence

The vector store persists to `.chroma_db/` directory. To reset:

```bash
rm -rf .chroma_db/
```

## Lecture Slides

The complete lecture slide deck is available in `SLIDES.md`. View it with any Markdown viewer or convert to presentation format.

## Contributing

Contributions welcome! Please open an issue or pull request.

## License

MIT License

## Author

Viraj Zaveri

## Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
