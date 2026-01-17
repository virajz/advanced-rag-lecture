# toy_corpus.py - The Knowledge Database

## What is this file?

This is your **fake knowledge base** - a small collection of documents that the RAG system will search through. Think of it as a mini Wikipedia just for SSL/security topics.

## The Big Picture

```
┌─────────────────────────────────────────┐
│           TOY CORPUS (DOCS)             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Doc 1   │ │ Doc 2   │ │ Doc 3   │   │
│  │ SSL     │ │ Dev     │ │ Rotation│   │
│  │ Formats │ │ Setup   │ │ Policy  │   │
│  └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Doc 4   │ │ Doc 5   │ │ Doc 6   │   │
│  │ Prod    │ │ Container│ │Compliance│  │
│  │ Steps   │ │ Ports   │ │ Rules   │   │
│  └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐                            │
│  │ Doc 7   │                            │
│  │ Infra   │                            │
│  │Constraint│                           │
│  └─────────┘                            │
└─────────────────────────────────────────┘
```

## What's Inside?

The file contains a list called `DOCS` with 7 mini-documents:

| ID | What It's About | Key Info |
|----|-----------------|----------|
| `ssl_formats` | Certificate file types | PEM, DER formats |
| `dev_setup_ssl_mention` | Local development | localhost:3000, SSL optional |
| `rotation_policy` | Certificate renewal | Rotate every 90 days |
| `prod_ssl_steps` | **Production setup** | Step-by-step instructions! |
| `container_port_note` | Docker/containers | Use port 8443, not 443 |
| `compliance_req` | Security rules | TLS 1.2+, no weak ciphers |
| `infra_constraint` | Server limitations | Only proxy can use port 443 |

## Explained Like You're 5

Imagine you're playing a treasure hunt game. This file is the **treasure chest** full of clue cards. Each card has:
- A **name tag** (the `id`) so you can find it later
- Some **information** (the `text`) that might help answer questions

When someone asks "How do I set up SSL?", the system searches through these cards to find helpful ones!

## The Code Structure

```python
DOCS = [
  {
    "id": "ssl_formats",           # Unique name for this document
    "text": "SSL certificates..."  # The actual content
  },
  # ... more documents ...
]
```

## Why Is It Called "Toy"?

Because it's **tiny and fake**! In a real system, you'd have:
- Thousands or millions of documents
- A real database (not a Python list)
- Vector embeddings for smart searching

This is just for **learning and demos**.

## How Other Files Use It

```python
from toy_corpus import DOCS  # Import the document list

for doc in DOCS:
    print(doc["id"])  # Access each document
```

---
*This is the "brain" that other RAG files search through!*
