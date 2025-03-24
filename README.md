# RAG-Bot: Chat with Your PDFs

A lightweight chatbot that uses **RAG (Retrieval-Augmented Generation)** and a **Local LLM** to answer questions based on the content of a PDF.

---

## Features

- Upload any PDF
- Automatically extracts and chunks content
- Uses sentence embeddings for semantic retrieval
- Answers questions using a local LLM (e.g., Microsoft Phi-1.5)
- No external API needed — runs fully local!

---

## How It Works

1. **Extracts text** from PDF using `PyMuPDF`
2. **Chunks** the text into overlapping segments
3. **Embeds** chunks using `sentence-transformers`
4. **Indexes** them using `FAISS`
5. **Retrieves** relevant chunks using cosine similarity
6. **Feeds** them to an LLM (Phi-1.5 by default)
7. **Generates** an answer based on the most relevant content

---

## Getting Started

### Clone the Repo

git clone https://github.com/your-username/rag-bot.git <br>
cd rag-bot

### Install Dependencies

> Requires **Python 3.8+** <br>
> pip install torch transformers sentence-transformers faiss-cpu pymupdf <br>
> python rag_bot.py <br>

---
### Example

```text
Enter path to PDF: research_paper.pdf
Extracted 15321 characters
Chunked into 45 segments
Embeddings created
FAISS index ready
LLM loaded

Ask a question (or type 'exit'):
```
---
### Model Info

Embeddings: all-MiniLM-L6-v2 (via sentence-transformers) <br>
LLM: microsoft/phi-1_5 (you can swap this in the code)


