# 🏦 Scoped Banking Assistant with Jailbreak Defense

A domain-restricted banking support chatbot built with **RAG (Retrieval-Augmented Generation)** and a **multi-layer guardrail system** that detects and blocks prompt-injection / jailbreak attempts — fully local, no API costs.

> Built as a portfolio project demonstrating LLM application design + LLM security for AI/ML roles in banking/fintech.

---

## 🎯 Why this project

Banks are increasingly deploying customer-facing AI assistants, which introduces two hard problems:
1. **Grounding** — the bot must answer using accurate, up-to-date bank data, not hallucinate
2. **Security** — the bot must resist prompt injection and jailbreak attempts trying to make it reveal internal instructions, go off-topic, or misbehave

This project tackles both with a RAG pipeline + a 3-layer guardrail system, and includes an evaluation suite that measures real block-rate metrics.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ Guardrail Layer 1: Regex Patterns   │  → catches common injection phrasing
├─────────────────────────────────────┤
│ Guardrail Layer 2: Embedding        │  → catches paraphrased/novel attacks
│ Similarity to Known Jailbreaks      │
├─────────────────────────────────────┤
│ Guardrail Layer 3: Domain           │  → blocks off-topic questions
│ Relevance Check                     │
└─────────────────────────────────────┘
    │ (passed)
    ▼
┌─────────────────────────────────────┐
│ RAG Retrieval                       │  → ChromaDB + sentence-transformers
│ (finds relevant banking FAQ context)│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ LLM Generation                      │  → Ministral 3B via Ollama (local)
└─────────────────────────────────────┘
    │
    ▼
Response
```

---

## ✨ Features

- 🔍 **Semantic search** over a banking FAQ knowledge base using sentence embeddings
- 🛡️ **3-layer guardrail pipeline**: regex → embedding similarity → domain relevance
- 🤖 **Fully local LLM inference** via Ollama (no API keys, no cost, no data leaves your machine)
- 📊 **Evaluation suite** with 33 test prompts (jailbreaks, off-topic, legitimate) producing measurable block-rate metrics
- 🧩 **Modular design** — swap the LLM, embedding model, or vector store independently

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Ministral 3B (via [Ollama](https://ollama.com)) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Language | Python 3.9+ |

---

## 🚀 Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed
- Ministral model pulled:
  ```bash
  ollama pull ministral-3:3b
  ```

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/banking-chatbot-guardrails.git
cd banking-chatbot-guardrails

# Install dependencies
pip install -r requirements.txt

# Build the vector index from the FAQ knowledge base
python build_index.py

# Run the chatbot
python chatbot.py
```

### Run the evaluation suite

```bash
python evaluate.py
```

This tests the guardrail pipeline against 33 prompts and reports:
- Jailbreak block rate
- Off-topic block rate
- Legitimate query pass-through rate (false positive check)

Results are saved to `evaluation_report.json`.

---

## 📁 Project Structure

```
banking-chatbot-guardrails/
├── knowledge_base.json     # Banking FAQ dataset (RAG source)
├── build_index.py          # Builds ChromaDB vector index
├── guardrails.py           # 3-layer jailbreak/off-topic detection
├── chatbot.py               # Main pipeline: guardrails → retrieval → LLM
├── test_prompts.json       # Jailbreak + off-topic + legitimate test set
├── evaluate.py              # Runs test suite, computes metrics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Results

_Run `python evaluate.py` and update this table with your actual numbers:_

| Metric | Result |
|---|---|
| Jailbreak Block Rate | XX% (X/15) |
| Off-Topic Block Rate | XX% (X/8) |
| Legitimate Query Pass Rate | XX% (X/10) |

---

## 🔮 Future Improvements

- [ ] Add an LLM-based classifier as a 4th guardrail layer for catching novel/unseen attack patterns
- [ ] Expand the knowledge base with more FAQ categories
- [ ] Build a web UI (Flask/FastAPI + HTML-JS frontend)
- [ ] Add conversation memory for multi-turn context
- [ ] Fine-tune guardrail thresholds using a larger labeled dataset

---

## 📄 License

MIT License — feel free to use this project as a learning reference.

---

## 🙋 About

Built as a hands-on project to explore RAG systems and LLM security patterns relevant to AI/ML roles in banking and fintech.
