# Scoped Banking Assistant with Jailbreak Defense

A domain-restricted chatbot for banking customer support, built with RAG (Retrieval-Augmented Generation) and a multi-layer guardrail system that detects and blocks jailbreak/prompt-injection attempts.

## Architecture

```
User Query
    |
    v
[Guardrail Layer 1: Regex pattern matching]  --> catches common injection phrasing
    |
    v
[Guardrail Layer 2: Embedding similarity]     --> catches paraphrased/novel jailbreak attempts
    |
    v
[Guardrail Layer 3: Domain relevance check]   --> blocks off-topic questions
    |
    v
[RAG Retrieval: ChromaDB + sentence-transformers]  --> fetches relevant banking FAQ context
    |
    v
[LLM Generation: Ministral 3B via Ollama]      --> generates grounded answer
    |
    v
Response
```

## Why this project matters (for your portfolio)

- **RAG**: Industry-standard technique for grounding LLM responses in real data, avoids hallucination
- **LLM Security**: Jailbreak/prompt-injection defense is directly relevant to any bank deploying customer-facing AI — very few candidates demonstrate this skill
- **Measurable results**: The evaluation script gives you concrete metrics (block rate %, false positive rate %) to cite in interviews/resume
- **Fully local**: No API costs, runs entirely offline using Ollama + open-source embedding models

## Setup

### 1. Prerequisites
- Python 3.9+
- Ollama installed with `ministral-3:3b` pulled (`ollama pull ministral-3:3b`)
- Ollama server running (`ollama serve` — usually auto-starts)

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the vector index
```bash
python build_index.py
```
This embeds the banking FAQ knowledge base and stores it in a local ChromaDB store.

### 4. Run the chatbot
```bash
python chatbot.py
```
Chat interactively in the terminal. Try normal banking questions, then try jailbreak attempts to see the guardrails in action.

### 5. Run the evaluation suite
```bash
python evaluate.py
```
This tests the guardrail pipeline against:
- 15 jailbreak attempts (should be blocked)
- 8 off-topic queries (should be blocked)
- 10 legitimate banking queries (should pass through)

Results print to console and save to `evaluation_report.json`.

## Files

| File | Purpose |
|---|---|
| `knowledge_base.json` | Banking FAQ dataset (RAG source) |
| `build_index.py` | Builds ChromaDB vector index from knowledge base |
| `guardrails.py` | 3-layer jailbreak/off-topic detection pipeline |
| `chatbot.py` | Main chat loop: guardrails -> retrieval -> LLM |
| `test_prompts.json` | Jailbreak + off-topic + legitimate test prompts |
| `evaluate.py` | Runs test suite, computes block-rate metrics |

## Extending this project

- **Add more FAQ entries** to `knowledge_base.json` for broader coverage
- **Expand jailbreak patterns** in `guardrails.py` as you discover new attack styles
- **Tune thresholds** (`threshold` params in `guardrails.py`) to balance security vs false positives
- **Add an LLM-based classifier** as a 4th guardrail layer for catching novel attacks the embedding/pattern layers miss
- **Build a web frontend** (Flask/FastAPI backend + simple HTML/JS chat UI) to demo this visually in interviews

## Resume bullet point (example)

> Built a RAG-based banking support chatbot with a 3-layer guardrail system (pattern matching, embedding similarity, domain relevance) achieving [X]% jailbreak block rate and [Y]% legitimate query pass-through, using Ollama, ChromaDB, and sentence-transformers.

Fill in X and Y with your actual `evaluate.py` results once you run it.
