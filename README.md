# Bharat RAG Eval

A multilingual Retrieval-Augmented Generation (RAG) system for Indian education/government documents — built with a **rigorous evaluation harness** that measures accuracy, retrieval quality, and language correctness across English, Hindi, and one additional Indic language.

## Features

- **Multilingual retrieval & generation** — handles Indic languages properly using multilingual embeddings, script-aware chunking, and generation that responds in the user's language instead of falling back to English.
- **Built-in evaluation harness** — a scored, repeatable evaluation across languages and models, not just a demo that "looks good."
- **Model-agnostic pipeline** — swap between OpenAI, Anthropic, or other providers without touching retrieval logic.
- **Cost tracking** — every query reports input/output tokens and estimated cost, so model tradeoffs are measurable, not guessed.

## Architecture

```
User question (any language)
        │
        ▼
 [Embed query] ──► bge-m3 (multilingual embedding model)
        │
        ▼
 [Retrieve top-k chunks] ──► ChromaDB vector store
        │
        ▼
 [Generate answer] ──► LLM (configurable: OpenAI / Anthropic / local model)
        │
        ▼
 Answer, in the same language as the question
```

The **eval harness** runs independently: it takes a labeled question set (multiple languages, same underlying content), runs it through the pipeline, and scores:
- **Retrieval hit-rate** — did we fetch the chunk that actually contains the answer?
- **Answer accuracy** — LLM-as-judge scoring against a reference answer
- **Language fidelity** — did the model answer in the language it was asked in, without code-switching?

## Project structure

```
bharat-rag-eval/
├── app/
│   ├── ingest.py        # chunk + embed documents into the vector store
│   ├── rag_pipeline.py  # retrieval + generation logic
│   ├── config.py        # model choices, paths, constants
│   └── api.py           # FastAPI server exposing /ask
├── data/
│   └── docs/            # put your source documents here (PDF/txt)
├── eval/
│   ├── eval_set.jsonl   # labeled question/answer pairs, multilingual
│   ├── run_eval.py      # runs the pipeline against eval_set.jsonl and scores it
│   └── results/         # eval run outputs land here
├── scripts/
│   └── quickstart.sh    # one-command setup
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Put some documents in data/docs/ (PDF or .txt)

# 4. Ingest documents into the vector store
python app/ingest.py

# 5. Run the API server
uvicorn app.api:app --reload

# 6. Ask a question
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "भारत में शिक्षा का अधिकार कब लागू हुआ?"}'

# 7. Run the evaluation harness
python eval/run_eval.py
```

## Evaluation results (fill in after running)

| Model              | Language | Retrieval hit-rate | Answer accuracy | Language fidelity | Avg tokens/query |
|--------------------|----------|--------------------:|-----------------:|--------------------:|-------------------:|
| gpt-4o-mini        | English  |                      |                   |                      |                     |
| gpt-4o-mini        | Hindi    |                      |                   |                      |                     |
| claude-haiku       | English  |                      |                   |                      |                     |
| claude-haiku       | Hindi    |                      |                   |                      |                     |

*(Run `python eval/run_eval.py` and this table will be regenerated in `eval/results/`.)*

## Design notes

- **bge-m3** was chosen for embeddings because it's trained explicitly for multilingual + cross-lingual retrieval, unlike English-only embedding models that silently degrade on Devanagari/other Indic scripts.
- Chunking respects sentence structure per-language rather than a fixed token count — naive fixed-token chunking splits mid-word more often on Indic scripts due to tokenizer inefficiency.
- The eval harness uses **LLM-as-judge with a rubric**, not exact-match, because Indic language generation has legitimate phrasing variation that exact-match would unfairly penalize.
- Retrieval and generation are decoupled so either the embedding model or the LLM can be swapped without touching the other half.

## Roadmap

- Add a reranking step (e.g. cross-encoder) to fix cases where embedding similarity ranks a topically-close-but-wrong chunk above the correct one.
- Expand the eval set beyond ~50 questions per language for statistical confidence.
- Add code-switching detection (Hindi written in Roman script mixed with English) as its own eval dimension.
