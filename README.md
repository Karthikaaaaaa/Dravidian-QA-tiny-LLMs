# Fine-Tuning Tiny LLMs for Question Answering in Dravidian Languages

> **Published Research** — Book chapter featured in *Large Language Models and AI: Challenges and Innovations* (2025)

A systematic study on adapting compact Large Language Models for Question Answering in **Malayalam and Tamil** — two morphologically rich, low-resource Dravidian languages. This work demonstrates that tiny LLMs fine-tuned with LoRA and QLoRA can achieve competitive QA performance while reducing model size by up to **1,008x**, making them deployable on edge devices and mobile phones.

---

## Key Results

| Technique | Model | Language | F1 Score | Cosine Similarity | Model Size |
|---|---|---|---|---|---|
| QLoRA | Open LLaMA 3.1B | Tamil | **0.93** | **1.0** | 11.9 MB (from 12 GB) |
| QLoRA | Open LLaMA 3.1B | Malayalam | 0.85 | 1.0 | 11.9 MB (from 12 GB) |
| QLoRA | Gemma 2B | Tamil | 0.80 | 0.58 | 8.9 MB (from 9.56 GB) |
| LoRA | Open LLaMA 3.1B | Malayalam | 0.865 | 0.99 | — |
| RAG | Gemma 2B | Malayalam | — | 0.58 | — |

**Model size reduction achieved via QLoRA:**

| Model | Original Size | Compressed Size | Reduction |
|---|---|---|---|
| Open LLaMA 3.1B | 12 GB | 11.9 MB | ~1,008x |
| Gemma 2B | 9.56 GB | 8.9 MB | ~1,000x |
| T5 Base | 0.85 GB | 1 MB | ~1,000x |

---

## Why This Matters

Malayalam and Tamil are agglutinative, morphologically rich languages with free word order and context-dependent particles — features that make standard NLP models fail. Most multilingual LLMs (mBERT, XLM-R) are trained predominantly on high-resource languages and struggle with Dravidian linguistic structures.

This work shows that **tiny LLMs + parameter-efficient fine-tuning** can bridge this gap without requiring massive compute — enabling real-time QA for ~90 million Dravidian language speakers on resource-constrained devices.

---

## Models Used

| Model | Parameters | Key Strength |
|---|---|---|
| **Gemma 2B** | 2B | Strong linguistic pattern understanding |
| **Open LLaMA 3.1B** | 3.1B | Best multilingual performance |
| **T5 Base** | 220M | Fastest inference, text-to-text versatility |

---

## Methodology

### 1. Dataset
**IndicQA** — a structured QA dataset covering 10 Indic languages.

| Language | Paragraphs | Questions | Unique Words |
|---|---|---|---|
| Malayalam | 247 | 1,589 | 30,534 |
| Tamil | 253 | 1,804 | 27,331 |

### 2. Approaches

**Baseline — Zero-shot & Few-shot Prompting**
- Zero-shot: All models scored 0 across BLEU, ROUGE, F1 — confirming that pre-trained LLMs lack sufficient Dravidian language representation
- Few-shot: Gemma 2B achieved cosine similarity of 1.0 on Malayalam with 2 examples — showing the power of in-context learning even for low-resource languages

**RAG — Retrieval-Augmented Generation**
```
Query → Embedding (paraphrase-multilingual-MiniLM-L12-v2)
     → ChromaDB vector search (mc4-ml / mc4-ta corpora)
     → Retrieved contexts → LLM response generation
```
- BERTScore F1 improved to 0.87–0.94 across models
- Demonstrates that external retrieval compensates for limited pre-training coverage

**LoRA — Low-Rank Adaptation**
```python
# Key hyperparameters
rank (r)         = 8
alpha            = 32
dropout          = 0.1
target_modules   = ['q_proj', 'v_proj']
learning_rate    = 1e-4
batch_size       = 1
epochs           = 3
```
- Freezes pre-trained weights, trains only low-rank matrices A and B
- Parameter reduction example: WQ ∈ R^(1024×512) with r=8 → 12,288 params vs 524,288 in full fine-tuning
- Open LLaMA achieved F1 of 0.865 and cosine similarity of 0.99 on Malayalam

**QLoRA — Quantized Low-Rank Adaptation**
```python
# Additional hyperparameters
quantization     = 8-bit (NF4)
rank (r)         = 8
alpha            = 32
dropout          = 0.1
target_modules   = ['q_proj', 'v_proj']
learning_rate    = 1e-4
```
- Compresses frozen weights to 4-bit/8-bit integers using NF4 quantization
- Final weights: W_final = Q(W) + ΔW
- Best result: Open LLaMA on Tamil — F1 0.93, cosine similarity 1.0

---

## Experimental Setup

**Hardware:**
- 3x NVIDIA A100 GPUs
- 21 CPUs, 96GB RAM, 120GB VRAM

**Software Stack:**
```
Python          3.10
PyTorch         2.0
Transformers    4.35 (HuggingFace)
ChromaDB        vector database
bitsandbytes    8-bit quantization
pandas          2.1
numpy           1.25
nltk            3.9
OS              Ubuntu 20.04 LTS
```

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| BLEU | N-gram overlap with reference answers |
| ROUGE-1 / ROUGE-L | Recall-oriented overlap |
| F1 Score | Harmonic mean of precision and recall |
| Cosine Similarity | Semantic vector similarity |
| BERTScore (P/R/F1) | Context-aware semantic similarity using BERT embeddings |

> **Note on BERTScore:** Traditional metrics like BLEU and ROUGE penalise morphological variation heavily — a major limitation for agglutinative languages like Malayalam and Tamil. BERTScore captures semantic similarity even when surface forms differ, making it the most reliable metric for this task.

---

## Key Findings

1. **Zero-shot prompting fails completely** for Dravidian QA — all models scored 0 across all metrics, confirming insufficient representation in pre-training corpora

2. **Few-shot prompting provides significant uplift** — Gemma 2B achieved cosine similarity of 1.0 on Malayalam with just 2 examples

3. **RAG improves semantic quality** — BERTScore F1 reaches 0.87–0.94, though exact-match metrics remain low due to morphological complexity

4. **LoRA outperforms RAG** on F1 and cosine similarity by focusing adaptation on attention projection layers (q_proj, v_proj)

5. **QLoRA achieves the best results** — combining 8-bit quantization with low-rank adaptation delivers the highest F1 scores while reducing model size by ~1000x

6. **Edge deployment is viable** — Open LLaMA compressed from 12GB to 11.9MB with QLoRA, suitable for mobile and IoT deployment

---

## Project Structure

```
dravidian-qa-tiny-llms/
├── Final_code.ipynb          # Complete experimental notebook
│   ├── Data loading & preprocessing
│   ├── Zero-shot & few-shot prompting
│   ├── RAG pipeline (ChromaDB + embeddings)
│   ├── LoRA fine-tuning
│   ├── QLoRA fine-tuning
│   └── Evaluation (BLEU, ROUGE, F1, BERTScore)
├── chapter_draft.pdf         # Published book chapter
└── README.md
```

---

## Publication

**Book Chapter:** Fine-Tuning Tiny LLMs for Question Answering in Dravidian Languages
**Featured in:** *Large Language Models and AI: Challenges and Innovations* (2025)
**Author:** Karthika S S

---

## Author

**Karthika S S** — ML Engineer | LLM & GenAI Systems | Dravidian Language AI
- LinkedIn: [linkedin.com/in/karthika-s-s-18574b1b2](https://linkedin.com/in/karthika-s-s-18574b1b2)
- GitHub: [github.com/Karthikaaaaaa](https://github.com/Karthikaaaaaa)

---

## Citation

If you use this work, please cite the book chapter:
```
Karthika S S. "Fine-Tuning Tiny LLMs for Question Answering in Dravidian Languages."
In Large Language Models and AI: Challenges and Innovations, 2025.
```
