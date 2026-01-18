# 10 Things to Try with Hugging Face Transformers

A comprehensive beginner's course for learning Hugging Face Transformers through 10 hands-on notebooks.

**Total Course Time**: ~20 hours (10 notebooks × ~2 hours each)

**Target Audience**: ML beginners with Python experience

**What You'll Build**: By the end of this course, you'll understand how transformer models work, be able to use Hugging Face pipelines for common NLP tasks, and have the skills to build your own custom NLP applications.

---

## Quick Start

Click any badge below to open a notebook directly in Google Colab—no setup required:

| # | Module | Open in Colab |
|---|--------|---------------|
| 1 | **Fill-Mask** - Learn masked language modeling with BERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/01-fill-mask.ipynb) |
| 2 | **Named Entity Recognition** - Extract people, places, and organizations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/02-named-entity-recognition.ipynb) |
| 3 | **Question Answering** - Build extractive Q&A systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/03-question-answering.ipynb) |
| 4 | **Text Summarization** - Compress long text with encoder-decoder models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/04-text-summarization.ipynb) |
| 5 | **Text Generation** - Master autoregressive generation and sampling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/05-text-generation.ipynb) |
| 6 | **Zero-Shot Classification** - Classify without training data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/06-zero-shot-classification.ipynb) |
| 7 | **Translation** - Translate between languages | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/07-translation.ipynb) |
| 8 | **Text Similarity & Embeddings** - Compare texts with vector representations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/08-text-similarity-embeddings.ipynb) |
| 9 | **Sentiment Analysis** - Compare different sentiment models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/09-sentiment-different-models.ipynb) |
| 10 | **Pipeline Internals** (Capstone) - Understand what happens under the hood | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buildLittleWorlds/10-things-to-try-with-hugging-face/blob/main/10-things-to-try/10-classification-pipeline-internals.ipynb) |

---

## Course Structure

The notebooks are designed to be completed **in order**. Each builds on concepts from previous ones:

```
Notebooks 1-3: Foundation
├── 1. Fill-Mask → Core pipeline concept, masked language models
├── 2. NER → Token-level predictions
└── 3. Question Answering → Context understanding

Notebooks 4-5: Text Generation
├── 4. Summarization → Encoder-decoder models
└── 5. Text Generation → Decoding strategies

Notebooks 6-7: Classification & Translation
├── 6. Zero-Shot → NLI-based classification
└── 7. Translation → Multilingual models

Notebooks 8-9: Analysis & Comparison
├── 8. Embeddings → Vector representations
└── 9. Sentiment Models → Model comparison

Notebook 10: Capstone
└── 10. Pipeline Internals → Everything under the hood
```

---

## What You'll Learn

| Module | Key Concepts | Mini-Project |
|--------|--------------|--------------|
| **1. Fill-Mask** | MLM, `[MASK]` token, BERT training, confidence scores | Word Fitness Scorer |
| **2. NER** | Entity types (PER, ORG, LOC), BIO tagging, aggregation | News Entity Analyzer |
| **3. Question Answering** | Extractive QA, span extraction, SQuAD 2.0 | Document Q&A System |
| **4. Summarization** | Abstractive vs extractive, beam search, hallucination | Article Digest Generator |
| **5. Text Generation** | Temperature, top-k, top-p sampling, repetition penalty | Creative Writing Assistant |
| **6. Zero-Shot** | NLI-based classification, hypothesis templates | Custom Content Tagger |
| **7. Translation** | Seq2seq, multilingual models, BLEU scores | Multilingual Content Adapter |
| **8. Embeddings** | Pooling strategies, cosine similarity, semantic search | Semantic FAQ Matcher |
| **9. Sentiment** | Binary vs multi-class, model comparison, ensembles | Multi-Model Dashboard |
| **10. Internals** | Tokenization, attention masks, logits, softmax | Custom Pipeline Builder |

---

## Time Estimates

| Activity | Time |
|----------|------|
| Reading conceptual sections | 20-25 min |
| Running code examples | 30-35 min |
| Completing exercises | 45-50 min |
| Mini-project | 20-25 min |
| **Total per notebook** | **~2 hours** |

---

## Prerequisites

- Python 3.8+
- Basic Python knowledge (functions, classes, lists, dictionaries)

If running locally (not required for Colab):
```bash
pip install transformers torch numpy
```

---

## Tips for Success

1. **Run every code cell** - Don't just read; execute and observe
2. **Complete all exercises** - They reinforce key concepts
3. **Build the mini-projects** - Real understanding comes from building
4. **Take breaks** - 2-3 notebooks per session is ideal
5. **Experiment** - Modify code to see what happens
6. **Check solutions only after trying** - Struggle builds understanding

---

## What's Next?

After completing this course:

- **Vision Course**: Continue with [10 Things to Try with Vision Transformers](https://github.com/buildLittleWorlds/10-things-to-try-vision-transformers) - the sequel course covering ViT, CLIP, Stable Diffusion, and more
- **Fine-tuning**: Train models on your own data with the [Hugging Face Course](https://huggingface.co/course)
- **Larger Models**: Try `deberta-v3-large`, `flan-t5-large`
- **Advanced**: Quantization, PEFT/LoRA, deployment with FastAPI or Gradio, RAG

---

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Course](https://huggingface.co/course)
- [Jay Alammar's Visual Guides](https://jalammar.github.io/)
- [Full Course Details](10-things-to-try/10_things_to_try.md)

---

## License

This course material is provided for educational purposes.
