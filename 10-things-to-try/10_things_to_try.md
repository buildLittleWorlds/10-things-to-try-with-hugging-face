# Hugging Face Transformers: A Comprehensive Beginner's Course

**Total Course Time**: ~20 hours (10 notebooks x ~2 hours each)

**Target Audience**: ML beginners with Python experience

**What You'll Build**: By the end of this course, you'll understand how transformer models work, be able to use Hugging Face pipelines for common NLP tasks, and have the skills to build your own custom NLP applications.

---

## Prerequisites

Before starting this course, ensure you have:

- [ ] Python 3.8+ installed
- [ ] Basic Python knowledge (functions, classes, lists, dictionaries)
- [ ] The following packages installed:
  ```bash
  pip install transformers torch numpy
  ```
- [ ] ~5GB of free disk space for model downloads (cached at `~/.cache/huggingface/hub/`)

---

## How to Use This Course

### Learning Path

The notebooks are designed to be completed **in order**. Each notebook builds on concepts from previous ones:

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

### Time Estimates

| Activity | Time |
|----------|------|
| Reading conceptual sections | 20-25 min |
| Running code examples | 30-35 min |
| Completing exercises | 45-50 min |
| Mini-project | 20-25 min |
| **Total per notebook** | **~2 hours** |

### Tips for Success

1. **Run every code cell** - Don't just read; execute and observe
2. **Complete all exercises** - They reinforce key concepts
3. **Build the mini-projects** - Real understanding comes from building
4. **Take breaks** - 2-3 notebooks per session is ideal
5. **Experiment** - Modify code to see what happens
6. **Check solutions only after trying** - Struggle builds understanding

---

## Course Modules

### Module 1: Fill-Mask (Masked Language Modeling)

**Time**: ~2 hours | **Difficulty**: Beginner | **Prerequisites**: None

**Description**: Learn how BERT-style models predict missing words using bidirectional context. This foundational notebook introduces the pipeline concept that you'll use throughout the course.

**Key Concepts**:
- Masked Language Modeling (MLM)
- The `[MASK]` token and how BERT was trained
- Confidence scores and top-k predictions
- Pipeline basics

**Mini-Project**: Word Fitness Scorer - Build a tool that suggests better alternatives for awkward word choices in writing.

---

### Module 2: Named Entity Recognition (NER)

**Time**: ~2 hours | **Difficulty**: Beginner | **Prerequisites**: Module 1

**Description**: Discover how models identify and classify entities (people, organizations, locations) in text. Learn about token-level classification and BIO tagging.

**Key Concepts**:
- Entity types (PER, ORG, LOC, MISC)
- BIO tagging scheme
- Entity aggregation and grouping
- Handling split entities

**Mini-Project**: News Article Entity Analyzer - Extract and summarize key entities from news articles automatically.

---

### Module 3: Question Answering

**Time**: ~2 hours | **Difficulty**: Beginner-Intermediate | **Prerequisites**: Modules 1-2

**Description**: Explore extractive question answering where models find answers within provided context. Learn about span extraction and confidence interpretation.

**Key Concepts**:
- Extractive vs. generative QA
- Start/end position prediction
- SQuAD 2.0 and unanswerable questions
- Context window limitations

**Mini-Project**: Document Q&A System - Build a system that answers questions from product manuals or documentation.

---

### Module 4: Text Summarization

**Time**: ~2 hours | **Difficulty**: Intermediate | **Prerequisites**: Modules 1-3

**Description**: Learn how encoder-decoder models compress long text while preserving key information. Explore generation parameters and their effects.

**Key Concepts**:
- Extractive vs. abstractive summarization
- Encoder-decoder architecture
- Beam search and generation parameters
- Hallucination and factual accuracy

**Mini-Project**: Article Digest Generator - Create summaries of different lengths (tweet, paragraph, abstract) for any article.

---

### Module 5: Text Generation

**Time**: ~2 hours | **Difficulty**: Intermediate | **Prerequisites**: Modules 1-4

**Description**: Dive into autoregressive text generation and learn how models generate text token by token. Master the creativity-coherence tradeoff.

**Key Concepts**:
- Next-token prediction
- Temperature and its effect on randomness
- Top-k and top-p (nucleus) sampling
- Repetition penalty

**Mini-Project**: Creative Writing Assistant - Generate story continuations with configurable mood and style.

---

### Module 6: Zero-Shot Classification

**Time**: ~2 hours | **Difficulty**: Intermediate | **Prerequisites**: Modules 1-5

**Description**: Classify text into categories without task-specific training. Learn how Natural Language Inference (NLI) enables zero-shot capabilities.

**Key Concepts**:
- Zero-shot learning concept
- NLI-based classification
- Hypothesis templates
- Multi-label classification
- Label design best practices

**Mini-Project**: Custom Content Tagger - Tag social media posts with user-defined categories without any training.

---

### Module 7: Translation

**Time**: ~2 hours | **Difficulty**: Intermediate | **Prerequisites**: Modules 1-6 (especially 4)

**Description**: Explore sequence-to-sequence models for translation. Learn about language pairs, multilingual models, and translation quality evaluation.

**Key Concepts**:
- Encoder-decoder for translation
- Language pair models vs. multilingual models
- BLEU score concept
- Handling idioms and cultural context

**Mini-Project**: Multilingual Content Adapter - Translate marketing copy with quality checks and style preservation.

---

### Module 8: Text Similarity with Embeddings

**Time**: ~2 hours | **Difficulty**: Intermediate | **Prerequisites**: Modules 1-7

**Description**: Learn how models represent text as numerical vectors. Use embeddings for similarity comparison, search, and clustering.

**Key Concepts**:
- Text embeddings and vector representations
- Pooling strategies (CLS, mean, max)
- Cosine similarity
- Semantic vs. lexical similarity
- Sentence-transformers models

**Mini-Project**: Semantic FAQ Matcher - Match user questions to FAQ answers using semantic similarity.

---

### Module 9: Sentiment with Different Models

**Time**: ~2 hours | **Difficulty**: Intermediate-Advanced | **Prerequisites**: Modules 1-8

**Description**: Compare how different models approach the same task. Learn why training data matters and how to choose the right model.

**Key Concepts**:
- Binary vs. multi-class sentiment
- Training data influence on model behavior
- Model cards and documentation
- Ensemble methods
- Domain-specific considerations

**Mini-Project**: Multi-Model Sentiment Dashboard - Analyze text with multiple models and identify disagreements.

---

### Module 10: Pipeline Internals (Capstone)

**Time**: ~2 hours | **Difficulty**: Advanced | **Prerequisites**: Modules 1-9 (all)

**Description**: The capstone notebook that ties everything together. Learn what happens inside a Hugging Face pipeline: tokenization, model inference, and post-processing.

**Key Concepts**:
- Three-stage pipeline architecture
- Manual tokenization and special tokens
- Attention masks and padding
- Logits and softmax
- torch.no_grad() for inference
- Custom pipeline building

**Mini-Project**: Custom Pipeline Builder - Implement a classification pipeline from scratch with logging and metrics.

---

## Quick Reference (Cheat Sheet)

### 1. Fill-Mask
```python
from transformers import pipeline
fill_mask = pipeline("fill-mask")
result = fill_mask("The capital of France is [MASK].")
```

### 2. Named Entity Recognition
```python
from transformers import pipeline
ner = pipeline("ner", grouped_entities=True)
result = ner("Elon Musk founded SpaceX in California.")
```

### 3. Question Answering
```python
from transformers import pipeline
qa = pipeline("question-answering")
result = qa(question="How tall is it?", context="The tower is 330 meters tall.")
```

### 4. Text Summarization
```python
from transformers import pipeline
summarizer = pipeline("summarization")
result = summarizer(article, max_length=50, min_length=20)
```

### 5. Text Generation
```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("In the year 2050,", max_length=50, num_return_sequences=3)
```

### 6. Zero-Shot Classification
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification")
result = classifier(text, candidate_labels=["tech", "sports", "politics"])
```

### 7. Translation
```python
from transformers import pipeline
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
```

### 8. Feature Extraction (Embeddings)
```python
from transformers import pipeline
extractor = pipeline("feature-extraction")
embeddings = extractor("Your text here")
```

### 9. Sentiment Analysis
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
```

### 10. Manual Pipeline
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForSequenceClassification.from_pretrained("model-name")

inputs = tokenizer("Your text", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=-1)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `OutOfMemoryError` | Use a smaller model (distilbert instead of bert) or reduce batch size |
| Model download slow | Check internet connection; models are cached after first download |
| `CUDA out of memory` | Add `model.cpu()` to run on CPU instead of GPU |
| `Token indices overflow` | Increase `max_length` or truncate longer texts |
| Different results each run | Set `torch.manual_seed(42)` for reproducibility |

### Model Download Locations

Models are cached at: `~/.cache/huggingface/hub/`

To clear cache: `rm -rf ~/.cache/huggingface/hub/`

### Getting Help

1. Check the [Hugging Face Documentation](https://huggingface.co/docs/transformers)
2. Search the [Hugging Face Forums](https://discuss.huggingface.co/)
3. Look up model-specific issues on the model's page at `huggingface.co/MODEL_NAME`

---

## What's Next?

After completing this course, you're ready to explore:

### Immediate Next Steps

1. **Fine-tuning**: Train models on your own data
   - Start with the [Hugging Face Course](https://huggingface.co/course)
   - Try fine-tuning on a small dataset with Trainer API

2. **Larger Models**: Experiment with larger, more capable models
   - Try `microsoft/deberta-v3-large` for classification
   - Explore `google/flan-t5-large` for text generation

3. **Multimodal**: Combine text with images
   - Vision Transformers (ViT) for image classification
   - CLIP for image-text similarity

### Advanced Topics

- **Quantization**: Make models smaller and faster
- **PEFT/LoRA**: Efficient fine-tuning with fewer parameters
- **Deployment**: Serve models with FastAPI or Gradio
- **RAG**: Retrieval-Augmented Generation for knowledge-grounded responses

### Recommended Resources

- [Hugging Face Course](https://huggingface.co/course) - Official free course
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - Comprehensive book
- [Jay Alammar's Blog](https://jalammar.github.io/) - Visual explanations of transformers
- [Papers With Code](https://paperswithcode.com/) - Find state-of-the-art models

---

## Course Completion Checklist

- [ ] Module 1: Fill-Mask completed
- [ ] Module 2: NER completed
- [ ] Module 3: Question Answering completed
- [ ] Module 4: Summarization completed
- [ ] Module 5: Text Generation completed
- [ ] Module 6: Zero-Shot Classification completed
- [ ] Module 7: Translation completed
- [ ] Module 8: Embeddings completed
- [ ] Module 9: Sentiment Models completed
- [ ] Module 10: Pipeline Internals completed
- [ ] All mini-projects built
- [ ] Experimented with custom inputs

**Congratulations on completing the course!**
