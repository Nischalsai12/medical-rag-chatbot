# Healthcare & Medical Knowledge Retrieval-Augmented Generation (RAG) Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for healthcare and medical knowledge retrieval, leveraging open-source LLMs (LLaMA-3, Mistral-7B, Phi-3) and semantic search for accurate, context-aware answers. This system aids in answering clinical, research, and healthcare-related questions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Evaluation & Results](#evaluation--results)
8. [Troubleshooting](#troubleshooting)
9. [Future Work](#future-work)
10. [Credits & References](#credits--references)

---

## Project Overview

**Domain:** Healthcare & Medical Knowledge Retrieval  
**Corpus:** Medical research papers, clinical guidelines, health reports from PubMed, WHO, CDC  
**Goal:** Retrieve relevant medical knowledge and generate accurate, user-friendly answers for clinical and healthcare queries.

**Key Objectives:**
- Automate responses to medical queries using up-to-date clinical information
- Use RAG to combine semantic retrieval with LLM-based generation
- Support multi-model architecture for flexibility and scalability

**Value Proposition:**
- Enhances healthcare professionals' access to knowledge
- Reduces response time in clinical decision-making
- Provides evidence-based, accurate answers grounded in medical literature

---

## Features
- **Retrieval-Augmented Generation:** Combines semantic search (Sentence-BERT + FAISS) with LLMs for grounded medical answers
- **Multiple LLMs Supported:** LLaMA-3, Mistral-7B, Phi-3 (selectable based on performance and resource requirements)
- **Multilingual:** English, Spanish, French (automatic translation of queries and responses)
- **Medical Focus:** Tailored for clinical and research queries, leveraging authoritative sources like PubMed and WHO
- **Web Interface:** Streamlit-based, with chat history, query input, and configuration options
- **Performance Metrics:** Real-time display of retrieval/generation times, memory usage, and model performance
- **User Feedback:** Rate answers and provide feedback to improve system performance
- **Sample Questions:** Quick testing with sample clinical queries and guidelines
- **Resource Adaptation:** Supports both CPU and GPU configurations for various model sizes

---

## System Architecture

1. **User Query:** Entered via web interface (e.g., "What is the treatment for diabetes?")
2. **Preprocessing:** Clean, translate, and augment query if needed
3. **Embedding & Retrieval:** Query embedded using Sentence-BERT, similar research papers or guidelines retrieved from FAISS index
4. **Context Construction:** Top retrievals are used as context for the LLM model
5. **Response Generation:** LLM generates a response based on the context and query
6. **Display & Feedback:** Response shown to the user along with the source, and feedback is collected for system improvement

**Main Components:**
- Data Processing (`src/data_processing.py`)
- Embedding & Retrieval (`src/embedding.py`)
- LLM Response Generation (`src/llm_response.py`)
- Utilities & Evaluation (`src/utils.py`)
- Web Interface (`app.py`)

---

## Project Structure
```
healthcare-medical-rag-chatbot/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md # User documentation manual
├── data/ # Medical datasets (CSV, JSON)
├── src/ # Source code modules
│ ├── data_processing.py # Data loading, cleaning, augmentation
│ ├── embedding.py # Embedding generation, FAISS retrieval
│ ├── llm_response.py # LLM loading, prompt generation
│ ├── utils.py # Evaluation, metrics, memory utils
│ └── init.py
├── embeddings/ # Persisted embedding model/index
├── docs/ # Project documentation (reports, guides)
├── venv/ # Python virtual environment
```



---

## Installation & Setup

### Prerequisites
- Python 3.8+
- 16GB+ RAM (32GB+ recommended for larger models)
- Optional: CUDA-compatible GPU with 8GB+ VRAM (15GB+ for Mistral-7B)
- Internet connection (for downloading models/datasets)
- Swap space (4GB) recommended for <32GB RAM

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/healthcare-medical-rag-chatbot.git
   cd healthcare-medical-rag-chatbot

**Create and Activate a Virtual Environment**

**Linux/macOS:**

```bash

python -m venv venv
source venv/bin/activate
```
**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```
**Install Dependencies**

```bash
pip install -r requirements.txt
```
(Optional) **Download NLTK Resources**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```
(Optional) **Add Swap Space (Linux, for low memory systems)**

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```
**Running the Application**
```bash
python chatbot.py
```
Access the web interface at http://localhost:8501.

## Usage Guide
**Interface Overview**
- **Main Chat Area:** Interact with the chatbot to ask medical queries

- **Retrieved Information:** See the research papers or clinical guidelines used for the answer

- **Configuration Sidebar:** Select model, language, and other settings

- **Sample Questions:** Click to test common clinical and healthcare-related queries

- **Performance Metrics:** View retrieval/generation times and memory usage

- **Feedback:** Rate responses and leave comments for improvements

## Configuration Options
- **Dataset Source:** PubMed, WHO guidelines, local CSV/JSON

- **FAQ Augmentation:** Enable paraphrasing for better query handling

- **Language:** English, Spanish, French

- **LLM Model:** LLaMA-3 (balanced), Mistral-7B (highest quality), Phi-3 (faster, less resource-intensive)

- **Memory Usage Display:** See current RAM/VRAM utilization

## Interacting with the Chatbot
**1.** Type a medical query (e.g., "What is the best treatment for hypertension?")

**2.** View the chatbot's response and the relevant literature used to generate it

**3.** Rate the answer (1-5) and leave feedback if needed

**4.** Test with sample questions for fast demonstration of functionality

## Performance Tips
- Use Mistral-7B for detailed answers and deep medical knowledge

- Use LLaMA-3 for balance between performance and resource usage

- Preload embeddings for faster responses

- Use GPU for best performance with larger models

## Evaluation & Results
**Retrieval Performance**
- **RAG (Sentence-BERT) outperforms keyword-based search:**

     - Precision@1: 0.85 (RAG) vs. 0.72 (TF-IDF)

     - Recall@3: 0.80 (RAG) vs. 0.65 (TF-IDF)

- Dense embeddings handle clinical and research queries with high relevance and accuracy

## Response Quality
- **BLEU/ROUGE-L/Word Overlap:** RAG+LLM consistently outperforms baselines

- **Human Ratings:** 4.5/5 (Phi-3+RAG), 4.2/5 (LLaMA-3+RAG), 4.7/5 (Mistral-7B+RAG)

- **Multilingual:** Maintains ~90% of English performance in Spanish/French medical queries

## System Performance
**Retrieval time:** ~0.02s (FAISS)

**Generation time:** 2-5s (GPU), 15-60s (CPU, large models)

**Memory usage:** 3-32GB RAM depending on model

## Troubleshooting
**Out of Memory Errors:**

- Switch to LLaMA-3 or Phi-3 for smaller memory footprint

- Disable FAQ augmentation

- Add swap space

- Reduce embedding batch size

**Slow Response Time:**

- Use LLaMA-3 or Phi-3 for better efficiency

- Preload embeddings for faster retrieval

- Use GPU for optimal performance with larger models

**Model Loading Failures:**

- Ensure proper internet connection for downloading models

- Check available disk space

- Update the transformers library if needed

**CUDA Errors:**

- Ensure GPU drivers are up to date

- Check CUDA compatibility with installed libraries

- Set CUDA_VISIBLE_DEVICES=0 if using multiple GPUs

For more help, see the GitHub issues or consult the relevant documentation.

## Future Work
**Hybrid Retrieval:** Integrate sparse search techniques (e.g., BM25) for improved precision

**Model Fine-tuning:** Train domain-specific LLMs with clinical data

**Multi-turn Conversations:** Handle back-and-forth exchanges for more complex queries

**Integration with EHR Systems:** Automate data extraction from Electronic Health Records

**Scaling:** Optimize for cloud deployment, microservices, and health data security

**User Feedback Loop:** Incorporate feedback to continuously improve system performance

## Credits & References
**Team:**

- Nitish Adla
- Kavya Rampalli
- Sai Nischal Dasari
  
**Key References:**

- Lewis et al. (2020) "Retrieval-augmented generation for knowledge-intensive NLP tasks"

- Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

- PubMed, WHO guidelines, CDC for medical datasets









