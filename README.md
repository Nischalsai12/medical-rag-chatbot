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

