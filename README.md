# RAG QA System

## Overview
This project is adapted and extended from the GitHub repository:
*https://github.com/techwithtim/LocalAIAgentWithRAG*

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Ollama (Llama 3)**, Chroma, and mxbai-embed-large to answer questions based on a custom CSV dataset

Users can easily replace the dataset with their own CSV file to build a domain-specific question-answering system.

---

## Features

* Semantic search over custom CSV data
* Context-aware answers using **LLM (Llama 3 via Ollama)**
* Plug-and-play dataset (just replace CSV file)
* Interactive CLI chatbot

---

## How It Works (RAG)

RAG stands for **Retrieval-Augmented Generation**.

Workflow:

1. User asks a question
2. The system converts it into an embedding
3. Retrieves top-k relevant rows from the CSV (via vector DB)
4. Injects them into the prompt
5. LLM generates an answer

The model answers based on your data, not just its training.

---

## Using Your Own Dataset

### Step 1: Prepare your CSV file

Your CSV should contain at least one text column.
Example:

```csv
Title,Review,Rating,Date
Great pizza,The crust was amazing,5,2024-01-01
Bad service,Waited too long,2,2024-01-02
```

---

### Step 2: Replace the dataset file

Put your CSV file in the project directory, e.g.:

```bash
my_data.csv
```

---

### Step 3: Update the filename in code

In `vector.py`, change:

```python
df = pd.read_csv("realistic_restaurant_reviews.csv")
```

to:

```python
df = pd.read_csv("my_data.csv")
```

---

### Step 4: Adjust text fields (if needed)

If your column names are different, update:

```python
page_content=row["Title"] + " " + row["Review"]
```

For example:

```python
page_content=row["content"]
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

### 2. Activate

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### Requirements

```txt
langchain
langchain-ollama
langchain-chroma
pandas
```

---

### 4. Install and run Ollama

Install Ollama and pull models:

```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```

---

### 5. Run the app

```bash
python main.py
```

---

## Notes on Customization

* You can use this system for:

  * Product reviews
  * Legal documents
  * Personal notes
  * Knowledge bases
* Only requirement:
  👉 You must define how text is extracted into `page_content`
