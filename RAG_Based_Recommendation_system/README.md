# RAG-Based Wine Recommendation System using LangChain

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to generate **personalized wine recommendations** based on user queries using an LLM (`LLaMA-3.1-8B-Instant` via Groq) and a Pinecone-powered vector store.

The application is deployed using **Streamlit**, and integrates the following:
- **Vector Search** with Pinecone & HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- **Structured Output Parsing** using `PydanticOutputParser`
- **LangGraph Workflow** to compose retrieval and generation nodes
- **LangChain PromptTemplate** for formatting instructions and context-aware generation


---

## Tech Stack

- Streamlit
- LangGraph
- LangChain
- Pinecone
- HuggingFace
- Groq

---

## Features

- Semantic search over wine reviews using Pinecone vector index
- Context-aware wine recommendation via LLaMA-3.1-8B-Instant
- Structured output with country, wine name, price, variety, and reason
- Lightweight UI using Streamlit for interactive querying
- Clean modular pipeline using LangGraph nodes

---

## Demo:

https://rag-wine-recommeder.streamlit.app/

### Query:
> _"Suggest australian wine with cherry taste within 30$"_

### Output:

Wine Recommendation :

Wine Name: D'Arenberg 2006 The Derelict Vineyard Grenache (McLaren Vale)

Wine Variety: Grenache

Price in (USD): $29

Reason: This wine has a hint of cherry fruit and is within the price of $30.
