# AGENTIC AI

In this repository, I explore and experiment with **AI agents**, **multi-agent systems**, **RAG**, **Agentic RAG**, and **Corrective RAG (CRAG)** using the **LangChain** and **LangGraph** frameworks — enabling dynamic, tool-augmented agent collaboration across diverse tasks.


---

## Repository Structure

AGENTIC_AI/
│
├── Langgraph_Multi_Agents/
│ ├── Network Multi Agents/ 
│ └── Supervisor Multi Agents/ 
│
├── langgraph/
│ ├── AI Travel Agent/ # LangGraph-powered agent for travel planning (LangChain + tools)
│ ├── Corrective RAG (CRAG)/ 
│ ├── langgraph_intro/ 
│ └── tools/ # Custom tools: Tavily, REPL Coder, Search API, etc.
│
├── requirements.txt
└── README.md

---

## Langgraph_Multi_Agents

### 1. Network Multi Agents

📄 [`1_Network_Multi_Agents.ipynb`](https://github.com/Divya-Chintala/Agentic_AI/blob/e7dd07492185ad69f7effbfd818aa1f835e5c2ed/Langgraph_Multi_Agents/1_Network_Multi_Agents.ipynb)  

**Use-Case:** A cooperative task executed by two agents using a LangGraph-based workflow.

| Agent       | Role             | Tool Bound To Agent   |
|-------------|------------------|------------------------|
| `researcher` | Web search       | `tavily_tool`          |
| `chartmaker` | Chart generation | `REPL_Coder_tool`      |

**Prompt Example:**

```python
app.invoke({
  "messages": [
    ("user", "get the UK's GDP over the past 3 years, then make a line chart of it. Once you make the chart, finish.")
  ]
})```

---

### 2. Supervisor Multi Agents

-  In Progress

---

## langgraph/

### AI Travel Agent

Agentic travel assistant using LangGraph + custom tools to:

- Fetch destinations

- Check live weather

- Suggest hotels

- Build itineraries within budget

---

### Corrective RAG (CRAG)

A Corrective RAG pipeline where:

- Initial retrieval results are validated by a tool-checking node

- Hallucinations/errors are corrected before synthesis

---

###  langgraph_intro

Getting started with langGrapgh:

- LangGraph workflows

- Agent state propagation

- Tool invocation flows

---

### tools/

Modular tool wrappers used across agents:

- TavilyTool for web search

- REPLCoderTool for code + plots

- Weather API, Building custom tools and Tool binding with LLm

---

## Run Locally

# Step 1: Clone repo
git clone https://github.com/Divya-Chintala/Agentic_AI.git
cd AGENTIC_AI

# Step 2: Create virtual environment


# Step 3: Install dependencies
pip install -r requirements.txt


