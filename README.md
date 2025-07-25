# AGENTIC AI

In this repository, I explore and experiment with **AI agents**, **multi-agent systems**, **RAG**, **Agentic RAG**, and **Corrective RAG (CRAG)** using the **LangChain** and **LangGraph** frameworks — enabling dynamic, tool-augmented agent collaboration across diverse tasks.


---

## Langgraph_Multi_Agents

### 1. Network Multi Agents

 [`1_Network_Multi_Agents.ipynb`](https://github.com/Divya-Chintala/Agentic_AI/blob/e7dd07492185ad69f7effbfd818aa1f835e5c2ed/Langgraph_Multi_Agents/1_Network_Multi_Agents.ipynb)  

**Use-Case:** A cooperative task executed by two agents using a LangGraph-based workflow.

| Agent       | Role             | Tool Bound To Agent   |
|-------------|------------------|------------------------|
| `researcher` | Web search       | `tavily_tool`          |
| `chartmaker` | Chart generation | `REPL_Coder_tool`      |



---

### 2. Supervisor Multi Agents

-  In Progress


 [`2_Supervisor__Multi_Agents.ipynb`](https://github.com/Divya-Chintala/Agentic_AI/blob/29fb465d93a22eb80ebb039a02aa9d79b6be85fd/Langgraph_Multi_Agents/2_Supervisor_Multi_Agents.ipynb)  

---

## langgraph/

### AI Travel Agent

[Agentic travel assistant](https://github.com/Divya-Chintala/Agentic_AI/tree/60a29dce7c4c624a5aee7c65b0b7e45f9a7fe9c2/langgraph/AI_Travel_Agent) using LangGraph + custom tools to:

- Fetch destinations

- Check live weather

- Suggest hotels

- Build itineraries within budget

---

### Corrective RAG (CRAG)

A [Corrective RAG](https://github.com/Divya-Chintala/Agentic_AI/blob/60a29dce7c4c624a5aee7c65b0b7e45f9a7fe9c2/langgraph/Corrective_RAG_CRAG.ipynb) pipeline where:

- Initial retrieval results are validated by a tool-checking node

- Hallucinations/errors are corrected before synthesis

---

###  langgraph_intro

Getting started with langGrapgh: [Intro](https://github.com/Divya-Chintala/Agentic_AI/tree/60a29dce7c4c624a5aee7c65b0b7e45f9a7fe9c2/langgraph)

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

```
# Step 1: Clone repo
git clone https://github.com/Divya-Chintala/Agentic_AI.git
cd AGENTIC_AI

# Step 2: Create virtual environment


# Step 3: Install dependencies
pip install -r requirements.txt


