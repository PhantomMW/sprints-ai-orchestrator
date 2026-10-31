# Sprints Multi-Agent AI Career & Assessment Orchestrator

An AI-powered multi-agent system designed to automate learner support, Python assignment assessment, and career guidance for Sprints learners.

Built with **LangGraph**, **LangChain**, **Google Gemini**, **ChromaDB**, **SQLite**, and **Docker**, the system combines specialized AI agents into a centralized workflow with structured outputs, retrieval evaluation, and human-in-the-loop approval.

---

## 1. Project Overview

The Sprints AI Orchestrator addresses three major workflows:

1. **Career Guidance:** Provides personalized career recommendations using learner profiles and retrieved knowledge.
2. **Python Assessment:** Executes submitted Python code in a Docker container and generates structured assessment feedback.
3. **Human-in-the-Loop:** Pauses selected career recommendations for human review before approval.

A centralized LangGraph workflow routes requests to specialized agents and sends successful results through an Evaluator-Optimizer stage.

## 2. System Architecture

```text
                    User Request
                         |
                         v
                  LangGraph Router
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Career Advisor           Python Assessor
             |                       |
       SQLite Learner DB        Docker Sandbox
             |
        Agentic RAG
             |
          ChromaDB
             |
             +-----------+-----------+
                         |
                         v
                Evaluator-Optimizer
                         |
                  Maximum 3 Iterations
                         |
                         v
                   Human Review
                 (Conditional HITL)
                         |
                         v
                   Final Response
```

## 3. Main Features

### 3.1 Career Advisor Agent

* Retrieves learner information from SQLite.
* Uses ChromaDB to retrieve relevant knowledge.
* Implements Agentic RAG with retrieval self-reflection.
* Reformulates search queries when retrieved evidence is insufficient.
* Supports up to three retrieval attempts.
* Generates structured career recommendations using Pydantic.
* Distinguishes verified learner information from AI-generated recommendations.
* Does not claim that an employer has an active vacancy without verified vacancy data.

### 3.2 Python Assessor Agent

* Accepts Python code and assignment requirements.
* Executes code inside a Docker container.
* Applies resource restrictions, execution timeouts, and bounded output capture (64 KiB per stream).
* Returns structured grading feedback.
* Preserves actual execution results and assessment scores.
* Returns an `Unable to Verify` response when execution fails.

### 3.3 Evaluator-Optimizer

* Evaluates successful agent responses.
* Checks factual accuracy, grounding, relevance, clarity, and completeness.
* Optimizes responses based on evaluator feedback.
* Enforces a maximum of three optimization iterations.
* Preserves verified facts and assessment results.

### 3.4 Human-in-the-Loop

* Uses LangGraph `interrupt()` to pause selected recommendations.
* Requests human approval or rejection.
* Resumes execution using LangGraph `Command(resume=...)`.
* Uses checkpointing to preserve workflow state.

Human approval represents approval of a recommendation, not confirmation of employment or an active vacancy.

## 4. Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python 3.10           |
| Agent Orchestration  | LangGraph             |
| LLM                  | Google Gemini         |
| LLM Integration      | LangChain             |
| Structured Outputs   | Pydantic              |
| Vector Database      | ChromaDB              |
| Relational Database  | SQLite                |
| Embeddings           | Sentence Transformers |
| Code Execution       | Docker                |
| Checkpointing        | LangGraph MemorySaver |
| Testing              | Pytest                |

## 5. Project Structure

```text
sprints-ai-orchestrator/
│
├── sandbox/
│   └── Dockerfile
│
├── src/
│   ├── agents/
│   │   ├── advisor_agent.py
│   │   ├── assessor_agent.py
│   │   ├── evaluator_optimizer.py
│   │   └── orchestrator.py
│   │
│   ├── tools/
│   │   ├── rag_retriever.py
│   │   ├── code_repl.py
│   │   ├── docker_executor.py
│   │   ├── db_queries.py
│   │   ├── document_loader.py
│   │   └── audio_transcriber.py
│   │
│   ├── state.py
│   ├── llm.py
│   └── main.py
│
├── docs/
│   ├── DEMO_SCRIPT.md
│   ├── SPRINTS_AI_ORCHESTRATOR_FINAL_REPORT.pdf
│   ├── SPRINTS_AI_ORCHESTRATOR_PRESENTATION.pptx
│   └── TECHNICAL_DOCUMENTATION.md
│
├── tests/
│   └── test_agents/
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** Local datasets, learner records, database files, and vector database artifacts are excluded from the public repository. They must be prepared locally before running workflows that depend on them.

## 6. Installation and Setup

### Requirements

* Python 3.10
* Docker Desktop
* Google Gemini API key
* Git

### Clone the Repository

```bash
git clone https://github.com/PhantomMW/sprints-ai-orchestrator.git
cd sprints-ai-orchestrator
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Never commit `.env` or expose API credentials.

### Build the Docker Sandbox

```powershell
docker build -t sprints-python-sandbox:latest -f sandbox/Dockerfile .
```

Ensure Docker Desktop is running before using the Python Assessor.

**Important:** Local learner databases, learner exports, client source materials, and ChromaDB data must be prepared separately. They are intentionally excluded from this public repository. Use synthetic learner records for demos; never publish identifiable learner data without explicit authorization.

## 7. Running the Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Run a specific test file:

```powershell
python -m pytest -v tests/test_agents/test_orchestrator_evaluator.py
```

The latest previously reported full test run completed with **9 passing tests**. This result should be revalidated before each final release.

## 8. Knowledge Retrieval

The RAG pipeline uses ChromaDB and Sentence Transformers.

Previously recorded evaluation results:

* 3 knowledge documents.
* 27 stored chunks.
* Maximum chunk size: 236 tokens.
* Embedding dimension: 384.
* Average retrieval latency: approximately 20.06 ms.

**Important:** The current knowledge collection contains project-context documents. It should not be represented as a complete official Sprints curriculum knowledge base until the official curriculum materials are ingested and validated.

## 9. Current Limitations

* No live employer vacancy database is connected.
* Career recommendations are not employment guarantees.
* Human approval does not represent employer acceptance.
* The current RAG collection requires additional official curriculum material.
* MemorySaver provides in-memory checkpointing and is not persistent across application restarts.
* Docker execution is a prototype sandbox. Output capture is capped at 64 KiB per stream, but additional isolation and an independent security review are required before accepting untrusted public workloads.
* Gemini API usage depends on the configured account's quota and rate limits.
* Local learner data and knowledge-base files are not distributed with the public repository.

## 10. Future Improvements

* Integrate an authenticated live job-opening database.
* Add persistent checkpoint storage.
* Expand the official curriculum knowledge base.
* Improve retrieval evaluation using a larger benchmark.
* Add a user interface for learner requests and human reviewers.
* Strengthen sandbox security and resource-output limits.
* Add end-to-end integration testing.

## 11. Project Goal

The ultimate goal is to provide a unified AI-powered learner support system that reduces manual workload, improves response times, supports reliable Python assessment, and delivers evidence-based career guidance.

---

**Project:** Sprints Multi-Agent AI Career & Assessment Orchestrator
**Language:** Python
**Orchestration:** LangGraph
**LLM:** Google Gemini