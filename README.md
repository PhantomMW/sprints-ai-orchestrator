# Sprints Multi-Agent AI Career & Assessment Orchestrator

An AI-powered multi-agent system designed to automate learner support, restricted Python assignment assessment, and career guidance for Sprints learners.

Built with **LangGraph**, **LangChain**, **Google Gemini**, **ChromaDB**, **SQLite**, and **Docker**, the system combines specialized AI agents into a centralized workflow with structured outputs, retrieval evaluation, automated test validation, and human-in-the-loop approval.

---

## 1. Project Overview

The Sprints AI Orchestrator addresses three major workflows:

1. **Career Guidance:** Provides personalized career recommendations using verified learner profiles and retrieved knowledge.
2. **Python Assessment:** Executes submitted Python code within a restricted Docker-based execution prototype and generates structured assessment feedback.
3. **Human-in-the-Loop:** Pauses selected career recommendations for human review before final approval.

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
       SQLite Learner DB      Docker Prototype
       (Privacy-Restricted)   (Resource-Bounded)
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

* Retrieves learner information from SQLite using privacy-hardened queries.
* Recommends `get_learner_by_id()` as the primary parameterized lookup to ensure strict learner isolation and prevent cross-learner data exposure.
* SQL tool strictly enforces a 5-row maximum, column allowlisting, and blocks SQL comments and system metadata (`sqlite_master`, `pragma`).
* Uses ChromaDB to retrieve relevant knowledge.
* Implements Agentic RAG with retrieval self-reflection.
* Reformulates search queries when retrieved evidence is insufficient (up to 3 retrieval attempts).
* Generates structured career recommendations using Pydantic.
* Distinguishes verified learner information from AI-generated recommendations.
* Does not claim that an employer has an active vacancy without verified vacancy data.

### 3.2 Python Assessor Agent

* Accepts Python code, assignment requirements, and optional test assertions (`test_suite`).
* Runs the **Automated Test Runner** (`src/tools/test_runner.py`) to execute deterministic unit assertions inside Docker sandbox, producing objective pass/fail metrics.
* Combines deterministic assertion metrics with Gemini qualitative code review (algorithmic efficiency, readability, PEP 8).
* Executes code inside a restricted Docker container prototype (`sprints-python-sandbox:latest`).
* Applies strict isolation and resource bounds:
  * Network disabled (`--network none`).
  * Read-only container root filesystem (`--read-only`).
  * Non-root user (UID 10001).
  * 128 MB RAM, 0.5 CPU quota, 64 PID limit.
  * Temporary storage capped at 16 MB tmpfs (`noexec`, `nosuid`).
  * All Linux capabilities dropped (`--cap-drop ALL`, `no-new-privileges`).
  * 5-second wall-clock execution timeout with automated container stopping and removal.
  * Bounded output capture: stdout and stderr capped at 64 KiB each with chunk draining to prevent pipe deadlocks.
  * Stdin-piped execution preventing shell injection and quote escaping anomalies.
* Returns structured grading feedback.
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

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python 3.10                    |
| Agent Orchestration  | LangGraph                      |
| LLM                  | Google Gemini                  |
| LLM Integration      | LangChain                      |
| Structured Outputs   | Pydantic                       |
| Vector Database      | ChromaDB                       |
| Relational Database  | SQLite (Read-Only, Restricted) |
| Embeddings           | Sentence Transformers          |
| Code Execution       | Docker Sandbox Prototype       |
| Checkpointing        | LangGraph MemorySaver          |
| Testing & CI         | Pytest, GitHub Actions         |

## 5. Project Structure

```text
sprints-ai-orchestrator/
│
├── .github/
│   └── workflows/
│       └── tests.yml            # Automated CI test workflow
│
├── sandbox/
│   └── Dockerfile               # Non-root, isolated Python execution container
│
├── src/
│   ├── agents/
│   │   ├── advisor_agent.py
│   │   ├── assessor_agent.py
│   │   ├── evaluator_optimizer.py
│   │   └── orchestrator.py      # Supports pluggable persistent checkpointers
│   │
│   ├── tools/
│   │   ├── test_runner.py       # Deterministic unit test runner harness
│   │   ├── docker_executor.py   # Bounded Docker executor with timeout & output caps
│   │   ├── db_queries.py        # Safe parameterized domain operations
│   │   ├── rag_retriever.py
│   │   ├── code_repl.py
│   │   ├── document_loader.py
│   │   └── audio_transcriber.py
│   │
│   ├── state.py
│   ├── llm.py
│   └── main.py                  # Interactive CLI and demonstration interface
│
├── docs/
│   ├── DEMO_SCRIPT.md
│   ├── SPRINTS_AI_ORCHESTRATOR_FINAL_REPORT.pdf
│   ├── SPRINTS_AI_ORCHESTRATOR_PRESENTATION.pptx
│   └── TECHNICAL_DOCUMENTATION.md
│
├── tests/
│   ├── test_agents/
│   │   ├── test_agentic_rag.py
│   │   ├── test_evaluator_optimizer.py
│   │   ├── test_orchestrator_evaluator.py
│   │   └── test_assessor_agent.py
│   │
│   └── test_tools/
│       ├── test_db_queries.py      # Synthetic isolated database fixture tests
│       ├── test_test_runner.py     # Deterministic unit test runner tests
│       └── test_docker_executor.py # Docker runtime, timeout, truncation & isolation tests
│
├── pytest.ini                   # Pytest configuration and path discovery
├── LICENSE                      # MIT Open Source License
├── requirements.txt             # Project dependencies (UTF-8)
├── README.md                    # Project documentation
└── .gitignore
```

**Note:** Local datasets, identifiable learner records, database files, and vector database artifacts are excluded from the public repository. They must be prepared locally before running workflows that depend on them.

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

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
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

```bash
docker build -t sprints-python-sandbox:latest -f sandbox/Dockerfile .
```

Ensure Docker Desktop is running before using the Python Assessor.

### Running the Interactive CLI

Launch the interactive terminal interface:

```bash
python src/main.py --mode interactive
```

Or run the automated demonstration:

```bash
python src/main.py --mode demo
```

## 7. Running the Tests

Run the complete test suite (agent workflows, database query privacy, and Docker runtime):

```bash
pytest -v
```

Run a specific test file:

```bash
pytest -v tests/test_tools/test_docker_executor.py
pytest -v tests/test_tools/test_db_queries.py
```

The comprehensive test suite currently consists of **45 passing tests** (100% pass rate) covering:
* **Agent Workflows & Assessment (11 tests):** Agentic RAG retries, evaluator-optimizer bounds, routing, Human-in-the-Loop interrupt/resume cycles, and Assessor Agent deterministic test runner integration.
* **Database Privacy & Domain Operations (14 tests):** Safe parameterized single-learner lookups, track searches, aggregate metrics, row bounding (5 max), column allowlisting, SQL comment and system table blocking, verified with an isolated synthetic database fixture.
* **Automated Test Runner (6 tests):** Test case parsing, deterministic assertion evaluation, pass percentage calculations, expected stdout matching, and compilation failure capture.
* **Docker Execution & Sandbox Runtime (14 tests):** Bounded 64 KiB stdout/stderr streams, empty/invalid submission handling, mock-based timeout and container cleanup, popen exception safety, real container execution, non-zero exit handling, syntax error capture, 5-second runtime timeout enforcement, massive output truncation, read-only root restriction, and network isolation.

Continuous Integration is automated via GitHub Actions (`.github/workflows/tests.yml`), running on every push and pull request.

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
* Docker execution is a restricted prototype sandbox with 64 KiB output caps, memory/CPU quotas, and dropped capabilities. It is not a certified production sandbox; untrusted public execution in production requires microVM isolation (e.g., gVisor, Firecracker) and dedicated security audits.
* Gemini API usage depends on the configured account's quota and rate limits.
* Local learner data and knowledge-base files are not distributed with the public repository.

## 10. Future Improvements

* Integrate an authenticated live job-opening database.
* Add persistent checkpoint storage (e.g., PostgreSQL).
* Expand the official curriculum knowledge base.
* Improve retrieval evaluation using a larger benchmark.
* Add a web user interface for learner requests and human reviewers.
* Upgrade prototype code execution to microVMs (gVisor or Firecracker).
* Add end-to-end integration testing.

## 11. Project Goal

The ultimate goal is to provide a unified AI-powered learner support system that reduces manual workload, improves response times, supports reliable Python assessment, and delivers evidence-based career guidance.

---

**Project:** Sprints Multi-Agent AI Career & Assessment Orchestrator
**Language:** Python
**Orchestration:** LangGraph
**LLM:** Google Gemini
**License:** MIT
