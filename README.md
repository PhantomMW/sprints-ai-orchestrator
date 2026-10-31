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

* Retrieves learner information from SQLite using safe parameterized domain tools (`get_learner_by_id_tool`, `search_learners_by_track_tool`, `get_track_aggregate_stats_tool`).
* **Zero Arbitrary SQL Execution:** Raw SQL tool (`learner_sql_tool`) has been removed from the agent registry, eliminating arbitrary SQL injection and browsing risks.
* **k-Anonymity Privacy on Cohort Stats:** `get_track_aggregate_stats()` enforces a minimum cohort threshold (`MIN_COHORT_SIZE = 3`) to prevent learner re-identification in sparse tracks (distinct from noise-based differential privacy).
* Recommends `get_learner_by_id()` as the primary parameterized lookup to ensure strict single-learner isolation and prevent cross-learner data exposure.
* Underlying query helper enforces a 5-row maximum, column allowlisting, read-only mode (`mode=ro`), and blocks SQL comments and system metadata (`sqlite_master`, `pragma`).
* Uses ChromaDB to retrieve relevant knowledge.
* Implements Agentic RAG with retrieval self-reflection.
* Reformulates search queries when retrieved evidence is insufficient (up to 3 retrieval attempts).
* Generates structured career recommendations using Pydantic.
* Distinguishes verified learner information from AI-generated recommendations.
* Does not claim that an employer has an active vacancy without verified vacancy data.

### 3.2 Python Assessor Agent

* Accepts Python code, assignment requirements, and optional test assertions (`test_suite`).
* Runs the **Automated Test Runner** (`src/tools/test_runner.py`) with **Master-Worker Process Isolation** inside the Docker sandbox:
  * **Supervisor & Worker Process Architecture:** The parent harness process (Supervisor) NEVER executes untrusted student code. It spawns a dedicated child Worker process via `os.fork()`.
  * **Reflection & Frame Inspection Neutralization:** Before executing learner code, the Worker process neutralizes `sys._getframe`, `sys._current_frames`, `sys.settrace`, and `sys.setprofile` to prevent walking or clearing parent call frames.
  * **Low-Level Attack Block:** Intercepts and blocks dangerous modules (`ctypes`, `_ctypes`) in `sys.modules`.
  * **Worker Suicide Trapping:** If learner code calls `os._exit(0)` or `sys.exit()`, the Supervisor safely catches the abnormal exit via `_safe_waitpid` without crashing the harness.
  * **Bounded IPC Payloads:** Worker-to-Supervisor IPC pipe reads are strictly capped at 512 KiB (and 64 KiB for assertion sub-forks) to prevent memory exhaustion attacks.
  * **Per-Assertion Sub-Forks:** Each unit test assertion executes in its own isolated sub-fork with piped IPC, preventing test contamination.
  * **Host Fallback:** On host systems without `os.fork()` (e.g. Windows), falls back safely to execution timeouts and in-process boundaries.
* **Fail-Closed Guardrails in Assessor Agent:**
  * Aborts immediately with `status: "error"` and message `"Unable to verify: Automated test runner execution failed"` before calling Gemini if the runner crashes or fails.
  * Automatically caps grades at 50/100 and enforces `status: "fail"` if automated unit assertions fail.
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

### 3.3 Evaluator-Optimizer

* Evaluates successful agent responses.
* Checks factual accuracy, grounding, relevance, clarity, and completeness.
* Optimizes responses based on evaluator feedback.
* Enforces a maximum of three optimization iterations.
* Preserves verified facts and assessment results.

### 3.4 Human-in-the-Loop

* Uses LangGraph `interrupt()` to pause selected recommendations.
* **Reviewer Authentication & RBAC Governance:** Requires authenticated reviewer identity against authorized institutional roles (`AUTHORIZED_HITL_REVIEWERS`), rejecting unauthorized approval attempts automatically.
* **Configurable Employer Tiers:** Qualifying placement review triggers are configurable via the `TIER_1_EMPLOYERS` environment variable (default: `fawry,vodafone,instapay`).
* Resumes execution using LangGraph `Command(resume=...)`.
* **Dual Checkpointing Architecture:** The core LangGraph workflow in `build_orchestrator()` defaults to `MemorySaver()` for isolated, fast unit tests, while `main.py` initializes a durable `SqliteSaver` (with WAL mode, 5-second busy timeout, and thread isolation) to guarantee restart-safe sessions and durable Human-in-the-Loop review state across restarts.

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
| Checkpointing        | Dual: MemorySaver (Default) & SqliteSaver (Persistent) |
| Testing & CI         | Pytest, GitHub Actions         |

## 5. Project Structure

```text
sprints-ai-orchestrator/
├── .github/
│   └── workflows/
│       └── tests.yml            # Automated CI test workflow
├── sandbox/
│   └── Dockerfile               # Non-root (UID 10001) isolated Python execution container
├── src/
│   ├── agents/
│   │   ├── advisor_agent.py
│   │   ├── assessor_agent.py    # Fail-closed grading & test runner integration
│   │   ├── evaluator_optimizer.py
│   │   └── orchestrator.py      # Dual checkpointer support (MemorySaver default)
│   ├── tools/
│   │   ├── test_runner.py       # Anti-tampering deterministic unit test runner
│   │   ├── docker_executor.py   # Bounded Docker executor with timeout & output caps
│   │   ├── db_queries.py        # Safe parameterized domain operations & cohort privacy
│   │   ├── sqlite_checkpoint.py # Persistent SQLite checkpointer with WAL & concurrency
│   │   ├── rag_retriever.py
│   │   ├── code_repl.py
│   │   ├── document_loader.py
│   │   └── audio_transcriber.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── auth.py              # Cryptographic HMAC session tokens & RBAC governance
│   │   └── logger.py            # Structured JSON telemetry and observability
│   ├── state.py
│   ├── llm.py
│   └── main.py                  # Interactive CLI and multi-stage demonstration
├── docs/
│   ├── DEMO_SCRIPT.md
│   ├── SPRINTS_AI_ORCHESTRATOR_FINAL_REPORT.pdf
│   ├── SPRINTS_AI_ORCHESTRATOR_PRESENTATION.pptx
│   └── TECHNICAL_DOCUMENTATION.md
├── tests/
│   ├── test_agents/
│   │   ├── test_agentic_rag.py       # Multi-track RAG quality & trust prioritization
│   │   ├── test_assessor_agent.py    # Fail-closed grading & runner integration
│   │   ├── test_e2e_integration.py   # End-to-end multi-agent lifecycle tests
│   │   ├── test_evaluator_optimizer.py
│   │   └── test_orchestrator_evaluator.py
│   └── test_tools/
│       ├── test_auth.py              # PBKDF2 reviewer login, replay store, caller tokens
│       ├── test_sqlite_checkpoint.py # Checkpointer persistence, coordinate binding, HMAC
│       ├── test_db_queries.py        # Parameterized queries, caller token RBAC, k-anonymity
│       ├── test_test_runner.py       # Master-worker process isolation, adversarial suites
│       └── test_docker_executor.py   # Docker runtime, timeout, truncation & isolation
├── pytest.ini
├── LICENSE
├── requirements.txt
├── README.md
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

The comprehensive test suite currently consists of **112 automated tests across 13 suites (111 passed in CI on Ubuntu with Docker, 1 skipped when live Gemini API key is absent)** with a 100% pass rate among runnable tests, covering:
* **Agent Workflows & Governance (21 tests):** Multi-track Agentic RAG retries, quality scoring, source trust prioritization, out-of-domain detection, evaluator-optimizer iteration limits, routing, Human-in-the-Loop interrupt/resume cycles, cryptographic HMAC session token verification, token signature tampering rejection, replay attack defense, assessor single-pass grading, fail-closed runner crash abortion, cryptographic caller token propagation in advisor node, and live Gemini API integration.
* **End-to-End Multi-Agent Integration (2 tests):** Complete lifecycle test traversing cryptographic caller token verification, learner DB lookup, Career Advisor RAG retrieval, Evaluator-Optimizer loop, conditional HITL review interrupt, human reviewer PBKDF2 login, HMAC session token issuance, graph resumption, and durable SQLite checkpoint restoration across instances.
* **Evaluator-Optimizer Invariant Guards (6 tests):** Deterministic evaluation, iteration bounding, regex-based invariant extraction (`extract_critical_invariants`), fact preservation validation (`verify_fact_preservation`), mutation detection (dropped IDs, changed tracks, altered scores), and programmatic rejection of corrupted candidates with automatic fallback.
* **Reviewer Authentication & Cryptographic Governance (11 tests):** PBKDF2-HMAC-SHA256 password hashing with secure salt and constant-time verification, institutional reviewer credential authentication, shared multi-worker brute-force login rate limiting backed by SQLite WAL mode, multi-threaded concurrent nonce race condition testing, persistent SQLite anti-replay nonce store with fail-closed behavior, production secret key separation between database caller tokens (`DB_CALLER_AUTH_SECRET`) and human reviewer tokens (`HITL_AUTH_SECRET`), cross-token verification rejection, and JSON-configured institutional directory support.
* **Database Privacy & Parameterized Operations (20 tests):** Parameterized single-learner lookups, track searches, row bounds, column allowlisting, SQL injection defense across all domain functions, caller role authorization (`requester_role`), cryptographic caller token authorization (`auth_token`), production token enforcement, aggregate query caller authorization, and k-anonymity cohort privacy (`MIN_COHORT_SIZE = 3` Minimum Cohort Suppression) with zero cohort size leakage in error messages.
* **Automated Test Runner & Adversarial Hardening (22 tests):** Master-worker process isolation (parent supervisor never executes untrusted code), concurrent non-blocking select-driven IPC draining loop eliminating OS pipe buffer deadlocks, 512 KiB payload bounding with explicit SIGKILL worker termination and clean reaping, 64 KiB child assertion bounding, test case parsing, empty code rejection, unverified status emission, combined assertion + stdout execution, full pass and partial failure metrics, compilation error handling, expected stdout matching, `SystemExit` trapping, delimiter spoofing defense, and 9 dedicated adversarial/IPC tests (monkey-patching `os.fork`/`os.waitpid`, `os._exit()` trapping, builtins corruption immunity, frame inspection defense, learner code suicide trapping, ctypes memory attack neutralization, large payload non-deadlocking IPC, and payload overflow SIGKILL handling).
* **Persistent SQLite Checkpointing & Restricted Deserialization (12 tests):** Cross-instance state restoration, Fernet AES-128-CBC authenticated encryption at rest, thread isolation, 10-thread concurrent writes with WAL mode, simulated process restart resumption, true cross-process persistence spanning separate OS Python processes, cryptographic HMAC disk tampering detection, coordinate-bound signature verification, production secret key enforcement, fail-closed anti-downgrade integrity protection (`CheckpointSecurityError`), retention policy purging (`purge_old_checkpoints`), and `SafeCheckpointUnpickler` blocking dangerous RCE gadget modules (`os`, `subprocess`, `sys`, `ctypes`, `eval`).
* **Quantitative RAG Retrieval Benchmark (1 test):** Automated regression test validating Precision @ 3 >= 0.80, Domain Recall = 100%, Groundedness Ratio = 100%, and sub-second retrieval latency (< 1000ms).
* **Structured Logging & Observability (2 tests):** Structured JSON log formatting, telemetry attributes, duration tracking, and event emission.
* **Docker Execution & MicroVM Runtime (16 tests):** Pinned immutable base image digest (`python:3.10-slim@sha256:6ff50646...`), configurable gVisor (`runsc`) and Kata Containers MicroVM support (`SPRINTS_DOCKER_RUNTIME`), runtime inspection (`get_sandbox_containment_mode`), bounded 64 KiB stdout/stderr streams, empty/invalid submission handling, mock-based timeout and container cleanup, popen exception safety, real container execution, non-zero exit handling, syntax error capture, 5-second runtime timeout enforcement, massive output truncation, read-only root restriction, and network isolation.

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
* Checkpointing is handled locally via `SqliteSaver` SQLite database; distributed deployments would benefit from PostgreSQL checkpointers.
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
