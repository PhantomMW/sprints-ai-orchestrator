# 3–5 Minute Demo Script
## Sprints Multi-Agent AI Career & Assessment Orchestrator

**Target duration:** 3 to 5 minutes  
**Demo mode:** Local interactive / automated CLI demo (`python src/main.py --mode demo` or `--mode interactive`). Automated test suite (`pytest -v`) with 121 automated tests across 13 suites (120 passed in CI on Ubuntu with Docker, 1 skipped [Live Gemini API connectivity]) and GitHub Actions CI.

> **Live vs Deterministic Fallback Notice:** The test runner with master-worker isolation, SQLite parameterized queries, HMAC authentication tokens, and LangGraph checkpoint state interruption/resumption execute 100% live. Gemini LLM inference falls back to deterministic structured outputs when `SPRINTS_LIVE_API` is not configured.

---

## 0:00–0:40 | Introduction & Architectural Vision

"Welcome. Today we present the **Sprints Multi-Agent AI Career & Assessment Orchestrator** (Release v1.1.0).

This system is an agentic AI prototype engineered to support technical learners through two specialized services:
1. **Personalized Career Guidance & RAG Assessment**
2. **Deterministic, Sandbox-Isolated Python Code Assessment**

At its core, a **LangGraph StateGraph** coordinates multi-agent routing, iterative optimization, and persistent state management, backed by strict guardrails:
- Safe, parameterized SQLite domain queries replacing unrestricted SQL.
- An automated unit test runner coupled with a secure Docker sandbox.
- Bounded Evaluator-Optimizer refinement loops.
- Durable SQLite checkpointing with Human-in-the-Loop (HITL) governance."

**Visual Action:** Display project architecture diagram or Presentation Slide 3.

---

## 0:40–1:30 | Career Advisor & Agentic RAG (Privacy & Security First)

"The first major workflow is the **Career Advisor**.

When a learner profile is requested, the system avoids risky open-ended SQL execution. Instead, it queries the learner database using **Safe Parameterized Domain Operations** (`get_learner_by_id`, `search_learners_by_track`, and `get_track_aggregate_stats`). These enforce row limits (maximum 5), restrict column visibility, and prevent unauthorized cross-learner data leakage.

For privacy assurance, our demonstrations utilize synthetic test profiles (such as `SYN-DEMO-01`) rather than identifiable learner records.

Once the profile is loaded, the Career Advisor initiates an **Agentic RAG loop** against the ChromaDB knowledge base:
- The agent retrieves curriculum context, prerequisites, and placement pathways.
- An internal evaluator inspects whether retrieved context is sufficient.
- If insufficient, it dynamically reformulates search queries (allowing up to 2 retries).
- Once sufficient, Google Gemini synthesizes structured, grounded career advice without fabricating employer requirements or open vacancies."

**Live Action:** Run `python src/main.py --mode demo` and highlight Step 1: Career Advisor execution and synthetic profile retrieval.

---

## 1:30–2:30 | Python Assessor: Docker Sandbox & Automated Test Runner

"Next is the **Python Assessor**, which evaluates student code submissions.

To ensure evaluation integrity and runtime safety, we implemented a two-tier assessment pipeline:

1. **Automated Test Runner with Master-Worker Process Isolation & Tampering Resistance:**
   - Master-Worker Process Isolation: The parent harness process NEVER executes untrusted student code directly. A dedicated worker child process handles compilation, execution, and stdout capture, while the parent harness remains immutable and traps worker exits.
   - Neutralizes reflection hooks (`_getframe`, `_current_frames`, `settrace`, `setprofile`) and blocks dangerous module imports (`ctypes`, `_ctypes`).  
   Rather than relying purely on LLM subjective grading, submissions undergo deterministic unit testing with True OS Process Isolation and single-pass execution:
   - Pre-compiles and executes learner code exactly once, capturing stdout/stderr without redundant re-runs.
   - Executes each unit assertion in a dedicated child process (`os.fork()` on Linux with copy-on-write memory and pipe IPC), providing true operating system process isolation.
   - Pre-binds safe builtins (`_safe_os`, `_safe_sys`, `_safe_json_dumps`) to protect runner internals.
   - Traps `os._exit()`, `sys.exit()`, and process crashes safely via parent `os.waitpid()` without aborting the harness.
   - Frames test metrics with random UUID session delimiters to eliminate student stdout spoofing.
   - Evaluates unit assertions and expected stdout concurrently.
   - If unit assertions fail, programmatic guardrails automatically cap the score at 50/100 and enforce a failing status.
   - **Fail-Closed Safety:** If the test runner encounters a technical failure (`status: "error"`), the Assessor Agent immediately aborts with an error response before invoking Gemini.

2. **Restricted Docker Execution Environment:**  
   Code executes in an ephemeral Docker container (`sprints-python-sandbox`) configured with:
   - Read-only root filesystem (`--read-only`) with isolated `/tmp` tmpfs.
   - Non-root user execution (`sandboxuser`, UID 10001).
   - Network disabled (`--network none`).
   - Hard execution timeout (5 seconds) with automated container termination (`docker stop` & `docker rm -f`).
   - Bounded stdout/stderr streaming (64 KiB cap with truncation indicators) preventing memory denial-of-service.

Our demo evaluates both an optimal list-average implementation and an error-prone submission to showcase immediate deterministic feedback."

**Live Action:** Highlight Step 2 in the CLI showing test runner metrics (`passed_count`, `failed_count`) and Docker execution outputs.

---

## 2:30–3:15 | Evaluator-Optimizer Loop & Persistent Checkpointing

"Every successful agent response passes through the **Evaluator-Optimizer**:
- The evaluator scores response quality across accuracy, completeness, and tone.
- If quality falls below threshold, the optimizer refines the response (capped at a hard maximum of 3 iterations).
- Immutable facts—such as Docker execution outputs and unit test scores—are strictly preserved during optimization.

The system features a **Dual Checkpointing Architecture** and **Institutional Governance**:
- `build_orchestrator()` defaults to `MemorySaver()` for isolated, lightning-fast unit tests.
- `main.py` initializes **`SqliteSaver`**, a durable SQLite-backed checkpointer storing workflow states in `data/orchestrator_checkpoints.sqlite3` with Write-Ahead Logging (WAL) and 5-second busy timeout, verified across real separate OS processes.
- **Reviewer Authentication & RBAC Governance:** Validates reviewer roles (`AUTHORIZED_HITL_REVIEWERS`), rejecting unauthorized approval attempts automatically.
- **Configurable Employer Tiers:** Qualifying placement review triggers are configurable via the `TIER_1_EMPLOYERS` environment variable (default: `fawry,vodafone,instapay`)."

**Visual Action:** Showcase CLI Step 3 displaying Evaluator-Optimizer iterations and persistent checkpoint thread IDs.

---

## 3:15–3:45 | Human-in-the-Loop (HITL) Placement Governance

"For high-stakes outcomes—such as career recommendations for learners flagged as 'Ready for Placement' with specific partner organizations—the orchestrator pauses execution at a dedicated **Human Review Node**.

The state snapshot is committed to SQLite, and the system prompts the human coordinator:
- If approved: the recommendation proceeds with verified placement advisor clearance.
- If rejected: the response is routed for revision or flagged with coordinator feedback.

This guarantees that AI never commits placement promises autonomously without institutional verification."

**Live Action:** Display Step 4 in the CLI demo simulating the HITL review pause and approved continuation.

---

## 3:45–4:15 | Automated Verification Suite & CI / CD (121 Tests: 120 Passed in CI, 1 Skipped)

"To prove engineering reliability, the repository features an automated test suite with **121 comprehensive tests** across 13 test suites:
- `test_db_queries.py`: 20 tests verifying row bounds, projection filters, parameterization, caller token cryptographic verification, caller role authorization (`requester_role`), production token enforcement, aggregate query caller authorization, Minimum Cohort Suppression (`MIN_COHORT_SIZE = 3`), and SQL injection resistance.
- `test_auth.py`: 15 tests verifying key separation between DB caller secrets and HITL reviewer secrets, external JSON directory configuration, PBKDF2-HMAC-SHA256 password hashing with salt, reviewer login authentication, brute-force login rate limiting (5 attempts / 15 min), concurrent nonce race condition atomicity, persistent SQLite anti-replay store with fail-closed behavior in production, production secret enforcement, adversarial payload tampering detection, cross-token boundary rejection, and database caller tokens.
- `test_e2e_integration.py`: 2 tests asserting complete end-to-end multi-agent workflow: signed caller token DB lookup, Career Advisor RAG retrieval, Evaluator-Optimizer loop, conditional HITL review interrupt, human reviewer PBKDF2 login, HMAC session token issuance, graph resumption, and durable SQLite checkpoint restoration across instances.
- `test_test_runner.py`: 22 tests asserting 512 KiB worker payload overflow SIGKILL handling, master-worker process isolation (parent harness process never executes untrusted code), concurrent non-blocking select-driven pipe draining eliminating OS pipe deadlocks on payloads > 64 KiB, bounded 512 KiB worker IPC payload, test case parsing, compilation error trapping, OS process isolation (`os.fork()`), delimiter protection, and 7 adversarial/IPC tampering defenses (monkey-patching `os.fork`/`os.waitpid`, `os._exit()` trapping, builtins corruption immunity, frame inspection defense, learner code suicide trapping, ctypes neutralization, and large IPC payload verification).
- `test_sqlite_checkpoint.py`: 14 tests ensuring restricted SafeCheckpointUnpickler blocking RCE gadgets (`subprocess.Popen`, `builtins.exec`, `eval`, `posix.system`, `ctypes`), database-level ciphertext tamper detection, cross-instance state persistence, Fernet AES-128-CBC authenticated encryption at rest, thread isolation, WAL concurrency, cross-process durability, HMAC tamper detection, coordinate-bound signature verification preventing thread swapping, production secret key enforcement, fail-closed anti-downgrade integrity protection (`CheckpointSecurityError`), and retention pruning.
- `test_agentic_rag.py`: 5 tests verifying self-reflective query reformulation, retrieval bounds, multi-track curriculum coverage, source trust prioritization, and out-of-domain detection.
- `test_docker_executor.py`: 17 tests validating gVisor (runsc) and Kata MicroVM support, runtime parameter sanitization and injection blocking, containment inspection (`get_sandbox_containment_mode`), pinned immutable base image digest (`python:3.10-slim@sha256:6ff50646...`), 64 KiB truncation, timeout enforcement, non-root user, read-only storage, and process failure recovery.
- `test_orchestrator_evaluator.py`: 10 tests verifying direct advisor caller token rejection in production, end-to-end graph routing, loop caps, HITL interrupts, cryptographic HMAC reviewer authentication, signature verification, anti-replay defense, and cryptographic caller token injection in advisor_node.
- `test_assessor_agent.py`: 4 tests verifying fail-closed error handling and scoring guardrails against failing unit tests.
- `test_evaluator_optimizer.py`: 6 tests verifying optimization loop bounds, programmatic fact-preservation verification (`verify_fact_preservation`), invariant extraction, and mutation rejection.
- `test_rag_benchmark.py`: 3 tests (CI metric calculation, out-of-domain unsupported claim rejection, and live ChromaDB retrieval benchmark auto-provisioned from fixtures) validating quantitative metric calculation and retrieval benchmarks enforcing Precision @ 3 >= 80% (achieved 91.7%), Domain Recall = 100%, Fact Recall >= 80%, Factual Grounding >= 85%, and Strict Groundedness >= 85%.
- `test_logger.py`: 2 tests verifying structured JSON logging and telemetry.
- `test_live_gemini_integration.py`: 1 test verifying live Gemini API connectivity.

All runnable tests run on every commit via **GitHub Actions CI (`.github/workflows/tests.yml`)**, maintaining a 100% green passing status (120 passed in CI on Ubuntu with Docker, 1 skipped for live Gemini API credentials)."

**Live Action:** Run `pytest -v` or display the GitHub Actions green checkmark badge.

---

## 4:15–4:30 | Conclusion & Roadmap

"In summary, the Sprints Multi-Agent AI Orchestrator proves how LangGraph multi-agent architectures, safe parameterized data access, automated deterministic testing, restricted Docker sandboxes, and persistent checkpoints combine to deliver an enterprise-grade AI prototype.

Future production milestones include microVM (gVisor/Firecracker) workload isolation, direct student portal UI integration, and live institutional curriculum synchronization.

Thank you!"

---

## Demo Quickstart Checklist

Before presenting, ensure:
- [ ] Virtual environment active: `.\.venv\Scripts\activate`
- [ ] Docker Desktop running with Linux engine: `docker info`
- [ ] Sandbox image built: `docker images sprints-python-sandbox:latest`
- [ ] Run full test suite: `pytest -v` (verify 121 tests collected: 95 passed locally on Windows, 26 skipped for Docker/Live API; or 120 passed, 1 skipped in CI on Ubuntu with Docker)
- [ ] Run automated demo: `python src/main.py --mode demo`
- [ ] Run interactive mode (optional): `python src/main.py --mode interactive`
- [ ] Verify verified database profile Ahmed Youssef (`SPR-1001`) is retrieved with parameterized SQL, caller token authorization, and zero hardcoded plaintext passwords in source code.
