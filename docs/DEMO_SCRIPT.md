# 3–5 Minute Demo Script
## Sprints Multi-Agent AI Career & Assessment Orchestrator

**Target duration:** approximately 4 minutes  
**Demo mode:** local prototype; use sample learner data and a working Gemini API quota. If Gemini quota is unavailable, present the prepared screenshots/test outputs and explain that automated tests use mocks.

---

## 0:00–0:30 | Introduction

"Hello. This is the Sprints Multi-Agent AI Career and Assessment Orchestrator.

The goal is to demonstrate how a coordinated AI workflow can support learners through two services: career guidance and Python assignment assessment.

The system uses LangGraph to route requests to specialized agents, while evidence checks, structured outputs, and a human review checkpoint help control the workflow."

## 0:30–1:10 | Architecture

"At the center is a LangGraph StateGraph. The router examines the request and sends it to either the Career Advisor or the Python Assessor.

Successful results move to an Evaluator-Optimizer, which can refine the response for up to three iterations. Failed or unverifiable results bypass optimization.

The graph uses a shared typed state and MemorySaver checkpointing. MemorySaver is in-memory for the current running process."

**Show:** presentation slide 3 and the orchestrator code.

## 1:10–2:00 | Career Advisor and Agentic RAG

"The Career Advisor first retrieves the learner's record from SQLite. It then searches the ChromaDB project knowledge collection.

This is an Agentic RAG workflow: the system evaluates whether retrieved evidence is sufficient. If it is not sufficient, it reformulates the search query and retries. It allows one initial retrieval and up to two retries.

When sufficient evidence is found, Gemini generates a structured career response using the Pydantic CareerAdvice schema.

The system is designed not to invent qualifications, employer hiring requirements, or active vacancies. The current knowledge base contains project context documents, not the complete official Sprints curriculum."

**Show:** advisor workflow code and, if available, a successful sample learner response.

## 2:00–2:50 | Python Assessor and Docker

"The second workflow is the Python Assessor. It receives the learner's code, assignment instructions, and optional expected output.

The code is executed inside a Docker container configured with restrictions such as disabled networking, a read-only root filesystem, a non-root user, resource limits, and a timeout.

After execution, the assessor returns structured grading feedback: a score, feedback points, and pass status.

The current demonstration in `src/main.py` compares a correct list-average solution with an incorrect one."

**Show:** `src/main.py`, then the Docker execution result or prepared test output.

## 2:50–3:30 | Evaluator and Human Review

"Successful agent responses are checked by the Evaluator-Optimizer. The optimization loop is capped at three iterations and is instructed to preserve verified execution results and numerical grades.

There is also a Human-in-the-Loop checkpoint for a specific placement recommendation case: when Fawry is a preferred employer and the learner is marked Ready for Placement.

The graph pauses for a human approval or rejection. This approval applies only to the recommendation; it does not mean that a vacancy exists or that the employer has accepted the learner."

**Show:** presentation slide 6 and the placement review node.

## 3:30–4:00 | Validation and Closing

"During development, the complete test suite was reported as passing nine tests. The tests cover agentic retrieval, evaluator optimization, orchestration, and human review paths. The retrieval benchmark used four test cases with three runs each; its reported average was approximately 20 milliseconds in that local test setup.

The current prototype still needs a complete approved curriculum knowledge base, durable checkpoint storage, stronger sandbox output controls, broader tool-level tests, and a user-facing interface.

To conclude, this project demonstrates a modular multi-agent workflow with retrieval grounding, isolated code execution, bounded response optimization, and human oversight. Thank you."

**Show:** presentation slides 8 and 9.

---

## Demo Readiness Checklist

- [ ] Activate the project virtual environment.
- [ ] Confirm Docker Desktop is running with Linux containers.
- [ ] Confirm the sandbox image exists: `sprints-python-sandbox:latest`.
- [ ] Confirm `.env` exists locally and contains a valid Gemini API key; never show the key on screen.
- [ ] Check API quota before attempting a live Gemini demonstration.
- [ ] Run `python -m pytest -v` and capture the current output.
- [ ] Run `python -m src.main` only if live Gemini quota is available.
- [ ] Keep a prepared screenshot or recorded output available in case the API quota is exhausted.
- [ ] Do not demonstrate arbitrary public code execution or claim production-grade sandbox security.
