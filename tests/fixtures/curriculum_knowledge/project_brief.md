# Sprints Multi-Agent AI Career & Assessment Orchestrator
## Project Brief

### Business Situation

Welcome to Sprints, the premier EdTech platform bridging the talent gap in the MENA region. We are currently scaling our employer-connected tech bootcamps, processing thousands of learners aiming for placements at top Egyptian tech firms. As our cohorts grow, our human advisors and technical graders are becoming the primary bottleneck.

We need you to build the "Sprints Multi-Agent AI Career & Assessment Orchestrator." This is not a simple chatbot; we require a robust, stateful agentic ecosystem. The system must autonomously ingest our curriculum to answer student questions, securely grade Python submissions using live execution tools, and query our internal databases to understand a student's progress and target salary in EGP.

This project requires a sophisticated orchestrator-worker architecture. You will leverage foundational chaining and prompt engineering, alongside dynamic graph-based state machines. Crucially, as we deal with high-stakes career placements, fully autonomous operations carry too much risk. You must implement human-in-the-loop (HITL) escalation protocols for final placement approvals, ensuring our human experts retain ultimate oversight over critical decisions.

Please review the provided kickoff deck and recorded discovery call transcript to begin.

### Proposed Technology Stack

- Python
- LangChain
- LangGraph
- OpenAI API
- Pydantic
- ChromaDB
- Jupyter Notebooks
- Python REPL
- SQLite/SQL Databases

### Scope of Work

| Technology | Requirement |
|---|---|
| LangChain | [Sprint 1] Implement a complete Document Ingestion and Vectorization pipeline that loads Sprints curriculum documents, applies chunking with a 500-token limit and 50-token overlap, and stores them in ChromaDB. |
| Pydantic | [Sprint 1] Create strongly typed Prompt Templates and Output Parsers using Pydantic schemas to strictly format the Assessor Agent's feedback on student code into predictable JSON structures (score, feedback_points, pass_status). |
| SQLite/SQL Databases | [Sprint 1] Equip a simple ReAct agent with a custom SQL tool that allows it to query the provided 'sprints_learner_db' to retrieve a student's completed modules, EGP target salary, and preferred Egyptian employers. |
| Python REPL | [Sprint 1] Implement a custom tool using the Python REPL utility that allows the Assessor Agent to securely execute user-submitted Python snippets and capture the stdout or runtime errors to inform grading. |
| LangGraph | [Sprint 2] Architect a unified StateGraph orchestration pipeline defining a global `TypedDict` state that routes incoming Sprints learner queries dynamically to either the Advisor Agent or the Assessor Agent based on intent. |
| LangGraph | [Sprint 2] Integrate a MemorySaver checkpointer within the LangGraph orchestrator to isolate and persist multi-turn conversational state per student using unique thread IDs, preventing context amnesia across sessions. |
| LangGraph | [Sprint 2] Build an Agentic RAG workflow inside the Advisor Agent node that features a self-reflection conditional edge: if retrieved curriculum context is deemed insufficient to answer the query, the agent reformulates the query and searches again. |
| LangGraph | [Sprint 2] Implement an Evaluator-Optimizer pattern for the Assessor Agent, utilizing a secondary 'LLM as a Judge' node to review the initial grading decision against an explicit rubric before returning it to the user. |
| LangGraph | [Sprint 2] Develop a Human-in-the-Loop (HITL) feedback escalation workflow using LangGraph's interrupt primitive, which pauses graph execution and waits for human approval whenever a job placement recommendation is generated for a tier-1 partner (e.g., Fawry). |
| LangChain | [Sprint 2] Standardize communication between the Orchestrator and Worker agents by enforcing structured Agentic Protocol message schemas (HumanMessage, AIMessage, ToolMessage) traceable via verbose logs. |

### Acceptance Criteria

- The system must use LangGraph for all multi-agent orchestration, avoiding deprecated linear chains for dynamic routing.
- All generated Python code and agentic tools must include detailed docstrings mapping to the specific Sprints curriculum functionality.
- The vector database implementation (ChromaDB) must consistently retrieve context within 1000 milliseconds to avoid orchestrator timeouts.
- The agent must fall back gracefully, returning a structured 'Unable to verify' message if the SQLite DB or Python REPL tools throw exceptions.
- The evaluator-optimizer loop must strictly halt after a maximum of 3 refinement iterations to control token consumption and latency.

### Expected Deliverables

- Shareable GitHub repository or workspace link containing all Python source code, Jupyter evaluation notebooks, and SQLite databases.
- Written technical report (PDF) explaining the agentic architecture, memory strategies chosen, and RAG configuration.
- An 8-10 slide final presentation deck (PPTX/PDF) tailored for Sprints stakeholders, summarizing the business impact and system design.
- A 3-5 minute recorded demo or walkthrough (MP4/Link) showing the multi-agent system executing a student query, grading code, and triggering a HITL escalation.

### Expected Submission Tree

```text
sprints-ai-orchestrator/
├── data/
│   ├── sprints_learner_db.csv
│   └── curriculum_docs/
├── src/
│   ├── __init__.py
│   ├── state.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── advisor_agent.py
│   │   └── assessor_agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_retriever.py
│   │   ├── code_repl.py
│   │   └── db_queries.py
│   └── main.py
├── notebooks/
│   └── evaluator_optimizer_loop.ipynb
├── docs/
│   ├── technical_report.pdf
│   └── final_presentation.pdf
├── requirements.txt
└── README.md
```

### Starter Assets from the Client

- `client_discovery_call.mp3` — the recorded discovery call: the stakeholder explains the situation, success criteria and constraints in their own words (MP3).
- `client_kickoff_deck.pptx` — the slides the Sprints stakeholder presented at kickoff (PPTX).
- `sprints_learner_db.csv`

The assignment instructs the team to start by reviewing the kickoff deck and listening to the discovery call because they contain client expectations that the written brief does not repeat. All files are attached to the assignment.

### Milestones (two weeks)

1. Days 1-2, Discovery: study the client kickoff deck and the recorded discovery call, write a one-page project charter (problem, scope, success criteria, risks, assumptions) and set up the repository or workspace.
2. Days 3-5, Foundation: build the core structure and a first thin end-to-end slice that proves the approach on the starter assets.
3. Days 6-9, Build: complete every functional requirement, test as you go, keep a short changelog of decisions.
4. Days 10-11, Quality: meet the non-functional requirements, fix defects, polish everything the Sprints stakeholder will see.
5. Days 12-13, Report and deck: write the PDF report and the 8-10 slide final presentation.
6. Day 14, Demo and submission: record the 3-5 minute walkthrough, check your work against the rubric, submit.

### Grading Rubric

| Criterion | Points | Description |
|---|---:|---|
| Functional completeness | 35 | Every functional requirement is implemented and demonstrably works on the provided starter assets. |
| Quality of the work | 20 | Non-functional requirements are met; structure, naming, documentation and setup are clean and reproducible. |
| Integration of the whole bundle | 15 | The solution draws on all sprints of the bundle, not only the last one, and the choices are justified. |
| Written report (PDF) | 10 | Problem, approach, decisions, trade-offs, results and limitations are explained clearly. |
| Final presentation deck | 10 | 8-10 slides telling the business story to the Sprints stakeholder: problem, solution, evidence, next steps. |
| Recorded demo | 10 | A 3-5 minute walkthrough showing the working deliverable end to end. |
| **Total** | **100** | **Pass mark: 60 / 100** |

### Submission Instructions

Submit the link to your repository or workspace, your written report as a PDF, and one compressed file holding the presentation deck, the demo recording (or a link to it) and every output your work produced.
