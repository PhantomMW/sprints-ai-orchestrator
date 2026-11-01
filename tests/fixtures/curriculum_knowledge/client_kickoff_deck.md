# Source: client_kickoff_deck.pptx

Source Type: Client Project Context

Original Format: .pptx

---

## Slide 1

Sprints: Multi-Agent AI Career & Assessment Orchestrator

Client kickoff  |  Advanced Agentic AI  |  presented by Youssef El-Sherif, Sprints, VP of Engineering, Sprints

• You recently joined Sprints (sprints.ai), the leading EdTech company in Egypt that bridges the gap between learning and employment through AI-enabled platforms and employer-connected pathways. You have been hired as the Lead AI Systems Architect for the Core Platforms team based in Cairo. The company has successfully launched its "Advanced Agentic AI" bootcamp and is rapidly scaling across the MENA region. However, the current manual systems are failing under the weight of thousands of active learners. The Sprints student success team is overwhelmed by repetitive curriculum queries, manual grading of Python assignments, and the intricate task of matching graduates to local hiring partners li

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 2

The business situation

• Welcome to Sprints, the premier EdTech platform bridging the talent gap in the MENA region. We are currently scaling our employer-connected tech bootcamps, processing thousands of learners aiming for placements at top Egyptian tech firms. As our cohorts grow, our human advisors and technical graders are becoming the primary bottleneck.
• We need you to build the "Sprints Multi-Agent AI Career & Assessment Orchestrator." This is not a simple chatbot; we require a robust, stateful agentic ecosystem. The system must autonomously ingest our curriculum to answer student questions, securely grade Python submissions using live execution tools, and query our internal databases to understand a student's progress and target salary in EGP. This project requires a sophisticated orchestrator-worker architecture.

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 3

What we need from you (1/2)

• {"requirement": "[Sprint 1] Implement a complete Document Ingestion and Vectorization pipeline that loads Sprints curric
• {"requirement": "[Sprint 1] Create strongly typed Prompt Templates and Output Parsers using Pydantic schemas to strictly
• {"requirement": "[Sprint 1] Equip a simple ReAct agent with a custom SQL tool that allows it to query the provided 'spri
• {"requirement": "[Sprint 1] Implement a custom tool using the Python REPL utility that allows the Assessor Agent to secu
• {"requirement": "[Sprint 2] Architect a unified StateGraph orchestration pipeline defining a global `TypedDict` state th
• {"requirement": "[Sprint 2] Integrate a MemorySaver checkpointer within the LangGraph orchestrator to isolate and persis

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 4

What we need from you (2/2)

• {"requirement": "[Sprint 2] Build an Agentic RAG workflow inside the Advisor Agent node that features a self-reflection
• {"requirement": "[Sprint 2] Implement an Evaluator-Optimizer pattern for the Assessor Agent, utilizing a secondary 'LLM
• {"requirement": "[Sprint 2] Develop a Human-in-the-Loop (HITL) feedback escalation workflow using LangGraph's interrupt
• {"requirement": "[Sprint 2] Standardize communication between the Orchestrator and Worker agents by enforcing structured

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 5

Quality bar

• The system must use LangGraph for all multi-agent orchestration, avoiding deprecated linear chains for dynamic routing.
• All generated Python code and agentic tools must include detailed docstrings mapping to the specific Sprints curriculum functionality.
• The vector database implementation (ChromaDB) must consistently retrieve context within 1000 milliseconds to avoid orchestrator timeouts.
• The agent must fall back gracefully, returning a structured 'Unable to verify' message if the SQLite DB or Python REPL tools throw exceptions.
• The evaluator-optimizer loop must strictly halt after a maximum of 3 refinement iterations to control token consumption and latency.

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 6

Deliverables

• Shareable GitHub repository or workspace link containing all Python source code, Jupyter evaluation notebooks, and SQLite databases.
• Written technical report (PDF) explaining the agentic architecture, memory strategies chosen, and RAG configuration.
• An 8-10 slide final presentation deck (PPTX/PDF) tailored for Sprints stakeholders, summarizing the business impact and system design.
• A 3-5 minute recorded demo or walkthrough (MP4/Link) showing the multi-agent system executing a student query, grading code, and triggering a HITL escalation.

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 7

Milestones (two weeks)

• Days 1-2, Discovery: study the client kickoff deck and the recorded discovery call, write a one-page project charter (problem, scope, success criteria
• Days 3-5, Foundation: build the core structure and a first thin end-to-end slice that proves the approach on the starter assets.
• Days 6-9, Build: complete every functional requirement, test as you go, keep a short changelog of decisions.
• Days 10-11, Quality: meet the non-functional requirements, fix defects, polish everything the Sprints stakeholder will see.
• Days 12-13, Report and deck: write the PDF report and the 8-10 slide final presentation.
• Day 14, Demo and submission: record the 3-5 minute walkthrough, check your work against the rubric, submit.

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 8

How you will be graded (pass mark 60/100)

• Functional completeness, 35 pts: Every functional requirement is implemented and demonstrably works on the provided starter assets.
• Quality of the work, 20 pts: Non-functional requirements are met; structure, naming, documentation and setup are clean and reproducible.
• Integration of the whole bundle, 15 pts: The solution draws on all sprints of the bundle, not only the last one, and the choices are justified.
• Written report (PDF), 10 pts: Problem, approach, decisions, trade-offs, results and limitations are explained clearly.
• Final presentation deck, 10 pts: 8-10 slides telling the business story to the Sprints stakeholder: problem, solution, evidence, next steps.
• Recorded demo, 10 pts: A 3-5 minute walkthrough showing the working deliverable end to end.

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning

## Slide 9

Starter assets

• This kickoff deck (PPTX)
• Recorded discovery call with the client (MP3)
• sprints_learner_db.csv: A CSV containing 500 rows of mock Egyptian learner data. Columns must include: learner_id, name, bootcamp_trac
• langgraph_orchestrator_starter.repo_link: A Python starter repository for a LangGraph multi-agent project. It should contain a basic `requirements.txt`

Sprints  |  Advanced Agentic AI  |  Graduation project kickoff  |  Illustrative scenario for learning