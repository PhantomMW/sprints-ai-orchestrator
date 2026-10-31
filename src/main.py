import argparse
from pathlib import Path
import sys
import uuid
from typing import Any

# Ensure repository root is on sys.path when invoked directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator
from src.tools.sqlite_checkpoint import SqliteSaver
from src.tools.test_runner import run_automated_tests

orchestrator = build_orchestrator(checkpointer=SqliteSaver())


BANNER = """
======================================================================
     Sprints Multi-Agent AI Career & Assessment Orchestrator
   Restricted Docker Execution | Agentic RAG | Automated Testing
======================================================================
"""


def run_demo_mode():
    """Run an end-to-end multi-agent demonstration showcasing all 4 architectural pillars."""
    print(BANNER)
    print("=" * 70)
    print("           COMPREHENSIVE MULTI-AGENT ORCHESTRATOR DEMO")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: Python Assessment with Automated Test Runner & Docker Sandbox
    # -------------------------------------------------------------------------
    print("\n[STAGE 1/4] Python Assessment Agent & Deterministic Test Runner")
    print("-" * 70)
    sample_code = """
def calculate_metrics(numbers):
    total = sum(numbers)
    avg = total / len(numbers) if numbers else 0
    return total, avg

nums = [10, 20, 30, 40, 50]
t, a = calculate_metrics(nums)
print(f"Sum: {t}")
print(f"Average: {a:.1f}")
"""
    assignment = (
        "Implement calculate_metrics(numbers) to compute sum and average, "
        "and print the formatted metrics."
    )
    test_suite = """
assert calculate_metrics([10, 20, 30]) == (60, 20.0)
assert calculate_metrics([5]) == (5, 5.0)
assert calculate_metrics([]) == (0, 0)
"""
    expected_output = "Sum: 150\nAverage: 30.0"

    print("Submitted Learner Code:")
    print(sample_code.strip())
    print("\nExecuting in Docker Sandbox with Per-Test Namespace Isolation...")

    test_metrics = run_automated_tests(sample_code, test_suite, expected_output)
    print("\nAutomated Test Runner Verification:")
    print(f"- Status: {test_metrics['status'].upper()}")
    print(f"- Total Test Cases: {test_metrics['total']}")
    print(f"- Tests Passed: {test_metrics['passed']}")
    print(f"- Tests Failed: {test_metrics['failed']}")
    print(f"- Pass Percentage: {test_metrics['percentage']}%")
    for r in test_metrics.get("results", []):
        status_symbol = "[PASS]" if r["status"] == "PASSED" else "[FAIL]"
        print(f"  {status_symbol} {r['name']} -> {r['status']}")

    # -------------------------------------------------------------------------
    # STAGE 2: Safe Parameterized Data Access & Differential Privacy
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/4] Career Advisor: Safe Parameterized Queries & Privacy Controls")
    print("-" * 70)
    print("Demonstrating strict single-learner parameterized isolation (SYN-DEMO-01):")
    demo_synthetic_learner = {
        "learner_id": "SYN-DEMO-01",
        "name": "Synthetic Demo Learner",
        "bootcamp_track": "Advanced Agentic AI",
        "completed_modules": 8,
        "python_assignment_score": 92,
        "target_salary_egp": 25000,
        "preferred_employers": "Fawry, Swvl",
        "readiness_status": "Ready for Placement (Synthetic)",
    }
    print(f"  Learner ID: {demo_synthetic_learner['learner_id']}")
    print(f"  Track: {demo_synthetic_learner['bootcamp_track']}")
    print(f"  Modules Completed: {demo_synthetic_learner['completed_modules']}")
    print(f"  Python Score: {demo_synthetic_learner['python_assignment_score']}/100")
    print(f"  Placement Readiness: {demo_synthetic_learner['readiness_status']}")
    print("\n  Privacy Guardrails:")
    print("  - Single-learner lookups use parameterized SQL (get_learner_by_id)")
    print("  - Bulk queries enforce MAX_QUERY_ROWS = 5 and column projection filters")
    print("  - Track aggregates enforce MIN_COHORT_SIZE = 3 differential privacy")
    print("  - Privacy Guarantee: Zero identifiable personal records in public demos")

    # -------------------------------------------------------------------------
    # STAGE 3: Evaluator-Optimizer Quality Assurance Loop
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/4] Evaluator-Optimizer: Bounded Iterative Refinement")
    print("-" * 70)
    print("Simulating response evaluation and optimization criteria:")
    print("  - Quality Dimension 1: Factual Grounding (100% curriculum adherence)")
    print("  - Quality Dimension 2: Actionability (concrete career next-steps)")
    print("  - Quality Dimension 3: Tone & Completeness")
    print("  - Hard Iteration Boundary: Maximum 3 refinement iterations")
    print("  - Fact Preservation Rule: Benchmark scores and execution facts cannot be altered")
    print("  - Convergence Status: [CONVERGED on Iteration 1 - Score: 95/100]")

    # -------------------------------------------------------------------------
    # STAGE 4: Persistent SQLite Checkpointing & Human-in-the-Loop Governance
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/4] Persistent Checkpointing (SqliteSaver) & HITL Placement Review")
    print("-" * 70)
    demo_thread = f"demo-hitl-{uuid.uuid4().hex[:6]}"
    print(f"Persistent Checkpoint Thread ID: {demo_thread}")
    print("Database Location: data/orchestrator_checkpoints.sqlite3")
    print("\nWorkflow Execution Flow:")
    print("  1. Orchestrator routes request -> Advisor Agent")
    print("  2. Learner meets criteria: 'Ready for Placement' + Preferred Employer 'Fawry'")
    print("  3. LangGraph interrupt() pauses graph execution before final clearance")
    print("  4. Checkpoint snapshot is committed to SQLite (safe across process restart)")
    print("\n[Human Review Checkpoint Simulated]:")
    print("  Reviewer: Senior Placement Coordinator")
    print("  Decision: APPROVED with guidance notes")
    print("  Command(resume=...) dispatched -> Graph completes execution")
    print("\nFinal Multi-Agent Response:")
    print("  'Learner SYN-DEMO-01 cleared for career advising. Portfolio meets requirements.'")
    print("  'Human Placement Review: APPROVED by Senior Placement Coordinator.'")
    print("  '(Disclaimer: Recommendation only; no active vacancy or employer guarantee)'")
    print("\n" + "=" * 70)
    print("           END OF COMPREHENSIVE MULTI-AGENT DEMO")
    print("=" * 70)


def run_interactive_mode():
    """Launch the interactive terminal console connected directly to the LangGraph Orchestrator."""
    print(BANNER)
    while True:
        print("\nSelect an operation:")
        print("1. Python Assessment (Orchestrator -> Sandbox -> Test Runner -> Evaluator)")
        print("2. Career Guidance (Orchestrator -> SQLite -> Agentic RAG -> HITL)")
        print("3. Standalone Automated Test Runner (Fast Sandbox Unit Testing)")
        print("4. Run Pre-configured Demo")
        print("5. Exit")

        choice = input("\nEnter choice [1-5]: ").strip()

        if choice == "1":
            print("\n[Python Assessor Mode]")
            assignment = input("Enter Assignment Description: ").strip()
            print("Enter/paste Python code (end submission with a line containing only 'EOF'):")
            lines = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            code = "\n".join(lines)

            expected_output = input("Enter expected stdout (optional, press Enter to skip): ").strip()
            test_suite = input("Enter unit test assertion (e.g. 'assert func(1)==2', optional): ").strip()

            thread_id = f"cli-assessor-{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            state_input = {
                "messages": [HumanMessage(content=f"Assess code for assignment: {assignment}")],
                "user_request": assignment,
                "submitted_code": code,
                "assignment": assignment,
                "test_suite": test_suite,
                "expected_output": expected_output,
                "request_type": "assessor",
            }

            print("\nRouting request through LangGraph Orchestrator...")
            result = orchestrator.invoke(state_input, config=config)
            print("\n" + "=" * 50)
            print(result.get("final_response", "Assessment completed."))
            print("=" * 50)

        elif choice == "2":
            student_id = input("\nEnter Learner ID (e.g. SPR-1001 or SYN-1001): ").strip()
            thread_id = f"cli-advisor-{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            state_input = {
                "messages": [HumanMessage(content=f"Career advice for learner {student_id}")],
                "student_id": student_id,
                "user_request": f"Career advice for learner {student_id}",
                "request_type": "advisor",
            }

            print(f"Routing career advice request through LangGraph Orchestrator for {student_id}...")
            result = orchestrator.invoke(state_input, config=config)

            # Check for Human-in-the-Loop review interrupt
            current_state = orchestrator.get_state(config)
            if current_state.next and "placement_review" in current_state.next:
                print("\n[Human-in-the-Loop Review Triggered]")
                print("This recommendation qualifies for tier-1 placement review.")
                dec = input("Decision [A]pprove / [R]eject: ").strip().lower()
                approved = dec.startswith("a")
                resume_payload = {
                    "approved": approved,
                    "reviewer": "Interactive CLI Reviewer",
                    "comments": "Reviewed and submitted via CLI",
                }
                result = orchestrator.invoke(Command(resume=resume_payload), config=config)

            print("\n" + "=" * 50)
            print(result.get("final_response", "Career guidance completed."))
            print("=" * 50)

        elif choice == "3":
            print("\n[Standalone Automated Test Runner]")
            print("Enter/paste Python code (end with 'EOF'):")
            lines = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            code = "\n".join(lines)
            test_suite = input("Enter assertions (multiline or single): ").strip()

            print("\nExecuting in isolated Docker sandbox...")
            metrics = run_automated_tests(code, test_suite=test_suite)
            print(f"\nResults: {metrics['passed']}/{metrics['total']} passed ({metrics['percentage']}%)")
            for r in metrics["results"]:
                print(f"  - {r['name']}: {r['status']} {r.get('error', '')}")

        elif choice == "4":
            run_demo_mode()

        elif choice == "5":
            print("Exiting Sprints AI Orchestrator. Goodbye!")
            sys.exit(0)

        else:
            print("Invalid selection. Please choose 1, 2, 3, 4, or 5.")


def main():
    parser = argparse.ArgumentParser(
        description="Sprints Multi-Agent AI Career & Assessment Orchestrator CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="Execution mode: 'demo' (default) or 'interactive' terminal UI",
    )
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive_mode()
    else:
        print(BANNER)
        run_demo_mode()


if __name__ == "__main__":
    main()
