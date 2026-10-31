import argparse
import sys
from typing import Any

from src.agents.advisor_agent import generate_career_advice
from src.agents.assessor_agent import assess_python_code
from src.tools.test_runner import run_automated_tests


BANNER = """
======================================================================
     Sprints Multi-Agent AI Career & Assessment Orchestrator
   Restricted Docker Execution | Agentic RAG | Automated Testing
======================================================================
"""


def run_demo_mode():
    """Run the pre-configured assessment and career guidance demonstration."""
    print("\n--- [DEMO 1] Python Assessment with Automated Test Runner ---")
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

    print("Submitting Code:")
    print(sample_code.strip())
    print("\nExecuting in Docker sandbox and running automated test assertions...")

    test_metrics = run_automated_tests(sample_code, test_suite, expected_output)
    print(f"\n[Automated Test Runner Results]:")
    print(f"- Total Test Cases: {test_metrics['total']}")
    print(f"- Tests Passed: {test_metrics['passed']}")
    print(f"- Tests Failed: {test_metrics['failed']}")
    print(f"- Pass Percentage: {test_metrics['percentage']}%")
    for r in test_metrics["results"]:
        status_symbol = "[PASS]" if r["status"] == "PASSED" else "[FAIL]"
        print(f"  {status_symbol} {r['name']} -> {r['status']}")

    print("\n--- [DEMO 2] Safe Parameterized Learner Profile Lookup ---")
    print("Testing Parameterized Learner Profile Verification (SPR-1001)...")
    from src.tools.db_queries import get_learner_by_id
    profile_result = get_learner_by_id("SPR-1001")
    if profile_result.get("status") == "success" and profile_result.get("row_count", 0) > 0:
        learner = profile_result["data"][0]
        print(f"Learner ID: {learner.get('learner_id')}")
        print(f"Name: {learner.get('name')}")
        print(f"Track: {learner.get('bootcamp_track')}")
        print(f"Python Score: {learner.get('python_assignment_score')}/100")
        print(f"Status: {learner.get('readiness_status')}")
    else:
        print("Note: Learner record lookup status:", profile_result.get("message"))


def run_interactive_mode():
    """Launch the interactive terminal console for learner and assessment queries."""
    print(BANNER)
    while True:
        print("\nSelect an operation:")
        print("1. Assess Python Code (with Automated Test Runner)")
        print("2. Career Guidance Lookup (by Learner ID)")
        print("3. Run Pre-configured Demo")
        print("4. Exit")

        choice = input("\nEnter choice [1-4]: ").strip()

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

            print("\nRunning assessment in isolated sandbox...")
            metrics = run_automated_tests(code, test_suite=test_suite, expected_output=expected_output)
            print(f"\nTest Runner Outcome: {metrics['passed']}/{metrics['total']} passed ({metrics['percentage']}%)")
            for r in metrics["results"]:
                print(f"  - {r['name']}: {r['status']} {r.get('error', '')}")

        elif choice == "2":
            student_id = input("\nEnter Learner ID (e.g. SPR-1001): ").strip()
            print(f"Querying profile for {student_id}...")
            res = generate_career_advice(student_id)
            if res.get("status") == "success":
                adv = res["advice"]
                print(f"\n=== Verified Career Advice for {adv.get('learner_name')} ===")
                print(f"Track: {adv.get('current_track')}")
                print(f"Roles: {', '.join(adv.get('recommended_roles', []))}")
                print(f"Overall: {adv.get('overall_recommendation')}")
            else:
                print(f"Result: {res.get('message')}")

        elif choice == "3":
            run_demo_mode()

        elif choice == "4":
            print("Exiting Sprints AI Orchestrator. Goodbye!")
            sys.exit(0)

        else:
            print("Invalid selection. Please choose 1, 2, 3, or 4.")


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
