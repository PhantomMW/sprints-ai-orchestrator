from src.agents.assessor_agent import assess_python_code


def run_assessment(test_name, code, expected_output):
    assignment = """
    Write a Python program that:
    1. Creates a list of numbers.
    2. Calculates the sum.
    3. Calculates the average.
    4. Prints both results.
    """

    print(f"\n{'=' * 50}")
    print(test_name)
    print("=" * 50)

    result = assess_python_code(
        code=code,
        assignment=assignment,
        expected_output=expected_output
    )

    if result["status"] == "success":
        print("Execution:", result["execution"])
        print("Grading:", result["grading"])
    else:
        print("Assessment Error:", result)


def main():

    correct_code = """
numbers = [10, 20, 30, 40, 50]

total = sum(numbers)
average = total / len(numbers)

print("Sum:", total)
print("Average:", average)
"""

    incorrect_code = """
numbers = [10, 20, 30, 40, 50]

total = sum(numbers)
average = total / 2

print("Sum:", total)
print("Average:", average)
"""

    expected_output = "Sum: 150\nAverage: 30.0"

    run_assessment(
        "TEST A - CORRECT SOLUTION",
        correct_code,
        expected_output
    )

    run_assessment(
        "TEST B - INCORRECT SOLUTION",
        incorrect_code,
        expected_output
    )


if __name__ == "__main__":
    main()