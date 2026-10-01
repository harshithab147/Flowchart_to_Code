from test_tanushri_logic import flowchart_ir
from modules.code_generator import (
    DeterministicCodeBuilder,
    ComplexityEngine,
    TestCaseEngine
)


print("\n========== HARSHITHA CODE GENERATOR TEST ==========\n")

# 1. Generate code in all supported languages
codes = DeterministicCodeBuilder.generate_all(flowchart_ir)

print("========== GENERATED CODE ==========\n")

for language, code in codes.items():
    print(f"\n--- {language} ---")
    print(code)


# 2. Analyze complexity
complexity = ComplexityEngine.analyze(flowchart_ir)

print("\n========== COMPLEXITY ANALYSIS ==========")
print("Time Complexity:", complexity.time_complexity)
print("Space Complexity:", complexity.space_complexity)
print("Reasoning:", complexity.reasoning)


# 3. Generate unit test cases
test_cases = TestCaseEngine.generate(flowchart_ir)

print("\n========== UNIT TEST CASES ==========")

for test in test_cases:
    print(test)

print("\n==============================================")
print("Harshitha's Code Generation Module Test Completed")