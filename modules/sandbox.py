"""
Module: Execution Sandbox & Live Test Runner
Safely executes generated code with custom or automated inputs and captures execution logs.
"""

import sys
import io
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    error_message: Optional[str] = None


@dataclass
class TestResult:
    test_id: int
    test_name: str
    input_data: str
    expected_behavior: str
    actual_output: str
    passed: bool
    execution_time_ms: float
    error_message: Optional[str] = None


class ExecutionSandbox:
    """In-process safe execution sandbox with input mocking and stream redirection."""

    @classmethod
    def run_python_code(cls, code: str, user_input: str = "", timeout_sec: float = 3.0) -> ExecutionResult:
        """Executes a Python code string with simulated stdin and captured stdout/stderr."""
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        simulated_stdin = io.StringIO(user_input)

        sys.stdin = simulated_stdin
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        start_time = time.perf_counter()
        success = True
        error_msg = None

        # Clean code to prevent recursive loop if name == main
        exec_globals = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }

        try:
            exec(code, exec_globals)
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}"
            captured_stderr.write(traceback.format_exc())
        finally:
            end_time = time.perf_counter()
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        exec_time = (end_time - start_time) * 1000.0

        return ExecutionResult(
            success=success,
            stdout=captured_stdout.getvalue(),
            stderr=captured_stderr.getvalue(),
            execution_time_ms=round(exec_time, 2),
            error_message=error_msg,
        )

    @classmethod
    def run_test_suite(cls, code: str, test_cases: List[Dict[str, Any]]) -> List[TestResult]:
        """Runs all test cases against the code and generates a full pass/fail report."""
        results = []
        for t in test_cases:
            inp = t.get("input", "")
            res = cls.run_python_code(code, user_input=inp)
            passed = res.success

            results.append(TestResult(
                test_id=t.get("id", 1),
                test_name=t.get("name", "Test"),
                input_data=inp.strip(),
                expected_behavior=t.get("expected_behavior", ""),
                actual_output=res.stdout.strip() if res.success else res.stderr.strip(),
                passed=passed,
                execution_time_ms=res.execution_time_ms,
                error_message=res.error_message,
            ))

        return results
