"""
Comprehensive Automated Test Suite for FlowLogic AI System
Tests all modules: Vision, OCR, Polyglot Code Gen, Reverse Engine, Sandbox, Exporter.
Run: python tests/test_all_modules.py
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.vision_processor import VisionProcessor, ShapeDetectionResult, DetectedNode, DetectedEdge
from modules.ocr_engine import OCREngine, FlowchartIR, FlowNode, FlowEdge, FlowNodeType
from modules.code_generator import CodeGenerator, SupportedLanguage, ComplexityEngine, TestSuiteGenerator
from modules.reverse_engine import ReverseEngine
from modules.sandbox import ExecutionSandbox
from modules.exporter import FlowchartExporter


class TestFlowLogicSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_decision = os.path.join("samples", "sample_decision.png")
        cls.sample_linear = os.path.join("samples", "sample_linear.png")

    # ------------------------------------------------------------------------
    # 1. Vision Processor Tests
    # ------------------------------------------------------------------------
    def test_01_vision_processor_detection(self):
        vp = VisionProcessor()
        res, img, annotated = vp.process_image(self.sample_decision)

        self.assertIsInstance(res, ShapeDetectionResult)
        self.assertGreaterEqual(len(res.nodes), 3, "Should detect at least 3 nodes")
        self.assertGreaterEqual(len(res.edges), 2, "Should detect edges")
        self.assertEqual(annotated.shape, img.shape)

    def test_02_vision_shape_classification(self):
        vp = VisionProcessor()
        res, _, _ = vp.process_image(self.sample_linear)
        shapes = [n.shape for n in res.nodes]
        self.assertTrue("oval" in shapes or "rectangle" in shapes or "parallelogram" in shapes)

    # ------------------------------------------------------------------------
    # 2. OCR & IR Engine Tests
    # ------------------------------------------------------------------------
    def test_03_ocr_ir_construction(self):
        vp = VisionProcessor()
        res, img, _ = vp.process_image(self.sample_decision)

        ocr = OCREngine()
        ir = ocr.build_ir_from_vision(img, res)

        self.assertIsInstance(ir, FlowchartIR)
        self.assertIsNotNone(ir.start_node_id)
        self.assertGreater(len(ir.nodes), 0)
        self.assertTrue(ir.is_valid)

    def test_04_ir_json_serialization(self):
        ir = FlowchartIR(
            title="Test Program",
            nodes=[
                FlowNode(0, FlowNodeType.START, "Start", "START", "oval", 0, 0, 100, 50),
                FlowNode(1, FlowNodeType.INPUT, "Input num", "INPUT num", "parallelogram", 0, 80, 100, 50),
                FlowNode(2, FlowNodeType.END, "End", "END", "oval", 0, 160, 100, 50),
            ],
            edges=[FlowEdge(0, 1), FlowEdge(1, 2)],
            start_node_id=0,
            end_node_ids=[2],
            variables=["num"],
        )
        json_str = ir.to_json()
        self.assertIn('"title": "Test Program"', json_str)
        self.assertIn('"START"', json_str)

    # ------------------------------------------------------------------------
    # 3. Polyglot Code Generator Tests
    # ------------------------------------------------------------------------
    def test_05_polyglot_code_generation(self):
        ir = FlowchartIR(
            title="Double Number",
            nodes=[
                FlowNode(0, FlowNodeType.START, "Start", "START", "oval", 0, 0, 100, 50),
                FlowNode(1, FlowNodeType.INPUT, "Input x", "INPUT x", "parallelogram", 0, 80, 100, 50),
                FlowNode(2, FlowNodeType.PROCESS, "result = x * 2", "result = x * 2", "rectangle", 0, 160, 100, 50),
                FlowNode(3, FlowNodeType.OUTPUT, "Print result", "Print result", "parallelogram", 0, 240, 100, 50),
                FlowNode(4, FlowNodeType.END, "End", "END", "oval", 0, 320, 100, 50),
            ],
            edges=[FlowEdge(0, 1), FlowEdge(1, 2), FlowEdge(2, 3), FlowEdge(3, 4)],
            start_node_id=0,
            end_node_ids=[4],
            variables=["x", "result"],
        )

        gen = CodeGenerator()
        res = gen.generate(ir, active_language="Python")

        self.assertIn("Python", res.codes)
        self.assertIn("C++", res.codes)
        self.assertIn("Java", res.codes)
        self.assertIn("JavaScript", res.codes)
        self.assertIn("C", res.codes)
        self.assertIn("Go", res.codes)
        self.assertIn("Rust", res.codes)

        # Verify Python syntax compiles
        py_code = res.codes["Python"]
        compile(py_code, "<string>", "exec")

    def test_06_complexity_analysis(self):
        ir = FlowchartIR(
            nodes=[
                FlowNode(0, FlowNodeType.START, "Start", "START", "oval", 0, 0, 100, 50),
                FlowNode(1, FlowNodeType.DECISION, "x > 0", "x > 0", "diamond", 0, 80, 100, 50),
                FlowNode(2, FlowNodeType.END, "End", "END", "oval", 0, 160, 100, 50),
            ],
            edges=[FlowEdge(0, 1), FlowEdge(1, 2)],
            variables=["x"],
        )
        comp = ComplexityEngine.analyze(ir)
        self.assertEqual(comp.time_complexity, "O(1)")
        self.assertEqual(comp.space_complexity, "O(1)")

    def test_07_unit_test_suite_generator(self):
        ir = FlowchartIR(variables=["x", "y"])
        tests = TestSuiteGenerator.generate(ir)
        self.assertGreaterEqual(len(tests), 5)
        for t in tests:
            self.assertIn("name", t)
            self.assertIn("input", t)

    # ------------------------------------------------------------------------
    # 4. Reverse Engine Tests (Code -> Flowchart IR -> Mermaid)
    # ------------------------------------------------------------------------
    def test_08_reverse_ast_engine(self):
        sample_code = """
x = float(input("Enter x: "))
if x >= 0:
    print("Positive")
else:
    print("Negative")
"""
        ir = ReverseEngine.code_to_ir(sample_code)
        self.assertIsInstance(ir, FlowchartIR)
        self.assertGreaterEqual(len(ir.nodes), 4)

        mermaid_code = ReverseEngine.ir_to_mermaid(ir)
        self.assertIn("flowchart TD", mermaid_code)
        self.assertIn("Positive", mermaid_code)

    # ------------------------------------------------------------------------
    # 5. Live Execution Sandbox Tests
    # ------------------------------------------------------------------------
    def test_09_sandbox_execution_success(self):
        code = """
x = float(input())
print(f"Calculated: {x * 2}")
"""
        result = ExecutionSandbox.run_python_code(code, user_input="21\n")
        self.assertTrue(result.success)
        self.assertIn("Calculated: 42.0", result.stdout)
        self.assertGreater(result.execution_time_ms, 0)

    def test_10_sandbox_execution_error_capture(self):
        code = "print(10 / 0)"
        result = ExecutionSandbox.run_python_code(code)
        self.assertFalse(result.success)
        self.assertIn("ZeroDivisionError", result.error_message)

    # ------------------------------------------------------------------------
    # 6. Exporter Engine Tests
    # ------------------------------------------------------------------------
    def test_11_exporter_outputs(self):
        ir = FlowchartIR(
            title="Unit Test Algorithm",
            nodes=[FlowNode(0, FlowNodeType.START, "Start", "START", "oval", 0, 0, 100, 50)],
            edges=[],
        )
        gen = CodeGenerator()
        code_res = gen.generate(ir)

        mermaid = FlowchartExporter.to_mermaid(ir)
        plantuml = FlowchartExporter.to_plantuml(ir)
        report = FlowchartExporter.generate_full_report(ir, code_res)

        self.assertIn("flowchart TD", mermaid)
        self.assertIn("@startuml", plantuml)
        self.assertIn("FlowLogic AI", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
