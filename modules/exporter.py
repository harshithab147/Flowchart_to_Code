"""
Module: Exporter Engine
Generates export formats: Mermaid.js, PlantUML, JSON IR, and full Markdown Project Reports.
"""

import json
from typing import Dict, Any
from .ocr_engine import FlowchartIR, FlowNodeType
from .code_generator import GeneratedCodeResult
from .reverse_engine import ReverseEngine


class FlowchartExporter:
    """Exports Flowchart IR and generated code to various documentation formats."""

    @classmethod
    def to_mermaid(cls, ir: FlowchartIR) -> str:
        """Exports to Mermaid.js flowchart code."""
        return ReverseEngine.ir_to_mermaid(ir)

    @classmethod
    def to_plantuml(cls, ir: FlowchartIR) -> str:
        """Exports to PlantUML Activity Diagram code."""
        lines = [
            "@startuml",
            "skinparam monochrome false",
            "skinparam ActivityBackgroundColor #2b303c",
            "skinparam ActivityBorderColor #00adb5",
            "skinparam ActivityFontColor #eeeeee",
            "start",
        ]

        for n in ir.nodes:
            if n.node_type == FlowNodeType.START:
                continue
            elif n.node_type == FlowNodeType.END:
                lines.append("stop")
            elif n.node_type == FlowNodeType.DECISION:
                lines.append(f"if ({n.cleaned_expression}) then (yes)")
                lines.append(f"  :Process branch;")
                lines.append(f"else (no)")
                lines.append(f"  :Alternative branch;")
                lines.append(f"endif")
            elif n.node_type in [FlowNodeType.INPUT, FlowNodeType.OUTPUT]:
                lines.append(f":>{n.raw_text}<;")
            else:
                lines.append(f":{n.raw_text};")

        if "stop" not in lines[-1]:
            lines.append("stop")
        lines.append("@enduml")
        return "\n".join(lines)

    @classmethod
    def to_json_ir(cls, ir: FlowchartIR) -> str:
        """Exports the raw JSON schema."""
        return ir.to_json(indent=2)

    @classmethod
    def generate_full_report(cls, ir: FlowchartIR, code_result: GeneratedCodeResult) -> str:
        """Generates a complete GitHub-ready Markdown Project Report."""
        lines = [
            f"# 📊 FlowLogic AI — Flowchart Algorithm Report",
            f"**Algorithm Title:** {ir.title}  ",
            f"**Validation Status:** {'✅ Valid Flow' if ir.is_valid else '⚠️ Warnings Detected'}  ",
            "",
            "## 1. Algorithmic Complexity (Big-O)",
            f"- **Time Complexity:** `{code_result.complexity.time_complexity if code_result.complexity else 'O(1)'}`",
            f"- **Space Complexity:** `{code_result.complexity.space_complexity if code_result.complexity else 'O(1)'}`",
            f"- **Complexity Proof:** {code_result.complexity.reasoning if code_result.complexity else 'N/A'}",
            "",
            "## 2. Detected Flowchart Nodes & Logic",
            "| ID | Shape | Semantic Type | Raw Text | Expression |",
            "|:--:|:-----:|:-------------:|:---------|:-----------|",
        ]

        for n in ir.nodes:
            lines.append(f"| #{n.id} | {n.shape} | {n.node_type.value} | {n.raw_text} | `{n.cleaned_expression}` |")

        lines.extend([
            "",
            "## 3. Generated Polyglot Source Code",
            f"### Primary Target: {code_result.active_language}",
            f"```{code_result.active_language.lower()}",
            code_result.codes.get(code_result.active_language, ""),
            "```",
            "",
            "## 4. Mermaid.js Flowchart Diagram",
            "```mermaid",
            cls.to_mermaid(ir),
            "```",
            "",
            "---",
            "*Generated automatically by FlowLogic AI Studio.*",
        ])

        return "\n".join(lines)
