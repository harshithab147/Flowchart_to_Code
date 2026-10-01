"""
Module: Reverse Engine (Code to Flowchart Visual IDE)
Converts Python / C-like code into Flowchart IR and Mermaid.js / Graphviz diagrams.
"""

import ast
import re
from typing import List, Dict, Optional, Tuple, Any

from .ocr_engine import FlowchartIR, FlowNode, FlowEdge, FlowNodeType


class ReverseEngine:
    """
    Parses source code (e.g. Python AST) and reconstructs a Flowchart IR and Mermaid.js diagram.
    """

    @classmethod
    def code_to_ir(cls, code: str) -> FlowchartIR:
        """Parses Python source code using Abstract Syntax Tree (AST) to build FlowchartIR."""
        nodes: List[FlowNode] = []
        edges: List[FlowEdge] = []
        variables = set()

        node_id = 0

        # 1. Start Node
        nodes.append(FlowNode(
            id=node_id,
            node_type=FlowNodeType.START,
            raw_text="Start",
            cleaned_expression="START",
            shape="oval",
            x=100, y=50, w=120, h=50,
        ))
        prev_id = node_id
        node_id += 1

        try:
            tree = ast.parse(code)
            for stmt in tree.body:
                # Handle functions by unwrapping body
                statements = stmt.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) else [stmt]
                for s in statements:
                    new_id, created_nodes, created_edges = cls._ast_stmt_to_nodes(s, node_id, prev_id, variables)
                    nodes.extend(created_nodes)
                    edges.extend(created_edges)
                    if created_nodes:
                        prev_id = new_id
                        node_id = max(n.id for n in created_nodes) + 1
        except Exception:
            # Fallback heuristic line-by-line parser if AST fails
            prev_id, nodes, edges, variables = cls._heuristic_code_parser(code, node_id, prev_id, nodes, edges)
            node_id = len(nodes)

        # End Node
        end_node = FlowNode(
            id=node_id,
            node_type=FlowNodeType.END,
            raw_text="End",
            cleaned_expression="END",
            shape="oval",
            x=100, y=50 + len(nodes) * 80, w=120, h=50,
        )
        nodes.append(end_node)
        edges.append(FlowEdge(source_id=prev_id, target_id=node_id))

        return FlowchartIR(
            title="Code to Flowchart Diagram",
            nodes=nodes,
            edges=edges,
            start_node_id=0,
            end_node_ids=[node_id],
            variables=sorted(list(variables)),
            is_valid=True,
            validation_messages=["Reconstructed from Code AST."],
        )

    @classmethod
    def _ast_stmt_to_nodes(cls, stmt: ast.AST, curr_id: int, prev_id: int,
                           variables: set) -> Tuple[int, List[FlowNode], List[FlowEdge]]:
        """Maps single AST node to FlowNode(s)."""
        nodes = []
        edges = []

        if isinstance(stmt, ast.Assign):
            target = ast.unparse(stmt.targets[0])
            value = ast.unparse(stmt.value)
            variables.add(target)
            is_input = "input(" in value
            node_type = FlowNodeType.INPUT if is_input else FlowNodeType.PROCESS
            shape = "parallelogram" if is_input else "rectangle"
            text = f"Input {target}" if is_input else f"{target} = {value}"

            n = FlowNode(curr_id, node_type, text, text, shape, 100, 50 + curr_id * 80, 140, 60)
            nodes.append(n)
            edges.append(FlowEdge(prev_id, curr_id))
            return curr_id, nodes, edges

        elif isinstance(stmt, ast.If):
            test_expr = ast.unparse(stmt.test)
            decision_node = FlowNode(curr_id, FlowNodeType.DECISION, f"Is {test_expr}?", test_expr, "diamond",
                                     100, 50 + curr_id * 80, 120, 80)
            nodes.append(decision_node)
            edges.append(FlowEdge(prev_id, curr_id))

            # Yes branch
            yes_id = curr_id + 1
            yes_text = ast.unparse(stmt.body[0]) if stmt.body else "pass"
            yes_node = FlowNode(yes_id, FlowNodeType.PROCESS, yes_text, yes_text, "rectangle",
                                260, 50 + curr_id * 80 + 70, 120, 50)
            nodes.append(yes_node)
            edges.append(FlowEdge(curr_id, yes_id, condition="Yes"))

            # No branch
            no_id = curr_id + 2
            no_text = ast.unparse(stmt.orelse[0]) if stmt.orelse else "pass"
            no_node = FlowNode(no_id, FlowNodeType.PROCESS, no_text, no_text, "rectangle",
                               -40, 50 + curr_id * 80 + 70, 120, 50)
            nodes.append(no_node)
            edges.append(FlowEdge(curr_id, no_id, condition="No"))

            # Joiner node
            join_id = curr_id + 3
            join_node = FlowNode(join_id, FlowNodeType.CONNECTOR, "Merge", "Merge", "circle",
                                 100, 50 + curr_id * 80 + 150, 40, 40)
            nodes.append(join_node)
            edges.append(FlowEdge(yes_id, join_id))
            edges.append(FlowEdge(no_id, join_id))

            return join_id, nodes, edges

        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_text = ast.unparse(stmt)
            is_print = "print" in call_text
            node_type = FlowNodeType.OUTPUT if is_print else FlowNodeType.PROCESS
            shape = "parallelogram" if is_print else "rectangle"
            n = FlowNode(curr_id, node_type, call_text, call_text, shape, 100, 50 + curr_id * 80, 140, 60)
            nodes.append(n)
            edges.append(FlowEdge(prev_id, curr_id))
            return curr_id, nodes, edges

        # Default fallback statement
        raw_code = ast.unparse(stmt)
        n = FlowNode(curr_id, FlowNodeType.PROCESS, raw_code, raw_code, "rectangle", 100, 50 + curr_id * 80, 140, 60)
        nodes.append(n)
        edges.append(FlowEdge(prev_id, curr_id))
        return curr_id, nodes, edges

    @classmethod
    def _heuristic_code_parser(cls, code: str, curr_id: int, prev_id: int,
                               nodes: list, edges: list) -> Tuple[int, list, list, set]:
        variables = set()
        for line in code.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "if" in line and ":" in line or "if (" in line:
                cond = line.replace("if", "").replace(":", "").replace("{", "").strip(" ()")
                n = FlowNode(curr_id, FlowNodeType.DECISION, f"Is {cond}?", cond, "diamond", 100, 50 + curr_id * 80, 120, 80)
            elif "input(" in line or "cin >>" in line or "scanf" in line:
                n = FlowNode(curr_id, FlowNodeType.INPUT, line, line, "parallelogram", 100, 50 + curr_id * 80, 140, 60)
            elif "print" in line or "cout <<" in line or "printf" in line or "System.out" in line:
                n = FlowNode(curr_id, FlowNodeType.OUTPUT, line, line, "parallelogram", 100, 50 + curr_id * 80, 140, 60)
            else:
                n = FlowNode(curr_id, FlowNodeType.PROCESS, line, line, "rectangle", 100, 50 + curr_id * 80, 140, 60)

            nodes.append(n)
            edges.append(FlowEdge(prev_id, curr_id))
            prev_id = curr_id
            curr_id += 1

        return prev_id, nodes, edges, variables

    @classmethod
    def ir_to_mermaid(cls, ir: FlowchartIR) -> str:
        """Converts FlowchartIR to a Mermaid.js flowchart definition."""
        lines = [
            "%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1e293b', 'edgeLabelBackground':'#0f172a', 'tertiaryColor': '#0f172a'}}}%%",
            "flowchart TD",
        ]

        # Shape formatting in Mermaid
        # Oval: ([Text])
        # Diamond: {Text}
        # Parallelogram: [/Text/]
        # Rectangle: [Text]
        # Circle: ((Text))

        for n in ir.nodes:
            text = n.raw_text.replace('"', "'").replace("\n", " ")
            if n.node_type in [FlowNodeType.START, FlowNodeType.END] or n.shape == "oval":
                lines.append(f'    node_{n.id}(["{text}"])')
            elif n.node_type == FlowNodeType.DECISION or n.shape == "diamond":
                lines.append(f'    node_{n.id}{{"{text}"}}')
            elif n.node_type in [FlowNodeType.INPUT, FlowNodeType.OUTPUT] or n.shape == "parallelogram":
                lines.append(f'    node_{n.id}[/"{text}"/]')
            elif n.shape == "circle" or n.node_type == FlowNodeType.CONNECTOR:
                lines.append(f'    node_{n.id}(("{text}"))')
            else:
                lines.append(f'    node_{n.id}["{text}"]')

        # Edges
        for e in ir.edges:
            if e.condition:
                lines.append(f'    node_{e.source_id} -->|{e.condition}| node_{e.target_id}')
            else:
                lines.append(f'    node_{e.source_id} --> node_{e.target_id}')

        # Styling
        lines.append("    classDef startEnd fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff;")
        lines.append("    classDef decision fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff;")
        lines.append("    classDef io fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff;")
        lines.append("    classDef process fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff;")

        for n in ir.nodes:
            if n.node_type in [FlowNodeType.START, FlowNodeType.END]:
                lines.append(f"    class node_{n.id} startEnd;")
            elif n.node_type == FlowNodeType.DECISION:
                lines.append(f"    class node_{n.id} decision;")
            elif n.node_type in [FlowNodeType.INPUT, FlowNodeType.OUTPUT]:
                lines.append(f"    class node_{n.id} io;")
            else:
                lines.append(f"    class node_{n.id} process;")

        return "\n".join(lines)
