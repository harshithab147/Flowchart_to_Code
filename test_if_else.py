from modules.ocr_engine import FlowchartIR, FlowNode, FlowEdge, FlowNodeType
from modules.code_generator import CodeGenerator

nodes = [
    FlowNode(
        id=0,
        node_type=FlowNodeType.END,
        raw_text="End",
        cleaned_expression="End",
        shape="", x=0, y=0, w=0, h=0
    ),
    FlowNode(
        id=1,
        node_type=FlowNodeType.OUTPUT,
        raw_text="Print Positive",
        cleaned_expression="Print Positive",
        shape="", x=0, y=0, w=0, h=0
    ),
    FlowNode(
        id=2,
        node_type=FlowNodeType.OUTPUT,
        raw_text="Print Negative",
        cleaned_expression="Print Negative",
        shape="", x=0, y=0, w=0, h=0
    ),
    FlowNode(
        id=3,
        node_type=FlowNodeType.DECISION,
        raw_text="number > 0",
        cleaned_expression="number > 0",
        shape="", x=0, y=0, w=0, h=0,
        branch_conditions={"Yes": 1, "No": 2}
    ),
    FlowNode(
        id=4,
        node_type=FlowNodeType.INPUT,
        raw_text="Input number",
        cleaned_expression="Input number",
        shape="", x=0, y=0, w=0, h=0
    ),
    FlowNode(
        id=5,
        node_type=FlowNodeType.START,
        raw_text="Start",
        cleaned_expression="Start",
        shape="", x=0, y=0, w=0, h=0
    ),
]

edges = [
    FlowEdge(source_id=5, target_id=4),
    FlowEdge(source_id=4, target_id=3),
    FlowEdge(source_id=3, target_id=1, condition="Yes"),
    FlowEdge(source_id=3, target_id=2, condition="No"),
    FlowEdge(source_id=1, target_id=0),
    FlowEdge(source_id=2, target_id=0),
]

flowchart_ir = FlowchartIR(
    nodes=nodes,
    edges=edges
)

result = CodeGenerator().generate(flowchart_ir)

for language, code in result.codes.items():
    print("\n==========", language, "==========")
    print(code)