import cv2
import sys
import pytesseract
import json
from modules.ocr_engine import FlowchartIR, FlowNode, FlowEdge, FlowNodeType
sys.path.append("../step2")
from flowchart_processor import FlowchartProcessor


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = "samples/sample_linear.png"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not read image")
    sys.exit()

processor = FlowchartProcessor()
structure = processor.process(image_path)


def classify_node(shape, text):

    text_lower = text.lower().strip()

    # Start / End
    if "start" in text_lower or "begin" in text_lower:
        return "START"

    if "end" in text_lower or "stop" in text_lower:
        return "END"

    # Decision
    if shape == "diamond":
        return "DECISION"

    # Input
    if any(word in text_lower for word in [
        "input",
        "enter",
        "read",
        "accept"
    ]):
        return "INPUT"

    # Output
    if any(word in text_lower for word in [
        "print",
        "display",
        "show",
        "output"
    ]):
        return "OUTPUT"

    # Process
    if shape == "rectangle":
        return "PROCESS"

    return "UNKNOWN"


print("\n========== TANUSHRI LOGIC OUTPUT ==========\n")

logical_nodes = []

for node in structure.nodes:

    x = node.x
    y = node.y
    w = node.w
    h = node.h

    crop = image[y:y+h, x:x+w]

    # Keep central text region
    left = int(w * 0.15)
    right = int(w * 0.85)
    top = int(h * 0.20)
    bottom = int(h * 0.80)

    text_crop = crop[top:bottom, left:right]

    gray = cv2.cvtColor(
        text_crop,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text = pytesseract.image_to_string(
        binary,
        config="--psm 7"
    )

    text = " ".join(text.split())

    node_type = classify_node(
        node.shape,
        text
    )

    logical_nodes.append({
        "id": node.id,
        "type": node_type,
        "text": text
    })

    print(
        f"Node {node.id}: "
        f"{node_type} "
        f"-> {text}"
    )


# Create edge list
logical_edges = []

for edge in structure.edges:

    logical_edges.append({
        "source": edge.source_id,
        "target": edge.target_id
    })


# Final IR
flow_nodes = []

for n in logical_nodes:
    flow_nodes.append(
        FlowNode(
            id=n["id"],
            node_type=FlowNodeType(n["type"]),
            raw_text=n["text"],
            cleaned_expression=n["text"],
            shape="",
            x=0,
            y=0,
            w=0,
            h=0
        )
    )

flow_edges = []

for e in logical_edges:
    flow_edges.append(
        FlowEdge(
            source_id=e["source"],
            target_id=e["target"]
        )
    )

flowchart_ir = FlowchartIR(
    nodes=flow_nodes,
    edges=flow_edges
)


print("\n========== FLOWCHART IR ==========\n")

print(
    json.dumps(
        flowchart_ir.to_dict(),
        indent=2
    )
)

print("\n===================================")