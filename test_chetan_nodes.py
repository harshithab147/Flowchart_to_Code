import sys

# Add Chetan's module
sys.path.append("../step2")

from flowchart_processor import FlowchartProcessor


# IMPORTANT:
# Chetan's processor expects an IMAGE PATH, not cv2.imread(...)
image_path = "samples/sample_linear.png"

processor = FlowchartProcessor()

structure = processor.process(image_path)

print("\n========== CHETAN NODE OUTPUT ==========\n")

for node in structure.nodes:
    print(
        f"Node {node.id}: "
        f"shape={node.shape}, "
        f"x={node.x}, y={node.y}, "
        f"w={node.w}, h={node.h}, "
        f"center=({node.cx}, {node.cy})"
    )

print("\n========== EDGES ==========\n")

for edge in structure.edges:
    print(
        f"{edge.source_id} -> {edge.target_id}"
    )

print("\n========================================")