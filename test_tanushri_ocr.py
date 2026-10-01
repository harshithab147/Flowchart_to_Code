import cv2
import sys
import pytesseract

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

print("\n========== TANUSHRI OCR OUTPUT ==========\n")

for node in structure.nodes:

    x = node.x
    y = node.y
    w = node.w
    h = node.h

    # Original detected block
    crop = image[y:y+h, x:x+w]

    # -----------------------------------------
    # Remove shape borders.
    # Keep only the central text region.
    # -----------------------------------------

    left = int(w * 0.15)
    right = int(w * 0.85)
    top = int(h * 0.20)
    bottom = int(h * 0.80)

    text_crop = crop[top:bottom, left:right]

    # Grayscale
    gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)

    # Enlarge text
    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    # Threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR as a single line
    text = pytesseract.image_to_string(
        binary,
        config="--psm 7"
    )

    # Clean OCR result
    text = " ".join(text.split())

    print(
        f"Node {node.id}: "
        f"shape={node.shape} "
        f"-> text='{text}'"
    )

print("\n========================================")