import cv2
import pytesseract

# Tell Python exactly where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load flowchart
image_path = "samples/sample_linear.png"
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image")
    exit()

print("Image loaded successfully")
print("Image size:", image.shape)

# --------------------------------------------------
# TEMPORARY TEST COORDINATES
# We will replace these with Chetan's detected nodes
# --------------------------------------------------

# Get image dimensions
height, width = image.shape[:2]

# OCR the complete image after preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(
    gray,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC
)

_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

text = pytesseract.image_to_string(
    thresh,
    config="--psm 6"
)

print("\n========== PREPROCESSED OCR ==========")
print(repr(text))
print("======================================")