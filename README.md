# 🤖 FlowLogic AI — Flowchart to Code & Reverse Visual IDE

An AI-powered system that accepts flowchart diagrams (digital, scans, or whiteboard photos), understands their algorithmic structure and logic, and automatically generates production code across **7 programming languages** (Python, C++, Java, JavaScript, C, Go, Rust), analyzes **Big-O complexity**, runs **automated unit tests**, and executes code live in a **browser sandbox**.

Includes a **Bidirectional Visual Engine** (Code ➔ Interactive Flowchart diagram).

---

## 👥 Team & Architecture

| Team Member | Role & Module | Tech Stack |
|---|---|---|
| **Chethan S** | **Flowchart Vision & Image Processing** | OpenCV, Contours, Deskewing, Hough Transforms |
| **Tanushri MP** | **OCR & Intermediate Representation (IR)** | Text Extraction, Semantic Classification, IR Schema |
| **Harshitha B** | **Polyglot Code Gen & Complexity Engine** | Python/C++/Java/JS/C/Go/Rust, Big-O Analyzer, Unit Tests |
| **Shubhada Umesh** | **Streamlit UI & System Integration** | Streamlit, Reverse AST IDE, Sandbox Runner, Exporter |

---

## 🚀 Key Features

1. **⚡ Polyglot Multi-Language Code Generation:**
   - One-click generation in **Python 3, C++17, Java 17, JavaScript (ES6), C99, Go, and Rust**.
2. **🔄 Bidirectional Reverse IDE (Code ➔ Flowchart):**
   - Paste any Python code; the AST parser automatically reconstructs an interactive **Mermaid.js flowchart diagram**.
3. **📊 Big-O Algorithmic Complexity Analyzer:**
   - Computes Time Complexity (e.g. $O(1), O(N), O(N^2)$) and Space Complexity with mathematical justifications and optimization recommendations.
4. **🧪 Live Execution Sandbox & Auto Test Suite:**
   - Run generated code in a safe browser sandbox with simulated user inputs.
   - Auto-generates 5+ boundary, edge, and stress test cases with a 1-click test runner.
5. **📐 Whiteboard & Camera Photo Preprocessor:**
   - 4-point perspective warp for angled phone captures.
   - Division normalization for shadow and uneven lighting removal.
6. **🎨 Futuristic UI with Light & Dark Mode:**
   - Sleek glassmorphism dashboard with side-by-side visualization, interactive node text editor, and multi-format exporters (Mermaid `.mmd`, PlantUML `.puml`, JSON `.json`, and Markdown Project Reports).

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running the Automated Test Suite

Run the full end-to-end test suite covering all modules:
```bash
python tests/test_all_modules.py
```
All unit tests validate Vision contour detection, OCR IR construction, 7-language syntax compilation, reverse AST parsing, execution sandbox, and export formats.

---

## 📁 Project Structure

```
FlowchartToCode/
├── app.py                          # Streamlit Web App (Dark/Light glassmorphism UI)
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
├── modules/
│   ├── __init__.py
│   ├── vision_processor.py         # Module 1 : OpenCV shape & arrow detection
│   ├── ocr_engine.py               # Module 2 : OCR, semantic blocks & IR builder
│   ├── code_generator.py           # Module 3 : Polyglot generation, Big-O, unit tests
│   ├── reverse_engine.py           # Reverse Engine: Python AST to Mermaid Flowchart
│   ├── sandbox.py                  # Live execution terminal & unit test runner
│   └── exporter.py                 # Mermaid, PlantUML, JSON, and Report export
├── samples/                        # Sample diagrams for instant testing
│   ├── generate_samples.py
│   ├── sample_decision.png
│   ├── sample_linear.png
│   └── sample_loop.png
└── tests/
    ├── __init__.py
    └── test_all_modules.py         # Automated test suite (11/11 passing)
```
