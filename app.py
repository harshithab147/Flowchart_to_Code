"""
FlowLogic AI: Flowchart to Code & Reverse Visual IDE
Full Team Integration (Chethan, Tanushri, Harshitha, Shubhada)

Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import json
import tempfile
from PIL import Image

# Import backend modules
from modules.vision_processor import VisionProcessor, ShapeDetectionResult
from modules.ocr_engine import OCREngine, FlowchartIR, FlowNodeType
from modules.code_generator import CodeGenerator, SupportedLanguage, ComplexityEngine
from modules.reverse_engine import ReverseEngine
from modules.sandbox import ExecutionSandbox
from modules.exporter import FlowchartExporter

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FlowLogic AI | Flowchart to Code Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for Dark & Light Glassmorphism UI
# -----------------------------------------------------------------------------
def apply_theme_css(theme: str):
    if theme == "Dark":
        bg_color = "#0b0f19"
        card_bg = "rgba(17, 24, 39, 0.75)"
        text_color = "#f3f4f6"
        subtext_color = "#9ca3af"
        border_color = "rgba(55, 65, 81, 0.6)"
        accent_color = "#06b6d4"
        accent_glow = "rgba(6, 182, 212, 0.25)"
        terminal_bg = "#030712"
    else:
        bg_color = "#f8fafc"
        card_bg = "rgba(255, 255, 255, 0.85)"
        text_color = "#0f172a"
        subtext_color = "#475569"
        border_color = "rgba(203, 213, 225, 0.8)"
        accent_color = "#0284c7"
        accent_glow = "rgba(2, 132, 199, 0.15)"
        terminal_bg = "#1e293b"

    css = f"""
    <style>
        /* Main Container */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* Glassmorphism Cards */
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 {accent_glow};
        }}

        .metric-badge {{
            display: inline-block;
            background: linear-gradient(135deg, {accent_color}, #8b5cf6);
            color: #ffffff;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        /* Header Logo */
        .header-title {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, {accent_color}, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }}

        .header-sub {{
            color: {subtext_color};
            font-size: 1.0rem;
            margin-top: -5px;
            margin-bottom: 20px;
        }}

        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }}

        /* Terminal Window */
        .terminal-box {{
            background-color: {terminal_bg};
            color: #10b981;
            font-family: 'Fira Code', 'Courier New', monospace;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid {border_color};
            font-size: 0.9rem;
            line-height: 1.5;
            white-space: pre-wrap;
            overflow-x: auto;
        }}

        /* Team Badges */
        .team-chip {{
            display: inline-block;
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.4);
            color: {text_color};
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.8rem;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ System Settings")
        theme = st.radio("🎨 Theme Mode", ["Dark", "Light"], horizontal=True)

        st.markdown("---")
        st.markdown("### 🚀 Engine Navigation")
        app_mode = st.selectbox(
            "Select Mode",
            [
                "⚡ Flowchart ➔ Polyglot Code Studio",
                "🔄 Code ➔ Flowchart (Reverse IDE)",
                "👥 Team & Architecture Overview",
            ],
        )

        st.markdown("---")
        st.markdown("### 🧠 AI Engine Config (Optional)")
        ai_provider = st.selectbox("AI Model Provider", ["Deterministic (Offline / Free)", "Google Gemini", "OpenAI GPT-4o"])
        api_key = ""
        if ai_provider != "Deterministic (Offline / Free)":
            api_key = st.text_input("Enter API Key", type="password", help="Leave blank to use the offline deterministic compiler")

        st.markdown("---")
        st.markdown("### 🛠️ Image Preprocessing")
        deskew_opt = st.checkbox("📐 Whiteboard Auto-Deskew", value=False, help="Straightens photos of whiteboards taken at an angle")
        shadow_opt = st.checkbox("💡 Shadow & Glare Removal", value=False, help="Normalizes uneven lighting and shadows")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94a3b8; text-align: center;">
                FlowLogic AI Studio v2.5<br>
                Integrated Engineering Project
            </div>
            """,
            unsafe_allow_html=True,
        )

        return theme, app_mode, ai_provider, api_key, deskew_opt, shadow_opt


# -----------------------------------------------------------------------------
# Mode 1: Flowchart ➔ Polyglot Code Studio
# -----------------------------------------------------------------------------
def render_flowchart_to_code_mode(ai_provider, api_key, deskew_opt, shadow_opt):
    st.markdown('<div class="header-title">⚡ Flowchart to Polyglot Code Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Transform flowchart diagrams into production code across 7 programming languages with Big-O complexity analysis and live sandbox execution.</div>', unsafe_allow_html=True)

    # Input Section: Upload or Preset Samples
    col_input_1, col_input_2 = st.columns([1, 1])

    with col_input_1:
        st.markdown("##### 📁 1. Upload Flowchart Image")
        uploaded_file = st.file_uploader(
            "Upload PNG, JPG, or JPEG diagram",
            type=["png", "jpg", "jpeg"],
            help="Upload any digital diagram or photo of a whiteboard",
        )

    with col_input_2:
        st.markdown("##### 🎯 Or Choose a Sample Preset")
        sample_choice = st.selectbox(
            "Select Preset Flowchart",
            [
                "None (Use Uploaded File)",
                "1. If-Else Decision (Positive or Negative Number)",
                "2. Linear Calculation (Double Input Value)",
                "3. Iterative Loop (Factorial Calculation)",
            ],
        )

    # Resolve Image Source
    img_to_process = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif sample_choice != "None (Use Uploaded File)":
        if "1." in sample_choice:
            sample_path = os.path.join("samples", "sample_decision.png")
        elif "2." in sample_choice:
            sample_path = os.path.join("samples", "sample_linear.png")
        else:
            sample_path = os.path.join("samples", "sample_loop.png")

        if os.path.exists(sample_path):
            img_to_process = cv2.imread(sample_path)

    if img_to_process is None:
        st.info("👆 Please upload a flowchart image or select a sample preset above to begin processing.")
        return

    # Process Diagram with Vision Engine (Chethan's Module)
    with st.spinner("🔍 OpenCV Vision Engine: Detecting shapes, deskewing & mapping connection arrows..."):
        vision_engine = VisionProcessor(deskew=deskew_opt, shadow_removal=shadow_opt)
        vision_res, processed_img, annotated_img = vision_engine.process_image(img_to_process)

    # Process with OCR Engine (Tanushri's Module)
    with st.spinner("📝 OCR & Logic Engine: Extracting node texts & building Intermediate Representation..."):
        ocr_engine = OCREngine()
        ir = ocr_engine.build_ir_from_vision(processed_img, vision_res)

    # -------------------------------------------------------------------------
    # Main Split View: Diagram & Overlays vs Code Studio
    # -------------------------------------------------------------------------
    col_vis, col_code = st.columns([1, 1], gap="large")

    with col_vis:
        st.markdown("### 🖼️ Flowchart Vision Canvas")
        tab_anno, tab_orig = st.tabs(["🎯 Detected Shapes Overlay", "📷 Original Image"])
        with tab_anno:
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True, caption=f"Detected {len(vision_res.nodes)} Nodes & {len(vision_res.edges)} Connecting Edges")
        with tab_orig:
            orig_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, use_container_width=True, caption="Source Flowchart")

        # Interactive Node Inspector & Editor
        with st.expander("✏️ Interactive Node Inspector & Text Override", expanded=False):
            st.markdown("Review and modify extracted texts if needed before code generation:")
            overrides = {}
            for n in ir.nodes:
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.write(f"**Node #{n.id}** ({n.shape})")
                with c2:
                    new_val = st.text_input(f"Text for #{n.id}", value=n.raw_text, key=f"node_text_{n.id}")
                    if new_val != n.raw_text:
                        overrides[n.id] = new_val

            if overrides:
                ir = ocr_engine.build_ir_from_vision(processed_img, vision_res, override_texts=overrides)
                st.success("✅ Intermediate Representation updated with your overrides!")

    with col_code:
        st.markdown("### 💻 Polyglot Multi-Language Code Studio")

        # Code Generation (Harshitha's Module)
        provider_name = "none"
        if "Gemini" in ai_provider:
            provider_name = "gemini"
        elif "OpenAI" in ai_provider:
            provider_name = "openai"

        code_gen = CodeGenerator(api_key=api_key if api_key else None, provider=provider_name)
        code_result = code_gen.generate(ir)

        # Multi-Language Tabs
        lang_tabs = st.tabs([
            "🐍 Python", "⚡ C++", "☕ Java", "🌐 JavaScript", "⚙️ C", "🐹 Go", "🦀 Rust"
        ])

        lang_map = {
            0: (SupportedLanguage.PYTHON.value, "python"),
            1: (SupportedLanguage.CPP.value, "cpp"),
            2: (SupportedLanguage.JAVA.value, "java"),
            3: (SupportedLanguage.JAVASCRIPT.value, "javascript"),
            4: (SupportedLanguage.C.value, "c"),
            5: (SupportedLanguage.GO.value, "go"),
            6: (SupportedLanguage.RUST.value, "rust"),
        }

        for idx, tab in enumerate(lang_tabs):
            with tab:
                lang_name, lang_code = lang_map[idx]
                code_text = code_result.codes.get(lang_name, "")
                st.code(code_text, language=lang_code)

                # Download button
                file_ext = {
                    "Python": ".py", "C++": ".cpp", "Java": ".java",
                    "JavaScript": ".js", "C": ".c", "Go": ".go", "Rust": ".rs"
                }.get(lang_name, ".txt")

                st.download_button(
                    label=f"💾 Download {lang_name} Source ({file_ext})",
                    data=code_text,
                    file_name=f"flowchart_algorithm{file_ext}",
                    mime="text/plain",
                    key=f"dl_{lang_name}",
                )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Bottom Section: Complexity Analyzer, Live Sandbox & Unit Tests
    # -------------------------------------------------------------------------
    col_bottom_1, col_bottom_2 = st.columns([1, 1], gap="large")

    with col_bottom_1:
        st.markdown("### 📊 Algorithmic Complexity Analyzer (Big-O)")
        if code_result.complexity:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("⏱️ Time Complexity", code_result.complexity.time_complexity)
            with c2:
                st.metric("💾 Space Complexity", code_result.complexity.space_complexity)
            with c3:
                st.metric("🔢 Total Nodes", len(ir.nodes))

            st.markdown(f"**Formal Proof:** {code_result.complexity.reasoning}")
            if code_result.complexity.optimizations:
                st.markdown(f"**💡 Recommended Optimizations:** {', '.join(code_result.complexity.optimizations)}")

        # Line-by-Line Logic Walkthrough
        with st.expander("📖 Line-by-Line Flowchart Logic Walkthrough", expanded=False):
            for exp in code_result.line_explanations:
                st.markdown(f"- **#{exp['node_id']} [{exp['type']}] ({exp['shape']}):** {exp['explanation']}")

    with col_bottom_2:
        st.markdown("### 🧪 Live Execution Sandbox & Test Runner")
        tab_live, tab_tests = st.tabs(["▶ Interactive Terminal", "⚡ Auto Unit Test Suite"])

        with tab_live:
            py_code = code_result.codes.get("Python", "")
            user_stdin = st.text_input("Simulate Console Input (stdin)", value="25", help="Value passed to input() statements")
            if st.button("▶ Run Python Code Live in Sandbox", type="primary"):
                with st.spinner("Executing in safe sandbox..."):
                    exec_res = ExecutionSandbox.run_python_code(py_code, user_input=user_stdin + "\n")
                    if exec_res.success:
                        st.markdown(
                            f'<div class="terminal-box"><b>[Execution Status: SUCCESS ({exec_res.execution_time_ms} ms)]</b><br><br>{exec_res.stdout}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="terminal-box" style="color: #ef4444;"><b>[Execution Status: ERROR ({exec_res.execution_time_ms} ms)]</b><br><br>{exec_res.error_message}<br><br>{exec_res.stderr}</div>',
                            unsafe_allow_html=True,
                        )

        with tab_tests:
            st.markdown("Run 5+ auto-generated boundary, edge, and stress test cases:")
            if st.button("⚡ Run Full Automated Test Suite"):
                with st.spinner("Running automated test suite..."):
                    test_results = ExecutionSandbox.run_test_suite(py_code, code_result.test_cases)
                    passed_count = sum(1 for t in test_results if t.passed)
                    st.markdown(f"**Test Results: {passed_count}/{len(test_results)} Passed**")
                    for tr in test_results:
                        badge = "✅ PASS" if tr.passed else "❌ FAIL"
                        with st.expander(f"{badge} | Test #{tr.test_id}: {tr.test_name} ({tr.execution_time_ms} ms)"):
                            st.write(f"**Input:** `{tr.input_data}`")
                            st.write(f"**Expected:** {tr.expected_behavior}")
                            st.write(f"**Actual Output:** `{tr.actual_output}`")

    # -------------------------------------------------------------------------
    # Export Center
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📦 Export Center")
    exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)

    with exp_c1:
        mermaid_code = FlowchartExporter.to_mermaid(ir)
        st.download_button("📊 Download Mermaid.js (.mmd)", data=mermaid_code, file_name="diagram.mmd", mime="text/plain")

    with exp_c2:
        plantuml_code = FlowchartExporter.to_plantuml(ir)
        st.download_button("📐 Download PlantUML (.puml)", data=plantuml_code, file_name="diagram.puml", mime="text/plain")

    with exp_c3:
        json_ir = FlowchartExporter.to_json_ir(ir)
        st.download_button("📄 Download JSON Schema (.json)", data=json_ir, file_name="flowchart_ir.json", mime="application/json")

    with exp_c4:
        full_report = FlowchartExporter.generate_full_report(ir, code_result)
        st.download_button("📋 Download Project Report (.md)", data=full_report, file_name="ALGORITHM_REPORT.md", mime="text/markdown")


# -----------------------------------------------------------------------------
# Mode 2: Code ➔ Flowchart (Reverse Visual IDE)
# -----------------------------------------------------------------------------
def render_code_to_flowchart_mode():
    st.markdown('<div class="header-title">🔄 Reverse Engine: Code to Interactive Flowchart</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Paste any Python algorithm to automatically parse its Abstract Syntax Tree (AST) and generate an interactive Mermaid.js diagram.</div>', unsafe_allow_html=True)

    default_code = """def check_number():
    num = float(input("Enter number: "))
    if num >= 0:
        print("Positive")
    else:
        print("Negative")
"""

    sample_code_choice = st.selectbox(
        "Load Preset Code Sample",
        [
            "Custom Code",
            "1. Positive / Negative Number Checker",
            "2. Double a Value Calculation",
        ],
    )

    if "1." in sample_code_choice:
        code_input = default_code
    elif "2." in sample_code_choice:
        code_input = """def double_val():
    x = float(input("Enter x: "))
    result = x * 2
    print(result)
"""
    else:
        code_input = default_code

    col_editor, col_diagram = st.columns([1, 1], gap="large")

    with col_editor:
        st.markdown("### 📝 Python Source Code")
        code_text = st.text_area("Write or paste Python code below:", value=code_input, height=350)
        generate_clicked = st.button("🔄 Generate Flowchart from Code", type="primary")

    with col_diagram:
        st.markdown("### 📊 Auto-Generated Flowchart Diagram")
        if code_text:
            rev_ir = ReverseEngine.code_to_ir(code_text)
            mermaid_markup = ReverseEngine.ir_to_mermaid(rev_ir)

            st.markdown("#### Mermaid.js Flow Diagram:")
            # Render using Streamlit component or markdown
            st.markdown(f"```mermaid\n{mermaid_markup}\n```")

            with st.expander("🔍 View Raw Mermaid Code"):
                st.code(mermaid_markup, language="mermaid")

            st.download_button(
                "💾 Download Mermaid Diagram (.mmd)",
                data=mermaid_markup,
                file_name="reverse_flowchart.mmd",
                mime="text/plain",
            )


# -----------------------------------------------------------------------------
# Mode 3: Team & Architecture Overview
# -----------------------------------------------------------------------------
def render_team_overview():
    st.markdown('<div class="header-title">👥 Project Architecture & Team Responsibilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">AI-Based Flowchart to Code Generator — Final Year Capstone Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h3>🏗️ End-to-End System Pipeline</h3>
            <p>
                <b>Flowchart Image</b> ➔ <b>[Chethan: OpenCV Vision]</b> ➔ <b>[Tanushri: OCR & IR Engine]</b> ➔ <b>[Harshitha: AI Polyglot Code Gen]</b> ➔ <b>[Shubhada: Streamlit Interactive Studio]</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h4>👤 Chethan S — Flowchart Image Processing</h4>
                <div class="team-chip">OpenCV 5.0</div>
                <div class="team-chip">Contour Geometry</div>
                <div class="team-chip">Perspective Deskewing</div>
                <div class="team-chip">Hough Arrow Detection</div>
                <ul>
                    <li>Image normalization and shadow removal.</li>
                    <li>Shape classification (Oval, Rect, Diamond, Parallelogram).</li>
                    <li>Directed arrow connection graph builder.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <h4>👤 Harshitha B — AI & Polyglot Code Generation</h4>
                <div class="team-chip">Python 3</div>
                <div class="team-chip">C++17 / Java</div>
                <div class="team-chip">Big-O Analyzer</div>
                <div class="team-chip">Unit Test Suite</div>
                <ul>
                    <li>Multi-language transpiler across 7 languages.</li>
                    <li>Big-O Time & Space complexity calculation engine.</li>
                    <li>Automated edge/boundary test case generator.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h4>👤 Tanushri MP — OCR & Logic Processing</h4>
                <div class="team-chip">Tesseract OCR</div>
                <div class="team-chip">Intermediate Representation (IR)</div>
                <div class="team-chip">Graph Topological Sort</div>
                <ul>
                    <li>Cropped region text extraction.</li>
                    <li>Block classification (Start/End, Input, Output, Decision).</li>
                    <li>Flowchart graph integrity validation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <h4>👤 Shubhada Umesh — UI & Full System Integration</h4>
                <div class="team-chip">Streamlit Web UI</div>
                <div class="team-chip">Reverse Visual IDE</div>
                <div class="team-chip">Interactive Sandbox</div>
                <div class="team-chip">Export Engine</div>
                <ul>
                    <li>Dark/Light responsive glassmorphic user interface.</li>
                    <li>Live in-browser Python execution terminal.</li>
                    <li>Mermaid.js, PlantUML, and Markdown Report exporter.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    theme, app_mode, ai_provider, api_key, deskew_opt, shadow_opt = render_sidebar()
    apply_theme_css(theme)

    if "Flowchart ➔ Polyglot" in app_mode:
        render_flowchart_to_code_mode(ai_provider, api_key, deskew_opt, shadow_opt)
    elif "Code ➔ Flowchart" in app_mode:
        render_code_to_flowchart_mode()
    else:
        render_team_overview()


if __name__ == "__main__":
    main()
