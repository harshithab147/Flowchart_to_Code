"""
Module 1: Flowchart Vision & Image Processor
Author: Chethan S (Enhanced & Production Grade)

Features:
- 4-Point Perspective Transform & Whiteboard Deskewing
- Illumination Normalization & Shadow Removal
- Multi-shape detection (Oval, Rectangle, Diamond, Parallelogram, Circle)
- Directed arrow connection mapping with Hough Lines
- Confidence scoring & Annotated visual overlay
"""

import cv2
import numpy as np
import math
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class DetectedNode:
    """A detected geometric shape block."""
    id: int
    shape: str              # "oval", "rectangle", "diamond", "parallelogram", "circle", "unknown"
    x: int                  # Bounding box x
    y: int                  # Bounding box y
    w: int                  # Bounding box width
    h: int                  # Bounding box height
    cx: int                 # Center x
    cy: int                 # Center y
    confidence: float = 0.95
    contour_points: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectedEdge:
    """A directed connection between two nodes."""
    source_id: int
    target_id: int
    label: str = ""         # "Yes", "No", "True", "False", or ""
    confidence: float = 0.90

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShapeDetectionResult:
    """Full result of the vision stage."""
    nodes: List[DetectedNode] = field(default_factory=list)
    edges: List[DetectedEdge] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    deskewed: bool = False

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "deskewed": self.deskewed,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ImageEnhancer:
    """Advanced preprocessing for Whiteboard photos, scans, and hand-drawn diagrams."""

    @staticmethod
    def remove_shadows(img: np.ndarray) -> np.ndarray:
        """Removes uneven shadows and background lighting using division normalization."""
        rgb_planes = cv2.split(img)
        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(norm_img)
        return cv2.merge(result_planes)

    @staticmethod
    def auto_deskew(img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Attempts 4-point perspective transform if a page/whiteboard contour is found.
        Falls back to original image if no quadrilateral is prominent.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screen_cnt = None
        img_area = h * w

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) == 4 and area > 0.40 * img_area and area < 0.98 * img_area:
                screen_cnt = approx
                break

        if screen_cnt is None:
            return img, False

        # Order 4 points
        pts = screen_cnt.reshape(4, 2).astype("float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]        # Top-left
        rect[2] = pts[np.argmax(s)]        # Bottom-right
        rect[1] = pts[np.argmin(diff)]     # Top-right
        rect[3] = pts[np.argmax(diff)]     # Bottom-left

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped, True


class VisionProcessor:
    """
    Main Vision Processing Engine.
    Handles image reading, deskewing, contour & shape analysis, arrow matching, and overlay generation.
    """

    COLOR_PALETTE = {
        "oval": (46, 204, 113),          # Emerald Green (Start/End)
        "rectangle": (52, 152, 219),     # Bright Blue (Process)
        "diamond": (243, 156, 18),       # Orange / Amber (Decision)
        "parallelogram": (155, 89, 182), # Amethyst Purple (I/O)
        "circle": (26, 188, 156),        # Turquoise
        "unknown": (149, 165, 166),      # Gray
    }

    def __init__(self, deskew: bool = False, shadow_removal: bool = False):
        self.deskew = deskew
        self.shadow_removal = shadow_removal

    def process_image(self, image_input) -> Tuple[ShapeDetectionResult, np.ndarray, np.ndarray]:
        """
        Processes an image from a path or numpy array.
        Returns:
            (ShapeDetectionResult, processed_img, annotated_overlay_img)
        """
        # Load image if string path
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to decode image from: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError("Expected image_input to be a file path string or numpy.ndarray")

        is_deskewed = False
        if self.deskew:
            img, is_deskewed = ImageEnhancer.auto_deskew(img)

        if self.shadow_removal:
            img = ImageEnhancer.remove_shadows(img)

        h, w = img.shape[:2]

        # Convert to binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=13,
            C=6,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Detect shapes
        nodes = self._detect_nodes(binary_cleaned, w, h)

        # Detect edges
        edges = self._detect_edges(binary_cleaned, nodes)

        result = ShapeDetectionResult(
            nodes=nodes,
            edges=edges,
            image_width=w,
            image_height=h,
            deskewed=is_deskewed,
        )

        # Build annotated overlay
        annotated = self.render_annotated(img, result)

        return result, img, annotated

    def _detect_nodes(self, binary: np.ndarray, img_w: int, img_h: int) -> List[DetectedNode]:
        """Detect and classify all flowchart shape contours."""
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img_w * img_h
        min_area = max(400, int(0.0005 * img_area))
        max_area = int(0.25 * img_area)

        raw_nodes = []
        node_id = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h < 25 or h > 0.40 * img_h or w < 40 or w > 0.75 * img_w:
                continue

            aspect_ratio = float(w) / max(h, 1)
            if aspect_ratio < 0.35 or aspect_ratio > 4.5:
                continue

            cx, cy = x + w // 2, y + h // 2

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            shape, confidence = self._classify_shape(cnt, approx, w, h)

            raw_nodes.append(DetectedNode(
                id=node_id,
                shape=shape,
                x=x, y=y, w=w, h=h,
                cx=cx, cy=cy,
                confidence=confidence,
                contour_points=approx.reshape(-1, 2).tolist(),
            ))
            node_id += 1

        # Filter nested/duplicate bounding boxes
        filtered = self._filter_nested_nodes(raw_nodes)

        # Sort top-to-bottom for intuitive ID sequencing
        filtered.sort(key=lambda n: (n.cy // 50, n.cx))
        for i, node in enumerate(filtered):
            node.id = i

        return filtered

    def _classify_shape(self, contour, approx, w: int, h: int) -> Tuple[str, float]:
        """Determines if contour is oval, diamond, rectangle, parallelogram, or circle."""
        num_vertices = len(approx)
        aspect_ratio = float(w) / max(h, 1)
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = area / max(rect_area, 1)

        # 1. Oval / Start-End (High vertex count + smooth ellipse fit or rounded aspect)
        if num_vertices >= 5 and extent > 0.65:
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    ell_w, ell_h = ellipse[1]
                    ratio = min(ell_w, ell_h) / max(ell_w, ell_h, 1e-3)
                    # If roughly circular, classify as circle/connector, else oval
                    if ratio > 0.85 and 0.8 < aspect_ratio < 1.2:
                        return "circle", 0.95
                    return "oval", 0.92
                except Exception:
                    pass

        # 2. Diamond / Decision Block (4 vertices or rotated square extent)
        if num_vertices == 4:
            pts = approx.reshape(4, 2)
            # Center of points
            mean_pt = pts.mean(axis=0)
            dists = [math.hypot(p[0] - mean_pt[0], p[1] - mean_pt[1]) for p in pts]
            spread = (max(dists) - min(dists)) / max(np.mean(dists), 1)
            if 0.40 < extent < 0.78 and 0.5 < aspect_ratio < 2.0:
                return "diamond", 0.94

        # 3. Parallelogram / I/O Block (Skewed sides)
        if num_vertices == 4 and extent > 0.68:
            pts = approx.reshape(4, 2)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1).flatten()
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            left_skew = abs(tl[0] - bl[0])
            right_skew = abs(tr[0] - br[0])
            if (left_skew > w * 0.08 or right_skew > w * 0.08) and abs(left_skew - right_skew) < w * 0.15:
                return "parallelogram", 0.91

        # 4. Rectangle / Process Block
        if 4 <= num_vertices <= 6 and extent > 0.75:
            return "rectangle", 0.96

        # Circularity fallback
        peri = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area) / max(peri * peri, 1)
        if circularity > 0.75 and 0.7 < aspect_ratio < 1.4:
            return "circle", 0.88
        elif circularity > 0.55:
            return "oval", 0.82
        elif extent > 0.75:
            return "rectangle", 0.85

        return "rectangle" if extent > 0.65 else "unknown", 0.70

    def _filter_nested_nodes(self, nodes: List[DetectedNode]) -> List[DetectedNode]:
        """Removes interior text contours or double borders by checking bounding box overlap."""
        to_remove = set()
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i == j or i in to_remove or j in to_remove:
                    continue
                x_left = max(a.x, b.x)
                y_top = max(a.y, b.y)
                x_right = min(a.x + a.w, b.x + b.w)
                y_bottom = min(a.y + a.h, b.y + b.h)

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    area_a = a.w * a.h
                    area_b = b.w * b.h
                    min_area = min(area_a, area_b)
                    max_area = max(area_a, area_b)
                    if intersection_area / max(min_area, 1) > 0.65:
                        if max_area > 2.5 * min_area:
                            to_remove.add(i if area_a > area_b else j)
                        else:
                            to_remove.add(i if area_a < area_b else j)

        return [n for i, n in enumerate(nodes) if i not in to_remove]

    def _detect_edges(self, binary: np.ndarray, nodes: List[DetectedNode]) -> List[DetectedEdge]:
        """Detect connections between shapes using line segments + proximity fallback."""
        if len(nodes) < 2:
            return []

        # Mask out shapes
        line_mask = binary.copy()
        for n in nodes:
            pad = 8
            x1 = max(0, n.x - pad)
            y1 = max(0, n.y - pad)
            x2 = min(line_mask.shape[1], n.x + n.w + pad)
            y2 = min(line_mask.shape[0], n.y + n.h + pad)
            line_mask[y1:y2, x1:x2] = 0

        # Detect line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(line_mask, kernel, iterations=1)
        lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=25, minLineLength=15, maxLineGap=25)

        edges = []
        existing_pairs = set()

        if lines is not None:
            for l in lines:
                pts = np.array(l).flatten()
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                src = self._find_closest_node(x1, y1, nodes)
                tgt = self._find_closest_node(x2, y2, nodes)
                if src is not None and tgt is not None and src != tgt:
                    # Direction: by default flowcharts flow downward / rightward
                    s_node = nodes[src]
                    t_node = nodes[tgt]
                    if s_node.cy > t_node.cy + 30:
                        src, tgt = tgt, src
                    pair = (src, tgt)
                    if pair not in existing_pairs:
                        existing_pairs.add(pair)
                        label = ""
                        # If source is a decision diamond, check branch position
                        if s_node.shape == "diamond":
                            label = "Yes" if t_node.cx >= s_node.cx else "No"
                        edges.append(DetectedEdge(source_id=src, target_id=tgt, label=label))

        # Proximity topological fallback if arrows are disconnected
        if len(edges) < len(nodes) - 1:
            sorted_by_y = sorted(nodes, key=lambda n: n.cy)
            for i in range(len(sorted_by_y) - 1):
                src = sorted_by_y[i]
                tgt = sorted_by_y[i + 1]
                pair = (src.id, tgt.id)
                rev_pair = (tgt.id, src.id)
                if pair not in existing_pairs and rev_pair not in existing_pairs:
                    label = "Yes" if src.shape == "diamond" and i % 2 == 0 else ("No" if src.shape == "diamond" else "")
                    edges.append(DetectedEdge(source_id=src.id, target_id=tgt.id, label=label))
                    existing_pairs.add(pair)

        return edges

    def _find_closest_node(self, px: int, py: int, nodes: List[DetectedNode], max_dist: float = 120) -> Optional[int]:
        best_id = None
        best_d = max_dist
        for n in nodes:
            dx = max(n.x - px, 0, px - (n.x + n.w))
            dy = max(n.y - py, 0, py - (n.y + n.h))
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best_id = n.id
        return best_id

    def render_annotated(self, img: np.ndarray, result: ShapeDetectionResult) -> np.ndarray:
        """Draws polished high-contrast bounding boxes, arrows, and tags onto the image."""
        canvas = img.copy()

        # 1. Draw directed edges with glowing arrows
        for e in result.edges:
            src = next((n for n in result.nodes if n.id == e.source_id), None)
            tgt = next((n for n in result.nodes if n.id == e.target_id), None)
            if src and tgt:
                # Arrow line
                color = (0, 140, 255) if not e.label else ((0, 200, 0) if e.label == "Yes" else (0, 0, 230))
                cv2.arrowedLine(canvas, (src.cx, src.cy), (tgt.cx, tgt.cy), color, 2, tipLength=0.04)
                if e.label:
                    mid_x = (src.cx + tgt.cx) // 2
                    mid_y = (src.cy + tgt.cy) // 2
                    cv2.putText(canvas, e.label, (mid_x + 5, mid_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 2. Draw shapes and labels
        for n in result.nodes:
            bgr = self.COLOR_PALETTE.get(n.shape, (140, 140, 140))
            pts = np.array(n.contour_points, dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(canvas, [pts], True, bgr, 2, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (n.x, n.y), (n.x + n.w, n.y + n.h), bgr, 2)

            # Node chip badge
            badge_text = f"#{n.id} {n.shape.upper()} ({int(n.confidence*100)}%)"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, n.y - th - 8)
            cv2.rectangle(canvas, (n.x, badge_y1), (n.x + tw + 10, badge_y1 + th + 6), bgr, -1)
            cv2.putText(canvas, badge_text, (n.x + 5, badge_y1 + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas
