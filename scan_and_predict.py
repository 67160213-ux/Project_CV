import cv2
import numpy as np
import os
import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "char_classifier.pkl")
OUTPUT_VISUALS_DIR = os.path.join(BASE_DIR, "outputs", "visuals")
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

os.makedirs(OUTPUT_VISUALS_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

def load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Classifier not found at {MODEL_PATH}. Run train_model.py first.")
    return joblib.load(MODEL_PATH)

def preprocess_and_warp(image_path, target_w=1000, target_h=1200):
    """
    Robust table alignment and perspective warp:
    Uses bilateral filter + edge detection with directional vertical dilation
    to isolate the 20x20 table grid without bridging into timing marks or question numbers.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 20, 60)
    
    # Vertical dilation bridges inter-row gaps vertically without expanding horizontally into timing marks
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    dilated = cv2.dilate(edges, kernel_v, iterations=2)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilated = cv2.dilate(dilated, kernel_h, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    c = contours[0]
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    pts = approx.reshape(-1, 2).astype("float32")
    
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    return warped

def align_centroid(img_28):
    """
    Shifts the character so that its center of mass (centroid)
    is positioned exactly at the center of the 28x28 canvas (14, 14).
    Uses image moments (standard technique from MNIST/EMNIST).
    """
    M = cv2.moments(img_28)
    if M["m00"] > 10:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = np.clip(14.0 - cx, -4.0, 4.0)
        shift_y = np.clip(14.0 - cy, -4.0, 4.0)
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(img_28, M_shift, (28, 28), borderValue=0)
    return img_28

def compute_hog_28(img_28):
    """
    Fast vectorized HOG feature extraction for 28x28 character images:
    - 7x7 grid of 4x4 pixel cells
    - 8 gradient orientation bins (0 to 180 degrees)
    - Output: 7 x 7 x 8 = 392 features
    """
    img = img_28.astype(np.float32)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0
    bin_idx = np.clip((ang / 22.5).astype(np.int32), 0, 7)

    mag_cells = mag.reshape(7, 4, 7, 4).transpose(0, 2, 1, 3)
    bin_cells = bin_idx.reshape(7, 4, 7, 4).transpose(0, 2, 1, 3)

    hist = np.zeros((7, 7, 8), dtype=np.float32)
    for b in range(8):
        hist[:, :, b] = np.sum(mag_cells * (bin_cells == b), axis=(2, 3))

    norm = np.sqrt(np.sum(hist**2) + 1e-6)
    hist = np.clip(hist / norm, 0, 0.2)
    hist = hist / np.sqrt(np.sum(hist**2) + 1e-6)
    return hist.flatten()

def extract_character_features(cell_bgr, return_crop=False, return_meta=False):
    """
    Isolates character inside cell, centers via Centroid Alignment,
    and returns combined (Raw Pixels 784 + HOG 392 = 1,176 features).
    Includes:
    1. Contrast & Variance Gatekeeper to eliminate pure blank paper noise.
    2. Gentle morphological closing to heal hair-thin pencil micro-breaks.
    3. Precise border filtering that never clips thin letters like 'I', 'L', 'T'.
    4. Multi-Component Intelligent Stroke Merging to assemble broken/multi-stroke characters (H, E, A, K, etc.).
    """
    ch, cw = cell_bgr.shape[:2]
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Gatekeeper: If cell is clean blank paper with virtually no contrast
    pixel_diff = int(np.max(gray)) - int(np.min(gray))
    pixel_std = float(np.std(gray))
    if pixel_std < 7.0 or pixel_diff < 35:
        if return_meta:
            return (None, None, None)
        return (None, None) if return_crop else None
        
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Gentle morphological close (2x2) to heal hair-thin pencil micro-breaks without bloating pen strokes
    k_heal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k_heal)
    
    # Clear outer boundary pixels to disconnect border remnants without cutting low strokes
    bin_inv[:1, :] = 0
    bin_inv[-1:, :] = 0
    bin_inv[:, :2] = 0
    bin_inv[:, -2:] = 0
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_inv)
    
    # 3. Filter border lines & collect valid character stroke candidates
    candidates = []
    for lbl in range(1, num_labels):
        bx, by, bw, bh, area = stats[lbl]
        
        # Precise grid border filter:
        # A true border line spans almost the full cell height/width or touches the boundary
        is_vertical_border = (
            bh >= ch - 5 and (by <= 1 or by + bh >= ch - 1) and
            (bh / max(bw, 1) >= 2.8) and (bx <= 7 or bx + bw >= cw - 7)
        )
        is_horizontal_border = (
            bw >= cw - 5 and (by <= 1 or by + bh >= ch - 1) and
            (bw / max(bh, 1) >= 2.8)
        )
        
        if is_vertical_border or is_horizontal_border:
            continue
            
        # Filter micro-noise dots
        if area > 12 and bh > 5:
            candidates.append((lbl, bx, by, bw, bh, area))
            
    if not candidates:
        if return_meta:
            return (None, None, None)
        return (None, None) if return_crop else None
        
    # 4. Multi-Component Intelligent Stroke Merging:
    # Sort candidates by area to find primary stroke (largest component)
    candidates.sort(key=lambda x: x[5], reverse=True)
    p_lbl, p_x, p_y, p_w, p_h, p_area = candidates[0]
    
    merged_lbls = [p_lbl]
    # Merge companion strokes of the same character (e.g., crossbars, second vertical stems)
    for (lbl, bx, by, bw, bh, area) in candidates[1:]:
        dist_x = max(0, bx - (p_x + p_w), p_x - (bx + bw))
        dist_y = max(0, by - (p_y + p_h), p_y - (by + bh))
        
        # Prevent merging border line remnants close to the outer cell edges
        if (bx <= 3 or bx + bw >= cw - 3) and (bh / max(bw, 1) >= 2.5):
            continue

        # A stroke belongs to the same character if it is within typical character span
        if dist_x <= 16 and dist_y <= 10 and (area >= 0.10 * p_area or bh >= 0.30 * p_h):
            merged_lbls.append(lbl)
            
    char_mask = np.isin(labels, merged_lbls).astype(np.uint8) * 255
    pts = np.argwhere(char_mask > 0)
    if len(pts) == 0:
        if return_meta:
            return (None, None, None)
        return (None, None) if return_crop else None
        
    y0, x0 = pts.min(axis=0)
    y1, x1 = pts.max(axis=0) + 1
    char_crop = char_mask[y0:y1, x0:x1]
    bh, bw = char_crop.shape[:2]
    
    max_dim = max(bw, bh)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    sq_x = (max_dim - bw) // 2
    sq_y = (max_dim - bh) // 2
    square[sq_y : sq_y + bh, sq_x : sq_x + bw] = char_crop
    
    pad = int(max_dim * 0.25)
    padded = cv2.copyMakeBorder(square, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    char_28 = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 1. Centroid Alignment (Image Moments / Center of Mass)
    char_28 = align_centroid(char_28)
    
    # 2. Raw Normalized Pixels (784 features)
    pix_feat = char_28.flatten() / 255.0
    
    # 3. HOG Gradient Features (392 features)
    hog_feat = compute_hog_28(char_28)
    
    # Combined feature vector (1,176 features)
    feat = np.hstack([pix_feat, hog_feat]).astype(np.float32)
    stroke_meta = {
        "x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1),
        "bw": int(bw), "bh": int(bh), "cw": int(cw), "ch": int(ch)
    }
    if return_meta:
        return (feat, char_crop, stroke_meta)
    return (feat, char_crop) if return_crop else feat

def detect_table_rows(warped):
    """Dynamically detects exact row vertical bounds (top, bottom) for all 20 rows"""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k)
    # Project horizontal lines on clean right margin (columns 20-19)
    h_proj = np.sum(h_lines[:, 500:950], axis=1) / 255.0
    
    h_peaks = np.where(h_proj > 30)[0]
    grouped_h = []
    if len(h_peaks) > 0:
        cur = [h_peaks[0]]
        for p in h_peaks[1:]:
            if p - cur[-1] <= 4:
                cur.append(p)
            else:
                grouped_h.append(int(np.mean(cur)))
                cur = [p]
        grouped_h.append(int(np.mean(cur)))
        
    row_boxes = []
    i = 0
    while i < len(grouped_h) - 1:
        d = grouped_h[i+1] - grouped_h[i]
        if 25 <= d <= 55:
            row_boxes.append((grouped_h[i], grouped_h[i+1]))
            i += 2
        else:
            i += 1
            
    # Robust fallback if some lines are faint
    if len(row_boxes) != 20:
        row_boxes = []
        for r in range(20):
            y_t = int(10 + r * 59.2)
            y_b = y_t + 45
            row_boxes.append((y_t, y_b))
            
    return row_boxes

def detect_table_columns(warped):
    """
    Dynamically detects exact column horizontal bounds (x1, x2) for all 20 columns.
    Identifies table left and right borders and computes true column boundaries,
    completely eliminating accumulative column drift.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k)
    v_proj = np.sum(v_lines[100:1100, :], axis=0) / 255.0
    
    left_candidates = np.where(v_proj[:35] > 150)[0]
    x_left = float(left_candidates[0]) if len(left_candidates) > 0 else 6.0
    
    right_candidates = np.where(v_proj[965:] > 150)[0]
    x_right = float(965 + right_candidates[-1]) if len(right_candidates) > 0 else (x_left + 990.0)
    
    col_w = (x_right - x_left) / 20.0
    col_boxes = []
    for c in range(20):
        x1 = int(round(x_left + c * col_w))
        x2 = int(round(x_left + (c + 1) * col_w))
        col_boxes.append((x1, x2))
    return col_boxes

def scan_sheet(image_path, save_annotated=True):
    clf = load_classifier()
    warped = preprocess_and_warp(image_path)
    
    row_boxes = detect_table_rows(warped)
    col_boxes = detect_table_columns(warped)
    annotated = warped.copy()
    sheet_data = {}
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    classes_list = list(clf.classes_)
    
    print("\n" + "="*50)
    print(f" SCANNING: {os.path.basename(image_path)} ")
    print("="*50)
    
    for r, (y_top, y_bot) in enumerate(row_boxes):
        question_num = r + 1
        sheet_data[question_num] = []
        
        for c, (x1, x2) in enumerate(col_boxes):
            pad_y = 4
            pad_x = 5
            cell_roi = warped[y_top + pad_y : y_bot - pad_y, x1 + pad_x : x2 - pad_x]
            
            feat, char_crop, stroke_meta = extract_character_features(cell_roi, return_crop=True, return_meta=True)
            
            if feat is not None:
                pred_label = clf.predict([feat])[0]
                probs = clf.predict_proba([feat])[0]
                confidence = float(np.max(probs))
                
                # --- Topological Disambiguation for Confusing Pairs ---
                # 1. B vs R: Distinguish B from R using hole count and bottom baseline width
                if pred_label == "B" and char_crop is not None:
                    contours, hierarchy = cv2.findContours(char_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    holes = sum(1 for h in hierarchy[0] if h[3] != -1) if hierarchy is not None else 0
                    if holes <= 1 and "R" in classes_list:
                        bh_c, bw_c = char_crop.shape[:2]
                        bot_row = char_crop[-3:, :]
                        bot_row_w = np.sum(np.any(bot_row > 0, axis=0))
                        bot_ratio = bot_row_w / max(bw_c, 1)
                        # An 'R' only touches the bottom with a narrow diagonal leg tip (< 0.45)
                        # A true 'B' touches the bottom with a wide loop base (>= 0.50)
                        if bot_ratio < 0.45:
                            prob_r = float(probs[classes_list.index("R")])
                            pred_label = "R"
                            confidence = max(prob_r, 0.85)

                # 2. F vs E: If bottom 25% has horizontal width extending to the right, it is an E
                elif pred_label == "F" and char_crop is not None:
                    bh_c, bw_c = char_crop.shape[:2]
                    bot_strip = char_crop[int(bh_c * 0.75):, :]
                    bot_w = np.sum(np.any(bot_strip > 0, axis=0))
                    if "E" in classes_list:
                        prob_e = float(probs[classes_list.index("E")])
                        if bot_w >= 0.30 * bw_c and prob_e > 0.005:
                            pred_label = "E"
                            confidence = max(prob_e, 0.85)

                # 3. C vs O: If right side has pixels closing the loop, check if O
                elif pred_label == "C" and char_crop is not None:
                    bh_c, bw_c = char_crop.shape[:2]
                    right_strip = char_crop[:, int(bw_c * 0.75):]
                    right_h = np.sum(np.any(right_strip > 0, axis=1))
                    if "O" in classes_list:
                        prob_o = float(probs[classes_list.index("O")])
                        if right_h >= 0.45 * bh_c and prob_o > 0.05:
                            pred_label = "O"
                            confidence = max(prob_o, 0.85)

                # 4. Topological Guard for 'I':
                # Tall thin vertical stroke centered in the cell (not touching cell borders)
                if char_crop is not None and stroke_meta is not None:
                    s_x0 = stroke_meta["x0"]
                    s_x1 = stroke_meta["x1"]
                    s_bw = stroke_meta["bw"]
                    s_bh = stroke_meta["bh"]
                    s_cw = stroke_meta["cw"]
                    s_aspect = s_bh / max(s_bw, 1)
                    stroke_cx = (s_x0 + s_x1) / 2.0
                    cell_cx = s_cw / 2.0
                    is_centered = (abs(stroke_cx - cell_cx) <= 8.5 and s_x0 >= 7 and s_x1 <= s_cw - 7)
                    if is_centered and s_bh >= 25 and s_aspect >= 4.5:
                        pred_label = "I"
                        confidence = max(confidence, 0.95)
            else:
                pred_label = "-"
                confidence = 1.0
                
            sheet_data[question_num].append({
                "col": c,
                "character": pred_label,
                "confidence": round(confidence, 3)
            })
            
            if pred_label != "-":
                cv2.rectangle(annotated, (x1, y_top), (x2, y_bot), (0, 255, 0), 1)
                cv2.putText(annotated, pred_label, (x1 + 6, y_bot - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated, (x1, y_top), (x2, y_bot), (200, 200, 200), 1)
                
        chars_found = "".join([item["character"] for item in sheet_data[question_num]])
        print(f" ข้อ {question_num:02d}: {chars_found}")
        
    print("="*50)
    
    if save_annotated:
        out_warped_path = os.path.join(OUTPUT_VISUALS_DIR, f"{base_name}_warped.jpg")
        cv2.imwrite(out_warped_path, warped)
        
        out_img_path = os.path.join(OUTPUT_VISUALS_DIR, f"{base_name}_result.jpg")
        cv2.imwrite(out_img_path, annotated)
        print(f"Clean warped image saved to: {out_warped_path}")
        print(f"Annotated result image saved to: {out_img_path}")
        
    json_path = os.path.join(OUTPUT_RESULTS_DIR, f"{base_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sheet_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to JSON: {json_path}")
    
    return sheet_data

if __name__ == "__main__":
    default_img = os.path.join(BASE_DIR, "input_images", "1.jpg")
    scan_sheet(default_img)
