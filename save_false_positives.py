import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import existing core functions
from scan_and_predict import (
    BASE_DIR,
    load_classifier,
    preprocess_and_warp,
    detect_table_rows,
    detect_table_columns,
    align_centroid,
    compute_hog_28
)
from eval_accuracy import GROUND_TRUTH_TECH, GROUND_TRUTH_ALPHABET

# Default folder to save false-positive clear square images
DEFAULT_OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "false_positives_clear_squares")

def extract_cell_details(cell_bgr):
    """
    Extracts features for prediction, but also returns intermediate
    visualizations (binary mask, 28x28 normalized image, bounding box) for debugging.
    """
    ch, cw = cell_bgr.shape[:2]
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    
    pixel_diff = int(np.max(gray)) - int(np.min(gray))
    pixel_std = float(np.std(gray))
    if pixel_std < 7.0 or pixel_diff < 35:
        return None, np.zeros((ch, cw), dtype=np.uint8), None, None
        
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Gentle morphological close (2x2)
    k_heal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k_heal)
    
    # Border clearing (2 px)
    bin_inv[:2, :] = 0
    bin_inv[-2:, :] = 0
    bin_inv[:, :2] = 0
    bin_inv[:, -2:] = 0
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_inv)
    
    candidates = []
    for lbl in range(1, num_labels):
        bx, by, bw, bh, area = stats[lbl]
        
        is_vertical_border = (
            bh >= ch - 6 and (by <= 1 or by + bh >= ch - 1) and
            (bh / max(bw, 1) >= 3.2) and (bx <= 3 or bx + bw >= cw - 3)
        )
        is_horizontal_border = (
            bw >= cw - 6 and (by <= 1 or by + bh >= ch - 1) and
            (bw / max(bh, 1) >= 3.2)
        )
        
        if is_vertical_border or is_horizontal_border:
            continue
            
        if area > 15 and bh > 6:
            candidates.append((lbl, bx, by, bw, bh, area))
            
    if not candidates:
        return None, bin_inv, None, None
        
    candidates.sort(key=lambda x: x[5], reverse=True)
    p_lbl, p_x, p_y, p_w, p_h, p_area = candidates[0]
    
    merged_lbls = [p_lbl]
    for (lbl, bx, by, bw, bh, area) in candidates[1:]:
        dist_x = max(0, bx - (p_x + p_w), p_x - (bx + bw))
        dist_y = max(0, by - (p_y + p_h), p_y - (by + bh))
        if dist_x <= 16 and dist_y <= 10 and (area >= 0.12 * p_area or bh >= 0.35 * p_h):
            merged_lbls.append(lbl)
            
    char_mask = np.isin(labels, merged_lbls).astype(np.uint8) * 255
    pts = np.argwhere(char_mask > 0)
    if len(pts) == 0:
        return None, bin_inv, None, None
        
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
    char_28 = align_centroid(char_28)
    
    pix_feat = char_28.flatten() / 255.0
    hog_feat = compute_hog_28(char_28)
    feat = np.hstack([pix_feat, hog_feat]).astype(np.float32)

    bbox_info = {
        "x": int(x0),
        "y": int(y0),
        "w": int(bw),
        "h": int(bh),
        "area": int(np.sum(char_mask > 0))
    }
    return feat, bin_inv, char_28, bbox_info

def create_diagnostic_card(cell_bgr, bin_mask, char_28, row_num, col_num, pred_char, conf, img_name):
    """
    Creates an informative visual diagnostic card combining:
    1. Raw cell crop from paper
    2. Thresholded binary mask
    3. 28x28 input fed to Neural Network
    4. Metadata banner
    """
    card_h = 160
    card_w = 460
    card = np.full((card_h, card_w, 3), 245, dtype=np.uint8)
    
    # 1. Header Banner
    cv2.rectangle(card, (0, 0), (card_w, 36), (35, 39, 42), -1)
    title_text = f"FALSE POSITIVE: {img_name} [R{row_num:02d}, C{col_num:02d}]"
    cv2.putText(card, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 2. Raw Cell Crop
    resized_raw = cv2.resize(cell_bgr, (80, 80))
    card[48:128, 15:95] = resized_raw
    cv2.rectangle(card, (15, 48), (95, 128), (0, 0, 255), 1)
    cv2.putText(card, "Raw Cell", (22, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 50), 1)
    
    # 3. Binary Mask
    bin_bgr = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)
    resized_bin = cv2.resize(bin_bgr, (80, 80))
    card[48:128, 115:195] = resized_bin
    cv2.rectangle(card, (115, 48), (195, 128), (100, 100, 100), 1)
    cv2.putText(card, "Otsu Mask", (122, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 50), 1)
    
    # 4. 28x28 Input (Enlarged)
    if char_28 is not None:
        c28_bgr = cv2.cvtColor(char_28, cv2.COLOR_GRAY2BGR)
        resized_c28 = cv2.resize(c28_bgr, (80, 80), interpolation=cv2.INTER_NEAREST)
        card[48:128, 215:295] = resized_c28
        cv2.rectangle(card, (215, 48), (295, 128), (100, 100, 100), 1)
        cv2.putText(card, "28x28 Input", (222, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 50), 1)
        
    # 5. Prediction Text Details
    cv2.putText(card, "Ground Truth: '-' (Clear)", (315, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 130, 0), 1)
    cv2.putText(card, f"Model Pred: '{pred_char}'", (315, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 220), 2)
    cv2.putText(card, f"Confidence: {conf*100:.1f}%", (315, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    
    return card

def find_matching_ground_truth(test_pred_matrix):
    """Determines whether Tech Sentences or Alphabet matches the sheet best"""
    score_tech = 0
    score_alpha = 0
    for r in range(1, 21):
        if r in test_pred_matrix:
            row_str = test_pred_matrix[r]
            gt_t = GROUND_TRUTH_TECH.get(r, "-" * 20)
            gt_a = GROUND_TRUTH_ALPHABET.get(r, "-" * 20)
            for c in range(min(20, len(row_str))):
                if row_str[c] == gt_t[c]:
                    score_tech += 1
                if row_str[c] == gt_a[c]:
                    score_alpha += 1
    return GROUND_TRUTH_TECH if score_tech >= score_alpha else GROUND_TRUTH_ALPHABET

def process_image_for_false_positives(image_path, clf, output_folder, custom_gt=None):
    """
    Scans a single image, compares predicted characters with Ground Truth,
    and extracts cell images where Ground Truth is '-' (clear square)
    but Model predicted a character ('A'-'Z').
    """
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    warped = preprocess_and_warp(image_path)
    row_boxes = detect_table_rows(warped)
    col_boxes = detect_table_columns(warped)
    
    # Directory setup
    cells_dir = os.path.join(output_folder, "cells")
    diags_dir = os.path.join(output_folder, "diagnostics")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(diags_dir, exist_ok=True)
    
    # Quick pass to get whole sheet prediction for template matching
    predictions = {}
    cell_cache = {}
    
    for r, (y_top, y_bot) in enumerate(row_boxes):
        row_num = r + 1
        predictions[row_num] = []
        for c, (x1, x2) in enumerate(col_boxes):
            pad_y = 4
            pad_x = 5
            
            cell_roi = warped[y_top + pad_y : y_bot - pad_y, x1 + pad_x : x2 - pad_x]
            feat, bin_mask, char_28, bbox_info = extract_cell_details(cell_roi)
            
            if feat is not None:
                pred = clf.predict([feat])[0]
                probs = clf.predict_proba([feat])[0]
                conf = float(np.max(probs))
            else:
                pred = "-"
                conf = 1.0
                
            predictions[row_num].append(pred)
            cell_cache[(row_num, c)] = {
                "cell_roi": cell_roi,
                "bin_mask": bin_mask,
                "char_28": char_28,
                "pred": pred,
                "conf": conf,
                "bbox": bbox_info
            }
            
    # Determine Ground Truth Template
    pred_str_map = {r: "".join(predictions[r]) for r in predictions}
    gt = custom_gt if custom_gt is not None else find_matching_ground_truth(pred_str_map)
    
    false_positives = []
    
    # Identify False Positives: GT == '-' AND Pred != '-'
    for r in range(1, 21):
        expected_row = gt.get(r, "-" * 20)
        for c in range(20):
            expected_char = expected_row[c]
            cached = cell_cache[(r, c)]
            pred_char = cached["pred"]
            conf = cached["conf"]
            
            # CORE CONDITION: Ground Truth is clear square ('-') but model found character ('A'-'Z')
            if expected_char == "-" and pred_char != "-":
                cell_roi = cached["cell_roi"]
                bin_mask = cached["bin_mask"]
                char_28 = cached["char_28"]
                
                # 1. Save raw cropped cell
                cell_filename = f"{img_name}_R{r:02d}_C{c:02d}_GT_clear_PRED_{pred_char}.png"
                cell_filepath = os.path.join(cells_dir, cell_filename)
                cv2.imwrite(cell_filepath, cell_roi)
                
                # 2. Save diagnostic card
                diag_card = create_diagnostic_card(
                    cell_roi, bin_mask, char_28,
                    row_num=r, col_num=c,
                    pred_char=pred_char, conf=conf,
                    img_name=img_name
                )
                diag_filename = f"diag_{img_name}_R{r:02d}_C{c:02d}_PRED_{pred_char}.png"
                diag_filepath = os.path.join(diags_dir, diag_filename)
                cv2.imwrite(diag_filepath, diag_card)
                
                false_positives.append({
                    "image": img_name,
                    "row": r,
                    "col": c,
                    "ground_truth": "-",
                    "predicted": pred_char,
                    "confidence": round(conf, 4),
                    "cell_image": os.path.relpath(cell_filepath, output_folder),
                    "diag_image": os.path.relpath(diag_filepath, output_folder),
                    "bbox": cached["bbox"]
                })
                
    return false_positives

def main():
    parser = argparse.ArgumentParser(
        description="Save cropped cell images where Ground Truth is Clear Square (-) but Model found a character (A-Z)"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(BASE_DIR, "test_image"),
        help="Input image file or folder containing images (default: test_image/)"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Target directory to save images (default: {DEFAULT_OUTPUT_FOLDER})"
    )
    args = parser.parse_args()
    
    print("\n" + "="*65)
    print(" 🔍 FALSE POSITIVE CELL EXTRACTOR (CLEAR SQUARE ERRORS) ")
    print("="*65)
    print(f"Target Output Folder: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    
    clf = load_classifier()
    
    # Collect images
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = sorted(
            glob.glob(os.path.join(args.input, "*.jpg")) +
            glob.glob(os.path.join(args.input, "*.jpeg")) +
            glob.glob(os.path.join(args.input, "*.png"))
        )
        
    if not image_paths:
        print(f"[!] No image files found in: {args.input}")
        return
        
    print(f"Found {len(image_paths)} image(s) to inspect.\n")
    
    all_fp_records = []
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)} ...")
        fp_list = process_image_for_false_positives(img_path, clf, args.output)
        all_fp_records.extend(fp_list)
        print(f"   -> Found {len(fp_list)} false positive cell(s) saved.")
        
    # Save Summary JSON
    summary_json_path = os.path.join(args.output, "false_positives_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_fp_records, f, indent=2, ensure_ascii=False)
        
    # Save Summary CSV
    summary_csv_path = os.path.join(args.output, "false_positives_summary.csv")
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write("Image,Row,Col,GroundTruth,Predicted,Confidence,CellFile,DiagFile\n")
        for rec in all_fp_records:
            f.write(f"{rec['image']},{rec['row']},{rec['col']},{rec['ground_truth']},{rec['predicted']},{rec['confidence']},{rec['cell_image']},{rec['diag_image']}\n")
            
    print("\n" + "="*65)
    print(" 🏁 SUMMARY OF EXTRACTED FALSE POSITIVES ")
    print("="*65)
    print(f"Total False Positive Cells Saved: {len(all_fp_records)}")
    print(f"📁 Raw Cropped Cells:   {os.path.join(args.output, 'cells')}")
    print(f"🖼️ Diagnostic Cards:   {os.path.join(args.output, 'diagnostics')}")
    print(f"📑 Summary CSV Report:  {summary_csv_path}")
    print(f"📋 Summary JSON Report: {summary_json_path}")
    print("="*65)
    
    if all_fp_records:
        print("\nBreakdown of false characters predicted on clear squares:")
        char_counts = {}
        for r in all_fp_records:
            char_counts[r["predicted"]] = char_counts.get(r["predicted"], 0) + 1
        for ch, count in sorted(char_counts.items(), key=lambda x: -x[1]):
            print(f"  • '{ch}': {count} occurrences")
        print("="*65 + "\n")

if __name__ == "__main__":
    main()
