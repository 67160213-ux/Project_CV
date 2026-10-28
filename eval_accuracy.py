import json
import os
import re
import glob
import sys
import numpy as np
import openpyxl
from sklearn.metrics import precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def natural_sort_key(s):
    """เรียงลำดับสตริงตามตัวเลขแบบธรรมชาติ เช่น test1 -> test2 -> ... -> test10"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

# Ground Truth Template: ชุดที่ 1 (ประโยคภาษาอังกฤษ & Tech 20 แถว x 20 ช่อง)
GROUND_TRUTH_TECH = {
    1:  "THE-QUICK-BROWN-FOX-",
    2:  "JUMPS-OVER-LAZY-DOG-",
    3:  "PACK-MY-BOX-WITH-TEA",
    4:  "FIVE-DOZEN-LIQUOR-JU",
    5:  "SPHINX-OF-BLACK-GEM-",
    6:  "JUDGE-MY-VOW-QUICKLY",
    7:  "BRIGHT-ZEPHYRS-BLOW-",
    8:  "COMPUTER-VISION-TEST",
    9:  "MACHINE-LEARNING-AI-",
    10: "DEEP-NEURAL-NETWORKS",
    11: "IMAGE-DATA-DETECT-OK",
    12: "PYTHON-OPENCV-ROBOT-",
    13: "SMART-OCR-RECOGNIZE-",
    14: "ACCURATE-PREDICTION-",
    15: "ARTIFICIAL-SYSTEM-OK",
    16: "HELLO-WORLD-WELCOME-",
    17: "KNOWLEDGE-IS-POWER--",
    18: "FIND-EXTRA-FINE-GOLD",
    19: "ZERO-ONE-TWO-SIX-TEN",
    20: "ABCDEFGHIJKLMNOPQRST",
}

# Template มาตรฐานดั้งเดิม (Alphabet A-M / N-Z)
GROUND_TRUTH_ALPHABET = {
    q: ("ABCDEFGHIJKLM-------" if q % 2 != 0 else "NOPQRSTUVWXYZ-------")
    for q in range(1, 21)
}

def evaluate_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ตรวจจับ Template ที่ตรงกับข้อมูลมากที่สุด (Default: Tech Sentences ชุดที่ 1)
    tech_score = 0
    alpha_score = 0
    for q_str, cols in data.items():
        q_num = int(q_str)
        exp_t = GROUND_TRUTH_TECH.get(q_num, "-" * 20)
        exp_a = GROUND_TRUTH_ALPHABET.get(q_num, "-" * 20)
        for item in cols:
            c = item["col"]
            p = item["character"]
            if c < 20:
                if p == exp_t[c]:
                    tech_score += 1
                if p == exp_a[c]:
                    alpha_score += 1

    if alpha_score > tech_score:
        gt_dict = GROUND_TRUTH_ALPHABET
        template_name = "Alphabet (A-Z)"
    else:
        gt_dict = GROUND_TRUTH_TECH
        template_name = "English & Tech Sentences (ชุดที่ 1)"

    total = 0
    correct = 0
    wrong = 0
    errors = []

    for q_str, cols in data.items():
        q_num = int(q_str)
        expected_str = gt_dict.get(q_num, "-" * 20)
        
        for item in cols:
            col = item["col"]
            pred = item["character"]
            conf = item["confidence"]
            
            if col < len(expected_str):
                expected = expected_str[col]
                total += 1
                if pred == expected:
                    correct += 1
                else:
                    wrong += 1
                    errors.append({
                        "question": q_num,
                        "col": col,
                        "expected": expected,
                        "predicted": pred,
                        "confidence": conf
                    })

    accuracy = (correct / total) * 100 if total > 0 else 0

    print("="*65)
    print(f"ผลการตรวจสอบความถูกต้องสำหรับ: {os.path.basename(json_path)}")
    print(f"ชุดเฉลยที่ใช้ตรวจ: {template_name}")
    print("="*65)
    print(f"จำนวนตัวอักษรทั้งหมด: {total} ตัว")
    print(f"ทายถูกต้อง: {correct} ตัว ({accuracy:.2f}%)")
    print(f"ทายผิด: {wrong} ตัว ({(wrong/total)*100 if total > 0 else 0:.2f}%)")
    
    if errors:
        print("-"*65)
        print(f"รายการที่ทายผิด ({wrong} ตำแหน่ง):")
        print(f"{'ข้อ':<6} | {'คอลัมน์':<8} | {'ค่าจริง (Ground Truth)':<22} | {'ค่าที่ทายได้':<12} | {'Confidence'}")
        print("-" * 65)
        for err in errors:
            print(f"ข้อ {err['question']:02d} | Col {err['col']:02d}    | {err['expected']:<22} | {err['predicted']:<12} | {err['confidence']:.3f}")
    else:
        print(" ยอดเยี่ยม! ทำนายถูกต้อง 100% ครบทุกตัวอักษร")
    print("="*65 + "\n")
    return correct, total

def evaluate_excel_benchmark(excel_path):
    """คำนวณ Accuracy (CRA), Error Rate (CER), Macro Precision, Macro Recall, Macro F1-Score จากไฟล์ Excel"""
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb["Detailed_Results"]
    y_true = []
    y_pred = []
    sheet_stats = {}

    for row in list(ws.iter_rows(values_only=True))[1:]:
        img_name, q_num, ans_str = row[0], int(row[1]), str(row[2])
        if not img_name:
            continue
        exp_str = GROUND_TRUTH_TECH.get(q_num, "-" * 20)
        ans_padded = (ans_str + "-" * 20)[:20]
        if img_name not in sheet_stats:
            sheet_stats[img_name] = [0, 0]
        for p_c, t_c in zip(ans_padded, exp_str):
            y_true.append(t_c)
            y_pred.append(p_c)
            sheet_stats[img_name][1] += 1
            if p_c == t_c:
                sheet_stats[img_name][0] += 1

    labels = sorted(list(set(y_true)))
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    total = len(y_true_arr)
    correct = int(np.sum(y_true_arr == y_pred_arr))
    cra = (correct / total) * 100.0 if total > 0 else 0.0
    cer = 100.0 - cra
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    return {
        "correct": correct,
        "total": total,
        "cra": cra,
        "cer": cer,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "sheets": sheet_stats,
    }


def print_paper_benchmark_table():
    """แสดงตารางที่ 1: ตารางวัดผลโมเดลหลัก อ้างอิงตามเปเปอร์วิจัย (สำหรับแคปหน้าจอ Terminal)"""
    models_cfg = [
        (
            "1. Dual-Ensemble MLP (HOG + Pixel)",
            os.path.join(RESULTS_DIR, "all_results_ensemble.xlsx"),
        ),
        (
            "2. Custom 2D CNN (Lin 2017 + Zhong 2020)",
            os.path.join(RESULTS_DIR, "all_results_cnn.xlsx"),
        ),
        (
            "3. Super-Ensemble (MLP + CNN)",
            os.path.join(RESULTS_DIR, "all_results.xlsx"),
        ),
    ]

    results = []
    for name, path in models_cfg:
        if os.path.exists(path):
            results.append((name, evaluate_excel_benchmark(path)))

    if not results:
        return

    # ตารางย่อย: เปรียบเทียบความแม่นยำรายแผ่น (12 แผ่นทดสอบ)
    all_sheets = sorted(results[-1][1]["sheets"].keys(), key=natural_sort_key)
    print("\n" + "=" * 108)
    print(" PER-SHEET ACCURACY BREAKDOWN : 27 Classes: A-Z & Blank)")
    print("=" * 108)
    print(f" {'Test Sheet':<14} | {'Dual-Ensemble MLP':^26} | {'Custom 2D CNN':^26} | {'Super-Ensemble':^28}")
    print("-" * 108)
    for sh in all_sheets:
        cols_str = []
        for _, r in results:
            c, t = r["sheets"].get(sh, [0, 400])
            acc = (c / t) * 100.0 if t > 0 else 0.0
            cols_str.append(f"{c:>3}/{t:<3} ({acc:>6.2f}%)")
        while len(cols_str) < 3:
            cols_str.append("N/A")
        print(f" {sh:<14} | {cols_str[0]:^26} | {cols_str[1]:^26} | {cols_str[2]:^28}")
    print("=" * 108)

    # ตารางที่ 1: ตารางวัดผลโมเดลหลัก อ้างอิงตามเปเปอร์วิจัย
    print("\n" + "=" * 118)
    print(" MAIN MODEL EVALUATION BENCHMARK")
    print(" Referenced Papers:")
    print("   - Accuracy (CRA) & Error Rate (CER) : Cohen et al. (IEEE IJCNN 2017) - EMNIST Benchmark")
    print("   - Precision, Recall & F1-Score      : Oqaibi et al. (IEEE ICECCME 2021) - Multi-Class HCR Evaluation")
    print("   - 2D CNN Architecture & Training    : Lin et al. (IEEE ICCV 2017, Focal Loss) & Zhong et al. (AAAI 2020, Random Erasing)")
    print("=" * 118)
    total_cells_label = f"Correct ({results[-1][1]['total']:,})"
    header = (
        f" {'Model Architecture':<40} | {total_cells_label:^15} | "
        f"{'Accuracy':^11} | {'Error Rate':^12} | {'Precision':^11} | {'Recall':^9} | {'F1-Score':^10}"
    )
    sub_header = (
        f" {'(Evaluated on 27 Classes: A-Z & -)':<40} | {'Cells':^15} | "
        f"{'(Cohen 17)':^11} | {'(Cohen 17)':^12} | {'(Oqaibi 21)':^11} | {'(Oqaibi 21)':^9} | {'(Oqaibi 21)':^10}"
    )
    print(header)
    print(sub_header)
    print("-" * 118)

    for name, r in results:
        corr_str = f"{r['correct']:,} / {r['total']:,}"
        print(
            f" {name:<40} | {corr_str:^15} | "
            f"{r['cra']:>9.2f}%  | {r['cer']:>10.2f}%  | {r['precision']:>9.2f}%  | {r['recall']:>7.2f}%  | {r['f1']:>8.2f}%"
        )

    print("=" * 118 + "\n")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in ("--table", "--benchmark", "-t"):
            print_paper_benchmark_table()
            return
        if os.path.exists(target):
            evaluate_file(target)
            return

    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "test*_results.json")), key=natural_sort_key)
    if not json_files:
        raise FileNotFoundError(f"No result json found in {RESULTS_DIR}. Run batch_processor.py first.")

    grand_correct = 0
    grand_total = 0
    for jf in json_files:
        c, t = evaluate_file(jf)
        grand_correct += c
        grand_total += t

    if len(json_files) > 1 and grand_total > 0:
        overall_acc = (grand_correct / grand_total) * 100
        print("*****************************************************************")
        print(f" สรุปภาพรวมทุกแผ่น ({len(json_files)} แผ่น): {grand_correct}/{grand_total} ตัว ({overall_acc:.2f}%)")
        print("*****************************************************************")
        print_paper_benchmark_table()

if __name__ == "__main__":
    main()
