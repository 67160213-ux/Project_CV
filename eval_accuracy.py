import json
import os
import re
import glob
import sys

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

def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.exists(target):
            evaluate_file(target)
            return
            
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.json")), key=natural_sort_key)
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

if __name__ == "__main__":
    main()
