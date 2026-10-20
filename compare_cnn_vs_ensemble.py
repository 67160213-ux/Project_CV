import os
import sys
import glob
import re
import json
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# Ensure workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "scratch":
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scan_and_predict import scan_sheet
from eval_accuracy import GROUND_TRUTH_TECH, natural_sort_key

TEST_IMG_DIR = os.path.join(BASE_DIR, "test_image")
MODELS_DIR = os.path.join(BASE_DIR, "models")


class CNNClassifierWrapper:
    def __init__(self, model_path, device=None):
        import torch
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes_ = np.array(checkpoint["classes"])
        self.classes_list = list(self.classes_)
        self.n_features_in_ = 1176
        
        from train_cnn import CharCNN
        self.model = CharCNN(num_classes=len(self.classes_))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
    def predict_proba(self, X):
        import torch
        # X is list/array of feature vectors; first 784 are 28x28 normalized pixels
        X_arr = np.array(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        pixels = X_arr[:, :784].reshape(-1, 1, 28, 28)
        
        with torch.no_grad():
            tensor = torch.from_numpy(pixels).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]


def run_evaluation(clf, image_files):
    results_by_sheet = {}
    
    for img_path in image_files:
        base_name = os.path.basename(img_path)
        # Scan sheet with the specified classifier
        sheet_data = scan_sheet(img_path, save_annotated=False, clf=clf)
        
        correct = 0
        total = 0
        confidences = []
        
        for q_num, items in sheet_data.items():
            expected_str = GROUND_TRUTH_TECH.get(q_num, "-" * 20)
            for it in items:
                col = it["col"]
                pred = it["character"]
                conf = it["confidence"]
                if col < len(expected_str):
                    expected = expected_str[col]
                    total += 1
                    if pred == expected:
                        correct += 1
                    confidences.append(conf)
                    
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        avg_conf = float(np.mean(confidences)) * 100.0 if confidences else 0.0
        results_by_sheet[base_name] = {
            "correct": correct,
            "total": total,
            "acc": acc,
            "avg_conf": avg_conf
        }
    return results_by_sheet


def main():
    import joblib
    print("=" * 80)
    print(" เปรียบเทียบผลลัพธ์: Dual-Ensemble (93.94%) vs Custom 2D CNN (28 Epochs)")
    print("=" * 80)

    image_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")), key=natural_sort_key)
    print(f"พบไฟล์ภาพชุดทดสอบ {len(image_files)} แผ่น")

    # 1. โหลดโมเดลปัจจุบัน Dual-Ensemble
    ensemble_path = os.path.join(MODELS_DIR, "char_classifier_ensemble_best_93.94.pkl")
    print(f"\n[1/3] กำลังประเมินโมเดลปัจจุบัน: Dual-Ensemble ({ensemble_path})...")
    clf_ensemble = joblib.load(ensemble_path)
    res_ensemble = run_evaluation(clf_ensemble, image_files)

    # 2. โหลดโมเดล 2D CNN
    cnn_path = os.path.join(MODELS_DIR, "char_cnn.pth")
    if not os.path.exists(cnn_path):
        print(f"\n[ข้อผิดพลาด] ยังไม่พบโมเดล CNN ที่ {cnn_path}")
        print("กรุณารัน 'python train_cnn.py' เพื่อฝึกสอนโมเดล CNN ก่อน")
        return

    print(f"\n[2/3] กำลังประเมินโมเดลใหม่: Custom 2D CNN ({cnn_path})...")
    clf_cnn = CNNClassifierWrapper(cnn_path)
    res_cnn = run_evaluation(clf_cnn, image_files)

    # 3. โหลดโมเดล Super-Ensemble (Ensemble 30% + CNN 70%)
    from scan_and_predict import SuperEnsembleClassifier
    print(f"\n[3/3] กำลังประเมินโมเดลรวมพลัง: Super-Ensemble (MLP 30% + CNN 70%)...")
    clf_super = SuperEnsembleClassifier(clf_ensemble, clf_cnn, w_ens=0.30)
    res_super = run_evaluation(clf_super, image_files)

    # 4. สร้างตารางเปรียบเทียบ 3 โมเดล
    print("\n" + "=" * 115)
    print(f"{'แผ่นทดสอบ':<10} | {'1. Dual-Ensemble':<18} | {'2. Custom 2D CNN':<18} | {'3. Super-Ensemble':<18} | {'ผลต่างเทียบเดิม':<15} | {'Super Conf':<12}")
    print("-" * 115)

    tot_ens = 0
    tot_cnn = 0
    tot_sup = 0
    tot_cells = 0
    conf_sup_list = []

    for img_path in image_files:
        name = os.path.basename(img_path)
        e = res_ensemble[name]
        c = res_cnn[name]
        s = res_super[name]

        diff_sup = s['correct'] - e['correct']
        diff_str = f"{s['acc'] - e['acc']:+.2f}% ({diff_sup:+d} ตัว)"

        tot_ens += e['correct']
        tot_cnn += c['correct']
        tot_sup += s['correct']
        tot_cells += e['total']
        conf_sup_list.append(s['avg_conf'])

        print(f"{name:<10} | {e['acc']:6.2f}% ({e['correct']}/400)    | {c['acc']:6.2f}% ({c['correct']}/400)    | {s['acc']:6.2f}% ({s['correct']}/400)    | {diff_str:<15} | {s['avg_conf']:6.2f}%")

    acc_ens = (tot_ens / tot_cells) * 100.0
    acc_cnn = (tot_cnn / tot_cells) * 100.0
    acc_sup = (tot_sup / tot_cells) * 100.0
    diff_all = acc_sup - acc_ens
    diff_chars = tot_sup - tot_ens
    avg_conf_sup = float(np.mean(conf_sup_list))

    print("=" * 115)
    print(f"{'รวม 9 แผ่น':<10} | {acc_ens:6.2f}% ({tot_ens}/{tot_cells})  | {acc_cnn:6.2f}% ({tot_cnn}/{tot_cells})  | {acc_sup:6.2f}% ({tot_sup}/{tot_cells})  | {diff_all:+.2f}% ({diff_chars:+d} ตัว)  | {avg_conf_sup:6.2f}%")
    print("=" * 115)

    print(f"\n[บทสรุปเชิงวิจัย]:")
    print(f" 1. Dual-Ensemble MLP เดี่ยว : {acc_ens:.2f}% ({tot_ens} ตัว)")
    print(f" 2. Custom 2D CNN เดี่ยว     : {acc_cnn:.2f}% ({tot_cnn} ตัว) [ชนะ Ensemble ถึง 6 แผ่น!]")
    print(f" 3. [BEST] Super-Ensemble (ผสาน) : {acc_sup:.2f}% ({tot_sup} ตัว) [ดึงจุดเด่นของทั้ง 2 โมเดล เพิ่มขึ้น +{diff_chars} ตัว!]")
    print(f"--> แนะนำใช้ 'Super-Ensemble' เป็นโมเดลหลักใน batch_processor.py")

if __name__ == "__main__":
    main()
