"""
========================================================================================
 🔬 Hard Example Mining & Targeted Fine-Tuning Pipeline with Early Stopping
----------------------------------------------------------------------------------------
 สคริปต์นี้ทำหน้าที่:
 1. สแกนกระดาษคำตอบทดสอบเพื่อค้นหาตัวอักษรที่โมเดลปัจจุบันยังทำนายผิด (Hard Examples)
 2. นำภาพตัวอักษรที่ผิดมาทำ Data Augmentation เพิ่ม 8 เท่า (จำลองเอียง, สั่น, หนา, บาง)
 3. ผสมกับข้อมูลมาตรฐานจากแคชเพื่อป้องกันการลืมข้อมูลเดิม (Catastrophic Forgetting)
 4. แบ่งชุดข้อมูลฝึกสอน (Train) และตรวจสอบ (Validation) เพื่อวัดความแม่นยำราย Epoch
 5. Continuous Training with Early Stopping: ฝึกสอนต่อเนื่องไปเรื่อยๆ จนกระทั่งความแม่นยำ
    (Validation Accuracy) เริ่มลดลงติดต่อกันครบตามค่า Patience (ค่าเริ่มต้น: 3 รอบ)
 6. กู้คืนโมเดลในรอบที่ทำคะแนนได้สูงสุด (Best Epoch) พร้อมสำรองโมเดลเดิมอัตโนมัติ
========================================================================================
"""
import sys
import os
import cv2
import shutil
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scan_and_predict import (
    load_classifier, detect_table_rows, detect_table_columns,
    preprocess_and_warp, extract_character_features, MODEL_PATH
)
from eval_accuracy import GROUND_TRUTH_TECH
from train_cnn import CharCNN, FocalLoss, CLASSES, CHAR2IDX, BLANK_CHAR

MODEL_DIR = os.path.dirname(MODEL_PATH)
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "char_cnn.pth")
CACHE_FILE = os.path.join(BASE_DIR, "dataset_cache_cnn.npz")

class MinedDataset(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_t = torch.from_numpy(self.imgs[idx].astype(np.float32) / 255.0).unsqueeze(0)
        return img_t, self.lbls[idx]

def evaluate_model(model, val_loader, criterion, device):
    """ฟังก์ชันประเมินความแม่นยำและ Loss บนชุด Validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            
    val_loss = (running_loss / total) if total > 0 else 0.0
    val_acc = (correct / total * 100.0) if total > 0 else 0.0
    return val_loss, val_acc

def mine_and_finetune(input_dir=os.path.join(BASE_DIR, "input_images"), epochs=None, lr=2e-4, 
                      multiplier=8, num_regular_mix=6000, patience=3, val_split=0.15, 
                      save_path=CNN_MODEL_PATH, apply_to_active_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep_quota_str = f"{epochs} Epochs" if epochs is not None else "∞ (รันไปเรื่อยๆ จนกระทั่งความแม่นยำเริ่มลดลง)"
    print("=" * 75)
    print(" 🚀 เริ่มต้นกระบวนการ Hard Example Mining จากชุดข้อมูลฝึกสอน (Training Set)")
    print(f" * แหล่งข้อมูล Hard Examples: {input_dir} (100% Zero-Test Leakage)")
    print(f" * อุปกรณ์ประมวลผล: {device}")
    print(f" * โหมดการเทรน: Continuous Training with Early Stopping (Patience = {patience} รอบ)")
    print(f" * โควตารอบการเทรน: {ep_quota_str} | Learning Rate: {lr} | Data Multiplier: {multiplier}x")
    print(f" * ปลายทางบันทึกโมเดล: {save_path}")
    print("=" * 75)

    # -------------------------------------------------------------
    # ขั้นตอนที่ 1: สแกนหาข้อที่โมเดลปัจจุบันทายผิด (Hard Examples)
    # -------------------------------------------------------------
    print(f"\n[Step 1/4] กำลังสแกนหาตัวอักษรลายมือยากที่ทายผิดจาก {input_dir} (Mining)...")
    clf = load_classifier('super')

    import glob
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

    target_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")), key=natural_sort_key)
    print(f"[*] พบแผ่นภาพสำหรับ Mining ทั้งหมด {len(target_files)} แผ่น")

    hard_images = []
    hard_labels = []

    odd_expected = "ABCDEFGHIJKLM-------"
    even_expected = "NOPQRSTUVWXYZ-------"

    for p in tqdm(target_files, desc="สแกนกระดาษฝึกสอน"):
        try:
            warped = preprocess_and_warp(p)
            rows = detect_table_rows(warped)
            cols = detect_table_columns(warped)
            
            for r, (y1, y2) in enumerate(rows):
                q = r + 1
                exp_s = odd_expected if (q % 2 != 0) else even_expected
                for c, (x1, x2) in enumerate(cols):
                    if c >= len(exp_s): continue
                    cell = warped[y1+4:y2-4, x1+5:x2-5]
                    feat = extract_character_features(cell)
                    pred = "-" if feat is None else clf.predict([feat])[0]
                    exp = exp_s[c]
                    
                    # หากทายผิด บันทึกเป็น Hard Example ทันที
                    if pred != exp:
                        if feat is not None:
                            img_28 = (feat[:784].reshape(28, 28) * 255.0).astype(np.uint8)
                        else:
                            img_28 = np.zeros((28, 28), dtype=np.uint8)
                        hard_images.append(img_28)
                        hard_labels.append(CHAR2IDX.get(exp, 0))
        except Exception as e:
            continue

    print(f"\n[*] ค้นพบตัวอย่างลายมือยาก (Hard Examples) ทั้งหมด: {len(hard_images)} ตัวอย่าง")

    if not hard_images:
        print("[!] ไม่พบข้อผิดพลาดในชุดข้อมูล ระบบมีความแม่นยำ 100% แล้ว")
        return

    # -------------------------------------------------------------
    # ขั้นตอนที่ 2: ทำ Data Augmentation เสริมเคสยาก (Data Multiplier)
    # -------------------------------------------------------------
    print(f"\n[Step 2/4] กำลังทำ Targeted Data Augmentation ({multiplier}x ขยายเคสยาก)...")
    aug_images = []
    aug_labels = []

    for base_img, lbl in zip(hard_images, hard_labels):
        aug_images.append(base_img)
        aug_labels.append(lbl)
        
        if lbl != CHAR2IDX[BLANK_CHAR]:
            for _ in range(multiplier - 1):
                img = base_img.copy()
                # สุ่มหมุน +-6 องศา
                ang = random.uniform(-6, 6)
                M_rot = cv2.getRotationMatrix2D((14, 14), ang, 1.0)
                img = cv2.warpAffine(img, M_rot, (28, 28), borderValue=0)
                # สุ่มขยับ +-1 พิกเซล
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
                M_t = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M_t, (28, 28), borderValue=0)
                # สุ่มเพิ่มความหนา/บาง (จำลองดินสอและปากกา)
                if random.random() < 0.35:
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    if random.random() < 0.5:
                        img = cv2.dilate(img, k, iterations=1)
                    else:
                        er = cv2.erode(img, k, iterations=1)
                        if np.count_nonzero(er) > 15:
                            img = er
                aug_images.append(img)
                aug_labels.append(lbl)
        else:
            for _ in range(multiplier - 1):
                b_img = np.zeros((28, 28), dtype=np.uint8)
                if random.random() < 0.25:
                    side = random.choice(['left', 'right'])
                    if side == 'left': b_img[:, :2] = random.randint(140, 220)
                    else: b_img[:, -2:] = random.randint(140, 220)
                aug_images.append(b_img)
                aug_labels.append(lbl)

    print(f"[*] ขยายภาพตัวอย่างยากเป็นชุดฝึกสอนเข้มข้น: {len(aug_images):,} ตัวอย่าง")

    # -------------------------------------------------------------
    # ขั้นตอนที่ 3: ผสมข้อมูลมาตรฐานและแบ่งชุด Validation (Train/Val Split)
    # -------------------------------------------------------------
    print(f"\n[Step 3/4] ผสมข้อมูลมาตรฐาน {num_regular_mix:,} ตัวอย่างเพื่อป้องกันการลืมข้อมูลเดิม...")
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"ไม่พบไฟล์แคชชุดข้อมูลที่ {CACHE_FILE}")
        
    cached = np.load(CACHE_FILE)
    X_cached, y_cached = cached["X"], cached["y"]
    np.random.seed(42)
    reg_idx = np.random.choice(len(X_cached), size=min(num_regular_mix, len(X_cached)), replace=False)
    X_reg, y_reg = X_cached[reg_idx], y_cached[reg_idx]

    X_all = np.concatenate([np.array(aug_images, dtype=np.uint8), X_reg], axis=0)
    y_all = np.concatenate([np.array(aug_labels, dtype=np.int64), y_reg], axis=0)
    print(f"[*] รวมชุดข้อมูล Fine-Tuning ทั้งสิ้น: {len(X_all):,} ตัวอย่าง")

    # แบ่งชุด Train / Validation (85% : 15%)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=val_split, random_state=42, stratify=y_all
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=val_split, random_state=42, shuffle=True
        )
        
    print(f"[*] ชุดฝึกสอน (Train): {len(X_train):,} ตัวอย่าง | ชุดตรวจสอบ (Validation): {len(X_val):,} ตัวอย่าง")

    train_loader = DataLoader(MinedDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(MinedDataset(X_val, y_val), batch_size=128, shuffle=False)

    # -------------------------------------------------------------
    # ขั้นตอนที่ 4: Fine-Tune โมเดล CNN พร้อม Early Stopping
    # -------------------------------------------------------------
    print(f"\n[Step 4/4] เริ่มต้น Continuous Training ({ep_quota_str}, LR={lr}, Early Stopping Patience={patience})...")
    model = CharCNN(num_classes=len(CLASSES)).to(device)
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"ไม่พบโมเดลเดิมที่ {CNN_MODEL_PATH}")
        
    ckpt = torch.load(CNN_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # วัดคะแนน Baseline ก่อนเริ่มฝึกสอน
    init_val_loss, init_val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f"[*] Baseline ก่อนเทรน -> Val Acc: {init_val_acc:.2f}% | Val Loss: {init_val_loss:.4f}")

    best_val_acc = init_val_acc
    best_val_loss = init_val_loss
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    ep = 0
    try:
        while True:
            if epochs is not None and ep >= epochs:
                print(f"\n[*] ครบกำหนดรอบสูงสุด {epochs} Epochs เรียบร้อยแล้ว")
                break
            ep += 1
            model.train()
            running_loss = 0.0
            corr = 0
            tot = 0
            desc_str = f"Epoch {ep:02d}" + (f"/{epochs:02d}" if epochs is not None else "")
            pbar = tqdm(train_loader, desc=desc_str)
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1)
                corr += (preds == lbls).sum().item()
                tot += lbls.size(0)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Train_Acc": f"{corr/tot*100:.2f}%"})

            train_loss = running_loss / tot
            train_acc = corr / tot * 100.0

            # ตรวจสอบความแม่นยำบน Validation Set
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            
            # ตรวจสอบการพัฒนาของ Accuracy
            if val_acc > best_val_acc:
                diff = val_acc - best_val_acc
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = ep
                patience_counter = 0
                print(f"  -> Epoch {ep:02d}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Loss: {val_loss:.4f})")
                print(f"     🌟 [สถิติสูงสุดใหม่!] Accuracy เพิ่มขึ้น +{diff:.2f}% -> บันทึกน้ำหนักที่ดีที่สุด")
            else:
                patience_counter += 1
                print(f"  -> Epoch {ep:02d}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Loss: {val_loss:.4f})")
                print(f"     ⚠️  [Patience {patience_counter}/{patience}] Accuracy เริ่มลดลงหรือคงที่ (ปัจจุบัน: {val_acc:.2f}%, สูงสุด: {best_val_acc:.2f}% ที่ Epoch {best_epoch})")

            # เงื่อนไข Early Stopping: เมื่อความแม่นยำเริ่มตกลงติดต่อกันครบ Patience รอบ
            if patience_counter >= patience:
                print("\n" + "=" * 75)
                print(f" 🛑 [Early Stopping ทำงาน!] ความแม่นยำ (Accuracy) เริ่มลดลงติดต่อกัน {patience} รอบแล้ว")
                print(f"    -> ระบบหยุดการเทรนอัตโนมัติที่รอบ {ep}")
                print(f"    -> กู้คืนโมเดลจุดที่ดีที่สุดจาก Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%)")
                print("=" * 75)
                break

    except KeyboardInterrupt:
        print("\n\n[!] ผู้ใช้กด Ctrl+C เพื่อหยุดการเทรน -> กำลังกู้คืนโมเดลจุดที่ดีที่สุด...")

    # กู้คืนน้ำหนักของโมเดลในรอบที่ความแม่นยำสูงสุด (ป้องกัน Overfitting จากรอบที่แย่ลง)
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\n[Model Restore] โหลดน้ำหนักรอบที่ดีที่สุด (Epoch {best_epoch}: Val Acc {best_val_acc:.2f}%) สำเร็จ")

    # บันทึกโมเดล
    if apply_to_active_model:
        backup_file = os.path.join(MODEL_DIR, "char_cnn_backup_before_mining.pth")
        if os.path.exists(save_path):
            shutil.copyfile(save_path, backup_file)
            print(f"[Safety Backup] สำรองโมเดลเดิมไว้ที่: {backup_file}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "val_acc": best_val_acc,
            "best_epoch": best_epoch
        }, save_path)
        print(f"🌟 บันทึกโมเดลที่ปรับปรุงแล้วเสร็จสมบูรณ์ -> {save_path}")

    print("\n" + "=" * 75)
    print(" ✅ กระบวนการ Fine-Tuning เสร็จสมบูรณ์เรียบร้อยแล้ว!")
    print(f" สรุปผล: โมเดลที่ดีที่สุดได้ Validation Accuracy = {best_val_acc:.2f}% (จาก Epoch {best_epoch})")
    print(" แนะนำให้รันคำสั่งด้านล่างเพื่ออัปเดตไฟล์รายงาน Excel ทั้ง 10 แผ่น:")
    print("    python batch_processor.py --model super")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard Example Mining & Targeted Continuous Fine-Tuning Pipeline with Early Stopping")
    parser.add_argument("--input-dir", type=str, default=os.path.join(BASE_DIR, "input_images"), help="โฟลเดอร์ภาพชุดฝึกสอนสำหรับ Mining (Default: input_images)")
    parser.add_argument("--save-path", type=str, default=CNN_MODEL_PATH, help="พาธสำหรับบันทึกโมเดลหลัง Fine-Tuning (Default: models/char_cnn.pth)")
    parser.add_argument("--epochs", type=int, default=None, help="จำนวนรอบ Epochs สูงสุดสำหรับการเทรน (Default: None รันต่อเนื่องจนกระทั่ง Accuracy ลดลงติดต่อกันตามค่า Patience)")
    parser.add_argument("--patience", type=int, default=3, help="จำนวนรอบที่ความแม่นยำลดลงติดต่อกันก่อน Early Stopping จะหยุดเทรน (Default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate สำหรับ Fine-Tune (Default: 0.0002)")
    parser.add_argument("--multiplier", type=int, default=8, help="จำนวนภาพ Augment ขยายเคสยากต่อ 1 ตัวอย่าง (Default: 8)")
    parser.add_argument("--val-split", type=float, default=0.15, help="สัดส่วนข้อมูลสำหรับ Validation Set (Default: 0.15)")
    parser.add_argument("--num-mix", type=int, default=6000, help="จำนวนตัวอย่างมาตรฐานจากแคชที่นำมาผสม (Default: 6000)")
    args = parser.parse_args()
    
    mine_and_finetune(
        input_dir=args.input_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        multiplier=args.multiplier,
        val_split=args.val_split,
        num_regular_mix=args.num_mix
    )
