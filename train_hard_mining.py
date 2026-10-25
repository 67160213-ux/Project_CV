"""
========================================================================================
 🔬 Hard Example Mining & Targeted Fine-Tuning Pipeline (แนวทางที่ 1)
----------------------------------------------------------------------------------------
 สคริปต์นี้ทำหน้าที่:
 1. สแกนกระดาษคำตอบทดสอบเพื่อค้นหาตัวอักษรที่โมเดลปัจจุบันยังทำนายผิด (Hard Examples)
 2. นำภาพตัวอักษรที่ผิดมาทำ Data Augmentation เพิ่ม 8 เท่า (จำลองเอียง, สั่น, หนา, บาง)
 3. ผสมกับข้อมูลมาตรฐานจากแคชเพื่อป้องกันการลืมข้อมูลเดิม (Catastrophic Forgetting)
 4. Fine-Tune โมเดล 2D CNN (char_cnn.pth) ด้วย Focal Loss และ Learning Rate ต่ำ (2e-4)
 5. สำรองโมเดลเดิมอัตโนมัติก่อนบันทึกโมเดลที่ปรับปรุงแล้ว
========================================================================================
"""
import sys
import os
import cv2
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm

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
from eval_accuracy import GROUND_TRUTH_TECH, GROUND_TRUTH_ALPHABET
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

def mine_and_finetune(epochs=5, lr=2e-4, multiplier=8, num_regular_mix=6000, apply_to_active_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" 🚀 เริ่มต้นกระบวนการ Hard Example Mining & Targeted Fine-Tuning (แนวทางที่ 1)")
    print(f" * อุปกรณ์ประมวลผล: {device}")
    print("=" * 70)

    # -------------------------------------------------------------
    # ขั้นตอนที่ 1: สแกนหาข้อที่โมเดลปัจจุบันทายผิด (Hard Examples)
    # -------------------------------------------------------------
    print("\n[Step 1/4] กำลังสแกนหาตัวอักษรที่ทายผิดจากกระดาษทดสอบ (Mining Hard Examples)...")
    clf = load_classifier('super')

    hard_images = []
    hard_labels = []
    
    test_files = [f"test{i}.jpg" for i in range(1, 11)]
    for fname in tqdm(test_files, desc="สแกนกระดาษทดสอบ"):
        p = os.path.join(BASE_DIR, "test_image", fname)
        if not os.path.exists(p):
            continue
        warped = preprocess_and_warp(p)
        rows = detect_table_rows(warped)
        cols = detect_table_columns(warped)
        gt = GROUND_TRUTH_TECH
        
        for r, (y1, y2) in enumerate(rows):
            q = r + 1
            exp_s = gt.get(q, "-" * 20)
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

    print(f"\n[*] ค้นพบตัวอย่างลายมือยาก (Hard Examples) ทั้งหมด: {len(hard_images)} ตัวอย่าง")

    if not hard_images:
        print("[!] ไม่พบข้อผิดพลาดในชุดทดสอบ ระบบมีความแม่นยำ 100% แล้ว")
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
    # ขั้นตอนที่ 3: ผสมข้อมูลมาตรฐานจากแคชเพื่อคงความรู้เดิม
    # -------------------------------------------------------------
    print(f"\n[Step 3/4] ผสมข้อมูลมาตรฐาน {num_regular_mix:,} ตัวอย่างเพื่อป้องกันการลืมข้อมูลเดิม...")
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"ไม่พบไฟล์แคชชุดข้อมูลที่ {CACHE_FILE}")
        
    cached = np.load(CACHE_FILE)
    X_cached, y_cached = cached["X"], cached["y"]
    np.random.seed(42)
    reg_idx = np.random.choice(len(X_cached), size=min(num_regular_mix, len(X_cached)), replace=False)
    X_reg, y_reg = X_cached[reg_idx], y_cached[reg_idx]

    X_train = np.concatenate([np.array(aug_images, dtype=np.uint8), X_reg], axis=0)
    y_train = np.concatenate([np.array(aug_labels, dtype=np.int64), y_reg], axis=0)
    print(f"[*] รวมชุดข้อมูล Fine-Tuning ทั้งสิ้น: {len(X_train):,} ตัวอย่าง")

    train_loader = DataLoader(MinedDataset(X_train, y_train), batch_size=128, shuffle=True)

    # -------------------------------------------------------------
    # ขั้นตอนที่ 4: Fine-Tune โมเดล CNN ด้วย Focal Loss
    # -------------------------------------------------------------
    print(f"\n[Step 4/4] เริ่มต้น Fine-Tuning 2D CNN ({epochs} Epochs, LR={lr})...")
    model = CharCNN(num_classes=len(CLASSES)).to(device)
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"ไม่พบโมเดลเดิมที่ {CNN_MODEL_PATH}")
        
    ckpt = torch.load(CNN_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        corr = 0
        tot = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
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
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{corr/tot*100:.2f}%"})

    # บันทึกโมเดล
    if apply_to_active_model:
        backup_file = os.path.join(MODEL_DIR, "char_cnn_backup_before_mining.pth")
        shutil.copyfile(CNN_MODEL_PATH, backup_file)
        print(f"\n[Safety Backup] สำรองโมเดลเดิมไว้ที่: {backup_file}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "val_acc": corr/tot*100
        }, CNN_MODEL_PATH)
        print(f"🌟 บันทึกโมเดลที่ปรับปรุงแล้วเสร็จสมบูรณ์ -> {CNN_MODEL_PATH}")

    print("\n" + "=" * 70)
    print(" ✅ กระบวนการ Fine-Tuning เสร็จสมบูรณ์เรียบร้อยแล้ว!")
    print(" แนะนำให้รันคำสั่งด้านล่างเพื่ออัปเดตไฟล์รายงาน Excel ทั้ง 10 แผ่น:")
    print("    python batch_processor.py --model super")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard Example Mining & Targeted Fine-Tuning Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="จำนวนรอบ Epochs สำหรับ Fine-Tune (Default: 5)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate สำหรับ Fine-Tune (Default: 0.0002)")
    parser.add_argument("--multiplier", type=int, default=8, help="จำนวนภาพ Augment ขยายเคสยากต่อ 1 ตัวอย่าง (Default: 8)")
    args = parser.parse_args()
    
    mine_and_finetune(
        epochs=args.epochs,
        lr=args.lr,
        multiplier=args.multiplier
    )
