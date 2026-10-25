import os
import glob
import string
import random
import time
import json
import csv
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from scan_and_predict import (
    preprocess_and_warp,
    detect_table_rows,
    detect_table_columns,
    extract_character_features,
    align_centroid
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "char_cnn.pth")
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
EMNIST_CSV_PATH = os.path.join(BASE_DIR, "emnist_data", "emnist-byclass-train.csv")
AZ_CSV_PATH = os.path.join(BASE_DIR, "A-Z Handwritten Alphabets in CSV Format", "A_Z Handwritten Data.csv")
HASY_DIR = os.path.join(BASE_DIR, "hasy_data")
HASY_CSV = os.path.join(HASY_DIR, "hasy-data-labels.csv")

LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG_FILE = os.path.join(LOG_DIR, "cnn_training_log.csv")
JSON_LOG_FILE = os.path.join(LOG_DIR, "cnn_training_log.json")
REPORT_FILE = os.path.join(LOG_DIR, "cnn_performance_report.md")

BLANK_CHAR = "-"
CLASSES = [BLANK_CHAR] + list(string.ascii_uppercase)
CHAR2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CHAR = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 27



# =========================================================================
# MODEL ARCHITECTURE: Lightweight 2D CNN (28x28 Grayscale Input)
# =========================================================================
class CharCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CharCNN, self).__init__()
        
        # Block 1: 28x28 -> 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )
        
        # Block 2: 14x14 -> 7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20)
        )
        
        # Block 3: 7x7 -> 3x3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with Hard-Negative Mining (Approach 3):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy well-classified examples (e.g. blank space, distinct letters)
    and forces the gradients to concentrate on hard, confusable characters (E vs F, O vs C, I vs T, I vs J, B vs R).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape (NUM_CLASSES,)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, label_smoothing=self.label_smoothing, reduction='none'
        )
        return (focal_weight * ce_loss).mean()


# =========================================================================
# DATASET & AUGMENTATION (With Random Stroke Erasing)
# =========================================================================
class CharDataset(Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images  # uint8 numpy array (N, 28, 28)
        self.labels = labels  # int64 numpy array (N,)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
        
        # If train, apply random slight variations
        if self.is_train:
            if lbl == CHAR2IDX[BLANK_CHAR]:
                # Simulate occasional grid line border remnants in blank cells (20% of the time)
                # This teaches the CNN that vertical/horizontal border edges are still BLANK (-) and NOT 'I'
                if random.random() < 0.20:
                    side = random.choice(["left", "right", "top", "bottom"])
                    b_img = np.zeros((28, 28), dtype=np.uint8)
                    thick = random.choice([1, 2])
                    h_val = random.randint(120, 240)
                    if side == "left":
                        b_img[:, :thick] = h_val
                    elif side == "right":
                        b_img[:, -thick:] = h_val
                    elif side == "top":
                        b_img[:thick, :] = h_val
                    else:
                        b_img[-thick:, :] = h_val
                    img = b_img
            else:
                # Random rotation +-10 deg
                angle = random.uniform(-10, 10)
                cx, cy = 14, 14
                M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                img = cv2.warpAffine(img, M_rot, (28, 28), borderValue=0)
                
                # Random translation +-1.5px
                dx = random.choice([-2, -1, 0, 1, 2])
                dy = random.choice([-2, -1, 0, 1, 2])
                M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M_trans, (28, 28), borderValue=0)
                
                # Random slight shear (-6 to +6 degrees)
                if random.random() < 0.30:
                    shear = random.uniform(-0.12, 0.12)
                    M_shear = np.float32([[1, shear, 0], [0, 1, 0]])
                    img = cv2.warpAffine(img, M_shear, (28, 28), borderValue=0)
                
                # Random slight erosion/dilation (mimic thin pencil vs bold pen)
                if random.random() < 0.30:
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    if random.random() < 0.5:
                        img = cv2.dilate(img, k, iterations=1)
                    else:
                        er = cv2.erode(img, k, iterations=1)
                        if np.count_nonzero(er) > 15:
                            img = er

                # Targeted F vs E Augmentation:
                # For 'F': add tiny smudge near bottom so model learns F with bottom artifact is still F
                if lbl == CHAR2IDX.get("F", -1) and random.random() < 0.25:
                    smudge_x = random.randint(14, 24)
                    smudge_y = random.randint(23, 27)
                    img[smudge_y:smudge_y+2, smudge_x:smudge_x+2] = random.randint(120, 220)

                # For 'E': occasionally shorten bottom bar so model learns to recognize weak-bar E
                if lbl == CHAR2IDX.get("E", -1) and random.random() < 0.25:
                    img[22:28, 18:27] = 0

                # Random Stroke Erasing Augmentation (Pencil Gap & Broken Stroke Simulator)
                if random.random() < 0.35:
                    fg_y, fg_x = np.where(img > 50)
                    if len(fg_x) > 20:
                        rand_idx = random.randint(0, len(fg_x) - 1)
                        rx, ry = fg_x[rand_idx], fg_y[rand_idx]
                        ew = random.randint(1, 3)
                        eh = random.randint(2, 4)
                        ex0 = max(0, rx - ew // 2)
                        ex1 = min(28, rx + ew // 2 + 1)
                        ey0 = max(0, ry - eh // 2)
                        ey1 = min(28, ry + eh // 2 + 1)
                        erased = img.copy()
                        erased[ey0:ey1, ex0:ex1] = 0
                        if np.count_nonzero(erased) > 15:
                            img = erased

                # Targeted O vs C gap simulation for O:
                if lbl == CHAR2IDX.get("O", -1) and random.random() < 0.25:
                    gap_y = random.randint(10, 16)
                    img[gap_y : gap_y + 3, 16:24] = 0
                            
        # Normalize to [0.0, 1.0] and add channel dim -> (1, 28, 28)
        tensor_img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor_img, lbl


def load_dataset_for_cnn(rebuild_cache=False, balance_classes=False, target_samples_per_class=None, max_emnist=400, max_az=400):
    cache_path = os.path.join(BASE_DIR, "dataset_cache_cnn.npz")
    if os.path.exists(cache_path) and not rebuild_cache:
        print(f"\n[Cache] พบไฟล์แคชชุดข้อมูล {cache_path} -> โหลดข้อมูลทันที...")
        cached = np.load(cache_path)
        X_all = cached["X"]
        y_all = cached["y"]
        print(f"[รวมชุดข้อมูล] ทั้งหมด {len(X_all):,} ตัวอย่าง 28x28 (จากแคช)")
        return X_all, y_all

    images_list, labels_list = [], []
    
    # 1. Blank images (500 samples)
    for _ in range(500):
        images_list.append(np.zeros((28, 28), dtype=np.uint8))
        labels_list.append(CHAR2IDX[BLANK_CHAR])
        
    # 2. Real sheets in input_images (รองรับ 87 แผ่นล่าสุด)
    import re
    def natural_sort_key(s):
        return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s)))

    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.*")), key=natural_sort_key)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    odd_expected = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"] + [BLANK_CHAR] * 7
    even_expected = ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] + [BLANK_CHAR] * 7
    
    print(f"\n[1/4] กำลังสกัดตัวอักษร 28x28 จากภาพจริง {len(image_paths)} แผ่น...")
    for img_path in tqdm(image_paths, desc="สกัดจากแผ่นงาน"):
        warped = preprocess_and_warp(img_path)
        row_boxes = detect_table_rows(warped)
        col_boxes = detect_table_columns(warped)
        
        for r, (y_top, y_bot) in enumerate(row_boxes):
            question_num = r + 1
            expected = odd_expected if (question_num % 2 != 0) else even_expected
            for c, (x1, x2) in enumerate(col_boxes):
                cell = warped[y_top + 4 : y_bot - 4, x1 + 5 : x2 - 5]
                feat = extract_character_features(cell)
                label = expected[c]
                
                if label == BLANK_CHAR:
                    images_list.append(np.zeros((28, 28), dtype=np.uint8))
                    labels_list.append(CHAR2IDX[BLANK_CHAR])
                else:
                    if feat is not None:
                        char_28 = (feat[:784].reshape(28, 28) * 255.0).astype(np.uint8)
                        char_28 = align_centroid(char_28)
                        images_list.append(char_28)
                        labels_list.append(CHAR2IDX[label])
                        
    # 3. EMNIST Dataset
    if os.path.exists(EMNIST_CSV_PATH):
        print(f"\n[2/4] กำลังโหลด EMNIST A-Z ({max_emnist} ตัวอย่าง/ตัวอักษร)...")
        counts = {i: 0 for i in range(10, 36)}
        ascii_map = {10 + i: char for i, char in enumerate(string.ascii_uppercase)}
        with open(EMNIST_CSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                lbl_str, pixel_str = line.split(",", 1)
                lbl = int(lbl_str)
                if 10 <= lbl <= 35 and counts[lbl] < max_emnist:
                    pix_28 = np.fromstring(pixel_str, dtype=np.uint8, sep=",").reshape(28, 28).T
                    aligned = align_centroid(pix_28)
                    images_list.append(aligned)
                    labels_list.append(CHAR2IDX[ascii_map[lbl]])
                    counts[lbl] += 1
                    if all(v >= max_emnist for v in counts.values()):
                        break
                        
    # 4. Kaggle A-Z Dataset
    if os.path.exists(AZ_CSV_PATH):
        print(f"\n[3/4] กำลังโหลด Kaggle A-Z ({max_az} ตัวอย่าง/ตัวอักษร)...")
        counts = {i: 0 for i in range(26)}
        ascii_map = {i: char for i, char in enumerate(string.ascii_uppercase)}
        with open(AZ_CSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                lbl_str, pixel_str = line.split(",", 1)
                lbl = int(lbl_str)
                if 0 <= lbl <= 25 and counts[lbl] < max_az:
                    pix_28 = np.fromstring(pixel_str, dtype=np.uint8, sep=",").reshape(28, 28)
                    aligned = align_centroid(pix_28)
                    images_list.append(aligned)
                    labels_list.append(CHAR2IDX[ascii_map[lbl]])
                    counts[lbl] += 1
                    if all(v >= max_az for v in counts.values()):
                        break

    # 5. HASYv2 Dataset (ใหม่: ขยายตัวอย่าง A-Z จากภาพเขียนมือสากล)
    if os.path.exists(HASY_CSV):
        print(f"\n[4/4] กำลังโหลด HASYv2 A-Z ({HASY_CSV})...")
        import pandas as pd
        df_hasy = pd.read_csv(HASY_CSV)
        az_hasy = df_hasy[df_hasy['latex'].isin(list(string.ascii_uppercase))]
        hasy_count = 0
        for _, row in tqdm(az_hasy.iterrows(), total=len(az_hasy), desc="โหลด HASYv2 A-Z"):
            img_rel = row['path']
            img_p = os.path.join(HASY_DIR, img_rel)
            if os.path.exists(img_p):
                raw = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    # Invert polarity: หมึกขาวบนพื้นหลังสีดำ (มาตรฐาน MNIST/EMNIST)
                    inv = 255 - raw
                    r28 = cv2.resize(inv, (28, 28))
                    aligned = align_centroid(r28)
                    images_list.append(aligned)
                    labels_list.append(CHAR2IDX[row['latex']])
                    hasy_count += 1
        print(f"[HASYv2] โหลดตัวอักษร A-Z สำเร็จ: {hasy_count:,} ตัวอย่าง")

    # 6. ปรับสมดุลข้อมูลทุกตัวอักษรให้เท่ากัน (Class Balancing)
    if balance_classes or target_samples_per_class is not None:
        print("\n[Balancing] กำลังปรับสมดุลจำนวนภาพตัวอย่างให้เท่ากันทุกคลาส A-Z...")
        az_indices = {c: [] for c in string.ascii_uppercase}
        blank_indices = []
        for idx, lbl_idx in enumerate(labels_list):
            char_name = IDX2CHAR[lbl_idx]
            if char_name == BLANK_CHAR:
                blank_indices.append(idx)
            else:
                az_indices[char_name].append(idx)

        if target_samples_per_class is None:
            target_count = min(len(idx_list) for idx_list in az_indices.values())
            print(f"[Balancing] ปรับ Undersampling ให้เท่ากับคลาสน้อยสุด: {target_count:,} ภาพ/ตัวอักษร")
        else:
            target_count = target_samples_per_class
            print(f"[Balancing] ปรับให้เท่ากับค่าเป้าหมาย: {target_count:,} ภาพ/ตัวอักษร")

        balanced_indices = []
        np.random.seed(42)
        for c, idx_list in az_indices.items():
            if len(idx_list) >= target_count:
                chosen = np.random.choice(idx_list, size=target_count, replace=False)
            else:
                chosen = np.random.choice(idx_list, size=target_count, replace=True)
            balanced_indices.extend(chosen)

        # เพิ่มช่องว่าง (Blank)
        balanced_indices.extend(blank_indices)
        images_list = [images_list[i] for i in balanced_indices]
        labels_list = [labels_list[i] for i in balanced_indices]
        print(f"[Balancing] ปรับสมดุลเสร็จสิ้น: รวม A-Z ทั้งหมด {target_count * 26:,} ภาพ ({target_count} ภาพ/คลาส)")

    X_all = np.array(images_list, dtype=np.uint8)
    y_all = np.array(labels_list, dtype=np.int64)
    print(f"\n[รวมชุดข้อมูล] ทั้งหมด {len(X_all):,} ตัวอย่าง 28x28")
    
    print(f"[Cache] กำลังบันทึกแคชชุดข้อมูลลง {cache_path}...")
    np.savez_compressed(cache_path, X=X_all, y=y_all)
    print("[Cache] บันทึกสำเร็จ!")
    return X_all, y_all


def train_cnn(fine_tune=True, max_epochs=None, rebuild_cache=False, balance_classes=False, target_samples=None, patience=6, lr=None, batch_size_arg=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cpus = os.cpu_count() or 8
    torch.set_num_threads(num_cpus)
    print("=" * 60)
    print(f"ฝึกสอน 2D CNN ด้วย Focal Loss & Hard-Negative Mining (อุปกรณ์ที่ใช้: {device}, Threads: {num_cpus})")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    X_all, y_all = load_dataset_for_cnn(
        rebuild_cache=rebuild_cache,
        balance_classes=balance_classes,
        target_samples_per_class=target_samples
    )

    # Train / Val Split (90% / 10%)
    indices = np.arange(len(X_all))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = CharDataset(X_all[train_idx], y_all[train_idx], is_train=True)
    val_dataset = CharDataset(X_all[val_idx], y_all[val_idx], is_train=False)
    
    batch_size = batch_size_arg if batch_size_arg is not None else (128 if device.type == "cuda" else 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = CharCNN(num_classes=NUM_CLASSES).to(device)

    # Class Weights for Focal Loss: Boost weights for top confusable characters diagnosed in benchmark:
    # E vs F, O vs C, I vs T, I vs J, B vs R, H
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    hard_chars = ["E", "F", "O", "C", "I", "T", "J", "R", "B", "H"]
    for ch in hard_chars:
        if ch in CHAR2IDX:
            class_weights[CHAR2IDX[ch]] = 1.35
            
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)
    
    # Auto-Backup existing model before starting
    if os.path.exists(MODEL_FILE):
        import shutil
        backup_path = os.path.join(MODEL_DIR, "char_cnn_backup.pth")
        shutil.copyfile(MODEL_FILE, backup_path)
        print(f"[Safety Backup] สำรองโมเดล CNN เดิมไว้ที่: {backup_path}")

    if fine_tune and os.path.exists(MODEL_FILE):
        print(f"\n[Transfer Learning] โหลดน้ำหนักเดิมจาก {MODEL_FILE} เพื่อทำ Fine-Tuning ต่อยอด...")
        ckpt = torch.load(MODEL_FILE, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        initial_lr = lr if lr is not None else 3e-4
        print(f"[Transfer Learning] โหลดสำเร็จ! เริ่มต้น Fine-Tuning ด้วย LR: {initial_lr}")
    else:
        initial_lr = lr if lr is not None else 1e-3
        print(f"\n[From Scratch] เริ่มต้นฝึกสอนจากศูนย์ด้วย LR: {initial_lr}")

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    
    # กำหนดค่า Early Stopping
    PATIENCE = patience
    patience_counter = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_train_acc = 0.0
    
    training_history = []
    
    # เตรียมไฟล์ CSV สำหรับบันทึก Log รายรอบ (Epoch) แบบ Real-time
    with open(CSV_LOG_FILE, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec", "is_best"])

    print("\n" + "=" * 75)
    print(" [Step 2/2] เริ่มต้นฝึกสอน 2D CNN (เทรนไปเรื่อยๆ จนกระทั่งโมเดลเริ่ม Overfit/ลดลง)")
    print(f" * บันทึกประวัติรายรอบอัตโนมัติลง: {CSV_LOG_FILE}")
    print(f" * เงื่อนไขหยุดอัตโนมัติ (Early Stopping): หาก Val Acc ต่ำกว่าสถิติสูงสุดติดต่อกัน {PATIENCE} รอบ")
    print(f" * สามารถกด Ctrl+C เพื่อหยุดเทรนได้ตลอดเวลา (ระบบจะสรุปและเซฟ Report ให้ทันที)")
    print("=" * 75 + "\n")
    
    epoch = 0
    start_total_time = time.time()
    
    try:
        while True:
            if max_epochs is not None and epoch >= max_epochs:
                print(f"\n[Epoch Limit] เทรนครบ {max_epochs} รอบตามที่กำหนดแล้ว สิ้นสุดการฝึกสอน")
                break
            epoch += 1
            t_epoch_start = time.time()
            model.train()
            running_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", unit="batch", ncols=100)
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == lbls).sum().item()
                train_total += lbls.size(0)
                
                batch_acc = (train_correct / train_total) * 100.0
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Train_Acc": f"{batch_acc:5.2f}%"
                })
                
            train_loss = running_train_loss / train_total
            train_acc = (train_correct / train_total) * 100.0
            
            # Validation Step
            model.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outputs = model(imgs)
                    v_loss = criterion(outputs, lbls)
                    running_val_loss += v_loss.item() * imgs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == lbls).sum().item()
                    val_total += lbls.size(0)
                    
            val_loss = running_val_loss / val_total
            val_acc = (val_correct / val_total) * 100.0
            t_epoch_end = time.time()
            epoch_duration = t_epoch_end - t_epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            # ปรับ Learning Rate เมื่อ Val Acc เริ่มนิ่ง
            scheduler.step(val_acc)
            
            is_new_best = val_acc > best_val_acc
            if is_new_best:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "val_acc": val_acc,
                    "epoch": epoch
                }, MODEL_FILE)
            else:
                patience_counter += 1
                
            # บันทึกลง Log รายรอบ (CSV & JSON)
            log_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "lr": current_lr,
                "time_sec": round(epoch_duration, 1),
                "is_best": is_new_best
            }
            training_history.append(log_row)
            
            with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([
                    epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                    f"{val_loss:.4f}", f"{val_acc:.2f}",
                    f"{current_lr:.6f}", f"{epoch_duration:.1f}", is_new_best
                ])
                
            with open(JSON_LOG_FILE, "w", encoding="utf-8") as f_json:
                json.dump(training_history, f_json, indent=2, ensure_ascii=False)
                
            # แสดงผลสรุปรายรอบ
            print(f"  -> จบ Epoch {epoch:03d} ({epoch_duration:.1f}s) | Train Acc: {train_acc:5.2f}% (Loss: {train_loss:.4f}) | Val Acc: {val_acc:5.2f}% (Loss: {val_loss:.4f}) | LR: {current_lr:.6f}")
            
            if is_new_best:
                print(f"     🌟 [สถิติใหม่!] บันทึกโมเดลชุดที่ดีที่สุด (Val Acc = {val_acc:5.2f}%) -> {MODEL_FILE}")
            else:
                print(f"     ⚠️  [Patience {patience_counter}/{PATIENCE}] Val Acc ({val_acc:5.2f}%) ต่ำกว่าสถิติสูงสุด ({best_val_acc:5.2f}% ที่ Epoch {best_epoch})")
                
            # ตรวจสอบเงื่อนไข Early Stopping: หากคะแนนต่ำกว่าสถิติเดิมติดต่อกันครบ PATIENCE รอบ
            if patience_counter >= PATIENCE:
                print("\n" + "=" * 75)
                print(f" [Early Stopping] ความแม่นยำต่ำกว่าสถิติสูงสุดติดต่อกัน {PATIENCE} รอบแล้ว")
                print(f" สิ้นสุดการฝึกสอนที่รอบ {epoch}! (สถิติที่ดีที่สุดอยู่ที่ Epoch {best_epoch}: Val Acc = {best_val_acc:5.2f}%)")
                print("=" * 75)
                break
                
    except KeyboardInterrupt:
        print("\n\n[หยุดการทำงานโดยผู้ใช้] ตรวจพบการกด Ctrl+C -> กำลังสรุปผลจากโมเดลที่ดีที่สุด...")

    total_training_time = time.time() - start_total_time
    
    # =========================================================================
    # MODEL PERFORMANCE REPORT: ประเมินโมเดลที่ดีที่สุดและสร้างรายงาน
    # =========================================================================
    print("\nกำลังโหลดโมเดลที่ดีที่สุดเพื่อสร้าง Performance Report...")
    if not os.path.exists(MODEL_FILE):
        print("[เตือน] ยังไม่มีโมเดลถูกบันทึก")
        return
        
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(lbls.numpy())
            
    clf_report_dict = classification_report(
        y_true_all, y_pred_all, target_names=CLASSES, output_dict=True, zero_division=0
    )
    clf_report_text = classification_report(
        y_true_all, y_pred_all, target_names=CLASSES, zero_division=0
    )
    
    # สร้าง Markdown Performance Report
    report_md = f"""# รายงานประสิทธิภาพโมเดล 2D CNN (CharCNN Performance Report)

- **วันที่และเวลาบันทึก:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **จำนวนรอบที่เทรนทั้งหมด (Total Epochs):** {epoch} รอบ
- **รอบที่ทำผลงานได้ดีที่สุด (Best Epoch):** รอบที่ {best_epoch}
- **เวลารวมในการฝึกสอน:** {total_training_time/60:.2f} นาที ({total_training_time:.1f} วินาที)
- **ไฟล์โมเดลที่บันทึก:** `{MODEL_FILE}`
- **ไฟล์ประวัติการเทรน (CSV):** `{CSV_LOG_FILE}`

---

## 1. ผลสรุปตัวชี้วัดหลัก (Executive Summary)

| ตัวชี้วัด (Metric) | ค่าที่ทำได้ | คำอธิบาย |
| :--- | :---: | :--- |
| **Best Validation Accuracy** | **{best_val_acc:.2f}%** | ความแม่นยำสูงสุดบนชุด Validation Data |
| **Best Training Accuracy** | **{best_train_acc:.2f}%** | ความแม่นยำบนชุดฝึกสอนในรอบที่ดีที่สุด |
| **Best Validation Loss** | **{best_val_loss:.4f}** | ค่า Loss ที่ต่ำที่สุดบนชุด Validation Data |
| **Early Stopping Trigger** | **หยุดที่รอบ {epoch}** | Val Acc ไม่พัฒนาขึ้นติดต่อกัน {PATIENCE} รอบ |

---

## 2. รายละเอียดประสิทธิภาพแยกตามตัวอักษร (Per-Class Performance)

```text
{clf_report_text}
```

---

## 3. ตารางประวัติการฝึกสอนทุกรอบ (Training Progression Log)

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate | Time (s) | Best? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in training_history:
        star = "🌟 (Best)" if row["is_best"] else ""
        report_md += f"| {row['epoch']:02d} | {row['train_loss']:.4f} | {row['train_acc']:.2f}% | {row['val_loss']:.4f} | {row['val_acc']:.2f}% | {row['lr']:.6f} | {row['time_sec']:.1f}s | {star} |\n"

    report_md += """
---

## 4. ขั้นตอนถัดไปในการวัดผล
รันคำสั่งเปรียบเทียบความแม่นยำและระดับความมั่นใจกับโมเดลเดิม Dual-Ensemble (93.94%) บนข้อสอบจริง 9 แผ่น (3,600 ช่อง):
```powershell
python scratch/compare_cnn_vs_ensemble.py
```
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_md)
        
    print("\n" + "=" * 75)
    print(" สรุปผลการฝึกสอน (Performance Summary):")
    print(f" * Best Epoch: {best_epoch} (Val Acc: {best_val_acc:.2f}%)")
    print(f" * บันทึกไฟล์ Log ประวัติ (CSV): {CSV_LOG_FILE}")
    print(f" * บันทึกไฟล์ Log ประวัติ (JSON): {JSON_LOG_FILE}")
    print(f" * บันทึกรายงานฉบับเต็ม (Performance Report): {REPORT_FILE}")
    print("=" * 75)
    print("\n[คำแนะนำ] รันคำสั่งด้านล่างเพื่อเปรียบเทียบกับโมเดลเดิม Dual-Ensemble (93.94%):")
    print("    python scratch/compare_cnn_vs_ensemble.py\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 2D CNN with Hard-Negative Mining & Focal Loss (Approach 3)")
    parser.add_argument("--scratch", action="store_true", help="ฝึกสอนใหม่ทั้งหมดตั้งแต่แรก (ไม่ใช้ weights เดิม)")
    parser.add_argument("--epochs", type=int, default=None, help="จำนวน epoch สูงสุด (Default: รันไปเรื่อยๆ จน Early Stopping)")
    parser.add_argument("--patience", type=int, default=6, help="จำนวนรอบที่คะแนนไม่เพิ่มก่อน Early Stopping (Default: 6)")
    parser.add_argument("--lr", type=float, default=None, help="Learning Rate เริ่มต้น (เช่น 0.0003)")
    parser.add_argument("--batch-size", type=int, default=None, help="ขนาด Batch Size (Default: 128 สำหรับ GPU, 64 สำหรับ CPU)")
    parser.add_argument("--rebuild-cache", action="store_true", help="สร้าง dataset_cache_cnn.npz ใหม่จากภาพกระดาษคำตอบ + HASYv2 + EMNIST + Kaggle")
    parser.add_argument("--balance", action="store_true", help="ปรับสมดุลจำนวนตัวอย่าง A-Z ให้เท่ากันทุกตัวอักษร")
    parser.add_argument("--target-samples", type=int, default=None, help="กำหนดจำนวนตัวอย่างเป้าหมายต่อตัวอักษร A-Z (เช่น 1700)")
    args = parser.parse_args()
    train_cnn(
        fine_tune=(not args.scratch),
        max_epochs=args.epochs,
        rebuild_cache=args.rebuild_cache,
        balance_classes=args.balance,
        target_samples=args.target_samples,
        patience=args.patience,
        lr=args.lr,
        batch_size_arg=args.batch_size
    )
