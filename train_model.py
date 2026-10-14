import cv2
import numpy as np
import os
import glob
import joblib
import string
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from scan_and_predict import (
    preprocess_and_warp,
    detect_table_rows,
    detect_table_columns,
    extract_character_features,
    align_centroid,
    compute_hog_28
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "char_classifier.pkl")
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
EMNIST_CSV_PATH = os.path.join(BASE_DIR, "emnist_data", "emnist-byclass-train.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

BLANK_CHAR = "-"
FEATURE_DIM = 784 + 392  # 784 Raw Pixels + 392 HOG Gradient Features = 1,176 Dimensions

def advanced_augment_sample(img_28):
    """
    Data Augmentation ขั้นสูง:
    1. องศาการเอียง (Rotations): -12, -8, -4, +4, +8, +12 องศา
    2. ตำแหน่งการเลื่อน (Translations): เลื่อน +-2px แนวนอน/แนวดิ่ง/ทแยงมุม
    3. ลายมือตวัดเอียง (Shear / Slant): เอียงซ้าย/ขวา สไตล์ตัวเขียน
    4. ขนาดหัวปากกา (Stroke Thickness): 
       - ขยายเส้นหนา (Dilation) เลียนแบบปากกาหัว 0.7 มม.
       - รีดเส้นบาง (Erosion) เลียนแบบปากกาหัว 0.38 มม. / ดินสอ
    5. ขนาดย่อ/ขยาย (Zoom / Scale): ซูมเข้า 110% และซูมออก 90%
    """
    augmented = [img_28]
    h, w = img_28.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. หมุนองศา (Rotations)
    for angle in [-12, -8, -4, 4, 8, 12]:
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        augmented.append(cv2.warpAffine(img_28, M_rot, (28, 28), borderValue=0))

    # 2. เลื่อนตำแหน่งพิกัด (Translations)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1)]:
        M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
        augmented.append(cv2.warpAffine(img_28, M_trans, (28, 28), borderValue=0))

    # 3. ลายมือเอียง (Slant / Shear)
    for shear in [-0.15, 0.15]:
        M_shear = np.float32([[1, shear, -shear * cy], [0, 1, 0]])
        augmented.append(cv2.warpAffine(img_28, M_shear, (28, 28), borderValue=0))

    # 4. ขนาดเส้นหนา / บาง (Morphological Thickness)
    k_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(img_28, k_thick, iterations=1)
    augmented.append(dilated)

    eroded = cv2.erode(img_28, k_thick, iterations=1)
    if np.count_nonzero(eroded) > 15:
        augmented.append(eroded)

    # จำลองลายมือดินสอเส้นบาง (Thin Pencil Stroke Simulation)
    k_thin_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    eroded_v = cv2.erode(img_28, k_thin_v, iterations=1)
    if np.count_nonzero(eroded_v) > 15:
        augmented.append(eroded_v)

    k_thin_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    eroded_h = cv2.erode(img_28, k_thin_h, iterations=1)
    if np.count_nonzero(eroded_h) > 15:
        augmented.append(eroded_h)

    # 5. ขนาดย่อ/ขยาย (Zoom Variations)
    M_zoom_in = cv2.getRotationMatrix2D((cx, cy), 0, 1.1)
    augmented.append(cv2.warpAffine(img_28, M_zoom_in, (28, 28), borderValue=0))

    M_zoom_out = cv2.getRotationMatrix2D((cx, cy), 0, 0.9)
    augmented.append(cv2.warpAffine(img_28, M_zoom_out, (28, 28), borderValue=0))

    return augmented

def build_dataset_from_images(image_paths):
    X, y = [], []

    odd_expected = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"] + [BLANK_CHAR] * 7
    even_expected = ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] + [BLANK_CHAR] * 7

    # เพิ่มข้อมูล Baseline สำหรับช่องว่างบริสุทธิ์ (Pure Blank Canvas) ขนาด 1,176 มิติ
    for _ in range(500):
        X.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        y.append(BLANK_CHAR)

    print("\n[Step 1/2] กำลังสกัดตัวอักษร, ทำ Centroid Alignment และสกัด HOG Features...")
    pbar_img = tqdm(image_paths, desc="ประมวลผลรูปภาพ", unit="ภาพ")

    for img_path in pbar_img:
        warped = preprocess_and_warp(img_path)
        row_boxes = detect_table_rows(warped)
        col_boxes = detect_table_columns(warped)

        for r, (y_top, y_bot) in enumerate(row_boxes):
            question_num = r + 1
            expected = odd_expected if (question_num % 2 != 0) else even_expected

            for c, (x1, x2) in enumerate(col_boxes):
                pad_y = 4
                pad_x = 5
                cell = warped[y_top + pad_y : y_bot - pad_y, x1 + pad_x : x2 - pad_x]

                feat = extract_character_features(cell)
                label = expected[c]

                if label == BLANK_CHAR:
                    # ช่องว่าง: บันทึกเฉพาะภาพที่ว่างเปล่าจริง (Zero vector) เท่านั้น
                    # ป้องกันไม่ให้เศษเส้นขอบตารางแนวตั้งเข้ามาปนเปื้อนจนโมเดลสับสนกับตัว 'I'
                    feat_blank = np.zeros(FEATURE_DIM, dtype=np.float32)
                    X.append(feat_blank)
                    y.append(BLANK_CHAR)
                else:
                    # ตัวอักษร: นำไปทำ Advanced Augmentation + Centroid Alignment + HOG Features
                    if feat is not None:
                        char_28 = (feat[:784].reshape(28, 28) * 255.0).astype(np.uint8)
                        samples = advanced_augment_sample(char_28)
                        for s in samples:
                            s_aligned = align_centroid(s)
                            pix_s = s_aligned.flatten() / 255.0
                            hog_s = compute_hog_28(s_aligned)
                            combined_s = np.hstack([pix_s, hog_s]).astype(np.float32)
                            X.append(combined_s)
                            y.append(label)

        pbar_img.set_postfix({"ตัวอย่างสะสม": f"{len(X):,}"})

    return np.array(X, dtype=np.float32), np.array(y)

def load_emnist_from_csv(csv_path=EMNIST_CSV_PATH, samples_per_class=500):
    """
    ดึงข้อมูลตัวพิมพ์ใหญ่ A-Z จาก EMNIST CSV (byclass)
    พร้อมแก้ไข Transpose (.T), ทำ Centroid Alignment และสกัด Hybrid Features 1,176 มิติ
    """
    if not os.path.exists(csv_path):
        print(f"[EMNIST] ไม่พบไฟล์ {csv_path} -> ข้ามการโหลด EMNIST")
        return [], []

    print(f"\n[EMNIST] กำลังโหลดตัวอักษร A-Z จาก EMNIST CSV (เป้าหมาย: {samples_per_class} ตัวอย่าง/ตัวอักษร)...")
    target_total = 26 * samples_per_class
    counts = {i: 0 for i in range(10, 36)}
    ascii_map = {10 + i: char for i, char in enumerate(string.ascii_uppercase)}

    X_emnist, y_emnist = [], []
    pbar_emnist = tqdm(total=target_total, desc="โหลด EMNIST A-Z", unit="ตัวอย่าง")

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            lbl_str, pixel_str = line.split(",", 1)
            lbl = int(lbl_str)

            if 10 <= lbl <= 35 and counts[lbl] < samples_per_class:
                # 1. แปลงเป็นภาพ 28x28 และสลับแกน Transpose (.T) เพื่อให้ตัวอักษรตั้งตรง
                pix_28 = np.fromstring(pixel_str, dtype=np.uint8, sep=",").reshape(28, 28).T

                # 2. จัดกึ่งกลางมวลหมึกด้วย Image Moments มาที่พิกัด (14, 14)
                aligned = align_centroid(pix_28)

                # 3. สกัด Hybrid Features 1,176 มิติ (Raw Pixels 784 + HOG 392)
                pix_feat = aligned.flatten() / 255.0
                hog_feat = compute_hog_28(aligned)
                comb_feat = np.hstack([pix_feat, hog_feat]).astype(np.float32)

                X_emnist.append(comb_feat)
                y_emnist.append(ascii_map[lbl])
                counts[lbl] += 1
                pbar_emnist.update(1)

                if len(X_emnist) == target_total:
                    break

    pbar_emnist.close()
    print(f"[EMNIST] โหลดเสร็จสิ้น: รวม {len(X_emnist):,} ตัวอย่าง ({samples_per_class} ตัวอย่าง/ตัวอักษร)")
    return X_emnist, y_emnist

def train_and_save():
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพในโฟลเดอร์ {INPUT_DIR}")

    print("=" * 65)
    print(f" เริ่มต้นกระบวนการฝึกสอนโมเดล (Hybrid Pixels + HOG + Centroid Alignment)")
    print(f" ประมวลผลจากกระดาษคำตอบ: {len(image_paths)} ภาพ (มิติ Feature: {FEATURE_DIM} มิติ)")
    print("=" * 65)

    X_sheet, y_sheet = build_dataset_from_images(image_paths)

    # ดึงข้อมูลตัวอักษรลายมือเพิ่มเติมจาก EMNIST (ตัวละ 500 ภาพ = 13,000 ตัวอย่าง)
    X_emnist, y_emnist = load_emnist_from_csv(EMNIST_CSV_PATH, samples_per_class=500)

    if len(X_emnist) > 0:
        print("\nกำลังผสานชุดข้อมูลกระดาษคำตอบกับ EMNIST...")
        X = np.vstack([X_sheet, np.array(X_emnist, dtype=np.float32)])
        y = np.concatenate([y_sheet, np.array(y_emnist)])
    else:
        X, y = X_sheet, y_sheet

    classes = np.unique(y)
    print(f"\nสรุปข้อมูลชุดฝึกสอนรวม:")
    print(f" - ข้อมูลจากกระดาษคำตอบ: {len(X_sheet):,} ตัวอย่าง")
    print(f" - ข้อมูลเสริมจาก EMNIST: {len(X_emnist):,} ตัวอย่าง")
    print(f" - จำนวนตัวอย่างทั้งหมด: {len(X):,} ตัวอย่าง")
    print(f" - มิติเวกเตอร์ (Features): {X.shape[1]} มิติ (784 Pixels + 392 HOG)")
    print(f" - จำนวนคลาสที่เรียนรู้: {len(classes)} คลาส ({', '.join(classes)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("\n[Step 2/2] กำลังฝึกสอนโครงข่ายประสาทเทียม (Training Neural Network)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(384, 192),
        activation="relu",
        solver="adam",
        alpha=0.001,
        random_state=42,
    )

    # วนลูป Epochs พร้อมแถบเปอร์เซ็นต์แบบ Real-Time ด้วย tqdm
    num_epochs = 60
    pbar_train = tqdm(range(1, num_epochs + 1), desc="รอบการเทรน (Epochs)", unit="epoch")

    for epoch in pbar_train:
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar_train.set_postfix({
            "Loss": f"{clf.loss_:.4f}",
            "Progress": f"{(epoch/num_epochs)*100:.1f}%"
        })

    print("\n" + "=" * 65)
    print(" วัดผลและประเมินความแม่นยำ (Evaluation Results):")
    print("=" * 65)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(f" ความแม่นยำชุดฝึกสอน (Train Accuracy): {train_acc * 100:.2f}%")
    print(f" ความแม่นยำชุดทดสอบ  (Test Accuracy):  {test_acc * 100:.2f}%")

    joblib.dump(clf, MODEL_FILE)
    print(f"\nบันทึกไฟล์โมเดลเสร็จสมบูรณ์ที่: {MODEL_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    train_and_save()
