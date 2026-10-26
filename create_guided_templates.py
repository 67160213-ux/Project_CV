import os
import argparse
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# GROUND TRUTH DEFINITIONS (อ้างอิงตาม batch_processor.py)
# ---------------------------------------------------------

# ชุดทดสอบ (Test Set): ประโยคภาษาอังกฤษ & Tech 20 แถว x 20 ช่อง (อ้างอิงจาก test_image/test1.jpg)
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

# ชุดเทรน (Training Set): ตัวอักษร A-Z (อ้างอิงจาก input_images/1.jpg)
# แถวคี่: A-M ในช่อง 1-13 (ช่อง 14-20 เว้นว่าง)
# แถวคู่: N-Z ในช่อง 1-13 (ช่อง 14-20 เว้นว่าง)
GROUND_TRUTH_ALPHABET = {
    q: ("ABCDEFGHIJKLM-------" if q % 2 != 0 else "NOPQRSTUVWXYZ-------")
    for q in range(1, 21)
}

# พิกัดคอลัมน์แนวตั้งทั้ง 21 เส้นบน template2_1_1.jpg (ความละเอียด 2480 x 3508, 300 DPI)
COL_LINES = [
    255, 353, 453, 553, 653, 753, 853, 953, 1053, 1153,
    1253, 1353, 1453, 1553, 1653, 1753, 1853, 1953, 2053, 2153, 2235
]

def generate_template(template_path, gt_dict, out_jpg, out_pdf, font_size=54, color=(20, 50, 140)):
    """
    สร้างรูปภาพกระดาษคำตอบพร้อมตัวอักษร Ground Truth สีน้ำเงินเข้ม (หรือสีที่ระบุ)
    สำหรับใช้พิมพ์เป็นใบต้นแบบให้เพื่อน/ผู้ทดสอบดูหรือเขียนตามได้ง่าย
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"ไม่พบไฟล์แม่แบบที่: {template_path}")
        
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # เลือกลำดับฟอนต์ที่มีในระบบ
    font_candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf"
    ]
    font_path = next((f for f in font_candidates if os.path.exists(f)), None)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    guide_color = color

    for row_num in range(1, 21):
        y1 = 868 + (row_num - 1) * 116
        y2 = y1 + 90
        row_str = gt_dict.get(row_num, "-" * 20)
        
        for col_idx, ch in enumerate(row_str):
            if ch != "-":
                x1 = COL_LINES[col_idx]
                x2 = COL_LINES[col_idx + 1]
                
                bbox = draw.textbbox((0, 0), ch, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x1 + (x2 - x1 - tw) // 2 - bbox[0]
                ty = y1 + (y2 - y1 - th) // 2 - bbox[1]
                
                draw.text((tx, ty), ch, font=font, fill=guide_color)

    # บันทึกเป็น JPG คุณภาพสูง และ PDF ขนาด A4 300 DPI
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    img.save(out_jpg, quality=98, dpi=(300, 300))
    img.save(out_pdf, "PDF", resolution=300.0)
    print(f"[OK] สร้างไฟล์เรียบร้อย:\n  - JPG: {out_jpg}\n  - PDF: {out_pdf}")

def main():
    parser = argparse.ArgumentParser(description="สร้างกระดาษคำตอบพร้อมตัวอักษร Ground Truth ให้เขียนตามได้ง่าย")
    parser.add_argument("--color", type=str, default="dark_blue", choices=["dark_blue", "navy", "gray", "black"], help="โทนสีตัวอักษร (dark_blue, navy, gray, black)")
    parser.add_argument("--size", type=int, default=54, help="ขนาดตัวอักษร (ค่าเริ่มต้น 54)")
    args = parser.parse_args()

    color_map = {
        "dark_blue": (20, 50, 140),
        "navy": (15, 45, 120),
        "gray": (185, 185, 185),
        "black": (30, 30, 30)
    }
    selected_color = color_map.get(args.color, (20, 50, 140))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "templete", "template2_1_1.jpg")
    out_dir = os.path.join(base_dir, "templete")

    # 1. สร้างสำหรับชุดทดสอบ (Test Set - English & Tech)
    test_jpg = os.path.join(out_dir, "template_test_groundtruth.jpg")
    test_pdf = os.path.join(out_dir, "template_test_groundtruth.pdf")
    print(f"\n--- กำลังสร้างแม่แบบชุดทดสอบ (Test Set - Tech Sentences) [สี: {args.color}] ---")
    generate_template(template_path, GROUND_TRUTH_TECH, test_jpg, test_pdf, font_size=args.size, color=selected_color)

    # 2. สร้างสำหรับชุดเทรน (Training Set - Alphabet A-Z)
    train_jpg = os.path.join(out_dir, "template_train_groundtruth.jpg")
    train_pdf = os.path.join(out_dir, "template_train_groundtruth.pdf")
    print(f"\n--- กำลังสร้างแม่แบบชุดเทรน (Training Set - Alphabet A-Z) [สี: {args.color}] ---")
    generate_template(template_path, GROUND_TRUTH_ALPHABET, train_jpg, train_pdf, font_size=args.size, color=selected_color)

if __name__ == "__main__":
    main()
