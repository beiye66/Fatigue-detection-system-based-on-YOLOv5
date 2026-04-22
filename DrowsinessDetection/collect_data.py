# -*- coding: utf-8 -*-
"""
数据采集脚本 - 摄像头拍照 + 自动打 YOLO 标签

操作说明：
  E 键 → 保存"闭眼"样本 (Eyeclosed, class 0)
  N 键 → 保存"正常"样本 (Neutral,   class 1)
  Y 键 → 保存"打哈欠"样本 (Yawn,    class 2)
  Q 键 → 退出

保存位置：DrowsinessDetection/collected_data/
  collected_data/
  ├── images/Eyeclosed/  ← 图片
  ├── images/Neutral/
  ├── images/Yawn/
  ├── labels/Eyeclosed/  ← YOLO 格式标注 (.txt)
  ├── labels/Neutral/
  └── labels/Yawn/

采集完成后运行 prepare_dataset.py 整理成训练集。
"""

import cv2
import os
from pathlib import Path

# ──────────────────── 配置 ────────────────────
SAVE_DIR = Path("collected_data")        # 保存根目录（相对于本脚本）
CAM_INDEX = 0                            # 摄像头编号，一般为 0
PAD_RATIO = 0.20                         # 检测框向外扩展比例（让框更完整地包住脸）

CLASS_KEYS = {
    ord('e'): ('Eyeclosed', 0),
    ord('n'): ('Neutral',   1),
    ord('y'): ('Yawn',      2),
}
# ─────────────────────────────────────────────

def setup_dirs():
    """创建保存目录"""
    for name, _ in CLASS_KEYS.values():
        (SAVE_DIR / 'images' / name).mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / 'labels' / name).mkdir(parents=True, exist_ok=True)

def load_face_detector():
    """加载 OpenCV 内置人脸检测器（无需额外安装）"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("找不到人脸检测器文件，请检查 OpenCV 安装")
    return detector

def detect_face(frame, detector):
    """
    检测画面中最大的人脸，返回扩展后的边界框 (x, y, w, h)。
    未检测到返回 None。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    if len(faces) == 0:
        return None

    # 取面积最大的脸
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    img_h, img_w = frame.shape[:2]

    # 向外扩展 PAD_RATIO，让框更完整
    pad_x = int(w * PAD_RATIO)
    pad_y = int(h * PAD_RATIO)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return x1, y1, x2 - x1, y2 - y1  # x, y, w, h

def save_sample(frame, face_box, class_name, class_id, count):
    """保存图片和对应的 YOLO 标注文件"""
    img_h, img_w = frame.shape[:2]
    x, y, w, h = face_box

    # YOLO 格式：归一化的中心点坐标 + 宽高
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    # 确保值在 [0, 1] 范围内
    cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))

    filename = f"{class_name}_{count:04d}"
    img_path   = SAVE_DIR / 'images' / class_name / f"{filename}.jpg"
    label_path = SAVE_DIR / 'labels' / class_name / f"{filename}.txt"

    cv2.imwrite(str(img_path), frame)
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

def draw_overlay(frame, face_box, counts):
    """在画面上绘制提示信息"""
    display = frame.copy()
    img_h, img_w = frame.shape[:2]

    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, "Face OK", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(display, "No face detected", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # 底部状态栏
    bar_text = (f"E: Eyeclosed({counts['Eyeclosed']})  "
                f"N: Neutral({counts['Neutral']})  "
                f"Y: Yawn({counts['Yawn']})  "
                f"| Q: 退出")
    cv2.rectangle(display, (0, img_h - 30), (img_w, img_h), (30, 30, 30), -1)
    cv2.putText(display, bar_text, (8, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # 保存提示（按键后短暂显示）
    return display

def main():
    setup_dirs()
    detector = load_face_detector()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {CAM_INDEX}")
        return

    # 读取已有文件数量（断点续采）
    counts = {}
    for name, _ in CLASS_KEYS.values():
        existing = list((SAVE_DIR / 'images' / name).glob('*.jpg'))
        counts[name] = len(existing)

    print("=" * 50)
    print("数据采集脚本启动")
    print("  E → 闭眼  N → 正常  Y → 打哈欠  Q → 退出")
    print(f"  已有样本: {counts}")
    print("=" * 50)

    flash_text = ""  # 保存成功时的闪烁提示
    flash_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 读取摄像头帧失败")
            break

        face_box = detect_face(frame, detector)
        display = draw_overlay(frame, face_box, counts)

        # 显示保存成功提示
        if flash_timer > 0:
            img_h, img_w = display.shape[:2]
            cv2.putText(display, flash_text,
                        (img_w // 2 - 120, img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2, cv2.LINE_AA)
            flash_timer -= 1

        cv2.imshow("数据采集 - 疲劳检测", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in CLASS_KEYS:
            class_name, class_id = CLASS_KEYS[key]
            if face_box is None:
                print(f"[跳过] 未检测到人脸，请正对摄像头")
                flash_text = "No face! 请正对摄像头"
                flash_timer = 25
            else:
                save_sample(frame, face_box, class_name, class_id, counts[class_name])
                counts[class_name] += 1
                print(f"[保存] {class_name} #{counts[class_name]:04d}")
                flash_text = f"Saved: {class_name} #{counts[class_name]}"
                flash_timer = 20

    cap.release()
    cv2.destroyAllWindows()

    print("\n采集结束，各类别数量：")
    for name, cnt in counts.items():
        print(f"  {name}: {cnt} 张")
    print(f"\n数据保存在: {SAVE_DIR.resolve()}")
    print("运行 prepare_dataset.py 整理成训练集。")


if __name__ == "__main__":
    main()
