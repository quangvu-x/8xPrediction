"""
detect_web.py - Bản detect nhẹ cho web: YOLOv8 ONNX (onnxruntime) + SimpleTracker.
Xuất raw.csv + meta.json ĐÚNG hợp đồng dữ liệu → dùng lại analytics.py, render.py.

Chạy:  python detect_web.py --video input/clip.mp4 --model models/yolov8n.onnx
In dòng "PROGRESS <đã xử lý> <tổng>" để web hiển thị thanh tiến độ.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config as C
from onnx_detector import OnnxYolo
from simple_tracker import SimpleTracker


def jersey_color(frame_bgr, x1, y1, x2, y2):
    """Giống detect.py: màu áo trung bình vùng thân trên, bỏ pixel cỏ."""
    h_img, w_img = frame_bgr.shape[:2]
    w, h = x2 - x1, y2 - y1
    ya, yb = int(max(0, y1 + C.JERSEY_Y[0] * h)), int(min(h_img, y1 + C.JERSEY_Y[1] * h))
    xa, xb = int(max(0, x1 + C.JERSEY_X[0] * w)), int(min(w_img, x1 + C.JERSEY_X[1] * w))
    crop = frame_bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return (np.nan, np.nan, np.nan)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    grass = ((hsv[:, 0] >= C.GRASS_HUE[0]) & (hsv[:, 0] <= C.GRASS_HUE[1])
             & (hsv[:, 1] > C.GRASS_MIN_SAT))
    keep = rgb[~grass] if (~grass).any() else rgb
    r, g, b = keep.mean(axis=0)
    return (round(r, 1), round(g, 1), round(b, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=str(C.ROOT / "models" / "yolov8n.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()

    if not Path(a.model).exists():
        sys.exit(f"[LỖI] Không thấy model ONNX: {a.model}")
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"[LỖI] Không đọc được video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    model = OnnxYolo(a.model, imgsz=a.imgsz)
    tracker = SimpleTracker(max_lost=int(fps * 0.6))
    rows, k, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        xyxy, cls, cf = model.predict(frame, conf=a.conf, classes=(C.PERSON_CLASS, C.BALL_CLASS))
        b = np.where(cls == C.BALL_CLASS)[0]
        if len(b):
            i = b[np.argmax(cf[b])]
            x1, y1, x2, y2 = xyxy[i]
            rows.append([k, -1, "ball", round(float(cf[i]), 4), round(x1, 1), round(y1, 1),
                         round(x2, 1), round(y2, 1), np.nan, np.nan, np.nan])
        p = np.where(cls == C.PERSON_CLASS)[0]
        for tid, (x1, y1, x2, y2), c in tracker.update(xyxy[p], cf[p]):
            r, g, bb = jersey_color(frame, x1, y1, x2, y2)
            rows.append([k, int(tid), "player", round(float(c), 4), round(x1, 1), round(y1, 1),
                         round(x2, 1), round(y2, 1), r, g, bb])
        k += 1
        if k % 10 == 0:
            print(f"PROGRESS {k} {total}", flush=True)
    cap.release()
    if k == 0:
        sys.exit("[LỖI] Video không có frame nào.")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.META_JSON.write_text(json.dumps({"fps": float(fps), "width": W, "height": H,
                                       "n_frames": k}, indent=2), encoding="utf-8")
    raw = pd.DataFrame(rows, columns=C.RAW_COLS).sort_values(["frame", "track_id"])
    raw.to_csv(C.RAW_CSV, index=False)
    print(f"XONG detect_web: {k} frame trong {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
