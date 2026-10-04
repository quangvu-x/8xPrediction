"""
detect.py - Bước 1: video -> results/raw.csv + results/meta.json
- YOLO (ultralytics, model COCO pretrained) phát hiện người (class 0) và bóng (class 32).
- ByteTrack (thư viện supervision) gán ID cố định cho cầu thủ.
- Bóng: mỗi frame chỉ giữ 1 detection có conf cao nhất, track_id = -1.
- Lấy màu áo trung bình của từng cầu thủ (bỏ pixel màu cỏ).

Vì sao không dùng model.track() của ultralytics: tracker của ultralytics bỏ các
detection không được gán vào track (thường là bóng có conf thấp). Tách riêng
"detect" và "track" giúp giữ lại bóng.

Chạy:  python detect.py --video input/clip3s.mp4
       python detect.py --video input/clip.mp4 --model yolov8m.pt --imgsz 960 --conf 0.1
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

import config as C


def parse_args():
    p = argparse.ArgumentParser(description="Phát hiện + theo dõi cầu thủ và bóng")
    p.add_argument("--video", default=str(C.DEFAULT_VIDEO), help="đường dẫn video đầu vào")
    p.add_argument("--model", default=C.MODEL, help="file weights YOLO")
    p.add_argument("--imgsz", type=int, default=C.IMGSZ)
    p.add_argument("--conf", type=float, default=C.CONF)
    return p.parse_args()


def open_video(path):
    """Mở video, báo lỗi tiếng Việt nếu không được."""
    from pathlib import Path
    if not Path(path).exists():
        sys.exit(f"[LỖI] Không tìm thấy video: {path}\n"
                 f"      Hãy đặt clip vào thư mục input/ (xem Bước 2 trong kế hoạch).")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"[LỖI] Không đọc được video: {path}. Thử cắt lại clip bằng ffmpeg (Bước 2).")
    return cap


def jersey_color(frame_bgr, x1, y1, x2, y2):
    """Màu áo trung bình (r, g, b) của vùng thân trên, đã bỏ pixel màu cỏ."""
    h_img, w_img = frame_bgr.shape[:2]
    w, h = x2 - x1, y2 - y1
    ya = int(max(0, y1 + C.JERSEY_Y[0] * h))
    yb = int(min(h_img, y1 + C.JERSEY_Y[1] * h))
    xa = int(max(0, x1 + C.JERSEY_X[0] * w))
    xb = int(min(w_img, x1 + C.JERSEY_X[1] * w))
    crop = frame_bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return (np.nan, np.nan, np.nan)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    is_grass = ((hsv[:, 0] >= C.GRASS_HUE[0]) & (hsv[:, 0] <= C.GRASS_HUE[1])
                & (hsv[:, 1] > C.GRASS_MIN_SAT))
    keep = rgb[~is_grass]
    if len(keep) == 0:          # toàn cỏ -> lấy trung bình cả vùng
        keep = rgb
    r, g, b = keep.mean(axis=0)
    return (round(r, 1), round(g, 1), round(b, 1))


def make_tracker(fps):
    import supervision as sv
    try:
        return sv.ByteTrack(frame_rate=int(round(fps)))
    except TypeError:            # phiên bản supervision khác chữ ký hàm
        return sv.ByteTrack()


def to_numpy(t):
    """Tensor torch (CPU/GPU) hoặc numpy -> numpy."""
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t)


def main():
    args = parse_args()
    import supervision as sv
    from ultralytics import YOLO

    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"Đang tải model {args.model} (lần đầu sẽ tự tải từ internet)...")
    model = YOLO(args.model)
    tracker = make_tracker(fps)

    rows = []
    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, classes=[C.PERSON_CLASS, C.BALL_CLASS],
                               conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = to_numpy(boxes.xyxy).astype(float)
            cls = to_numpy(boxes.cls).astype(int)
            conf = to_numpy(boxes.conf).astype(float)

            # --- Bóng: giữ detection có conf cao nhất ---
            ball_idx = np.where(cls == C.BALL_CLASS)[0]
            if len(ball_idx) > 0:
                i = ball_idx[np.argmax(conf[ball_idx])]
                x1, y1, x2, y2 = xyxy[i]
                rows.append([frame_idx, -1, "ball", round(conf[i], 4),
                             round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                             np.nan, np.nan, np.nan])

            # --- Cầu thủ: đưa qua ByteTrack để có ID ---
            person_idx = np.where(cls == C.PERSON_CLASS)[0]
            dets = sv.Detections(
                xyxy=xyxy[person_idx] if len(person_idx) else np.empty((0, 4)),
                confidence=conf[person_idx] if len(person_idx) else np.empty(0),
                class_id=np.zeros(len(person_idx), dtype=int),
            )
        else:
            dets = sv.Detections.empty()

        tracked = tracker.update_with_detections(dets)
        if tracked.tracker_id is not None:
            for (x1, y1, x2, y2), tid, cf in zip(tracked.xyxy, tracked.tracker_id,
                                                 tracked.confidence):
                r, g, b = jersey_color(frame, x1, y1, x2, y2)
                rows.append([frame_idx, int(tid), "player", round(float(cf), 4),
                             round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                             r, g, b])

        frame_idx += 1
        if frame_idx % 25 == 0:
            total = f"/{total_hint}" if total_hint else ""
            print(f"  Frame {frame_idx}{total}  ({time.time() - t0:.0f}s)")

    cap.release()
    if frame_idx == 0:
        sys.exit("[LỖI] Video không có frame nào đọc được.")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"fps": float(fps), "width": width, "height": height, "n_frames": frame_idx}
    C.META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    raw = pd.DataFrame(rows, columns=C.RAW_COLS)
    raw = raw.sort_values(["frame", "track_id"]).reset_index(drop=True)
    raw.to_csv(C.RAW_CSV, index=False)

    n_ball = raw.loc[raw.cls == "ball", "frame"].nunique()
    n_player_rows = (raw.cls == "player").sum()
    print(f"\nXONG trong {time.time() - t0:.0f}s - {frame_idx} frame")
    print(f"  Trung bình {n_player_rows / frame_idx:.1f} cầu thủ/frame; "
          f"bóng xuất hiện ở {n_ball}/{frame_idx} frame ({100 * n_ball / frame_idx:.1f}%)")
    print(f"  Đã ghi {C.RAW_CSV} và {C.META_JSON}")


if __name__ == "__main__":
    main()
