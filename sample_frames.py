"""
sample_frames.py - Tạo "bảng kiểm tra bằng mắt": 12 frame rải đều từ results/output.mp4
ghép thành 1 ảnh results/_qa_sheet.jpg, kèm số frame/giây. Dùng để chấm độ chính xác thủ công
(đếm cầu thủ sai đội, bóng sai vị trí, người giữ bóng sai) trước và sau mỗi lần cải tiến.

Chạy:  python sample_frames.py [--n 12]
"""
import argparse

import cv2
import numpy as np

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    cap = cv2.VideoCapture(str(C.OUTPUT_MP4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if total == 0:
        raise SystemExit("[LỖI] Không đọc được results/output.mp4 - chạy render.py trước.")
    tiles = []
    for k in np.linspace(0, total - 1, a.n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
        cv2.rectangle(img, (0, img.shape[0] - 26), (200, img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"#{k}  {k / fps:.1f}s", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = C.RESULTS_DIR / "_qa_sheet.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"Đã tạo {out}. Mở ảnh và chấm theo bảng trong GUIDELINE.md mục A0.")


if __name__ == "__main__":
    main()
