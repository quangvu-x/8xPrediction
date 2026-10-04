"""
make_mock.py - Sinh dữ liệu GIẢ nhưng đúng hợp đồng để làm web/deploy trước khi YOLO chạy xong.

Cách làm: vẽ một video tổng hợp (sân xanh, 2 đội x 8 cầu thủ, 1 quả bóng) ->
ghi input/mock_clip.mp4 + results/raw.csv + results/meta.json (bỏ qua YOLO) ->
chạy analytics.py và render.py thật. Nhờ vậy dữ liệu giả đi qua đúng code thật.

Chạy:  python make_mock.py
CHÚ Ý: lệnh này GHI ĐÈ thư mục results/. Chạy pipeline thật sau đó sẽ ghi đè lại.
"""
import json
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

import config as C
from detect import jersey_color

FPS, W, H, N = 25, 1280, 720, 375
N_PER_TEAM = 8
SHIRTS = [(240, 240, 240), (200, 30, 40)]   # RGB: đội áo trắng, đội áo đỏ
MOCK_VIDEO = C.INPUT_DIR / "mock_clip.mp4"
rng = np.random.default_rng(42)


def player_paths():
    """Quỹ đạo mượt (tổng các sóng sin) cho 16 cầu thủ, shape (16, N, 2)."""
    t = np.arange(N) / FPS
    paths = []
    for i in range(2 * N_PER_TEAM):
        team = i // N_PER_TEAM
        base_x = (0.15 + 0.35 * team) * W + rng.uniform(0, 0.35 * W)
        base_y = rng.uniform(0.25 * H, 0.9 * H)
        ax, ay = rng.uniform(40, 120), rng.uniform(20, 60)
        fx, fy, ph = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25), rng.uniform(0, 6.28)
        x = base_x + ax * np.sin(2 * np.pi * fx * t + ph)
        y = base_y + ay * np.cos(2 * np.pi * fy * t + ph)
        paths.append(np.stack([x, y], axis=1))
    return np.array(paths)


def ball_path(paths):
    """Bóng dính chân người giữ ~1.2 giây rồi chuyền sang người khác trong 10 frame."""
    pos = np.zeros((N, 2))
    holder = rng.integers(0, 2 * N_PER_TEAM)
    f = 0
    while f < N:
        hold = int(rng.integers(20, 40))
        for k in range(f, min(N, f + hold)):
            pos[k] = paths[holder, k] + [12, -4]
        f += hold
        nxt = int(rng.integers(0, 2 * N_PER_TEAM))
        for j, k in enumerate(range(f, min(N, f + 10))):
            a = (j + 1) / 10
            pos[k] = (1 - a) * (paths[holder, k] + [12, -4]) + a * (paths[nxt, k] + [12, -4])
        f += 10
        holder = nxt
    return pos


def draw_frame(paths, ball, k):
    img = np.full((H, W, 3), (40, 140, 60), np.uint8)         # BGR: cỏ xanh
    for x in range(0, W, 160):
        cv2.rectangle(img, (x, 0), (x + 80, H), (45, 150, 66), cv2.FILLED)
    cv2.line(img, (W // 2, 0), (W // 2, H), (230, 230, 230), 3)
    boxes = []
    for i, (cx, cy) in enumerate(paths[:, k]):
        team = i // N_PER_TEAM
        w, h = 30, 80
        x1, y1, x2, y2 = cx - w / 2, cy - h, cx + w / 2, cy
        r, g, b = SHIRTS[team]
        cv2.rectangle(img, (int(x1), int(y1 + 0.15 * h)), (int(x2), int(y1 + 0.55 * h)),
                      (b, g, r), cv2.FILLED)                               # áo
        cv2.rectangle(img, (int(x1), int(y1 + 0.55 * h)), (int(x2), int(y2)), (30, 30, 30),
                      cv2.FILLED)                                          # quần
        cv2.circle(img, (int(cx), int(y1 + 0.08 * h)), 8, (150, 180, 220), cv2.FILLED)
        boxes.append((x1, y1, x2, y2))
    cv2.circle(img, (int(ball[k, 0]), int(ball[k, 1])), 6, (255, 255, 255), cv2.FILLED)
    return img, boxes


def main():
    C.INPUT_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    paths = player_paths()
    ball = ball_path(paths)
    # Mất bóng ngẫu nhiên ~30% frame (các đoạn ngắn) để thử nội suy
    lost = np.zeros(N, bool)
    for s in rng.integers(0, N, 25):
        lost[s:s + int(rng.integers(2, 8))] = True

    writer = cv2.VideoWriter(str(MOCK_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []
    for k in range(N):
        img, boxes = draw_frame(paths, ball, k)
        writer.write(img)
        for tid, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            r, g, b = jersey_color(img, x1, y1, x2, y2)
            rows.append([k, tid, "player", 0.9, round(x1, 1), round(y1, 1), round(x2, 1),
                         round(y2, 1), r, g, b])
        if not lost[k]:
            bx, by = ball[k]
            rows.append([k, -1, "ball", 0.5, round(bx - 6, 1), round(by - 6, 1),
                         round(bx + 6, 1), round(by + 6, 1), np.nan, np.nan, np.nan])
    writer.release()

    pd.DataFrame(rows, columns=C.RAW_COLS).to_csv(C.RAW_CSV, index=False)
    C.META_JSON.write_text(json.dumps({"fps": float(FPS), "width": W, "height": H,
                                       "n_frames": N}, indent=2), encoding="utf-8")
    print(f"Đã tạo {MOCK_VIDEO}, {C.RAW_CSV}, {C.META_JSON}")

    py = sys.executable
    subprocess.run([py, "analytics.py"], check=True, cwd=C.ROOT)
    subprocess.run([py, "render.py", "--video", str(MOCK_VIDEO)], check=True, cwd=C.ROOT)
    print("\nDữ liệu giả đã sẵn sàng trong results/. Chạy: streamlit run app.py")


if __name__ == "__main__":
    main()
