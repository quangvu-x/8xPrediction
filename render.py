"""
render.py - Bước 3: video gốc + tracks.csv + frames.csv + stats.json -> results/output.mp4
Video H.264 (libx264, yuv420p) để trình duyệt/Streamlit phát được.

Chạy:  python render.py --video input/clip.mp4 [--max_width 1280] [--crf 26]
"""
import argparse
import json
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

import config as C


def parse_args():
    p = argparse.ArgumentParser(description="Vẽ chú thích và xuất video H.264")
    p.add_argument("--video", default=str(C.DEFAULT_VIDEO))
    p.add_argument("--max_width", type=int, default=C.MAX_WIDTH)
    p.add_argument("--crf", type=int, default=C.CRF)
    return p.parse_args()


def load_results():
    for p in (C.TRACKS_CSV, C.FRAMES_CSV, C.STATS_JSON):
        if not p.exists():
            sys.exit(f"[LỖI] Thiếu {p}. Hãy chạy analytics.py trước.")
    tracks = pd.read_csv(C.TRACKS_CSV)
    frames = pd.read_csv(C.FRAMES_CSV).set_index("frame")
    stats = json.loads(C.STATS_JSON.read_text(encoding="utf-8"))
    return tracks, frames, stats


def rgb2bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def draw_triangle(img, x, y_tip, color, size):
    """Tam giác chĩa xuống, đỉnh tại (x, y_tip)."""
    pts = np.array([[x, y_tip], [x - size, y_tip - 2 * size], [x + size, y_tip - 2 * size]],
                   dtype=np.int32)
    cv2.drawContours(img, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(img, [pts], 0, (0, 0, 0), 1)


def draw_player(img, row, color, scale):
    x1, y1, x2, y2 = row.x1, row.y1, row.x2, row.y2
    w = max(1.0, x2 - x1)
    cx, cy = int((x1 + x2) / 2), int(y2)
    cv2.ellipse(img, (cx, cy), (int(w * 0.6), max(2, int(w * 0.6 * 0.35))), 0,
                -45, 235, color, max(2, int(2 * scale)), cv2.LINE_AA)
    # nhãn ID
    label = str(int(row.track_id))
    fs = 0.45 * scale
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    bx1, by1 = cx - tw // 2 - 3, cy + int(8 * scale)
    cv2.rectangle(img, (bx1, by1), (bx1 + tw + 6, by1 + th + 6), color, cv2.FILLED)
    txt_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(img, label, (bx1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                txt_color, 1, cv2.LINE_AA)
    if row.has_ball == 1:
        draw_triangle(img, cx, int(y1) - 4, (0, 0, 255), int(7 * scale))


def draw_panel(img, pct0, pct1, colors_bgr, scale):
    """Bảng nền mờ góc trên trái: % kiểm soát bóng tích luỹ."""
    pw, ph = int(330 * scale), int(78 * scale)
    x0, y0 = int(15 * scale), int(15 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    fs = 0.6 * scale
    for i, (pct, col) in enumerate(((pct0, colors_bgr[0]), (pct1, colors_bgr[1]))):
        yy = y0 + int((28 + 32 * i) * scale)
        sq = int(16 * scale)
        cv2.rectangle(img, (x0 + 10, yy - sq), (x0 + 10 + sq, yy), col, cv2.FILLED)
        cv2.rectangle(img, (x0 + 10, yy - sq), (x0 + 10 + sq, yy), (0, 0, 0), 1)
        cv2.putText(img, f"Team {i}: {pct:5.1f}% ball control", (x0 + 20 + sq, yy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)


def encode_h264(src, dst, crf):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-preset", "veryfast", "-movflags", "+faststart", "-an", str(dst)]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    tracks, frames, stats = load_results()
    colors_bgr = [rgb2bgr(c) for c in stats["team_colors"]]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[LỖI] Không mở được video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or stats["fps"]
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w0, h0) != (stats["width"], stats["height"]):
        print("[CẢNH BÁO] Kích thước video khác stats.json - có thể bạn đang dùng nhầm clip.")

    ratio = min(1.0, args.max_width / w0)
    out_w, out_h = int(w0 * ratio) // 2 * 2, int(h0 * ratio) // 2 * 2   # H.264 cần số chẵn
    scale = max(0.6, w0 / 1280)          # cỡ nét/chữ theo độ phân giải gốc

    by_frame = {f: g for f, g in tracks.groupby("frame")}   # dựng sẵn 1 lần
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(C.RAW_RENDER_MP4), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        g = by_frame.get(idx)
        if g is not None:
            for row in g[g["cls"] == "player"].itertuples():
                draw_player(img, row, colors_bgr[int(row.team)], scale)
            for row in g[g["cls"] == "ball"].itertuples():
                draw_triangle(img, int(row.x), int(row.y) - int(8 * scale), (0, 215, 255),
                              int(7 * scale))
        if idx in frames.index:
            fr = frames.loc[idx]
            draw_panel(img, fr["cum_pct_team0"], fr["cum_pct_team1"], colors_bgr, scale)
        if (out_w, out_h) != (w0, h0):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(img)
        idx += 1
        if idx % 50 == 0:
            print(f"  Đã vẽ {idx} frame")
    cap.release()
    writer.release()

    print("Đang encode H.264...")
    encode_h264(C.RAW_RENDER_MP4, C.OUTPUT_MP4, args.crf)
    C.RAW_RENDER_MP4.unlink(missing_ok=True)

    size_mb = C.OUTPUT_MP4.stat().st_size / 1e6
    print(f"XONG: {C.OUTPUT_MP4} - {idx} frame, {out_w}x{out_h}, {size_mb:.1f} MB")
    if size_mb > C.MAX_OUTPUT_MB:
        print(f"[CẢNH BÁO] File > {C.MAX_OUTPUT_MB} MB: tăng --crf (28-30) "
              f"hoặc giảm --max_width (960).")


if __name__ == "__main__":
    main()
