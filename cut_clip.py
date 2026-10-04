"""
cut_clip.py - Cắt clip chính (15s) và clip thử (3s) bằng ffmpeg đi kèm imageio-ffmpeg.
Chạy:  python cut_clip.py --src input/full.mp4 --start 00:00:05 --dur 15
Kết quả: input/clip.mp4 và input/clip3s.mp4 (3 giây đầu của cùng đoạn).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

import config as C


def cut(src, start, dur, out):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Encode lại (không -c copy) để cắt chính xác từng frame và ra H.264 chuẩn
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", start, "-i", str(src), "-t", str(dur),
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
    subprocess.run(cmd, check=True)
    print(f"Đã tạo {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(C.INPUT_DIR / "full.mp4"))
    ap.add_argument("--start", default="00:00:05", help="thời điểm bắt đầu hh:mm:ss")
    ap.add_argument("--dur", type=float, default=15)
    a = ap.parse_args()
    if not Path(a.src).exists():
        sys.exit(f"[LỖI] Không thấy {a.src}. Đặt video gốc vào input/full.mp4.")
    C.INPUT_DIR.mkdir(exist_ok=True)
    cut(a.src, a.start, a.dur, C.DEFAULT_VIDEO)
    cut(a.src, a.start, 3, C.TEST_VIDEO)


if __name__ == "__main__":
    main()
