"""
web_pipeline.py - Chạy pipeline cho video người dùng upload trên web.
Mỗi lần upload có thư mục tạm riêng (không ghi đè results/ của bản demo).
Các bước: chuẩn hoá video (cắt ≤ MAX_SECONDS, ≤ 720p, giảm fps) → detect_web → analytics → render.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 15
MAX_HEIGHT = 720


def _run(cmd, env, on_line=None):
    proc = subprocess.Popen([sys.executable] + cmd, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = []
    for line in proc.stdout:
        log.append(line.rstrip())
        if on_line:
            on_line(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError("\n".join(log[-15:]))
    return log


def prepare_video(src, dst, start_s=0.0, seconds=MAX_SECONDS, fps=12):
    """Cắt đoạn, thu nhỏ về ≤720p, giảm fps để xử lý nhanh trên CPU yếu."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"scale=-2:'min({MAX_HEIGHT},ih)',fps={fps}"
    cmd = [ff, "-y", "-loglevel", "error", "-ss", str(start_s), "-i", str(src), "-t", str(seconds),
           "-an", "-vf", vf, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)]
    subprocess.run(cmd, check=True)


def run_uploaded(video_file, suffix=".mp4", start_s=0.0, seconds=MAX_SECONDS, fps=12,
                 conf=0.1, on_progress=None):
    """Trả về đường dẫn thư mục kết quả (có meta/raw/tracks/frames/stats/output.mp4).
    video_file: file-like (UploadedFile của Streamlit) - chép xuống đĩa theo khối 8 MB,
    không tạo thêm bản sao bytes trong RAM. Vẫn nhận bytes để chạy thử từ dòng lệnh."""
    if isinstance(video_file, (bytes, bytearray)):
        video_file = io.BytesIO(video_file)
    work = Path(tempfile.mkdtemp(prefix="fa_"))
    src, clip = work / f"upload{suffix}", work / "clip.mp4"
    video_file.seek(0)
    with open(src, "wb") as out:
        shutil.copyfileobj(video_file, out, length=8 * 1024 * 1024)
    report = on_progress or (lambda p, msg: None)

    report(0.03, "Chuẩn hoá video...")
    prepare_video(src, clip, start_s, seconds, fps)
    src.unlink(missing_ok=True)

    env = dict(os.environ, FA_RESULTS_DIR=str(work / "results"))

    def on_detect(line):
        if line.startswith("PROGRESS"):
            _, done, total = line.split()
            total = int(total) or 1
            report(0.05 + 0.75 * min(1.0, int(done) / total), f"Phát hiện & theo dõi: frame {done}/{total}")

    _run(["detect_web.py", "--video", str(clip), "--conf", str(conf)], env, on_detect)
    report(0.82, "Phân tích đội & kiểm soát bóng...")
    _run(["analytics.py"], env)
    report(0.90, "Vẽ video chú thích...")
    _run(["render.py", "--video", str(clip), "--max_width", "960", "--crf", "28"], env)
    report(1.0, "Hoàn tất")
    return work / "results"
