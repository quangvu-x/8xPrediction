"""
run_pipeline.py - Chạy toàn bộ pipeline bằng MỘT lệnh, dừng ngay khi có bước lỗi.
Chạy:  python run_pipeline.py --video input/clip3s.mp4           (thử nhanh)
       python run_pipeline.py --video input/clip.mp4             (bản thật)
       python run_pipeline.py --video input/clip.mp4 --skip-detect  (chỉ chỉnh analytics/render)
Tham số YOLO (--model, --imgsz, --conf) được chuyển tiếp cho detect.py.
"""
import argparse
import subprocess
import sys
import time

import config as C


def run(step, cmd):
    print(f"\n========== {step} ==========")
    t0 = time.time()
    rc = subprocess.run([sys.executable] + cmd, cwd=C.ROOT).returncode
    if rc != 0:
        sys.exit(f"\n[DỪNG] Bước '{step}' lỗi (mã {rc}). Sửa lỗi rồi chạy lại "
                 f"(dùng --skip-detect nếu detect đã xong).")
    print(f"-> {step} xong sau {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(C.DEFAULT_VIDEO))
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--imgsz")
    ap.add_argument("--conf")
    a = ap.parse_args()

    if not a.skip_detect:
        cmd = ["detect.py", "--video", a.video]
        for k in ("model", "imgsz", "conf"):
            if getattr(a, k):
                cmd += [f"--{k}", getattr(a, k)]
        run("1/4 detect", cmd)
    elif not C.RAW_CSV.exists():
        sys.exit("[LỖI] --skip-detect nhưng chưa có results/raw.csv.")
    run("2/4 analytics", ["analytics.py"])
    run("3/4 render", ["render.py", "--video", a.video])
    run("4/4 check_outputs", ["check_outputs.py", "--strict"])
    print("\nHOÀN TẤT. Xem kết quả: streamlit run app.py")


if __name__ == "__main__":
    main()
