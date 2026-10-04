"""
check_outputs.py - Kiểm tra thư mục results/ có đúng hợp đồng dữ liệu không.
Thay cho "người review thứ hai": chạy sau MỖI bước của pipeline.

Chạy:  python check_outputs.py            (thiếu output.mp4 chỉ cảnh báo)
       python check_outputs.py --strict   (thiếu output.mp4 = FAIL, dùng trước khi commit)
Mã thoát: 0 nếu không có FAIL, 1 nếu có.
"""
import argparse
import json
import sys

import pandas as pd

import config as C

results = []   # (trạng thái, tên kiểm tra, chi tiết)


def check(name, ok, detail="", warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    results.append((status, name, detail))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    # 1. File tồn tại
    files = [C.META_JSON, C.TRACKS_CSV, C.FRAMES_CSV, C.STATS_JSON]
    for f in files:
        check(f"Tồn tại {f.name}", f.exists())
    if not all(f.exists() for f in files):
        report()
        return

    meta = json.loads(C.META_JSON.read_text(encoding="utf-8"))
    tracks = pd.read_csv(C.TRACKS_CSV)
    frames = pd.read_csv(C.FRAMES_CSV)
    stats = json.loads(C.STATS_JSON.read_text(encoding="utf-8"))

    # 2. Đúng cột / khoá
    check("meta.json đủ khoá", all(k in meta for k in ["fps", "width", "height", "n_frames"]))
    check("tracks.csv đúng cột", list(tracks.columns) == C.TRACKS_COLS, str(list(tracks.columns)))
    check("frames.csv đúng cột", list(frames.columns) == C.FRAMES_COLS, str(list(frames.columns)))
    miss = [k for k in C.STATS_KEYS if k not in stats]
    check("stats.json đủ khoá", not miss, f"thiếu {miss}" if miss else "")
    if C.RAW_CSV.exists():
        raw_cols = list(pd.read_csv(C.RAW_CSV, nrows=1).columns)
        check("raw.csv đúng cột", raw_cols == C.RAW_COLS, str(raw_cols))

    # 3. Không NaN ở cột bắt buộc
    req_tracks = ["frame", "track_id", "cls", "team", "x1", "y1", "x2", "y2", "x", "y", "has_ball"]
    check("tracks.csv không NaN", not tracks[req_tracks].isna().any().any())
    req_frames = ["frame", "time_s", "holder_id", "possession_team",
                  "cum_pct_team0", "cum_pct_team1"]
    check("frames.csv không NaN (trừ ball_x/ball_y)", not frames[req_frames].isna().any().any())

    # 4. Logic
    n = int(meta["n_frames"])
    check("frames.csv có đúng n_frames dòng", len(frames) == n, f"{len(frames)} vs {n}")
    balls = tracks[tracks["cls"] == "ball"]
    check("Tối đa 1 bóng/frame", balls.groupby("frame").size().max() <= 1 if len(balls) else True)
    check("Bóng có track_id=-1, team=-1",
          ((balls["track_id"] == -1) & (balls["team"] == -1)).all() if len(balls) else True)
    pl = tracks[tracks["cls"] == "player"]
    check("Team của player chỉ là 0/1", pl["team"].isin([0, 1]).all())
    check("cls chỉ là player/ball", tracks["cls"].isin(["player", "ball"]).all())
    hb = pl.groupby("frame")["has_ball"].sum()
    check("has_ball tối đa 1 người/frame", (hb <= 1).all() if len(hb) else True)
    check("possession_team thuộc {-1,0,1}", frames["possession_team"].isin([-1, 0, 1]).all())
    cp = frames[["cum_pct_team0", "cum_pct_team1"]]
    check("cum_pct trong 0-100", ((cp >= 0) & (cp <= 100)).all().all())
    s = sum(stats["possession_pct"])
    check("Tổng possession_pct ~ 100", abs(s - 100) <= 0.5, f"tổng = {s:.1f}")
    check("Mỗi đội có 6-11 cầu thủ (tham khảo)",
          all(5 <= k <= 15 for k in stats["n_players"]), str(stats["n_players"]), warn_only=True)
    check("Phát hiện bóng >= 40% (tham khảo)", stats["ball_detected_pct"] >= 40,
          f"{stats['ball_detected_pct']}%", warn_only=True)

    # 5. Video
    if C.OUTPUT_MP4.exists():
        mb = C.OUTPUT_MP4.stat().st_size / 1e6
        check(f"output.mp4 < {C.MAX_OUTPUT_MB} MB", mb < C.MAX_OUTPUT_MB, f"{mb:.1f} MB",
              warn_only=True)
    else:
        check("Tồn tại output.mp4", False, "chưa chạy render.py", warn_only=not args.strict)

    # Số liệu tóm tắt
    print(f"Số frame: {n} | Cầu thủ mỗi đội: {stats['n_players']} | "
          f"Phát hiện bóng: {stats['ball_detected_pct']}% | "
          f"Possession: {stats['possession_pct']} | Đổi quyền: {stats['possession_changes']}")
    report()


def report():
    print("-" * 72)
    for status, name, detail in results:
        print(f"[{status}] {name}" + (f"  ({detail})" if detail and status != "PASS" else ""))
    n_fail = sum(1 for r in results if r[0] == "FAIL")
    n_warn = sum(1 for r in results if r[0] == "WARN")
    print("-" * 72)
    print(f"Kết quả: {len(results) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
