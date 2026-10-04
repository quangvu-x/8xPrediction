"""
analytics.py - Bước 2: raw.csv + meta.json -> tracks.csv + frames.csv + stats.json
Chỉ dùng pandas, numpy, scikit-learn (KHÔNG dùng cv2/ultralytics).

Chạy:  python analytics.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config as C
from accuracy_utils import assign_teams_robust, filter_ball_outliers


# ---------------------------------------------------------------- đọc dữ liệu
def load_inputs():
    for p in (C.RAW_CSV, C.META_JSON):
        if not p.exists():
            sys.exit(f"[LỖI] Thiếu {p}. Hãy chạy detect.py trước (hoặc make_mock.py để thử).")
    raw = pd.read_csv(C.RAW_CSV)
    missing = [c for c in C.RAW_COLS if c not in raw.columns]
    if missing:
        sys.exit(f"[LỖI] raw.csv thiếu cột {missing} - sai hợp đồng dữ liệu.")
    meta = json.loads(C.META_JSON.read_text(encoding="utf-8"))
    return raw, meta


# ---------------------------------------------------------------- bóng
def interpolate_ball(raw, meta):
    """Trả về (DataFrame index=frame, cột ball_x/ball_y; ball_detected_pct)."""
    n, fps = int(meta["n_frames"]), float(meta["fps"])
    ball = raw[raw["cls"] == "ball"].copy()
    ball["bx"] = (ball["x1"] + ball["x2"]) / 2
    ball["by"] = (ball["y1"] + ball["y2"]) / 2
    ball = ball.groupby("frame")[["bx", "by"]].first()
    ball = filter_ball_outliers(ball, meta["width"], meta["fps"], C.BALL_MAX_SPEED_RATIO)

    detected_pct = 100.0 * len(ball) / n if n else 0.0
    s = ball.reindex(range(n))

    # Chỉ nội suy khoảng trống "bên trong" và ngắn hơn hoặc bằng max_gap frame.
    max_gap = max(1, int(round(fps * C.BALL_INTERP_MAX_GAP_S)))
    isna = s["bx"].isna()
    gap_id = (isna != isna.shift()).cumsum()
    gap_len = isna.groupby(gap_id).transform("sum")
    too_long = isna & (gap_len > max_gap)

    s = s.interpolate(method="linear", limit_area="inside")
    s.loc[too_long, ["bx", "by"]] = np.nan
    s = s.rename(columns={"bx": "ball_x", "by": "ball_y"})
    s.index.name = "frame"
    return s, round(detected_pct, 1)


# ---------------------------------------------------------------- phân đội
def drop_referees(players):
    """Loại trọng tài: track có màu áo trung vị gần REFEREE_COLORS không được chia đội."""
    if not C.REFEREE_COLORS:
        return players
    med = players.dropna(subset=["r", "g", "b"]).groupby("track_id")[["r", "g", "b"]].median()
    ref = np.array(C.REFEREE_COLORS, dtype=float)
    d_ref = np.linalg.norm(med.values[:, None, :] - ref[None, :, :], axis=2)
    ref_ids = med.index[d_ref.min(axis=1) < C.REFEREE_COLOR_DIST].astype(int).tolist()
    if ref_ids:
        print(f"Loại trọng tài : track {ref_ids}")
    return players[~players["track_id"].isin(ref_ids)]


def assign_teams(players):
    """(Bản cũ, RGB) KMeans 2 cụm trên màu áo trung vị của từng track. Team 0 = áo sáng hơn."""
    players = drop_referees(players)
    colored = players.dropna(subset=["r", "g", "b"])
    agg = colored.groupby("track_id").agg(r=("r", "median"), g=("g", "median"),
                                          b=("b", "median"), n=("frame", "count"))

    fit = agg[agg["n"] >= C.MIN_TRACK_FRAMES]
    if len(fit) < 2:
        fit = agg
    if len(fit) < 2:
        sys.exit("[LỖI] Có ít hơn 2 cầu thủ có màu áo - không chia được đội. "
                 "Kiểm tra lại raw.csv hoặc chọn clip khác.")

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(fit[["r", "g", "b"]].values)
    centers = km.cluster_centers_
    order = np.argsort(-centers.sum(axis=1))      # cụm sáng hơn đứng trước
    rank = np.empty(2, dtype=int)
    rank[order] = [0, 1]                          # cluster gốc -> team

    feats = agg[["r", "g", "b"]].values
    dist = np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2)
    teams = rank[dist.argmin(axis=1)]
    team_of = dict(zip(agg.index.astype(int), teams.astype(int)))
    team_colors = np.round(centers[order]).astype(int).clip(0, 255).tolist()
    return team_of, team_colors


def print_excluded(players, excluded):
    """In các track bị loại khỏi 2 đội: track_id, số frame, màu RGB trung vị, lý do."""
    if not excluded:
        print("Loại khỏi đội  : không có")
        return
    info = players.groupby("track_id").agg(n=("frame", "nunique"), r=("r", "median"),
                                           g=("g", "median"), b=("b", "median"))
    print(f"Loại khỏi đội  : {len(excluded)} track")
    for tid in sorted(excluded, key=lambda t: -info.loc[t, "n"]):
        r = info.loc[tid]
        print(f"  track {tid:>4} | {int(r.n):>4} frame | RGB ({r.r:.0f}, {r.g:.0f}, {r.b:.0f})"
              f" | {excluded[tid]}")


# ---------------------------------------------------------------- giữ bóng
def find_holders(players, ball_pos, width):
    """holder_id cho từng frame (-1 nếu không ai đủ gần bóng)."""
    thr = C.HOLD_DIST_RATIO * width
    holder = pd.Series(-1, index=ball_pos.index, dtype=int, name="holder_id")

    bp = ball_pos.dropna().reset_index()
    if bp.empty or players.empty:
        return holder
    m = players[["frame", "track_id", "x", "y"]].merge(bp, on="frame", how="inner")
    if m.empty:
        return holder
    m["dist"] = np.hypot(m["x"] - m["ball_x"], m["y"] - m["ball_y"])
    best = m.loc[m.groupby("frame")["dist"].idxmin()]
    best = best[best["dist"] < thr]
    holder.loc[best["frame"].values] = best["track_id"].astype(int).values
    return holder


def compute_possession(holder, team_of):
    """possession_team theo frame + % tích luỹ + số lần đổi quyền kiểm soát."""
    holder_team = holder.map(lambda h: team_of.get(int(h), -1) if h != -1 else -1)

    # Carry-forward, có lọc nhiễu: đội mới phải giữ bóng N frame liên tiếp (N=1: tắt lọc)
    current, cand, cand_count = -1, -1, 0
    poss = []
    for t in holder_team.values:
        if t != -1 and t != current:
            if current == -1:
                current = t
            else:
                cand_count = cand_count + 1 if t == cand else 1
                cand = t
                if cand_count >= C.MIN_POSSESSION_FRAMES:
                    current, cand, cand_count = t, -1, 0
        elif t == current:
            cand, cand_count = -1, 0
        poss.append(current)
    poss = pd.Series(poss, index=holder.index, dtype=int)

    c0 = (poss == 0).cumsum()
    c1 = (poss == 1).cumsum()
    tot = (c0 + c1).replace(0, np.nan)
    cum0 = (100 * c0 / tot).fillna(0.0).round(2)
    cum1 = (100 * c1 / tot).fillna(0.0).round(2)

    valid = poss[poss != -1]
    changes = int((valid.diff().fillna(0) != 0).sum())
    return poss, cum0, cum1, changes


# ---------------------------------------------------------------- xuất file
def export_results(players, ball_pos, holder, poss, cum0, cum1, changes,
                   team_colors, detected_pct, meta):
    fps = float(meta["fps"])
    n = int(meta["n_frames"])

    # tracks.csv
    p = players.copy()
    p["has_ball"] = 0
    hold_pairs = set(zip(holder.index[holder != -1], holder[holder != -1]))
    if hold_pairs:
        p["has_ball"] = [1 if (f, t) in hold_pairs else 0
                         for f, t in zip(p["frame"], p["track_id"])]
    b = ball_pos.dropna().reset_index()
    h = C.BALL_BOX_HALF
    ball_rows = pd.DataFrame({
        "frame": b["frame"], "track_id": -1, "cls": "ball", "team": -1,
        "x1": b["ball_x"] - h, "y1": b["ball_y"] - h,
        "x2": b["ball_x"] + h, "y2": b["ball_y"] + h,
        "x": b["ball_x"], "y": b["ball_y"], "has_ball": 0,
    })
    tracks = pd.concat([p[C.TRACKS_COLS], ball_rows[C.TRACKS_COLS]], ignore_index=True)
    tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
    for col in ["x1", "y1", "x2", "y2", "x", "y"]:
        tracks[col] = tracks[col].round(1)
    tracks.to_csv(C.TRACKS_CSV, index=False)

    # frames.csv
    frames = pd.DataFrame({
        "frame": range(n),
        "time_s": (np.arange(n) / fps).round(3),
        "ball_x": ball_pos["ball_x"].round(1).values,
        "ball_y": ball_pos["ball_y"].round(1).values,
        "holder_id": holder.values,
        "possession_team": poss.values,
        "cum_pct_team0": cum0.values,
        "cum_pct_team1": cum1.values,
    })
    frames.to_csv(C.FRAMES_CSV, index=False)

    # stats.json
    per_player = (p.groupby("track_id")
                   .agg(team=("team", "first"), frames_seen=("frame", "nunique"),
                        frames_with_ball=("has_ball", "sum"))
                   .reset_index())
    # n_players = số người nhiều nhất có mặt CÙNG LÚC trong 1 frame (camera lia làm đổi ID,
    # nên không đếm số ID). Chỉ xét track đủ dài; track vụn vẫn có trong per_player.
    long_enough = per_player["frames_seen"] >= C.MIN_PLAYER_SECONDS * fps
    long_ids = per_player.loc[long_enough, "track_id"]
    on_screen = (p[p["track_id"].isin(long_ids)]
                 .groupby(["frame", "team"])["track_id"].nunique().unstack(fill_value=0))
    n_players = [int(on_screen[t].max()) if t in on_screen else 0 for t in (0, 1)]
    stats = {
        "fps": fps, "n_frames": n,
        "width": int(meta["width"]), "height": int(meta["height"]),
        "team_colors": team_colors,
        "possession_pct": [float(round(cum0.iloc[-1], 1)), float(round(cum1.iloc[-1], 1))],
        "n_players": n_players,
        "ball_detected_pct": float(detected_pct),
        "possession_changes": int(changes),
        "per_player": [
            {"track_id": int(r.track_id), "team": int(r.team),
             "frames_seen": int(r.frames_seen), "frames_with_ball": int(r.frames_with_ball)}
            for r in per_player.itertuples()
        ],
    }
    C.STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


# ---------------------------------------------------------------- main
def main():
    raw, meta = load_inputs()
    ball_pos, detected_pct = interpolate_ball(raw, meta)

    players = raw[raw["cls"] == "player"].copy()
    players["track_id"] = players["track_id"].astype(int)
    players = drop_referees(players)
    try:
        team_of, team_colors, excluded = assign_teams_robust(
            players, C.TEAM_CLUSTERS, C.MIN_TRACK_FRAMES, 1.0,
            C.TEAM_OUTLIER_FACTOR, C.EXCLUDE_TRACK_IDS)
    except ValueError:
        sys.exit("[LỖI] Có ít hơn 2 cầu thủ có màu áo - không chia được đội. "
                 "Kiểm tra lại raw.csv hoặc chọn clip khác.")
    # Thủ môn mặc áo khác màu đội -> gán đội thủ công, bỏ khỏi danh sách loại
    present = set(players["track_id"].unique())
    for tid, team in C.GOALKEEPER_TEAM.items():
        if tid in present:
            team_of[int(tid)] = int(team)
            excluded.pop(int(tid), None)
            print(f"Thủ môn        : track {tid} -> Team {team}")
    print_excluded(players, excluded)
    # Track không có trong team_of (người không phải cầu thủ) bị bỏ TRƯỚC khi tìm người giữ bóng
    players["team"] = players["track_id"].map(team_of)
    players = players.dropna(subset=["team"])
    players["team"] = players["team"].astype(int)
    players["x"] = (players["x1"] + players["x2"]) / 2     # điểm chân
    players["y"] = players["y2"]

    holder = find_holders(players, ball_pos, meta["width"])
    poss, cum0, cum1, changes = compute_possession(holder, team_of)
    stats = export_results(players, ball_pos, holder, poss, cum0, cum1, changes,
                           team_colors, detected_pct, meta)

    p0, p1 = stats["possession_pct"]
    print("===== KẾT QUẢ PHÂN TÍCH =====")
    print(f"Kiểm soát bóng : Team 0 {p0}% | Team 1 {p1}%")
    print(f"Màu đội (RGB)  : {team_colors}")
    print(f"Số cầu thủ (tối đa cùng lúc): Team 0 = {stats['n_players'][0]}, Team 1 = {stats['n_players'][1]}")
    print(f"Phát hiện bóng : {detected_pct}% số frame")
    print(f"Đổi quyền k.s. : {changes} lần")
    if p0 + p1 == 0:
        print("[CẢNH BÁO] Không frame nào xác định được người giữ bóng. "
              "Thử tăng HOLD_DIST_RATIO trong config.py hoặc cải thiện phát hiện bóng.")
    print(f"Đã ghi {C.TRACKS_CSV.name}, {C.FRAMES_CSV.name}, {C.STATS_JSON.name} vào results/")


if __name__ == "__main__":
    main()
