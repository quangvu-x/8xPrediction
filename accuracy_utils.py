"""
accuracy_utils.py - Các hàm tăng độ chính xác, gắn vào analytics.py (xem GUIDELINE.md, mục A).
Chỉ dùng numpy, pandas, scikit-learn → chạy được cả local lẫn web.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ------------------------------------------------------------------ bóng
def filter_ball_outliers(ball, width, fps, max_speed_ratio=1.5):
    """
    Loại các điểm "bóng" nhảy vọt rồi quay lại (spike) - thường là đầu, giày, vạch sân bị nhận nhầm.
    ball: DataFrame index = frame, cột bx, by (tâm bóng). Trả về DataFrame đã lọc.
    Một điểm bị loại khi nó cách CẢ điểm trước LẪN điểm sau quá xa
    (tốc độ > max_speed_ratio x chiều rộng video mỗi giây). Không lan lỗi sang các điểm khác.
    """
    if len(ball) < 3:
        return ball
    thr = max_speed_ratio * width / fps                   # px tối đa mỗi frame
    f = ball.index.values.astype(float)
    xy = ball[["bx", "by"]].values.astype(float)
    step = np.hypot(*(xy[1:] - xy[:-1]).T) / np.maximum(np.diff(f), 1)   # tốc độ giữa 2 điểm liền kề
    fast_prev = np.r_[False, step > thr]
    fast_next = np.r_[step > thr, False]
    return ball[~(fast_prev & fast_next)]


# ------------------------------------------------------------------ màu áo
def rgb_to_lab(rgb):
    """Đổi mảng (N,3) RGB 0-255 sang CIE Lab (D65). Lab gần với cảm nhận màu của mắt hơn RGB."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def assign_teams_lab(players, min_track_frames=5, light_weight=0.5):
    """
    Thay thế trực tiếp assign_teams(players) trong analytics.py: cùng đầu vào/đầu ra
    (team_of: dict track_id -> 0/1, team_colors: [[r,g,b],[r,g,b]]).
    Khác bản gốc: (1) phân cụm trong không gian Lab, giảm trọng số độ sáng L để bóng râm
    ít ảnh hưởng; (2) team_colors = trung vị RGB thật của các cầu thủ mỗi đội.
    """
    colored = players.dropna(subset=["r", "g", "b"])
    agg = colored.groupby("track_id").agg(r=("r", "median"), g=("g", "median"),
                                          b=("b", "median"), n=("frame", "count"))
    lab = rgb_to_lab(agg[["r", "g", "b"]].values) * np.array([light_weight, 1.0, 1.0])
    fit_mask = (agg["n"] >= min_track_frames).values
    if fit_mask.sum() < 2:
        fit_mask[:] = True
    if fit_mask.sum() < 2:
        raise ValueError("Có ít hơn 2 cầu thủ có màu áo - không chia được đội.")
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(lab[fit_mask])
    labels = km.predict(lab)
    rgb_med = [np.median(agg[["r", "g", "b"]].values[labels == k], axis=0) for k in (0, 1)]
    order = np.argsort([-m.sum() for m in rgb_med])   # đội áo sáng hơn = team 0
    remap = {int(order[0]): 0, int(order[1]): 1}
    team_of = {int(t): remap[int(l)] for t, l in zip(agg.index, labels)}
    team_colors = [np.round(rgb_med[order[i]]).astype(int).clip(0, 255).tolist() for i in (0, 1)]
    return team_of, team_colors


# ------------------------------------------------------------------ loại người không phải cầu thủ
def assign_teams_robust(players, n_clusters=4, min_track_frames=5, light_weight=1.0,
                        outlier_factor=2.5, exclude_ids=()):
    """
    Chia đội và LOẠI người không phải cầu thủ (ban huấn luyện, trọng tài, khán giả...).
    Cách làm:
      1. Gom màu áo (Lab, mặc định GIỮ độ sáng: light_weight=1.0) thành n_clusters cụm.
      2. Hai cụm có TỔNG số frame xuất hiện lớn nhất = 2 đội (cầu thủ có mặt nhiều nhất).
         Các cụm còn lại (vest tối, áo trọng tài...) bị loại.
      3. Trong mỗi đội, track cách tâm cụm > outlier_factor x khoảng cách trung vị cũng bị loại.
      4. exclude_ids: danh sách ID loại thủ công (dùng cho clip demo cố định).
    Trả về (team_of, team_colors, excluded): team_of chỉ chứa track thuộc 2 đội;
    excluded = {track_id: lý do}. Track không có trong team_of sẽ bị bỏ khỏi tracks.csv.
    """
    colored = players.dropna(subset=["r", "g", "b"])
    agg = colored.groupby("track_id").agg(r=("r", "median"), g=("g", "median"),
                                          b=("b", "median"), n=("frame", "count"))
    excluded = {int(t): "loại thủ công" for t in exclude_ids if t in agg.index}
    agg = agg.drop(index=[t for t in exclude_ids if t in agg.index])

    lab = rgb_to_lab(agg[["r", "g", "b"]].values) * np.array([light_weight, 1.0, 1.0])
    fit = (agg["n"] >= min_track_frames).values
    if fit.sum() < n_clusters:
        fit[:] = True
    k = int(min(n_clusters, fit.sum()))
    if k < 2:
        raise ValueError("Có ít hơn 2 cầu thủ có màu áo - không chia được đội.")
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(lab[fit])
    labels = km.predict(lab)
    dist = np.linalg.norm(lab - km.cluster_centers_[labels], axis=1)

    # 2 cụm "nặng" nhất theo số frame = 2 đội
    weight = pd.Series(agg["n"].values).groupby(labels).sum()
    teams_cl = weight.sort_values(ascending=False).index[:2].tolist()

    team_of, members = {}, {0: [], 1: []}
    rgb_vals = agg[["r", "g", "b"]].values
    for cl_rank, cl in enumerate(teams_cl):
        idx = np.where(labels == cl)[0]
        ref = np.median(dist[idx[fit[idx]]]) if fit[idx].any() else np.median(dist[idx])
        for i in idx:
            tid = int(agg.index[i])
            if dist[i] > outlier_factor * max(ref, 1.0):
                excluded[tid] = "màu lệch xa tâm đội"
            else:
                members[cl_rank].append(i)
                team_of[tid] = cl_rank
    for i in np.where(~np.isin(labels, teams_cl))[0]:
        excluded[int(agg.index[i])] = "không thuộc 2 nhóm màu chính"

    med = [np.median(rgb_vals[members[t]], axis=0) for t in (0, 1)]
    if med[1].sum() > med[0].sum():                 # đội áo sáng hơn = team 0 (giữ quy ước cũ)
        team_of = {t: 1 - v for t, v in team_of.items()}
        med = med[::-1]
    team_colors = [np.round(m).astype(int).clip(0, 255).tolist() for m in med]
    return team_of, team_colors, excluded
