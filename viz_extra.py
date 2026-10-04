"""
viz_extra.py - Biểu đồ tương tác bổ sung cho web (pandas, numpy, plotly; KHÔNG cv2).
- make_replay: "phát lại chiến thuật" - chấm cầu thủ/bóng chạy theo thời gian, có nút Play và thanh trượt.
- make_momentum: % kiểm soát theo từng cửa sổ thời gian (biểu đồ cột hai chiều).
- make_timeline_with_events: đường % tích luỹ + vạch đánh dấu các lần đổi quyền kiểm soát.
- player_summary: bảng thống kê từng cầu thủ (thời gian xuất hiện, giữ bóng, quãng đường px).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PITCH = "#1f6f3f"


def rgb(c, a=1.0):
    return f"rgba({int(c[0])},{int(c[1])},{int(c[2])},{a})"


def _layout(fig, width, height, title=None):
    fig.update_xaxes(visible=False, range=[0, width], constrain="domain")
    fig.update_yaxes(visible=False, range=[height, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor=PITCH, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=40 if title else 0, b=0), title=title,
                      font=dict(color="#E8F5EC"), legend=dict(orientation="h", y=1.08))
    return fig


def make_replay(tracks, stats, frames_df=None, step=2, names=None):
    """Animation vị trí cầu thủ theo khung hình. step: lấy 1 frame mỗi `step` frame cho nhẹ."""
    names = names or ("Team 0", "Team 1")
    W, H, fps = stats["width"], stats["height"], stats["fps"]
    colors = stats["team_colors"]
    keep = sorted(tracks["frame"].unique())[::step]
    t = tracks[tracks["frame"].isin(keep)]
    groups = {f: g for f, g in t.groupby("frame")}

    def traces(f):
        g = groups.get(f, t.iloc[0:0])
        out = []
        for team in (0, 1):
            p = g[(g["cls"] == "player") & (g["team"] == team)]
            out.append(go.Scatter(
                x=p["x"], y=p["y"], mode="markers+text", name=names[team],
                text=p["track_id"].astype(str), textposition="top center",
                textfont=dict(size=9, color="#ffffff"),
                marker=dict(size=np.where(p["has_ball"] == 1, 18, 12), color=rgb(colors[team]),
                            line=dict(width=np.where(p["has_ball"] == 1, 3, 1),
                                      color=["#FFD60A" if h else "#111" for h in p["has_ball"]])),
                hovertemplate="ID %{text}<extra>" + names[team] + "</extra>"))
        b = g[g["cls"] == "ball"]
        out.append(go.Scatter(x=b["x"], y=b["y"], mode="markers", name="Bóng",
                              marker=dict(size=10, color="#ffffff", symbol="circle",
                                          line=dict(width=2, color="#FFD60A")),
                              hoverinfo="skip"))
        return out

    fig = go.Figure(data=traces(keep[0]),
                    frames=[go.Frame(data=traces(f), name=str(f)) for f in keep])
    ms = int(1000 * step / fps)
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.02, xanchor="left", yanchor="top",
                          buttons=[dict(label="▶ Phát", method="animate",
                                        args=[None, dict(frame=dict(duration=ms, redraw=True),
                                                         transition=dict(duration=0), fromcurrent=True)]),
                                   dict(label="⏸ Dừng", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(x=0.15, len=0.85, y=-0.02, currentvalue=dict(prefix="Giây: "),
                      steps=[dict(method="animate", label=f"{f / fps:.1f}",
                                  args=[[str(f)], dict(mode="immediate", frame=dict(duration=0))])
                             for f in keep])])
    return _layout(fig, W, H)


def make_momentum(frames_df, team_colors, window_s=2.0, fps=25.0, names=None):
    """Mỗi cột = % thời gian đội 0 (lên) / đội 1 (xuống) giữ bóng trong cửa sổ window_s giây."""
    names = names or ("Team 0", "Team 1")
    f = frames_df[frames_df["possession_team"] != -1].copy()
    if f.empty:
        return go.Figure()
    f["win"] = (f["time_s"] // window_s) * window_s
    g = f.groupby("win")["possession_team"].agg(lambda s: (s == 0).mean() * 100).reset_index()
    g.columns = ["win", "p0"]
    fig = go.Figure()
    fig.add_bar(x=g["win"], y=g["p0"], name=names[0], marker_color=rgb(team_colors[0]),
                hovertemplate="%{x:.0f}s: %{y:.0f}%<extra>" + names[0] + "</extra>")
    fig.add_bar(x=g["win"], y=-(100 - g["p0"]), name=names[1], marker_color=rgb(team_colors[1]),
                customdata=100 - g["p0"], hovertemplate="%{x:.0f}s: %{customdata:.0f}%<extra>" + names[1] + "</extra>")
    fig.update_layout(barmode="relative", bargap=0.15, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[-100, 100], title=f"← {names[1]} | {names[0]} →",
                                                               zeroline=True, zerolinecolor="#888"),
                      xaxis_title="Thời gian (giây)", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    return fig


def make_timeline_with_events(frames_df, team_colors, names=None):
    names = names or ("Team 0", "Team 1")
    fig = go.Figure()
    for t in (0, 1):
        fig.add_scatter(x=frames_df["time_s"], y=frames_df[f"cum_pct_team{t}"], mode="lines",
                        name=names[t], line=dict(color=rgb(team_colors[t]), width=3))
    pt = frames_df["possession_team"]
    changes = frames_df[(pt != pt.shift()) & (pt != -1) & (pt.shift() != -1)]
    for _, r in changes.iterrows():
        fig.add_vline(x=r["time_s"], line=dict(color="#FFD60A", width=1, dash="dot"))
    fig.update_layout(yaxis_range=[0, 100], yaxis_title="% kiểm soát tích luỹ",
                      xaxis_title="Thời gian (giây) — vạch vàng: đổi quyền kiểm soát",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    return fig


def player_summary(tracks, fps):
    """Thống kê từng cầu thủ; quãng đường tính theo pixel (đã làm mượt), chưa quy ra mét."""
    p = tracks[tracks["cls"] == "player"].sort_values(["track_id", "frame"]).copy()
    p["xs"] = p.groupby("track_id")["x"].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
    p["ys"] = p.groupby("track_id")["y"].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
    p["d"] = np.hypot(p.groupby("track_id")["xs"].diff(), p.groupby("track_id")["ys"].diff()).fillna(0)
    s = p.groupby("track_id").agg(team=("team", "first"), frames=("frame", "nunique"),
                                  ball=("has_ball", "sum"), dist=("d", "sum")).reset_index()
    s["giây xuất hiện"] = (s["frames"] / fps).round(1)
    s["giây giữ bóng"] = (s["ball"] / fps).round(1)
    s["quãng đường (px)"] = s["dist"].round(0).astype(int)
    return s.rename(columns={"track_id": "ID", "team": "Đội"})[
        ["ID", "Đội", "giây xuất hiện", "giây giữ bóng", "quãng đường (px)"]]


def make_player_trail(tracks, track_id, stats):
    p = tracks[(tracks["cls"] == "player") & (tracks["track_id"] == track_id)].sort_values("frame")
    team = int(p["team"].iloc[0]) if len(p) else 0
    fig = go.Figure(go.Scatter(x=p["x"], y=p["y"], mode="lines+markers",
                               line=dict(color=rgb(stats["team_colors"][team]), width=3),
                               marker=dict(size=np.where(p["has_ball"] == 1, 9, 3), color="#FFD60A"),
                               hovertemplate="frame %{customdata}<extra></extra>",
                               customdata=p["frame"]))
    return _layout(fig, stats["width"], stats["height"], f"Đường di chuyển cầu thủ #{track_id}")
