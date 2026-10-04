"""
viz.py - Hàm vẽ biểu đồ cho web. Chỉ dùng pandas, numpy, plotly (KHÔNG import cv2).
Chạy thử:  python viz.py   -> tạo results/_heatmap_team0.html và results/_timeline.html
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def rgb_str(c):
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def make_heatmap(tracks_df, team, width, height, bins=(32, 18), name=None):
    """Heatmap vị trí điểm chân của các cầu thủ thuộc đội `team` (toạ độ pixel)."""
    p = tracks_df[(tracks_df["cls"] == "player") & (tracks_df["team"] == team)]
    hist, xe, ye = np.histogram2d(p["x"], p["y"], bins=bins,
                                  range=[[0, width], [0, height]])
    fig = go.Figure(go.Heatmap(
        z=hist.T, x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2,
        colorscale="YlOrRd", showscale=False,
        hovertemplate="x=%{x:.0f}, y=%{y:.0f}<br>số lần: %{z:.0f}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, width], constrain="domain")
    fig.update_yaxes(visible=False, range=[height, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="#2e7d32",
                      title=f"Vị trí {name or f'Team {team}'} (theo khung hình)")
    return fig


def make_possession_timeline(frames_df, team_colors, names=None):
    """Đường % kiểm soát bóng tích luỹ của 2 đội theo thời gian."""
    names = names or ("Team 0", "Team 1")
    fig = go.Figure()
    for t in (0, 1):
        fig.add_trace(go.Scatter(x=frames_df["time_s"], y=frames_df[f"cum_pct_team{t}"],
                                 mode="lines", name=names[t],
                                 line=dict(color=rgb_str(team_colors[t]), width=3)))
    fig.update_layout(xaxis_title="Thời gian (giây)", yaxis_title="% kiểm soát tích luỹ",
                      yaxis_range=[0, 100], margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    return fig


def make_possession_donut(possession_pct, team_colors, names=None):
    names = names or ("Team 0", "Team 1")
    fig = go.Figure(go.Pie(labels=list(names), values=possession_pct, hole=0.55,
                           marker=dict(colors=[rgb_str(c) for c in team_colors],
                                       line=dict(color="#333", width=1)),
                           sort=False))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    return fig


if __name__ == "__main__":
    import json
    import config as C
    stats = json.loads(C.STATS_JSON.read_text(encoding="utf-8"))
    tracks = pd.read_csv(C.TRACKS_CSV)
    frames = pd.read_csv(C.FRAMES_CSV)
    make_heatmap(tracks, 0, stats["width"], stats["height"]).write_html(
        C.RESULTS_DIR / "_heatmap_team0.html")
    make_possession_timeline(frames, stats["team_colors"]).write_html(
        C.RESULTS_DIR / "_timeline.html")
    print("Đã ghi results/_heatmap_team0.html và results/_timeline.html")
