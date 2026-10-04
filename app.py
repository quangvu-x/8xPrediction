"""
app.py (v2) - Web Streamlit: dashboard trận mẫu + tự upload video để phân tích.
Upload dùng YOLOv8n ONNX (onnxruntime) nên chạy được trên Streamlit Community Cloud.
Chạy:  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config as C
import ui
import viz_extra as VX
from viz import make_heatmap, make_possession_donut

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")
ui.inject_css()
ONNX_MODEL = C.ROOT / "models" / "yolov8n.onnx"
NEEDED = ["stats.json", "tracks.csv", "frames.csv"]


# ------------------------------------------------------------------ dữ liệu
@st.cache_data(show_spinner=False)
def load_data(folder: str):
    d = Path(folder)
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    return stats, pd.read_csv(d / "tracks.csv"), pd.read_csv(d / "frames.csv")


def current_dir():
    if st.session_state.get("source") == "upload" and st.session_state.get("upload_dir"):
        return Path(st.session_state["upload_dir"])
    return C.RESULTS_DIR


def team_names():
    """Tên 2 đội hiển thị trên web (sidebar sửa được; ô trống -> "Team 0/1")."""
    return [st.session_state.get(f"team_name_{t}", "").strip() or f"Team {t}" for t in (0, 1)]


# ------------------------------------------------------------------ sidebar
def sidebar():
    with st.sidebar:
        st.markdown("### ⚽ Nguồn dữ liệu")
        options = ["Trận mẫu"] + (["Video của bạn"] if st.session_state.get("upload_dir") else [])
        choice = st.radio("Hiển thị", options, label_visibility="collapsed",
                          index=options.index("Video của bạn") if st.session_state.get("source") == "upload"
                          and "Video của bạn" in options else 0)
        st.session_state["source"] = "upload" if choice == "Video của bạn" else "demo"
        st.markdown("### 👕 Tên đội")
        for t in (0, 1):
            st.session_state.setdefault(f"team_name_{t}", C.TEAM_NAMES[t])
            st.text_input(f"Đội {t} ({'áo sáng hơn' if t == 0 else 'áo tối hơn'})", key=f"team_name_{t}")
        with st.expander("Cách hệ thống hoạt động"):
            st.markdown("1. **Phát hiện** cầu thủ & bóng (YOLO)\n2. **Theo dõi** ID qua từng frame\n"
                        "3. **Chia đội** theo màu áo (KMeans)\n4. **Giữ bóng** = cầu thủ gần bóng nhất\n"
                        "5. **Vẽ** video & biểu đồ")
        with st.expander("Hạn chế"):
            st.markdown("- Bóng nhỏ dễ bị bỏ sót\n- Trọng tài/thủ môn có thể bị tính như cầu thủ\n"
                        "- ID có thể đổi khi bị che\n- Toạ độ theo pixel, chưa quy ra mét\n"
                        "- Video upload dùng model nhỏ (YOLOv8n) nên kém chính xác hơn trận mẫu")


# ------------------------------------------------------------------ các tab
def tab_overview(stats, d):
    p0, p1 = stats["possession_pct"]
    names = team_names()
    ui.scoreboard(p0, p1, *stats["team_colors"], names=names)
    left, right = st.columns([3, 2], gap="large")
    with left:
        video = d / "output.mp4"
        st.video(video.read_bytes()) if video.exists() else st.warning("Chưa có output.mp4.")
    with right:
        a, b = st.columns(2)
        with a:
            ui.stat_card("Phát hiện bóng", f"{stats['ball_detected_pct']:.0f}%", "tỷ lệ frame thấy bóng", "🎯")
        with b:
            ui.stat_card("Đổi quyền", stats["possession_changes"], "số lần đổi đội giữ bóng", "🔁")
        a, b = st.columns(2)
        with a:
            ui.stat_card(f"Cầu thủ {names[0]}", stats["n_players"][0], "ID khác nhau", "👕")
        with b:
            ui.stat_card(f"Cầu thủ {names[1]}", stats["n_players"][1], "ID khác nhau", "👕")
        st.plotly_chart(make_possession_donut(stats["possession_pct"], stats["team_colors"], names),
                        width="stretch")
    with st.expander("⬇️ Tải dữ liệu"):
        c1, c2, c3 = st.columns(3)
        c1.download_button("stats.json", (d / "stats.json").read_bytes(), "stats.json")
        c2.download_button("tracks.csv", (d / "tracks.csv").read_bytes(), "tracks.csv")
        c3.download_button("frames.csv", (d / "frames.csv").read_bytes(), "frames.csv")


def tab_replay(stats, tracks, frames):
    ui.section("Phát lại chiến thuật", "🎬")
    st.caption("Bấm ▶ hoặc kéo thanh trượt. Viền vàng = cầu thủ đang giữ bóng. Rê chuột để xem ID.")
    step = st.select_slider("Độ mượt", options=[1, 2, 3, 5], value=2,
                            format_func=lambda s: {1: "Tối đa", 2: "Cao", 3: "Vừa", 5: "Nhẹ"}[s])
    st.plotly_chart(VX.make_replay(tracks, stats, frames, step=step, names=team_names()), width="stretch")


def tab_trend(stats, frames):
    ui.section("Momentum theo thời gian", "📊")
    win = st.slider("Độ dài mỗi cửa sổ (giây)", 1.0, 5.0, 2.0, 0.5)
    st.plotly_chart(VX.make_momentum(frames, stats["team_colors"], win, stats["fps"], team_names()), width="stretch")
    ui.section("% kiểm soát tích luỹ", "📈")
    st.plotly_chart(VX.make_timeline_with_events(frames, stats["team_colors"], team_names()), width="stretch")


def tab_players(stats, tracks):
    names = team_names()
    summary = VX.player_summary(tracks, stats["fps"])
    team = st.segmented_control("Đội", ["Tất cả", 0, 1], default="Tất cả",
                                format_func=lambda t: t if t == "Tất cả" else names[t])
    view = summary if team in (None, "Tất cả") else summary[summary["Đội"] == team]
    left, right = st.columns([2, 3], gap="large")
    with left:
        ui.section("Bảng cầu thủ", "👟")
        st.dataframe(view.assign(**{"Đội": view["Đội"].map(dict(enumerate(names)))})
                     .sort_values("giây giữ bóng", ascending=False), hide_index=True, width="stretch",
                     column_config={"giây giữ bóng": st.column_config.ProgressColumn(
                         "giây giữ bóng", min_value=0, max_value=float(max(summary["giây giữ bóng"].max(), 0.1)),
                         format="%.1f s")})
    with right:
        if len(view):
            pid = st.selectbox("Xem đường di chuyển của cầu thủ", view["ID"].tolist())
            st.plotly_chart(VX.make_player_trail(tracks, pid, stats), width="stretch")


def tab_heatmap(stats, tracks):
    names = team_names()
    team = st.radio("Chọn đội", [0, 1], format_func=lambda t: names[t], horizontal=True)
    st.plotly_chart(make_heatmap(tracks, team, stats["width"], stats["height"], name=names[team]),
                    width="stretch")
    st.caption("Camera di chuyển nên heatmap phản ánh vị trí trên khung hình, không phải sân thật.")


def tab_upload():
    ui.section("Tự phân tích video của bạn", "📤")
    if not ONNX_MODEL.exists():
        st.info("Chưa có models/yolov8n.onnx trong repo. Xem GUIDELINE.md mục B1 để tạo file này.")
        return
    st.caption("Video được cắt tối đa 15 giây, thu nhỏ ≤ 720p để chạy được trên máy chủ miễn phí. "
               "Không lưu video của bạn sau phiên làm việc.")
    f = st.file_uploader("Chọn video (mp4, mov, avi, mkv)", type=["mp4", "mov", "avi", "mkv"])
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Bắt đầu từ giây", 0.0, 3600.0, 0.0, 1.0)
    dur = c2.slider("Độ dài (giây)", 3, 15, 8)
    mode = c3.selectbox("Chế độ", ["⚡ Nhanh (8 fps)", "⚖️ Cân bằng (12 fps)", "🎯 Chi tiết (25 fps)"], index=1)
    fps = {"⚡": 8, "⚖": 12, "🎯": 25}[mode[0]]
    conf = st.slider("Ngưỡng tin cậy (thấp = bắt nhiều bóng hơn, nhiều nhầm hơn)", 0.05, 0.5, 0.10, 0.05)
    if f and st.button("🚀 Phân tích", type="primary", width="stretch"):
        from web_pipeline import run_uploaded          # import muộn: chỉ cần khi upload
        bar = st.progress(0.0, text="Bắt đầu...")
        try:
            out = run_uploaded(f.getvalue(), Path(f.name).suffix or ".mp4", start, dur, fps, conf,
                               on_progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
        except Exception as e:                         # hiện log để người dùng gửi lại khi cần
            st.error("Phân tích thất bại. Thử đoạn video khác, giảm độ dài hoặc chọn chế độ Nhanh.")
            with st.expander("Chi tiết lỗi"):
                st.code(str(e))
            return
        missing = [n for n in NEEDED if not (out / n).exists()]
        if missing:
            st.error(f"Thiếu kết quả: {missing}")
            return
        st.session_state.update(upload_dir=str(out), source="upload")
        st.balloons()
        st.rerun()


# ------------------------------------------------------------------ main
def main():
    sidebar()
    d = current_dir()
    is_upload = st.session_state.get("source") == "upload"
    ui.hero("FOOTBALL ANALYTICS",
            "Phát hiện cầu thủ và bóng bằng YOLO, chia đội theo màu áo, đo kiểm soát bóng và phát lại "
            "chiến thuật — ngay trên trình duyệt.",
            ["YOLOv8", "ByteTrack / IoU tracker", "KMeans", "Streamlit",
             "🎥 Video của bạn" if is_upload else "🏟️ Trận mẫu"])

    tabs = st.tabs(["🏟️ Tổng quan", "🎬 Phát lại", "📈 Diễn biến", "👟 Cầu thủ", "🔥 Heatmap", "📤 Tự phân tích"])
    with tabs[5]:
        tab_upload()

    missing = [n for n in NEEDED if not (d / n).exists()]
    if missing:
        with tabs[0]:
            st.error(f"Thiếu {', '.join(missing)} trong {d.name}/. Chạy run_pipeline.py hoặc make_mock.py.")
        return
    try:
        stats, tracks, frames = load_data(str(d))
    except Exception as e:
        with tabs[0]:
            st.error(f"Không đọc được dữ liệu: {e}")
        return

    with tabs[0]:
        tab_overview(stats, d)
    with tabs[1]:
        tab_replay(stats, tracks, frames)
    with tabs[2]:
        tab_trend(stats, frames)
    with tabs[3]:
        tab_players(stats, tracks)
    with tabs[4]:
        tab_heatmap(stats, tracks)
    st.caption("Tham chiếu: \"Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python\". "
               "Người thực hiện: [ĐIỀN TÊN].")


main()
