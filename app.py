"""
app.py - Web Streamlit: CHỈ đọc thư mục results/ và hiển thị. Không chạy YOLO.
Chỉ import streamlit, pandas, plotly, json, pathlib (+ config.py, viz.py thuần Python).
Chạy:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

import config as C
from viz import make_heatmap, make_possession_donut, make_possession_timeline

st.set_page_config(page_title="Phân tích bóng đá", page_icon="⚽", layout="wide")

REQUIRED = [C.STATS_JSON, C.TRACKS_CSV, C.FRAMES_CSV]


@st.cache_data
def load_data():
    stats = json.loads(C.STATS_JSON.read_text(encoding="utf-8"))
    tracks = pd.read_csv(C.TRACKS_CSV)
    frames = pd.read_csv(C.FRAMES_CSV)
    return stats, tracks, frames


def render_sidebar():
    with st.sidebar:
        st.header("Cách hệ thống hoạt động")
        st.markdown(
            "1. **detect.py** - YOLO phát hiện cầu thủ, bóng; ByteTrack gán ID.\n"
            "2. **analytics.py** - nội suy bóng, chia đội theo màu áo (KMeans), "
            "xác định người giữ bóng, tính % kiểm soát.\n"
            "3. **render.py** - vẽ chú thích, xuất video H.264.\n"
            "4. **app.py** - web này, chỉ đọc kết quả.")
        st.header("Hạn chế")
        st.markdown(
            "- Bóng nhỏ, model pretrained có thể bỏ sót.\n"
            "- Trọng tài/thủ môn có thể bị tính như cầu thủ.\n"
            "- ID có thể đổi khi cầu thủ bị che khuất.\n"
            "- Một clip ngắn không đại diện cho cả trận.\n"
            "- Toạ độ theo pixel, chưa quy đổi ra mét.")


def render_overview(stats):
    col_video, col_metrics = st.columns([3, 2])
    with col_video:
        if C.OUTPUT_MP4.exists():
            st.video(C.OUTPUT_MP4.read_bytes())
        else:
            st.warning("Chưa có results/output.mp4 - hãy chạy render.py.")
    with col_metrics:
        p0, p1 = stats["possession_pct"]
        m1, m2 = st.columns(2)
        m1.metric("Kiểm soát bóng - Team 0", f"{p0:.1f}%")
        m2.metric("Kiểm soát bóng - Team 1", f"{p1:.1f}%")
        m3, m4 = st.columns(2)
        m3.metric("Tỷ lệ phát hiện bóng", f"{stats['ball_detected_pct']:.1f}%")
        m4.metric("Số lần đổi quyền kiểm soát", stats["possession_changes"])
        st.plotly_chart(make_possession_donut(stats["possession_pct"], stats["team_colors"]),
                        width="stretch")
    st.caption(f"Clip: {stats['n_frames']} frame, {stats['fps']:.0f} fps, "
               f"{stats['width']}x{stats['height']} - cầu thủ: Team 0 = {stats['n_players'][0]}, "
               f"Team 1 = {stats['n_players'][1]}")


def render_players(stats):
    df = pd.DataFrame(stats["per_player"])
    if df.empty:
        st.info("Không có dữ liệu cầu thủ.")
        return
    df["ty_le_giu_bong"] = (100 * df["frames_with_ball"] / df["frames_seen"]).round(1)
    df = df.rename(columns={"track_id": "ID", "team": "Đội", "frames_seen": "Số frame xuất hiện",
                            "frames_with_ball": "Số frame giữ bóng",
                            "ty_le_giu_bong": "Tỉ lệ giữ bóng (%)"})
    choice = st.selectbox("Lọc theo đội", ["Tất cả", "Team 0", "Team 1"])
    if choice != "Tất cả":
        df = df[df["Đội"] == int(choice[-1])]
    st.dataframe(df.sort_values("Số frame giữ bóng", ascending=False),
                 width="stretch", hide_index=True)


def main():
    st.title("⚽ Phân tích bóng đá bằng YOLO + OpenCV")
    st.write("Phát hiện và theo dõi cầu thủ, bóng trong một clip ngắn; chia đội theo màu áo "
             "và tính tỷ lệ kiểm soát bóng.")
    render_sidebar()

    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        st.error(f"Thiếu file trong results/: {', '.join(missing)}. "
                 "Chạy pipeline (python run_pipeline.py) hoặc tạo dữ liệu giả "
                 "(python make_mock.py), rồi tải lại trang.")
        st.stop()

    try:
        stats, tracks, frames = load_data()
    except Exception as e:   # file hỏng / sai định dạng
        st.error(f"Không đọc được dữ liệu trong results/: {e}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan", "Diễn biến", "Cầu thủ", "Heatmap"])
    with tab1:
        render_overview(stats)
    with tab2:
        st.plotly_chart(make_possession_timeline(frames, stats["team_colors"]),
                        width="stretch")
    with tab3:
        render_players(stats)
    with tab4:
        team = st.radio("Chọn đội", [0, 1], format_func=lambda t: f"Team {t}", horizontal=True)
        st.plotly_chart(make_heatmap(tracks, team, stats["width"], stats["height"]),
                        width="stretch")
        st.caption("Camera di chuyển nên heatmap phản ánh vị trí trên khung hình, không phải sân thật.")

    st.divider()
    st.caption("Tham chiếu: \"Build an AI/ML Football Analysis system with YOLO, OpenCV, "
               "and Python\" (YouTube). Người thực hiện: [ĐIỀN TÊN].")


main()
