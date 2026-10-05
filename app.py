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

st.set_page_config(page_title="8xPrediction", page_icon="⚽", layout="wide")
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


MAX_MOMENTS = 8


def moments(frames, team_colors, names):
    """Nút "Khoảnh khắc" = các lần đổi quyền kiểm soát; bấm để tua video tới giây đó."""
    pt = frames.loc[frames["possession_team"] != -1, ["time_s", "possession_team"]]
    ch = pt[pt["possession_team"] != pt["possession_team"].shift()].iloc[1:].head(MAX_MOMENTS)
    ui.section("Khoảnh khắc", "⏱")
    if ch.empty:
        st.caption("Không có lần đổi quyền kiểm soát nào.")
        return
    st.caption("Bấm để tua video tới lúc đội nhận bóng.")
    css = []
    for i, team in enumerate(ch["possession_team"].astype(int)):
        r, g, b = team_colors[team]
        fg = "#07130C" if r + g + b > 382 else "#FFFFFF"   # chữ tối trên nền sáng, chữ trắng trên nền tối
        css.append(f".st-key-moment_{i} button{{background:rgb({r},{g},{b});color:{fg};"
                   f"border-color:rgb({r},{g},{b})}}")
    st.markdown(f"<style>{''.join(css)}</style>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (t, team) in enumerate(zip(ch["time_s"], ch["possession_team"].astype(int))):
        cols[i % 4].button(f"⏱ {t:.1f}s → {names[team]}", key=f"moment_{i}", width="stretch",
                           on_click=st.session_state.__setitem__, args=("seek", float(t)))


PLAYERS_HELP = "Số cầu thủ nhiều nhất cùng lúc trong một frame; camera không quay hết sân nên có thể dưới 11."


def quality_warnings(stats, names):
    """Cảnh báo chất lượng dữ liệu ở đầu tab Tổng quan (ngưỡng trong config.py)."""
    if stats["ball_detected_pct"] < C.WARN_MIN_BALL_PCT:
        st.warning(f"⚠️ Chỉ thấy bóng ở {stats['ball_detected_pct']:.0f}% số frame "
                   f"(< {C.WARN_MIN_BALL_PCT}%): % kiểm soát bóng và khoảnh khắc có thể sai.")
    for t, n in enumerate(stats["n_players"]):
        if n > C.WARN_MAX_PLAYERS:
            st.warning(f"⚠️ {names[t]} có {n} người cùng lúc (> {C.WARN_MAX_PLAYERS}): có thể lẫn trọng tài, "
                       "ban huấn luyện hoặc khán giả vào đội.")


# ------------------------------------------------------------------ các tab
def tab_overview(stats, d, frames):
    p0, p1 = stats["possession_pct"]
    names = team_names()
    quality_warnings(stats, names)
    ui.scoreboard(p0, p1, *stats["team_colors"], names=names,
                  help="% thời gian mỗi đội giữ bóng, chỉ tính các frame xác định được đội đang giữ bóng.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        video = d / "output.mp4"
        if video.exists():
            st.video(video.read_bytes(), start_time=int(st.session_state.get("seek", 0)))
        else:
            st.warning("Chưa có output.mp4.")
        moments(frames, stats["team_colors"], names)
    with right:
        a, b = st.columns(2)
        with a:
            ui.stat_card("Phát hiện bóng", f"{stats['ball_detected_pct']:.0f}%", "tỷ lệ frame thấy bóng", "🎯",
                         help=f"% frame model thấy bóng, sau khi lọc điểm nhảy. Dưới {C.WARN_MIN_BALL_PCT}% thì % kiểm soát kém tin cậy.")
        with b:
            ui.stat_card("Đổi quyền", stats["possession_changes"], "số lần đổi đội giữ bóng", "🔁",
                         help="Số lần đội giữ bóng thay đổi; đội mới phải giữ bóng liên tục vài frame mới được tính.")
        a, b = st.columns(2)
        with a:
            ui.stat_card(f"Cầu thủ {names[0]}", stats["n_players"][0], "nhiều nhất cùng lúc", "👕", help=PLAYERS_HELP)
        with b:
            ui.stat_card(f"Cầu thủ {names[1]}", stats["n_players"][1], "nhiều nhất cùng lúc", "👕", help=PLAYERS_HELP)
        st.plotly_chart(make_possession_donut(stats["possession_pct"], stats["team_colors"], names),
                        width="stretch")
        st.caption("Cách đọc: mỗi phần màu là % thời gian đội đó kiểm soát bóng trong clip.")
    with st.expander("⬇️ Tải dữ liệu"):
        c1, c2, c3 = st.columns(3)
        c1.download_button("stats.json", (d / "stats.json").read_bytes(), "stats.json")
        c2.download_button("tracks.csv", (d / "tracks.csv").read_bytes(), "tracks.csv")
        c3.download_button("frames.csv", (d / "frames.csv").read_bytes(), "frames.csv")


def tab_replay(stats, tracks, frames):
    ui.section("Phát lại chiến thuật", "🎬")
    st.caption("Bấm ▶ hoặc kéo thanh trượt. Viền vàng = cầu thủ đang giữ bóng. Rê chuột để xem ID.")
    step = st.select_slider("Độ mượt", options=[1, 2, 3, 5], value=3,
                            format_func=lambda s: {1: "Tối đa", 2: "Cao", 3: "Vừa", 5: "Nhẹ"}[s])
    st.plotly_chart(VX.make_replay(tracks, stats, frames, step=step, names=team_names()), width="stretch")
    st.caption("Cách đọc: mỗi chấm là một cầu thủ, viền vàng là người giữ bóng, chấm trắng là bóng.")


def tab_trend(stats, frames):
    ui.section("Momentum theo thời gian", "📊")
    win = st.slider("Độ dài mỗi cửa sổ (giây)", 1.0, 5.0, 2.0, 0.5)
    names = team_names()
    st.plotly_chart(VX.make_momentum(frames, stats["team_colors"], win, stats["fps"], names), width="stretch")
    st.caption(f"Cách đọc: cột lên là % giữ bóng của {names[0]}, cột xuống của {names[1]}, trong từng cửa sổ.")
    ui.section("% kiểm soát tích luỹ", "📈")
    st.plotly_chart(VX.make_timeline_with_events(frames, stats["team_colors"], names), width="stretch")
    st.caption("Cách đọc: đường cao hơn là đội kiểm soát nhiều hơn tính đến giây đó; vạch vàng là đổi quyền.")


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
        st.caption("Cách đọc: thanh xanh là số giây giữ bóng; quãng đường tính bằng pixel, chưa quy ra mét.")
    with right:
        if len(view):
            pid = st.selectbox("Xem đường di chuyển của cầu thủ", view["ID"].tolist())
            st.plotly_chart(VX.make_player_trail(tracks, pid, stats), width="stretch")
            st.caption("Cách đọc: đường đi của cầu thủ trên khung hình; chấm vàng là lúc cầu thủ giữ bóng.")


def tab_heatmap(stats, tracks):
    names = team_names()
    team = st.radio("Chọn đội", [0, 1], format_func=lambda t: names[t], horizontal=True)
    st.plotly_chart(make_heatmap(tracks, team, stats["width"], stats["height"], name=names[team]),
                    width="stretch")
    st.caption("Cách đọc: ô càng đỏ thì đội xuất hiện ở vùng đó của khung hình càng nhiều.")
    st.caption("Camera di chuyển nên heatmap phản ánh vị trí trên khung hình, không phải sân thật.")


def video_info(f):
    """Đọc thời lượng video upload bằng cv2 (nhớ theo file_id để không đọc lại mỗi lần rerun)."""
    key = f"vinfo_{f.file_id}"
    if key not in st.session_state:
        import tempfile
        import cv2                                      # import muộn: chỉ cần khi có file upload
        import shutil
        with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix or ".mp4") as tmp:
            f.seek(0)
            shutil.copyfileobj(f, tmp, length=8 * 1024 * 1024)   # không gọi getvalue(): tránh nhân đôi RAM
            tmp.flush()
            f.seek(0)
            cap = cv2.VideoCapture(tmp.name)
            ok, _ = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            info = {"ok": ok and fps > 0, "fps": fps, "duration": n / fps if fps else 0,
                    "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
            cap.release()
        st.session_state[key] = info
    return st.session_state[key]


def classify_error(e):
    """(tiêu đề, gợi ý) cho lỗi khi phân tích video upload."""
    msg = str(e)
    if "cầu thủ" in msg or "không chia được đội" in msg:
        return ("Không thấy đủ cầu thủ trong video.",
                "Chọn đoạn camera quay rộng, thấy rõ cầu thủ 2 đội; thử giảm ngưỡng tin cậy.")
    if ("Không đọc được video" in msg or "không có frame" in msg or "ffmpeg" in msg.lower()
            or type(e).__name__ == "CalledProcessError"):
        return ("Không đọc được video.", "Thử xuất lại video sang mp4 (H.264) hoặc chọn file khác.")
    return ("Phân tích thất bại.", "Thử đoạn video khác, giảm độ dài hoặc chọn chế độ Nhanh.")


def show_error(title, hint, detail=""):
    st.session_state["upload_state"] = "error"
    st.error(f"❌ {title}")
    st.caption(f"Gợi ý: {hint}")
    if detail:
        with st.expander("Chi tiết lỗi"):
            st.code(detail)


def tab_upload():
    """Trạng thái: empty → selected → running → done / error."""
    ui.section("Tự phân tích video của bạn", "📤")
    if not ONNX_MODEL.exists():
        st.info("Chưa có models/yolov8n.onnx trong repo. Xem GUIDELINE.md mục B1 để tạo file này.")
        return
    st.caption(f"File tối đa {C.UPLOAD_MAX_MB} MB. Video được cắt tối đa 15 giây, thu nhỏ ≤ 720p để chạy được trên máy chủ miễn phí. "
               "Không lưu video của bạn sau phiên làm việc.")
    f = st.file_uploader("Chọn video (mp4, mov, avi, mkv)", type=["mp4", "mov", "avi", "mkv"])

    # done: vừa phân tích xong ở lượt trước (chỉ hiện khi vẫn là file đó)
    same_file = f is None or f.name == st.session_state.get("upload_name")
    if st.session_state.get("upload_state") == "done" and st.session_state.get("upload_dir") and same_file:
        st.success(f"✅ Đã phân tích xong \"{st.session_state.get('upload_name', 'video')}\". "
                   "Xem ở các tab bên trái; sidebar đang chọn \"Video của bạn\".")

    # empty: chưa chọn file
    if f is None:
        if st.session_state.get("upload_state") != "done":
            st.session_state["upload_state"] = "empty"
            st.info("👆 Chọn một video bóng đá ngắn (camera quay rộng, thấy rõ 2 đội) để bắt đầu.")
        return

    # selected: đã chọn file -> đọc thời lượng, kiểm tra trước khi chạy
    info = video_info(f)
    if not info["ok"]:
        show_error("Không đọc được video.", "Thử xuất lại video sang mp4 (H.264) hoặc chọn file khác.")
        return
    if info["duration"] > C.UPLOAD_MAX_VIDEO_S:
        show_error(f"Video quá dài ({info['duration'] / 60:.1f} phút > {C.UPLOAD_MAX_VIDEO_S / 60:.0f} phút).",
                   "Cắt đoạn cần phân tích (≤ 15 giây) bằng app chỉnh video trên máy rồi upload lại.")
        return

    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Bắt đầu từ giây", 0.0, max(0.0, info["duration"] - 1), 0.0, 1.0)
    dur = c2.slider("Độ dài (giây)", 3, 15, 8)
    mode = c3.selectbox("Chế độ", ["⚡ Nhanh (8 fps)", "⚖️ Cân bằng (12 fps)", "🎯 Chi tiết (25 fps)"], index=1)
    fps = {"⚡": 8, "⚖": 12, "🎯": 25}[mode[0]]
    conf = st.slider("Ngưỡng tin cậy (thấp = bắt nhiều bóng hơn, nhiều nhầm hơn)", 0.05, 0.5, 0.10, 0.05)

    real_dur = max(0.0, min(dur, info["duration"] - start))
    n_frames = int(real_dur * fps)
    lo, hi = (n_frames * s for s in C.UPLOAD_SEC_PER_FRAME)
    size_mb = f.size / 1e6
    if size_mb > C.UPLOAD_BIG_MB:
        st.info("File lớn: chỉ đoạn bạn chọn (tối đa 15 giây) được phân tích; tải lên có thể mất vài phút.")
    st.markdown(f"**{f.name}** · {size_mb:.1f} MB · {info['w']}×{info['h']} · "
                f"{info['duration']:.1f} s · {info['fps']:.0f} fps")
    st.caption(f"Sẽ xử lý {real_dur:.1f} s × {fps} fps = {n_frames} frame. "
               f"Thời gian **ước lượng** (chưa đo trên máy chủ): {lo / 60:.1f}–{hi / 60:.1f} phút.")
    if real_dur < 1:
        show_error("Đoạn đã chọn nằm ngoài video.", f"Chọn \"Bắt đầu từ giây\" nhỏ hơn {info['duration']:.0f}.")
        return
    if not (st.session_state.get("upload_state") == "done" and same_file):
        st.session_state["upload_state"] = "selected"

    if st.button("🚀 Phân tích", type="primary", width="stretch"):
        st.session_state["upload_state"] = "running"
        from web_pipeline import run_uploaded          # import muộn: chỉ cần khi upload
        bar = st.progress(0.0, text="Bắt đầu...")
        try:
            out = run_uploaded(f, Path(f.name).suffix or ".mp4", start, real_dur, fps, conf,
                               on_progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
        except Exception as e:                         # phân loại lỗi, kèm log để người dùng gửi lại
            show_error(*classify_error(e), detail=str(e))
            return
        missing = [n for n in NEEDED if not (out / n).exists()]
        if missing:
            show_error("Phân tích thất bại.", "Thử lại hoặc chọn đoạn video khác.", f"Thiếu kết quả: {missing}")
            return
        st.session_state.update(upload_dir=str(out), source="upload", upload_state="done", upload_name=f.name)
        st.balloons()
        st.rerun()


# ------------------------------------------------------------------ main
def main():
    sidebar()
    d = current_dir()
    is_upload = st.session_state.get("source") == "upload"
    ui.hero("8xPrediction",
            "Phát hiện cầu thủ và bóng bằng YOLO, chia đội theo màu áo, đo kiểm soát bóng và phát lại "
            "chiến thuật — ngay trên trình duyệt.",
            ["YOLOv8", "ByteTrack / IoU tracker", "KMeans", "Streamlit",
             "🎥 Video của bạn" if is_upload else "🏟️ Trận mẫu"], accent_prefix="8x")

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
        tab_overview(stats, d, frames)
    with tabs[1]:
        tab_replay(stats, tracks, frames)
    with tabs[2]:
        tab_trend(stats, frames)
    with tabs[3]:
        tab_players(stats, tracks)
    with tabs[4]:
        tab_heatmap(stats, tracks)
    st.caption("Tham chiếu: \"Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python\". "
               )


main()
