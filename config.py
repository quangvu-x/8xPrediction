"""
config.py - NGUỒN DUY NHẤT cho đường dẫn và tham số của toàn dự án.
Mọi script (detect, analytics, render, check_outputs, make_mock, viz, app)
đều import từ đây, nên muốn đổi tham số chỉ cần sửa ở MỘT chỗ.
File này chỉ dùng thư viện chuẩn -> web Streamlit import được mà không cần torch/cv2.
"""
import os
from pathlib import Path

# ---------- Đường dẫn ----------
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
RESULTS_DIR = Path(os.environ.get("FA_RESULTS_DIR", ROOT / "results"))

DEFAULT_VIDEO = INPUT_DIR / "clip.mp4"
TEST_VIDEO = INPUT_DIR / "clip3s.mp4"

META_JSON = RESULTS_DIR / "meta.json"
RAW_CSV = RESULTS_DIR / "raw.csv"
TRACKS_CSV = RESULTS_DIR / "tracks.csv"
FRAMES_CSV = RESULTS_DIR / "frames.csv"
STATS_JSON = RESULTS_DIR / "stats.json"
OUTPUT_MP4 = RESULTS_DIR / "output.mp4"
RAW_RENDER_MP4 = RESULTS_DIR / "raw_render.mp4"

# ---------- Hợp đồng dữ liệu (tên cột) ----------
RAW_COLS = ["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "r", "g", "b"]
TRACKS_COLS = ["frame", "track_id", "cls", "team", "x1", "y1", "x2", "y2", "x", "y", "has_ball"]
FRAMES_COLS = ["frame", "time_s", "ball_x", "ball_y", "holder_id",
               "possession_team", "cum_pct_team0", "cum_pct_team1"]
STATS_KEYS = ["fps", "n_frames", "width", "height", "team_colors", "possession_pct",
              "n_players", "ball_detected_pct", "possession_changes", "per_player"]

# ---------- detect.py ----------
MODEL = "yolov8m.pt"        # yolov8n.pt: nhanh/kém hơn; yolov8l.pt: chậm/tốt hơn
IMGSZ = 960                 # 1280 giúp bắt bóng nhỏ tốt hơn nhưng chậm hơn
CONF = 0.10                 # thấp -> nhiều bóng hơn nhưng nhiều nhận diện sai hơn
PERSON_CLASS = 0            # COCO: person
BALL_CLASS = 32             # COCO: sports ball
# Vùng lấy màu áo (tỉ lệ trong bbox): thân trên
JERSEY_Y = (0.20, 0.50)
JERSEY_X = (0.25, 0.75)
# Pixel cỏ bị loại (thang HSV của OpenCV: hue 0-179)
GRASS_HUE = (35, 85)
GRASS_MIN_SAT = 40

# ---------- web ----------
TEAM_NAMES = ["Man City", "Man United"]  # tên hiển thị trên web, theo thứ tự team 0 / team 1
WARN_MIN_BALL_PCT = 40      # web cảnh báo khi ball_detected_pct thấp hơn mức này
WARN_MAX_PLAYERS = 14       # web cảnh báo khi n_players của một đội lớn hơn mức này
UPLOAD_MAX_VIDEO_S = 600    # video upload dài hơn (giây) bị từ chối: cắt ngắn trước khi upload
UPLOAD_SEC_PER_FRAME = (0.6, 2.5)  # ƯỚC LƯỢNG giây xử lý/frame trên Streamlit Cloud (GUIDELINE B4, chưa đo)

# ---------- analytics.py ----------
HOLD_DIST_RATIO = 0.04      # khoảng cách giữ bóng tối đa = ratio x chiều rộng video
MIN_TRACK_FRAMES = 5        # track ngắn hơn không tham gia KMeans
BALL_INTERP_MAX_GAP_S = 1.0 # khoảng trống bóng dài hơn (giây) thì không nội suy
BALL_MAX_SPEED_RATIO = 1.5  # điểm bóng nhảy vọt > ratio x chiều rộng video mỗi giây -> loại
MIN_POSSESSION_FRAMES = 10  # >1: đội mới phải giữ bóng liên tục N frame mới tính đổi quyền
BALL_BOX_HALF = 6           # bbox giả của bóng trong tracks.csv = tâm +/- 6 px
# Thiết lập THỦ CÔNG bên dưới chỉ đúng với trận mẫu; tắt khi chạy cho video người dùng upload
_UPLOAD = "FA_RESULTS_DIR" in os.environ
REFEREE_COLORS = [] if _UPLOAD else [[56, 64, 45],    # trọng tài chính, đo trên clip3s.mp4
                                     [189, 198, 42]]  # trọng tài biên (vàng chanh), đo trên clip.mp4
REFEREE_COLOR_DIST = 25     # track có màu gần REFEREE_COLORS hơn mức này -> trọng tài, bị loại
MIN_PLAYER_SECONDS = 0.5    # track xuất hiện ngắn hơn (giây) không được đếm vào n_players
TEAM_CLUSTERS = 4           # số nhóm màu; 2 nhóm có nhiều frame nhất = 2 đội, còn lại bị loại
TEAM_OUTLIER_FACTOR = 3.0   # track cách tâm đội > factor x khoảng cách trung vị -> loại
EXCLUDE_TRACK_IDS = []      # loại thủ công theo track_id (chỉ đúng với raw.csv hiện tại)
GOALKEEPER_TEAM = {} if _UPLOAD else {6: 1}  # gán thủ môn thủ công {track_id: team} (chỉ đúng với raw.csv hiện tại)

# ---------- render.py ----------
MAX_WIDTH = 1280
CRF = 26                    # tăng 28-30 nếu output.mp4 quá nặng
MAX_OUTPUT_MB = 30
