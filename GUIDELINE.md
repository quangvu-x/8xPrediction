# GUIDELINE v2 – Tăng độ chính xác, tương tác và giao diện

> Đọc cùng `CLAUDE.md`. Mỗi mục có **prompt dán cho Claude Code** và **tiêu chí đạt**.
> Nguyên tắc: **mỗi lần một thay đổi → đo lại → commit**. Không đổi hợp đồng dữ liệu.

## 0. Thứ tự làm và thời gian ước lượng

| # | Hạng mục | Giá trị | Rủi ro | Thời gian [Inference] |
|---|---|---|---|---|
| 1 | Cài gói nâng cấp + giao diện mới (C) | Cao, thấy ngay | Thấp | 30–45 phút |
| 2 | Đo baseline (A0) | Bắt buộc để biết cải tiến có thật | Thấp | 20 phút |
| 3 | Lọc bóng nhảy + chia đội Lab + lọc nhiễu possession (A1–A3) | Cao | Thấp | 45 phút |
| 4 | Tự upload video trên web (B) | Rất cao (tương tác) | Trung bình | 1–2 giờ |
| 5 | Pass bóng thứ 2 + model lớn trên Colab (A4–A5) | Trung bình–cao | Trung bình | 1 giờ |
| 6 | Weights fine-tune bóng đá (A6) | Cao | Cao | ≥ 2 giờ |

Cắt giảm theo thứ tự ngược lại nếu thiếu giờ. Luôn giữ bản web đang chạy được.

### Trạng thái thực hiện (04/10/2026, trên `input/clip.mp4`, 60 fps)

| Mục | Trạng thái | Ghi chú |
|---|---|---|
| C – giao diện | ✅ Đã cài | `.streamlit/config.toml` đã tạo. AppTest đủ 6 tab, chưa xem trên trình duyệt/điện thoại |
| A0 – baseline | ⚠️ Một phần | Số liệu ghi trong README "Nhật ký tinh chỉnh"; 3 chỉ số chấm bằng mắt chưa điền |
| A1 – lọc bóng nhảy | ✅ | `BALL_MAX_SPEED_RATIO = 1.5`; ball_detected_pct 93.3 → 92.1 |
| A2 – chia đội | ✅ Thay bằng `assign_teams_robust` | `assign_teams_lab` đẩy ban huấn luyện vào Team 0 → dùng bản robust: `TEAM_CLUSTERS = 4`, `TEAM_OUTLIER_FACTOR = 3.0`, thủ môn gán tay `GOALKEEPER_TEAM = {6: 1}` |
| A3 – lọc đổi quyền | ✅ Giữ 10 | Clip 60 fps: 10 frame ≈ 0.17 s (3 → 29 lần, 5 → 19 lần, 10 → 9 lần đổi quyền) |
| A4 – pass bóng 1280 px | ⏭️ Bỏ qua | Chạy lại YOLO sẽ đổi track_id → phải đặt lại `GOALKEEPER_TEAM` |
| A5, A6 | ⏳ Chưa làm | Cần Colab/GPU |
| B1 – yolov8n.onnx | ✅ | Export trên máy (12.3 MB) vào `models/` |
| B2 – `FA_RESULTS_DIR` | ✅ | Thiết lập thủ công (`REFEREE_COLORS`, `GOALKEEPER_TEAM`) tự tắt khi có `FA_RESULTS_DIR` (video upload) |
| B3 – thử upload trên máy | ✅ | `run_uploaded(clip3s)` ≈ 6 s trên máy, 23 PASS. Chưa thử trên Streamlit Cloud |
| B4 – giới hạn web | ✅ | Đã ghi trong README "Hạn chế" |
| D – GitHub | ⏳ | Chờ upload |

---

## A. Độ chính xác

### A0. Đo trước khi sửa (baseline)

```bash
python check_outputs.py --strict
python sample_frames.py          # tạo results/_qa_sheet.jpg (12 frame rải đều)
```

Mở `_qa_sheet.jpg`, chấm và ghi vào README (mục "Nhật ký tinh chỉnh"):

| Chỉ số | Cách đếm trên 12 frame | Baseline | Sau cải tiến |
|---|---|---|---|
| Đúng đội | số cầu thủ có elip đúng màu áo / tổng cầu thủ thấy | | |
| Bóng | đúng / sai vị trí / không có tam giác vàng | | |
| Người giữ bóng | tam giác đỏ đúng người (khi bóng đang ở chân ai đó) | | |
| ball_detected_pct | lấy từ stats.json | | |
| possession_changes | lấy từ stats.json (càng sát thực tế càng tốt) | | |

Chỉ giữ một cải tiến khi ít nhất một chỉ số tốt lên và không chỉ số nào xấu đi rõ rệt.

### A1. Lọc bóng nhảy (spike) – `accuracy_utils.filter_ball_outliers`

Detection "bóng" giả (đầu, giày, vạch sân) thường nhảy vọt rồi quay lại. Hàm loại điểm cách **cả** điểm trước **lẫn** điểm sau quá xa (đã thử: loại 16/19 điểm nhiễu cấy vào, không mất điểm đúng).

```text
Prompt: Đọc @accuracy_utils.py và @analytics.py. Trong interpolate_ball, ngay sau dòng
ball = ball.groupby("frame")[["bx", "by"]].first(), gọi
ball = filter_ball_outliers(ball, meta["width"], meta["fps"], C.BALL_MAX_SPEED_RATIO).
Thêm BALL_MAX_SPEED_RATIO = 1.5 vào config.py (phần analytics).
Tính ball_detected_pct SAU khi lọc. Chạy run_pipeline.py --skip-detect, check_outputs --strict,
sample_frames.py và báo ball_detected_pct trước/sau.
```

Tinh chỉnh: bóng thật bị loại khi chuyền mạnh → tăng lên 2.0; còn nhiều điểm nhảy → giảm 1.0.

### A2. Chia đội trong không gian màu Lab – `accuracy_utils.assign_teams_lab`

Lab gần với cảm nhận của mắt; giảm trọng số độ sáng giúp bóng râm/ánh đèn ít làm sai đội. Hàm có **cùng đầu vào/đầu ra** với `assign_teams`.

```text
Prompt: Trong @analytics.py, thay lời gọi assign_teams(players) bằng
assign_teams_lab(players, C.MIN_TRACK_FRAMES) từ accuracy_utils. GIỮ NGUYÊN phần lọc trọng tài
và cách đếm n_players hiện có. Bọc try/except ValueError để in thông báo tiếng Việt như cũ.
Chạy lại, so team_colors trong stats.json với màu áo thật, chạy sample_frames.py.
```

### A3. Lọc nhiễu đổi quyền kiểm soát

```text
Prompt: Trong config.py đặt MIN_POSSESSION_FRAMES = 3 (12 fps) hoặc 5 (25 fps).
Chạy --skip-detect, báo possession_changes trước/sau.
```

Quy đổi theo fps của clip: giữ khoảng 0.2–0.25 s → `MIN_POSSESSION_FRAMES ≈ 0.2 × fps`
(60 fps → 10–15; trận mẫu dùng 10).

### A4. Pass phát hiện bóng thứ 2 ở độ phân giải cao (detect.py, chạy local/Colab)

Bóng nhỏ → YOLO ở 960 px hay bỏ sót. Chạy thêm 1 lần chỉ tìm bóng ở 1280 px (chậm hơn khoảng 1,5–2 lần [Inference]).

```text
Prompt: Trong @detect.py, thêm config BALL_IMGSZ = 1280 (0 = tắt). Nếu BALL_IMGSZ > imgsz:
mỗi frame gọi thêm model.predict(frame, classes=[C.BALL_CLASS], conf=args.conf,
imgsz=C.BALL_IMGSZ, verbose=False), gộp ứng viên bóng của cả 2 lần, giữ conf cao nhất.
Không đổi phần cầu thủ/ByteTrack. Thử trên clip3s trước.
```

### A5. Model lớn hơn trên Google Colab (GPU)

Máy thiếu dung lượng → chạy trên Colab (Phụ lục C của kế hoạch), sau đó chỉ tải `results/` về:

```bash
!python run_pipeline.py --video input/clip.mp4 --model yolov8l.pt --imgsz 1280
```

### A6. (Nâng cao) Weights fine-tune bóng đá

Có class `player / goalkeeper / referee / ball` riêng → bóng chính xác hơn, loại trọng tài đúng nghĩa. Theo video gốc, dùng dataset bóng đá trên Roboflow Universe và train trên Colab. Chỉ làm khi A1–A5 xong và còn ≥ 2 giờ. [Unverified] Cần kiểm tra giấy phép dataset.

---

## B. Tương tác: tự upload video và phân tích trên web

### B0. Kiến trúc

```
Upload (≤200 MB) → web_pipeline.py: ffmpeg cắt ≤15 s, ≤720p, 8/12/25 fps
  → detect_web.py: YOLOv8n ONNX (onnxruntime) + SimpleTracker (IoU + Hungarian)
  → analytics.py (dùng CHUNG với bản local) → render.py → thư mục tạm riêng cho mỗi lần upload
  → dashboard hiển thị như trận mẫu
```

- Không dùng torch/ultralytics trên web: onnxruntime nhẹ hơn nhiều, vừa với gói miễn phí (khoảng 1 GB RAM).
- `config.py` đọc biến môi trường `FA_RESULTS_DIR`, nên mỗi lượt upload ghi vào thư mục tạm và không đè `results/` của trận mẫu.
- Đã thử end-to-end bằng model ONNX giả lập: giải mã box đúng toạ độ, NMS đúng, pipeline ra đủ 6 file. **Chưa thử với yolov8n.onnx thật và chưa thử trên Streamlit Cloud.**

### B1. Tạo `models/yolov8n.onnx` (trên Colab, không tốn dung lượng máy)

```python
!pip -q install ultralytics onnx onnxslim
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)
from google.colab import files; files.download("yolov8n.onnx")
```

- [Unverified] File khoảng 12 MB, dưới giới hạn 25 MB của GitHub web. yolov8s (khoảng 43 MB) và lớn hơn **không** upload qua web được.
- Muốn bắt bóng tốt hơn trên web: export `imgsz=960` (chậm hơn khoảng 2,25 lần; `onnx_detector.py` tự đọc kích thước đầu vào).

### B2. Cập nhật config.py (1 dòng) – bắt buộc

```text
Prompt: Trong @config.py thêm "import os" và đổi dòng RESULTS_DIR thành
RESULTS_DIR = Path(os.environ.get("FA_RESULTS_DIR", ROOT / "results")).
Không đổi gì khác. Kiểm tra mọi đường dẫn results được tính SAU dòng này.
Thêm "onnxruntime>=1.17" vào requirements-local.txt và "results/_*.jpg" vào .gitignore.
```

### B3. Thử trên máy trước khi upload GitHub

```bash
pip install onnxruntime
python detect_web.py --video input/clip3s.mp4      # ghi vào results/ (sau đó chạy lại pipeline thật!)
streamlit run app.py                                # tab "📤 Tự phân tích" → upload clip thử
```

Đạt khi: thanh tiến độ chạy, sau khi xong sidebar có lựa chọn "Video của bạn", các tab hiển thị.
**Lưu ý:** lệnh `detect_web.py` trực tiếp sẽ ghi đè `results/`; chạy lại `run_pipeline.py` bản thật trước khi upload GitHub.

### B4. Giới hạn cần ghi trong README và nói khi demo

- Web dùng YOLOv8n nên kém chính xác hơn trận mẫu (yolov8m + ByteTrack).
- [Inference] Thời gian xử lý trên Streamlit Cloud có thể 1–4 phút cho 8 giây ở 12 fps; chọn "⚡ Nhanh" nếu chậm.
- Kết quả upload chỉ tồn tại trong phiên; app khởi động lại thì mất.
- Nhiều người upload cùng lúc sẽ chậm hoặc hết RAM. Đây là bản demo, không phải dịch vụ.

---

## C. Giao diện (UI guideline)

### C1. Hệ thống thiết kế

| Thành phần | Giá trị | Dùng cho |
|---|---|---|
| Nền | `#07130C` (sân đêm) | Toàn trang |
| Nền thẻ | `#0F2418`, viền `rgba(255,255,255,.08)`, bo góc 16 px | Card, scoreboard |
| Nhấn chính | `#22C55E` (xanh cỏ) | Nút, hover, tab đang chọn |
| Nhấn phụ | `#FFD60A` (vàng bóng) | Người giữ bóng, mốc đổi quyền, bóng |
| Màu đội | **Luôn lấy từ `team_colors`** | Mọi thứ thuộc về đội |
| Chữ | `#E8F5EC`; phụ `#9DB8A6` | Nội dung, chú thích |
| Font tiêu đề | Bebas Neue (kiểu bảng tỷ số) | Hero, số liệu lớn, tiêu đề mục |
| Font nội dung | Inter | Mọi chữ còn lại |

Theme nằm ở `.streamlit/config.toml`; CSS và component ở `ui.py` (`hero`, `scoreboard`, `stat_card`, `section`).

### C2. Bố cục từng tab

| Tab | Nội dung chính | Tương tác |
|---|---|---|
| 🏟️ Tổng quan | Scoreboard possession, video, 4 stat card, donut, tải dữ liệu | Hover card, download |
| 🎬 Phát lại | Sơ đồ chấm cầu thủ/bóng chạy theo thời gian, viền vàng = giữ bóng | Play/Pause, kéo thời gian, chọn độ mượt |
| 📈 Diễn biến | Momentum theo cửa sổ + đường tích luỹ có vạch đổi quyền | Kéo độ dài cửa sổ |
| 👟 Cầu thủ | Bảng có thanh tiến độ "giây giữ bóng" + đường di chuyển | Lọc đội, chọn cầu thủ |
| 🔥 Heatmap | Heatmap theo đội | Chọn đội |
| 📤 Tự phân tích | Upload, chọn đoạn/chế độ/ngưỡng, thanh tiến độ | Upload và chạy |

### C3. Quy tắc

- **Nên:**
  - Mỗi tab mở đầu bằng **một** con số/biểu đồ chính, chi tiết để phía dưới.
  - Mỗi biểu đồ có caption 1 câu giải thích cách đọc.
  - Màu đội nhất quán giữa video, scoreboard, biểu đồ.
  - Số lớn dùng Bebas Neue.
  - Hiệu ứng nhẹ: hover nâng thẻ 3 px, thanh possession chạy 1,2 s, bóng bay khi phân tích xong.
- **Không nên:**
  - Quá 2 màu nhấn.
  - Chỉ dùng màu để phân biệt (đã có nhãn ID và chữ "Team 0/1").
  - Nhồi > 4 stat card một hàng.
  - Dùng chữ có dấu trên video OpenCV.
- **Mobile:** kiểm tra trên điện thoại. CSS đã thu nhỏ hero/scoreboard dưới 640 px; cột Streamlit tự xếp dọc.
- **Giọng văn:** tiếng Việt ngắn, chủ động ("Bấm ▶ để xem lại"), số kèm đơn vị.

### C4. Ý tưởng tiếp (nếu còn giờ)

- Ảnh chụp frame "khoảnh khắc đổi quyền" dưới timeline.
- So sánh 2 lần chạy (trận mẫu và video của bạn) cạnh nhau.
- Nút chia sẻ link kèm `?source=demo`.

[Unverified] Giao diện mới mới được kiểm tra bằng Streamlit AppTest (không lỗi, đủ 6 tab), **chưa xem bằng trình duyệt thật**. Hãy mở trên máy và điện thoại rồi chỉnh CSS trong `ui.py` nếu cần.

---

## D. Đưa lên GitHub (web, không cần git)

File mới/cập nhật trong gói nâng cấp:

| File | Hành động |
|---|---|
| `app.py`, `requirements.txt` | Thay thế file cũ |
| `ui.py`, `viz_extra.py`, `onnx_detector.py`, `simple_tracker.py`, `detect_web.py`, `web_pipeline.py`, `accuracy_utils.py`, `sample_frames.py`, `GUIDELINE.md` | Thêm mới ở thư mục gốc |
| `.streamlit/config.toml` | Add file → Create new file → gõ `.streamlit/config.toml`, dán nội dung |
| `models/yolov8n.onnx` | Create new file `models/README.md` (ghi 1 dòng mô tả) → vào `models/` → Upload files |
| `config.py`, `analytics.py`, `requirements-local.txt`, `.gitignore` | Do Claude Code sửa theo prompt A1–A3, B2, rồi upload bản đã sửa |
| `results/` | Upload lại nếu analytics thay đổi số liệu |

Sau khi upload: Streamlit Cloud cài lại thư viện (lâu hơn lần đầu vì có onnxruntime, opencv, sklearn). Nếu lỗi build → Prompt Deploy (mục 8.3 kế hoạch) kèm log.
