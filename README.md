# ⚽ Phân tích bóng đá bằng YOLO + OpenCV (phiên bản đơn giản hoá)

Xử lý một clip bóng đá 10-15 giây: phát hiện và theo dõi cầu thủ, bóng → chia 2 đội theo màu áo →
tính % kiểm soát bóng → xuất video có chú thích → hiển thị trên web Streamlit.
Web có thêm tab "📤 Tự phân tích": người xem tự upload video ngắn để chạy phân tích.

**Link web:** https://8xprediction-xdapcygbbmkaspmubstbkr.streamlit.app/ · **Người thực hiện:** [ĐIỀN TÊN]

## Kiến trúc

```
Trận mẫu (chạy trên máy):
input/clip.mp4 ─► detect.py ─► results/raw.csv, meta.json        (YOLOv8m + ByteTrack + màu áo)
               ─► analytics.py ─► tracks.csv, frames.csv, stats.json (lọc bóng nhảy, nội suy bóng,
               │                                                   chia đội robust, giữ bóng)
               ─► render.py ─► output.mp4 (H.264)
results/ ─► upload GitHub ─► Streamlit Cloud ─► app.py (đọc results/)

Tab "Tự phân tích" (chạy trên web, CPU):
video upload ─► web_pipeline.py: cắt ≤ 15 s, ≤ 720p, giảm fps
             ─► detect_web.py (YOLOv8n ONNX + SimpleTracker) ─► analytics.py ─► render.py
             ─► thư mục tạm riêng (biến FA_RESULTS_DIR), không ghi đè results/ của trận mẫu
```
Mọi tham số nằm trong `config.py`. Web không dùng torch/ultralytics (gói miễn phí không đủ RAM);
tab upload dùng onnxruntime. Các thiết lập thủ công cho trận mẫu (`REFEREE_COLORS`, `GOALKEEPER_TEAM`)
tự tắt khi xử lý video upload.

## Cài đặt

```bash
# Windows (PowerShell)
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
# macOS / Linux
python3.11 -m venv venv && source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-local.txt   # xử lý trên máy (gồm cả thư viện web)
```
`requirements.txt` chỉ dành cho Streamlit Cloud.

## Chạy

```bash
python make_mock.py                                   # (tuỳ chọn) dữ liệu giả để thử web
python cut_clip.py --src input/full.mp4 --start 00:00:00 --dur 15
python run_pipeline.py --video input/clip3s.mp4       # thử nhanh 3 giây
python run_pipeline.py --video input/clip.mp4         # bản thật
python check_outputs.py --strict                      # kiểm tra hợp đồng dữ liệu (phải 0 FAIL)
python sample_frames.py                               # ảnh results/_qa_sheet.jpg (12 frame) để chấm bằng mắt
streamlit run app.py
```
Chỉ chỉnh analytics/render (không chạy lại YOLO): `python run_pipeline.py --video input/clip.mp4 --skip-detect`.

Thử tab "Tự phân tích" trên máy: `streamlit run app.py` → tab 📤 → upload `input/clip3s.mp4`
(cần `models/yolov8n.onnx`; kết quả ghi vào thư mục tạm, không đụng `results/`).

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| config.py | Đường dẫn, tên cột, tham số (nguồn duy nhất); `RESULTS_DIR` đổi được qua biến `FA_RESULTS_DIR` |
| cut_clip.py | Cắt clip 15 giây + 3 giây |
| detect.py | Video → raw.csv, meta.json (YOLOv8m + ByteTrack, chạy trên máy) |
| analytics.py | raw.csv → tracks.csv, frames.csv, stats.json |
| accuracy_utils.py | Lọc bóng nhảy, chia đội Lab / robust (loại người không phải cầu thủ) |
| render.py | Video + kết quả → output.mp4 |
| check_outputs.py | Kiểm tra hợp đồng dữ liệu (PASS/FAIL) |
| run_pipeline.py | Chạy cả pipeline bằng một lệnh |
| make_mock.py | Sinh dữ liệu giả đi qua code thật |
| sample_frames.py | Ảnh 12 frame `results/_qa_sheet.jpg` để chấm chất lượng |
| app.py, ui.py | Web Streamlit (6 tab) và giao diện (CSS, hero, scoreboard, stat card) |
| viz.py, viz_extra.py | Biểu đồ (donut, heatmap, phát lại, diễn biến, cầu thủ) |
| web_pipeline.py | Pipeline cho video upload trên web |
| detect_web.py, onnx_detector.py, simple_tracker.py | Phát hiện bằng YOLOv8n ONNX + tracker IoU/Hungarian (bản web) |
| models/yolov8n.onnx | Model cho tab upload (12 MB) |
| .streamlit/config.toml | Theme tối và giới hạn upload 200 MB |
| GUIDELINE.md | Hướng dẫn nâng cấp v2 (độ chính xác, upload, giao diện) |

## Hợp đồng dữ liệu

| File | Cột / khoá |
|---|---|
| meta.json | fps, width, height, n_frames |
| raw.csv | frame, track_id, cls, conf, x1, y1, x2, y2, r, g, b |
| tracks.csv | frame, track_id, cls, team, x1, y1, x2, y2, x, y, has_ball |
| frames.csv | frame, time_s, ball_x, ball_y, holder_id, possession_team, cum_pct_team0, cum_pct_team1 |
| stats.json | fps, n_frames, width, height, team_colors, possession_pct, n_players, ball_detected_pct, possession_changes, per_player |

## Kết quả mẫu

<img width="1439" height="709" alt="Screenshot 2026-10-05 at 17 34 09" src="https://github.com/user-attachments/assets/c1b445e0-5c90-4362-b15c-28bfd518ef26" />
<img width="1437" height="706" alt="Screenshot 2026-10-05 at 17 36 06" src="https://github.com/user-attachments/assets/ed1ac75d-a006-414c-a414-715573643ae7" />
<img width="1436" height="704" alt="Screenshot 2026-10-05 at 17 36 37" src="https://github.com/user-attachments/assets/6f0b1e7e-46c5-4eb7-b391-79872ad52f7b" />

## Nhật ký tinh chỉnh

Clip `input/clip.mp4` (15 s, 60 fps, 900 frame), chạy lại bằng `--skip-detect` trên cùng `raw.csv`.

| Bước | Thay đổi | possession_pct | ball_detected_pct | possession_changes | n_players |
|---|---|---|---|---|---|
| Baseline | `MIN_POSSESSION_FRAMES = 10`, lọc trọng tài theo màu, đếm cùng lúc | 75.4 / 24.6 | 93.3 | 9 | [8, 11] |
| A1 | Lọc bóng nhảy, `BALL_MAX_SPEED_RATIO = 1.5` | 75.6 / 24.4 | 92.1 | 9 | [8, 11] |
| A2 | Chia đội Lab (`assign_teams_lab`) | 75.6 / 24.4 | 92.1 | 9 | [11, 11] (ban huấn luyện lọt vào Team 0) |
| A2' | `assign_teams_robust`, `TEAM_CLUSTERS = 4`, `TEAM_OUTLIER_FACTOR = 3.0`, `GOALKEEPER_TEAM = {6: 1}` | 75.6 / 24.4 | 92.1 | 9 | [8, 11] |
| A3 | Giữ `MIN_POSSESSION_FRAMES = 10` (≈ 0.17 s ở 60 fps; 3 → 29 lần, 5 → 19 lần đổi quyền) | 75.6 / 24.4 | 92.1 | 9 | [8, 11] |

Chấm trên `results/_qa_sheet.jpg` (12 frame, `python sample_frames.py`):

| Chỉ số | Baseline | Sau A1–A3 |
|---|---|---|
| Đúng đội (elip đúng màu / tổng cầu thủ thấy) | [ĐIỀN] | [ĐIỀN] |
| Bóng (đúng / sai vị trí / không có) | [ĐIỀN] | [ĐIỀN] |
| Người giữ bóng (tam giác đỏ đúng người) | [ĐIỀN] | [ĐIỀN] |

## Hạn chế

- Bóng nhỏ, model pretrained COCO có thể bỏ sót; nội suy chỉ bù khoảng trống ≤ 1 giây.
  Điểm bóng nhảy vọt (nhận nhầm đầu, giày, vạch sân) bị loại theo `BALL_MAX_SPEED_RATIO`,
  nên `ball_detected_pct` được tính sau khi lọc.
- Chia đội bằng `assign_teams_robust` (`accuracy_utils.py`): gom màu áo (Lab) thành `TEAM_CLUSTERS` nhóm,
  2 nhóm xuất hiện nhiều nhất = 2 đội, các nhóm khác (ban huấn luyện, trọng tài thứ tư, vest tối)
  và track lệch màu xa tâm đội (> `TEAM_OUTLIER_FACTOR`) bị loại khỏi `tracks.csv`.
  Các tham số được chỉnh và kiểm tra bằng ảnh trên **clip này**; clip khác cần kiểm tra lại.
- Trọng tài chính và trọng tài biên còn được loại thêm theo màu áo đo sẵn (`REFEREE_COLORS`).
  Trên clip này bước này trùng với `assign_teams_robust`, nhưng được giữ lại phòng khi
  trọng tài xuất hiện nhiều đến mức thành một nhóm màu chính.
- Thủ môn mặc áo khác màu đội nên bị coi là "không thuộc 2 nhóm màu chính";
  thủ môn được gán đội **thủ công** bằng `GOALKEEPER_TEAM` theo track_id.
  `GOALKEEPER_TEAM` và `EXCLUDE_TRACK_IDS` chỉ đúng với `results/raw.csv` hiện tại:
  chạy lại detect hoặc đổi clip thì ID thay đổi, phải đặt lại.
- ID có thể đổi sau khi cầu thủ bị che khuất hoặc camera lia; vì vậy `n_players` là
  số người nhiều nhất có mặt cùng lúc (track ≥ `MIN_PLAYER_SECONDS`), không phải số ID.
  Camera không quay hết sân nên `n_players` có thể < 11.
- Lọc trọng tài / người ngoài sân, gán thủ môn và cách đếm `n_players` ở trên là thay đổi
  so với kế hoạch gốc (kế hoạch gốc chia 2 cụm KMeans RGB cho mọi người và đếm số ID theo đội).
- Một clip ngắn không đại diện cho cả trận; toạ độ theo pixel, chưa quy đổi ra mét.
- Tab "Tự phân tích" (upload video trên web) dùng YOLOv8n ONNX + tracker đơn giản nên kém chính xác hơn
  trận mẫu (yolov8m + ByteTrack). Video được cắt ≤ 15 s, ≤ 720p, giảm fps để chạy được trên CPU yếu;
  thời gian xử lý trên Streamlit Cloud chưa đo. Kết quả upload chỉ tồn tại trong phiên,
  app khởi động lại thì mất; nhiều người upload cùng lúc sẽ chậm hoặc hết RAM (bản demo, không phải dịch vụ).

## Hướng phát triển

- Pass tìm bóng thứ 2 ở 1280 px (GUIDELINE A4, chưa làm) và model lớn hơn trên Colab (A5).
- Weights fine-tune (có class trọng tài/thủ môn) để bỏ gán thủ môn thủ công (A6).
- Đếm đường chuyền, bù chuyển động camera, perspective transform để tính tốc độ/quãng đường.

## Nguồn tham chiếu

- Video: "Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python" – https://www.youtube.com/watch?v=neBZ6huolkg
- Nguồn clip: "FULL MATCH | Manchester City v Manchester United | Final | Emirates FA Cup 2023-24" – https://www.youtube.com/watch?v=zUTUA0TKvfQ
  Đoạn dùng: 15 giây đầu của `input/full.mp4` (đồng hồ trận đấu 03:2x), 2880×1620, 60 fps.
  Clip chỉ dùng cho mục đích học tập, không phát hành lại; bản quyền thuộc chủ sở hữu.
