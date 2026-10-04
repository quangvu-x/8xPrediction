# ⚽ Phân tích bóng đá bằng YOLO + OpenCV (phiên bản đơn giản hoá)

Xử lý một clip bóng đá 10-15 giây: phát hiện và theo dõi cầu thủ, bóng → chia 2 đội theo màu áo →
tính % kiểm soát bóng → xuất video có chú thích → hiển thị trên web Streamlit.

**Link web:** [ĐIỀN LINK] · **Người thực hiện:** [ĐIỀN TÊN]

## Kiến trúc

```
input/clip.mp4 ─► detect.py ─► results/raw.csv, meta.json        (YOLO + ByteTrack + màu áo)
               ─► analytics.py ─► tracks.csv, frames.csv, stats.json (nội suy bóng, KMeans, giữ bóng)
               ─► render.py ─► output.mp4 (H.264)
results/ ─► git push ─► GitHub ─► Streamlit Cloud ─► app.py (chỉ đọc results/)
```
Mọi tham số nằm trong `config.py`. Web không chạy YOLO (gói host miễn phí không đủ RAM cho torch).

## Cài đặt

```bash
# Windows (PowerShell)
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
# macOS / Linux
python3.11 -m venv venv && source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

## Chạy

```bash
python make_mock.py                                   # (tuỳ chọn) dữ liệu giả để thử web
python cut_clip.py --src input/full.mp4 --start 00:00:00 --dur 15
python run_pipeline.py --video input/clip3s.mp4       # thử nhanh 3 giây
python run_pipeline.py --video input/clip.mp4         # bản thật
streamlit run app.py
```
Chỉ chỉnh analytics/render (không chạy lại YOLO): `python run_pipeline.py --skip-detect`.

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| config.py | Đường dẫn, tên cột, tham số (nguồn duy nhất) |
| cut_clip.py | Cắt clip 15 giây + 3 giây |
| detect.py | Video → raw.csv, meta.json |
| analytics.py | raw.csv → tracks.csv, frames.csv, stats.json |
| render.py | Video + kết quả → output.mp4 |
| check_outputs.py | Kiểm tra hợp đồng dữ liệu (PASS/FAIL) |
| run_pipeline.py | Chạy cả pipeline bằng một lệnh |
| make_mock.py | Sinh dữ liệu giả đi qua code thật |
| viz.py, app.py | Biểu đồ và web Streamlit |

## Hợp đồng dữ liệu

| File | Cột / khoá |
|---|---|
| meta.json | fps, width, height, n_frames |
| raw.csv | frame, track_id, cls, conf, x1, y1, x2, y2, r, g, b |
| tracks.csv | frame, track_id, cls, team, x1, y1, x2, y2, x, y, has_ball |
| frames.csv | frame, time_s, ball_x, ball_y, holder_id, possession_team, cum_pct_team0, cum_pct_team1 |
| stats.json | fps, n_frames, width, height, team_colors, possession_pct, n_players, ball_detected_pct, possession_changes, per_player |

## Kết quả mẫu

[CHÈN ẢNH CHỤP MÀN HÌNH]

## Hạn chế

- Bóng nhỏ, model pretrained COCO có thể bỏ sót; nội suy chỉ bù khoảng trống ≤ 1 giây.
- Trọng tài chính và trọng tài biên được loại theo màu áo (`REFEREE_COLORS` trong `config.py`, đo trên clip này);
  clip khác cần đo lại màu. Thủ môn được tính vào đội của mình.
- Ban huấn luyện / trọng tài thứ tư đứng sát đường biên dọc vẫn có thể bị tính như cầu thủ:
  màu áo vest tối gần màu áo đội khi bị bóng râm, và họ đứng chỉ cách vạch biên vài chục pixel
  nên không lọc được theo màu hay theo vị trí (đã thử dò vạch biên, nhưng nhầm với vạch vòng cấm).
- ID có thể đổi sau khi cầu thủ bị che khuất hoặc camera lia; vì vậy `n_players` là
  số người nhiều nhất có mặt cùng lúc (track ≥ `MIN_PLAYER_SECONDS`), không phải số ID.
- Lọc trọng tài và cách đếm `n_players` ở trên là thay đổi so với kế hoạch gốc
  (kế hoạch gốc không lọc trọng tài và đếm số ID theo đội).
- Một clip ngắn không đại diện cho cả trận; toạ độ theo pixel, chưa quy đổi ra mét.

## Hướng phát triển

Weights fine-tune (có class trọng tài/thủ môn), đếm đường chuyền, bù chuyển động camera,
perspective transform để tính tốc độ/quãng đường.

## Nguồn tham chiếu

- Video: "Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python" – https://www.youtube.com/watch?v=neBZ6huolkg
- Nguồn clip: "FULL MATCH | Manchester City v Manchester United | Final | Emirates FA Cup 2023-24" – https://www.youtube.com/watch?v=zUTUA0TKvfQ
  Đoạn dùng: 15 giây đầu của `input/full.mp4` (đồng hồ trận đấu 03:2x), 2880×1620, 60 fps.
  Clip chỉ dùng cho mục đích học tập, không phát hành lại; bản quyền thuộc chủ sở hữu.
