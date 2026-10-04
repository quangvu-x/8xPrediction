# CLAUDE.md – Dự án phân tích bóng đá (football-analysis)

## Bối cảnh
- Một người triển khai (Owner), mới dùng VS Code và ít dùng Python. Giải thích ngắn, bằng tiếng Việt.
- Code đã có sẵn và đã chạy được trên dữ liệu giả. Nhiệm vụ của bạn: chạy, sửa lỗi, tinh chỉnh, giải thích.
  KHÔNG viết lại file từ đầu khi chỉ cần sửa vài dòng.

## Môi trường
- Python 3.11, venv tại `venv/`. Luôn chạy lệnh Python trong venv.
- Xử lý (chỉ trên máy/Colab): ultralytics, supervision<0.31, opencv-python, scikit-learn, pandas, numpy, imageio-ffmpeg.
- Web: chỉ streamlit, pandas, plotly. `app.py`, `viz.py`, `config.py` KHÔNG được import cv2/ultralytics/torch.
- `requirements.txt` chỉ dành cho web; thư viện xử lý để ở `requirements-local.txt`.

## Lệnh chuẩn
- Dữ liệu giả: `python make_mock.py`
- Cắt clip: `python cut_clip.py --src input/full.mp4 --start 00:00:05 --dur 15`
- Pipeline thử: `python run_pipeline.py --video input/clip3s.mp4`
- Pipeline thật: `python run_pipeline.py --video input/clip.mp4`
- Tinh chỉnh (không chạy lại YOLO): `python run_pipeline.py --video input/clip.mp4 --skip-detect`
- Kiểm tra: `python check_outputs.py --strict`
- Web: `streamlit run app.py`

## Luật bắt buộc
1. Hợp đồng dữ liệu (tên file, tên cột trong `config.py`: RAW_COLS, TRACKS_COLS, FRAMES_COLS, STATS_KEYS) KHÔNG được đổi
   nếu tôi chưa đồng ý rõ ràng.
2. Mọi tham số chỉ đổi trong `config.py` hoặc qua tham số dòng lệnh; không hardcode số trong script.
3. Mỗi lần chỉ đổi MỘT thứ. Sau mỗi thay đổi: chạy pipeline phù hợp + `python check_outputs.py --strict`,
   báo kết quả PASS/FAIL và số liệu possession_pct, ball_detected_pct.
4. Trước khi sửa từ 2 file trở lên: trình bày kế hoạch ngắn và chờ tôi đồng ý.
5. Không xoá file trong `input/`, không xoá `results/` trừ khi tôi yêu cầu.
6. Không commit `venv/`, `input/`, `*.pt`, `results/raw.csv`, `results/raw_render.mp4`.
7. Commit message dạng: `CP4: ket qua that clip 15s` hoặc `Tinh chinh: HOLD_DIST_RATIO 0.04 -> 0.05`.
8. Không đưa token, mật khẩu, đường dẫn cá nhân vào code hoặc README.

## Cách trả lời
- Khi gặp lỗi: (1) nguyên nhân 1–2 câu, (2) sửa, (3) lệnh để tôi kiểm tra lại.
- Khi giải thích code: theo khối, đầu vào → xử lý → đầu ra, không dùng thuật ngữ khó nếu không cần.
