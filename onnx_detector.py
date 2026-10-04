"""
onnx_detector.py - Chạy YOLOv8 dạng ONNX bằng onnxruntime (KHÔNG cần torch/ultralytics).
Dùng cho web: nhẹ hơn nhiều so với torch, chạy được trên Streamlit Community Cloud.

Tạo file ONNX (làm 1 lần, trên Colab hoặc máy có ultralytics):
    yolo export model=yolov8n.pt format=onnx imgsz=640
→ models/yolov8n.onnx (khoảng 12 MB).

Đầu ra của YOLOv8 ONNX: (1, 84, N) = 4 toạ độ (cx, cy, w, h) + 80 điểm lớp COCO.
"""
import cv2
import numpy as np
import onnxruntime as ort


class OnnxYolo:
    def __init__(self, model_path, imgsz=640, threads=None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape          # [1, 3, H, W] hoặc tên động
        self.imgsz = shape[2] if isinstance(shape[2], int) else imgsz

    def _letterbox(self, img):
        """Resize giữ tỉ lệ + đệm viền xám về hình vuông imgsz."""
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def predict(self, frame_bgr, conf=0.1, classes=(0, 32), iou=0.5):
        """Trả về (xyxy[N,4] theo pixel ảnh gốc, cls[N], conf[N])."""
        img, r, left, top = self._letterbox(frame_bgr)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: blob})[0]
        return self.decode(out, r, left, top, frame_bgr.shape, conf, classes, iou)

    @staticmethod
    def decode(out, r, left, top, shape, conf, classes, iou):
        pred = out[0].T                                   # (N, 84)
        boxes, scores = pred[:, :4], pred[:, 4:]
        cls = scores.argmax(axis=1)
        cf = scores[np.arange(len(scores)), cls]
        keep = (cf >= conf) & np.isin(cls, classes)
        boxes, cls, cf = boxes[keep], cls[keep], cf[keep]
        if len(cf) == 0:
            return np.empty((0, 4)), np.empty(0, int), np.empty(0)

        cx, cy, bw, bh = boxes.T
        x1 = (cx - bw / 2 - left) / r
        y1 = (cy - bh / 2 - top) / r
        x2 = (cx + bw / 2 - left) / r
        y2 = (cy + bh / 2 - top) / r
        H, W = shape[:2]
        xyxy = np.stack([x1.clip(0, W), y1.clip(0, H), x2.clip(0, W), y2.clip(0, H)], axis=1)

        # NMS riêng cho từng lớp
        final = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            b = xyxy[idx]
            rects = [[float(a[0]), float(a[1]), float(a[2] - a[0]), float(a[3] - a[1])] for a in b]
            kept = cv2.dnn.NMSBoxes(rects, cf[idx].astype(float).tolist(), conf, iou)
            final.extend(idx[np.array(kept).flatten()] if len(kept) else [])
        final = np.array(final, dtype=int)
        return xyxy[final], cls[final], cf[final]
