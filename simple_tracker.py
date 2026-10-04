"""
simple_tracker.py - Tracker IoU + Hungarian (scipy, có sẵn khi cài scikit-learn).
Thay cho ByteTrack trên web để tránh phụ thuộc supervision/opencv-python (xung đột với
opencv-python-headless). Chất lượng thấp hơn ByteTrack một chút nhưng đủ cho clip ngắn.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """IoU giữa từng box a[i] và b[j] (định dạng x1,y1,x2,y2)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_lost=15, min_conf=0.25):
        self.iou_thresh, self.max_lost, self.min_conf = iou_thresh, max_lost, min_conf
        self.tracks = {}          # id -> {"box": array(4), "vel": array(4), "lost": int}
        self.next_id = 1

    def update(self, boxes, confs):
        """boxes: (N,4). Trả về list (track_id, box, conf) cho các box đã gán ID."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        confs = np.asarray(confs, float).reshape(-1)
        ids = list(self.tracks)
        # dự đoán vị trí mới = box cũ + vận tốc (mô hình chuyển động đơn giản)
        pred = np.array([self.tracks[i]["box"] + self.tracks[i]["vel"] for i in ids]).reshape(-1, 4)
        iou = iou_matrix(pred, boxes)
        matched_t, matched_d, out = set(), set(), []
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_thresh:
                    tid = ids[r]
                    old = self.tracks[tid]["box"]
                    self.tracks[tid].update(box=boxes[c], vel=0.5 * (boxes[c] - old), lost=0)
                    matched_t.add(tid); matched_d.add(c)
                    out.append((tid, boxes[c], confs[c]))
        for tid in ids:                                   # track không khớp
            if tid not in matched_t:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
        for c in range(len(boxes)):                       # detection mới → track mới
            if c not in matched_d and confs[c] >= self.min_conf:
                self.tracks[self.next_id] = {"box": boxes[c], "vel": np.zeros(4), "lost": 0}
                out.append((self.next_id, boxes[c], confs[c]))
                self.next_id += 1
        return out
