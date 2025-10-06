"""Collection of evaluation helpers for detection and forecasting tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support


def compute_detection_pr(detections_csv: Path, ground_truth_csv: Path, iou_threshold: float = 0.5) -> Dict[str, float]:
    """Compute overall precision/recall/F1 by matching detections with ground-truth boxes."""
    import torch

    def boxes_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    detections = pd.read_csv(detections_csv)
    ground_truth = pd.read_csv(ground_truth_csv)

    y_true, y_pred = [], []
    for image_id, gt_group in ground_truth.groupby("image_path"):
        det_group = detections[detections["image_path"] == image_id]
        matched = set()
        for _, gt in gt_group.iterrows():
            gt_box = gt[["x1", "y1", "x2", "y2"]].to_numpy()
            gt_class = int(gt["class_id"])
            best_match = None
            best_iou = 0.0
            for det_idx, det in det_group.iterrows():
                if det_idx in matched:
                    continue
                det_box = det[["x1", "y1", "x2", "y2"]].to_numpy()
                iou = boxes_iou(gt_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            if best_match is not None and best_iou >= iou_threshold:
                y_true.append(gt_class)
                y_pred.append(int(best_match["class_id"]))
                matched.add(best_match.name)
            else:
                y_true.append(gt_class)
                y_pred.append(-1)  # miss detection
        for det_idx, det in det_group.iterrows():
            if det_idx not in matched:
                y_true.append(-1)
                y_pred.append(int(det["class_id"]))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_forecasting_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    mae = mean_absolute_error(truth, prediction)
    rmse = np.sqrt(mean_squared_error(truth, prediction))
    mape = float(np.mean(np.abs((truth - prediction) / (truth + 1e-6))) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape}


def summarize_forecast_csv(
    forecast_csv: Path,
    actual_csv: Path,
    target_column: str,
    horizon: int,
) -> Dict[str, float]:
    forecast_df = pd.read_csv(forecast_csv)
    actual_df = pd.read_csv(actual_csv, parse_dates=["date"])
    errors = []
    for _, row in forecast_df.iterrows():
        group = row["group"]
        predicted_values = np.array(eval(row["predicted_values"]))
        forecast_end = row["forecast_end_date"]
        hist = actual_df[actual_df[target_column].notnull()]
        y_true = hist[target_column].values[-horizon:]
        errors.append(compute_forecasting_metrics(y_true, predicted_values))
    return {
        f"avg_{metric}": float(np.mean([err[metric] for err in errors]))
        for metric in errors[0].keys()
    }
