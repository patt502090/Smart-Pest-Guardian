# YOLOv8 Pest Detector — RTX 4060 Ti Run (2025-10-09)

## Configuration
- **Model**: `yolov8s.pt` (pretrained)
- **Dataset**: `data/processed/detection/pests_2xlvx_yolo/dataset.yaml`
- **Epoch budget**: 180 (stopped at epoch 115 by early stopping with `patience=40`)
- **Batch size**: 28 @ 640×640
- **Optimizer**: auto-selected AdamW (lr≈3.1e-4 initial with cosine schedule)
- **Augmentations**: Ultralytics defaults (mosaic on for 10 epochs, mixup off)
- **Hardware**: RTX 4060 Ti (AMP enabled)

## Training Timeline
- Losses converged steadily through ~80 epochs; validation metrics plateaued after epoch 75.
- Early stopping triggered when no mAP50 gain was observed for 40 consecutive epochs.
- Total wall-clock time ≈ 2.0 hours.

## Key Metrics (best epoch 75)
| Metric | Value |
| --- | --- |
| Precision (B) | 0.224 |
| Recall (B) | 0.217 |
| mAP50 (B) | 0.249 |
| mAP50-95 (B) | 0.181 |

### Per-class highlights (best weights)
- **Strong classes**: `Creatonotus transiens` (mAP50 ≈ 0.71), `Diaphania indica` (≈ 1.00 with very low support), `Maruca testulalis` (≈ 0.80), `Cnaphalocrocis medinalis` (≈ 0.64).
- **Weak / zero detections**: `Gryllidae`, `Sirthenea flavipes`, `Spoladea recurvalis`, `Trichoptera`, `Holotrichia oblita`, `Sogatella furcifera`, `Spodoptera exigua` — all show near-zero precision/recall.
- **Moderate**: `Mamestra brassicae` (~0.37), `Sesamia inferens` (~0.19), `Nilaparvata` (~0.14).

## Interpretation
- Overall mAP50 ≈ 0.25 is a noticeable lift versus earlier CPU-bound runs but still below a "production-ready" threshold (typically ≥0.45 for balanced detection tasks).
- Precision and recall remain low because numerous rare classes do not have enough positive samples; the detector over-fits to dominant species.
- Confusion matrix visuals (see `models/detector/yolov8-pest-rtx4060ti/confusion_matrix*.png`) confirm frequent confusion among hopper/planthopper categories and near-misses on noctuid moths.

## Recommendations
1. **Balance the dataset**: Augment or up-sample under-represented species (e.g., targeted data collection or synthetic augmentation) before retraining.
2. **Curriculum-style fine-tuning**: Start from the current `best.pt` and fine-tune with class-balanced sampling or focal loss to prioritise difficult classes.
3. **Hyperparameter sweep**: Explore higher image sizes (768–896) and longer patience (≥80) once data is balanced; consider switching to SGD with warm restarts for better recall.
4. **Validation granularity**: Create a hold-out test set to measure generalisation; current validation may still be optimistic where classes are scarce.
5. **Post-processing**: Tune confidence thresholds per class once recall improves; currently the detector misses many instances regardless of confidence.

## Next Steps
- Prepare a new training job after data curation, aiming for ≥10–15 images per class and re-run for 250–300 epochs with staged patience.
- Track improvements by comparing future `*_summary.md` files against this baseline.
