from ultralytics import YOLO
import json

model = YOLO('models/detector/yolov8-pest-rtx4060ti/weights/best.pt')
results = model.val(data='data/processed/detection/pests_2xlvx_yolo/dataset.yaml', split='val')
metrics = {
    'precision': float(results.results_dict.get('metrics/precision(B)', 0.0)),
    'recall': float(results.results_dict.get('metrics/recall(B)', 0.0)),
    'map50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
    'map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))
}
per_class = []
if results.box and results.box.map_per_class is not None:
    for class_id, name in results.names.items():
        value = float(results.box.map_per_class[class_id])
        per_class.append({'class_id': int(class_id), 'class_name': name, 'map50': value})

print(json.dumps({'metrics': metrics, 'per_class': per_class}, ensure_ascii=False, indent=2))
