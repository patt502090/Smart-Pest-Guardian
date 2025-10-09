# โครงการระบบตรวจจับและคาดการณ์ศัตรูพืชอัจฉริยะ (Smart Pest Guardian)

> ระบบ AI สองเฟสเพื่อช่วยเกษตรกรไทยตรวจจับศัตรูพืชและคาดการณ์การระบาดล่วงหน้า ด้วยโมเดลตรวจจับภาพ (YOLOv8) และโมเดลลำดับเวลา (LSTM / Temporal Fusion Transformer)

## Quick Links
- Repository: https://github.com/patt502090/Smart-Pest-Guardian
- รายงานฉบับล่าสุด: reports/drafts/final_report.pdf
- แกลเลอรีผลตรวจจับ: results/detections/report_examples/
- Log เทรนและกราฟ: models/detector/yolov8-pest-balanced-ft44/

## Highlights
- เทรน YOLOv8s บนชุด pests_2xlvx_yolo_balanced หลังปรับสมดุลแต่ละคลาส (train 868 / val 140 / test 138 ภาพ)
- ค่าความเชื่อมั่นที่ให้ F1 สูงสุดอยู่ที่ 0.125 (จาก runs/detect/val14/BoxF1_curve.png) พร้อม threshold รายคลาสสำหรับ deploy หน้างาน
- ประเมินบน validation ได้ Precision 0.203 / Recall 0.285 / mAP@0.5 0.236 / mAP@0.5:0.95 0.177; test set ให้ Precision 0.517 / Recall 0.420 / mAP@0.5 0.428
- วิเคราะห์ข้อผิดพลาดด้วย confusion matrix (runs/detect/val14/confusion_matrix_normalized.png) และรวบรวมเคสที่โมเดลพลาด (results/detections/report_examples/failure/) เพื่อชี้แนะแนวทางปรับปรุง

## โครงสร้างไดเรกทอรีโดยรวม
```
.
 data/                   # ชุดข้อมูลดิบและหลัง preprocess (YOLO format)
 models/detector/        # checkpoints, logs, figures จากการเทรน
 reports/drafts/         # เนื้อหารายงาน, รูปประกอบ, ตารางผลลัพธ์
 results/detections/     # ภาพ inference สำหรับรายงาน
 scripts/                # CLI workflow (download / split / train / evaluate)
 src/                    # โมดูลหลักสำหรับ data, detection, forecasting
 tmp/                    # สคริปต์ช่วยงานเฉพาะกิจ (เช่น หา failure cases)
```

## เวิร์กโฟลว์หลัก
1. scripts/download_data.py ดาวน์โหลดและจัดเก็บชุดข้อมูลอัตโนมัติ  
2. scripts/prepare_detection_dataset.py แปลง annotation และจัด train/val/test split  
3. scripts/train_detector.py เทรน YOLOv8 พร้อมบันทึก metrics, loss curve, confusion matrix  
4. scripts/generate_incident_series.py รวมผลตรวจจับเป็น time series แยกพื้นที่/ชนิดศัตรูพืช  
5. scripts/train_forecaster.py เทรน LSTM/TFT เพื่อพยากรณ์ระดับการระบาดล่วงหน้า  
6. scripts/evaluate_system.py รวมผลสองเฟส สร้างรายงานและกราฟสรุป  
7. scripts/run_dashboard.py สาธิตระบบผ่าน Streamlit dashboard

## Model Performance Snapshot
| Dataset | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|-----------|--------|---------|---------------|
| Validation (balanced) | 0.203 | 0.285 | 0.236 | 0.177 |
| Test (hold-out) | 0.517 | 0.420 | 0.428 | 0.346 |

> ผล test สะท้อน distribution ที่ต่างจาก train/val และมีคลาสหายาก จึงควรวางแผนเก็บข้อมูลเพิ่ม ทำ active learning และทวน threshold บนชุดจริง

## Detection Gallery
- ความมั่นใจสูง: results/detections/report_examples/good/01314_jpg.rf.d87f7a1cb9c0f155c01f189963dedf1f.jpg  
- หลายชนิดรวมเฟรม: results/detections/report_examples/good/00160_jpg.rf.62ddc0aeb5dbd3e75513646cab81aa60_aug_4_000.jpg  
- เคสโมเดลสับสน: results/detections/report_examples/failure/00379_jpg.rf.5b6ef3eb58c3d1e431bd653c105a8d4e_aug_0_001.jpg  

## เทคโนโลยีหลัก
- ภาษา: Python 3.10+
- ไลบรารีตรวจจับ: ultralytics (YOLOv8)
- ไลบรารีพยากรณ์: PyTorch, PyTorch Lightning, pytorch-forecasting
- การติดตามการทดลอง: MLflow / Weights & Biases (ตามความเหมาะสม)
- Visualization & Dashboard: Streamlit, Plotly, Altair

## Roadmap (6–11 ต.ค. 2025)
| วัน | Milestone |
|-----|------------|
| 6 ต.ค. | ดาวน์โหลดข้อมูล, ทำ EDA, เตรียม repo + environment |
| 7 ต.ค. | เทรน baseline YOLOv8, ประเมินบน validation |
| 8 ต.ค. | เทรน LSTM/TFT รอบแรก, ปรับ hyperparameter |
| 9 ต.ค. | สร้าง pipeline อัตโนมัติ, ทำ Grad-CAM/SHAP, เก็บเคสศึกษา |
| 10 ต.ค. | เขียนรายงานฉบับเต็ม, เตรียมสไลด์และสคริปต์วิดีโอ |
| 11 ต.ค. | อัดวิดีโอ 5 นาที, ตรวจ checklist, สรุปแพ็กส่ง |

## ขั้นตอนถัดไป
1. ขยายชุดข้อมูลคลาสหายากและทวน threshold บน test set  
2. ทดลอง active learning หรือ semi-supervised เพื่อลด labeling cost  
3. ลดขนาดโมเดล (YOLOv8n) สำหรับ Jetson/Edge device และทดสอบ latency  
4. รวมผลพยากรณ์กับ detection เพื่อสร้าง dashboard เตือนภัยแบบ near real-time  

## ภาคผนวกที่แนบในรายงาน
- สคริปต์หลัก: src/data/balance_dataset.py, src/detection/train_detector.py, tmp/find_failure_samples.py  
- ไฟล์สรุป: dataset_balance_summary.json, test_split_counts.csv, models/detector/yolov8-pest-balanced-ft44/results.csv  
- กราฟอ้างอิง: runs/detect/val14/BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png, BoxPR_curve.png  

พร้อมต่อยอดเก็บข้อมูลเพิ่มและ fine-tune รอบใหม่ เพื่อให้ Smart Pest Guardian เป็นผู้ช่วยเกษตรกรที่แม่นยำและใช้งานได้จริงมากขึ้น
