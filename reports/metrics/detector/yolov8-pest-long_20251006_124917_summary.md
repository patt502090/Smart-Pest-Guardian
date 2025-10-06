# สรุปผลการเทรน YOLOv8 รุ่น `yolov8-pest-long`

- วันที่รัน: 6 ตุลาคม 2025 เวลา 12:49 (ระบบ)
- จำนวน epoch: 5
- ขนาดชุดข้อมูล: `data/processed/detection/pests_2xlvx_yolo`
- พารามิเตอร์สำคัญ
  - `imgsz=512`, `batch=8`
  - ใช้ pretrained `yolov8n.pt`
  - ความเชื่อมั่นขั้นต่ำ 0.25 ระหว่างอินเฟอเรนซ์ทดสอบ

## เมตริกสุดท้าย (Epoch 5)

| Metric | ค่า |
| --- | --- |
| Precision (B) | 0.63221 |
| Recall (B) | 0.04427 |
| mAP50 (B) | 0.02875 |
| mAP50-95 (B) | 0.01614 |
| Val Box Loss | 1.37868 |
| Val Cls Loss | 4.46644 |
| Val DFL Loss | 1.21686 |

> *หมายเหตุ*: ค่า Recall และ mAP ยังต่ำ เนื่องจากจำนวน epoch น้อยและข้อมูลมีความหลากหลายสูง แนะนำให้เพิ่มรอบการเทรนและปรับสมดุลข้อมูลเพิ่มเติมในงานต่อไป

## ไฟล์ที่เกี่ยวข้อง

- `yolov8-pest-long_20251006_124917_results.csv` – รายละเอียดต่อ epoch
- `yolov8-pest-long_20251006_124917_summary.json` – ค่าเมตริกสุดท้ายในรูป JSON
- `yolov8-pest-long_20251006_124917_curves.png` – กราฟ loss/metric ตลอดการเทรน
