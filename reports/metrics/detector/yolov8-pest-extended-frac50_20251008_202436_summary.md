# สรุปผลการเทรน YOLOv8 รอบ `yolov8-pest-extended-frac50`

- วันที่รัน: 8 ตุลาคม 2025 เวลา 20:24 (ระบบ)
- จำนวน epoch: 15 (early stop patience = 4)
- สัดส่วนข้อมูลที่ใช้: 50% (`fraction=0.5`)
- ขนาดชุดข้อมูล: `data/processed/detection/pests_2xlvx_yolo`
- พารามิเตอร์สำคัญ
  - `imgsz=448`, `batch=16`
  - ใช้ pretrained `yolov8n.pt`
  - อุปกรณ์: CPU (`device=cpu`)

## เมตริกสุดท้าย (Epoch 15)

| Metric | ค่า |
| --- | --- |
| Precision (B) | 0.66173 |
| Recall (B) | 0.04393 |
| mAP50 (B) | 0.04953 |
| mAP50-95 (B) | 0.03320 |
| Val Box Loss | 1.36554 |
| Val Cls Loss | 4.56229 |
| Val DFL Loss | 1.22240 |

> *สังเกต*: การเพิ่ม epoch เป็น 15 และใช้ครึ่งหนึ่งของข้อมูลช่วยให้ mAP50 เพิ่มขึ้นกว่ารุ่น 5 epoch เดิม (~0.02875 → ~0.04953) แม้ recall ยังต่ำอยู่ที่ ~0.044 จำเป็นต้องทดลองปรับจูนเพิ่มเติม (เช่น ใช้ข้อมูลครบชุด, ปรับ augmentation, หรือ reweight class) เพื่อให้โมเดลเห็นตัวอย่างที่ขาดหายและเพิ่มการตรวจจับได้ครอบคลุมยิ่งขึ้น

## ไฟล์ที่เกี่ยวข้อง

- `yolov8-pest-extended-frac50_20251008_202436_results.csv` – บันทึกผลทุก epoch
- `yolov8-pest-extended-frac50_20251008_202436_summary.json` – ค่าเมตริกล่าสุดแบบ JSON
- `yolov8-pest-extended-frac50_20251008_202436_curves.png` – กราฟ loss/metric ตลอดการเทรน
