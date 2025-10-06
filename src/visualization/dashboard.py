"""Streamlit dashboard for showcasing detection and forecasting results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_DETECTIONS = Path("results/detections/detections.csv")
DEFAULT_FORECAST = Path("results/forecasting/forecast.csv")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"ไม่พบไฟล์ {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="Smart Pest Guardian", layout="wide")
    st.title("📸 Smart Pest Guardian Dashboard")

    st.sidebar.header("Dataset Selection")
    detections_path = Path(st.sidebar.text_input("ไฟล์ผลตรวจจับ", value=str(DEFAULT_DETECTIONS)))
    forecast_path = Path(st.sidebar.text_input("ไฟล์ผลพยากรณ์", value=str(DEFAULT_FORECAST)))

    detections_df = load_csv(detections_path)
    forecast_df = load_csv(forecast_path)

    tab1, tab2, tab3 = st.tabs(["ผลตรวจจับ", "พยากรณ์", "วิเคราะห์ร่วม"])

    with tab1:
        st.subheader("สถิติเชิงสรุป")
        if not detections_df.empty:
            st.write(detections_df.head())
            agg = detections_df.groupby("class_id").agg(
                detections=("confidence", "count"),
                avg_confidence=("confidence", "mean"),
            )
            st.bar_chart(agg["detections"])
        else:
            st.info("ยังไม่มีข้อมูลการตรวจจับ กรุณารันสคริปต์ตรวจจับก่อน")

    with tab2:
        st.subheader("พยากรณ์การระบาด")
        if not forecast_df.empty:
            st.write(forecast_df.head())
        else:
            st.info("ยังไม่มีข้อมูลการพยากรณ์ กรุณารันสคริปต์เทรนและคำนวณพยากรณ์")

    with tab3:
        if not detections_df.empty and not forecast_df.empty:
            st.subheader("เชื่อมโยงการตรวจจับกับการพยากรณ์")
            st.write("เมื่อตรวจจับศัตรูพืชจำนวนมาก ระบบจะพยากรณ์ระดับความเสี่ยงสูงขึ้นในพื้นที่เดียวกัน")
        else:
            st.info("ต้องมีทั้งข้อมูลตรวจจับและพยากรณ์เพื่อวิเคราะห์ร่วม")


if __name__ == "__main__":
    main()
