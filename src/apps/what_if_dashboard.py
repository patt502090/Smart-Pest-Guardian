"""Interactive what-if simulator for Smart Pest Guardian forecasts."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.forecasting.dataset import SequenceConfig, SequenceDataset, load_and_prepare
from src.forecasting.train_lstm import LSTMForecaster

DEFAULT_MODEL_PATH = Path("models/forecaster/sweeps/hs128_lr0.0005_do0.2/lstm_best.pt")
DEFAULT_DATA_PATH = Path("data/processed/forecasting/pest_incidents_daily_integration.csv")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_group_key(key) -> Tuple:
    if key is None:
        return ("__global__",)
    if isinstance(key, tuple):
        return key
    return (key,)


def format_group_label(group_columns: Sequence[str] | None, group_key: Tuple) -> str:
    if not group_columns:
        return "overall"
    if len(group_columns) == 1:
        return f"{group_columns[0]}={group_key[0]}"
    return ", ".join(f"{col}={val}" for col, val in zip(group_columns, group_key))


def load_lstm_model(model_path: Path, device: str = DEFAULT_DEVICE) -> Tuple[LSTMForecaster, SequenceConfig, torch.device]:
    """Load an LSTM checkpoint and rebuild its configuration."""
    if not model_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    cfg_dict = checkpoint["config"]
    seq_config = SequenceConfig(
        time_column=cfg_dict.get("time_column", "date"),
        group_columns=cfg_dict.get("group_columns"),
        feature_columns=cfg_dict.get("feature_columns"),
        target_column=cfg_dict.get("target_column", "count_total"),
        input_window=cfg_dict["input_window"],
        forecast_horizon=cfg_dict["forecast_horizon"],
    )

    model = LSTMForecaster(
        num_features=cfg_dict["num_features"],
        hidden_size=cfg_dict["hidden_size"],
        num_layers=cfg_dict["num_layers"],
        horizon=cfg_dict["horizon"],
        dropout=cfg_dict["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model_device = torch.device(device)
    model.to(model_device)
    model.eval()
    return model, seq_config, model_device


def prepare_dataframe(csv_path: Path, config: SequenceConfig) -> pd.DataFrame:
    """Load and prepare the base dataframe used for inference."""
    return load_and_prepare(csv_path, config).copy()


def compute_group_forecast(
    model: LSTMForecaster,
    df: pd.DataFrame,
    config: SequenceConfig,
    device: torch.device,
    target_group: Tuple,
) -> np.ndarray:
    """Generate a forecast for the specified group key."""
    dataset = SequenceDataset(df, config)
    last_indices: Dict[Tuple, Tuple[int, Iterable]] = {}
    for idx, (group_key_raw, meta_timestamp) in enumerate(dataset.groups):
        norm_key = _normalize_group_key(group_key_raw)
        last_indices[norm_key] = (idx, group_key_raw, meta_timestamp)

    if target_group not in last_indices:
        available = ", ".join(format_group_label(config.group_columns, key) for key in last_indices)
        raise KeyError(f"ไม่พบกลุ่ม {target_group}. มีกลุ่มที่ใช้ได้: {available}")

    idx, raw_group_key, _ = last_indices[target_group]
    features = torch.tensor(dataset.data[idx], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        preds_norm = model(features).cpu().numpy().flatten()

    scaler = dataset.get_scaler(raw_group_key)
    target_mean = scaler["mean"][-1]
    target_std = scaler["std"][-1]
    preds = preds_norm * target_std + target_mean
    return preds


def apply_scenario(
    df: pd.DataFrame,
    config: SequenceConfig,
    group_key: Tuple,
    recent_days: int,
    incident_multiplier: float,
    confidence_shift: float,
) -> pd.DataFrame:
    """Apply what-if adjustments to the dataframe for the selected group."""
    if not config.group_columns:
        group_mask = pd.Series(True, index=df.index)
    elif len(config.group_columns) == 1:
        group_mask = df[config.group_columns[0]] == group_key[0]
    else:
        group_mask = pd.Series(True, index=df.index)
        for col, value in zip(config.group_columns, group_key):
            group_mask &= df[col] == value

    group_df = df.loc[group_mask].sort_values(config.time_column)
    if group_df.empty:
        raise ValueError("ไม่มีข้อมูลในกลุ่มที่เลือกสำหรับ scenario")

    effective_days = min(recent_days, len(group_df))
    idx_to_update = group_df.tail(effective_days).index
    scenario_df = df.copy()

    row_positions = scenario_df.index.get_indexer(idx_to_update)
    target_cols = [idx for idx, col in enumerate(scenario_df.columns) if col == config.target_column]
    if target_cols:
        target_values = scenario_df.iloc[row_positions, target_cols].to_numpy(dtype=float)
        adjusted = np.clip(target_values * incident_multiplier, a_min=0.0, a_max=None)
        for col_offset, col_pos in enumerate(target_cols):
            scenario_df.iloc[row_positions, col_pos] = adjusted[:, col_offset]

    if "mean_confidence" in scenario_df.columns:
        confidence_slice = scenario_df.loc[idx_to_update, "mean_confidence"].astype(float)
        scenario_df.loc[idx_to_update, "mean_confidence"] = (
            confidence_slice + confidence_shift
        ).clip(lower=0.0, upper=1.0)

    return scenario_df


def build_action_recommendation(baseline: np.ndarray, scenario: np.ndarray) -> Tuple[str, str]:
    """Create an action recommendation comparing two forecast trajectories."""
    baseline_mean = float(np.mean(baseline))
    scenario_mean = float(np.mean(scenario))
    delta_mean = scenario_mean - baseline_mean
    change_pct = (delta_mean / baseline_mean * 100.0) if baseline_mean > 1e-6 else np.nan

    if baseline_mean < 1.0 and scenario_mean < 1.5:
        headline = "ความเสี่ยงต่ำ"
        details = "คาดการณ์การระบาดยังต่ำกว่า 1.5 ตัว/วัน สามารถติดตามสถานการณ์ต่อเนื่อง"
    elif change_pct != change_pct:  # NaN
        if delta_mean > 5:
            headline = "เตรียมแผนรับมือ"
            details = "การเพิ่มขึ้นแบบสัมบูรณ์สูง ควรเสริมกับดัก/พ่นสารตามความเหมาะสม"
        else:
            headline = "จับตาดูใกล้ชิด"
            details = "ข้อมูลฐานต่ำมาก ใช้การเฝ้าระวังเชิงรุกเพื่อยืนยันแนวโน้ม"
    elif change_pct >= 30 or scenario_mean >= 10:
        headline = "เตรียมปฏิบัติการควบคุม"
        details = "ปริมาณคาดการณ์เพิ่มขึ้นมากกว่า 30% หรือสูงกว่า 10 ตัว/วัน แนะนำวางแผนพ่นสารและเสริมมาตรการควบคุม"
    elif change_pct >= 15 or scenario_mean >= 6:
        headline = "เพิ่มความถี่การสำรวจ"
        details = "การระบาดมีแนวโน้มเพิ่ม 15-30% ควรปรับรอบการสำรวจและเตรียมมาตรการสำรอง"
    elif change_pct >= 5:
        headline = "มีสัญญาณขยับขึ้น"
        details = "พบสัญญาณเตือนเล็กน้อย เพิ่มกับดัก/การสำรวจเชิงรุกเพื่อยืนยัน"
    else:
        headline = "แนวโน้มทรงตัว"
        details = "การคาดการณ์ใกล้เคียงฐานเดิม รักษามาตรการเดิมและติดตามตัวเลข"

    return headline, details


def render_dashboard():
    st.set_page_config(page_title="Smart Pest Guardian – Scenario Simulator", layout="wide")
    st.title("🐛 Smart Pest Guardian – Scenario Simulator")
    st.caption("ทดลองสถานการณ์สมมุติและดูว่าแนวโน้มการระบาดจะเปลี่ยนไปอย่างไร")

    sidebar = st.sidebar
    sidebar.header("Scenario Controls")

    model_path = Path(sidebar.text_input("โมเดล LSTM", value=str(DEFAULT_MODEL_PATH)))
    data_path = Path(sidebar.text_input("ไฟล์ข้อมูล", value=str(DEFAULT_DATA_PATH)))
    device_choice = sidebar.selectbox("อุปกรณ์", options=[DEFAULT_DEVICE, "cpu"], index=0)

    try:
        model, seq_config, device = load_lstm_model(model_path, device_choice)
        base_df = prepare_dataframe(data_path, seq_config)
    except Exception as exc:  # noqa: BLE001
        st.error(f"ไม่สามารถโหลดโมเดล/ข้อมูลได้: {exc}")
        st.stop()

    group_cols = seq_config.group_columns or []
    if group_cols:
        unique_groups = (
            base_df[group_cols[0]].sort_values().unique() if len(group_cols) == 1 else base_df[group_cols].drop_duplicates().itertuples(index=False, name=None)
        )
        if len(group_cols) == 1:
            selection = sidebar.selectbox("จุดติดตั้ง", options=list(unique_groups))
            group_key = (selection,)
        else:
            selection = sidebar.selectbox("จุดติดตั้ง", options=list(unique_groups))
            group_key = tuple(selection)
    else:
        selection = "overall"
        group_key = ("__global__",)

    horizon = seq_config.forecast_horizon
    default_days = min(3, horizon)
    recent_days = sidebar.slider("จำนวนวันที่ต้องการปรับ", min_value=1, max_value=horizon, value=default_days)
    multiplier = sidebar.slider("ตัวคูณจำนวนศัตรูพืช (ล่าสุด)", min_value=0.2, max_value=3.0, value=1.4, step=0.1)
    confidence_shift = sidebar.slider("ปรับ mean_confidence", min_value=-0.3, max_value=0.3, value=0.05, step=0.05)

    baseline_forecast = compute_group_forecast(model, base_df, seq_config, device, group_key)
    scenario_df = apply_scenario(base_df, seq_config, group_key, recent_days, multiplier, confidence_shift)
    scenario_forecast = compute_group_forecast(model, scenario_df, seq_config, device, group_key)

    days = np.arange(1, len(baseline_forecast) + 1)
    comparison_df = pd.DataFrame(
        {
            "Day": days,
            "Baseline": baseline_forecast,
            "Scenario": scenario_forecast,
        }
    )
    comparison_df["Δ"] = comparison_df["Scenario"] - comparison_df["Baseline"]
    comparison_df["Δ%"] = np.where(
        comparison_df["Baseline"].abs() > 1e-6,
        comparison_df["Δ"] / comparison_df["Baseline"] * 100,
        np.nan,
    )

    col_chart, col_metrics = st.columns([2, 1])
    with col_chart:
        chart_df = comparison_df.set_index("Day")[["Baseline", "Scenario"]]
        st.line_chart(chart_df)
    with col_metrics:
        headline, details = build_action_recommendation(baseline_forecast, scenario_forecast)
        st.metric("เฉลี่ย Scenario", f"{np.mean(scenario_forecast):.2f} ตัว/วัน", delta=f"{np.mean(scenario_forecast) - np.mean(baseline_forecast):+.2f}")
        st.metric("เปลี่ยนแปลง (%)", f"{np.nanmean(comparison_df['Δ%']):.1f}%")
        st.subheader(headline)
        st.write(details)

    st.subheader("รายละเอียดการคาดการณ์")
    st.dataframe(
        comparison_df.style.format({"Baseline": "{:.2f}", "Scenario": "{:.2f}", "Δ": "{:+.2f}", "Δ%": "{:+.1f}"})
    )

    with st.expander("ข้อมูลอ้างอิง" ):
        st.json(asdict(seq_config), expanded=False)
        st.write("กลุ่มที่เลือก:", format_group_label(seq_config.group_columns, group_key))
        st.write(
            "ปรับข้อมูลล่าสุด",
            {
                "recent_days": recent_days,
                "incident_multiplier": multiplier,
                "confidence_shift": confidence_shift,
            },
        )

    st.caption(
        "เคล็ดลับ: บันทึกกราฟ/ตารางเหล่านี้เพื่อแทรกในรายงานหรือวิดีโอเดโม และทดลองหลายสถานการณ์เพื่อเน้นความสามารถเชิงคาดการณ์"
    )


def main():
    render_dashboard()


if __name__ == "__main__":
    main()
