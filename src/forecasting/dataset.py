"""Utilities to transform aggregated detections into training-ready time series."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


@dataclass
class SequenceConfig:
    time_column: str = "date"
    group_columns: Optional[List[str]] = None
    feature_columns: Optional[List[str]] = None
    target_column: str = "count_total"
    input_window: int = 14
    forecast_horizon: int = 7


def load_and_prepare(
    csv_path: Path,
    config: SequenceConfig,
    fill_method: str = "ffill",
    add_time_features: bool = True,
) -> pd.DataFrame:
    """Load aggregated incidents and produce a feature-complete dataframe."""
    if not csv_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[config.time_column])
    df.sort_values(config.time_column, inplace=True)

    group_cols = config.group_columns or []
    if add_time_features:
        df["dayofweek"] = df[config.time_column].dt.dayofweek
        df["month"] = df[config.time_column].dt.month
        df["weekofyear"] = df[config.time_column].dt.isocalendar().week.astype(int)

    if fill_method:
        for _, group_df in df.groupby(group_cols or [config.target_column]):
            if fill_method == "ffill":
                df.loc[group_df.index] = group_df.fillna(method="ffill").fillna(0)
            elif fill_method == "zero":
                df.loc[group_df.index] = group_df.fillna(0)
            else:
                raise ValueError(f"ไม่รองรับ fill_method: {fill_method}")

    feature_cols = config.feature_columns
    if feature_cols is None:
        ignore_cols = set(group_cols + [config.time_column, config.target_column])
        feature_cols = [col for col in df.columns if col not in ignore_cols]
        console.log(f"เลือก feature columns อัตโนมัติ: {feature_cols}")

    df = df[group_cols + [config.time_column] + feature_cols + [config.target_column]]
    return df


class SequenceDataset:
    """Simple sliding-window dataset for LSTM forecasting."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: SequenceConfig,
        scaler: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config
        self.group_columns = config.group_columns or []
        self.feature_columns = config.feature_columns or [
            col
            for col in df.columns
            if col not in set(self.group_columns + [config.time_column, config.target_column])
        ]
        self.target_column = config.target_column
        self.input_window = config.input_window
        self.forecast_horizon = config.forecast_horizon

        data, targets = [], []
        groups = []
        for group_key, group_df in df.groupby(self.group_columns) if self.group_columns else [(None, df)]:
            features = group_df[self.feature_columns + [self.target_column]].to_numpy(dtype=np.float32)
            timestamps = group_df[config.time_column].to_numpy()
            if scaler is None:
                f_mean = features.mean(axis=0)
                f_std = features.std(axis=0) + 1e-6
            else:
                f_mean = scaler.get("mean")
                f_std = scaler.get("std")
            norm_features = (features - f_mean) / f_std
            for idx in range(len(group_df) - self.input_window - self.forecast_horizon + 1):
                window = norm_features[idx : idx + self.input_window, : len(self.feature_columns)]
                target = norm_features[
                    idx + self.input_window : idx + self.input_window + self.forecast_horizon,
                    -1,
                ]
                data.append(window)
                targets.append(target)
                groups.append((group_key, timestamps[idx + self.input_window + self.forecast_horizon - 1]))
        if not data:
            raise ValueError("ข้อมูลไม่เพียงพอสำหรับสร้าง sequence โปรดลด input_window หรือเพิ่มข้อมูล")
        self.data = np.stack(data)
        self.targets = np.stack(targets)
        self.groups = groups

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index], self.groups[index]
