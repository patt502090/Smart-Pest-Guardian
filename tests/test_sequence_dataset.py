import pandas as pd

from src.forecasting.dataset import SequenceConfig, SequenceDataset


def test_sequence_dataset_generates_samples():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "location_id": ["A"] * 30,
            "date": dates,
            "temp": range(30),
            "humidity": range(30, 60),
            "count_total": [i % 5 for i in range(30)],
        }
    )
    config = SequenceConfig(
        time_column="date",
        group_columns=["location_id"],
        feature_columns=["temp", "humidity"],
        target_column="count_total",
        input_window=7,
        forecast_horizon=3,
    )
    dataset = SequenceDataset(df, config)
    assert len(dataset) == 21
    features, target, meta = dataset[0]
    assert features.shape == (7, 2)
    assert target.shape == (3,)
