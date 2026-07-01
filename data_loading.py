from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_path: Path
    label_col: str
    test_path: Path | None = None
    drop_cols: tuple[str, ...] = ()
    train_cap: int = 50000
    test_cap: int = 20000
    official_split: bool = False


EDGE_IIOT_LEAKAGE_COLUMNS = (
    "Attack_label",
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "http.file_data",
    "http.request.uri.query",
    "http.request.full_uri",
    "http.referer",
    "tcp.payload",
    "mqtt.msg",
    "dns.qry.name",
)


def paper1_dataset_specs(data_root: str | Path) -> list[DatasetSpec]:
    root = Path(data_root)
    return [
        DatasetSpec(
            name="CICIoT23_multiclass",
            train_path=root / "CICIOT23" / "CICIOT23" / "train" / "train.csv",
            test_path=root / "CICIOT23" / "CICIOT23" / "test" / "test.csv",
            label_col="label",
            train_cap=50000,
            test_cap=20000,
            official_split=True,
        ),
        DatasetSpec(
            name="EdgeIIoT_multiclass",
            train_path=root / "Edge-IIoTset" / "Edge-IIoTset dataset" / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv",
            label_col="Attack_type",
            drop_cols=EDGE_IIOT_LEAKAGE_COLUMNS,
            train_cap=50000,
            test_cap=20000,
        ),
        DatasetSpec(
            name="RT_IOT2022_attack_type",
            train_path=root / "RT_IOT2022" / "RT_IOT2022.csv",
            label_col="Attack_type",
            train_cap=45000,
            test_cap=18000,
        ),
    ]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.replace([np.inf, -np.inf], np.nan)


def normalize_label(s: pd.Series) -> pd.Series:
    return s.fillna("Normal").astype(str).str.strip().replace({"": "Normal", "nan": "Normal"})


def stratified_cap(df: pd.DataFrame, label_col: str, cap: int, seed: int) -> pd.DataFrame:
    df = df[df[label_col].notna()].copy()
    if len(df) <= cap:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    counts = df[label_col].value_counts()
    weights = np.sqrt(counts)
    allocation = (weights / weights.sum() * cap).astype(int).clip(lower=2)
    while allocation.sum() > cap:
        allocation[allocation.idxmax()] -= 1
    while allocation.sum() < cap:
        for cls in counts.index:
            if allocation.sum() >= cap:
                break
            if allocation[cls] < counts[cls]:
                allocation[cls] += 1
    parts = []
    for cls, n_rows in allocation.items():
        group = df[df[label_col] == cls]
        parts.append(group.sample(n=min(int(n_rows), len(group)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(spec: DatasetSpec, seed: int):
    train_df = clean_columns(pd.read_csv(spec.train_path, low_memory=False))
    train_df[spec.label_col] = normalize_label(train_df[spec.label_col])
    if spec.official_split:
        if spec.test_path is None:
            raise ValueError(f"{spec.name} requires a test_path for official_split=True")
        test_df = clean_columns(pd.read_csv(spec.test_path, low_memory=False))
        test_df[spec.label_col] = normalize_label(test_df[spec.label_col])
        train_counts = train_df[spec.label_col].value_counts()
        test_counts = test_df[spec.label_col].value_counts()
        common = sorted(set(train_counts[train_counts >= 30].index) & set(test_counts[test_counts >= 30].index))
        train_df = train_df[train_df[spec.label_col].isin(common)]
        test_df = test_df[test_df[spec.label_col].isin(common)]
        train = stratified_cap(train_df, spec.label_col, spec.train_cap, seed)
        test = stratified_cap(test_df, spec.label_col, spec.test_cap, seed + 99)
    else:
        total = spec.train_cap + spec.test_cap
        df = stratified_cap(train_df, spec.label_col, total, seed)
        train, test = train_test_split(
            df,
            test_size=spec.test_cap / total,
            stratify=df[spec.label_col],
            random_state=seed,
        )
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    y_train = train[spec.label_col].to_numpy()
    y_test = test[spec.label_col].to_numpy()
    drop = [spec.label_col, *spec.drop_cols]
    X_train = train.drop(columns=drop, errors="ignore")
    X_test = test.drop(columns=drop, errors="ignore")
    return X_train, X_test, y_train, y_test
