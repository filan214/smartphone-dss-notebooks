# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tahap 5 — Rekayasa Fitur Supervised
#
# Tahap ini menerapkan D3 (split temporal), D5 (ablation Google Trends), dan D6
# (Fourier musiman) dengan fitur lag/rolling masa-lalu saja untuk mencegah leakage.

# %%
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# %matplotlib inline

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.yaml").exists())
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
np.random.seed(SEED)

def rp(*parts: str) -> Path:
    """Resolve a repo-relative path (parts are path fragments under the repo root)."""
    return ROOT.joinpath(*parts)

BRAND_COLORS = {"OPPO": "#0072B2", "Realme": "#D55E00", "Samsung": "#009E73",
                "Vivo": "#CC79A7", "Xiaomi": "#E69F00"}
ALGO_COLORS = {"SARIMAX": "#0072B2", "RF": "#D55E00", "LSTM": "#009E73"}
plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (9, 4.5),
                     "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 12})

# %% [markdown]
# ## Input Tahap 2 dan 3
#
# Grid mingguan lengkap dipertahankan sebelum fitur dibuat. Cache Google Trends
# hanya digabung berdasarkan merek dan minggu; notebook ini tidak melakukan fetch jaringan.

# %%
weekly = pd.read_parquet(rp("data", "interim", "weekly_store_brand.parquet"))
gt = pd.read_csv(rp("data", "interim", "google_trends.csv"), parse_dates=["week_start"])
weekly["week_start"] = pd.to_datetime(weekly["week_start"])
gt["week_start"] = pd.to_datetime(gt["week_start"])

week_grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
assert len(week_grid) == 184, f"grid mingguan berisi {len(week_grid)} minggu, diharapkan 184"
assert set(gt["week_start"]) == set(week_grid), "kalender Google Trends tidak selaras dengan data mingguan"
assert gt.duplicated(["brand", "week_start"]).sum() == 0, "cache Google Trends duplikat"

series_keys = sorted(weekly[["store", "brand"]].drop_duplicates().itertuples(index=False, name=None))
assert len(series_keys) == 20, f"{len(series_keys)} deret, diharapkan 20"
assert weekly.groupby(["store", "brand"])["week_start"].nunique().eq(len(week_grid)).all()
assert not gt["gt_index"].isna().any(), "cache Google Trends memuat gt_index kosong"
print(f"Input selaras: {len(series_keys)} deret × {len(week_grid)} minggu.")

# %% [markdown]
# ## Fungsi fitur dan split temporal
#
# Semua lag dan rolling menggunakan target historis. Cutoff D3 dihitung dari grid
# 184 minggu, lalu digunakan kembali sesudah baris warm-up dihapus.

# %%
LEBARAN_DATES = pd.to_datetime(["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31"])
RAMADAN_WEEKS_BEFORE = 4
LEBARAN_WINDOW_WEEKS = 1
HARBOLNAS_DATES_MMDD = [(11, 11), (12, 12)]

def _week_monday(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("W").start_time

def add_lag_roll_features(df: pd.DataFrame, lags: int, rolling_windows: list[int]) -> pd.DataFrame:
    u = df["units"]
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = u.shift(k)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = u.rolling(w).mean().shift(1)
    df[f"roll_std_{rolling_windows[0]}"] = u.rolling(rolling_windows[0]).std().shift(1)
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ws = df["week_start"]
    df["weekofyear"] = ws.dt.isocalendar().week.astype(int)
    df["month"] = ws.dt.month.astype(int)
    lebaran_mondays = [_week_monday(day) for day in LEBARAN_DATES]
    lebaran_window, ramadan = set(), set()
    for monday in lebaran_mondays:
        for offset in range(-LEBARAN_WINDOW_WEEKS, LEBARAN_WINDOW_WEEKS + 1):
            lebaran_window.add(monday + pd.Timedelta(weeks=offset))
        for offset in range(1, RAMADAN_WEEKS_BEFORE + 1):
            ramadan.add(monday - pd.Timedelta(weeks=offset))
    years = range(int(ws.dt.year.min()), int(ws.dt.year.max()) + 1)
    harbolnas = {
        _week_monday(pd.Timestamp(year=year, month=month, day=day))
        for year in years for month, day in HARBOLNAS_DATES_MMDD
    }
    df["is_ramadan"] = ws.isin(ramadan).astype(int)
    df["is_lebaran_window"] = ws.isin(lebaran_window).astype(int)
    df["is_harbolnas"] = ws.isin(harbolnas).astype(int)
    df["year_trend"] = np.arange(len(df))
    return df

def add_fourier_features(df: pd.DataFrame, period: int, harmonics: int) -> pd.DataFrame:
    time_index = np.arange(len(df))
    for harmonic in range(1, harmonics + 1):
        df[f"fourier_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * time_index / period)
        df[f"fourier_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * time_index / period)
    return df

def build_features_for_series(series_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = series_df.sort_values("week_start").reset_index(drop=True).copy()
    df = add_lag_roll_features(df, int(cfg["lags"]), list(cfg["rolling_windows"]))
    df = add_calendar_features(df)
    return add_fourier_features(df, int(cfg["seasonal_period"]), int(cfg["fourier_harmonics"]))

def build_all_features(weekly: pd.DataFrame, gt: pd.DataFrame, cfg: dict, processed_dir: Path) -> dict[str, pd.DataFrame]:
    merged = weekly.merge(gt[["week_start", "brand", "gt_index"]], on=["brand", "week_start"], how="left", validate="many_to_one")
    assert not merged["gt_index"].isna().any(), "join Google Trends menyisakan nilai kosong"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for (store, brand), subset in merged.groupby(["store", "brand"], sort=True):
        features = build_features_for_series(subset, cfg)
        features.to_parquet(processed_dir / f"features_{store}_{brand}.parquet", index=False)
        out[f"{store}|{brand}"] = features
    return out

def resolve_feature_columns(all_cols, cfg: dict, variant: str) -> list[str]:
    resolved = []
    for entry in cfg["feature_variants"][variant]:
        if entry.endswith("*"):
            resolved.extend(column for column in all_cols if column.startswith(entry[:-1]))
        elif entry in all_cols:
            resolved.append(entry)
        else:
            raise KeyError(f"Kolom fitur '{entry}' (varian {variant}) tak ada di features.")
    return list(dict.fromkeys(resolved))

def make_supervised(features_df: pd.DataFrame, cfg: dict, variant: str = "baseline") -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    columns = resolve_feature_columns(features_df.columns, cfg, variant)
    subset = features_df.dropna(subset=columns).reset_index(drop=True)
    return subset[columns].copy(), subset["units"].astype(float).copy(), pd.DatetimeIndex(subset["week_start"])

def train_cutoff_week(weeks: pd.DatetimeIndex, train_ratio: float = 0.8) -> pd.Timestamp:
    unique_weeks = pd.DatetimeIndex(sorted(pd.Index(weeks).unique()))
    return unique_weeks[int(np.floor(len(unique_weeks) * train_ratio))]

def temporal_masks(week_series: pd.Series, train_ratio: float = 0.8, cutoff: pd.Timestamp | None = None) -> tuple[pd.Series, pd.Series]:
    if cutoff is None:
        cutoff = train_cutoff_week(pd.DatetimeIndex(week_series), train_ratio)
    is_test = week_series >= cutoff
    return ~is_test, is_test

def fit_scaler_on_train(X: pd.DataFrame, index: pd.DatetimeIndex, cutoff: pd.Timestamp):
    from sklearn.preprocessing import MinMaxScaler
    mask_train, _ = temporal_masks(pd.Series(index), cutoff=cutoff)
    scaler = MinMaxScaler()
    scaler.fit(X.loc[mask_train.to_numpy()])
    return scaler

# %% [markdown]
# ## Bangun 20 matriks fitur
#
# Setiap parquet tetap menyimpan 184 baris kalender; hanya `make_supervised` yang
# membuang delapan baris warm-up sesuai lag terpanjang.

# %%
features_by_series = build_all_features(weekly, gt, CFG, rp("data", "processed"))
assert set(features_by_series) == {f"{store}|{brand}" for store, brand in series_keys}
print(f"Menulis {len(features_by_series)} file fitur ke {rp('data', 'processed')}")

# %% [markdown]
# ## Pemeriksaan representatif: Toko_A × Xiaomi
#
# Perbandingan dua varian memastikan bahwa D5 hanya menambahkan `gt_index`,
# sementara D3 dan fitur D6 sama untuk kedua matriks.

# %%
representative = features_by_series["Toko_A|Xiaomi"].copy()
X_baseline, y_baseline, index_baseline = make_supervised(representative, CFG, "baseline")
X_gt, y_gt, index_gt = make_supervised(representative, CFG, "gt")
cutoff = train_cutoff_week(week_grid, float(CFG["train_ratio"]))
train_mask, test_mask = temporal_masks(pd.Series(index_baseline), cutoff=cutoff)
scaler_baseline = fit_scaler_on_train(X_baseline, index_baseline, cutoff)

display(representative[["week_start", "units", "gt_index", "lag_1", "lag_8", "roll_mean_4", "roll_mean_8"]].head(12))
print({
    "baseline_columns": X_baseline.columns.tolist(),
    "gt_only_columns": sorted(set(X_gt.columns) - set(X_baseline.columns)),
    "d3_full_grid": {"train": int((week_grid < cutoff).sum()), "test": int((week_grid >= cutoff).sum())},
    "d3_supervised": {"train": int(train_mask.sum()), "test": int(test_mask.sum())},
    "scaler_train_min": scaler_baseline.data_min_.tolist(),
    "scaler_train_max": scaler_baseline.data_max_.tolist(),
})

# %%
import fnmatch
feat = pd.read_parquet(rp("data", "processed", "features_Toko_A_Xiaomi.parquet"))
columns = feat.columns.tolist()
for pattern in CFG["feature_variants"]["gt"]:
    assert any(fnmatch.fnmatch(column, pattern) for column in columns), f"pola fitur '{pattern}' tak ada"
feature_files = list(rp("data", "processed").glob("features_*.parquet"))
assert len(feature_files) == 20, f"{len(feature_files)} file fitur, diharapkan 20"
ordered = feat.sort_values("week_start").reset_index(drop=True)
assert np.allclose(ordered["lag_1"].to_numpy()[1:], ordered["units"].to_numpy()[:-1], equal_nan=True)
print("✅ Tahap 5: 20 file fitur; pola feature_variants[gt] hadir; lag_1 anti-bocor")

expected_keys = {f"{store}|{brand}" for store, brand in series_keys}
found_keys = set()
expected_base_columns = resolve_feature_columns(feat.columns, CFG, "baseline")
expected_gt_columns = resolve_feature_columns(feat.columns, CFG, "gt")
for feature_path in feature_files:
    current = pd.read_parquet(feature_path).sort_values("week_start").reset_index(drop=True)
    found_keys.add(f"{current.loc[0, 'store']}|{current.loc[0, 'brand']}")
    assert len(current) == 184, f"{feature_path.name}: {len(current)} baris, diharapkan 184"
    assert pd.DatetimeIndex(current["week_start"]).equals(week_grid), f"{feature_path.name}: kalender tidak selaras"
    assert {"store", "brand", "week_start", "units", "gt_index", *expected_gt_columns}.issubset(current.columns)
    assert np.allclose(current["lag_1"].iloc[1:].to_numpy(), current["units"].iloc[:-1].to_numpy(), equal_nan=True)
    expected_roll_mean_4 = np.array([current["units"].iloc[row - 4:row].mean() for row in range(4, len(current))])
    assert np.allclose(current["roll_mean_4"].iloc[4:].to_numpy(), expected_roll_mean_4, equal_nan=True)
assert found_keys == expected_keys, "ada deret fitur yang hilang atau tidak diharapkan"

assert "gt_index" not in expected_base_columns and "gt_index" in expected_gt_columns
assert set(expected_gt_columns) - set(expected_base_columns) == {"gt_index"}
assert len(week_grid) == 184 and (week_grid < cutoff).sum() == 147 and (week_grid >= cutoff).sum() == 37
assert len(X_baseline) == len(X_gt) == 176
assert index_baseline.equals(index_gt) and y_baseline.equals(y_gt)
assert int(train_mask.sum()) == 139 and int(test_mask.sum()) == 37
assert np.array_equal(test_mask.to_numpy(), index_baseline >= cutoff)
assert X_gt.columns.tolist() == [*X_baseline.columns, "gt_index"]

independent_train = X_baseline.loc[train_mask.to_numpy()]
assert np.allclose(scaler_baseline.data_min_, independent_train.min().to_numpy())
assert np.allclose(scaler_baseline.data_max_, independent_train.max().to_numpy())
assert np.any(scaler_baseline.data_min_ != X_baseline.min().to_numpy()) or np.any(scaler_baseline.data_max_ != X_baseline.max().to_numpy())
print("✅ Penguatan: 20×184, rolling t−1, D3 147/37, D5 hanya gt_index, dan scaler train-only tervalidasi")
