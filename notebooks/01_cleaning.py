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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tahap 1 — Pembersihan transaksi POS
#
# Notebook ini menjalankan kontrak Tahap 1: CSV transaksi mentah diubah menjadi
# transaksi Paid yang terdeduplikasi, memiliki merek yang valid, dan disimpan sebagai
# `data/interim/transactions_clean.parquet` untuk Tahap 2.

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
# Pembersihan mempertahankan hanya transaksi berstatus `Paid`, membuang duplikat
# baris penuh, lalu memetakan setiap `product_name` ke satu dari lima merek.

# %%
import logging

logger = logging.getLogger(__name__)

_BRAND_KEYS: list[tuple[str, str]] = [
    ("xiaomi", "Xiaomi"),
    ("redmi", "Xiaomi"),
    ("oppo", "OPPO"),
    ("samsung", "Samsung"),
    ("realme", "Realme"),
    ("vivo", "Vivo"),
]


def brand_from_product_name(name: object) -> str:
    """Kembalikan merek dari product_name; 'Other' jika tak dikenali."""
    n = str(name).lower()
    for key, brand in _BRAND_KEYS:
        if key in n:
            return brand
    return "Other"


def clean_transactions(raw_path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """Bersihkan CSV POS mentah menjadi DataFrame transaksi Paid, terdedup, ber-merek.

    Statistik ringkas tersimpan di ``df.attrs['stats']`` untuk assertion DoD.
    """
    raw_path = Path(raw_path)
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    n_raw = len(raw)

    raw["created_at"] = pd.to_datetime(raw["created_at"])
    raw["paid_at"] = pd.to_datetime(raw["paid_at"], errors="coerce")

    n_void = int((raw["status"] == "Void").sum())
    paid = raw[raw["status"] == "Paid"].copy()

    n_before = len(paid)
    clean = paid.drop_duplicates().reset_index(drop=True)
    n_dup = n_before - len(clean)

    clean["brand"] = clean["product_name"].map(brand_from_product_name)
    n_other = int((clean["brand"] == "Other").sum())
    assert n_other == 0, (
        f"{n_other} baris ter-map ke 'Other'; product_name tak dikenal BRAND_MAP: "
        f"{list(clean.loc[clean['brand'] == 'Other', 'product_name'].unique()[:5])}"
    )

    clean["week_start"] = clean["created_at"].dt.to_period("W").dt.start_time

    stats = {
        "n_raw": n_raw,
        "n_void": n_void,
        "n_duplicates": n_dup,
        "n_clean": len(clean),
        "units": int(clean["qty"].sum()),
    }
    clean.attrs["stats"] = stats
    logger.info(
        "clean: raw=%d void=%d dup=%d -> clean=%d (Σqty=%d)",
        n_raw, n_void, n_dup, len(clean), stats["units"],
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(out_path, index=False)
        logger.info("Tersimpan: %s", out_path)

    return clean


# %% [markdown]
# Jalankan transformasi dengan input dan output yang selalu di-resolve dari akar
# repositori; output ini menjadi kontrak data untuk notebook Tahap 2.

# %%
clean = clean_transactions(
    rp("data", "raw", "pos_transactions_raw.csv"),
    rp("data", "interim", "transactions_clean.parquet"),
)
print("Statistik pembersihan:", clean.attrs["stats"])
clean.head()

# %%
st = clean.attrs["stats"]
expected = {"n_raw": 6320, "n_void": 76, "n_duplicates": 19, "n_clean": 6225, "units": 6907}
assert st == expected, f"❌ Tahap 1: {st} != {expected}"
assert clean["brand"].nunique() == 5 and clean["store"].nunique() == 4
assert (clean["brand"] != "Other").all()
print("✅ Tahap 1:", st, "| 5 merek × 4 gerai, tak ada 'Other'")
