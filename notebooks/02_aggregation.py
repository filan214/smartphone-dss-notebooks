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
# # Tahap 2 — Agregasi mingguan gerai × merek
#
# Notebook ini membentuk kontrak data Tahap 2 dari transaksi yang telah bersih.
# Keputusan D1 mempertahankan granularitas gerai × merek: mean fraksi minggu
# tanpa penjualan harus sekitar 0.258 dan tidak boleh melebihi 0.30. Keputusan
# D2 menerapkan resampling mingguan agar tiap deret memiliki kalender yang sama.

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
# Transaksi bersih Tahap 1 dimuat dari lokasi interim yang di-resolve relatif
# terhadap akar repositori.

# %%
clean = pd.read_parquet(rp("data", "interim", "transactions_clean.parquet"))
clean.head()

# %% [markdown]
# Fungsi berikut membangun grid penuh semua gerai, merek, dan Senin dalam rentang
# data. Minggu tanpa penjualan diisi nol, bukan dianggap nilai hilang.

# %%
import logging

logger = logging.getLogger(__name__)

N_WEEKS_EXPECTED = 184  # grid penuh (§1.2, diverifikasi pada data riil)
MAX_ZERO_WEEK_FRAC = 0.30  # ambang D1


def zero_week_fraction(weekly: pd.DataFrame) -> pd.Series:
    """Fraksi minggu dengan units==0 per deret (store, brand), urut menurun."""
    return (
        weekly.assign(is_zero=weekly["units"].eq(0))
        .groupby(["store", "brand"])["is_zero"]
        .mean()
        .sort_values(ascending=False)
    )


# %%
def aggregate_weekly(
    clean: pd.DataFrame,
    out_path: str | Path | None = None,
    max_zero_week_frac: float = MAX_ZERO_WEEK_FRAC,
) -> pd.DataFrame:
    """Agregasi transaksi bersih -> deret mingguan gerai x merek pada grid penuh."""
    g = (
        clean.groupby(["store", "brand", "week_start"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "units"})
    )

    stores = sorted(clean["store"].unique())
    brands = sorted(clean["brand"].unique())
    weeks = pd.date_range(
        clean["week_start"].min(), clean["week_start"].max(), freq="W-MON"
    )
    full_index = pd.MultiIndex.from_product(
        [stores, brands, weeks], names=["store", "brand", "week_start"]
    )
    weekly = (
        g.set_index(["store", "brand", "week_start"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["store", "brand", "week_start"])
        .reset_index(drop=True)
    )
    weekly["units"] = weekly["units"].astype(int)

    n_series = weekly.groupby(["store", "brand"]).ngroups
    zero_frac = zero_week_fraction(weekly)
    mean_zero_frac = float(zero_frac.mean())

    stats = {
        "n_rows": len(weekly),
        "n_series": int(n_series),
        "n_weeks": len(weeks),
        "units": int(weekly["units"].sum()),
        "mean_zero_week_frac": mean_zero_frac,
        "zero_week_frac_by_series": {
            f"{s}|{b}": round(float(v), 4) for (s, b), v in zero_frac.items()
        },
    }
    weekly.attrs["stats"] = stats

    logger.info(
        "aggregate: %d baris (%d deret x %d minggu), Σunits=%d, mean_zero_frac=%.4f",
        stats["n_rows"], stats["n_series"], stats["n_weeks"], stats["units"], mean_zero_frac,
    )
    logger.info("5 deret dengan fraksi minggu-nol tertinggi:")
    for (s, b), v in zero_frac.head(5).items():
        logger.info("    %s x %s: %.3f", s, b, v)

    assert weekly["units"].notna().all(), "Ada NaN pada units setelah reindex/fill."
    assert (weekly["units"] >= 0).all(), "units negatif — tak seharusnya."
    assert len(weeks) == N_WEEKS_EXPECTED, (
        f"Grid minggu = {len(weeks)}, diharapkan {N_WEEKS_EXPECTED} (§1.2)."
    )
    assert mean_zero_frac <= max_zero_week_frac, (
        f"mean fraksi minggu-nol = {mean_zero_frac:.4f} > {max_zero_week_frac} "
        "-> keputusan D1 (gerai x merek) tidak valid pada data ini; STOP & lapor."
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(out_path, index=False)
        logger.info("Tersimpan: %s", out_path)

    return weekly


# %% [markdown]
# Agregasi disimpan tanpa kondisi sebagai artefak deterministik yang dikonsumsi
# notebook tahap berikutnya.

# %%
weekly = aggregate_weekly(
    clean,
    rp("data", "interim", "weekly_store_brand.parquet"),
)
weekly.attrs["stats"]

# %% [markdown]
# Grafik berikut memperlihatkan fraksi minggu tanpa penjualan untuk setiap deret.
# Warna batang mengikuti identitas merek agar konsisten di seluruh notebook.

# %%
zero_frac = zero_week_fraction(weekly)
labels = [f"{store} × {brand}" for store, brand in zero_frac.index]
colors = [BRAND_COLORS[brand] for _, brand in zero_frac.index]
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.bar(labels, zero_frac.values, color=colors)
ax.axhline(MAX_ZERO_WEEK_FRAC, color="black", linestyle="--", linewidth=1, label="Batas D1 (0,30)")
ax.set_title("Fraksi minggu tanpa penjualan per gerai × merek")
ax.set_xlabel("Kombinasi gerai × merek")
ax.set_ylabel("Fraksi minggu tanpa penjualan")
ax.set_ylim(0, 0.58)
ax.tick_params(axis="x", rotation=60)
ax.legend()
plt.show()

# %%
st = weekly.attrs["stats"]
assert st["n_rows"] == 3680, st
assert st["n_series"] == 20 and st["n_weeks"] == 184
assert st["units"] == 6907
assert st["mean_zero_week_frac"] <= 0.30, "D1 dilanggar"
assert abs(st["mean_zero_week_frac"] - 0.258) < 0.01, st["mean_zero_week_frac"]
assert weekly["units"].notna().all() and (weekly["units"] >= 0).all()
print(f"✅ Tahap 2: 20×184=3680, Σ=6907, mean_zero_frac={st['mean_zero_week_frac']:.3f} ≤0.30 (D1)")
