__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PATH_AMSCLAE = ROOT / "data" / "external" / "batimetria_amsclae.tif"
C1_DIR = ROOT / "outputs" / "hipsometria_total_16C1"
COAST_DIR = ROOT / "outputs" / "costa_future_compatible"
PATH_CURVE_C1 = C1_DIR / "16C1_curva_hipsometrica_total.csv"
PATH_REFERENCES_C1 = C1_DIR / "16C1_referencias_verticales.csv"
PATH_DIAGNOSTIC_C1 = C1_DIR / "16C1_diagnostico_fisico.csv"
PATH_CURRENT_LAKE = COAST_DIR / "mascara_lago_actual_2024_2026.tif"
PATH_COAST_METADATA = COAST_DIR / "parametros_costa_future_compatible.json"
PATH_LEVEL_CANONICAL = ROOT / "data" / "processed" / "nivel_canonico.csv"
PATH_FUTURE = ROOT / "outputs" / "nivel_future_compatible" / "nivel_diario_cmip6_future_compatible.pkl"

OUT_DIR = ROOT / "outputs" / "hipsometria_future_compatible"
OUT_CURVE = OUT_DIR / "curva_hipsometrica_relativa_frozen.csv"
OUT_ANCHOR = OUT_DIR / "ancla_area_volumen_referencias.csv"
OUT_FUTURE = OUT_DIR / "area_volumen_horizontes_cmip6.csv"
OUT_SUMMARY = OUT_DIR / "resumen_horizontes_area_volumen.csv"
OUT_MANIFEST = OUT_DIR / "manifiesto_hipsometria_future_compatible.csv"
OUT_METADATA = OUT_DIR / "parametros_hipsometria_future_compatible.json"
OUT_FIG_AREA = OUT_DIR / "area_futura_p50.png"
OUT_FIG_VOLUME = OUT_DIR / "volumen_futuro_p50.png"

H2014_MED = 1557.560
H2014_MIN = 1557.310
H2014_MAX = 1558.090
H_ANCHOR = 1552.770
ANCHOR_DATE = "2026-08-29"
CURRENT_AREA_EXPECTED_KM2 = 122.0232
PUBLISHED_VOLUME_2014_KM3 = 25.46
RASTER_AREA_EXPECTED_KM2 = 126.5563
RASTER_VOLUME_EXPECTED_KM3 = 25.31516
RASTER_MAX_DEPTH_EXPECTED_M = 325.0

SCENARIOS = ["ssp245", "ssp585"]
HORIZONS = [2030, 2050, 2090]

EXPECTED_P50 = {
    ("ssp245", 2030): 1552.348610,
    ("ssp245", 2050): 1550.336502,
    ("ssp245", 2090): 1542.639156,
    ("ssp585", 2030): 1552.293235,
    ("ssp585", 2050): 1548.144740,
    ("ssp585", 2090): 1530.124355,
}

REFERENCES = {
    "primary_temporal_2014_median": H2014_MED,
    "sensitivity_temporal_2014_min": H2014_MIN,
    "sensitivity_temporal_2014_max": H2014_MAX,
}


def banner(text):
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stage_metrics(sorted_depths, cumulative_depths, pixel_area_m2, relative_level):
    k = int(np.searchsorted(sorted_depths, relative_level, side="right"))
    if k <= 0:
        return 0, 0.0, 0.0, np.nan
    sum_d = cumulative_depths[k - 1]
    area_km2 = k * pixel_area_m2 / 1e6
    volume_km3 = (k * relative_level - sum_d) * pixel_area_m2 / 1e9
    mean_depth_m = volume_km3 * 1000.0 / area_km2 if area_km2 > 0 else np.nan
    return k, float(area_km2), float(volume_km3), float(mean_depth_m)


def normalize_scenario(value):
    s = str(value).lower().replace("_", "").replace("-", "")
    if "245" in s:
        return "ssp245"
    if "585" in s:
        return "ssp585"
    return s


def find_date_column(df):
    for col in ["obs_date", "date", "fecha"]:
        if col in df.columns:
            return col
    raise RuntimeError("No se encontro columna de fecha.")


for p in [
    PATH_AMSCLAE,
    PATH_CURVE_C1,
    PATH_REFERENCES_C1,
    PATH_DIAGNOSTIC_C1,
    PATH_CURRENT_LAKE,
    PATH_COAST_METADATA,
    PATH_LEVEL_CANONICAL,
    PATH_FUTURE,
]:
    if not p.exists():
        raise FileNotFoundError(p)

banner("PASO 16C2 - CONGELAR HIPSOMETRIA FUTURE-COMPATIBLE")

with open(PATH_COAST_METADATA, "r", encoding="utf-8") as f:
    coast_metadata = json.load(f)

if coast_metadata.get("status") != "FROZEN_COAST_FUTURE_COMPATIBLE":
    raise RuntimeError("La rama costera frozen no tiene el status esperado.")

with rasterio.open(PATH_CURRENT_LAKE) as ds:
    current_lake = ds.read(1) > 0
    current_transform = ds.transform

current_pixel_area_m2 = abs(
    current_transform.a * current_transform.e
    - current_transform.b * current_transform.d
)

current_area_km2 = current_lake.sum() * current_pixel_area_m2 / 1e6

if abs(current_area_km2 - CURRENT_AREA_EXPECTED_KM2) > 5e-4:
    raise RuntimeError("El area Sentinel frozen no reproduce 122.0232 km2.")

references_c1 = pd.read_csv(PATH_REFERENCES_C1)

required_refs = {
    "temporal_2014_median",
    "temporal_2014_min",
    "temporal_2014_max",
    "area_anchored_2026",
}

if not required_refs.issubset(set(references_c1["reference"])):
    raise RuntimeError("16C1 no contiene todas las referencias verticales esperadas.")

h0_area_anchor = float(
    references_c1.loc[
        references_c1["reference"] == "area_anchored_2026",
        "H0_m",
    ].iloc[0]
)

area_anchor_shift_m = h0_area_anchor - H2014_MED

diagnostic_c1 = pd.read_csv(PATH_DIAGNOSTIC_C1)
diag = dict(zip(diagnostic_c1["metric"], diagnostic_c1["value"]))

if abs(float(diag["reference_area_km2"]) - RASTER_AREA_EXPECTED_KM2) > 5e-4:
    raise RuntimeError("16C1 no reproduce el area AMSCLAE esperada.")

if abs(float(diag["reference_volume_km3"]) - RASTER_VOLUME_EXPECTED_KM3) > 5e-5:
    raise RuntimeError("16C1 no reproduce el volumen AMSCLAE esperado.")

if abs(float(diag["max_depth_m"]) - RASTER_MAX_DEPTH_EXPECTED_M) > 1e-6:
    raise RuntimeError("16C1 no reproduce la profundidad maxima esperada.")

banner("QC 1 - MORFOMETRIA ORIGINAL")

with rasterio.open(PATH_AMSCLAE) as ds:
    bathy = ds.read(1).astype(np.float64)
    nodata = ds.nodata
    transform = ds.transform
    crs = ds.crs
    shape = ds.shape

valid = np.isfinite(bathy)

if nodata is not None:
    valid &= bathy != nodata

depth_values = bathy[valid]
depth_values = depth_values[depth_values <= 0.0]

pixel_area_m2 = abs(
    transform.a * transform.e
    - transform.b * transform.d
)

sorted_depths = np.sort(depth_values)
cumulative_depths = np.cumsum(sorted_depths, dtype=np.float64)
positive_depths = -depth_values

reference_area_km2 = len(sorted_depths) * pixel_area_m2 / 1e6
reference_volume_km3 = np.sum(positive_depths, dtype=np.float64) * pixel_area_m2 / 1e9
max_depth_m = float(np.max(positive_depths))

if abs(reference_area_km2 - RASTER_AREA_EXPECTED_KM2) > 5e-4:
    raise RuntimeError("Area AMSCLAE inesperada.")

if abs(reference_volume_km3 - RASTER_VOLUME_EXPECTED_KM3) > 5e-5:
    raise RuntimeError("Volumen AMSCLAE inesperado.")

if abs(max_depth_m - RASTER_MAX_DEPTH_EXPECTED_M) > 1e-6:
    raise RuntimeError("Profundidad maxima AMSCLAE inesperada.")

print(f"[AREA REFERENCE] = {reference_area_km2:.4f} km2")
print(f"[VOLUME REFERENCE] = {reference_volume_km3:.5f} km3")
print(f"[MAX DEPTH] = {max_depth_m:.2f} m")

banner("QC 2 - REFERENCIA PRODUCTIVA")

print(f"[PRIMARY H0] = {H2014_MED:.3f} m")
print(f"[SENSITIVITY H0 MIN/MAX] = {H2014_MIN:.3f} / {H2014_MAX:.3f} m")
print(f"[AREA-ANCHORED H0 DIAGNOSTIC] = {h0_area_anchor:.3f} m")
print(f"[AREA-ANCHORED SHIFT VS PRIMARY] = {area_anchor_shift_m:+.3f} m")
print("[DECISION] area_anchored_2026 queda solo como diagnostico; no se usa como datum productivo.")

level = pd.read_csv(PATH_LEVEL_CANONICAL)
date_col = find_date_column(level)
level["_date"] = pd.to_datetime(level[date_col], errors="coerce")
level["_level"] = pd.to_numeric(level["nivel_final_m"], errors="coerce")
level = level[level["_date"].notna() & level["_level"].notna()].copy()

survey = level[
    (level["_date"] >= pd.Timestamp("2014-05-01"))
    & (level["_date"] <= pd.Timestamp("2014-06-30"))
]

if len(survey) != 5:
    raise RuntimeError("No se reproducen las 5 observaciones mayo-junio 2014.")

if abs(float(survey["_level"].median()) - H2014_MED) > 1e-9:
    raise RuntimeError("No se reproduce H2014_MED.")

if abs(float(survey["_level"].min()) - H2014_MIN) > 1e-9:
    raise RuntimeError("No se reproduce H2014_MIN.")

if abs(float(survey["_level"].max()) - H2014_MAX) > 1e-9:
    raise RuntimeError("No se reproduce H2014_MAX.")

anchor_obs = level[level["_date"] == pd.Timestamp(ANCHOR_DATE)]

if len(anchor_obs) != 1:
    raise RuntimeError("No se encontro exactamente el ancla 2026-08-29.")

if abs(float(anchor_obs.iloc[0]["_level"]) - H_ANCHOR) > 1e-9:
    raise RuntimeError("No se reproduce H_ANCHOR.")

anchor_rows = []

for reference, H0 in REFERENCES.items():
    relative_anchor = H_ANCHOR - H0
    n, area_raw, volume_raw, mean_depth = stage_metrics(
        sorted_depths,
        cumulative_depths,
        pixel_area_m2,
        relative_anchor,
    )
    volume_published_anchored = (
        PUBLISHED_VOLUME_2014_KM3
        + volume_raw
        - reference_volume_km3
    )
    anchor_rows.append(
        {
            "reference": reference,
            "H0_m": H0,
            "anchor_level_m": H_ANCHOR,
            "relative_anchor_m": relative_anchor,
            "wet_pixels": n,
            "area_raw_km2": area_raw,
            "area_observed_anchor_km2": current_area_km2,
            "area_raw_minus_observed_km2": area_raw - current_area_km2,
            "volume_raw_km3": volume_raw,
            "volume_published_anchored_km3": volume_published_anchored,
            "mean_water_depth_m": mean_depth,
        }
    )

anchor_df = pd.DataFrame(anchor_rows)

banner("QC 3 - ANCLA POR REFERENCIA")

print(
    anchor_df.to_string(
        index=False,
        formatters={
            "H0_m": "{:.3f}".format,
            "relative_anchor_m": "{:+.3f}".format,
            "area_raw_km2": "{:.4f}".format,
            "area_observed_anchor_km2": "{:.4f}".format,
            "area_raw_minus_observed_km2": "{:+.4f}".format,
            "volume_raw_km3": "{:.5f}".format,
            "volume_published_anchored_km3": "{:.5f}".format,
            "mean_water_depth_m": "{:.2f}".format,
        },
    )
)

future = pd.read_pickle(PATH_FUTURE)

if isinstance(future, pd.Series):
    future = future.to_frame()

required_columns = {"date", "model", "scenario", "level_m"}

if not required_columns.issubset(set(future.columns)):
    raise RuntimeError("El futuro congelado no contiene las columnas esperadas.")

future = future.copy()
future["_date"] = pd.to_datetime(future["date"], errors="coerce")
future["_year"] = future["_date"].dt.year
future["_scenario"] = future["scenario"].map(normalize_scenario)
future["_level"] = pd.to_numeric(future["level_m"], errors="coerce")
future = future[
    future["_date"].notna()
    & future["_level"].notna()
    & future["_scenario"].isin(SCENARIOS)
].copy()

quantile_levels = []

for scenario in SCENARIOS:
    for year in HORIZONS:
        sub = future[
            (future["_scenario"] == scenario)
            & (future["_year"] == year)
        ]

        annual_by_model = sub.groupby("model")["_level"].mean().dropna()

        if len(annual_by_model) != 15:
            raise RuntimeError(f"{scenario} {year}: se esperaban 15 GCM.")

        q = {
            "p05": float(annual_by_model.quantile(0.05)),
            "p50": float(annual_by_model.quantile(0.50)),
            "p95": float(annual_by_model.quantile(0.95)),
        }

        if abs(q["p50"] - EXPECTED_P50[(scenario, year)]) > 5e-4:
            raise RuntimeError(f"{scenario} {year}: p50 no reproduce 15C.")

        for quantile, H in q.items():
            quantile_levels.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "quantile": quantile,
                    "level_m": H,
                }
            )

quantile_levels_df = pd.DataFrame(quantile_levels)

anchor_lookup = anchor_df.set_index("reference").to_dict(orient="index")
future_rows = []

for _, qrow in quantile_levels_df.iterrows():
    scenario = str(qrow["scenario"])
    year = int(qrow["year"])
    quantile = str(qrow["quantile"])
    H = float(qrow["level_m"])

    for reference, H0 in REFERENCES.items():
        relative_level = H - H0
        n, area_raw, volume_raw, mean_depth = stage_metrics(
            sorted_depths,
            cumulative_depths,
            pixel_area_m2,
            relative_level,
        )

        anchor_area_raw = float(anchor_lookup[reference]["area_raw_km2"])
        anchor_volume_raw = float(anchor_lookup[reference]["volume_raw_km3"])

        delta_area_vs_anchor = area_raw - anchor_area_raw
        delta_volume_vs_anchor = volume_raw - anchor_volume_raw

        area_anchor_adjusted = current_area_km2 + delta_area_vs_anchor
        volume_published_anchored = (
            PUBLISHED_VOLUME_2014_KM3
            + volume_raw
            - reference_volume_km3
        )

        future_rows.append(
            {
                "scenario": scenario,
                "year": year,
                "quantile": quantile,
                "reference": reference,
                "H0_m": H0,
                "level_m": H,
                "relative_level_m": relative_level,
                "wet_pixels_raw": n,
                "area_raw_km2": area_raw,
                "anchor_area_raw_km2": anchor_area_raw,
                "delta_area_vs_anchor_km2": delta_area_vs_anchor,
                "area_anchor_adjusted_km2": area_anchor_adjusted,
                "area_loss_vs_current_km2": current_area_km2 - area_anchor_adjusted,
                "volume_raw_km3": volume_raw,
                "anchor_volume_raw_km3": anchor_volume_raw,
                "delta_volume_vs_anchor_km3": delta_volume_vs_anchor,
                "storage_loss_vs_anchor_km3": anchor_volume_raw - volume_raw,
                "volume_published_anchored_km3": volume_published_anchored,
                "mean_water_depth_m": mean_depth,
            }
        )

future_df = pd.DataFrame(future_rows)

banner("QC 4 - HORIZONTES PRODUCTIVOS")

primary_p50 = future_df[
    (future_df["quantile"] == "p50")
    & (future_df["reference"] == "primary_temporal_2014_median")
].copy()

print(
    primary_p50[
        [
            "scenario",
            "year",
            "level_m",
            "area_raw_km2",
            "area_anchor_adjusted_km2",
            "area_loss_vs_current_km2",
            "volume_raw_km3",
            "volume_published_anchored_km3",
            "storage_loss_vs_anchor_km3",
        ]
    ].to_string(
        index=False,
        formatters={
            "level_m": "{:.3f}".format,
            "area_raw_km2": "{:.3f}".format,
            "area_anchor_adjusted_km2": "{:.3f}".format,
            "area_loss_vs_current_km2": "{:.3f}".format,
            "volume_raw_km3": "{:.3f}".format,
            "volume_published_anchored_km3": "{:.3f}".format,
            "storage_loss_vs_anchor_km3": "{:.3f}".format,
        },
    )
)

summary_rows = []

for scenario in SCENARIOS:
    for year in HORIZONS:
        primary = future_df[
            (future_df["scenario"] == scenario)
            & (future_df["year"] == year)
            & (future_df["reference"] == "primary_temporal_2014_median")
        ].set_index("quantile")

        sens = future_df[
            (future_df["scenario"] == scenario)
            & (future_df["year"] == year)
            & (future_df["quantile"] == "p50")
        ]

        summary_rows.append(
            {
                "scenario": scenario,
                "year": year,
                "level_p05_m": float(primary.loc["p05", "level_m"]),
                "level_p50_m": float(primary.loc["p50", "level_m"]),
                "level_p95_m": float(primary.loc["p95", "level_m"]),
                "area_anchor_adjusted_p05_km2": float(primary.loc["p05", "area_anchor_adjusted_km2"]),
                "area_anchor_adjusted_p50_km2": float(primary.loc["p50", "area_anchor_adjusted_km2"]),
                "area_anchor_adjusted_p95_km2": float(primary.loc["p95", "area_anchor_adjusted_km2"]),
                "area_loss_p50_vs_current_km2": float(primary.loc["p50", "area_loss_vs_current_km2"]),
                "area_p50_vertical_sensitivity_min_km2": float(sens["area_anchor_adjusted_km2"].min()),
                "area_p50_vertical_sensitivity_max_km2": float(sens["area_anchor_adjusted_km2"].max()),
                "volume_raw_p05_km3": float(primary.loc["p05", "volume_raw_km3"]),
                "volume_raw_p50_km3": float(primary.loc["p50", "volume_raw_km3"]),
                "volume_raw_p95_km3": float(primary.loc["p95", "volume_raw_km3"]),
                "volume_published_anchored_p50_km3": float(primary.loc["p50", "volume_published_anchored_km3"]),
                "storage_loss_p50_vs_anchor_km3": float(primary.loc["p50", "storage_loss_vs_anchor_km3"]),
                "volume_p50_vertical_sensitivity_min_km3": float(sens["volume_raw_km3"].min()),
                "volume_p50_vertical_sensitivity_max_km3": float(sens["volume_raw_km3"].max()),
            }
        )

summary_df = pd.DataFrame(summary_rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy2(PATH_CURVE_C1, OUT_CURVE)

if sha256_file(PATH_CURVE_C1) != sha256_file(OUT_CURVE):
    raise RuntimeError("La curva frozen no es identica a 16C1.")

anchor_df.to_csv(OUT_ANCHOR, index=False, encoding="utf-8-sig")
future_df.to_csv(OUT_FUTURE, index=False, encoding="utf-8-sig")
summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

banner("QC 5 - RESUMEN FINAL")

print(
    summary_df[
        [
            "scenario",
            "year",
            "level_p50_m",
            "area_anchor_adjusted_p50_km2",
            "area_loss_p50_vs_current_km2",
            "volume_raw_p50_km3",
            "storage_loss_p50_vs_anchor_km3",
        ]
    ].to_string(
        index=False,
        formatters={
            "level_p50_m": "{:.3f}".format,
            "area_anchor_adjusted_p50_km2": "{:.3f}".format,
            "area_loss_p50_vs_current_km2": "{:.3f}".format,
            "volume_raw_p50_km3": "{:.3f}".format,
            "storage_loss_p50_vs_anchor_km3": "{:.3f}".format,
        },
    )
)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

for scenario in SCENARIOS:
    sub = summary_df[summary_df["scenario"] == scenario].sort_values("year")
    ax.plot(
        sub["year"],
        sub["area_anchor_adjusted_p50_km2"],
        marker="o",
        label=scenario.upper(),
    )

ax.axhline(current_area_km2, linestyle="--", label="Area Sentinel 2026")
ax.set_xlabel("Horizonte")
ax.set_ylabel("Area agregada anclada a 2026 (km2)")
ax.set_title("16C2 - Area futura p50")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_AREA, dpi=180, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

for scenario in SCENARIOS:
    sub = summary_df[summary_df["scenario"] == scenario].sort_values("year")
    ax.plot(
        sub["year"],
        sub["volume_raw_p50_km3"],
        marker="o",
        label=scenario.upper(),
    )

ax.set_xlabel("Horizonte")
ax.set_ylabel("Volumen hipsometrico p50 (km3)")
ax.set_title("16C2 - Volumen futuro p50")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_VOLUME, dpi=180, bbox_inches="tight")
plt.close(fig)

manifest_sources = [
    ("bathymetry", PATH_AMSCLAE),
    ("coast_metadata", PATH_COAST_METADATA),
    ("current_lake_mask", PATH_CURRENT_LAKE),
    ("canonical_level", PATH_LEVEL_CANONICAL),
    ("future_level_pickle", PATH_FUTURE),
    ("16C1_curve", PATH_CURVE_C1),
    ("16C1_references", PATH_REFERENCES_C1),
    ("16C1_diagnostic", PATH_DIAGNOSTIC_C1),
]

manifest_outputs = [
    ("frozen_curve", OUT_CURVE),
    ("anchor_table", OUT_ANCHOR),
    ("future_area_volume", OUT_FUTURE),
    ("future_summary", OUT_SUMMARY),
    ("figure_area", OUT_FIG_AREA),
    ("figure_volume", OUT_FIG_VOLUME),
]

manifest_rows = []

for role, path in manifest_sources:
    manifest_rows.append(
        {
            "kind": "source",
            "role": role,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    )

for role, path in manifest_outputs:
    manifest_rows.append(
        {
            "kind": "output",
            "role": role,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")

metadata = {
    "step": "16C2",
    "status": "FROZEN_HYPSOMETRY_FUTURE_COMPATIBLE",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "production_reference": {
        "name": "primary_temporal_2014_median",
        "H0_m": H2014_MED,
        "basis": "median canonical lake level during May-June 2014",
    },
    "vertical_sensitivity_references": {
        "min_m": H2014_MIN,
        "max_m": H2014_MAX,
    },
    "rejected_as_production_datum": {
        "name": "area_anchored_2026",
        "H0_m": h0_area_anchor,
        "shift_vs_primary_m": area_anchor_shift_m,
        "reason": "derived from matching modern Sentinel area to the aggregate raster footprint; retained only as a diagnostic and not used to shift the spatial bathymetry",
    },
    "current_anchor": {
        "date": ANCHOR_DATE,
        "level_m": H_ANCHOR,
        "observed_area_km2": current_area_km2,
    },
    "aggregate_area_method": "observed Sentinel anchor area plus hypsometric area change relative to the 2026 anchor under the same temporal reference",
    "aggregate_volume_method": "raw AMSCLAE hypsometric volume under the temporal reference; changes are reported relative to the 2026 anchor",
    "published_volume_secondary_series": "published 2014 volume 25.46 km3 plus modeled storage change from relative level 0",
    "reference_morphometry": {
        "raster_area_km2": reference_area_km2,
        "raster_volume_km3": reference_volume_km3,
        "max_depth_m": max_depth_m,
    },
    "future": {
        "scenarios": SCENARIOS,
        "horizons": HORIZONS,
        "quantiles": ["p05", "p50", "p95"],
        "models_per_scenario": 15,
    },
    "scope": {
        "hypsometry": "aggregate area-volume only",
        "spatial_shoreline": "resultados/costa_future_compatible remains authoritative",
        "future_levels": "conditional CMIP6 scenario trajectories, not deterministic forecasts",
    },
    "limitations": [
        "The aggregate hypsometric curve does not repair or redefine unsupported spatial shoreline pixels.",
        "The absolute raw area from AMSCLAE is not used as the modern area anchor.",
        "Vertical sensitivity from the May-June 2014 level bracket is retained separately from CMIP6 ensemble spread.",
        "The area-anchored 2026 H0 is not treated as a validated vertical datum correction.",
        "2090 is retained as a stress horizon and carries greater scenario and model uncertainty than 2030/2050.",
    ],
    "outputs_sha256": {
        OUT_CURVE.name: sha256_file(OUT_CURVE),
        OUT_ANCHOR.name: sha256_file(OUT_ANCHOR),
        OUT_FUTURE.name: sha256_file(OUT_FUTURE),
        OUT_SUMMARY.name: sha256_file(OUT_SUMMARY),
        OUT_MANIFEST.name: sha256_file(OUT_MANIFEST),
        OUT_FIG_AREA.name: sha256_file(OUT_FIG_AREA),
        OUT_FIG_VOLUME.name: sha256_file(OUT_FIG_VOLUME),
    },
}

with open(OUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

banner("DECISION 16C2")

print(f"[PRIMARY H0] = {H2014_MED:.3f} m")
print(f"[VERTICAL SENSITIVITY] = {H2014_MIN:.3f} a {H2014_MAX:.3f} m")
print(f"[AREA-ANCHORED H0] = {h0_area_anchor:.3f} m | shift {area_anchor_shift_m:+.3f} m | DIAGNOSTIC ONLY")
print(f"[CURRENT OBSERVED AREA ANCHOR] = {current_area_km2:.4f} km2")
print("[AREA PRODUCTION] Sentinel 2026 + delta hipsometrica desde el ancla")
print("[VOLUME PRODUCTION] volumen hipsometrico temporal + delta vs ancla")
print("[STATUS] FROZEN_HYPSOMETRY_FUTURE_COMPATIBLE")
print("[NEXT] integrar nivel, costa espacial frozen e hipsometria frozen en productos finales 2030/2050.")

banner("FIN PASO 16C2")
print("[OUTPUT DIRECTORY]")
print(OUT_DIR)
