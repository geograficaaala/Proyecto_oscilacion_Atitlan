__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PATH_AMSCLAE = ROOT / "data" / "external" / "batimetria_amsclae.tif"
COAST_FROZEN = ROOT / "outputs" / "costa_future_compatible"
PATH_CURRENT_LAKE = COAST_FROZEN / "mascara_lago_actual_2024_2026.tif"
PATH_COAST_METADATA = COAST_FROZEN / "parametros_costa_future_compatible.json"
PATH_LEVEL_CANONICAL = ROOT / "data" / "processed" / "nivel_canonico.csv"
PATH_FUTURE = ROOT / "outputs" / "nivel_future_compatible" / "nivel_diario_cmip6_future_compatible.pkl"

OUT_DIR = ROOT / "outputs" / "hipsometria_total_16C1"
OUT_CURVE = OUT_DIR / "16C1_curva_hipsometrica_total.csv"
OUT_CONTOURS = OUT_DIR / "16C1_niveles_relativos_clave.csv"
OUT_BANDS = OUT_DIR / "16C1_bandas_profundidad.csv"
OUT_PERCENTILES = OUT_DIR / "16C1_percentiles_profundidad.csv"
OUT_REFERENCES = OUT_DIR / "16C1_referencias_verticales.csv"
OUT_FUTURE = OUT_DIR / "16C1_hipsometria_futura_2030_2050_2090.csv"
OUT_PUBLISHED = OUT_DIR / "16C1_comparacion_morfometria_publicada.csv"
OUT_DIAGNOSTIC = OUT_DIR / "16C1_diagnostico_fisico.csv"
OUT_METADATA = OUT_DIR / "16C1_metadata.json"
OUT_FIG_AREA = OUT_DIR / "16C1_area_vs_nivel_relativo.png"
OUT_FIG_VOLUME = OUT_DIR / "16C1_volumen_vs_nivel_relativo.png"
OUT_FIG_AREA_VOLUME = OUT_DIR / "16C1_area_vs_volumen.png"
OUT_FIG_DEPTH = OUT_DIR / "16C1_distribucion_profundidad.png"
OUT_FIG_SENSITIVITY = OUT_DIR / "16C1_sensibilidad_referencia_vertical.png"

H2014_MED = 1557.560
H2014_MIN = 1557.310
H2014_MAX = 1558.090
H_ANCHOR = 1552.770
ANCHOR_DATE = "2026-08-29"

CURRENT_AREA_EXPECTED_KM2 = 122.0232
RASTER_AREA_EXPECTED_KM2 = 126.5563
RASTER_VOLUME_EXPECTED_KM3 = 25.31516
RASTER_MAX_DEPTH_EXPECTED_M = 325.0

PUBLISHED_AREA_KM2 = 125.77
PUBLISHED_VOLUME_KM3 = 25.46
PUBLISHED_MAX_DEPTH_M = 327.56
PUBLISHED_MEAN_DEPTH_M = PUBLISHED_VOLUME_KM3 / PUBLISHED_AREA_KM2 * 1000.0

CURVE_STEP_M = 0.10
HORIZONS = [2030, 2050, 2090]
SCENARIOS = ["ssp245", "ssp585"]

EXPECTED_P50 = {
    ("ssp245", 2030): 1552.348610,
    ("ssp245", 2050): 1550.336502,
    ("ssp245", 2090): 1542.639156,
    ("ssp585", 2030): 1552.293235,
    ("ssp585", 2050): 1548.144740,
    ("ssp585", 2090): 1530.124355,
}

DEPTH_BANDS = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, 50.0),
    (50.0, 75.0),
    (75.0, 100.0),
    (100.0, 150.0),
    (150.0, 200.0),
    (200.0, 250.0),
    (250.0, 300.0),
    (300.0, 400.0),
]

KEY_RELATIVE_LEVELS = [
    0.0,
    -0.5,
    -1.0,
    -2.0,
    -5.0,
    -10.0,
    -20.0,
    -30.0,
    -40.0,
    -50.0,
    -75.0,
    -100.0,
    -150.0,
    -200.0,
    -250.0,
    -300.0,
    -325.0,
]

DEPTH_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]


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


def find_date_column(df):
    for col in ["obs_date", "date", "fecha"]:
        if col in df.columns:
            return col
    raise RuntimeError("No se encontro columna de fecha en nivel_canonico.csv.")


def stage_metrics(sorted_depths, cumulative_depths, pixel_area_m2, relative_level):
    k = int(np.searchsorted(sorted_depths, relative_level, side="right"))
    if k <= 0:
        return 0, 0.0, 0.0, np.nan
    sum_d = cumulative_depths[k - 1]
    area_km2 = k * pixel_area_m2 / 1e6
    volume_km3 = (k * relative_level - sum_d) * pixel_area_m2 / 1e9
    mean_depth_m = volume_km3 * 1000.0 / area_km2 if area_km2 > 0 else np.nan
    return k, float(area_km2), float(volume_km3), float(mean_depth_m)


def solve_relative_level_for_area(sorted_depths, cumulative_depths, pixel_area_m2, target_area_km2):
    target_pixels_float = target_area_km2 * 1e6 / pixel_area_m2
    target_pixels = int(np.clip(np.rint(target_pixels_float), 1, len(sorted_depths)))
    relative_level = float(sorted_depths[target_pixels - 1])
    n, area_km2, volume_km3, mean_depth_m = stage_metrics(
        sorted_depths,
        cumulative_depths,
        pixel_area_m2,
        relative_level,
    )
    return {
        "relative_level_m": relative_level,
        "target_pixels_float": float(target_pixels_float),
        "target_pixels_rounded": target_pixels,
        "wet_pixels": n,
        "area_km2": area_km2,
        "volume_km3": volume_km3,
        "mean_depth_m": mean_depth_m,
        "area_error_km2": area_km2 - target_area_km2,
    }


def normalize_scenario(value):
    s = str(value).lower().replace("_", "").replace("-", "")
    if "245" in s:
        return "ssp245"
    if "585" in s:
        return "ssp585"
    return s


for p in [
    PATH_AMSCLAE,
    PATH_CURRENT_LAKE,
    PATH_COAST_METADATA,
    PATH_LEVEL_CANONICAL,
    PATH_FUTURE,
]:
    if not p.exists():
        raise FileNotFoundError(p)

OUT_DIR.mkdir(parents=True, exist_ok=True)

banner("PASO 16C1 - HIPSOMETRIA TOTAL Y COMPLETA")

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
    raise RuntimeError("El area actual Sentinel no reproduce 122.0232 km2.")

level = pd.read_csv(PATH_LEVEL_CANONICAL)
date_col = find_date_column(level)
level["_date"] = pd.to_datetime(level[date_col], errors="coerce")
level["_level"] = pd.to_numeric(level["nivel_final_m"], errors="coerce")
level = level[level["_date"].notna() & level["_level"].notna()].copy()

survey = level[
    (level["_date"] >= pd.Timestamp("2014-05-01"))
    & (level["_date"] <= pd.Timestamp("2014-06-30"))
].copy()

if len(survey) != 5:
    raise RuntimeError(f"Se esperaban 5 observaciones mayo-junio 2014; hay {len(survey)}.")

survey_med = float(survey["_level"].median())
survey_min = float(survey["_level"].min())
survey_max = float(survey["_level"].max())

if abs(survey_med - H2014_MED) > 1e-9:
    raise RuntimeError("La mediana mayo-junio 2014 no reproduce 1557.560 m.")

if abs(survey_min - H2014_MIN) > 1e-9:
    raise RuntimeError("El minimo mayo-junio 2014 no reproduce 1557.310 m.")

if abs(survey_max - H2014_MAX) > 1e-9:
    raise RuntimeError("El maximo mayo-junio 2014 no reproduce 1558.090 m.")

anchor = level[level["_date"] == pd.Timestamp(ANCHOR_DATE)]

if len(anchor) != 1:
    raise RuntimeError("No se encontro exactamente el ancla 2026-08-29.")

anchor_level = float(anchor.iloc[0]["_level"])

if abs(anchor_level - H_ANCHOR) > 1e-9:
    raise RuntimeError("El ancla no reproduce 1552.770 m.")

banner("QC 1 - MORFOMETRIA AMSCLAE ORIGINAL")

with rasterio.open(PATH_AMSCLAE) as ds:
    bathy = ds.read(1).astype(np.float64)
    nodata = ds.nodata
    bathy_transform = ds.transform
    bathy_crs = ds.crs
    bathy_shape = ds.shape

valid = np.isfinite(bathy)

if nodata is not None:
    valid &= bathy != nodata

depth_values = bathy[valid].astype(np.float64)
depth_values = depth_values[depth_values <= 0.0]

pixel_area_m2 = abs(
    bathy_transform.a * bathy_transform.e
    - bathy_transform.b * bathy_transform.d
)

sorted_depths = np.sort(depth_values)
cumulative_depths = np.cumsum(sorted_depths, dtype=np.float64)
positive_depths = -depth_values

n_valid = len(sorted_depths)
reference_area_km2 = n_valid * pixel_area_m2 / 1e6
reference_volume_km3 = np.sum(positive_depths, dtype=np.float64) * pixel_area_m2 / 1e9
max_depth_m = float(np.max(positive_depths))
mean_depth_m = reference_volume_km3 * 1000.0 / reference_area_km2
median_depth_m = float(np.median(positive_depths))

if abs(reference_area_km2 - RASTER_AREA_EXPECTED_KM2) > 5e-4:
    raise RuntimeError("El area AMSCLAE no reproduce el diagnostico previo.")

if abs(reference_volume_km3 - RASTER_VOLUME_EXPECTED_KM3) > 5e-5:
    raise RuntimeError("El volumen AMSCLAE no reproduce el diagnostico previo.")

if abs(max_depth_m - RASTER_MAX_DEPTH_EXPECTED_M) > 1e-6:
    raise RuntimeError("La profundidad maxima AMSCLAE no reproduce el diagnostico previo.")

print(f"[VALID PIXELS] = {n_valid:,}")
print(f"[PIXEL AREA] = {pixel_area_m2:.4f} m2")
print(f"[REFERENCE AREA] = {reference_area_km2:.4f} km2")
print(f"[REFERENCE VOLUME] = {reference_volume_km3:.5f} km3")
print(f"[MAX DEPTH] = {max_depth_m:.2f} m")
print(f"[MEAN DEPTH] = {mean_depth_m:.2f} m")
print(f"[MEDIAN DEPTH] = {median_depth_m:.2f} m")

banner("QC 2 - REFERENCIAS VERTICALES")

area_anchor_solution = solve_relative_level_for_area(
    sorted_depths,
    cumulative_depths,
    pixel_area_m2,
    current_area_km2,
)

relative_anchor_area = float(area_anchor_solution["relative_level_m"])
H0_area_anchor = H_ANCHOR - relative_anchor_area

references = {
    "temporal_2014_median": H2014_MED,
    "temporal_2014_min": H2014_MIN,
    "temporal_2014_max": H2014_MAX,
    "area_anchored_2026": H0_area_anchor,
}

reference_rows = []

for ref_name, H0 in references.items():
    relative_anchor = H_ANCHOR - H0
    n, area_km2, volume_km3, stage_mean_depth_m = stage_metrics(
        sorted_depths,
        cumulative_depths,
        pixel_area_m2,
        relative_anchor,
    )
    reference_rows.append(
        {
            "reference": ref_name,
            "H0_m": H0,
            "anchor_level_m": H_ANCHOR,
            "anchor_relative_level_m": relative_anchor,
            "anchor_wet_pixels": n,
            "anchor_area_km2": area_km2,
            "anchor_area_error_vs_sentinel_km2": area_km2 - current_area_km2,
            "anchor_area_error_vs_sentinel_pct": 100.0 * (area_km2 - current_area_km2) / current_area_km2,
            "anchor_volume_km3": volume_km3,
            "anchor_mean_depth_m": stage_mean_depth_m,
        }
    )

references_df = pd.DataFrame(reference_rows)
references_df.to_csv(OUT_REFERENCES, index=False, encoding="utf-8-sig")

print(
    references_df[
        [
            "reference",
            "H0_m",
            "anchor_relative_level_m",
            "anchor_area_km2",
            "anchor_area_error_vs_sentinel_km2",
            "anchor_volume_km3",
        ]
    ].to_string(
        index=False,
        formatters={
            "H0_m": "{:.3f}".format,
            "anchor_relative_level_m": "{:+.3f}".format,
            "anchor_area_km2": "{:.4f}".format,
            "anchor_area_error_vs_sentinel_km2": "{:+.4f}".format,
            "anchor_volume_km3": "{:.5f}".format,
        },
    )
)

banner("QC 3 - CURVA HIPSOMETRICA COMPLETA")

relative_min = -max_depth_m
relative_levels = np.arange(
    relative_min,
    0.0 + CURVE_STEP_M / 2.0,
    CURVE_STEP_M,
)

curve_rows = []

for relative_level in relative_levels:
    n, area_km2, volume_km3, stage_mean_depth_m = stage_metrics(
        sorted_depths,
        cumulative_depths,
        pixel_area_m2,
        float(relative_level),
    )
    curve_rows.append(
        {
            "relative_level_m": float(relative_level),
            "wet_pixels": n,
            "area_km2": area_km2,
            "volume_km3": volume_km3,
            "mean_water_depth_m": stage_mean_depth_m,
            "area_pct_reference": 100.0 * area_km2 / reference_area_km2,
            "volume_pct_reference": 100.0 * volume_km3 / reference_volume_km3,
            "exposed_area_from_reference_km2": reference_area_km2 - area_km2,
            "storage_loss_from_reference_km3": reference_volume_km3 - volume_km3,
            "absolute_level_temporal_2014_median_m": H2014_MED + float(relative_level),
            "absolute_level_temporal_2014_min_m": H2014_MIN + float(relative_level),
            "absolute_level_temporal_2014_max_m": H2014_MAX + float(relative_level),
            "absolute_level_area_anchored_2026_m": H0_area_anchor + float(relative_level),
        }
    )

curve_df = pd.DataFrame(curve_rows)

curve_df["dA_dH_km2_per_m"] = np.gradient(
    curve_df["area_km2"].to_numpy(dtype=float),
    curve_df["relative_level_m"].to_numpy(dtype=float),
)

curve_df["dV_dH_numeric_km2"] = np.gradient(
    curve_df["volume_km3"].to_numpy(dtype=float) * 1000.0,
    curve_df["relative_level_m"].to_numpy(dtype=float),
)

curve_df["dVdH_minus_area_km2"] = (
    curve_df["dV_dH_numeric_km2"] - curve_df["area_km2"]
)

if not np.all(np.diff(curve_df["area_km2"].to_numpy()) >= -1e-12):
    raise RuntimeError("La curva area-nivel no es monotona.")

if not np.all(np.diff(curve_df["volume_km3"].to_numpy()) >= -1e-12):
    raise RuntimeError("La curva volumen-nivel no es monotona.")

curve_df.to_csv(OUT_CURVE, index=False, encoding="utf-8-sig")

closure_abs = np.abs(curve_df["dVdH_minus_area_km2"].to_numpy())
closure_core = closure_abs[1:-1] if len(closure_abs) > 2 else closure_abs

print(f"[CURVE ROWS] = {len(curve_df):,}")
print(f"[RELATIVE RANGE] = {relative_levels.min():.2f} a {relative_levels.max():.2f} m")
print(f"[dV/dH - A] median abs = {np.nanmedian(closure_core):.4f} km2")
print(f"[dV/dH - A] p95 abs = {np.nanpercentile(closure_core, 95):.4f} km2")

banner("QC 4 - NIVELES RELATIVOS CLAVE")

contour_rows = []

for relative_level in KEY_RELATIVE_LEVELS:
    if relative_level < relative_min:
        continue
    n, area_km2, volume_km3, stage_mean_depth_m = stage_metrics(
        sorted_depths,
        cumulative_depths,
        pixel_area_m2,
        relative_level,
    )
    contour_rows.append(
        {
            "relative_level_m": relative_level,
            "wet_pixels": n,
            "area_km2": area_km2,
            "volume_km3": volume_km3,
            "mean_water_depth_m": stage_mean_depth_m,
            "area_pct_reference": 100.0 * area_km2 / reference_area_km2,
            "volume_pct_reference": 100.0 * volume_km3 / reference_volume_km3,
            "exposed_area_km2": reference_area_km2 - area_km2,
            "storage_loss_km3": reference_volume_km3 - volume_km3,
        }
    )

contours_df = pd.DataFrame(contour_rows)
contours_df.to_csv(OUT_CONTOURS, index=False, encoding="utf-8-sig")

print(
    contours_df.to_string(
        index=False,
        formatters={
            "relative_level_m": "{:+.1f}".format,
            "area_km2": "{:.3f}".format,
            "volume_km3": "{:.3f}".format,
            "mean_water_depth_m": "{:.2f}".format,
            "area_pct_reference": "{:.2f}".format,
            "volume_pct_reference": "{:.2f}".format,
            "exposed_area_km2": "{:.3f}".format,
            "storage_loss_km3": "{:.3f}".format,
        },
    )
)

banner("QC 5 - BANDAS DE PROFUNDIDAD")

band_rows = []
total_depth_volume_m3 = np.sum(positive_depths, dtype=np.float64) * pixel_area_m2

for lo, hi in DEPTH_BANDS:
    if hi >= 400.0:
        mask = (positive_depths >= lo) & (positive_depths <= max_depth_m)
        hi_label = max_depth_m
    else:
        mask = (positive_depths >= lo) & (positive_depths < hi)
        hi_label = hi

    n = int(mask.sum())
    area_km2 = n * pixel_area_m2 / 1e6
    volume_km3 = np.sum(positive_depths[mask], dtype=np.float64) * pixel_area_m2 / 1e9
    mean_band_depth = float(np.mean(positive_depths[mask])) if n > 0 else np.nan

    band_rows.append(
        {
            "depth_lo_m": lo,
            "depth_hi_m": hi_label,
            "pixels": n,
            "area_km2": area_km2,
            "area_pct_reference": 100.0 * area_km2 / reference_area_km2,
            "reference_volume_contribution_km3": volume_km3,
            "volume_pct_reference": 100.0 * volume_km3 / reference_volume_km3,
            "mean_depth_in_band_m": mean_band_depth,
        }
    )

bands_df = pd.DataFrame(band_rows)
bands_df.to_csv(OUT_BANDS, index=False, encoding="utf-8-sig")

print(
    bands_df.to_string(
        index=False,
        formatters={
            "depth_lo_m": "{:.1f}".format,
            "depth_hi_m": "{:.1f}".format,
            "area_km2": "{:.3f}".format,
            "area_pct_reference": "{:.2f}".format,
            "reference_volume_contribution_km3": "{:.3f}".format,
            "volume_pct_reference": "{:.2f}".format,
            "mean_depth_in_band_m": "{:.2f}".format,
        },
    )
)

banner("QC 6 - PERCENTILES DE PROFUNDIDAD")

percentile_values = np.percentile(positive_depths, DEPTH_PERCENTILES)

percentiles_df = pd.DataFrame(
    {
        "percentile": DEPTH_PERCENTILES,
        "depth_m": percentile_values,
    }
)

percentiles_df.to_csv(OUT_PERCENTILES, index=False, encoding="utf-8-sig")
print(percentiles_df.to_string(index=False, formatters={"depth_m": "{:.2f}".format}))

banner("QC 7 - COMPARACION CON MORFOMETRIA PUBLICADA")

published_df = pd.DataFrame(
    [
        {
            "metric": "area_km2",
            "raster_value": reference_area_km2,
            "published_value": PUBLISHED_AREA_KM2,
            "difference": reference_area_km2 - PUBLISHED_AREA_KM2,
            "difference_pct_published": 100.0 * (reference_area_km2 - PUBLISHED_AREA_KM2) / PUBLISHED_AREA_KM2,
        },
        {
            "metric": "volume_km3",
            "raster_value": reference_volume_km3,
            "published_value": PUBLISHED_VOLUME_KM3,
            "difference": reference_volume_km3 - PUBLISHED_VOLUME_KM3,
            "difference_pct_published": 100.0 * (reference_volume_km3 - PUBLISHED_VOLUME_KM3) / PUBLISHED_VOLUME_KM3,
        },
        {
            "metric": "max_depth_m",
            "raster_value": max_depth_m,
            "published_value": PUBLISHED_MAX_DEPTH_M,
            "difference": max_depth_m - PUBLISHED_MAX_DEPTH_M,
            "difference_pct_published": 100.0 * (max_depth_m - PUBLISHED_MAX_DEPTH_M) / PUBLISHED_MAX_DEPTH_M,
        },
        {
            "metric": "mean_depth_m",
            "raster_value": mean_depth_m,
            "published_value": PUBLISHED_MEAN_DEPTH_M,
            "difference": mean_depth_m - PUBLISHED_MEAN_DEPTH_M,
            "difference_pct_published": 100.0 * (mean_depth_m - PUBLISHED_MEAN_DEPTH_M) / PUBLISHED_MEAN_DEPTH_M,
        },
    ]
)

published_df.to_csv(OUT_PUBLISHED, index=False, encoding="utf-8-sig")

print(
    published_df.to_string(
        index=False,
        formatters={
            "raster_value": "{:.4f}".format,
            "published_value": "{:.4f}".format,
            "difference": "{:+.4f}".format,
            "difference_pct_published": "{:+.2f}".format,
        },
    )
)

banner("QC 8 - HIPSOMETRIA FUTURA 2030/2050/2090")

future = pd.read_pickle(PATH_FUTURE)

if isinstance(future, pd.Series):
    future = future.to_frame()

required_future_columns = {"date", "model", "scenario", "level_m"}

if not required_future_columns.issubset(set(future.columns)):
    raise RuntimeError(
        f"Faltan columnas en futuro: {sorted(required_future_columns - set(future.columns))}"
    )

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

future_rows = []

for scenario in SCENARIOS:
    for year in HORIZONS:
        sub = future[
            (future["_scenario"] == scenario)
            & (future["_year"] == year)
        ].copy()

        if len(sub) == 0:
            raise RuntimeError(f"No hay datos futuros para {scenario} {year}.")

        annual_by_model = sub.groupby("model")["_level"].mean().dropna()

        if len(annual_by_model) != 15:
            raise RuntimeError(f"{scenario} {year}: se esperaban 15 GCM; hay {len(annual_by_model)}.")

        quantiles = {
            "p05": float(annual_by_model.quantile(0.05)),
            "p50": float(annual_by_model.quantile(0.50)),
            "p95": float(annual_by_model.quantile(0.95)),
        }

        expected = EXPECTED_P50[(scenario, year)]

        if abs(quantiles["p50"] - expected) > 5e-4:
            raise RuntimeError(
                f"{scenario} {year}: p50 {quantiles['p50']:.6f} no reproduce {expected:.6f}."
            )

        for quantile_name, absolute_level in quantiles.items():
            for reference_name, H0 in references.items():
                relative_level = absolute_level - H0
                n, area_km2, volume_km3, stage_mean_depth_m = stage_metrics(
                    sorted_depths,
                    cumulative_depths,
                    pixel_area_m2,
                    relative_level,
                )

                future_rows.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "quantile": quantile_name,
                        "absolute_level_m": absolute_level,
                        "reference": reference_name,
                        "H0_m": H0,
                        "relative_level_m": relative_level,
                        "wet_pixels": n,
                        "area_km2": area_km2,
                        "volume_km3": volume_km3,
                        "mean_water_depth_m": stage_mean_depth_m,
                        "area_pct_reference": 100.0 * area_km2 / reference_area_km2,
                        "volume_pct_reference": 100.0 * volume_km3 / reference_volume_km3,
                        "exposed_area_vs_reference_km2": reference_area_km2 - area_km2,
                        "storage_loss_vs_reference_km3": reference_volume_km3 - volume_km3,
                        "area_change_vs_current_sentinel_km2": area_km2 - current_area_km2,
                    }
                )

future_df = pd.DataFrame(future_rows)
future_df.to_csv(OUT_FUTURE, index=False, encoding="utf-8-sig")

future_p50 = future_df[future_df["quantile"] == "p50"].copy()

print(
    future_p50[
        [
            "scenario",
            "year",
            "reference",
            "absolute_level_m",
            "relative_level_m",
            "area_km2",
            "volume_km3",
            "exposed_area_vs_reference_km2",
            "storage_loss_vs_reference_km3",
        ]
    ].to_string(
        index=False,
        formatters={
            "absolute_level_m": "{:.3f}".format,
            "relative_level_m": "{:+.3f}".format,
            "area_km2": "{:.3f}".format,
            "volume_km3": "{:.3f}".format,
            "exposed_area_vs_reference_km2": "{:.3f}".format,
            "storage_loss_vs_reference_km3": "{:.3f}".format,
        },
    )
)

banner("QC 9 - DIAGNOSTICO FISICO")

area_band_sum = float(bands_df["area_km2"].sum())
volume_band_sum = float(bands_df["reference_volume_contribution_km3"].sum())

diagnostic_df = pd.DataFrame(
    [
        {
            "metric": "current_sentinel_area_km2",
            "value": current_area_km2,
            "unit": "km2",
        },
        {
            "metric": "reference_area_km2",
            "value": reference_area_km2,
            "unit": "km2",
        },
        {
            "metric": "reference_volume_km3",
            "value": reference_volume_km3,
            "unit": "km3",
        },
        {
            "metric": "max_depth_m",
            "value": max_depth_m,
            "unit": "m",
        },
        {
            "metric": "mean_depth_m",
            "value": mean_depth_m,
            "unit": "m",
        },
        {
            "metric": "median_depth_m",
            "value": median_depth_m,
            "unit": "m",
        },
        {
            "metric": "area_anchored_H0_m",
            "value": H0_area_anchor,
            "unit": "m",
        },
        {
            "metric": "area_anchored_H0_minus_H2014_median_m",
            "value": H0_area_anchor - H2014_MED,
            "unit": "m",
        },
        {
            "metric": "area_band_sum_km2",
            "value": area_band_sum,
            "unit": "km2",
        },
        {
            "metric": "area_band_closure_error_km2",
            "value": area_band_sum - reference_area_km2,
            "unit": "km2",
        },
        {
            "metric": "volume_band_sum_km3",
            "value": volume_band_sum,
            "unit": "km3",
        },
        {
            "metric": "volume_band_closure_error_km3",
            "value": volume_band_sum - reference_volume_km3,
            "unit": "km3",
        },
        {
            "metric": "dVdH_area_median_abs_error_km2",
            "value": float(np.nanmedian(closure_core)),
            "unit": "km2",
        },
        {
            "metric": "dVdH_area_p95_abs_error_km2",
            "value": float(np.nanpercentile(closure_core, 95)),
            "unit": "km2",
        },
    ]
)

diagnostic_df.to_csv(OUT_DIAGNOSTIC, index=False, encoding="utf-8-sig")
print(diagnostic_df.to_string(index=False))

banner("FIGURAS")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(curve_df["relative_level_m"], curve_df["area_km2"])
ax.axvline(H_ANCHOR - H2014_MED, linestyle="--", label="Ancla con H0 temporal 2014")
ax.axvline(H_ANCHOR - H0_area_anchor, linestyle="--", label="Ancla con H0 por area")
ax.set_xlabel("Nivel relativo al H0 (m)")
ax.set_ylabel("Area inundada (km2)")
ax.set_title("16C1 - Curva hipsometrica total de area")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_AREA, dpi=180, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(curve_df["relative_level_m"], curve_df["volume_km3"])
ax.axvline(H_ANCHOR - H2014_MED, linestyle="--", label="Ancla con H0 temporal 2014")
ax.axvline(H_ANCHOR - H0_area_anchor, linestyle="--", label="Ancla con H0 por area")
ax.set_xlabel("Nivel relativo al H0 (m)")
ax.set_ylabel("Volumen (km3)")
ax.set_title("16C1 - Curva hipsometrica total de volumen")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_VOLUME, dpi=180, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(curve_df["area_km2"], curve_df["volume_km3"])
ax.set_xlabel("Area inundada (km2)")
ax.set_ylabel("Volumen (km3)")
ax.set_title("16C1 - Relacion area-volumen")
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_FIG_AREA_VOLUME, dpi=180, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(positive_depths, bins=80)
ax.set_xlabel("Profundidad (m)")
ax.set_ylabel("Pixeles")
ax.set_title("16C1 - Distribucion de profundidad AMSCLAE")
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_FIG_DEPTH, dpi=180, bbox_inches="tight")
plt.close(fig)

sensitivity_plot = future_p50[
    future_p50["reference"].isin(["temporal_2014_median", "area_anchored_2026"])
].copy()

sensitivity_plot["label"] = (
    sensitivity_plot["scenario"].str.upper()
    + " "
    + sensitivity_plot["year"].astype(str)
    + " "
    + sensitivity_plot["reference"]
)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
x = np.arange(len(sensitivity_plot))
ax.bar(x, sensitivity_plot["area_km2"].to_numpy())
ax.set_xticks(x)
ax.set_xticklabels(sensitivity_plot["label"].to_list(), rotation=45, ha="right")
ax.set_ylabel("Area hipsometrica (km2)")
ax.set_title("16C1 - Sensibilidad del area futura a la referencia vertical")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
fig.savefig(OUT_FIG_SENSITIVITY, dpi=180, bbox_inches="tight")
plt.close(fig)

banner("METADATA")

metadata = {
    "step": "16C1",
    "status": "TOTAL_HYPSOMETRY_DIAGNOSTIC_COMPLETE",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "amsclae_source": str(PATH_AMSCLAE.resolve()),
    "amsclae_sha256": sha256_file(PATH_AMSCLAE),
    "raster": {
        "shape": list(bathy_shape),
        "crs": str(bathy_crs),
        "valid_pixels": int(n_valid),
        "pixel_area_m2": float(pixel_area_m2),
        "reference_area_km2": float(reference_area_km2),
        "reference_volume_km3": float(reference_volume_km3),
        "max_depth_m": float(max_depth_m),
        "mean_depth_m": float(mean_depth_m),
        "median_depth_m": float(median_depth_m),
    },
    "published_reference": {
        "area_km2": PUBLISHED_AREA_KM2,
        "volume_km3": PUBLISHED_VOLUME_KM3,
        "max_depth_m": PUBLISHED_MAX_DEPTH_M,
        "mean_depth_m": PUBLISHED_MEAN_DEPTH_M,
    },
    "current_lake": {
        "date": ANCHOR_DATE,
        "level_m": H_ANCHOR,
        "sentinel_area_km2": float(current_area_km2),
    },
    "vertical_references": {
        name: float(value)
        for name, value in references.items()
    },
    "curve": {
        "relative_min_m": float(relative_min),
        "relative_max_m": 0.0,
        "step_m": CURVE_STEP_M,
        "rows": int(len(curve_df)),
    },
    "future": {
        "scenarios": SCENARIOS,
        "horizons": HORIZONS,
        "quantiles": ["p05", "p50", "p95"],
        "models_per_scenario": 15,
    },
    "interpretation": {
        "spatial_coast_authority": "resultados/costa_future_compatible",
        "hypsometry_role": "aggregate area-volume morphometry only",
        "area_anchored_reference": "bulk scalar sensitivity reference only, not a spatial datum correction",
        "future_levels": "conditional CMIP6 scenario trajectories, not deterministic forecasts",
    },
    "outputs_sha256": {},
}

output_files = [
    OUT_CURVE,
    OUT_CONTOURS,
    OUT_BANDS,
    OUT_PERCENTILES,
    OUT_REFERENCES,
    OUT_FUTURE,
    OUT_PUBLISHED,
    OUT_DIAGNOSTIC,
    OUT_FIG_AREA,
    OUT_FIG_VOLUME,
    OUT_FIG_AREA_VOLUME,
    OUT_FIG_DEPTH,
    OUT_FIG_SENSITIVITY,
]

metadata["outputs_sha256"] = {
    p.name: sha256_file(p)
    for p in output_files
}

with open(OUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

banner("DECISION QC 16C1")

temporal_anchor_row = references_df[
    references_df["reference"] == "temporal_2014_median"
].iloc[0]

area_anchor_row = references_df[
    references_df["reference"] == "area_anchored_2026"
].iloc[0]

print(f"[REFERENCE AREA] = {reference_area_km2:.4f} km2")
print(f"[REFERENCE VOLUME] = {reference_volume_km3:.5f} km3")
print(f"[MAX / MEAN / MEDIAN DEPTH] = {max_depth_m:.2f} / {mean_depth_m:.2f} / {median_depth_m:.2f} m")
print(f"[CURRENT SENTINEL AREA] = {current_area_km2:.4f} km2")
print(f"[TEMPORAL 2014 H0] = {H2014_MED:.3f} m")
print(f"[TEMPORAL 2014 ANCHOR AREA] = {temporal_anchor_row['anchor_area_km2']:.4f} km2")
print(f"[AREA-ANCHORED H0] = {H0_area_anchor:.3f} m")
print(f"[AREA-ANCHORED H0 - TEMPORAL H0] = {H0_area_anchor - H2014_MED:+.3f} m")
print(f"[AREA-ANCHORED ANCHOR AREA] = {area_anchor_row['anchor_area_km2']:.4f} km2")
print(f"[FULL HIPSOMETRIC CURVE] = {relative_min:.2f} a 0.00 m | step {CURVE_STEP_M:.2f} m")
print(f"[FUTURE HORIZONS] = {HORIZONS}")
print("[STATUS] TOTAL_HYPSOMETRY_DIAGNOSTIC_COMPLETE")
print("[NEXT] decidir la referencia agregada que puede congelarse para area/volumen sin alterar la costa espacial.")

banner("FIN PASO 16C1")
print("[OUTPUT DIRECTORY]")
print(OUT_DIR)
