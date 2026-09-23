__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PATH_AMSCLAE = ROOT / "data" / "external" / "batimetria_amsclae.tif"

COAST_DIR = ROOT / "outputs" / "costa_future_compatible"
HYPSO_DIR = ROOT / "outputs" / "hipsometria_future_compatible"
C3_DIR = ROOT / "outputs" / "consistencia_costa_hipsometria_16C3"

PATH_CURRENT = COAST_DIR / "mascara_lago_actual_2024_2026.tif"
PATH_CAPABILITY = COAST_DIR / "capacidad_espacial_costera.tif"
PATH_THRESHOLD = COAST_DIR / "nivel_umbral_costero_m.tif"
PATH_LEVELS_2030_2050 = COAST_DIR / "niveles_ensemble_2030_2050.csv"
PATH_FUTURE = ROOT / "outputs" / "nivel_future_compatible" / "nivel_diario_cmip6_future_compatible.pkl"
PATH_HYPSO_SUMMARY = HYPSO_DIR / "resumen_horizontes_area_volumen.csv"
PATH_C3 = C3_DIR / "16C3_auditoria_consistencia_area_volumen.csv"

OUT_DIR = ROOT / "outputs" / "area_futura_acotada_16C4"
OUT_TABLE = OUT_DIR / "16C4_limites_area_futura.csv"
OUT_P50 = OUT_DIR / "16C4_resumen_p50_area_volumen.csv"
OUT_METADATA = OUT_DIR / "16C4_metadata.json"
OUT_FIG = OUT_DIR / "16C4_limites_area_p50.png"

H2014_MIN = 1557.310
H2014_MAX = 1558.090
H_ANCHOR = 1552.770
INTERIOR_DISTANCE_M = 120.0
FLOAT_NODATA = -9999.0

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


def normalize_scenario(value):
    s = str(value).lower().replace("_", "").replace("-", "")
    if "245" in s:
        return "ssp245"
    if "585" in s:
        return "ssp585"
    return s


for p in [
    PATH_AMSCLAE,
    PATH_CURRENT,
    PATH_CAPABILITY,
    PATH_THRESHOLD,
    PATH_FUTURE,
    PATH_HYPSO_SUMMARY,
    PATH_C3,
]:
    if not p.exists():
        raise FileNotFoundError(p)

OUT_DIR.mkdir(parents=True, exist_ok=True)

banner("PASO 16C4 - AREA FUTURA ACOTADA Y VOLUMEN RETENIDO")

c3 = pd.read_csv(PATH_C3)

if not np.all(c3["area_consistency_status"] == "CONTRADICTION"):
    raise RuntimeError("16C3 no rechazo todos los puntos de area 16C2 para 2030/2050.")

if not np.all(c3["storage_physical_status"] == "PLAUSIBLE"):
    raise RuntimeError("16C3 no retuvo el almacenamiento como fisicamente plausible.")

with rasterio.open(PATH_CURRENT) as ds:
    current = ds.read(1) > 0
    profile = ds.profile.copy()
    shape = ds.shape
    transform = ds.transform
    crs = ds.crs

with rasterio.open(PATH_CAPABILITY) as ds:
    if ds.shape != shape or ds.transform != transform or ds.crs != crs:
        raise RuntimeError("Capacidad espacial en grilla incompatible.")
    capability = ds.read(1).astype(np.uint8)

with rasterio.open(PATH_THRESHOLD) as ds:
    if ds.shape != shape or ds.transform != transform or ds.crs != crs:
        raise RuntimeError("Umbral costero en grilla incompatible.")
    threshold = ds.read(1).astype(np.float32)
    threshold_nodata = ds.nodata

pixel_area_m2 = abs(
    transform.a * transform.e
    - transform.b * transform.d
)

current_area_km2 = current.sum() * pixel_area_m2 / 1e6

threshold_valid = np.isfinite(threshold)

if threshold_nodata is not None:
    threshold_valid &= threshold != threshold_nodata

known_capability = np.isin(capability, [1, 2])

if not np.array_equal(threshold_valid, known_capability):
    raise RuntimeError("Los umbrales validos no coinciden con la capacidad conocida.")

banner("QC 1 - AMSCLAE EN GRILLA MAESTRA")

with rasterio.open(PATH_AMSCLAE) as ds:
    bathy_native = ds.read(1).astype(np.float32)
    bathy_nodata = ds.nodata
    bathy_transform = ds.transform
    bathy_crs = ds.crs

native_valid = np.isfinite(bathy_native)

if bathy_nodata is not None:
    native_valid &= bathy_native != bathy_nodata

bathy_20 = np.full(shape, FLOAT_NODATA, dtype=np.float32)

reproject(
    source=bathy_native,
    destination=bathy_20,
    src_transform=bathy_transform,
    src_crs=bathy_crs,
    src_nodata=bathy_nodata,
    dst_transform=transform,
    dst_crs=crs,
    dst_nodata=FLOAT_NODATA,
    resampling=Resampling.bilinear,
)

valid_20_u8 = np.zeros(shape, dtype=np.uint8)

reproject(
    source=native_valid.astype(np.uint8),
    destination=valid_20_u8,
    src_transform=bathy_transform,
    src_crs=bathy_crs,
    src_nodata=0,
    dst_transform=transform,
    dst_crs=crs,
    dst_nodata=0,
    resampling=Resampling.nearest,
)

valid_20 = (
    (valid_20_u8 > 0)
    & np.isfinite(bathy_20)
    & (bathy_20 != FLOAT_NODATA)
)

distance_inside_m = distance_transform_edt(
    current,
    sampling=(abs(transform.e), abs(transform.a)),
)

interior_guard = current & (distance_inside_m >= INTERIOR_DISTANCE_M)

zlow = np.full(shape, np.nan, dtype=np.float32)
zhigh = np.full(shape, np.nan, dtype=np.float32)

zlow[valid_20] = H2014_MIN + bathy_20[valid_20]
zhigh[valid_20] = H2014_MAX + bathy_20[valid_20]

anchor_robust_wet_ams = (
    current
    & interior_guard
    & valid_20
    & (H_ANCHOR >= zhigh)
)

print(f"[CURRENT AREA] = {current_area_km2:.4f} km2")
print(f"[AMSCLAE VALID CURRENT] = {np.sum(current & valid_20):,} pix")
print(f"[INTERIOR >= {INTERIOR_DISTANCE_M:.0f} m] = {np.sum(interior_guard):,} pix")
print(f"[INTERIOR ROBUST WET AT ANCHOR] = {np.sum(anchor_robust_wet_ams):,} pix")

banner("QC 2 - NIVELES FUTUROS")

future = pd.read_pickle(PATH_FUTURE)

if isinstance(future, pd.Series):
    future = future.to_frame()

required = {"date", "model", "scenario", "level_m"}

if not required.issubset(set(future.columns)):
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

level_rows = []

for scenario in SCENARIOS:
    for year in HORIZONS:
        sub = future[
            (future["_scenario"] == scenario)
            & (future["_year"] == year)
        ]

        annual = sub.groupby("model")["_level"].mean().dropna()

        if len(annual) != 15:
            raise RuntimeError(f"{scenario} {year}: se esperaban 15 GCM.")

        q = {
            "p05": float(annual.quantile(0.05)),
            "p50": float(annual.quantile(0.50)),
            "p95": float(annual.quantile(0.95)),
        }

        if abs(q["p50"] - EXPECTED_P50[(scenario, year)]) > 5e-4:
            raise RuntimeError(f"{scenario} {year}: p50 no reproduce la rama congelada.")

        for quantile, H in q.items():
            level_rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "quantile": quantile,
                    "level_m": H,
                }
            )

levels_df = pd.DataFrame(level_rows)

print(
    levels_df.to_string(
        index=False,
        formatters={"level_m": "{:.3f}".format},
    )
)

banner("QC 3 - CLASIFICACION Y LIMITES")

rows = []
generated = []

uint_profile = profile.copy()
uint_profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate",
)

for _, lr in levels_df.iterrows():
    scenario = str(lr["scenario"])
    year = int(lr["year"])
    quantile = str(lr["quantile"])
    H = float(lr["level_m"])
    below_anchor = H <= H_ANCHOR

    known_anchor_wet = (
        current
        & known_capability
        & (H_ANCHOR >= threshold)
    )

    known_anchor_conflict = (
        current
        & known_capability
        & (~known_anchor_wet)
    )

    known_future_wet = (
        known_anchor_wet
        & (H >= threshold)
    )

    known_future_exposed = (
        known_anchor_wet
        & (H < threshold)
    )

    class0_current = current & (capability == 0)

    interior_candidate = (
        class0_current
        & anchor_robust_wet_ams
    )

    interior_future_wet = (
        interior_candidate
        & (H >= zhigh)
    )

    interior_future_exposed = (
        interior_candidate
        & (H < zlow)
    )

    interior_future_uncertain = (
        interior_candidate
        & (~interior_future_wet)
        & (~interior_future_exposed)
    )

    coastal_unknown = (
        current
        & (capability == 3)
    )

    class0_not_classified = (
        class0_current
        & (~interior_future_wet)
        & (~interior_future_exposed)
    )

    certain_wet = (
        known_future_wet
        | interior_future_wet
    )

    certain_exposed = (
        known_future_exposed
        | interior_future_exposed
    )

    uncertain = (
        current
        & (~certain_wet)
        & (~certain_exposed)
    )

    if np.any(certain_wet & certain_exposed):
        raise RuntimeError("Clases wet/exposed se superponen.")

    if not np.array_equal(
        current,
        certain_wet | certain_exposed | uncertain,
    ):
        raise RuntimeError("La particion del lago actual no cierra.")

    certain_wet_area = certain_wet.sum() * pixel_area_m2 / 1e6
    certain_exposed_area = certain_exposed.sum() * pixel_area_m2 / 1e6
    uncertain_area = uncertain.sum() * pixel_area_m2 / 1e6

    if below_anchor:
        water_area_lower = certain_wet_area
        water_area_upper = current_area_km2 - certain_exposed_area
        loss_area_lower = certain_exposed_area
        loss_area_upper = current_area_km2 - certain_wet_area
        bound_status = "VALID_FOR_DECLINE_FROM_CURRENT_MASK"
    else:
        water_area_lower = np.nan
        water_area_upper = np.nan
        loss_area_lower = np.nan
        loss_area_upper = np.nan
        bound_status = "NOT_VALID_LEVEL_ABOVE_2026_ANCHOR"

    rows.append(
        {
            "scenario": scenario,
            "year": year,
            "quantile": quantile,
            "level_m": H,
            "level_change_vs_anchor_m": H - H_ANCHOR,
            "bound_status": bound_status,
            "certain_wet_pixels": int(certain_wet.sum()),
            "certain_wet_area_km2": certain_wet_area,
            "certain_exposed_pixels": int(certain_exposed.sum()),
            "certain_exposed_area_km2": certain_exposed_area,
            "uncertain_pixels": int(uncertain.sum()),
            "uncertain_area_km2": uncertain_area,
            "water_area_lower_bound_km2": water_area_lower,
            "water_area_upper_bound_km2": water_area_upper,
            "area_loss_lower_bound_km2": loss_area_lower,
            "area_loss_upper_bound_km2": loss_area_upper,
            "known_threshold_anchor_conflict_area_km2": known_anchor_conflict.sum() * pixel_area_m2 / 1e6,
            "coastal_unknown_area_km2": coastal_unknown.sum() * pixel_area_m2 / 1e6,
            "interior_bracket_uncertain_area_km2": interior_future_uncertain.sum() * pixel_area_m2 / 1e6,
            "class0_unclassified_area_km2": class0_not_classified.sum() * pixel_area_m2 / 1e6,
        }
    )

    if quantile == "p50":
        out_class = np.zeros(shape, dtype=np.uint8)
        out_class[current & known_future_wet] = 1
        out_class[current & known_future_exposed] = 2
        out_class[current & interior_future_wet] = 3
        out_class[current & interior_future_exposed] = 4
        out_class[uncertain] = 5

        out_path = OUT_DIR / f"16C4_{scenario}_{year}_p50_clases_area_acotada.tif"

        with rasterio.open(out_path, "w", **uint_profile) as dst:
            dst.write(out_class, 1)
            dst.set_band_description(1, "clase_area_futura_acotada")

        generated.append(out_path)

        print()
        print(f"[{scenario.upper()} {year} p50] H={H:.3f} m")
        print(f"  agua cierta = {certain_wet_area:.4f} km2")
        print(f"  expuesto cierto = {certain_exposed_area:.4f} km2")
        print(f"  incierto = {uncertain_area:.4f} km2")
        print(f"  area futura bounds = {water_area_lower:.4f} a {water_area_upper:.4f} km2")

bounds_df = pd.DataFrame(rows)
bounds_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

banner("QC 4 - COMPARACION CON 16B8 Y 16C3")

for scenario in SCENARIOS:
    for year in [2030, 2050]:
        row = bounds_df[
            (bounds_df["scenario"] == scenario)
            & (bounds_df["year"] == year)
            & (bounds_df["quantile"] == "p50")
        ].iloc[0]

        c3row = c3[
            (c3["scenario"] == scenario)
            & (c3["year"] == year)
        ].iloc[0]

        hard_c3 = float(c3row["hard_minimum_area_loss_km2"])
        hard_c4 = float(row["certain_exposed_area_km2"])

        if abs(hard_c4 - hard_c3) > 5e-4:
            raise RuntimeError(
                f"{scenario} {year}: la exposicion cierta 16C4 no reproduce 16C3."
            )

        print(
            f"[{scenario.upper()} {year}] "
            f"16C3={hard_c3:.4f} km2 | "
            f"16C4={hard_c4:.4f} km2 | identical=True"
        )

hypso_summary = pd.read_csv(PATH_HYPSO_SUMMARY)

p50_rows = []

for scenario in SCENARIOS:
    for year in HORIZONS:
        b = bounds_df[
            (bounds_df["scenario"] == scenario)
            & (bounds_df["year"] == year)
            & (bounds_df["quantile"] == "p50")
        ].iloc[0]

        h = hypso_summary[
            (hypso_summary["scenario"] == scenario)
            & (hypso_summary["year"] == year)
        ]

        if len(h) != 1:
            raise RuntimeError(f"Resumen hipsometrico invalido {scenario} {year}.")

        h = h.iloc[0]

        p50_rows.append(
            {
                "scenario": scenario,
                "year": year,
                "level_p50_m": float(b["level_m"]),
                "water_area_lower_bound_km2": float(b["water_area_lower_bound_km2"]),
                "water_area_upper_bound_km2": float(b["water_area_upper_bound_km2"]),
                "certain_exposed_area_km2": float(b["certain_exposed_area_km2"]),
                "uncertain_area_km2": float(b["uncertain_area_km2"]),
                "storage_loss_p50_vs_anchor_km3": float(h["storage_loss_p50_vs_anchor_km3"]),
                "volume_raw_p50_km3": float(h["volume_raw_p50_km3"]),
                "area_point_estimate_status": "REJECTED_BY_16C3",
                "storage_change_status": "RETAINED_AS_PHYSICALLY_PLAUSIBLE",
            }
        )

p50_df = pd.DataFrame(p50_rows)
p50_df.to_csv(OUT_P50, index=False, encoding="utf-8-sig")

banner("QC 5 - RESUMEN P50")

print(
    p50_df.to_string(
        index=False,
        formatters={
            "level_p50_m": "{:.3f}".format,
            "water_area_lower_bound_km2": "{:.3f}".format,
            "water_area_upper_bound_km2": "{:.3f}".format,
            "certain_exposed_area_km2": "{:.3f}".format,
            "uncertain_area_km2": "{:.3f}".format,
            "storage_loss_p50_vs_anchor_km3": "{:.3f}".format,
            "volume_raw_p50_km3": "{:.3f}".format,
        },
    )
)

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111)

labels = [
    f"{r['scenario'].upper()} {int(r['year'])}"
    for _, r in p50_df.iterrows()
]

x = np.arange(len(p50_df))
lower = p50_df["water_area_lower_bound_km2"].to_numpy()
upper = p50_df["water_area_upper_bound_km2"].to_numpy()
mid = (lower + upper) / 2.0
err = np.vstack([mid - lower, upper - mid])

ax.errorbar(
    x,
    mid,
    yerr=err,
    fmt="o",
    capsize=5,
)

ax.axhline(current_area_km2, linestyle="--", label="Area actual Sentinel")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Area de agua compatible con evidencia (km2)")
ax.set_title("16C4 - Limites conservadores de area futura p50")
ax.grid(axis="y", alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
plt.close(fig)

metadata = {
    "step": "16C4",
    "status": "FUTURE_AREA_BOUNDS_WITH_STORAGE_RETAINED",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "current_reference": {
        "level_m": H_ANCHOR,
        "area_km2": current_area_km2,
    },
    "interior_guard_distance_m": INTERIOR_DISTANCE_M,
    "vertical_bracket_m": {
        "H2014_min": H2014_MIN,
        "H2014_max": H2014_MAX,
    },
    "p50_class_codes": {
        "0": "outside_current_reference_lake",
        "1": "certain_water_direct_or_locally_supported_threshold",
        "2": "certain_exposed_direct_or_locally_supported_threshold",
        "3": "certain_water_deep_interior_AMSCLAE_bracket",
        "4": "certain_exposed_deep_interior_AMSCLAE_bracket",
        "5": "uncertain_or_unsupported",
    },
    "area_method": {
        "lower_water_bound": "certain-water pixels inside the frozen current-water reference mask",
        "upper_water_bound": "current reference area minus certainly exposed pixels",
        "applicability": "only levels at or below the 2026 anchor; higher levels may inundate land outside the current reference mask",
    },
    "decisions": {
        "16C2_area_point_estimates": "REJECTED_BY_16C3",
        "16C2_storage_change": "RETAINED_AS_PHYSICALLY_PLAUSIBLE",
    },
    "limitations": [
        "The bounds are referenced to the 2024-2026 Sentinel 50% current-water mask.",
        "Unsupported coastal pixels remain uncertain.",
        "AMSCLAE is used outside the supported coastal domain only for robust deep-interior classification at least 120 m inside the current-water mask and with the full 2014 vertical bracket.",
        "No continuous future shoreline is invented.",
        "Storage change is physically plausible but not independently validated by a second bathymetric survey.",
        "Future levels are conditional CMIP6 scenario trajectories, not deterministic forecasts.",
    ],
    "outputs_sha256": {
        OUT_TABLE.name: sha256_file(OUT_TABLE),
        OUT_P50.name: sha256_file(OUT_P50),
        OUT_FIG.name: sha256_file(OUT_FIG),
        **{p.name: sha256_file(p) for p in generated},
    },
}

with open(OUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

banner("DECISION 16C4")
print("[AREA] puntos 16C2 rechazados; se sustituyen por limites conservadores.")
print("[VOLUME] cambio de almacenamiento 16C2 retenido como fisicamente plausible.")
print("[SPATIAL] la costa frozen sigue siendo la autoridad para ubicacion.")
print("[STATUS] FUTURE_AREA_BOUNDS_WITH_STORAGE_RETAINED")
print("[NEXT] integrar productos finales 2030/2050 usando area acotada, almacenamiento y costa espacial frozen.")

banner("FIN PASO 16C4")
print("[OUTPUT DIRECTORY]")
print(OUT_DIR)
