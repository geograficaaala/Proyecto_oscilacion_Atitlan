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

COAST_DIR = ROOT / "outputs" / "costa_future_compatible"
HYPSO_DIR = ROOT / "outputs" / "hipsometria_future_compatible"
C4_DIR = ROOT / "outputs" / "area_futura_acotada_16C4"

PATH_COAST_METADATA = COAST_DIR / "parametros_costa_future_compatible.json"
PATH_HYPSO_METADATA = HYPSO_DIR / "parametros_hipsometria_future_compatible.json"
PATH_C4_METADATA = C4_DIR / "16C4_metadata.json"

PATH_CURRENT = COAST_DIR / "mascara_lago_actual_2024_2026.tif"
PATH_LEVELS = COAST_DIR / "niveles_ensemble_2030_2050.csv"
PATH_EXPOSURE_SUMMARY = COAST_DIR / "resumen_exposicion_2030_2050.csv"
PATH_C4_ALL = C4_DIR / "16C4_limites_area_futura.csv"
PATH_C4_P50 = C4_DIR / "16C4_resumen_p50_area_volumen.csv"
PATH_HYPSO_ALL = HYPSO_DIR / "area_volumen_horizontes_cmip6.csv"
PATH_HYPSO_SUMMARY = HYPSO_DIR / "resumen_horizontes_area_volumen.csv"

OUT_DIR = ROOT / "outputs" / "hidromorfologia_future_compatible"
OUT_FINAL = OUT_DIR / "resumen_hidromorfologico_final.csv"
OUT_QUANTILES = OUT_DIR / "resumen_hidromorfologico_cuantiles.csv"
OUT_LIMITS = OUT_DIR / "limites_area_futura_p50.csv"
OUT_MANIFEST = OUT_DIR / "manifiesto_hidromorfologia_future_compatible.csv"
OUT_METADATA = OUT_DIR / "parametros_hidromorfologia_future_compatible.json"
OUT_FIG_AREA = OUT_DIR / "resumen_area_futura_p50.png"
OUT_FIG_STORAGE = OUT_DIR / "resumen_perdida_almacenamiento_p50.png"

SCENARIOS = ["ssp245", "ssp585"]
PRIMARY_YEARS = [2030, 2050]
STRESS_YEAR = 2090

EXPECTED_CURRENT_AREA_KM2 = 122.0232
EXPECTED_UNKNOWN_COASTAL_AREA_KM2 = 6.7992
EXPECTED_KNOWN_CAPABILITY_PCT = 27.64


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


def copy_exact(src, dst):
    shutil.copy2(src, dst)
    h1 = sha256_file(src)
    h2 = sha256_file(dst)
    if h1 != h2:
        raise RuntimeError(f"Copia no identica: {src}")
    return h2


required = [
    PATH_COAST_METADATA,
    PATH_HYPSO_METADATA,
    PATH_C4_METADATA,
    PATH_CURRENT,
    PATH_LEVELS,
    PATH_EXPOSURE_SUMMARY,
    PATH_C4_ALL,
    PATH_C4_P50,
    PATH_HYPSO_ALL,
    PATH_HYPSO_SUMMARY,
]

for scenario in SCENARIOS:
    for year in PRIMARY_YEARS:
        required.extend(
            [
                COAST_DIR / f"exposicion_litoral_{scenario}_{year}.tif",
                C4_DIR / f"16C4_{scenario}_{year}_p50_clases_area_acotada.tif",
            ]
        )

for year in PRIMARY_YEARS:
    required.append(COAST_DIR / f"consenso_exposicion_{year}.tif")

for p in required:
    if not p.exists():
        raise FileNotFoundError(p)

banner("PASO 16C5 - INTEGRACION HIDROMORFOLOGICA FINAL")

with open(PATH_COAST_METADATA, "r", encoding="utf-8") as f:
    coast_meta = json.load(f)

with open(PATH_HYPSO_METADATA, "r", encoding="utf-8") as f:
    hypso_meta = json.load(f)

with open(PATH_C4_METADATA, "r", encoding="utf-8") as f:
    c4_meta = json.load(f)

if coast_meta.get("status") != "FROZEN_COAST_FUTURE_COMPATIBLE":
    raise RuntimeError("Costa frozen invalida.")

if hypso_meta.get("status") != "FROZEN_HYPSOMETRY_FUTURE_COMPATIBLE":
    raise RuntimeError("Hipsometria frozen invalida.")

if c4_meta.get("status") != "FUTURE_AREA_BOUNDS_WITH_STORAGE_RETAINED":
    raise RuntimeError("16C4 no tiene el status esperado.")

with rasterio.open(PATH_CURRENT) as ds:
    current = ds.read(1) > 0
    transform = ds.transform
    current_shape = ds.shape
    current_crs = ds.crs

pixel_area_m2 = abs(
    transform.a * transform.e
    - transform.b * transform.d
)

current_area_km2 = current.sum() * pixel_area_m2 / 1e6

if abs(current_area_km2 - EXPECTED_CURRENT_AREA_KM2) > 5e-4:
    raise RuntimeError("Area actual inesperada.")

coastal_unknown = float(coast_meta["spatial_capability"]["unknown_area_km2"])
known_pct = float(coast_meta["spatial_capability"]["known_pct"])

if abs(coastal_unknown - EXPECTED_UNKNOWN_COASTAL_AREA_KM2) > 5e-4:
    raise RuntimeError("Area costera desconocida inesperada.")

if abs(known_pct - EXPECTED_KNOWN_CAPABILITY_PCT) > 0.02:
    raise RuntimeError("Capacidad espacial conocida inesperada.")

levels = pd.read_csv(PATH_LEVELS)
exposure = pd.read_csv(PATH_EXPOSURE_SUMMARY)
c4_all = pd.read_csv(PATH_C4_ALL)
c4_p50 = pd.read_csv(PATH_C4_P50)
hypso_all = pd.read_csv(PATH_HYPSO_ALL)
hypso_summary = pd.read_csv(PATH_HYPSO_SUMMARY)

primary_hypso = hypso_all[
    hypso_all["reference"] == "primary_temporal_2014_median"
].copy()

quantile_rows = []

for scenario in SCENARIOS:
    for year in [2030, 2050, 2090]:
        for quantile in ["p05", "p50", "p95"]:
            c = c4_all[
                (c4_all["scenario"] == scenario)
                & (c4_all["year"] == year)
                & (c4_all["quantile"] == quantile)
            ]

            h = primary_hypso[
                (primary_hypso["scenario"] == scenario)
                & (primary_hypso["year"] == year)
                & (primary_hypso["quantile"] == quantile)
            ]

            if len(c) != 1 or len(h) != 1:
                raise RuntimeError(f"Fila invalida {scenario} {year} {quantile}.")

            c = c.iloc[0]
            h = h.iloc[0]

            quantile_rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "quantile": quantile,
                    "level_m": float(c["level_m"]),
                    "level_change_vs_2026_m": float(c["level_change_vs_anchor_m"]),
                    "area_bounds_status": str(c["bound_status"]),
                    "water_area_lower_bound_km2": float(c["water_area_lower_bound_km2"]) if pd.notna(c["water_area_lower_bound_km2"]) else np.nan,
                    "water_area_upper_bound_km2": float(c["water_area_upper_bound_km2"]) if pd.notna(c["water_area_upper_bound_km2"]) else np.nan,
                    "certain_exposed_area_km2": float(c["certain_exposed_area_km2"]),
                    "uncertain_area_km2": float(c["uncertain_area_km2"]),
                    "storage_loss_vs_2026_km3": float(h["storage_loss_vs_anchor_km3"]),
                    "volume_raw_km3": float(h["volume_raw_km3"]),
                    "mean_water_depth_raw_m": float(h["mean_water_depth_m"]),
                }
            )

quantiles_df = pd.DataFrame(quantile_rows)

final_rows = []

for scenario in SCENARIOS:
    for year in [2030, 2050, 2090]:
        q = quantiles_df[
            (quantiles_df["scenario"] == scenario)
            & (quantiles_df["year"] == year)
        ].set_index("quantile")

        p50 = q.loc["p50"]

        exp_row = exposure[
            (exposure["scenario"] == scenario)
            & (exposure["year"] == year)
        ] if year in PRIMARY_YEARS else pd.DataFrame()

        if year in PRIMARY_YEARS:
            if len(exp_row) != 1:
                raise RuntimeError(f"Resumen de exposicion invalido {scenario} {year}.")
            exp_row = exp_row.iloc[0]
            known_new_exposed_km2 = float(exp_row["new_exposed_total_area_km2"])
            robust_new_exposed_km2 = float(exp_row["new_exposed_robust_area_km2"])
            ensemble_sensitive_exposed_km2 = float(exp_row["new_exposed_ensemble_sensitive_area_km2"])
            unknown_coastal_km2 = float(exp_row["unknown_area_km2"])
        else:
            known_new_exposed_km2 = float(p50["certain_exposed_area_km2"])
            robust_new_exposed_km2 = np.nan
            ensemble_sensitive_exposed_km2 = np.nan
            unknown_coastal_km2 = coastal_unknown

        lower = float(p50["water_area_lower_bound_km2"])
        upper = float(p50["water_area_upper_bound_km2"])
        width = upper - lower

        final_rows.append(
            {
                "scenario": scenario,
                "year": year,
                "horizon_role": "PRIMARY" if year in PRIMARY_YEARS else "STRESS",
                "level_p05_m": float(q.loc["p05", "level_m"]),
                "level_p50_m": float(q.loc["p50", "level_m"]),
                "level_p95_m": float(q.loc["p95", "level_m"]),
                "level_p50_change_vs_2026_m": float(p50["level_change_vs_2026_m"]),
                "current_reference_area_km2": current_area_km2,
                "water_area_lower_bound_p50_km2": lower,
                "water_area_upper_bound_p50_km2": upper,
                "water_area_bound_width_p50_km2": width,
                "certain_exposed_area_p50_km2": float(p50["certain_exposed_area_km2"]),
                "uncertain_current_mask_area_p50_km2": float(p50["uncertain_area_km2"]),
                "known_new_exposed_coastal_area_p50_km2": known_new_exposed_km2,
                "robust_new_exposed_coastal_area_km2": robust_new_exposed_km2,
                "ensemble_sensitive_new_exposed_coastal_area_km2": ensemble_sensitive_exposed_km2,
                "unknown_coastal_domain_area_km2": unknown_coastal_km2,
                "storage_loss_p50_vs_2026_km3": float(p50["storage_loss_vs_2026_km3"]),
                "volume_raw_p50_km3": float(p50["volume_raw_km3"]),
                "area_point_estimate_status": "REJECTED",
                "area_reporting_status": "BOUNDS_ONLY",
                "storage_change_status": "PHYSICALLY_PLAUSIBLE_NOT_INDEPENDENTLY_VALIDATED",
                "spatial_shoreline_status": "PARTIAL_WITH_EXPLICIT_UNKNOWN",
            }
        )

final_df = pd.DataFrame(final_rows)

banner("QC 1 - RESUMEN FINAL")

print(
    final_df[
        [
            "scenario",
            "year",
            "horizon_role",
            "level_p50_m",
            "water_area_lower_bound_p50_km2",
            "water_area_upper_bound_p50_km2",
            "certain_exposed_area_p50_km2",
            "storage_loss_p50_vs_2026_km3",
        ]
    ].to_string(
        index=False,
        formatters={
            "level_p50_m": "{:.3f}".format,
            "water_area_lower_bound_p50_km2": "{:.3f}".format,
            "water_area_upper_bound_p50_km2": "{:.3f}".format,
            "certain_exposed_area_p50_km2": "{:.3f}".format,
            "storage_loss_p50_vs_2026_km3": "{:.3f}".format,
        },
    )
)

for _, row in final_df.iterrows():
    if row["water_area_lower_bound_p50_km2"] > row["water_area_upper_bound_p50_km2"]:
        raise RuntimeError("Limites de area invertidos.")

    if row["certain_exposed_area_p50_km2"] < -1e-12:
        raise RuntimeError("Area expuesta negativa.")

    if row["storage_loss_p50_vs_2026_km3"] < -1e-12:
        raise RuntimeError("Perdida de almacenamiento negativa.")

OUT_DIR.mkdir(parents=True, exist_ok=True)

final_df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")
quantiles_df.to_csv(OUT_QUANTILES, index=False, encoding="utf-8-sig")
c4_p50.to_csv(OUT_LIMITS, index=False, encoding="utf-8-sig")

banner("QC 2 - CONGELACION DE RASTERS 2030/2050")

copy_plan = [
    (
        PATH_CURRENT,
        OUT_DIR / "mascara_lago_actual_2024_2026.tif",
        "current_reference_lake",
    )
]

for scenario in SCENARIOS:
    for year in PRIMARY_YEARS:
        copy_plan.extend(
            [
                (
                    C4_DIR / f"16C4_{scenario}_{year}_p50_clases_area_acotada.tif",
                    OUT_DIR / f"area_acotada_{scenario}_{year}_p50.tif",
                    f"bounded_area_classes_{scenario}_{year}",
                ),
                (
                    COAST_DIR / f"exposicion_litoral_{scenario}_{year}.tif",
                    OUT_DIR / f"exposicion_litoral_{scenario}_{year}.tif",
                    f"coastal_exposure_{scenario}_{year}",
                ),
            ]
        )

for year in PRIMARY_YEARS:
    copy_plan.append(
        (
            COAST_DIR / f"consenso_exposicion_{year}.tif",
            OUT_DIR / f"consenso_exposicion_{year}.tif",
            f"coastal_exposure_consensus_{year}",
        )
    )

manifest_rows = []

for src, dst, role in copy_plan:
    h = copy_exact(src, dst)
    manifest_rows.append(
        {
            "kind": "raster",
            "role": role,
            "source_path": str(src.resolve()),
            "frozen_path": str(dst.resolve()),
            "sha256": h,
        }
    )
    print(f"[COPY] {dst.name} | identical=True")

for role, path in [
    ("final_summary", OUT_FINAL),
    ("quantile_summary", OUT_QUANTILES),
    ("bounded_area_p50_table", OUT_LIMITS),
]:
    manifest_rows.append(
        {
            "kind": "table",
            "role": role,
            "source_path": "",
            "frozen_path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    )

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111)

primary = final_df[final_df["year"].isin(PRIMARY_YEARS)].copy()
labels = [
    f"{r['scenario'].upper()} {int(r['year'])}"
    for _, r in primary.iterrows()
]
x = np.arange(len(primary))
lower = primary["water_area_lower_bound_p50_km2"].to_numpy()
upper = primary["water_area_upper_bound_p50_km2"].to_numpy()
mid = (lower + upper) / 2.0
err = np.vstack([mid - lower, upper - mid])

ax.errorbar(x, mid, yerr=err, fmt="o", capsize=6)
ax.axhline(current_area_km2, linestyle="--", label="Area actual 2026")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Area de agua (km2)")
ax.set_title("Area futura p50: limites compatibles con evidencia")
ax.grid(axis="y", alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_AREA, dpi=180, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111)

for scenario in SCENARIOS:
    sub = final_df[final_df["scenario"] == scenario].sort_values("year")
    ax.plot(
        sub["year"],
        sub["storage_loss_p50_vs_2026_km3"],
        marker="o",
        label=scenario.upper(),
    )

ax.set_xlabel("Horizonte")
ax.set_ylabel("Perdida de almacenamiento vs 2026 (km3)")
ax.set_title("Cambio de almacenamiento p50")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG_STORAGE, dpi=180, bbox_inches="tight")
plt.close(fig)

for role, path in [
    ("figure_area", OUT_FIG_AREA),
    ("figure_storage", OUT_FIG_STORAGE),
]:
    manifest_rows.append(
        {
            "kind": "figure",
            "role": role,
            "source_path": "",
            "frozen_path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")

metadata = {
    "step": "16C5",
    "status": "FROZEN_INTEGRATED_HYDROMORPHOLOGY",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "current_reference": {
        "area_km2": current_area_km2,
        "source": "Sentinel 2024-2026 50% modern-water reference mask",
    },
    "spatial_coast": {
        "status": "PARTIAL_WITH_EXPLICIT_UNKNOWN",
        "known_capability_pct": known_pct,
        "unknown_coastal_domain_area_km2": coastal_unknown,
        "authority": "resultados/costa_future_compatible",
    },
    "future_area": {
        "status": "BOUNDS_ONLY",
        "point_estimates": "REJECTED_BY_16C3",
        "source": "16C4 conservative certain-wet / certain-exposed partition",
        "primary_horizons": PRIMARY_YEARS,
        "stress_horizon": STRESS_YEAR,
    },
    "storage": {
        "status": "PHYSICALLY_PLAUSIBLE_NOT_INDEPENDENTLY_VALIDATED",
        "source": "frozen aggregate AMSCLAE hypsometry with temporal 2014 reference",
    },
    "future_levels": {
        "status": "CONDITIONAL_CMIP6_SCENARIO_TRAJECTORIES",
        "scenarios": SCENARIOS,
        "primary_horizons": PRIMARY_YEARS,
        "stress_horizon": STRESS_YEAR,
    },
    "reporting_rules": [
        "Do not report a single total future lake area; report lower and upper bounds.",
        "Do not draw a continuous future shoreline through unsupported pixels.",
        "Report certain exposed area separately from uncertain current-mask area.",
        "Storage change may be reported as physically plausible aggregate sensitivity, not independently validated bathymetric truth.",
        "Treat 2090 as a stress horizon with substantially greater uncertainty.",
    ],
    "outputs_sha256": {
        OUT_FINAL.name: sha256_file(OUT_FINAL),
        OUT_QUANTILES.name: sha256_file(OUT_QUANTILES),
        OUT_LIMITS.name: sha256_file(OUT_LIMITS),
        OUT_MANIFEST.name: sha256_file(OUT_MANIFEST),
        OUT_FIG_AREA.name: sha256_file(OUT_FIG_AREA),
        OUT_FIG_STORAGE.name: sha256_file(OUT_FIG_STORAGE),
    },
}

with open(OUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

banner("DECISION 16C5")

print("[FROZEN] hidromorfologia integrada congelada")
print("[AREA] solo limites inferior/superior; no punto unico")
print("[COSTA] parcial con desconocidos explicitos")
print("[VOLUMEN] cambio de almacenamiento retenido como plausible, no validado independientemente")
print("[HORIZONTES PRINCIPALES] 2030 y 2050")
print("[HORIZONTE DE ESTRES] 2090")
print("[STATUS] FROZEN_INTEGRATED_HYDROMORPHOLOGY")
print("[NEXT] usar esta rama congelada como base para productos de planificacion litoral y tul.")

banner("FIN PASO 16C5")
print("[OUTPUT DIRECTORY]")
print(OUT_DIR)
