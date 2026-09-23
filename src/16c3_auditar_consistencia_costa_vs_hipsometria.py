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
COAST_DIR = ROOT / "outputs" / "costa_future_compatible"
HYPSO_DIR = ROOT / "outputs" / "hipsometria_future_compatible"

PATH_CURRENT = COAST_DIR / "mascara_lago_actual_2024_2026.tif"
PATH_LEVELS = COAST_DIR / "niveles_ensemble_2030_2050.csv"
PATH_COAST_SUMMARY = COAST_DIR / "resumen_exposicion_2030_2050.csv"
PATH_HYPSO_SUMMARY = HYPSO_DIR / "resumen_horizontes_area_volumen.csv"
PATH_HYPSO_METADATA = HYPSO_DIR / "parametros_hipsometria_future_compatible.json"

OUT_DIR = ROOT / "outputs" / "consistencia_costa_hipsometria_16C3"
OUT_TABLE = OUT_DIR / "16C3_auditoria_consistencia_area_volumen.csv"
OUT_CLASSES = OUT_DIR / "16C3_resumen_clases_espaciales.csv"
OUT_METADATA = OUT_DIR / "16C3_metadata.json"
OUT_FIG = OUT_DIR / "16C3_comparacion_perdida_area.png"

SCENARIOS = ["ssp245", "ssp585"]
YEARS = [2030, 2050]

H_ANCHOR = 1552.770
CURRENT_AREA_EXPECTED_KM2 = 122.0232
PIXEL_AREA_EXPECTED_M2 = 400.0

AREA_TOL_KM2 = 5e-4
PHYSICAL_TOL_KM2 = 1e-6


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


for p in [
    PATH_CURRENT,
    PATH_LEVELS,
    PATH_COAST_SUMMARY,
    PATH_HYPSO_SUMMARY,
    PATH_HYPSO_METADATA,
]:
    if not p.exists():
        raise FileNotFoundError(p)

for scenario in SCENARIOS:
    for year in YEARS:
        p = COAST_DIR / f"exposicion_litoral_{scenario}_{year}.tif"
        if not p.exists():
            raise FileNotFoundError(p)

banner("PASO 16C3 - AUDITORIA CONSISTENCIA COSTA VS HIPSOMETRIA")

with open(PATH_HYPSO_METADATA, "r", encoding="utf-8") as f:
    hypso_metadata = json.load(f)

if hypso_metadata.get("status") != "FROZEN_HYPSOMETRY_FUTURE_COMPATIBLE":
    raise RuntimeError("La hipsometria frozen no tiene el status esperado.")

with rasterio.open(PATH_CURRENT) as ds:
    current = ds.read(1) > 0
    shape = ds.shape
    transform = ds.transform
    crs = ds.crs

pixel_area_m2 = abs(
    transform.a * transform.e
    - transform.b * transform.d
)

current_area_km2 = current.sum() * pixel_area_m2 / 1e6

if abs(pixel_area_m2 - PIXEL_AREA_EXPECTED_M2) > 1e-6:
    raise RuntimeError("La grilla maestra no tiene pixeles de 400 m2.")

if abs(current_area_km2 - CURRENT_AREA_EXPECTED_KM2) > AREA_TOL_KM2:
    raise RuntimeError("El area actual no reproduce 122.0232 km2.")

levels = pd.read_csv(PATH_LEVELS)
coast_summary = pd.read_csv(PATH_COAST_SUMMARY)
hypso_summary = pd.read_csv(PATH_HYPSO_SUMMARY)

rows = []
class_rows = []

banner("QC 1 - INTERSECCION EXPOSICION CON MASCARA ACTUAL")

for scenario in SCENARIOS:
    for year in YEARS:
        path = COAST_DIR / f"exposicion_litoral_{scenario}_{year}.tif"

        with rasterio.open(path) as ds:
            if ds.shape != shape or ds.transform != transform or ds.crs != crs:
                raise RuntimeError(f"Grilla incompatible: {path.name}")
            exposure = ds.read(1).astype(np.uint8)

        class_counts = {
            cls: int(np.sum(exposure == cls))
            for cls in range(0, 8)
        }

        class_current_counts = {
            cls: int(np.sum((exposure == cls) & current))
            for cls in range(0, 8)
        }

        new_total_mask = np.isin(exposure, [2, 3, 4, 5])
        new_current_mask = new_total_mask & current
        unknown_current_mask = (exposure == 7) & current
        remains_water_current_mask = (exposure == 1) & current
        already_exposed_current_mask = (exposure == 6) & current
        outside_domain_current_mask = (exposure == 0) & current

        new_total_area = new_total_mask.sum() * pixel_area_m2 / 1e6
        new_current_area = new_current_mask.sum() * pixel_area_m2 / 1e6
        unknown_current_area = unknown_current_mask.sum() * pixel_area_m2 / 1e6
        remains_water_current_area = remains_water_current_mask.sum() * pixel_area_m2 / 1e6
        already_exposed_current_area = already_exposed_current_mask.sum() * pixel_area_m2 / 1e6
        outside_domain_current_area = outside_domain_current_mask.sum() * pixel_area_m2 / 1e6

        coast_row = coast_summary[
            (coast_summary["scenario"] == scenario)
            & (coast_summary["year"] == year)
        ]

        if len(coast_row) != 1:
            raise RuntimeError(f"Resumen costa invalido {scenario} {year}.")

        coast_row = coast_row.iloc[0]
        reported_new_area = float(coast_row["new_exposed_total_area_km2"])

        if abs(new_total_area - reported_new_area) > AREA_TOL_KM2:
            raise RuntimeError(f"{scenario} {year}: raster y resumen 16B8 no coinciden.")

        level_row = levels[
            (levels["scenario"] == scenario)
            & (levels["year"] == year)
        ]

        if len(level_row) != 1:
            raise RuntimeError(f"Niveles invalidos {scenario} {year}.")

        level_row = level_row.iloc[0]
        level_p50 = float(level_row["level_p50_m"])
        drawdown_m = H_ANCHOR - level_p50

        hypso_row = hypso_summary[
            (hypso_summary["scenario"] == scenario)
            & (hypso_summary["year"] == year)
        ]

        if len(hypso_row) != 1:
            raise RuntimeError(f"Resumen hipsometrico invalido {scenario} {year}.")

        hypso_row = hypso_row.iloc[0]

        area_hypso = float(hypso_row["area_anchor_adjusted_p50_km2"])
        area_loss_hypso = float(hypso_row["area_loss_p50_vs_current_km2"])
        storage_loss_hypso = float(hypso_row["storage_loss_p50_vs_anchor_km3"])

        hard_minimum_area_loss = new_current_area
        future_area_upper_bound = current_area_km2 - hard_minimum_area_loss
        contradiction_km2 = hard_minimum_area_loss - area_loss_hypso
        area_excess_vs_upper_bound = area_hypso - future_area_upper_bound
        contradiction = contradiction_km2 > PHYSICAL_TOL_KM2

        hydraulic_upper_storage_loss = (
            current_area_km2
            * drawdown_m
            / 1000.0
        )

        equivalent_mean_area = (
            storage_loss_hypso
            * 1000.0
            / drawdown_m
            if drawdown_m > 0
            else np.nan
        )

        storage_physically_plausible = (
            storage_loss_hypso >= -1e-12
            and storage_loss_hypso <= hydraulic_upper_storage_loss + 1e-6
            and equivalent_mean_area <= current_area_km2 + 1e-6
        )

        rows.append(
            {
                "scenario": scenario,
                "year": year,
                "level_p50_m": level_p50,
                "drawdown_vs_2026_m": drawdown_m,
                "current_area_km2": current_area_km2,
                "spatial_new_exposed_total_km2": new_total_area,
                "spatial_new_exposed_intersect_current_km2": new_current_area,
                "spatial_unknown_intersect_current_km2": unknown_current_area,
                "spatial_remains_water_intersect_current_km2": remains_water_current_area,
                "spatial_already_exposed_intersect_current_km2": already_exposed_current_area,
                "spatial_outside_domain_intersect_current_km2": outside_domain_current_area,
                "hard_minimum_area_loss_km2": hard_minimum_area_loss,
                "future_area_upper_bound_km2": future_area_upper_bound,
                "hypso_area_p50_km2": area_hypso,
                "hypso_area_loss_p50_km2": area_loss_hypso,
                "hypso_area_excess_vs_spatial_upper_bound_km2": area_excess_vs_upper_bound,
                "area_loss_shortfall_vs_known_spatial_exposure_km2": contradiction_km2,
                "area_consistency_status": "CONTRADICTION" if contradiction else "NOT_CONTRADICTED",
                "hypso_storage_loss_p50_km3": storage_loss_hypso,
                "hydraulic_storage_loss_upper_bound_km3": hydraulic_upper_storage_loss,
                "equivalent_mean_area_from_storage_loss_km2": equivalent_mean_area,
                "storage_physical_status": "PLAUSIBLE" if storage_physically_plausible else "REJECT",
            }
        )

        for cls in range(0, 8):
            class_rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "class": cls,
                    "pixels_total": class_counts[cls],
                    "area_total_km2": class_counts[cls] * pixel_area_m2 / 1e6,
                    "pixels_intersect_current": class_current_counts[cls],
                    "area_intersect_current_km2": class_current_counts[cls] * pixel_area_m2 / 1e6,
                }
            )

        print()
        print(f"[{scenario.upper()} {year}]")
        print(f"  nivel p50 = {level_p50:.3f} m | descenso = {drawdown_m:.3f} m")
        print(f"  exposicion espacial total = {new_total_area:.4f} km2")
        print(f"  exposicion espacial dentro mascara actual = {new_current_area:.4f} km2")
        print(f"  perdida de area 16C2 = {area_loss_hypso:.4f} km2")
        print(f"  area futura maxima permitida por exposicion conocida = {future_area_upper_bound:.4f} km2")
        print(f"  area futura 16C2 = {area_hypso:.4f} km2")
        print(f"  contradiccion = {contradiction_km2:+.4f} km2")
        print(f"  almacenamiento 16C2 = {storage_loss_hypso:.5f} km3")
        print(f"  area media equivalente = {equivalent_mean_area:.3f} km2")
        print(f"  estado area = {'CONTRADICTION' if contradiction else 'NOT_CONTRADICTED'}")
        print(f"  estado volumen = {'PLAUSIBLE' if storage_physically_plausible else 'REJECT'}")

audit_df = pd.DataFrame(rows)
classes_df = pd.DataFrame(class_rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)
audit_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
classes_df.to_csv(OUT_CLASSES, index=False, encoding="utf-8-sig")

banner("QC 2 - DECISION GLOBAL")

n_area_contradictions = int(np.sum(audit_df["area_consistency_status"] == "CONTRADICTION"))
n_storage_rejects = int(np.sum(audit_df["storage_physical_status"] == "REJECT"))

if n_area_contradictions > 0:
    area_decision = "REJECT_C2_AREA_POINT_ESTIMATES"
else:
    area_decision = "AREA_POINT_ESTIMATES_NOT_CONTRADICTED"

if n_storage_rejects == 0:
    volume_decision = "RETAIN_STORAGE_CHANGE_AS_PLAUSIBLE"
else:
    volume_decision = "REJECT_STORAGE_CHANGE"

print(
    audit_df[
        [
            "scenario",
            "year",
            "hard_minimum_area_loss_km2",
            "hypso_area_loss_p50_km2",
            "future_area_upper_bound_km2",
            "hypso_area_p50_km2",
            "area_consistency_status",
            "hypso_storage_loss_p50_km3",
            "storage_physical_status",
        ]
    ].to_string(
        index=False,
        formatters={
            "hard_minimum_area_loss_km2": "{:.4f}".format,
            "hypso_area_loss_p50_km2": "{:.4f}".format,
            "future_area_upper_bound_km2": "{:.4f}".format,
            "hypso_area_p50_km2": "{:.4f}".format,
            "hypso_storage_loss_p50_km3": "{:.5f}".format,
        },
    )
)

print()
print(f"[AREA DECISION] {area_decision}")
print(f"[VOLUME DECISION] {volume_decision}")

labels = [
    f"{r['scenario'].upper()} {int(r['year'])}"
    for _, r in audit_df.iterrows()
]

x = np.arange(len(audit_df))
width = 0.36

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111)
ax.bar(
    x - width / 2,
    audit_df["hard_minimum_area_loss_km2"].to_numpy(),
    width,
    label="Perdida minima observada espacialmente",
)
ax.bar(
    x + width / 2,
    audit_df["hypso_area_loss_p50_km2"].to_numpy(),
    width,
    label="Perdida total estimada por 16C2",
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Perdida de area desde 2026 (km2)")
ax.set_title("16C3 - Consistencia entre costa observada e hipsometria agregada")
ax.grid(axis="y", alpha=0.25)
ax.legend()
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
plt.close(fig)

metadata = {
    "step": "16C3",
    "status": "COAST_HYPSOMETRY_CONSISTENCY_AUDIT_COMPLETE",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "area_decision": area_decision,
    "volume_decision": volume_decision,
    "n_area_contradictions": n_area_contradictions,
    "n_storage_rejects": n_storage_rejects,
    "current_area_km2": current_area_km2,
    "logic": {
        "hard_minimum_area_loss": "newly exposed p50 pixels from the frozen coastal product intersected with the frozen current-water mask",
        "future_area_upper_bound": "current observed area minus hard minimum known exposure",
        "area_contradiction": "aggregate hypsometric loss is smaller than spatially observed-supported loss inside the current-water mask",
        "storage_check": "storage loss must be nonnegative and no larger than current surface area times drawdown",
    },
    "interpretation": {
        "area": "If contradiction is detected, the C2 aggregate area point estimate cannot be used as total future lake area.",
        "volume": "A plausible storage change is not automatically independently validated; it may be retained only as aggregate storage sensitivity until further validation.",
        "spatial_coast": "Frozen coastal products remain authoritative for supported spatial exposure.",
    },
    "outputs_sha256": {
        OUT_TABLE.name: sha256_file(OUT_TABLE),
        OUT_CLASSES.name: sha256_file(OUT_CLASSES),
        OUT_FIG.name: sha256_file(OUT_FIG),
    },
}

with open(OUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

banner("FIN PASO 16C3")
print("[OUTPUT DIRECTORY]")
print(OUT_DIR)
