__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
AMS_PATH = ROOT / "data" / "external" / "batimetria_amsclae.tif"

COAST_DIR = ROOT / "outputs" / "costa_future_compatible"
HYDRO_DIR = ROOT / "outputs" / "hidromorfologia_future_compatible"
LEVEL_DIR = ROOT / "outputs" / "nivel_future_compatible"
C4_DIR = ROOT / "outputs" / "area_futura_acotada_16C4"

PATH_CURRENT = COAST_DIR / "mascara_lago_actual_2024_2026.tif"
PATH_CAP = COAST_DIR / "capacidad_espacial_costera.tif"
PATH_THRESHOLD = COAST_DIR / "nivel_umbral_costero_m.tif"
PATH_PROVENANCE = COAST_DIR / "procedencia_umbral_costero.tif"
PATH_DOMAIN = COAST_DIR / "dominio_costero_evaluado.tif"
PATH_FUTURE = LEVEL_DIR / "nivel_diario_cmip6_future_compatible.pkl"
PATH_FROZEN_QUANTILES = HYDRO_DIR / "resumen_hidromorfologico_cuantiles.csv"
PATH_FROZEN_FINAL = HYDRO_DIR / "resumen_hidromorfologico_final.csv"
PATH_C4_P50 = C4_DIR / "16C4_resumen_p50_area_volumen.csv"

OUT_DIR = ROOT / "outputs" / "productos_qgis_anuales_final_18D"
RASTER_DIR = OUT_DIR / "rasters_anuales"
VECTOR_DIR = OUT_DIR / "vectores_anuales"
STACK_DIR = OUT_DIR / "stacks_multibanda"
STYLE_DIR = OUT_DIR / "estilos_qml"
TABLE_DIR = OUT_DIR / "tablas"

START_YEAR = 2027
END_YEAR = 2050
SCENARIOS = ["ssp245", "ssp585"]

H_ANCHOR = 1552.770
H2014_MIN = 1557.310
H2014_MAX = 1558.090
INTERIOR_DISTANCE_M = 120.0
FLOAT_NODATA = -9999.0


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


def normalize_scenario(s):
    x = str(s).lower().replace("_", "").replace("-", "")
    if "245" in x:
        return "ssp245"
    if "585" in x:
        return "ssp585"
    return x


def write_single(path, arr, profile, dtype, nodata, description):
    p = profile.copy()
    p.update(dtype=dtype, count=1, nodata=nodata, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)
        dst.set_band_description(1, description)


def write_stack(path, arrays, profile, dtype, nodata, descriptions):
    p = profile.copy()
    p.update(dtype=dtype, count=len(arrays), nodata=nodata, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr.astype(dtype), i)
            dst.set_band_description(i, descriptions[i - 1])


def write_geojson(path, arr, class_labels, transform, crs, properties):
    feats = []
    mask = arr > 0
    for geom, value in shapes(arr.astype(np.int16), mask=mask, transform=transform):
        cls = int(value)
        props = dict(properties)
        props["class_id"] = cls
        props["class_label"] = class_labels.get(cls, "unknown")
        feats.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": geom,
            }
        )
    data = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": str(crs)},
        },
        "features": feats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def qml_state():
    return """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1">
      <colorPalette>
        <paletteEntry value="0" color="#ffffff" alpha="0" label="Fuera"/>
        <paletteEntry value="1" color="#2c7fb8" alpha="255" label="Agua cierta - evidencia costera"/>
        <paletteEntry value="2" color="#f03b20" alpha="255" label="Expuesto cierto - evidencia costera"/>
        <paletteEntry value="3" color="#225ea8" alpha="255" label="Agua cierta - interior AMSCLAE"/>
        <paletteEntry value="4" color="#bd0026" alpha="255" label="Expuesto cierto - interior AMSCLAE"/>
        <paletteEntry value="5" color="#bdbdbd" alpha="255" label="Incierto / sin soporte suficiente"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def qml_robust():
    return """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1">
      <colorPalette>
        <paletteEntry value="0" color="#ffffff" alpha="0" label="Fuera"/>
        <paletteEntry value="1" color="#2c7fb8" alpha="255" label="Agua robusta p05-p95"/>
        <paletteEntry value="2" color="#fdae61" alpha="255" label="Sensible al ensemble o al intervalo vertical"/>
        <paletteEntry value="3" color="#d73027" alpha="255" label="Exposicion robusta p05-p95"/>
        <paletteEntry value="4" color="#bdbdbd" alpha="255" label="Incertidumbre espacial"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def qml_consensus():
    return """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1">
      <colorPalette>
        <paletteEntry value="0" color="#ffffff" alpha="0" label="Fuera"/>
        <paletteEntry value="1" color="#2c7fb8" alpha="255" label="Agua cierta en ambos escenarios"/>
        <paletteEntry value="2" color="#d73027" alpha="255" label="Expuesto cierto en ambos escenarios"/>
        <paletteEntry value="3" color="#fdae61" alpha="255" label="Diferencia entre escenarios"/>
        <paletteEntry value="4" color="#bdbdbd" alpha="255" label="Incierto en uno o ambos escenarios"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def classify_level(H, current, cap, threshold, threshold_valid, robust_interior, zlow, zhigh):
    out = np.zeros(current.shape, dtype=np.uint8)

    known = current & np.isin(cap, [1, 2]) & threshold_valid
    anchor_supported = known & (H_ANCHOR >= threshold)
    known_wet = anchor_supported & (H >= threshold)
    known_exposed = anchor_supported & (H < threshold)

    class0 = current & (cap == 0)
    interior = class0 & robust_interior
    interior_wet = interior & (H >= zhigh)
    interior_exposed = interior & (H < zlow)

    out[known_wet] = 1
    out[known_exposed] = 2
    out[interior_wet] = 3
    out[interior_exposed] = 4
    out[current & (out == 0)] = 5

    return out


def robust_classification(H05, H95, current, cap, threshold, threshold_valid, robust_interior, zlow, zhigh):
    low = classify_level(H05, current, cap, threshold, threshold_valid, robust_interior, zlow, zhigh)
    high = classify_level(H95, current, cap, threshold, threshold_valid, robust_interior, zlow, zhigh)

    low_wet = np.isin(low, [1, 3])
    high_wet = np.isin(high, [1, 3])
    low_exp = np.isin(low, [2, 4])
    high_exp = np.isin(high, [2, 4])
    low_unc = low == 5
    high_unc = high == 5

    out = np.zeros(current.shape, dtype=np.uint8)
    out[current & low_wet] = 1
    out[current & high_exp] = 3
    out[current & low_unc & high_unc] = 4

    remaining = current & (out == 0)
    supported_transition = remaining & (
        low_wet | high_wet | low_exp | high_exp
    )
    out[supported_transition] = 2
    out[current & (out == 0)] = 4

    return out


required = [
    AMS_PATH,
    PATH_CURRENT,
    PATH_CAP,
    PATH_THRESHOLD,
    PATH_PROVENANCE,
    PATH_DOMAIN,
    PATH_FUTURE,
    PATH_FROZEN_QUANTILES,
    PATH_FROZEN_FINAL,
    PATH_C4_P50,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(p)

for p in [RASTER_DIR, VECTOR_DIR, STACK_DIR, STYLE_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

banner("PASO 18D - PRODUCTOS ANUALES QGIS VALIDADOS 2027-2050")

with rasterio.open(PATH_CURRENT) as ds:
    current = ds.read(1) > 0
    profile = ds.profile.copy()
    shape = ds.shape
    transform = ds.transform
    crs = ds.crs

with rasterio.open(PATH_CAP) as ds:
    if ds.shape != shape or ds.transform != transform or ds.crs != crs:
        raise RuntimeError("Capacidad espacial incompatible.")
    cap = ds.read(1)

with rasterio.open(PATH_THRESHOLD) as ds:
    if ds.shape != shape or ds.transform != transform or ds.crs != crs:
        raise RuntimeError("Umbral costero incompatible.")
    threshold = ds.read(1).astype(np.float64)
    threshold_nodata = ds.nodata

threshold_valid = np.isfinite(threshold)
if threshold_nodata is not None:
    threshold_valid &= threshold != threshold_nodata

known = np.isin(cap, [1, 2])

if np.any(current & known & (~threshold_valid)):
    raise RuntimeError("Hay pixeles costeros conocidos sin umbral.")

with rasterio.open(AMS_PATH) as src:
    depth_src = src.read(1).astype(np.float32)
    src_nodata = src.nodata

    depth_dst = np.full(shape, np.nan, dtype=np.float32)

    reproject(
        source=depth_src,
        destination=depth_dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src_nodata,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    valid_src = np.isfinite(depth_src)
    if src_nodata is not None:
        valid_src &= depth_src != src_nodata

    valid_dst = np.zeros(shape, dtype=np.uint8)

    reproject(
        source=valid_src.astype(np.uint8),
        destination=valid_dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=0,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

ams_valid = (valid_dst > 0) & np.isfinite(depth_dst)
zlow = H2014_MIN + depth_dst
zhigh = H2014_MAX + depth_dst

distance_inside = distance_transform_edt(
    current,
    sampling=(abs(transform.e), abs(transform.a)),
)

robust_interior = (
    current
    & (distance_inside >= INTERIOR_DISTANCE_M)
    & ams_valid
    & (H_ANCHOR >= zhigh)
)

pixel_area_km2 = abs(
    transform.a * transform.e
    - transform.b * transform.d
) / 1e6

current_area_km2 = current.sum() * pixel_area_km2

banner("QC 1 - BASE ESPACIAL 16C4")

print(f"[CURRENT AREA] = {current_area_km2:.4f} km2")
print(f"[AMS VALID CURRENT] = {int((current & ams_valid).sum()):,} pix")
print(f"[INTERIOR >= {INTERIOR_DISTANCE_M:.0f} m] = {int((current & (distance_inside >= INTERIOR_DISTANCE_M)).sum()):,} pix")
print(f"[INTERIOR ROBUST WET AT ANCHOR] = {int(robust_interior.sum()):,} pix")

future = pd.read_pickle(PATH_FUTURE).copy()
future["date"] = pd.to_datetime(future["date"], errors="coerce")
future["scenario_norm"] = future["scenario"].map(normalize_scenario)
future["year"] = future["date"].dt.year
future["level_m"] = pd.to_numeric(future["level_m"], errors="coerce")

future = future[
    future["date"].notna()
    & future["level_m"].notna()
    & future["scenario_norm"].isin(SCENARIOS)
    & future["year"].between(START_YEAR, END_YEAR)
].copy()

model_year = (
    future.groupby(["scenario_norm", "model", "year"], as_index=False)["level_m"]
    .mean()
    .rename(columns={"level_m": "annual_mean_level_m"})
)

rows = []

for (scenario, year), sub in model_year.groupby(["scenario_norm", "year"]):
    vals = sub["annual_mean_level_m"].to_numpy(dtype=float)
    rows.append(
        {
            "scenario": scenario,
            "year": int(year),
            "level_p05_m": float(np.quantile(vals, 0.05)),
            "level_p50_m": float(np.quantile(vals, 0.50)),
            "level_p95_m": float(np.quantile(vals, 0.95)),
            "n_models": int(len(vals)),
            "level_source": "annual_mean_by_model_then_ensemble_quantile",
        }
    )

levels = pd.DataFrame(rows).sort_values(["scenario", "year"]).reset_index(drop=True)

frozen_q = pd.read_csv(PATH_FROZEN_QUANTILES)

for scenario in SCENARIOS:
    for year in [2030, 2050]:
        fq = frozen_q[
            (frozen_q["scenario"] == scenario)
            & (frozen_q["year"] == year)
        ]

        if len(fq) != 3:
            raise RuntimeError(f"Cuantiles frozen invalidos: {scenario} {year}")

        for quantile, col in [
            ("p05", "level_p05_m"),
            ("p50", "level_p50_m"),
            ("p95", "level_p95_m"),
        ]:
            value = float(
                fq.loc[fq["quantile"] == quantile, "level_m"].iloc[0]
            )
            mask = (
                (levels["scenario"] == scenario)
                & (levels["year"] == year)
            )
            levels.loc[mask, col] = value
            levels.loc[mask, "level_source"] = "frozen_key_horizon"

levels["delta_p50_vs_2026_m"] = levels["level_p50_m"] - H_ANCHOR
levels["uncertainty_width_p95_p05_m"] = levels["level_p95_m"] - levels["level_p05_m"]

levels.to_csv(
    TABLE_DIR / "18D_niveles_anuales_ensemble_2027_2050.csv",
    index=False,
    encoding="utf-8-sig",
)

banner("QC 2 - NIVELES ANUALES")

print(
    levels[
        [
            "scenario",
            "year",
            "level_p05_m",
            "level_p50_m",
            "level_p95_m",
            "uncertainty_width_p95_p05_m",
            "level_source",
        ]
    ].to_string(
        index=False,
        formatters={
            "level_p05_m": "{:.3f}".format,
            "level_p50_m": "{:.3f}".format,
            "level_p95_m": "{:.3f}".format,
            "uncertainty_width_p95_p05_m": "{:.3f}".format,
        },
    )
)

(STYLE_DIR / "18D_estado_p50.qml").write_text(qml_state(), encoding="utf-8")
(STYLE_DIR / "18D_robustez_p05_p95.qml").write_text(qml_robust(), encoding="utf-8")
(STYLE_DIR / "18D_consenso_escenarios.qml").write_text(qml_consensus(), encoding="utf-8")

annual_rows = []
state_cache = {}
state_stacks = {s: [] for s in SCENARIOS}
robust_stacks = {s: [] for s in SCENARIOS}
level_stacks = {s: [] for s in SCENARIOS}
delta_stacks = {s: [] for s in SCENARIOS}
uncertainty_stacks = {s: [] for s in SCENARIOS}

descriptions = [str(y) for y in range(START_YEAR, END_YEAR + 1)]

for scenario in SCENARIOS:
    banner(f"ESCENARIO {scenario.upper()}")

    for year in range(START_YEAR, END_YEAR + 1):
        r = levels[
            (levels["scenario"] == scenario)
            & (levels["year"] == year)
        ]

        if len(r) != 1:
            raise RuntimeError(f"Nivel anual faltante: {scenario} {year}")

        r = r.iloc[0]
        H05 = float(r["level_p05_m"])
        H50 = float(r["level_p50_m"])
        H95 = float(r["level_p95_m"])

        state = classify_level(
            H50,
            current,
            cap,
            threshold,
            threshold_valid,
            robust_interior,
            zlow,
            zhigh,
        )

        robust = robust_classification(
            H05,
            H95,
            current,
            cap,
            threshold,
            threshold_valid,
            robust_interior,
            zlow,
            zhigh,
        )

        certain_wet = np.isin(state, [1, 3])
        certain_exposed = np.isin(state, [2, 4])
        uncertain = state == 5

        n_wet = int(certain_wet.sum())
        n_exp = int(certain_exposed.sum())
        n_unc = int(uncertain.sum())

        if n_wet + n_exp + n_unc != int(current.sum()):
            raise RuntimeError(f"Particion invalida: {scenario} {year}")

        water_lower = n_wet * pixel_area_km2
        water_upper = (current.sum() - n_exp) * pixel_area_km2
        exposed_area = n_exp * pixel_area_km2
        uncertain_area = n_unc * pixel_area_km2

        direct_exposed = int(np.sum((state == 2) & (cap == 1))) * pixel_area_km2
        ams_coastal_exposed = int(np.sum((state == 2) & (cap == 2))) * pixel_area_km2
        interior_exposed = int(np.sum(state == 4)) * pixel_area_km2

        robust_wet = int(np.sum(robust == 1)) * pixel_area_km2
        sensitive = int(np.sum(robust == 2)) * pixel_area_km2
        robust_exposed = int(np.sum(robust == 3)) * pixel_area_km2
        spatial_unknown = int(np.sum(robust == 4)) * pixel_area_km2

        scenario_raster_dir = RASTER_DIR / scenario
        scenario_vector_dir = VECTOR_DIR / scenario

        state_path = scenario_raster_dir / f"18D_{scenario}_{year}_estado_p50.tif"
        robust_path = scenario_raster_dir / f"18D_{scenario}_{year}_robustez_p05_p95.tif"
        level_path = scenario_raster_dir / f"18D_{scenario}_{year}_nivel_p50_m.tif"
        delta_path = scenario_raster_dir / f"18D_{scenario}_{year}_delta_nivel_vs_2026_m.tif"
        unc_path = scenario_raster_dir / f"18D_{scenario}_{year}_ancho_incertidumbre_m.tif"

        write_single(
            state_path,
            state,
            profile,
            "uint8",
            0,
            f"estado_p50_{scenario}_{year}",
        )

        write_single(
            robust_path,
            robust,
            profile,
            "uint8",
            0,
            f"robustez_p05_p95_{scenario}_{year}",
        )

        level_arr = np.full(shape, FLOAT_NODATA, dtype=np.float32)
        delta_arr = np.full(shape, FLOAT_NODATA, dtype=np.float32)
        uncertainty_arr = np.full(shape, FLOAT_NODATA, dtype=np.float32)

        level_arr[current] = H50
        delta_arr[current] = H50 - H_ANCHOR
        uncertainty_arr[current] = H95 - H05

        write_single(
            level_path,
            level_arr,
            profile,
            "float32",
            FLOAT_NODATA,
            f"nivel_p50_m_{scenario}_{year}",
        )

        write_single(
            delta_path,
            delta_arr,
            profile,
            "float32",
            FLOAT_NODATA,
            f"delta_nivel_vs_2026_m_{scenario}_{year}",
        )

        write_single(
            unc_path,
            uncertainty_arr,
            profile,
            "float32",
            FLOAT_NODATA,
            f"ancho_incertidumbre_m_{scenario}_{year}",
        )

        write_geojson(
            scenario_vector_dir / f"18D_{scenario}_{year}_estado_p50.geojson",
            state,
            {
                1: "agua_cierta_costera",
                2: "expuesto_cierto_costero",
                3: "agua_cierta_interior",
                4: "expuesto_cierto_interior",
                5: "incierto",
            },
            transform,
            crs,
            {
                "scenario": scenario,
                "year": int(year),
                "level_p50_m": H50,
                "delta_vs_2026_m": H50 - H_ANCHOR,
            },
        )

        annual_rows.append(
            {
                "scenario": scenario,
                "year": int(year),
                "level_p05_m": H05,
                "level_p50_m": H50,
                "level_p95_m": H95,
                "delta_p50_vs_2026_m": H50 - H_ANCHOR,
                "uncertainty_width_p95_p05_m": H95 - H05,
                "current_reference_area_km2": current_area_km2,
                "water_area_lower_bound_p50_km2": water_lower,
                "water_area_upper_bound_p50_km2": water_upper,
                "certain_exposed_area_p50_km2": exposed_area,
                "uncertain_area_p50_km2": uncertain_area,
                "direct_coastal_exposed_p50_km2": direct_exposed,
                "ams_coastal_exposed_p50_km2": ams_coastal_exposed,
                "interior_ams_exposed_p50_km2": interior_exposed,
                "robust_wet_area_km2": robust_wet,
                "ensemble_or_vertical_sensitive_area_km2": sensitive,
                "robust_exposed_area_km2": robust_exposed,
                "spatially_uncertain_area_km2": spatial_unknown,
                "level_source": r["level_source"],
            }
        )

        state_cache[(scenario, year)] = state
        state_stacks[scenario].append(state)
        robust_stacks[scenario].append(robust)
        level_stacks[scenario].append(level_arr)
        delta_stacks[scenario].append(delta_arr)
        uncertainty_stacks[scenario].append(uncertainty_arr)

        print(
            f"[{scenario.upper()} {year}] "
            f"H={H50:.3f} | "
            f"area={water_lower:.4f}-{water_upper:.4f} km2 | "
            f"expuesto cierto={exposed_area:.4f} | "
            f"incierto={uncertain_area:.4f}"
        )

annual = pd.DataFrame(annual_rows)
annual.to_csv(
    TABLE_DIR / "18D_resumen_hidromorfologico_anual_2027_2050.csv",
    index=False,
    encoding="utf-8-sig",
)

banner("QC 3 - REPRODUCCION 16C4 EN 2030/2050")

c4 = pd.read_csv(PATH_C4_P50)

for scenario in SCENARIOS:
    for year in [2030, 2050]:
        a = annual[
            (annual["scenario"] == scenario)
            & (annual["year"] == year)
        ].iloc[0]

        b = c4[
            (c4["scenario"] == scenario)
            & (c4["year"] == year)
        ].iloc[0]

        dif_low = abs(
            float(a["water_area_lower_bound_p50_km2"])
            - float(b["water_area_lower_bound_km2"])
        )
        dif_up = abs(
            float(a["water_area_upper_bound_p50_km2"])
            - float(b["water_area_upper_bound_km2"])
        )
        dif_exp = abs(
            float(a["certain_exposed_area_p50_km2"])
            - float(b["certain_exposed_area_km2"])
        )

        ok = max(dif_low, dif_up, dif_exp) <= 0.0005

        print(
            f"[{scenario.upper()} {year}] "
            f"dLow={dif_low:.4f} | dUp={dif_up:.4f} | dExp={dif_exp:.4f} | identical={ok}"
        )

        if not ok:
            raise RuntimeError(
                f"18D no reproduce 16C4 en {scenario} {year}."
            )

banner("QC 4 - CONSENSO ANUAL ENTRE ESCENARIOS")

consensus_rows = []
consensus_stack = []

for year in range(START_YEAR, END_YEAR + 1):
    a = state_cache[("ssp245", year)]
    b = state_cache[("ssp585", year)]

    a_wet = np.isin(a, [1, 3])
    b_wet = np.isin(b, [1, 3])
    a_exp = np.isin(a, [2, 4])
    b_exp = np.isin(b, [2, 4])
    unknown = (a == 5) | (b == 5)

    consensus = np.zeros(shape, dtype=np.uint8)
    consensus[current & (~unknown) & a_wet & b_wet] = 1
    consensus[current & (~unknown) & a_exp & b_exp] = 2
    consensus[current & (~unknown) & ((a_exp & b_wet) | (a_wet & b_exp))] = 3
    consensus[current & unknown] = 4

    consensus_stack.append(consensus)

    out = RASTER_DIR / "consenso" / f"18D_consenso_ssp245_ssp585_{year}.tif"

    write_single(
        out,
        consensus,
        profile,
        "uint8",
        0,
        f"consenso_ssp245_ssp585_{year}",
    )

    write_geojson(
        VECTOR_DIR / "consenso" / f"18D_consenso_ssp245_ssp585_{year}.geojson",
        consensus,
        {
            1: "agua_cierta_ambos",
            2: "expuesto_cierto_ambos",
            3: "diferencia_entre_escenarios",
            4: "incierto",
        },
        transform,
        crs,
        {"year": int(year)},
    )

    consensus_rows.append(
        {
            "year": int(year),
            "water_both_km2": int(np.sum(consensus == 1)) * pixel_area_km2,
            "exposed_both_km2": int(np.sum(consensus == 2)) * pixel_area_km2,
            "scenario_difference_km2": int(np.sum(consensus == 3)) * pixel_area_km2,
            "uncertain_km2": int(np.sum(consensus == 4)) * pixel_area_km2,
        }
    )

consensus_df = pd.DataFrame(consensus_rows)
consensus_df.to_csv(
    TABLE_DIR / "18D_consenso_anual_ssp245_ssp585_2027_2050.csv",
    index=False,
    encoding="utf-8-sig",
)

for scenario in SCENARIOS:
    write_stack(
        STACK_DIR / scenario / f"18D_{scenario}_estado_p50_2027_2050.tif",
        state_stacks[scenario],
        profile,
        "uint8",
        0,
        descriptions,
    )

    write_stack(
        STACK_DIR / scenario / f"18D_{scenario}_robustez_p05_p95_2027_2050.tif",
        robust_stacks[scenario],
        profile,
        "uint8",
        0,
        descriptions,
    )

    write_stack(
        STACK_DIR / scenario / f"18D_{scenario}_nivel_p50_m_2027_2050.tif",
        level_stacks[scenario],
        profile,
        "float32",
        FLOAT_NODATA,
        descriptions,
    )

    write_stack(
        STACK_DIR / scenario / f"18D_{scenario}_delta_nivel_vs_2026_m_2027_2050.tif",
        delta_stacks[scenario],
        profile,
        "float32",
        FLOAT_NODATA,
        descriptions,
    )

    write_stack(
        STACK_DIR / scenario / f"18D_{scenario}_ancho_incertidumbre_m_2027_2050.tif",
        uncertainty_stacks[scenario],
        profile,
        "float32",
        FLOAT_NODATA,
        descriptions,
    )

write_stack(
    STACK_DIR / "consenso" / "18D_consenso_ssp245_ssp585_2027_2050.tif",
    consensus_stack,
    profile,
    "uint8",
    0,
    descriptions,
)

readme = f"""PAQUETE QGIS ANUAL FINAL 18D

Periodo principal: {START_YEAR}-{END_YEAR}
Escenarios: SSP245 y SSP585
Ancla 2026: {H_ANCHOR:.3f} m
CRS: {crs}

RASTERS ANUALES POR ESCENARIO
estado_p50:
1 agua cierta por evidencia costera
2 expuesto cierto por evidencia costera
3 agua cierta por interior AMSCLAE conservador
4 expuesto cierto por interior AMSCLAE conservador
5 incierto o sin soporte espacial suficiente

robustez_p05_p95:
1 agua robusta
2 sensible al ensemble o al intervalo vertical
3 exposicion robusta
4 incertidumbre espacial

consenso:
1 agua cierta en ambos escenarios
2 expuesto cierto en ambos escenarios
3 diferencia entre SSP245 y SSP585
4 incierto en uno o ambos escenarios

Los GeoTIFF nivel_p50_m contienen el nivel anual p50 dentro de la mascara actual.
Los GeoTIFF delta_nivel_vs_2026_m contienen el cambio respecto a 1552.770 m.
Los GeoTIFF ancho_incertidumbre_m contienen p95-p05.
Los stacks multibanda tienen una banda por ano, de {START_YEAR} a {END_YEAR}.

El area futura total no debe representarse con un unico valor.
Usar water_area_lower_bound_p50_km2 y water_area_upper_bound_p50_km2.
Las zonas inciertas no deben convertirse en una costa continua interpolada.
"""

(OUT_DIR / "LEEME_QGIS_18D.txt").write_text(readme, encoding="utf-8")

metadata = {
    "step": "18D",
    "status": "ANNUAL_QGIS_FINAL_VALIDATED",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "years": [START_YEAR, END_YEAR],
    "scenarios": SCENARIOS,
    "anchor_level_m": H_ANCHOR,
    "master_crs": str(crs),
    "current_reference_area_km2": current_area_km2,
    "interior_distance_guard_m": INTERIOR_DISTANCE_M,
    "h2014_vertical_bracket_m": [H2014_MIN, H2014_MAX],
    "annual_level_method": "annual mean per GCM followed by ensemble p05/p50/p95; frozen 2030 and 2050 quantiles preserved exactly",
    "area_method": "16C4 conservative certain-water / certain-exposed / uncertain partition",
    "key_horizon_validation": "must reproduce 16C4 within 0.0005 km2",
    "outputs": {
        "rasters": str(RASTER_DIR),
        "vectors": str(VECTOR_DIR),
        "stacks": str(STACK_DIR),
        "styles": str(STYLE_DIR),
        "tables": str(TABLE_DIR),
    },
}

with open(OUT_DIR / "18D_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

files_for_manifest = [
    p
    for p in OUT_DIR.rglob("*")
    if p.is_file()
]

manifest = pd.DataFrame(
    [
        {
            "file": p.name,
            "relative_path": str(p.relative_to(OUT_DIR)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files_for_manifest
        if p.name != "18D_manifiesto.csv"
    ]
)

manifest.to_csv(
    OUT_DIR / "18D_manifiesto.csv",
    index=False,
    encoding="utf-8-sig",
)

banner("DECISION 18D")
print("[LEVELS] media anual por GCM + cuantiles ensemble")
print("[2030/2050] cuantiles frozen preservados")
print("[AREA] metodologia conservadora 16C4")
print("[UNCERTAINTY] p05-p95 + incertidumbre espacial separadas")
print("[QGIS] GeoTIFF anuales, GeoJSON, QML y stacks multibanda")
print("[STATUS] ANNUAL_QGIS_FINAL_VALIDATED")

banner("FIN PASO 18D")
print("[OUTPUT DIRECTORY]")
print(OUT_DIR)
