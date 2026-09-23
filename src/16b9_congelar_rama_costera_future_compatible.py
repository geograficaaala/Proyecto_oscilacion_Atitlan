                       















































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import shutil
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

B4_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
)

B6_DIR = (
    B4_DIR
    / "capacidad_espacial_16B6"
)

B7_DIR = (
    ROOT
    / "outputs"
    / "costa_futura_16B7"
)

B8_DIR = (
    ROOT
    / "outputs"
    / "exposicion_litoral_16B8"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "costa_future_compatible"
)

PATH_CURRENT = (
    B4_DIR
    / "16B4a_mascara_lago_actual_Sentinel_50pct.tif"
)

PATH_CAPABILITY = (
    B6_DIR
    / "16B6_capacidad_espacial_costera.tif"
)

PATH_THRESHOLD = (
    B6_DIR
    / "16B6_nivel_umbral_con_fuente_m.tif"
)

PATH_PROVENANCE = (
    B6_DIR
    / "16B6_procedencia_umbral.tif"
)

PATH_DOMAIN = (
    B6_DIR
    / "16B6_dominio_costero_evaluado.tif"
)

PATH_LEVELS = (
    B7_DIR
    / "16B7_niveles_ensemble_2030_2050.csv"
)

PATH_EXPOSURE_SUMMARY = (
    B8_DIR
    / "16B8_resumen_exposicion.csv"
)

PATH_SECTOR_SUMMARY = (
    B8_DIR
    / "16B8_resumen_sectores.csv"
)

SCENARIOS = [
    "ssp245",
    "ssp585",
]

YEARS = [
    2030,
    2050,
]


                                                                               
                     
                                                                               

EXPECTED = {
    "current_lake_pixels":
        305_058,

    "current_lake_area_km2":
        122.0232,

    "domain_pixels":
        23_490,

    "observed_direct_pixels":
        6_181,

    "ams_supported_pixels":
        311,

    "unknown_pixels":
        16_998,

    "known_pct":
        27.64,
}

EXPECTED_LEVEL_P50 = {
    ("ssp245", 2030):
        1552.348610,

    ("ssp245", 2050):
        1550.336502,

    ("ssp585", 2030):
        1552.293235,

    ("ssp585", 2050):
        1548.144740,
}

EXPECTED_EXPOSURE_AREA = {
    ("ssp245", 2030):
        1.9312,

    ("ssp245", 2050):
        1.9440,

    ("ssp585", 2030):
        1.9320,

    ("ssp585", 2050):
        1.9572,
}

EXPECTED_UNKNOWN_AREA_KM2 = 6.7992

TOL_AREA_KM2 = 5e-4
TOL_LEVEL_M = 5e-4
TOL_PCT = 0.02


                                                                               
              
                                                                               

def banner(text):

    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def almost_equal(
    a,
    b,
    tol
):

    return abs(
        float(a)
        -
        float(b)
    ) <= tol


def copy_exact(
    src,
    dst
):

    shutil.copy2(
        src,
        dst
    )

    h_src = sha256_file(
        src
    )

    h_dst = sha256_file(
        dst
    )

    if h_src != h_dst:

        raise RuntimeError(
            f"Copia no identica:\n{src}\n{dst}"
        )

    print(
        "[COPY]",
        src.name,
        "->",
        dst.name
    )

    print(
        "       SHA256 =",
        h_dst
    )

    print(
        "       identical = True"
    )

    return h_dst


                                                                               
                 
                                                                               

banner(
    "PASO 16B9 - LOCALIZACION"
)


source_files = [
    PATH_CURRENT,
    PATH_CAPABILITY,
    PATH_THRESHOLD,
    PATH_PROVENANCE,
    PATH_DOMAIN,
    PATH_LEVELS,
    PATH_EXPOSURE_SUMMARY,
    PATH_SECTOR_SUMMARY,
]


for scenario in SCENARIOS:

    for year in YEARS:

        source_files.append(
            B8_DIR
            /
            f"16B8_{scenario}_{year}_clases_exposicion_litoral.tif"
        )


for year in YEARS:

    source_files.append(
        B8_DIR
        /
        f"16B8_consenso_nueva_exposicion_{year}.tif"
    )


for p in source_files:

    if not p.exists():

        raise FileNotFoundError(
            p
        )

    print(
        "[FILE]",
        p
    )


                                                                               
                   
                                                                               

banner(
    "QC 1 - LAGO ACTUAL"
)


with rasterio.open(
    PATH_CURRENT
) as ds:

    current = (
        ds.read(1)
        >
        0
    )

    transform = ds.transform

    shape = ds.shape

    crs = ds.crs


pixel_area_m2 = abs(
    transform.a
    *
    transform.e
)


current_pixels = int(
    current.sum()
)


current_area_km2 = (
    current_pixels
    *
    pixel_area_m2
    /
    1e6
)


print(
    "[CURRENT] pixels =",
    f"{current_pixels:,}"
)


print(
    "[CURRENT] area =",
    f"{current_area_km2:.4f} km2"
)


if current_pixels != EXPECTED[
    "current_lake_pixels"
]:

    raise RuntimeError(
        "Conteo de lago actual no reproduce 16B4a."
    )


if not almost_equal(
    current_area_km2,
    EXPECTED[
        "current_lake_area_km2"
    ],
    TOL_AREA_KM2
):

    raise RuntimeError(
        "Area de lago actual no reproduce 16B4a."
    )


                                                                               
                          
                                                                               

banner(
    "QC 2 - CAPACIDAD ESPACIAL 16B6"
)


with rasterio.open(
    PATH_CAPABILITY
) as ds:

    capability = (
        ds.read(1)
        .astype(
            np.uint8
        )
    )


with rasterio.open(
    PATH_DOMAIN
) as ds:

    domain = (
        ds.read(1)
        >
        0
    )


n_domain = int(
    domain.sum()
)


n_direct = int(
    np.sum(
        capability
        ==
        1
    )
)


n_ams = int(
    np.sum(
        capability
        ==
        2
    )
)


n_unknown = int(
    np.sum(
        capability
        ==
        3
    )
)


known_pct = (
    100.0
    *
    (
        n_direct
        +
        n_ams
    )
    /
    n_domain
)


print(
    "[DOMAIN] =",
    f"{n_domain:,}"
)


print(
    "[OBSERVED DIRECT] =",
    f"{n_direct:,}"
)


print(
    "[AMS SUPPORTED] =",
    f"{n_ams:,}"
)


print(
    "[UNKNOWN] =",
    f"{n_unknown:,}"
)


print(
    "[KNOWN %] =",
    f"{known_pct:.2f}%"
)


if n_domain != EXPECTED[
    "domain_pixels"
]:

    raise RuntimeError(
        "Dominio 16B6 no coincide."
    )


if n_direct != EXPECTED[
    "observed_direct_pixels"
]:

    raise RuntimeError(
        "Conteo direct 16B6 no coincide."
    )


if n_ams != EXPECTED[
    "ams_supported_pixels"
]:

    raise RuntimeError(
        "Conteo AMS 16B6 no coincide."
    )


if n_unknown != EXPECTED[
    "unknown_pixels"
]:

    raise RuntimeError(
        "Conteo desconocido 16B6 no coincide."
    )


if not almost_equal(
    known_pct,
    EXPECTED[
        "known_pct"
    ],
    TOL_PCT
):

    raise RuntimeError(
        "Porcentaje conocido 16B6 no coincide."
    )


                                                                               
                            
                                                                               

banner(
    "QC 3 - NIVELES Y EXPOSICION 2030/2050"
)


levels = pd.read_csv(
    PATH_LEVELS
)


exposure_summary = pd.read_csv(
    PATH_EXPOSURE_SUMMARY
)


for scenario in SCENARIOS:

    for year in YEARS:

        key = (
            scenario,
            year
        )


        row_level = levels[
            (
                levels[
                    "scenario"
                ]
                ==
                scenario
            )
            &
            (
                levels[
                    "year"
                ]
                ==
                year
            )
        ]


        if len(
            row_level
        ) != 1:

            raise RuntimeError(
                f"Niveles: fila invalida {scenario} {year}."
            )


        row_level = row_level.iloc[
            0
        ]


        p50 = float(
            row_level[
                "level_p50_m"
            ]
        )


        expected_p50 = EXPECTED_LEVEL_P50[
            key
        ]


        if not almost_equal(
            p50,
            expected_p50,
            TOL_LEVEL_M
        ):

            raise RuntimeError(
                f"{scenario} {year}: p50 no reproduce 16B7."
            )


        row_exp = exposure_summary[
            (
                exposure_summary[
                    "scenario"
                ]
                ==
                scenario
            )
            &
            (
                exposure_summary[
                    "year"
                ]
                ==
                year
            )
        ]


        if len(
            row_exp
        ) != 1:

            raise RuntimeError(
                f"Exposicion: fila invalida {scenario} {year}."
            )


        row_exp = row_exp.iloc[
            0
        ]


        area_exp = float(
            row_exp[
                "new_exposed_total_area_km2"
            ]
        )


        area_unknown = float(
            row_exp[
                "unknown_area_km2"
            ]
        )


        if not almost_equal(
            area_exp,
            EXPECTED_EXPOSURE_AREA[
                key
            ],
            TOL_AREA_KM2
        ):

            raise RuntimeError(
                f"{scenario} {year}: area expuesta no reproduce 16B8."
            )


        if not almost_equal(
            area_unknown,
            EXPECTED_UNKNOWN_AREA_KM2,
            TOL_AREA_KM2
        ):

            raise RuntimeError(
                f"{scenario} {year}: area desconocida no reproduce 16B8."
            )


        print(
            f"[{scenario.upper()} {year}] "
            f"p50={p50:.3f} m | "
            f"expuesto conocido={area_exp:.4f} km2 | "
            f"desconocido={area_unknown:.4f} km2"
        )


print(
    "[PASS] niveles y resumen de exposicion reproducidos."
)


                                                                               
                               
                                                                               

banner(
    "CONGELACION BYTE-A-BYTE"
)


if OUT_DIR.exists():

    shutil.rmtree(
        OUT_DIR
    )


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


copy_plan = [
    (
        PATH_CURRENT,
        OUT_DIR
        /
        "mascara_lago_actual_2024_2026.tif",
        "current_lake",
    ),

    (
        PATH_CAPABILITY,
        OUT_DIR
        /
        "capacidad_espacial_costera.tif",
        "spatial_capability",
    ),

    (
        PATH_THRESHOLD,
        OUT_DIR
        /
        "nivel_umbral_costero_m.tif",
        "coastal_threshold",
    ),

    (
        PATH_PROVENANCE,
        OUT_DIR
        /
        "procedencia_umbral_costero.tif",
        "threshold_provenance",
    ),

    (
        PATH_DOMAIN,
        OUT_DIR
        /
        "dominio_costero_evaluado.tif",
        "evaluated_domain",
    ),

    (
        PATH_LEVELS,
        OUT_DIR
        /
        "niveles_ensemble_2030_2050.csv",
        "future_levels_summary",
    ),

    (
        PATH_EXPOSURE_SUMMARY,
        OUT_DIR
        /
        "resumen_exposicion_2030_2050.csv",
        "future_exposure_summary",
    ),

    (
        PATH_SECTOR_SUMMARY,
        OUT_DIR
        /
        "resumen_exposicion_sectores.csv",
        "future_exposure_sector_summary",
    ),
]


for scenario in SCENARIOS:

    for year in YEARS:

        copy_plan.append(
            (
                B8_DIR
                /
                f"16B8_{scenario}_{year}_clases_exposicion_litoral.tif",

                OUT_DIR
                /
                f"exposicion_litoral_{scenario}_{year}.tif",

                f"exposure_{scenario}_{year}",
            )
        )


for year in YEARS:

    copy_plan.append(
        (
            B8_DIR
            /
            f"16B8_consenso_nueva_exposicion_{year}.tif",

            OUT_DIR
            /
            f"consenso_exposicion_{year}.tif",

            f"exposure_consensus_{year}",
        )
    )


manifest_rows = []


for src, dst, role in copy_plan:

    h = copy_exact(
        src,
        dst
    )


    manifest_rows.append(
        {
            "role":
                role,

            "source_path":
                str(
                    src.resolve()
                ),

            "frozen_path":
                str(
                    dst.resolve()
                ),

            "sha256":
                h,
        }
    )


                                                                               
               
                                                                               

manifest = pd.DataFrame(
    manifest_rows
)


PATH_MANIFEST = (
    OUT_DIR
    /
    "manifiesto_costa_future_compatible.csv"
)


manifest.to_csv(
    PATH_MANIFEST,
    index=False,
    encoding="utf-8-sig"
)


manifest_hash = sha256_file(
    PATH_MANIFEST
)


print(
    "[MANIFEST]",
    PATH_MANIFEST
)


print(
    "[SHA256 MANIFEST] =",
    manifest_hash
)


                                                                               
             
                                                                               

banner(
    "METADATA FROZEN"
)


PATH_METADATA = (
    OUT_DIR
    /
    "parametros_costa_future_compatible.json"
)


metadata = {
    "step":
        "16B9",

    "status":
        "FROZEN_COAST_FUTURE_COMPATIBLE",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "current_lake": {
        "reference":
            "Sentinel 2024-2026 50% modern water mask",

        "pixels":
            current_pixels,

        "area_km2":
            float(
                current_area_km2
            ),
    },

    "spatial_capability": {
        "domain_pixels":
            n_domain,

        "observed_direct_pixels":
            n_direct,

        "amsclae_supported_pixels":
            n_ams,

        "unknown_pixels":
            n_unknown,

        "known_pct":
            float(
                known_pct
            ),

        "unknown_area_km2":
            float(
                n_unknown
                *
                pixel_area_m2
                /
                1e6
            ),
    },

    "direct_observation_range_m": {
        "min":
            1552.450,

        "max":
            1554.050,
    },

    "future_horizons": [
        2030,
        2050,
    ],

    "future_scenarios": [
        "ssp245",
        "ssp585",
    ],

    "interpretation": {
        "future_levels":
            (
                "Conditional CMIP6 scenario trajectories, not deterministic forecasts."
            ),

        "exposure_area":
            (
                "Area of newly exposed pixels only within the spatially supported "
                "part of the coastal domain. It is NOT total lake-wide exposed area."
            ),

        "unknown":
            (
                "Pixels without sufficient spatial support. They must remain unknown "
                "and must not be treated as land or water in downstream analyses."
            ),

        "deep_bathymetry":
            (
                "The poor shallow-coast alignment does not imply that the entire "
                "deep AMSCLAE bathymetry is invalid. The limitation applies primarily "
                "to exact shoreline positioning in poorly supported shallow zones."
            ),
    },

    "key_results": exposure_summary[
        [
            "scenario",
            "year",
            "level_p50_m",
            "new_exposed_total_area_km2",
            "new_exposed_robust_area_km2",
            "new_exposed_ensemble_sensitive_area_km2",
            "unknown_area_km2",
        ]
    ].to_dict(
        orient="records"
    ),

    "manifest_sha256":
        manifest_hash,

    "limitations": [
        (
            "Only 27.64% of the evaluated coastal domain has direct or locally "
            "supported spatial thresholds."
        ),
        (
            "The 6.7992 km2 classified as unknown are not predicted."
        ),
        (
            "The observed direct shoreline relation is supported only over the "
            "2024-2026 exact Sentinel/NASA range 1552.450-1554.050 m."
        ),
        (
            "AMSCLAE is used for shoreline extrapolation only in the small subset "
            "accepted by the local compatibility audit."
        ),
        (
            "These products must not be represented as a continuous deterministic "
            "future shoreline around the whole lake."
        ),
    ],
}


with open(
    PATH_METADATA,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


metadata_hash = sha256_file(
    PATH_METADATA
)


print(
    "[METADATA]",
    PATH_METADATA
)


print(
    "[SHA256 METADATA] =",
    metadata_hash
)


                                                                               
                     
                                                                               

banner(
    "QC 4 - RELECTURA FROZEN"
)


with rasterio.open(
    OUT_DIR
    /
    "capacidad_espacial_costera.tif"
) as ds:

    frozen_capability = (
        ds.read(1)
        .astype(
            np.uint8
        )
    )


if not np.array_equal(
    frozen_capability,
    capability
):

    raise RuntimeError(
        "Capacidad frozen no es identica al origen."
    )


with rasterio.open(
    OUT_DIR
    /
    "mascara_lago_actual_2024_2026.tif"
) as ds:

    frozen_current = (
        ds.read(1)
        >
        0
    )


if not np.array_equal(
    frozen_current,
    current
):

    raise RuntimeError(
        "Mascara actual frozen no es identica al origen."
    )


print(
    "[PASS] productos principales frozen releidos exactamente."
)


                                                                               
                   
                                                                               

banner(
    "RESUMEN FROZEN 16B9"
)


print(
    "[CURRENT LAKE] =",
    f"{current_area_km2:.4f} km2"
)


print(
    "[COASTAL DOMAIN] =",
    f"{n_domain*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[OBSERVED DIRECT] =",
    f"{n_direct*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[AMS SUPPORTED] =",
    f"{n_ams*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[UNKNOWN] =",
    f"{n_unknown*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[KNOWN CAPABILITY] =",
    f"{known_pct:.2f}%"
)


print()


print(
    "[FROZEN] RAMA COSTERA FUTURE-COMPATIBLE CONGELADA"
)


print()


print(
    "[NEXT] PASO 16C:"
)


print(
    "retomar la relacion nivel-area-volumen con una separacion estricta "
    "entre morfometria batimetrica del vaso y posicion exacta de la costa."
)


banner(
    "FIN PASO 16B9"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
