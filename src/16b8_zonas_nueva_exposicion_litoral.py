                       

































































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

B7_DIR = (
    ROOT
    / "outputs"
    / "costa_futura_16B7"
)

B4_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
)

PATH_CURRENT_LAKE = (
    B4_DIR
    / "16B4a_mascara_lago_actual_Sentinel_50pct.tif"
)

PATH_LEVELS = (
    B7_DIR
    / "16B7_niveles_ensemble_2030_2050.csv"
)

PATH_B7_SUMMARY = (
    B7_DIR
    / "16B7_resumen_espacial_2030_2050.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "exposicion_litoral_16B8"
)

OUT_SUMMARY = (
    OUT_DIR
    / "16B8_resumen_exposicion.csv"
)

OUT_SECTORS = (
    OUT_DIR
    / "16B8_resumen_sectores.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B8_metadata.json"
)


SCENARIOS = [
    "ssp245",
    "ssp585",
]

HORIZONS = [
    2030,
    2050,
]


                                                                               
              
                                                                               

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


def check_same_grid(
    ds,
    shape,
    transform,
    crs,
    name
):

    if (
        ds.shape != shape
        or
        ds.transform != transform
        or
        ds.crs != crs
    ):

        raise RuntimeError(
            f"{name}: grilla incompatible."
        )


def build_exposure_class(
    change_state,
    robust_state
):


















    out = np.zeros(
        change_state.shape,
        dtype=np.uint8
    )

                     
    out[
        change_state
        ==
        1
    ] = 1

                                       
    out[
        (
            change_state
            ==
            2
        )
        &
        (
            robust_state
            ==
            3
        )
    ] = 2

                                                    
    out[
        (
            change_state
            ==
            2
        )
        &
        (
            robust_state
            ==
            2
        )
    ] = 3

                                   
    out[
        (
            change_state
            ==
            3
        )
        &
        (
            robust_state
            ==
            3
        )
    ] = 4

                                                
    out[
        (
            change_state
            ==
            3
        )
        &
        (
            robust_state
            ==
            2
        )
    ] = 5

                                     
    out[
        change_state
        ==
        4
    ] = 6

                  
    out[
        change_state
        ==
        5
    ] = 7

                                                                                 
    new_exposed = np.isin(
        change_state,
        [
            2,
            3,
        ]
    )

    bad = (
        new_exposed
        &
        (~np.isin(
            robust_state,
            [
                2,
                3,
            ]
        ))
    )

    if bad.any():

        raise RuntimeError(
            f"Hay {bad.sum()} pixeles de nueva exposicion con clase de robustez incompatible."
        )

    return out


                                                                               
                 
                                                                               

banner(
    "PASO 16B8 - LOCALIZACION"
)


required = [
    PATH_CURRENT_LAKE,
    PATH_LEVELS,
    PATH_B7_SUMMARY,
]


for scenario in SCENARIOS:

    for year in HORIZONS:

        required.extend(
            [
                B7_DIR
                /
                f"16B7_{scenario}_{year}_cambio_mediano_vs_2026.tif",

                B7_DIR
                /
                f"16B7_{scenario}_{year}_robustez_p05_p95.tif",
            ]
        )


for p in required:

    if not p.exists():

        raise FileNotFoundError(
            p
        )

    print(
        "[FILE]",
        p
    )


                                                                               
                   
                                                                               

banner(
    "QC 1 - GRILLA"
)


with rasterio.open(
    PATH_CURRENT_LAKE
) as ds:

    current_lake = (
        ds.read(1)
        > 0
    )

    profile = ds.profile.copy()

    shape = ds.shape

    transform = ds.transform

    crs = ds.crs


pixel_area_m2 = abs(
    transform.a
    *
    transform.e
)


print(
    "[GRID] shape =",
    shape
)


print(
    "[GRID] CRS =",
    crs
)


                                                                               
                    
                                                                               

banner(
    "QC 2 - NIVELES FUTUROS"
)


levels_df = pd.read_csv(
    PATH_LEVELS
)


b7_summary = pd.read_csv(
    PATH_B7_SUMMARY
)


print(
    levels_df[
        [
            "scenario",
            "year",
            "level_p05_m",
            "level_p50_m",
            "level_p95_m",
            "change_p50_vs_anchor_m",
        ]
    ].to_string(
        index=False,
        formatters={
            "level_p05_m":
                "{:.3f}".format,

            "level_p50_m":
                "{:.3f}".format,

            "level_p95_m":
                "{:.3f}".format,

            "change_p50_vs_anchor_m":
                "{:+.3f}".format,
        }
    )
)


                                                                               
                              
                                                                               

banner(
    "QC 3 - MAPAS DE EXPOSICION"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


uint_profile = profile.copy()


uint_profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate",
)


exposure_maps = {}

summary_rows = []

generated = []


for scenario in SCENARIOS:

    for year in HORIZONS:

        change_path = (
            B7_DIR
            /
            f"16B7_{scenario}_{year}_cambio_mediano_vs_2026.tif"
        )


        robust_path = (
            B7_DIR
            /
            f"16B7_{scenario}_{year}_robustez_p05_p95.tif"
        )


        with rasterio.open(
            change_path
        ) as ds:

            check_same_grid(
                ds,
                shape,
                transform,
                crs,
                "change raster"
            )

            change_state = (
                ds.read(1)
                .astype(
                    np.uint8
                )
            )


        with rasterio.open(
            robust_path
        ) as ds:

            check_same_grid(
                ds,
                shape,
                transform,
                crs,
                "robust raster"
            )

            robust_state = (
                ds.read(1)
                .astype(
                    np.uint8
                )
            )


        exposure = build_exposure_class(
            change_state,
            robust_state
        )


        exposure_maps[
            (
                scenario,
                year
            )
        ] = exposure


        out_path = (
            OUT_DIR
            /
            f"16B8_{scenario}_{year}_clases_exposicion_litoral.tif"
        )


        with rasterio.open(
            out_path,
            "w",
            **uint_profile
        ) as dst:

            dst.write(
                exposure,
                1
            )

            dst.set_band_description(
                1,
                "clase_exposicion_litoral"
            )


        generated.append(
            out_path
        )


                  
        level_row = levels_df[
            (
                levels_df[
                    "scenario"
                ]
                ==
                scenario
            )
            &
            (
                levels_df[
                    "year"
                ]
                ==
                year
            )
        ]


        if len(
            level_row
        ) != 1:

            raise RuntimeError(
                f"No hay exactamente una fila de nivel para {scenario} {year}."
            )


        level_row = level_row.iloc[
            0
        ]


                  
        class_counts = {
            cls:
                int(
                    np.sum(
                        exposure
                        ==
                        cls
                    )
                )
            for cls in range(
                1,
                8
            )
        }


        robust_direct = class_counts[
            2
        ]


        sensitive_direct = class_counts[
            3
        ]


        robust_ams = class_counts[
            4
        ]


        sensitive_ams = class_counts[
            5
        ]


        new_total = (
            robust_direct
            +
            sensitive_direct
            +
            robust_ams
            +
            sensitive_ams
        )


        robust_total = (
            robust_direct
            +
            robust_ams
        )


        sensitive_total = (
            sensitive_direct
            +
            sensitive_ams
        )


        print()
        print(
            f"[{scenario.upper()} {year}]"
        )


        print(
            "  nueva exposicion total =",
            f"{new_total:,} pix | "
            f"{new_total*pixel_area_m2/1e6:.4f} km2"
        )


        print(
            "  robusta p05-p95 =",
            f"{robust_total:,} pix | "
            f"{robust_total*pixel_area_m2/1e6:.4f} km2"
        )


        print(
            "  dependiente ensemble =",
            f"{sensitive_total:,} pix | "
            f"{sensitive_total*pixel_area_m2/1e6:.4f} km2"
        )


        print(
            "  directo / AMS =",
            f"{robust_direct+sensitive_direct:,} / "
            f"{robust_ams+sensitive_ams:,} pix"
        )


        print(
            "  desconocido =",
            f"{class_counts[7]:,} pix | "
            f"{class_counts[7]*pixel_area_m2/1e6:.4f} km2"
        )


        summary_rows.append(
            {
                "scenario":
                    scenario,

                "year":
                    year,

                "level_p05_m":
                    float(
                        level_row[
                            "level_p05_m"
                        ]
                    ),

                "level_p50_m":
                    float(
                        level_row[
                            "level_p50_m"
                        ]
                    ),

                "level_p95_m":
                    float(
                        level_row[
                            "level_p95_m"
                        ]
                    ),

                "change_p50_vs_anchor_m":
                    float(
                        level_row[
                            "change_p50_vs_anchor_m"
                        ]
                    ),

                "new_exposed_total_pixels":
                    new_total,

                "new_exposed_total_area_km2":
                    new_total
                    *
                    pixel_area_m2
                    /
                    1e6,

                "new_exposed_robust_pixels":
                    robust_total,

                "new_exposed_robust_area_km2":
                    robust_total
                    *
                    pixel_area_m2
                    /
                    1e6,

                "new_exposed_ensemble_sensitive_pixels":
                    sensitive_total,

                "new_exposed_ensemble_sensitive_area_km2":
                    sensitive_total
                    *
                    pixel_area_m2
                    /
                    1e6,

                "new_exposed_direct_pixels":
                    robust_direct
                    +
                    sensitive_direct,

                "new_exposed_direct_area_km2":
                    (
                        robust_direct
                        +
                        sensitive_direct
                    )
                    *
                    pixel_area_m2
                    /
                    1e6,

                "new_exposed_ams_supported_pixels":
                    robust_ams
                    +
                    sensitive_ams,

                "new_exposed_ams_supported_area_km2":
                    (
                        robust_ams
                        +
                        sensitive_ams
                    )
                    *
                    pixel_area_m2
                    /
                    1e6,

                "unknown_pixels":
                    class_counts[
                        7
                    ],

                "unknown_area_km2":
                    class_counts[
                        7
                    ]
                    *
                    pixel_area_m2
                    /
                    1e6,

                "already_exposed_anchor_pixels":
                    class_counts[
                        6
                    ],

                "remains_water_pixels":
                    class_counts[
                        1
                    ],
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)


                                                                               
                                            
                                                                               

banner(
    "QC 4 - CONSENSO SSP245/SSP585"
)


consensus_paths = []


for year in HORIZONS:

    a = exposure_maps[
        (
            "ssp245",
            year
        )
    ]


    b = exposure_maps[
        (
            "ssp585",
            year
        )
    ]


    unknown = (
        (a == 7)
        |
        (b == 7)
    )


    new_a = np.isin(
        a,
        [
            2,
            3,
            4,
            5,
        ]
    )


    new_b = np.isin(
        b,
        [
            2,
            3,
            4,
            5,
        ]
    )


    outside = (
        (a == 0)
        &
        (b == 0)
    )


    consensus = np.zeros(
        shape,
        dtype=np.uint8
    )


                                                       
    known_domain = (
        (~outside)
        &
        (~unknown)
    )


    consensus[
        known_domain
        &
        (~new_a)
        &
        (~new_b)
    ] = 1


    consensus[
        known_domain
        &
        (
            new_a
            ^
            new_b
        )
    ] = 2


    consensus[
        known_domain
        &
        new_a
        &
        new_b
    ] = 3


    consensus[
        unknown
    ] = 4


    out_path = (
        OUT_DIR
        /
        f"16B8_consenso_nueva_exposicion_{year}.tif"
    )


    with rasterio.open(
        out_path,
        "w",
        **uint_profile
    ) as dst:

        dst.write(
            consensus,
            1
        )

        dst.set_band_description(
            1,
            "consenso_nueva_exposicion_entre_escenarios"
        )


    generated.append(
        out_path
    )


    consensus_paths.append(
        out_path
    )


    print(
        f"[{year}] exposed both scenarios =",
        f"{np.sum(consensus==3):,} pix | "
        f"{np.sum(consensus==3)*pixel_area_m2/1e6:.4f} km2"
    )


    print(
        f"[{year}] exposed one scenario =",
        f"{np.sum(consensus==2):,} pix | "
        f"{np.sum(consensus==2)*pixel_area_m2/1e6:.4f} km2"
    )


    print(
        f"[{year}] unknown =",
        f"{np.sum(consensus==4):,} pix | "
        f"{np.sum(consensus==4)*pixel_area_m2/1e6:.4f} km2"
    )


                                                                               
                       
                                                                               

banner(
    "QC 5 - RESUMEN POR SECTOR"
)


lake_rows, lake_cols = np.where(
    current_lake
)


cy = float(
    np.mean(
        lake_rows
    )
)


cx = float(
    np.mean(
        lake_cols
    )
)


rr_grid, cc_grid = np.indices(
    shape
)


dx = (
    cc_grid
    -
    cx
)


dy = (
    cy
    -
    rr_grid
)


azimuth = (
    np.degrees(
        np.arctan2(
            dx,
            dy
        )
    )
    +
    360.0
) % 360.0


sector_grid = (
    np.floor(
        azimuth
        /
        30.0
    ).astype(
        np.int16
    )
    +
    1
)


sector_rows = []


for scenario in SCENARIOS:

    for year in HORIZONS:

        exposure = exposure_maps[
            (
                scenario,
                year
            )
        ]


        new_exposed = np.isin(
            exposure,
            [
                2,
                3,
                4,
                5,
            ]
        )


        robust_new = np.isin(
            exposure,
            [
                2,
                4,
            ]
        )


        unknown = (
            exposure
            ==
            7
        )


        for sector in range(
            1,
            13
        ):

            sm = (
                sector_grid
                ==
                sector
            )


            new_n = int(
                np.sum(
                    new_exposed
                    &
                    sm
                )
            )


            robust_n = int(
                np.sum(
                    robust_new
                    &
                    sm
                )
            )


            unknown_n = int(
                np.sum(
                    unknown
                    &
                    sm
                )
            )


            sector_rows.append(
                {
                    "scenario":
                        scenario,

                    "year":
                        year,

                    "sector":
                        sector,

                    "azimuth_lo":
                        (sector - 1) * 30,

                    "azimuth_hi":
                        sector * 30,

                    "new_exposed_pixels":
                        new_n,

                    "new_exposed_area_km2":
                        new_n
                        *
                        pixel_area_m2
                        /
                        1e6,

                    "robust_new_exposed_pixels":
                        robust_n,

                    "robust_new_exposed_area_km2":
                        robust_n
                        *
                        pixel_area_m2
                        /
                        1e6,

                    "unknown_pixels":
                        unknown_n,

                    "unknown_area_km2":
                        unknown_n
                        *
                        pixel_area_m2
                        /
                        1e6,
                }
            )


sector_df = pd.DataFrame(
    sector_rows
)


                                                                               
                     
                                                                               

banner(
    "ESCRITURA TABLAS 16B8"
)


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


sector_df.to_csv(
    OUT_SECTORS,
    index=False,
    encoding="utf-8-sig"
)


print(
    "[FILE]",
    OUT_SUMMARY
)


print(
    "[FILE]",
    OUT_SECTORS
)


                                                                               
            
                                                                               

banner(
    "FIGURAS 16B8"
)


for scenario in SCENARIOS:

    for year in HORIZONS:

        exposure = exposure_maps[
            (
                scenario,
                year
            )
        ]


        fig = plt.figure(
            figsize=(12, 10)
        )


        ax = fig.add_subplot(
            111
        )


        im = ax.imshow(
            exposure,
            vmin=0,
            vmax=7
        )


        ax.set_title(
            f"16B8 - Nueva exposicion litoral {scenario.upper()} {year}\n"
            "2/4=robusta | 3/5=depende ensemble | 7=desconocido"
        )


        ax.set_axis_off()


        cb = plt.colorbar(
            im,
            ax=ax,
            shrink=0.75,
            ticks=list(
                range(
                    0,
                    8
                )
            )
        )


        cb.set_label(
            "Clase"
        )


        plt.tight_layout()


        fig_path = (
            OUT_DIR
            /
            f"16B8_{scenario}_{year}_clases_exposicion_litoral.png"
        )


        fig.savefig(
            fig_path,
            dpi=180,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


                                                                               
              
                                                                               

hashes = {
    p.name:
        sha256_file(
            p
        )
    for p in generated
}


hashes[
    OUT_SUMMARY.name
] = sha256_file(
    OUT_SUMMARY
)


hashes[
    OUT_SECTORS.name
] = sha256_file(
    OUT_SECTORS
)


metadata = {
    "step":
        "16B8",

    "status":
        "LITTORAL_EXPOSURE_SCREENING_READY",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "purpose":
        (
            "Screen newly exposed littoral zones for planning while retaining "
            "spatial provenance, ensemble dependence, and explicit unknown areas."
        ),

    "important_scope":
        (
            "This is hydrological exposure screening only. It is not ecological "
            "suitability mapping for tul or any other planting."
        ),

    "exposure_classes": {
        "0":
            "outside_evaluated_domain",

        "1":
            "remains_water_at_p50",

        "2":
            "newly_exposed_OBSERVADO_DIRECTO_robust_p05_p95",

        "3":
            "newly_exposed_OBSERVADO_DIRECTO_ensemble_sensitive",

        "4":
            "newly_exposed_AMSCLAE_SOPORTADO_robust_p05_p95",

        "5":
            "newly_exposed_AMSCLAE_SOPORTADO_ensemble_sensitive",

        "6":
            "already_exposed_at_2026_anchor",

        "7":
            "DESCONOCIDO",
    },

    "consensus_classes": {
        "0":
            "outside_domain",

        "1":
            "not_newly_exposed_in_either_scenario",

        "2":
            "newly_exposed_in_one_scenario",

        "3":
            "newly_exposed_in_both_scenarios",

        "4":
            "DESCONOCIDO",
    },

    "interpretation_of_robust":
        (
            "Robust means exposed throughout the scenario/horizon p05-p95 ensemble "
            "level interval on spatially supported pixels. It is not a calibrated "
            "probability or certainty statement."
        ),

    "limitations": [
        (
            "Large parts of the evaluated coastal domain remain spatially unknown."
        ),
        (
            "Future levels are conditional CMIP6 scenario trajectories, not deterministic forecasts."
        ),
        (
            "Ecological planting suitability requires additional variables such as "
            "substrate, slope, wave exposure, water quality, land use/access and habitat constraints."
        ),
        (
            "Do not interpret these rasters as a continuous future shoreline where class 7 occurs."
        ),
    ],

    "sha256":
        hashes,
}


with open(
    OUT_METADATA,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


                                                                               
              
                                                                               

banner(
    "DECISION QC 16B8"
)


print(
    summary_df[
        [
            "scenario",
            "year",
            "level_p50_m",
            "new_exposed_total_area_km2",
            "new_exposed_robust_area_km2",
            "new_exposed_ensemble_sensitive_area_km2",
            "unknown_area_km2",
        ]
    ].to_string(
        index=False,
        formatters={
            "level_p50_m":
                "{:.3f}".format,

            "new_exposed_total_area_km2":
                "{:.4f}".format,

            "new_exposed_robust_area_km2":
                "{:.4f}".format,

            "new_exposed_ensemble_sensitive_area_km2":
                "{:.4f}".format,

            "unknown_area_km2":
                "{:.4f}".format,
        }
    )
)


print()


print(
    "[PASS EXPOSURE SCREENING]"
)


print(
    "Se generaron zonas de nueva exposicion con procedencia, robustez "
    "ensemble y DESCONOCIDO explicito."
)


print(
    "Estas capas sirven como screening hidrologico para planificacion; "
    "todavia NO son mapas de aptitud para plantar tul."
)


banner(
    "FIN PASO 16B8"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
