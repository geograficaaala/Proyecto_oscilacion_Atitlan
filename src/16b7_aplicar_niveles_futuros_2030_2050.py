                       








































































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

PATH_FUTURE = (
    ROOT
    / "outputs"
    / "nivel_future_compatible"
    / "nivel_diario_cmip6_future_compatible.pkl"
)

B6_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
    / "capacidad_espacial_16B6"
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

OUT_DIR = (
    ROOT
    / "outputs"
    / "costa_futura_16B7"
)

OUT_LEVELS = (
    OUT_DIR
    / "16B7_niveles_ensemble_2030_2050.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "16B7_resumen_espacial_2030_2050.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B7_metadata.json"
)


                                                                               
                        
                                                                               

H_ANCHOR = 1552.770

SCENARIOS = [
    "ssp245",
    "ssp585",
]

HORIZONS = [
    2030,
    2050,
]

EXPECTED_MEDIANS = {
    (
        "ssp245",
        2030
    ):
        1552.348610,

    (
        "ssp245",
        2050
    ):
        1550.336502,

    (
        "ssp585",
        2030
    ):
        1552.293235,

    (
        "ssp585",
        2050
    ):
        1548.144740,
}

EXPECTED_DIRECT = 6181
EXPECTED_AMS = 311
EXPECTED_UNKNOWN = 16998
EXPECTED_DOMAIN = 23490

MEDIAN_TOL_M = 0.05

FLOAT_NODATA = -9999.0


                  
CLASS_OUTSIDE = 0
CLASS_DIRECT = 1
CLASS_AMS = 2
CLASS_UNKNOWN = 3


                                                                               
              
                                                                               

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


def normalize_name(value):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(
            value
        ).lower()
    )


def choose_date_column(df):

    preferred = [
        "date",
        "fecha",
        "datetime",
        "time",
        "obs_date",
    ]

    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    for key in preferred:

        nk = normalize_name(
            key
        )

        if nk in normalized:
            return normalized[nk]

                        
    for c in df.columns:

        n = normalize_name(
            c
        )

        if (
            "date" in n
            or
            "fecha" in n
            or
            "time" in n
        ):

            parsed = pd.to_datetime(
                df[c],
                errors="coerce"
            )

            if parsed.notna().mean() > 0.95:
                return c

    raise RuntimeError(
        "No se pudo identificar la columna de fecha del futuro."
    )


def choose_scenario_column(df):

    preferred = [
        "scenario",
        "escenario",
        "ssp",
    ]

    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    for key in preferred:

        nk = normalize_name(
            key
        )

        if nk in normalized:
            return normalized[nk]

                                                 
    for c in df.columns:

        if not (
            pd.api.types.is_object_dtype(
                df[c]
            )
            or
            pd.api.types.is_string_dtype(
                df[c]
            )
        ):
            continue

        sample = (
            df[c]
            .dropna()
            .astype(str)
            .str.lower()
            .head(1000)
        )

        if len(sample) > 0 and sample.str.contains(
            r"ssp\s*2?45|ssp\s*5?85|ssp245|ssp585",
            regex=True
        ).mean() > 0.5:

            return c

    raise RuntimeError(
        "No se pudo identificar la columna de escenario SSP."
    )


def choose_level_column(df):

                         
    name_candidates = []

    for c in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[c]
        ):
            continue

        n = normalize_name(
            c
        )

        score = 0

        if "nivel" in n:
            score += 10

        if "level" in n:
            score += 10

        if n.endswith("m"):
            score += 2

        if any(
            bad in n
            for bad in [
                "delta",
                "change",
                "anom",
                "diff",
                "p05",
                "p95",
            ]
        ):
            score -= 20

        values = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        med = values.median()

        if np.isfinite(
            med
        ) and 1500 < med < 1600:
            score += 20

        if score > 0:

            name_candidates.append(
                (
                    score,
                    c,
                    med
                )
            )

    if len(
        name_candidates
    ) == 0:

        raise RuntimeError(
            "No se pudo identificar la columna de nivel futuro."
        )

    name_candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print(
        "[LEVEL COLUMN CANDIDATES]",
        name_candidates[:5]
    )

    return name_candidates[
        0
    ][
        1
    ]


def choose_model_column(
    df,
    date_col,
    scenario_col,
    level_col
):

    preferred_tokens = [
        "gcm",
        "model",
        "modelo",
        "member",
        "miembro",
        "trajectory",
        "trayectoria",
    ]

    candidates = []

    for c in df.columns:

        if c in [
            date_col,
            scenario_col,
            level_col,
        ]:
            continue

        n_unique = df[c].nunique(
            dropna=True
        )

        if not (
            2
            <=
            n_unique
            <=
            100
        ):
            continue

        n = normalize_name(
            c
        )

        score = 0

        for token in preferred_tokens:

            if token in n:
                score += 20

                                              
        if (
            10
            <=
            n_unique
            <=
            30
        ):
            score += 10

        if score > 0:

            candidates.append(
                (
                    score,
                    c,
                    n_unique
                )
            )

    if len(
        candidates
    ) == 0:

        raise RuntimeError(
            "No se pudo identificar la columna GCM/modelo."
        )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print(
        "[MODEL COLUMN CANDIDATES]",
        candidates[:5]
    )

    return candidates[
        0
    ][
        1
    ]


def normalize_scenario(value):

    s = normalize_name(
        value
    )

    if "245" in s:
        return "ssp245"

    if "585" in s:
        return "ssp585"

    return s


def state_median(
    capability,
    threshold,
    H
):

    state = np.zeros(
        capability.shape,
        dtype=np.uint8
    )

    direct = (
        capability
        ==
        CLASS_DIRECT
    )

    ams = (
        capability
        ==
        CLASS_AMS
    )

    unknown = (
        capability
        ==
        CLASS_UNKNOWN
    )

    state[
        direct
        &
        (
            H
            >=
            threshold
        )
    ] = 1

    state[
        direct
        &
        (
            H
            <
            threshold
        )
    ] = 2

    state[
        ams
        &
        (
            H
            >=
            threshold
        )
    ] = 3

    state[
        ams
        &
        (
            H
            <
            threshold
        )
    ] = 4

    state[
        unknown
    ] = 5

    return state


def robustness_state(
    capability,
    threshold,
    H_p05,
    H_p95
):

    state = np.zeros(
        capability.shape,
        dtype=np.uint8
    )

    known = np.isin(
        capability,
        [
            CLASS_DIRECT,
            CLASS_AMS,
        ]
    )

    unknown = (
        capability
        ==
        CLASS_UNKNOWN
    )

                                          
    robust_wet = (
        known
        &
        (
            threshold
            <=
            H_p05
        )
    )

                                              
    robust_dry = (
        known
        &
        (
            threshold
            >
            H_p95
        )
    )

    ensemble_sensitive = (
        known
        &
        (~robust_wet)
        &
        (~robust_dry)
    )

    state[
        robust_wet
    ] = 1

    state[
        ensemble_sensitive
    ] = 2

    state[
        robust_dry
    ] = 3

    state[
        unknown
    ] = 4

    return state


def change_vs_anchor(
    capability,
    threshold,
    H_future
):

    state = np.zeros(
        capability.shape,
        dtype=np.uint8
    )

    direct = (
        capability
        ==
        CLASS_DIRECT
    )

    ams = (
        capability
        ==
        CLASS_AMS
    )

    known = (
        direct
        |
        ams
    )

    unknown = (
        capability
        ==
        CLASS_UNKNOWN
    )

    wet_anchor = (
        known
        &
        (
            H_ANCHOR
            >=
            threshold
        )
    )

    wet_future = (
        known
        &
        (
            H_future
            >=
            threshold
        )
    )

    remains_wet = (
        wet_anchor
        &
        wet_future
    )

    newly_exposed_direct = (
        direct
        &
        wet_anchor
        &
        (~wet_future)
    )

    newly_exposed_ams = (
        ams
        &
        wet_anchor
        &
        (~wet_future)
    )

    already_exposed = (
        known
        &
        (~wet_anchor)
    )

    state[
        remains_wet
    ] = 1

    state[
        newly_exposed_direct
    ] = 2

    state[
        newly_exposed_ams
    ] = 3

    state[
        already_exposed
    ] = 4

    state[
        unknown
    ] = 5

    return state


                                                                               
                 
                                                                               

banner(
    "PASO 16B7 - LOCALIZACION"
)


required = [
    PATH_FUTURE,
    PATH_CAPABILITY,
    PATH_THRESHOLD,
    PATH_PROVENANCE,
    PATH_DOMAIN,
]


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
    "QC 1 - FUTURO CONGELADO"
)


future = pd.read_pickle(
    PATH_FUTURE
)


if isinstance(
    future,
    pd.Series
):

    future = future.to_frame()


if not isinstance(
    future,
    pd.DataFrame
):

    raise RuntimeError(
        "El PKL futuro no contiene un DataFrame/Series."
    )


future = future.reset_index()


print(
    "[FUTURE] rows =",
    f"{len(future):,}"
)


print(
    "[FUTURE] columns =",
    list(
        future.columns
    )
)


date_col = choose_date_column(
    future
)


scenario_col = choose_scenario_column(
    future
)


level_col = choose_level_column(
    future
)


model_col = choose_model_column(
    future,
    date_col,
    scenario_col,
    level_col
)


print(
    "[DETECTED] date =",
    date_col
)


print(
    "[DETECTED] scenario =",
    scenario_col
)


print(
    "[DETECTED] model =",
    model_col
)


print(
    "[DETECTED] level =",
    level_col
)


future[
    "_date"
] = pd.to_datetime(
    future[
        date_col
    ],
    errors="coerce"
)


future[
    "_scenario"
] = (
    future[
        scenario_col
    ]
    .map(
        normalize_scenario
    )
)


future[
    "_level"
] = pd.to_numeric(
    future[
        level_col
    ],
    errors="coerce"
)


future = future[
    future[
        "_date"
    ].notna()
    &
    future[
        "_level"
    ].notna()
].copy()


future[
    "_year"
] = future[
    "_date"
].dt.year


print(
    "[DATE RANGE] =",
    future[
        "_date"
    ].min().date(),
    "a",
    future[
        "_date"
    ].max().date()
)


print(
    "[SCENARIOS] =",
    sorted(
        future[
            "_scenario"
        ].unique()
    )
)


                                                                               
                         
                                                                               

banner(
    "QC 2 - NIVELES ENSEMBLE 2030/2050"
)


level_rows = []


for scenario in SCENARIOS:

    for year in HORIZONS:

        sub = future[
            (
                future[
                    "_scenario"
                ]
                ==
                scenario
            )
            &
            (
                future[
                    "_year"
                ]
                ==
                year
            )
        ].copy()


        if len(
            sub
        ) == 0:

            raise RuntimeError(
                f"No hay datos para {scenario} {year}."
            )


        annual_by_model = (
            sub
            .groupby(
                model_col,
                dropna=False
            )[
                "_level"
            ]
            .mean()
            .dropna()
        )


        n_models = len(
            annual_by_model
        )


        if n_models < 5:

            raise RuntimeError(
                f"{scenario} {year}: solo {n_models} modelos."
            )


        p05 = float(
            annual_by_model.quantile(
                0.05
            )
        )


        p50 = float(
            annual_by_model.quantile(
                0.50
            )
        )


        p95 = float(
            annual_by_model.quantile(
                0.95
            )
        )


        expected = EXPECTED_MEDIANS[
            (
                scenario,
                year
            )
        ]


        diff = (
            p50
            -
            expected
        )


        print(
            f"[{scenario.upper()} {year}] "
            f"n_models={n_models} | "
            f"p05={p05:.3f} | "
            f"p50={p50:.3f} | "
            f"p95={p95:.3f} | "
            f"diff expected={diff:+.4f} m"
        )


        if abs(
            diff
        ) > MEDIAN_TOL_M:

            raise RuntimeError(
                f"{scenario} {year}: la mediana calculada no reproduce 15C "
                f"dentro de {MEDIAN_TOL_M:.2f} m."
            )


        level_rows.append(
            {
                "scenario":
                    scenario,

                "year":
                    year,

                "n_models":
                    n_models,

                "level_p05_m":
                    p05,

                "level_p50_m":
                    p50,

                "level_p95_m":
                    p95,

                "expected_p50_15C_m":
                    expected,

                "difference_vs_expected_m":
                    diff,

                "change_p50_vs_anchor_m":
                    p50
                    -
                    H_ANCHOR,
            }
        )


levels_df = pd.DataFrame(
    level_rows
)


                                                                               
              
                                                                               

banner(
    "QC 3 - MAPA DE CAPACIDAD 16B6"
)


with rasterio.open(
    PATH_CAPABILITY
) as ds:

    capability = (
        ds.read(
            1
        )
        .astype(
            np.uint8
        )
    )

    profile = ds.profile.copy()

    shape = ds.shape

    transform = ds.transform

    crs = ds.crs


with rasterio.open(
    PATH_THRESHOLD
) as ds:

    if (
        ds.shape
        !=
        shape
        or
        ds.transform
        !=
        transform
        or
        ds.crs
        !=
        crs
    ):

        raise RuntimeError(
            "Threshold 16B6 en grilla incompatible."
        )

    threshold = (
        ds.read(
            1
        )
        .astype(
            np.float32
        )
    )

    threshold_nodata = ds.nodata


with rasterio.open(
    PATH_PROVENANCE
) as ds:

    provenance = (
        ds.read(
            1
        )
        .astype(
            np.uint8
        )
    )


with rasterio.open(
    PATH_DOMAIN
) as ds:

    domain = (
        ds.read(
            1
        )
        >
        0
    )


threshold_valid = np.isfinite(
    threshold
)


if threshold_nodata is not None:

    threshold_valid &= (
        threshold
        !=
        threshold_nodata
    )


n_direct = int(
    np.sum(
        capability
        ==
        CLASS_DIRECT
    )
)


n_ams = int(
    np.sum(
        capability
        ==
        CLASS_AMS
    )
)


n_unknown = int(
    np.sum(
        capability
        ==
        CLASS_UNKNOWN
    )
)


n_domain = int(
    domain.sum()
)


print(
    "[DIRECT] =",
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
    "[DOMAIN] =",
    f"{n_domain:,}"
)


if n_direct != EXPECTED_DIRECT:

    raise RuntimeError(
        "Conteo OBSERVADO_DIRECTO diferente a 16B6."
    )


if n_ams != EXPECTED_AMS:

    raise RuntimeError(
        "Conteo AMSCLAE_SOPORTADO diferente a 16B6."
    )


if n_unknown != EXPECTED_UNKNOWN:

    raise RuntimeError(
        "Conteo DESCONOCIDO diferente a 16B6."
    )


if n_domain != EXPECTED_DOMAIN:

    raise RuntimeError(
        "Conteo del dominio diferente a 16B6."
    )


known = np.isin(
    capability,
    [
        CLASS_DIRECT,
        CLASS_AMS,
    ]
)


if not np.array_equal(
    threshold_valid,
    known
):

    raise RuntimeError(
        "Umbral valido no coincide con clases conocidas."
    )


pixel_area_m2 = abs(
    transform.a
    *
    transform.e
)


                                                                               
                            
                                                                               

banner(
    "QC 4 - ESTADOS FUTUROS"
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


summary_rows = []


generated_files = []


for _, row in levels_df.iterrows():

    scenario = str(
        row[
            "scenario"
        ]
    )


    year = int(
        row[
            "year"
        ]
    )


    p05 = float(
        row[
            "level_p05_m"
        ]
    )


    p50 = float(
        row[
            "level_p50_m"
        ]
    )


    p95 = float(
        row[
            "level_p95_m"
        ]
    )


                                                                               
                    
                                                                               

    median_state = state_median(
        capability,
        threshold,
        p50
    )


    robust_state = robustness_state(
        capability,
        threshold,
        p05,
        p95
    )


    change_state = change_vs_anchor(
        capability,
        threshold,
        p50
    )


    tag = (
        f"{scenario}_{year}"
    )


    path_median = (
        OUT_DIR
        /
        f"16B7_{tag}_estado_mediano_p50.tif"
    )


    path_robust = (
        OUT_DIR
        /
        f"16B7_{tag}_robustez_p05_p95.tif"
    )


    path_change = (
        OUT_DIR
        /
        f"16B7_{tag}_cambio_mediano_vs_2026.tif"
    )


    for path, array, description in [
        (
            path_median,
            median_state,
            "estado_costero_mediano_p50"
        ),
        (
            path_robust,
            robust_state,
            "robustez_costera_ensemble_p05_p95"
        ),
        (
            path_change,
            change_state,
            "cambio_costero_mediano_vs_ancla_2026"
        ),
    ]:

        with rasterio.open(
            path,
            "w",
            **uint_profile
        ) as dst:

            dst.write(
                array.astype(
                    np.uint8
                ),
                1
            )

            dst.set_band_description(
                1,
                description
            )


        generated_files.append(
            path
        )


                                                                               
                      
                                                                               

    direct_wet = int(
        np.sum(
            median_state
            ==
            1
        )
    )


    direct_dry = int(
        np.sum(
            median_state
            ==
            2
        )
    )


    ams_wet = int(
        np.sum(
            median_state
            ==
            3
        )
    )


    ams_dry = int(
        np.sum(
            median_state
            ==
            4
        )
    )


    unknown_n = int(
        np.sum(
            median_state
            ==
            5
        )
    )


    robust_wet = int(
        np.sum(
            robust_state
            ==
            1
        )
    )


    ensemble_sensitive = int(
        np.sum(
            robust_state
            ==
            2
        )
    )


    robust_dry = int(
        np.sum(
            robust_state
            ==
            3
        )
    )


    robust_unknown = int(
        np.sum(
            robust_state
            ==
            4
        )
    )


    remains_wet = int(
        np.sum(
            change_state
            ==
            1
        )
    )


    newly_direct = int(
        np.sum(
            change_state
            ==
            2
        )
    )


    newly_ams = int(
        np.sum(
            change_state
            ==
            3
        )
    )


    already_exposed = int(
        np.sum(
            change_state
            ==
            4
        )
    )


    change_unknown = int(
        np.sum(
            change_state
            ==
            5
        )
    )


    known_wet = (
        direct_wet
        +
        ams_wet
    )


    known_dry = (
        direct_dry
        +
        ams_dry
    )


    newly_exposed = (
        newly_direct
        +
        newly_ams
    )


    print()
    print(
        f"[{scenario.upper()} {year}] p50={p50:.3f} m"
    )


    print(
        "  median known wet/exposed/unknown =",
        f"{known_wet:,} / "
        f"{known_dry:,} / "
        f"{unknown_n:,} pix"
    )


    print(
        "  newly exposed vs 2026 =",
        f"{newly_exposed:,} pix | "
        f"{newly_exposed*pixel_area_m2/1e6:.4f} km2"
    )


    print(
        "  robust wet / ensemble-dependent / robust exposed =",
        f"{robust_wet:,} / "
        f"{ensemble_sensitive:,} / "
        f"{robust_dry:,} pix"
    )


    summary_rows.append(
        {
            "scenario":
                scenario,

            "year":
                year,

            "level_p05_m":
                p05,

            "level_p50_m":
                p50,

            "level_p95_m":
                p95,

            "p50_change_vs_2026_m":
                p50
                -
                H_ANCHOR,

            "direct_wet_pixels":
                direct_wet,

            "direct_exposed_pixels":
                direct_dry,

            "ams_supported_wet_pixels":
                ams_wet,

            "ams_supported_exposed_pixels":
                ams_dry,

            "known_wet_area_km2":
                known_wet
                *
                pixel_area_m2
                /
                1e6,

            "known_exposed_area_km2":
                known_dry
                *
                pixel_area_m2
                /
                1e6,

            "unknown_area_km2":
                unknown_n
                *
                pixel_area_m2
                /
                1e6,

            "newly_exposed_direct_pixels":
                newly_direct,

            "newly_exposed_ams_pixels":
                newly_ams,

            "newly_exposed_area_vs_2026_km2":
                newly_exposed
                *
                pixel_area_m2
                /
                1e6,

            "remains_wet_area_km2":
                remains_wet
                *
                pixel_area_m2
                /
                1e6,

            "already_exposed_at_anchor_area_km2":
                already_exposed
                *
                pixel_area_m2
                /
                1e6,

            "robust_wet_area_km2":
                robust_wet
                *
                pixel_area_m2
                /
                1e6,

            "ensemble_sensitive_area_km2":
                ensemble_sensitive
                *
                pixel_area_m2
                /
                1e6,

            "robust_exposed_area_km2":
                robust_dry
                *
                pixel_area_m2
                /
                1e6,

            "robust_unknown_area_km2":
                robust_unknown
                *
                pixel_area_m2
                /
                1e6,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


                                                                               
                        
                                                                               

banner(
    "ESCRITURA TABLAS 16B7"
)


levels_df.to_csv(
    OUT_LEVELS,
    index=False,
    encoding="utf-8-sig"
)


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


print(
    "[FILE]",
    OUT_LEVELS
)


print(
    "[FILE]",
    OUT_SUMMARY
)


                                                                               
            
                                                                               

banner(
    "FIGURAS 16B7"
)


for _, row in levels_df.iterrows():

    scenario = str(
        row[
            "scenario"
        ]
    )


    year = int(
        row[
            "year"
        ]
    )


    p50 = float(
        row[
            "level_p50_m"
        ]
    )


    tag = (
        f"{scenario}_{year}"
    )


    path_median = (
        OUT_DIR
        /
        f"16B7_{tag}_estado_mediano_p50.tif"
    )


    with rasterio.open(
        path_median
    ) as ds:

        state = ds.read(
            1
        )


    fig = plt.figure(
        figsize=(12, 10)
    )


    ax = fig.add_subplot(
        111
    )


    im = ax.imshow(
        state,
        vmin=0,
        vmax=5
    )


    ax.set_title(
        f"16B7 - {scenario.upper()} {year} | p50={p50:.3f} m\n"
        "1=Obs agua | 2=Obs expuesto | 3=AMS agua | "
        "4=AMS expuesto | 5=Desconocido"
    )


    ax.set_axis_off()


    cb = plt.colorbar(
        im,
        ax=ax,
        shrink=0.75,
        ticks=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    )


    cb.set_label(
        "Clase"
    )


    plt.tight_layout()


    fig_path = (
        OUT_DIR
        /
        f"16B7_{tag}_estado_mediano_p50.png"
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
    for p in generated_files
}


hashes[
    OUT_LEVELS.name
] = sha256_file(
    OUT_LEVELS
)


hashes[
    OUT_SUMMARY.name
] = sha256_file(
    OUT_SUMMARY
)


metadata = {
    "step":
        "16B7",

    "status":
        "FUTURE_COASTAL_STATE_PARTIAL_WITH_EXPLICIT_UNKNOWN",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "future_source":
        str(
            PATH_FUTURE
        ),

    "detected_columns": {
        "date":
            str(
                date_col
            ),

        "scenario":
            str(
                scenario_col
            ),

        "model":
            str(
                model_col
            ),

        "level":
            str(
                level_col
            ),
    },

    "anchor": {
        "date":
            "2026-08-29",

        "level_m":
            H_ANCHOR,
    },

    "horizons":
        HORIZONS,

    "scenarios":
        SCENARIOS,

    "ensemble_method":
        (
            "For each scenario and horizon year, compute annual mean lake level "
            "for each GCM, then calculate p05/p50/p95 across GCM annual means."
        ),

    "capability_counts": {
        "observed_direct_pixels":
            n_direct,

        "amsclae_supported_pixels":
            n_ams,

        "unknown_pixels":
            n_unknown,

        "domain_pixels":
            n_domain,
    },

    "median_state_classes": {
        "0":
            "outside_domain",

        "1":
            "OBSERVADO_DIRECTO_water",

        "2":
            "OBSERVADO_DIRECTO_exposed",

        "3":
            "AMSCLAE_SOPORTADO_water",

        "4":
            "AMSCLAE_SOPORTADO_exposed",

        "5":
            "DESCONOCIDO",
    },

    "robustness_classes": {
        "0":
            "outside_domain",

        "1":
            "water_for_all_levels_p05_to_p95",

        "2":
            "state_depends_on_ensemble_level",

        "3":
            "exposed_for_all_levels_p05_to_p95",

        "4":
            "DESCONOCIDO_spatially",
    },

    "change_vs_anchor_classes": {
        "0":
            "outside_domain",

        "1":
            "remains_water",

        "2":
            "newly_exposed_OBSERVADO_DIRECTO",

        "3":
            "newly_exposed_AMSCLAE_SOPORTADO",

        "4":
            "already_exposed_at_2026_anchor",

        "5":
            "DESCONOCIDO",
    },

    "important_limitations": [
        (
            "These maps are conditional CMIP6 scenario products, not deterministic forecasts."
        ),
        (
            "The shoreline is not continuous where 16B6 classified pixels as DESCONOCIDO."
        ),
        (
            "OBSERVADO_DIRECTO thresholds come from Sentinel/NASA transitions observed "
            "within approximately 1552.45-1554.05 m; future levels below that range mean "
            "those observed transition pixels are exposed, but the deeper replacement "
            "shoreline may still be unknown."
        ),
        (
            "AMSCLAE thresholds are used only in the very small subset accepted by 16B4d0."
        ),
        (
            "Unknown pixels remain unknown in all scenarios and are never silently filled."
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
    "DECISION QC 16B7"
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


print()


print(
    "[PASS PARTIAL FUTURE MAPS]"
)


print(
    "Se aplicaron 2030/2050 al mapa 16B6 conservando explicitamente "
    "OBSERVADO_DIRECTO, AMSCLAE_SOPORTADO y DESCONOCIDO."
)


print(
    "Los mapas NO deben interpretarse como una costa continua en zonas desconocidas."
)


print()


print(
    "[NEXT] 16B8:"
)


print(
    "extraer zonas de exposicion relevantes para planificacion litoral "
    "(incluido tul) y resumirlas por escenario/horizonte, manteniendo "
    "procedencia e incertidumbre espacial."
)


banner(
    "FIN PASO 16B7"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
