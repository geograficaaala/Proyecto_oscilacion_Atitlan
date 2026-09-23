                       














































































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import shutil
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

DOWNLOADS = ROOT / "data" / "external"

B4_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
)

                            
OBSOLETE_B4C_DIR = (
    B4_DIR
    / "corredor_oscilacion_16B4c"
)

PATH_LAKE_CURRENT = (
    B4_DIR
    / "16B4a_mascara_lago_actual_Sentinel_50pct.tif"
)

PATH_NO_BATHY = (
    B4_DIR
    / "16B4a_agua_actual_sin_batimetria_AMSCLAE.tif"
)

PATH_SHORE = (
    B4_DIR
    / "16B4a_corredor_litoral_Sentinel_20_80.tif"
)

EXACT_STACK_NAMES = [
    "atitlan_16A2b_costas_nivel_NASA_exacto_01_stack_2024.tif",
    "atitlan_16A2b_costas_nivel_NASA_exacto_02_stack_2025.tif",
    "atitlan_16A2b_costas_nivel_NASA_exacto_03_stack_2026.tif",
]

EXACT_MANIFEST_NAME = (
    "atitlan_16A2b_costas_nivel_NASA_exacto_04_manifiesto.csv"
)

OUT_DIR = (
    B4_DIR
    / "corredor_directo_16B4c2"
)

OUT_CORRIDOR = (
    OUT_DIR
    / "16B4c2_corredor_transicion_directa.tif"
)

OUT_THRESHOLD = (
    OUT_DIR
    / "16B4c2_nivel_umbral_directo_m.tif"
)

OUT_CLASS = (
    OUT_DIR
    / "16B4c2_clase_temporal_exacta.tif"
)

OUT_STATE_LOW = (
    OUT_DIR
    / "16B4c2_estado_transicion_Hmin.tif"
)

OUT_STATE_ANCHOR = (
    OUT_DIR
    / "16B4c2_estado_transicion_Hanchor.tif"
)

OUT_STATE_HIGH = (
    OUT_DIR
    / "16B4c2_estado_transicion_Hmax.tif"
)

OUT_GAP_DIRECT = (
    OUT_DIR
    / "16B4c2_agua_actual_sin_AMSCLAE_con_transicion_directa.tif"
)

OUT_GAP_NO_DIRECT = (
    OUT_DIR
    / "16B4c2_agua_actual_sin_AMSCLAE_sin_transicion_directa.tif"
)

OUT_VALIDATION = (
    OUT_DIR
    / "16B4c2_validacion_interna_por_fecha.csv"
)

OUT_CURVE = (
    OUT_DIR
    / "16B4c2_curva_cambio_area_corredor.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "16B4c2_resumen.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B4c2_metadata.json"
)

OUT_FIG_CORRIDOR = (
    OUT_DIR
    / "16B4c2_corredor_transicion_directa.png"
)

OUT_FIG_VALIDATION = (
    OUT_DIR
    / "16B4c2_consistencia_transicion_por_fecha.png"
)


                                                                               
               
                                                                               

                                       
ROI_DILATION_PIXELS = 3                                           

MIN_VALID = 10
MIN_WET = 2
MIN_DRY = 2
MIN_LEVEL_SPAN_M = 0.50
MIN_ACCURACY = 0.80

                        
PERSISTENT_WET_FRACTION = 0.90
PERSISTENT_DRY_FRACTION = 0.10

H_ANCHOR = 1552.770
FLOAT_NODATA = -9999.0


                                                                               
              
                                                                               

def banner(text):
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def find_file(name):

    candidates = [
        DOWNLOADS / name,
        ROOT / name,
        ROOT / "data" / name,
        ROOT / "outputs" / name,
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    hits = []

    if DOWNLOADS.exists():
        hits.extend(
            DOWNLOADS.rglob(name)
        )

    if ROOT.exists():
        hits.extend(
            ROOT.rglob(name)
        )

    hits = list(
        dict.fromkeys(
            p.resolve()
            for p in hits
        )
    )

    if len(hits) == 0:
        raise FileNotFoundError(
            f"No se encontro:\n{name}"
        )

    if len(hits) > 1:
        print(
            f"[WARN] {name}: {len(hits)} copias; usando {hits[0]}"
        )

    return hits[0]


def lookup(path):

    with rasterio.open(path) as ds:

        return {
            str(desc): i
            for i, desc
            in enumerate(
                ds.descriptions,
                start=1
            )
            if desc is not None
        }


def load_wv(
    manifest,
    raster_paths,
    pixel_index
):

    manifest = manifest.reset_index(
        drop=True
    )

    lookups = {
        p: lookup(p)
        for p in raster_paths
    }

    sources = {}

    for _, row in manifest.iterrows():

        for column in [
            "water_band",
            "valid_band",
        ]:

            name = str(
                row[column]
            )

            found = [
                p
                for p in raster_paths
                if name in lookups[p]
            ]

            if len(found) != 1:
                raise RuntimeError(
                    f"{name}: encontrado {len(found)} veces."
                )

            p = found[0]

            sources[name] = (
                p,
                lookups[p][name]
            )

    W = np.zeros(
        (
            len(pixel_index),
            len(manifest)
        ),
        dtype=bool
    )

    V = np.zeros_like(
        W
    )

    datasets = {
        p: rasterio.open(p)
        for p in raster_paths
    }

    try:

        for j, row in manifest.iterrows():

            wb = str(
                row["water_band"]
            )

            vb = str(
                row["valid_band"]
            )

            pw, iw = sources[wb]
            pv, iv = sources[vb]

            water = (
                datasets[pw]
                .read(iw)
                .ravel()[pixel_index]
                > 0
            )

            valid = (
                datasets[pv]
                .read(iv)
                .ravel()[pixel_index]
                > 0
            )

            W[:, j] = (
                water
                &
                valid
            )

            V[:, j] = valid

    finally:

        for ds in datasets.values():
            ds.close()

    return W, V


def fit_monotonic(
    W,
    V,
    H
):

    levels = np.unique(
        H
    )

    levels.sort()

    if len(levels) < 2:
        raise RuntimeError(
            "No hay suficientes niveles."
        )

    candidates = (
        levels[:-1]
        +
        levels[1:]
    ) / 2.0

    errors = np.zeros(
        (
            W.shape[0],
            len(candidates)
        ),
        dtype=np.int16
    )

    for k, z in enumerate(
        candidates
    ):

        predicted = (
            H
            >=
            z
        )

        errors[:, k] = (
            (
                (
                    W
                    !=
                    predicted[None, :]
                )
                &
                V
            )
            .sum(
                axis=1
            )
        )

    min_error = errors.min(
        axis=1
    )

    winners = (
        errors
        ==
        min_error[:, None]
    )

    z_est = (
        winners
        *
        candidates[None, :]
    ).sum(
        axis=1
    ) / winners.sum(
        axis=1
    )

    n_valid = V.sum(
        axis=1
    )

    n_wet = (
        W
        &
        V
    ).sum(
        axis=1
    )

    n_dry = (
        (~W)
        &
        V
    ).sum(
        axis=1
    )

    accuracy = np.full(
        W.shape[0],
        np.nan,
        dtype=float
    )

    usable = (
        n_valid
        > 0
    )

    accuracy[usable] = (
        1.0
        -
        min_error[usable]
        /
        n_valid[usable]
    )

    h_min = np.full(
        W.shape[0],
        np.nan,
        dtype=float
    )

    h_max = np.full(
        W.shape[0],
        np.nan,
        dtype=float
    )

    for i in range(
        W.shape[0]
    ):

        m = V[i]

        if m.any():

            vals = H[m]

            h_min[i] = vals.min()
            h_max[i] = vals.max()

    span = (
        h_max
        -
        h_min
    )

    good = (
        (n_valid >= MIN_VALID)
        &
        (n_wet >= MIN_WET)
        &
        (n_dry >= MIN_DRY)
        &
        (span >= MIN_LEVEL_SPAN_M)
        &
        (accuracy >= MIN_ACCURACY)
    )

    return {
        "z":
            z_est,

        "good":
            good,

        "accuracy":
            accuracy,

        "n_valid":
            n_valid,

        "n_wet":
            n_wet,

        "n_dry":
            n_dry,

        "span":
            span,
    }


def confusion_metrics(
    observed,
    predicted
):

    obs = np.asarray(
        observed,
        dtype=bool
    )

    pred = np.asarray(
        predicted,
        dtype=bool
    )

    if len(obs) == 0:

        return {
            "n":
                0,
            "accuracy":
                np.nan,
            "balanced_accuracy":
                np.nan,
            "wet_iou":
                np.nan,
            "wet_f1":
                np.nan,
        }

    tp = int(
        np.sum(
            obs
            &
            pred
        )
    )

    tn = int(
        np.sum(
            (~obs)
            &
            (~pred)
        )
    )

    fp = int(
        np.sum(
            (~obs)
            &
            pred
        )
    )

    fn = int(
        np.sum(
            obs
            &
            (~pred)
        )
    )

    accuracy = (
        tp
        +
        tn
    ) / len(obs)

    wet_recall = (
        tp
        /
        (
            tp
            +
            fn
        )
        if (
            tp
            +
            fn
        ) > 0
        else
        np.nan
    )

    dry_recall = (
        tn
        /
        (
            tn
            +
            fp
        )
        if (
            tn
            +
            fp
        ) > 0
        else
        np.nan
    )

    balanced_accuracy = np.nanmean(
        [
            wet_recall,
            dry_recall,
        ]
    )

    wet_iou = (
        tp
        /
        (
            tp
            +
            fp
            +
            fn
        )
        if (
            tp
            +
            fp
            +
            fn
        ) > 0
        else
        np.nan
    )

    wet_f1 = (
        2
        *
        tp
        /
        (
            2
            *
            tp
            +
            fp
            +
            fn
        )
        if (
            2
            *
            tp
            +
            fp
            +
            fn
        ) > 0
        else
        np.nan
    )

    return {
        "n":
            int(
                len(obs)
            ),

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "wet_iou":
            float(
                wet_iou
            ),

        "wet_f1":
            float(
                wet_f1
            ),
    }


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


                                                                               
                             
                                                                               

banner(
    "LIMPIEZA - ELIMINAR 16B4c ANTERIOR RECHAZADO"
)


print(
    "[OBSOLETE]",
    OBSOLETE_B4C_DIR
)


if OBSOLETE_B4C_DIR.exists():

    shutil.rmtree(
        OBSOLETE_B4C_DIR
    )

    print(
        "[DELETED] salida 16B4c anterior eliminada."
    )

else:

    print(
        "[OK] no existe salida 16B4c anterior."
    )


                                                                               
                 
                                                                               

banner(
    "PASO 16B4c2 - LOCALIZACION"
)


for p in [
    PATH_LAKE_CURRENT,
    PATH_NO_BATHY,
    PATH_SHORE,
]:

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    print(
        "[FILE]",
        p
    )


exact_paths = [
    find_file(
        name
    )
    for name in EXACT_STACK_NAMES
]


manifest_path = find_file(
    EXACT_MANIFEST_NAME
)


for p in exact_paths + [
    manifest_path
]:

    print(
        "[FILE]",
        p
    )


                                                                               
           
                                                                               

banner(
    "QC 1 - GRILLA Y ROI"
)


with rasterio.open(
    PATH_LAKE_CURRENT
) as ds:

    lake_current = (
        ds.read(1)
        > 0
    )

    profile = ds.profile.copy()

    shape = ds.shape

    transform = ds.transform

    crs = ds.crs


with rasterio.open(
    PATH_NO_BATHY
) as ds:

    no_bathy = (
        ds.read(1)
        > 0
    )


with rasterio.open(
    PATH_SHORE
) as ds:

    shore = (
        ds.read(1)
        > 0
    )


pixel_area_m2 = abs(
    transform.a
    *
    transform.e
)


roi = binary_dilation(
    shore,
    iterations=ROI_DILATION_PIXELS
)


roi_index = np.flatnonzero(
    roi.ravel()
)


print(
    "[GRID] shape =",
    shape
)


print(
    "[GRID] CRS =",
    crs
)


print(
    "[SHORE 20-80] =",
    f"{shore.sum():,} pix | "
    f"{shore.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[ROI ~60m] =",
    f"{roi.sum():,} pix | "
    f"{roi.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                          
                                                                               

banner(
    "QC 2 - OBSERVACIONES EXACTAS"
)


exact = pd.read_csv(
    manifest_path
).reset_index(
    drop=True
)


exact[
    "acquisition_date"
] = pd.to_datetime(
    exact[
        "acquisition_date"
    ]
)


H = (
    exact[
        "lake_level_m"
    ]
    .to_numpy(
        float
    )
)


H_MIN = float(
    np.min(
        H
    )
)


H_MAX = float(
    np.max(
        H
    )
)


W, V = load_wv(
    exact,
    exact_paths,
    roi_index
)


fit = fit_monotonic(
    W,
    V,
    H
)


print(
    "[DATES] =",
    len(
        exact
    )
)


print(
    "[H RANGE] =",
    f"{H_MIN:.3f} - {H_MAX:.3f} m"
)


print(
    "[EXACT GOOD TRANSITION] =",
    f"{fit['good'].sum():,}"
)


if fit[
    "good"
].sum() != 6181:

    raise RuntimeError(
        "No se reprodujeron exactamente los 6,181 pixeles EXACT GOOD."
    )


                                                                               
                               
                                                                               

banner(
    "QC 3 - CLASES TEMPORALES DIRECTAS"
)


n_valid = fit[
    "n_valid"
]


n_wet = fit[
    "n_wet"
]


n_dry = fit[
    "n_dry"
]


wet_fraction = np.full(
    len(
        roi_index
    ),
    np.nan,
    dtype=float
)


m_valid = (
    n_valid
    > 0
)


wet_fraction[
    m_valid
] = (
    n_wet[
        m_valid
    ]
    /
    n_valid[
        m_valid
    ]
)


sufficient_temporal = (
    (n_valid >= MIN_VALID)
    &
    (fit["span"] >= MIN_LEVEL_SPAN_M)
)


persistent_wet = (
    sufficient_temporal
    &
    (
        wet_fraction
        >=
        PERSISTENT_WET_FRACTION
    )
    &
    (~fit["good"])
)


persistent_dry = (
    sufficient_temporal
    &
    (
        wet_fraction
        <=
        PERSISTENT_DRY_FRACTION
    )
    &
    (~fit["good"])
)


transition = fit[
    "good"
]


ambiguous = (
    ~(
        persistent_wet
        |
        persistent_dry
        |
        transition
    )
)


class_map = np.zeros(
    shape,
    dtype=np.uint8
)


roi_rows = (
    roi_index
    //
    shape[
        1
    ]
)


roi_cols = (
    roi_index
    %
    shape[
        1
    ]
)


class_map[
    roi_rows[
        persistent_wet
    ],
    roi_cols[
        persistent_wet
    ]
] = 1


class_map[
    roi_rows[
        transition
    ],
    roi_cols[
        transition
    ]
] = 2


class_map[
    roi_rows[
        persistent_dry
    ],
    roi_cols[
        persistent_dry
    ]
] = 3


class_map[
    roi_rows[
        ambiguous
    ],
    roi_cols[
        ambiguous
    ]
] = 4


for cls, name in [
    (
        1,
        "persistente agua"
    ),
    (
        2,
        "transicion directa"
    ),
    (
        3,
        "persistente seco"
    ),
    (
        4,
        "ambiguo/insuficiente"
    ),
]:

    n = int(
        np.sum(
            class_map
            ==
            cls
        )
    )

    print(
        f"[CLASS {cls}] {name}: "
        f"{n:,} pix | "
        f"{n*pixel_area_m2/1e6:.4f} km2"
    )


                                                                               
                              
                                                                               

banner(
    "QC 4 - CORREDOR DIRECTAMENTE OBSERVADO"
)


transition_linear = roi_index[
    transition
]


transition_rows = (
    transition_linear
    //
    shape[
        1
    ]
)


transition_cols = (
    transition_linear
    %
    shape[
        1
    ]
)


transition_z = fit[
    "z"
][
    transition
]


transition_accuracy = fit[
    "accuracy"
][
    transition
]


corridor = np.zeros(
    shape,
    dtype=bool
)


corridor[
    transition_rows,
    transition_cols
] = True


threshold = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


threshold[
    transition_rows,
    transition_cols
] = transition_z.astype(
    np.float32
)


corridor_area_km2 = (
    corridor.sum()
    *
    pixel_area_m2
    /
    1e6
)


intersect_shore = (
    corridor
    &
    shore
)


print(
    "[DIRECT CORRIDOR] pixels =",
    f"{corridor.sum():,}"
)


print(
    "[DIRECT CORRIDOR] area =",
    f"{corridor_area_km2:.4f} km2"
)


print(
    "[DIRECT CORRIDOR ∩ 20-80] pixels =",
    f"{intersect_shore.sum():,}"
)


print(
    "[DIRECT CORRIDOR ∩ 20-80] area =",
    f"{intersect_shore.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[THRESHOLD Z] min/median/max =",
    f"{np.min(transition_z):.3f} / "
    f"{np.median(transition_z):.3f} / "
    f"{np.max(transition_z):.3f} m"
)


print(
    "[PIXEL ACCURACY] median/p10 =",
    f"{np.median(transition_accuracy):.3f} / "
    f"{np.percentile(transition_accuracy,10):.3f}"
)


                                                                               
                                             
                                                                               

banner(
    "QC 5 - AGUA ACTUAL SIN AMSCLAE"
)


gap_direct = (
    no_bathy
    &
    corridor
)


gap_no_direct = (
    no_bathy
    &
    (~corridor)
)


print(
    "[CURRENT GAP TOTAL] =",
    f"{no_bathy.sum():,} pix | "
    f"{no_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT GAP WITH DIRECT TRANSITION] =",
    f"{gap_direct.sum():,} pix | "
    f"{gap_direct.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT GAP WITHOUT DIRECT TRANSITION] =",
    f"{gap_no_direct.sum():,} pix | "
    f"{gap_no_direct.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[DIRECT GAP COVERAGE] =",
    f"{100*gap_direct.sum()/max(int(no_bathy.sum()),1):.2f}%"
)


                                                                               
                                          
                                                                               

banner(
    "QC 6 - ESTADOS DEL CORREDOR"
)


def transition_state(
    H_level
):

    state = np.zeros(
        shape,
        dtype=np.uint8
    )


    wet = (
        corridor
        &
        (
            H_level
            >=
            threshold
        )
    )


    dry = (
        corridor
        &
        (
            H_level
            <
            threshold
        )
    )


    state[
        wet
    ] = 1


    state[
        dry
    ] = 2


    return (
        state,
        wet,
        dry
    )


state_low, wet_low, dry_low = transition_state(
    H_MIN
)


state_anchor, wet_anchor, dry_anchor = transition_state(
    H_ANCHOR
)


state_high, wet_high, dry_high = transition_state(
    H_MAX
)


for name, wet, dry in [
    (
        "LOW",
        wet_low,
        dry_low
    ),
    (
        "ANCHOR",
        wet_anchor,
        dry_anchor
    ),
    (
        "HIGH",
        wet_high,
        dry_high
    ),
]:

    print(
        f"[{name}] wet corridor = "
        f"{wet.sum():,} pix | "
        f"{wet.sum()*pixel_area_m2/1e6:.4f} km2"
    )


    print(
        f"[{name}] dry corridor = "
        f"{dry.sum():,} pix | "
        f"{dry.sum()*pixel_area_m2/1e6:.4f} km2"
    )


                                                                               
                                                      
                                                                               

banner(
    "QC 7 - CAMBIO DE AREA DENTRO DEL CORREDOR"
)


levels = np.unique(
    np.concatenate(
        [
            H,
            np.array(
                [
                    H_ANCHOR
                ],
                dtype=float
            )
        ]
    )
)


levels.sort()


anchor_wet_n = int(
    wet_anchor.sum()
)


curve_rows = []


for H_level in levels:

    wet_n = int(
        np.sum(
            transition_z
            <=
            H_level
        )
    )


    delta_n = (
        wet_n
        -
        anchor_wet_n
    )


    curve_rows.append(
        {
            "lake_level_m":
                float(
                    H_level
                ),

            "wet_corridor_pixels":
                wet_n,

            "wet_corridor_area_km2":
                float(
                    wet_n
                    *
                    pixel_area_m2
                    /
                    1e6
                ),

            "delta_area_vs_anchor_km2":
                float(
                    delta_n
                    *
                    pixel_area_m2
                    /
                    1e6
                ),

            "interpretation":
                "direct_observed_transition_only",
        }
    )


curve_df = pd.DataFrame(
    curve_rows
)


print(
    curve_df[
        [
            "lake_level_m",
            "wet_corridor_area_km2",
            "delta_area_vs_anchor_km2",
        ]
    ].to_string(
        index=False,
        formatters={
            "lake_level_m":
                "{:.3f}".format,

            "wet_corridor_area_km2":
                "{:.4f}".format,

            "delta_area_vs_anchor_km2":
                "{:+.4f}".format,
        }
    )
)


                                                                               
                                    
                                                                               

banner(
    "QC 8 - CONSISTENCIA INTERNA EN PIXELES DE TRANSICION"
)


W_transition = W[
    transition,
    :
]


V_transition = V[
    transition,
    :
]


validation_rows = []


for j, row in exact.iterrows():

    valid_j = V_transition[
        :,
        j
    ]


    obs_j = W_transition[
        :,
        j
    ][
        valid_j
    ]


    pred_j = (
        H[
            j
        ]
        >=
        transition_z
    )[
        valid_j
    ]


    metrics = confusion_metrics(
        obs_j,
        pred_j
    )


    validation_rows.append(
        {
            "acquisition_date":
                row[
                    "acquisition_date"
                ].date(),

            "lake_level_m":
                float(
                    H[
                        j
                    ]
                ),

            **metrics,
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


print(
    "[DATES] =",
    len(
        validation_df
    )
)


print(
    "[ACCURACY] median/p10 =",
    f"{validation_df['accuracy'].median():.3f} / "
    f"{validation_df['accuracy'].quantile(0.10):.3f}"
)


print(
    "[BAL ACC] median/p10 =",
    f"{validation_df['balanced_accuracy'].median():.3f} / "
    f"{validation_df['balanced_accuracy'].quantile(0.10):.3f}"
)


print(
    "[WET IOU] median/p10 =",
    f"{validation_df['wet_iou'].median():.3f} / "
    f"{validation_df['wet_iou'].quantile(0.10):.3f}"
)


print(
    "[WET F1] median/p10 =",
    f"{validation_df['wet_f1'].median():.3f} / "
    f"{validation_df['wet_f1'].quantile(0.10):.3f}"
)


                  
                                                            
                                                                         
median_accuracy = float(
    validation_df[
        "accuracy"
    ].median()
)


                                                                               
               
                                                                               

banner(
    "ESCRITURA 16B4c2"
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


def write_uint(
    path,
    arr,
    description
):

    with rasterio.open(
        path,
        "w",
        **uint_profile
    ) as dst:

        dst.write(
            arr.astype(
                np.uint8
            ),
            1
        )

        dst.set_band_description(
            1,
            description
        )


write_uint(
    OUT_CORRIDOR,
    corridor,
    "corredor_transicion_directa_exacta"
)


write_uint(
    OUT_CLASS,
    class_map,
    "clase_temporal_exacta"
)


write_uint(
    OUT_STATE_LOW,
    state_low,
    "estado_corredor_Hmin"
)


write_uint(
    OUT_STATE_ANCHOR,
    state_anchor,
    "estado_corredor_Hanchor"
)


write_uint(
    OUT_STATE_HIGH,
    state_high,
    "estado_corredor_Hmax"
)


write_uint(
    OUT_GAP_DIRECT,
    gap_direct,
    "agua_actual_sin_AMSCLAE_con_transicion_directa"
)


write_uint(
    OUT_GAP_NO_DIRECT,
    gap_no_direct,
    "agua_actual_sin_AMSCLAE_sin_transicion_directa"
)


float_profile = profile.copy()


float_profile.update(
    dtype="float32",
    count=1,
    nodata=FLOAT_NODATA,
    compress="deflate",
)


with rasterio.open(
    OUT_THRESHOLD,
    "w",
    **float_profile
) as dst:

    dst.write(
        threshold,
        1
    )

    dst.set_band_description(
        1,
        "nivel_umbral_transicion_directa_m"
    )


validation_df.to_csv(
    OUT_VALIDATION,
    index=False,
    encoding="utf-8-sig"
)


curve_df.to_csv(
    OUT_CURVE,
    index=False,
    encoding="utf-8-sig"
)


summary_df = pd.DataFrame(
    [
        {
            "metric":
                "exact_dates",
            "value":
                len(exact),
            "unit":
                "count",
        },
        {
            "metric":
                "H_min_exact",
            "value":
                H_MIN,
            "unit":
                "m",
        },
        {
            "metric":
                "H_max_exact",
            "value":
                H_MAX,
            "unit":
                "m",
        },
        {
            "metric":
                "direct_transition_pixels",
            "value":
                int(
                    corridor.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "direct_transition_area",
            "value":
                corridor_area_km2,
            "unit":
                "km2",
        },
        {
            "metric":
                "shore_20_80_area",
            "value":
                float(
                    shore.sum()
                    *
                    pixel_area_m2
                    /
                    1e6
                ),
            "unit":
                "km2",
        },
        {
            "metric":
                "direct_gap_pixels",
            "value":
                int(
                    gap_direct.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "gap_without_direct_transition_pixels",
            "value":
                int(
                    gap_no_direct.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "transition_pixel_accuracy_median",
            "value":
                float(
                    np.median(
                        transition_accuracy
                    )
                ),
            "unit":
                "fraction",
        },
        {
            "metric":
                "per_date_accuracy_median",
            "value":
                median_accuracy,
            "unit":
                "fraction",
        },
    ]
)


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


                                                                               
             
                                                                               

fig = plt.figure(
    figsize=(12, 10)
)


ax = fig.add_subplot(
    111
)


display_map = np.zeros(
    shape,
    dtype=np.uint8
)


display_map[
    shore
] = 1


display_map[
    corridor
] = 2


display_map[
    gap_no_direct
] = 3


im = ax.imshow(
    display_map,
    vmin=0,
    vmax=3
)


ax.set_title(
    "16B4c2 - Corredor directamente observado\n"
    "1=franja Sentinel 20-80 | 2=transicion exacta | "
    "3=agua actual sin AMSCLAE y sin transicion directa"
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
    ]
)


cb.set_label(
    "Clase"
)


plt.tight_layout()


fig.savefig(
    OUT_FIG_CORRIDOR,
    dpi=180,
    bbox_inches="tight"
)


plt.close(
    fig
)


fig = plt.figure(
    figsize=(10, 6)
)


ax = fig.add_subplot(
    111
)


dates = pd.to_datetime(
    validation_df[
        "acquisition_date"
    ]
)


ax.plot(
    dates,
    validation_df[
        "accuracy"
    ],
    marker="o",
    label="Accuracy"
)


ax.plot(
    dates,
    validation_df[
        "balanced_accuracy"
    ],
    marker="o",
    label="Balanced accuracy"
)


ax.plot(
    dates,
    validation_df[
        "wet_iou"
    ],
    marker="o",
    label="Wet IoU"
)


ax.set_ylim(
    0,
    1.02
)


ax.set_ylabel(
    "Fraccion"
)


ax.set_title(
    "16B4c2 - Consistencia interna en transiciones directas"
)


ax.grid(
    alpha=0.25
)


ax.legend()


fig.autofmt_xdate()


plt.tight_layout()


fig.savefig(
    OUT_FIG_VALIDATION,
    dpi=180,
    bbox_inches="tight"
)


plt.close(
    fig
)


                                                                               
              
                                                                               

metadata = {
    "step":
        "16B4c2",

    "status":
        (
            "DIRECT_OBSERVED_CORRIDOR_CANDIDATE"
            if median_accuracy >= 0.75
            else
            "DIRECT_OBSERVED_CORRIDOR_NEEDS_REVIEW"
        ),

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "reason_for_replacement":
        (
            "16B4c IDW-expanded corridor was rejected because it covered the "
            "entire interpolated littoral surface and had insufficient internal "
            "consistency (median accuracy 0.645, balanced accuracy 0.618, wet IoU 0.567)."
        ),

    "method":
        (
            "Direct exact Sentinel/NASA temporal transitions only. "
            "No spatial IDW expansion is used to define the corridor."
        ),

    "exact_levels": {
        "n_dates":
            int(
                len(
                    exact
                )
            ),

        "min_m":
            H_MIN,

        "max_m":
            H_MAX,

        "anchor_m":
            H_ANCHOR,
    },

    "direct_corridor": {
        "pixels":
            int(
                corridor.sum()
            ),

        "area_km2":
            float(
                corridor_area_km2
            ),

        "threshold_min_m":
            float(
                np.min(
                    transition_z
                )
            ),

        "threshold_median_m":
            float(
                np.median(
                    transition_z
                )
            ),

        "threshold_max_m":
            float(
                np.max(
                    transition_z
                )
            ),

        "pixel_accuracy_median":
            float(
                np.median(
                    transition_accuracy
                )
            ),

        "pixel_accuracy_p10":
            float(
                np.percentile(
                    transition_accuracy,
                    10
                )
            ),
    },

    "current_water_without_amsclae": {
        "total_pixels":
            int(
                no_bathy.sum()
            ),

        "with_direct_transition_pixels":
            int(
                gap_direct.sum()
            ),

        "without_direct_transition_pixels":
            int(
                gap_no_direct.sum()
            ),
    },

    "internal_consistency": {
        "important_note":
            (
                "These are internal consistency metrics because the same exact "
                "Sentinel/NASA observations are used to estimate and test each "
                "pixel threshold. They are not independent predictive validation."
            ),

        "per_date_accuracy_median":
            median_accuracy,

        "per_date_balanced_accuracy_median":
            float(
                validation_df[
                    "balanced_accuracy"
                ].median()
            ),

        "per_date_wet_iou_median":
            float(
                validation_df[
                    "wet_iou"
                ].median()
            ),

        "per_date_wet_f1_median":
            float(
                validation_df[
                    "wet_f1"
                ].median()
            ),
    },

    "limitations": [
        (
            "The direct corridor only includes pixels that actually switched "
            "between wet and dry in exact observations with adequate support."
        ),
        (
            "Pixels without a direct transition are not assigned a shoreline "
            "threshold in this product."
        ),
        (
            "No extrapolation below the exact observed minimum level is performed."
        ),
        (
            "Future shoreline mapping below the observed range must use AMSCLAE "
            "where available and retain explicit extrapolation provenance."
        ),
    ],

    "sha256": {
        OUT_CORRIDOR.name:
            sha256_file(
                OUT_CORRIDOR
            ),

        OUT_THRESHOLD.name:
            sha256_file(
                OUT_THRESHOLD
            ),

        OUT_CLASS.name:
            sha256_file(
                OUT_CLASS
            ),

        OUT_VALIDATION.name:
            sha256_file(
                OUT_VALIDATION
            ),

        OUT_CURVE.name:
            sha256_file(
                OUT_CURVE
            ),
    },
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
    "DECISION QC 16B4c2"
)


print(
    "[QC] 16B4c anterior eliminado =",
    not OBSOLETE_B4C_DIR.exists()
)


print(
    "[QC] corredor directo =",
    f"{corridor.sum():,} pix | "
    f"{corridor_area_km2:.4f} km2"
)


print(
    "[QC] franja 20-80 =",
    f"{shore.sum():,} pix | "
    f"{shore.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[QC] agua actual sin AMSCLAE con transicion directa =",
    f"{gap_direct.sum():,} / {no_bathy.sum():,} pix"
)


print(
    "[QC] pixel accuracy mediana =",
    f"{np.median(transition_accuracy):.3f}"
)


print(
    "[QC] per-date accuracy mediana =",
    f"{median_accuracy:.3f}"
)


print(
    "[QC] per-date balanced accuracy mediana =",
    f"{validation_df['balanced_accuracy'].median():.3f}"
)


print(
    "[QC] per-date wet IoU mediana =",
    f"{validation_df['wet_iou'].median():.3f}"
)


print()


if median_accuracy >= 0.75:

    print(
        "[PASS DIRECT CORRIDOR]"
    )


    print(
        "El corredor queda definido por transiciones Sentinel/NASA realmente observadas."
    )


    print(
        "No se uso IDW para ampliar artificialmente la zona de oscilacion."
    )


    print()


    print(
        "[NEXT] 16B4d:"
    )


    print(
        "usar este corredor directo dentro del rango observado y AMSCLAE "
        "solo para retiros por debajo del minimo observado, marcando "
        "cada pixel como OBSERVADO, AMSCLAE o DESCONOCIDO."
    )


else:

    print(
        "[NO PASS]"
    )


    print(
        "La consistencia temporal agregada sigue siendo insuficiente. "
        "No avanzar a escenarios futuros."
    )


banner(
    "FIN PASO 16B4c2"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
