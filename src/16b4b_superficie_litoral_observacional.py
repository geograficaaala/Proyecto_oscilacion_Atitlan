                       


































































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio

from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

DOWNLOADS = ROOT / "data" / "external"


B4_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
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


INTERP_STACK_NAME = (
    "atitlan_16A2g_nivel_bajo_interpolado_01_stack_WV.tif"
)


INTERP_MANIFEST_NAME = (
    "atitlan_16A2g_nivel_bajo_interpolado_02_manifiesto.csv"
)


OUT_DIR = (
    B4_DIR
    / "superficie_litoral_16B4b"
)


OUT_Z = (
    OUT_DIR
    / "16B4b_superficie_litoral_Z_absoluta_m.tif"
)


OUT_SUPPORT = (
    OUT_DIR
    / "16B4b_soporte_superficie_litoral.tif"
)


OUT_DISTANCE = (
    OUT_DIR
    / "16B4b_distancia_restriccion_estable_m.tif"
)


OUT_SPREAD = (
    OUT_DIR
    / "16B4b_dispersion_local_m.tif"
)


OUT_GAP_SUPPORTED = (
    OUT_DIR
    / "16B4b_agua_actual_sin_AMSCLAE_con_soporte.tif"
)


OUT_GAP_UNSUPPORTED = (
    OUT_DIR
    / "16B4b_agua_actual_sin_AMSCLAE_sin_soporte.tif"
)


OUT_SUMMARY = (
    OUT_DIR
    / "16B4b_resumen.csv"
)


OUT_CV = (
    OUT_DIR
    / "16B4b_validacion_cruzada.csv"
)


OUT_METADATA = (
    OUT_DIR
    / "16B4b_metadata.json"
)


OUT_FIG_Z = (
    OUT_DIR
    / "16B4b_superficie_litoral_Z.png"
)


OUT_FIG_GAPS = (
    OUT_DIR
    / "16B4b_soporte_agua_actual_sin_AMSCLAE.png"
)


                                                                               
               
                                                                               

                                                                   
MIN_INTERP_COVERAGE = 85.0
DILATION_PIXELS = 3

MIN_VALID = 10
MIN_WET = 2
MIN_DRY = 2
MIN_LEVEL_SPAN_M = 0.50
MIN_ACCURACY = 0.80

MAX_INTERP_SENSITIVITY_M = 0.20
MAX_COMBINED_SHIFT_M = 0.25


                     
COASTAL_EXPANSION_M = 120.0

MAX_DISTANCE_M = 120.0

K_NEIGHBORS = 8

MIN_NEIGHBORS = 3

IDW_POWER = 2.0

IDW_EPS_M = 0.25


                                                                               
              
                                                                               

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
            DOWNLOADS.rglob(
                name
            )
        )


    if ROOT.exists():

        hits.extend(
            ROOT.rglob(
                name
            )
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

    with rasterio.open(
        path
    ) as ds:

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
                row[
                    column
                ]
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


            sources[
                name
            ] = (
                p,
                lookups[p][name]
            )


    W = np.zeros(
        (
            len(
                pixel_index
            ),
            len(
                manifest
            ),
        ),
        dtype=bool
    )


    V = np.zeros_like(
        W
    )


    datasets = {
        p: rasterio.open(
            p
        )
        for p in raster_paths
    }


    try:

        for j, row in manifest.iterrows():

            wb = str(
                row[
                    "water_band"
                ]
            )


            vb = str(
                row[
                    "valid_band"
                ]
            )


            pw, iw = sources[
                wb
            ]


            pv, iv = sources[
                vb
            ]


            water = (
                datasets[pw]
                .read(
                    iw
                )
                .ravel()[
                    pixel_index
                ]
                >
                0
            )


            valid = (
                datasets[pv]
                .read(
                    iv
                )
                .ravel()[
                    pixel_index
                ]
                >
                0
            )


            W[
                :,
                j
            ] = (
                water
                &
                valid
            )


            V[
                :,
                j
            ] = valid


    finally:

        for ds in datasets.values():

            ds.close()


    return (
        W,
        V
    )


def fit_monotonic(
    W,
    V,
    H
):

    levels = np.unique(
        H
    )


    levels.sort()


    if len(
        levels
    ) < 2:

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
            len(
                candidates
            ),
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


        errors[
            :,
            k
        ] = (
            (
                (
                    W
                    !=
                    predicted[
                        None,
                        :
                    ]
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
        min_error[
            :,
            None
        ]
    )


    z_est = (
        winners
        *
        candidates[
            None,
            :
        ]
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
        >
        0
    )


    accuracy[
        usable
    ] = (
        1.0
        -
        min_error[
            usable
        ]
        /
        n_valid[
            usable
        ]
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

        m = V[
            i
        ]


        if m.any():

            vals = H[
                m
            ]


            h_min[
                i
            ] = vals.min()


            h_max[
                i
            ] = vals.max()


    span = (
        h_max
        -
        h_min
    )


    good = (
        (
            n_valid
            >=
            MIN_VALID
        )
        &
        (
            n_wet
            >=
            MIN_WET
        )
        &
        (
            n_dry
            >=
            MIN_DRY
        )
        &
        (
            span
            >=
            MIN_LEVEL_SPAN_M
        )
        &
        (
            accuracy
            >=
            MIN_ACCURACY
        )
    )


    return {
        "z":
            z_est,

        "good":
            good,

        "accuracy":
            accuracy,
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


def idw_predict(
    tree,
    source_z,
    target_xy,
    k=K_NEIGHBORS,
    max_distance=MAX_DISTANCE_M,
    min_neighbors=MIN_NEIGHBORS
):

    dists, idxs = tree.query(
        target_xy,
        k=k,
        distance_upper_bound=max_distance
    )


    if dists.ndim == 1:

        dists = dists[
            :,
            None
        ]


        idxs = idxs[
            :,
            None
        ]


    good_neighbor = (
        np.isfinite(
            dists
        )
        &
        (
            idxs
            <
            len(
                source_z
            )
        )
    )


    n_neighbors = good_neighbor.sum(
        axis=1
    )


    safe_idx = np.where(
        good_neighbor,
        idxs,
        0
    )


    neighbor_z = source_z[
        safe_idx
    ]


    weights = np.zeros_like(
        dists,
        dtype=np.float64
    )


    weights[
        good_neighbor
    ] = (
        1.0
        /
        (
            dists[
                good_neighbor
            ]
            +
            IDW_EPS_M
        )
        **
        IDW_POWER
    )


    weight_sum = weights.sum(
        axis=1
    )


    supported = (
        (
            n_neighbors
            >=
            min_neighbors
        )
        &
        (
            weight_sum
            >
            0
        )
    )


    pred = np.full(
        len(
            target_xy
        ),
        np.nan,
        dtype=np.float64
    )


    spread = np.full(
        len(
            target_xy
        ),
        np.nan,
        dtype=np.float64
    )


    nearest = np.full(
        len(
            target_xy
        ),
        np.inf,
        dtype=np.float64
    )


    if len(
        target_xy
    ) > 0:

        nearest = dists[
            :,
            0
        ].astype(
            np.float64
        )


    if supported.any():

        pred[
            supported
        ] = (
            (
                weights[
                    supported
                ]
                *
                neighbor_z[
                    supported
                ]
            ).sum(
                axis=1
            )
            /
            weight_sum[
                supported
            ]
        )


        centered = (
            neighbor_z[
                supported
            ]
            -
            pred[
                supported
            ][
                :,
                None
            ]
        )


        variance = (
            (
                weights[
                    supported
                ]
                *
                centered
                **
                2
            ).sum(
                axis=1
            )
            /
            weight_sum[
                supported
            ]
        )


        spread[
            supported
        ] = np.sqrt(
            np.maximum(
                variance,
                0.0
            )
        )


    return (
        pred,
        spread,
        nearest,
        n_neighbors,
        supported,
    )


                                                                               
                 
                                                                               

banner(
    "PASO 16B4b - LOCALIZACION"
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


exact_paths = [
    find_file(
        name
    )
    for name in EXACT_STACK_NAMES
]


exact_manifest_path = find_file(
    EXACT_MANIFEST_NAME
)


interp_stack_path = find_file(
    INTERP_STACK_NAME
)


interp_manifest_path = find_file(
    INTERP_MANIFEST_NAME
)


for p in (
    [
        PATH_LAKE_CURRENT,
        PATH_NO_BATHY,
        PATH_SHORE,
    ]
    +
    exact_paths
    +
    [
        exact_manifest_path,
        interp_stack_path,
        interp_manifest_path,
    ]
):

    print(
        "[FILE]",
        p
    )


                                                                               
                        
                                                                               

banner(
    "QC 1 - GRILLA MAESTRA"
)


with rasterio.open(
    PATH_LAKE_CURRENT
) as ds:

    lake_current = (
        ds.read(
            1
        )
        >
        0
    )


    profile = ds.profile.copy()

    shape = ds.shape

    transform = ds.transform

    crs = ds.crs


with rasterio.open(
    PATH_NO_BATHY
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
            "16B4a agua sin batimetria no coincide con la grilla maestra."
        )


    no_bathy = (
        ds.read(
            1
        )
        >
        0
    )


with rasterio.open(
    PATH_SHORE
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
            "16B4a corredor litoral no coincide con la grilla maestra."
        )


    shore = (
        ds.read(
            1
        )
        >
        0
    )


pixel_m = abs(
    transform.a
)


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


print(
    "[CURRENT LAKE] =",
    f"{lake_current.sum():,} pix | "
    f"{lake_current.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT WATER WITHOUT AMSCLAE] =",
    f"{no_bathy.sum():,} pix | "
    f"{no_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                                       
                                                                               

banner(
    "QC 2 - RESTRICCIONES SENTINEL/NASA"
)


shore_roi = binary_dilation(
    shore,
    iterations=DILATION_PIXELS
)


shore_index = np.flatnonzero(
    shore_roi.ravel()
)


exact = pd.read_csv(
    exact_manifest_path
).reset_index(
    drop=True
)


interp = pd.read_csv(
    interp_manifest_path
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


interp[
    "acquisition_date"
] = pd.to_datetime(
    interp[
        "acquisition_date"
    ]
)


interp_lookup = lookup(
    interp_stack_path
)


lake_flat = lake_current.ravel()


coverage = []


with rasterio.open(
    interp_stack_path
) as ds:

    for _, row in interp.iterrows():

        vb = str(
            row[
                "valid_band"
            ]
        )


        valid_band = (
            ds.read(
                interp_lookup[
                    vb
                ]
            )
            .ravel()
            >
            0
        )


        pct = (
            100.0
            *
            np.sum(
                valid_band
                &
                lake_flat
            )
            /
            lake_current.sum()
        )


        coverage.append(
            pct
        )


interp[
    "coverage_pct"
] = coverage


interp[
    "usable"
] = (
    interp[
        "coverage_pct"
    ]
    >=
    MIN_INTERP_COVERAGE
)


print(
    "[INTERP] utilizables =",
    int(
        interp[
            "usable"
        ].sum()
    )
)


W_exact, V_exact = load_wv(
    exact,
    exact_paths,
    shore_index
)


H_exact = (
    exact[
        "lake_level_m"
    ]
    .to_numpy(
        float
    )
)


fit_exact = fit_monotonic(
    W_exact,
    V_exact,
    H_exact
)


W_interp_all, V_interp_all = load_wv(
    interp,
    [
        interp_stack_path
    ],
    shore_index
)


usable_interp = (
    interp[
        "usable"
    ]
    .to_numpy(
        bool
    )
)


W_interp = W_interp_all[
    :,
    usable_interp
]


V_interp = V_interp_all[
    :,
    usable_interp
]


H_interp = (
    interp.loc[
        usable_interp,
        "lake_level_interp_m"
    ]
    .to_numpy(
        float
    )
)


P90_interp = (
    interp.loc[
        usable_interp,
        "p90_error_proxy_m"
    ]
    .to_numpy(
        float
    )
)


W_comb = np.concatenate(
    [
        W_exact,
        W_interp,
    ],
    axis=1
)


V_comb = np.concatenate(
    [
        V_exact,
        V_interp,
    ],
    axis=1
)


H_comb = np.concatenate(
    [
        H_exact,
        H_interp,
    ]
)


fit_comb = fit_monotonic(
    W_comb,
    V_comb,
    H_comb
)


fit_low = fit_monotonic(
    W_comb,
    V_comb,
    np.concatenate(
        [
            H_exact,
            H_interp
            -
            P90_interp,
        ]
    )
)


fit_high = fit_monotonic(
    W_comb,
    V_comb,
    np.concatenate(
        [
            H_exact,
            H_interp
            +
            P90_interp,
        ]
    )
)


three_z = np.column_stack(
    [
        fit_low[
            "z"
        ],
        fit_comb[
            "z"
        ],
        fit_high[
            "z"
        ],
    ]
)


interp_sensitivity = (
    np.nanmax(
        three_z,
        axis=1
    )
    -
    np.nanmin(
        three_z,
        axis=1
    )
)


combined_shift = np.abs(
    fit_comb[
        "z"
    ]
    -
    fit_exact[
        "z"
    ]
)


stable = (
    fit_exact[
        "good"
    ]
    &
    fit_comb[
        "good"
    ]
    &
    (
        interp_sensitivity
        <=
        MAX_INTERP_SENSITIVITY_M
    )
    &
    (
        combined_shift
        <=
        MAX_COMBINED_SHIFT_M
    )
)


print(
    "[EXACT GOOD] =",
    f"{fit_exact['good'].sum():,}"
)


print(
    "[COMBINED GOOD] =",
    f"{fit_comb['good'].sum():,}"
)


print(
    "[STABLE] =",
    f"{stable.sum():,}"
)


if fit_exact[
    "good"
].sum() != 6181:

    raise RuntimeError(
        "No se reprodujeron exactamente los 6,181 EXACT GOOD."
    )


if fit_comb[
    "good"
].sum() != 7853:

    raise RuntimeError(
        "No se reprodujeron exactamente los 7,853 COMBINED GOOD."
    )


if stable.sum() != 5103:

    raise RuntimeError(
        "No se reprodujeron exactamente las 5,103 restricciones estables."
    )


                                                                               
                                        
                                                                               

stable_linear = shore_index[
    stable
]


stable_z = (
    fit_comb[
        "z"
    ][
        stable
    ]
    .astype(
        np.float64
    )
)


stable_rows = (
    stable_linear
    //
    shape[
        1
    ]
)


stable_cols = (
    stable_linear
    %
    shape[
        1
    ]
)


stable_x = (
    transform.c
    +
    (
        stable_cols
        +
        0.5
    )
    *
    transform.a
)


stable_y = (
    transform.f
    +
    (
        stable_rows
        +
        0.5
    )
    *
    transform.e
)


stable_xy = np.column_stack(
    [
        stable_x,
        stable_y,
    ]
)


stable_map = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


stable_map[
    stable_rows,
    stable_cols
] = stable_z.astype(
    np.float32
)


Z_MIN_OBS = float(
    np.min(
        stable_z
    )
)


Z_MAX_OBS = float(
    np.max(
        stable_z
    )
)


print(
    "[STABLE Z] min/max =",
    f"{Z_MIN_OBS:.3f} / "
    f"{Z_MAX_OBS:.3f} m"
)


                                                                               
                                  
                                                                               

banner(
    "QC 3 - DOMINIO LITORAL"
)


n_expand = int(
    round(
        COASTAL_EXPANSION_M
        /
        pixel_m
    )
)


coastal_domain = binary_dilation(
    shore,
    iterations=n_expand
)


print(
    "[COAST DOMAIN] expansion =",
    COASTAL_EXPANSION_M,
    "m"
)


print(
    "[COAST DOMAIN] pixels =",
    f"{coastal_domain.sum():,}"
)


print(
    "[COAST DOMAIN] area =",
    f"{coastal_domain.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                          
                                                                               

banner(
    "QC 4 - SUPERFICIE LITORAL IDW"
)


target_rows, target_cols = np.where(
    coastal_domain
)


target_x = (
    transform.c
    +
    (
        target_cols
        +
        0.5
    )
    *
    transform.a
)


target_y = (
    transform.f
    +
    (
        target_rows
        +
        0.5
    )
    *
    transform.e
)


target_xy = np.column_stack(
    [
        target_x,
        target_y,
    ]
)


tree = cKDTree(
    stable_xy
)


(
    z_pred,
    local_spread,
    nearest_distance,
    n_neighbors,
    supported,
) = idw_predict(
    tree,
    stable_z,
    target_xy
)


                                                                   
linear_target = (
    target_rows
    *
    shape[
        1
    ]
    +
    target_cols
)


stable_position = {
    int(
        idx
    ):
        float(
            z
        )
    for idx, z
    in zip(
        stable_linear,
        stable_z
    )
}


is_direct = np.zeros(
    len(
        target_rows
    ),
    dtype=bool
)


for i, lin in enumerate(
    linear_target
):

    if int(
        lin
    ) in stable_position:

        is_direct[
            i
        ] = True


        z_pred[
            i
        ] = stable_position[
            int(
                lin
            )
        ]


        local_spread[
            i
        ] = 0.0


        nearest_distance[
            i
        ] = 0.0


        supported[
            i
        ] = True


                                                                  
within_observed_range = (
    supported
    &
    np.isfinite(
        z_pred
    )
    &
    (
        z_pred
        >=
        Z_MIN_OBS
    )
    &
    (
        z_pred
        <=
        Z_MAX_OBS
    )
)


supported = (
    supported
    &
    within_observed_range
)


z_surface = np.full(
    shape,
    -9999.0,
    dtype=np.float32
)


spread_surface = np.full(
    shape,
    -9999.0,
    dtype=np.float32
)


distance_surface = np.full(
    shape,
    -9999.0,
    dtype=np.float32
)


support_class = np.zeros(
    shape,
    dtype=np.uint8
)


rr = target_rows[
    supported
]


cc = target_cols[
    supported
]


z_surface[
    rr,
    cc
] = z_pred[
    supported
].astype(
    np.float32
)


spread_surface[
    rr,
    cc
] = local_spread[
    supported
].astype(
    np.float32
)


distance_surface[
    rr,
    cc
] = nearest_distance[
    supported
].astype(
    np.float32
)


         
support_class[
    target_rows[
        supported
        &
        is_direct
    ],
    target_cols[
        supported
        &
        is_direct
    ]
] = 1


m = (
    supported
    &
    (~is_direct)
    &
    (
        nearest_distance
        <=
        20.0
    )
)


support_class[
    target_rows[
        m
    ],
    target_cols[
        m
    ]
] = 2


m = (
    supported
    &
    (~is_direct)
    &
    (
        nearest_distance
        >
        20.0
    )
    &
    (
        nearest_distance
        <=
        60.0
    )
)


support_class[
    target_rows[
        m
    ],
    target_cols[
        m
    ]
] = 3


m = (
    supported
    &
    (~is_direct)
    &
    (
        nearest_distance
        >
        60.0
    )
    &
    (
        nearest_distance
        <=
        120.0
    )
)


support_class[
    target_rows[
        m
    ],
    target_cols[
        m
    ]
] = 4


print(
    "[SURFACE] supported pixels =",
    f"{supported.sum():,}"
)


print(
    "[SURFACE] supported area =",
    f"{supported.sum()*pixel_area_m2/1e6:.4f} km2"
)


for cls, label_name in [
    (
        1,
        "direct stable"
    ),
    (
        2,
        "<=20m"
    ),
    (
        3,
        "20-60m"
    ),
    (
        4,
        "60-120m"
    ),
]:

    n = int(
        np.sum(
            support_class
            ==
            cls
        )
    )


    print(
        f"[SUPPORT {cls}] {label_name}: {n:,} pix"
    )


                                                                               
                            
                                                                               

banner(
    "QC 5 - AGUA ACTUAL SIN AMSCLAE"
)


surface_valid = (
    support_class
    >
    0
)


gap_supported = (
    no_bathy
    &
    surface_valid
)


gap_unsupported = (
    no_bathy
    &
    (~surface_valid)
)


print(
    "[GAP TOTAL] =",
    f"{no_bathy.sum():,} pix | "
    f"{no_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[GAP SUPPORTED] =",
    f"{gap_supported.sum():,} pix | "
    f"{gap_supported.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[GAP UNSUPPORTED] =",
    f"{gap_unsupported.sum():,} pix | "
    f"{gap_unsupported.sum()*pixel_area_m2/1e6:.4f} km2"
)


gap_support_pct = (
    100.0
    *
    gap_supported.sum()
    /
    max(
        int(
            no_bathy.sum()
        ),
        1
    )
)


print(
    "[GAP COVERAGE] =",
    f"{gap_support_pct:.2f}%"
)


if gap_supported.any():

    gap_z = z_surface[
        gap_supported
    ].astype(
        float
    )


    gap_dist = distance_surface[
        gap_supported
    ].astype(
        float
    )


    gap_spread = spread_surface[
        gap_supported
    ].astype(
        float
    )


    print(
        "[GAP Z] min/median/max =",
        f"{np.min(gap_z):.3f} / "
        f"{np.median(gap_z):.3f} / "
        f"{np.max(gap_z):.3f} m"
    )


    print(
        "[GAP DIST] median/p90/max =",
        f"{np.median(gap_dist):.1f} / "
        f"{np.percentile(gap_dist,90):.1f} / "
        f"{np.max(gap_dist):.1f} m"
    )


    print(
        "[GAP LOCAL SPREAD] median/p90 =",
        f"{np.median(gap_spread):.3f} / "
        f"{np.percentile(gap_spread,90):.3f} m"
    )


                                                                               
                                       
                                                                               

banner(
    "QC 6 - VALIDACION CRUZADA DE LA INTERPOLACION"
)


                                                                    
                  
cv_tree = cKDTree(
    stable_xy
)


cv_k = min(
    K_NEIGHBORS
    +
    1,
    len(
        stable_z
    )
)


cv_dist, cv_idx = cv_tree.query(
    stable_xy,
    k=cv_k,
    distance_upper_bound=MAX_DISTANCE_M
)


if cv_dist.ndim == 1:

    cv_dist = cv_dist[
        :,
        None
    ]


    cv_idx = cv_idx[
        :,
        None
    ]


                                                          
cv_pred = np.full(
    len(
        stable_z
    ),
    np.nan,
    dtype=np.float64
)


cv_n = np.zeros(
    len(
        stable_z
    ),
    dtype=np.int16
)


cv_nearest = np.full(
    len(
        stable_z
    ),
    np.nan,
    dtype=np.float64
)


for i in range(
    len(
        stable_z
    )
):

    d = cv_dist[
        i
    ]


    ind = cv_idx[
        i
    ]


    valid_neighbor = (
        np.isfinite(
            d
        )
        &
        (
            ind
            <
            len(
                stable_z
            )
        )
        &
        (
            ind
            !=
            i
        )
    )


    d = d[
        valid_neighbor
    ]


    ind = ind[
        valid_neighbor
    ]


    if len(
        d
    ) < MIN_NEIGHBORS:

        continue


    d = d[
        :
        K_NEIGHBORS
    ]


    ind = ind[
        :
        K_NEIGHBORS
    ]


    w = (
        1.0
        /
        (
            d
            +
            IDW_EPS_M
        )
        **
        IDW_POWER
    )


    cv_pred[
        i
    ] = (
        np.sum(
            w
            *
            stable_z[
                ind
            ]
        )
        /
        np.sum(
            w
        )
    )


    cv_n[
        i
    ] = len(
        d
    )


    cv_nearest[
        i
    ] = np.min(
        d
    )


cv_valid = np.isfinite(
    cv_pred
)


cv_resid = (
    cv_pred[
        cv_valid
    ]
    -
    stable_z[
        cv_valid
    ]
)


cv_abs = np.abs(
    cv_resid
)


cv_mae = float(
    np.mean(
        cv_abs
    )
)


cv_rmse = float(
    np.sqrt(
        np.mean(
            cv_resid
            **
            2
        )
    )
)


cv_med = float(
    np.median(
        cv_abs
    )
)


cv_p90 = float(
    np.percentile(
        cv_abs,
        90
    )
)


print(
    "[CV] n =",
    f"{cv_valid.sum():,}"
)


print(
    "[CV] MAE =",
    f"{cv_mae:.3f} m"
)


print(
    "[CV] RMSE =",
    f"{cv_rmse:.3f} m"
)


print(
    "[CV] median abs =",
    f"{cv_med:.3f} m"
)


print(
    "[CV] p90 abs =",
    f"{cv_p90:.3f} m"
)


cv_df = pd.DataFrame(
    {
        "stable_z_m":
            stable_z[
                cv_valid
            ],

        "predicted_z_m":
            cv_pred[
                cv_valid
            ],

        "residual_m":
            cv_resid,

        "abs_residual_m":
            cv_abs,

        "nearest_neighbor_m":
            cv_nearest[
                cv_valid
            ],

        "n_neighbors":
            cv_n[
                cv_valid
            ],
    }
)


                                                                               
               
                                                                               

banner(
    "ESCRITURA 16B4b"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


float_profile = profile.copy()


float_profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="deflate",
)


with rasterio.open(
    OUT_Z,
    "w",
    **float_profile
) as dst:

    dst.write(
        z_surface,
        1
    )


    dst.set_band_description(
        1,
        "elevacion_litoral_observacional_m"
    )


with rasterio.open(
    OUT_DISTANCE,
    "w",
    **float_profile
) as dst:

    dst.write(
        distance_surface,
        1
    )


    dst.set_band_description(
        1,
        "distancia_restriccion_estable_m"
    )


with rasterio.open(
    OUT_SPREAD,
    "w",
    **float_profile
) as dst:

    dst.write(
        spread_surface,
        1
    )


    dst.set_band_description(
        1,
        "dispersion_local_IDW_m"
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
    OUT_SUPPORT,
    support_class,
    "clase_soporte_superficie_litoral"
)


write_uint(
    OUT_GAP_SUPPORTED,
    gap_supported,
    "agua_actual_sin_AMSCLAE_con_superficie_litoral"
)


write_uint(
    OUT_GAP_UNSUPPORTED,
    gap_unsupported,
    "agua_actual_sin_AMSCLAE_sin_superficie_litoral"
)


summary_df = pd.DataFrame(
    [
        {
            "metric":
                "stable_constraints",

            "value":
                int(
                    len(
                        stable_z
                    )
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "observed_z_min",

            "value":
                Z_MIN_OBS,

            "unit":
                "m",
        },
        {
            "metric":
                "observed_z_max",

            "value":
                Z_MAX_OBS,

            "unit":
                "m",
        },
        {
            "metric":
                "surface_supported_pixels",

            "value":
                int(
                    surface_valid.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_water_without_amsclae_pixels",

            "value":
                int(
                    no_bathy.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_gap_supported_pixels",

            "value":
                int(
                    gap_supported.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_gap_unsupported_pixels",

            "value":
                int(
                    gap_unsupported.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_gap_support_pct",

            "value":
                gap_support_pct,

            "unit":
                "percent",
        },
        {
            "metric":
                "cv_mae",

            "value":
                cv_mae,

            "unit":
                "m",
        },
        {
            "metric":
                "cv_rmse",

            "value":
                cv_rmse,

            "unit":
                "m",
        },
        {
            "metric":
                "cv_p90_abs",

            "value":
                cv_p90,

            "unit":
                "m",
        },
    ]
)


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


cv_df.to_csv(
    OUT_CV,
    index=False,
    encoding="utf-8-sig"
)


                                                                               
             
                                                                               

z_plot = np.where(
    z_surface
    !=
    -9999.0,
    z_surface,
    np.nan
)


fig = plt.figure(
    figsize=(12, 10)
)


ax = fig.add_subplot(
    111
)


im = ax.imshow(
    z_plot
)


ax.set_title(
    "16B4b - Superficie litoral observacional Sentinel/NASA\n"
    "Independiente de AMSCLAE"
)


ax.set_axis_off()


cb = plt.colorbar(
    im,
    ax=ax,
    shrink=0.75
)


cb.set_label(
    "Elevacion litoral (m)"
)


plt.tight_layout()


fig.savefig(
    OUT_FIG_Z,
    dpi=180,
    bbox_inches="tight"
)


plt.close(
    fig
)


gap_map = np.zeros(
    shape,
    dtype=np.uint8
)


gap_map[
    gap_supported
] = 1


gap_map[
    gap_unsupported
] = 2


fig = plt.figure(
    figsize=(12, 10)
)


ax = fig.add_subplot(
    111
)


im = ax.imshow(
    gap_map,
    vmin=0,
    vmax=2
)


ax.set_title(
    "16B4b - Agua actual sin AMSCLAE\n"
    "1=con soporte Sentinel/NASA | 2=sin soporte"
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
    ]
)


cb.set_label(
    "Clase"
)


plt.tight_layout()


fig.savefig(
    OUT_FIG_GAPS,
    dpi=180,
    bbox_inches="tight"
)


plt.close(
    fig
)


                                                                               
              
                                                                               

metadata = {
    "step":
        "16B4b",

    "status":
        "OBSERVATIONAL_LITTORAL_SURFACE_CANDIDATE",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "principle":
        (
            "The observational littoral surface is built only from Sentinel/NASA "
            "shoreline constraints and is independent of the AMSCLAE bathymetry edge."
        ),

    "stable_constraints": {
        "n":
            int(
                len(
                    stable_z
                )
            ),

        "z_min_m":
            Z_MIN_OBS,

        "z_max_m":
            Z_MAX_OBS,
    },

    "interpolation": {
        "method":
            "local IDW",

        "k_neighbors":
            K_NEIGHBORS,

        "min_neighbors":
            MIN_NEIGHBORS,

        "max_distance_m":
            MAX_DISTANCE_M,

        "power":
            IDW_POWER,

        "coastal_domain_expansion_m":
            COASTAL_EXPANSION_M,
    },

    "current_water_without_amsclae": {
        "total_pixels":
            int(
                no_bathy.sum()
            ),

        "supported_pixels":
            int(
                gap_supported.sum()
            ),

        "unsupported_pixels":
            int(
                gap_unsupported.sum()
            ),

        "supported_pct":
            float(
                gap_support_pct
            ),
    },

    "cross_validation": {
        "n":
            int(
                cv_valid.sum()
            ),

        "mae_m":
            cv_mae,

        "rmse_m":
            cv_rmse,

        "median_abs_m":
            cv_med,

        "p90_abs_m":
            cv_p90,
    },

    "important_limitations": [
        (
            "This is a shoreline-elevation surface, not a replacement lake-wide bathymetry."
        ),
        (
            "No AMSCLAE depth value or AMSCLAE edge condition is used in the interpolation."
        ),
        (
            "Values are kept within the directly observed stable Sentinel/NASA elevation range."
        ),
        (
            "Pixels of current Sentinel water without sufficient local stable constraints remain "
            "explicitly unsupported rather than being invented."
        ),
        (
            "Future levels below the directly observed shoreline range require a separate "
            "extrapolation treatment and must be flagged as such."
        ),
    ],

    "sha256": {
        OUT_Z.name:
            sha256_file(
                OUT_Z
            ),

        OUT_SUPPORT.name:
            sha256_file(
                OUT_SUPPORT
            ),

        OUT_DISTANCE.name:
            sha256_file(
                OUT_DISTANCE
            ),

        OUT_SPREAD.name:
            sha256_file(
                OUT_SPREAD
            ),

        OUT_GAP_SUPPORTED.name:
            sha256_file(
                OUT_GAP_SUPPORTED
            ),

        OUT_GAP_UNSUPPORTED.name:
            sha256_file(
                OUT_GAP_UNSUPPORTED
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
    "DECISION QC 16B4b"
)


print(
    "[QC] restricciones estables =",
    f"{len(stable_z):,}"
)


print(
    "[QC] rango vertical directo =",
    f"{Z_MIN_OBS:.3f} - "
    f"{Z_MAX_OBS:.3f} m"
)


print(
    "[QC] agua actual sin AMSCLAE =",
    f"{no_bathy.sum():,} pix"
)


print(
    "[QC] de esa agua, con superficie Sentinel/NASA =",
    f"{gap_supported.sum():,} pix "
    f"({gap_support_pct:.2f}%)"
)


print(
    "[QC] sin soporte suficiente =",
    f"{gap_unsupported.sum():,} pix"
)


print(
    "[QC] CV MAE / RMSE / p90 =",
    f"{cv_mae:.3f} / "
    f"{cv_rmse:.3f} / "
    f"{cv_p90:.3f} m"
)


print()


print(
    "[PASS CANDIDATE]"
)


print(
    "Se construyo una superficie litoral observacional independiente de AMSCLAE."
)


print(
    "La mascara del lago actual se conserva completa aunque falte batimetria."
)


print(
    "Los pixeles sin soporte permanecen explicitamente SIN estimar."
)


print()


print(
    "[NEXT] 16B4c:"
)


print(
    "usar esta superficie para reconstruir posiciones de costa por nivel, "
    "validarlas contra las observaciones Sentinel/NASA y definir el corredor "
    "de oscilacion espacial."
)


banner(
    "FIN PASO 16B4b"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
