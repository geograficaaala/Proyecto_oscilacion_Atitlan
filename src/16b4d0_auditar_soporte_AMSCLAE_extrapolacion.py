                       


































































































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

PATH_AMSCLAE = ROOT / "data" / "external" / "batimetria_amsclae.tif"

B4_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
)

C2_DIR = (
    B4_DIR
    / "corredor_directo_16B4c2"
)

PATH_CURRENT_LAKE = (
    B4_DIR
    / "16B4a_mascara_lago_actual_Sentinel_50pct.tif"
)

PATH_DIRECT_CORRIDOR = (
    C2_DIR
    / "16B4c2_corredor_transicion_directa.tif"
)

PATH_DIRECT_THRESHOLD = (
    C2_DIR
    / "16B4c2_nivel_umbral_directo_m.tif"
)

OUT_DIR = (
    B4_DIR
    / "anclaje_AMSCLAE_16B4d0"
)

OUT_SUPPORT = (
    OUT_DIR
    / "16B4d0_soporte_extrapolacion_AMSCLAE.tif"
)

OUT_RESIDUAL = (
    OUT_DIR
    / "16B4d0_residual_directo_Sentinel_menos_AMSCLAE_m.tif"
)

OUT_LOCAL_FRACTION = (
    OUT_DIR
    / "16B4d0_fraccion_compatible_local.tif"
)

OUT_LOCAL_MEDABS = (
    OUT_DIR
    / "16B4d0_mediana_abs_residual_local_m.tif"
)

OUT_DISTANCE = (
    OUT_DIR
    / "16B4d0_distancia_transicion_m.tif"
)

OUT_SECTORS = (
    OUT_DIR
    / "16B4d0_resumen_sectores.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "16B4d0_resumen.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B4d0_metadata.json"
)

OUT_FIGURE = (
    OUT_DIR
    / "16B4d0_soporte_extrapolacion_AMSCLAE.png"
)


                                                                               
               
                                                                               

H2014_MED = 1557.560
H2014_MIN = 1557.310
H2014_MAX = 1558.090

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 30.0

DIRECT_COMPAT_TOL_M = 1.0

LOCAL_RADIUS_M = 200.0
LOCAL_NEAREST_MAX_M = 120.0
LOCAL_MIN_NEIGHBORS = 6
LOCAL_MIN_COMPAT_FRACTION = 0.60
LOCAL_MAX_MEDIAN_ABS_RESIDUAL_M = 1.5

FLOAT_NODATA = -9999.0


                                                                               
              
                                                                               

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


def safe_median(values):

    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.median(
            x
        )
    )


def safe_p90(values):

    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.percentile(
            x,
            90
        )
    )


                                                                               
                 
                                                                               

banner(
    "PASO 16B4d0 - LOCALIZACION"
)


for p in [
    PATH_AMSCLAE,
    PATH_CURRENT_LAKE,
    PATH_DIRECT_CORRIDOR,
    PATH_DIRECT_THRESHOLD,
]:

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    print(
        "[FILE]",
        p
    )


                                                                               
                   
                                                                               

banner(
    "QC 1 - GRILLA MAESTRA"
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


with rasterio.open(
    PATH_DIRECT_CORRIDOR
) as ds:

    if (
        ds.shape != shape
        or
        ds.transform != transform
        or
        ds.crs != crs
    ):
        raise RuntimeError(
            "Corredor directo en grilla incompatible."
        )

    direct_corridor = (
        ds.read(1)
        > 0
    )


with rasterio.open(
    PATH_DIRECT_THRESHOLD
) as ds:

    if (
        ds.shape != shape
        or
        ds.transform != transform
        or
        ds.crs != crs
    ):
        raise RuntimeError(
            "Threshold directo en grilla incompatible."
        )

    z_obs = (
        ds.read(1)
        .astype(
            np.float32
        )
    )

    z_obs_nodata = ds.nodata


z_obs_valid = np.isfinite(
    z_obs
)


if z_obs_nodata is not None:

    z_obs_valid &= (
        z_obs
        !=
        z_obs_nodata
    )


if not np.array_equal(
    direct_corridor,
    z_obs_valid
):

    raise RuntimeError(
        "Mascara corredor directo y threshold no coinciden."
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
    "[DIRECT TRANSITION] =",
    f"{direct_corridor.sum():,} pix | "
    f"{direct_corridor.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                          
                                                                               

banner(
    "QC 2 - AMSCLAE EN GRILLA MAESTRA"
)


with rasterio.open(
    PATH_AMSCLAE
) as ds:

    bathy_native = (
        ds.read(1)
        .astype(
            np.float32
        )
    )

    bathy_nodata = ds.nodata

    bathy_transform = ds.transform

    bathy_crs = ds.crs


if bathy_nodata is None:

    valid_native = np.isfinite(
        bathy_native
    )

else:

    valid_native = (
        np.isfinite(
            bathy_native
        )
        &
        (
            bathy_native
            !=
            bathy_nodata
        )
    )


                                          
bathy_20 = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


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


                                            
valid_20_u8 = np.zeros(
    shape,
    dtype=np.uint8
)


reproject(
    source=valid_native.astype(
        np.uint8
    ),
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
    &
    np.isfinite(
        bathy_20
    )
    &
    (
        bathy_20
        !=
        FLOAT_NODATA
    )
)


depth_abs = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


depth_abs[
    valid_20
] = np.maximum(
    -bathy_20[
        valid_20
    ],
    0.0
)


shallow = (
    valid_20
    &
    (
        depth_abs
        >=
        DEPTH_MIN_M
    )
    &
    (
        depth_abs
        <=
        DEPTH_MAX_M
    )
)


print(
    "[AMSCLAE valid 20m] =",
    f"{valid_20.sum():,} pix"
)


print(
    "[AMSCLAE shallow 0.5-30m] =",
    f"{shallow.sum():,} pix | "
    f"{shallow.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                           
                                                                               

banner(
    "QC 3 - COMPATIBILIDAD DIRECTA SENTINEL/NASA vs AMSCLAE"
)


overlap = (
    direct_corridor
    &
    shallow
)


if overlap.sum() == 0:

    raise RuntimeError(
        "No existe solape entre transiciones directas y AMSCLAE somero."
    )


z_ams_central = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


z_ams_low = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


z_ams_high = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


z_ams_central[
    valid_20
] = (
    H2014_MED
    +
    bathy_20[
        valid_20
    ]
)


z_ams_low[
    valid_20
] = (
    H2014_MIN
    +
    bathy_20[
        valid_20
    ]
)


z_ams_high[
    valid_20
] = (
    H2014_MAX
    +
    bathy_20[
        valid_20
    ]
)


residual = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


residual[
    overlap
] = (
    z_obs[
        overlap
    ]
    -
    z_ams_central[
        overlap
    ]
)


direct_compatible = (
    overlap
    &
    (
        z_obs
        >=
        (
            z_ams_low
            -
            DIRECT_COMPAT_TOL_M
        )
    )
    &
    (
        z_obs
        <=
        (
            z_ams_high
            +
            DIRECT_COMPAT_TOL_M
        )
    )
)


direct_incompatible = (
    overlap
    &
    (~direct_compatible)
)


inferred_h2014 = (
    z_obs[
        overlap
    ].astype(
        np.float64
    )
    -
    bathy_20[
        overlap
    ].astype(
        np.float64
    )
)


abs_resid_overlap = np.abs(
    residual[
        overlap
    ].astype(
        np.float64
    )
)


print(
    "[OVERLAP] =",
    f"{overlap.sum():,} pix"
)


print(
    "[DIRECT COMPATIBLE] =",
    f"{direct_compatible.sum():,} pix | "
    f"{100*direct_compatible.sum()/overlap.sum():.2f}%"
)


print(
    "[DIRECT INCOMPATIBLE] =",
    f"{direct_incompatible.sum():,} pix"
)


print(
    "[RESIDUAL central] signed median =",
    f"{np.median(residual[overlap]):+.3f} m"
)


print(
    "[RESIDUAL central] median abs / p90 =",
    f"{np.median(abs_resid_overlap):.3f} / "
    f"{np.percentile(abs_resid_overlap,90):.3f} m"
)


print(
    "[INFERRED H2014] median / p10 / p90 =",
    f"{np.median(inferred_h2014):.3f} / "
    f"{np.percentile(inferred_h2014,10):.3f} / "
    f"{np.percentile(inferred_h2014,90):.3f} m"
)


                                                                               
                                      
                                                                               

banner(
    "QC 4 - SOPORTE LOCAL"
)


cal_rows, cal_cols = np.where(
    overlap
)


cal_x = (
    transform.c
    +
    (
        cal_cols
        +
        0.5
    )
    *
    transform.a
)


cal_y = (
    transform.f
    +
    (
        cal_rows
        +
        0.5
    )
    *
    transform.e
)


cal_xy = np.column_stack(
    [
        cal_x,
        cal_y,
    ]
)


cal_resid = residual[
    overlap
].astype(
    np.float64
)


cal_compat = direct_compatible[
    overlap
].astype(
    np.uint8
)


tree = cKDTree(
    cal_xy
)


target_rows, target_cols = np.where(
    shallow
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


neighbor_lists = tree.query_ball_point(
    target_xy,
    r=LOCAL_RADIUS_M
)


local_fraction = np.full(
    len(
        target_rows
    ),
    np.nan,
    dtype=np.float32
)


local_med_abs = np.full(
    len(
        target_rows
    ),
    np.nan,
    dtype=np.float32
)


local_n = np.zeros(
    len(
        target_rows
    ),
    dtype=np.int16
)


nearest_distance = np.full(
    len(
        target_rows
    ),
    np.inf,
    dtype=np.float32
)


                                                
near_dist, _ = tree.query(
    target_xy,
    k=1
)


nearest_distance[:] = near_dist.astype(
    np.float32
)


for i, inds in enumerate(
    neighbor_lists
):

    if len(
        inds
    ) == 0:
        continue

    inds = np.asarray(
        inds,
        dtype=int
    )

    local_n[
        i
    ] = len(
        inds
    )

    local_fraction[
        i
    ] = float(
        np.mean(
            cal_compat[
                inds
            ]
        )
    )

    local_med_abs[
        i
    ] = float(
        np.median(
            np.abs(
                cal_resid[
                    inds
                ]
            )
        )
    )


local_supported_vector = (
    (
        local_n
        >=
        LOCAL_MIN_NEIGHBORS
    )
    &
    np.isfinite(
        local_fraction
    )
    &
    (
        local_fraction
        >=
        LOCAL_MIN_COMPAT_FRACTION
    )
    &
    np.isfinite(
        local_med_abs
    )
    &
    (
        local_med_abs
        <=
        LOCAL_MAX_MEDIAN_ABS_RESIDUAL_M
    )
    &
    (
        nearest_distance
        <=
        LOCAL_NEAREST_MAX_M
    )
)


local_supported = np.zeros(
    shape,
    dtype=bool
)


local_supported[
    target_rows[
        local_supported_vector
    ],
    target_cols[
        local_supported_vector
    ]
] = True


fraction_map = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


medabs_map = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


distance_map = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


fraction_map[
    target_rows,
    target_cols
] = np.where(
    np.isfinite(
        local_fraction
    ),
    local_fraction,
    FLOAT_NODATA
)


medabs_map[
    target_rows,
    target_cols
] = np.where(
    np.isfinite(
        local_med_abs
    ),
    local_med_abs,
    FLOAT_NODATA
)


distance_map[
    target_rows,
    target_cols
] = np.where(
    np.isfinite(
        nearest_distance
    ),
    nearest_distance,
    FLOAT_NODATA
)


print(
    "[LOCAL SUPPORTED] =",
    f"{local_supported.sum():,} pix | "
    f"{local_supported.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[LOCAL SUPPORTED fraction shallow] =",
    f"{100*local_supported.sum()/max(int(shallow.sum()),1):.2f}%"
)


                                                                               
                      
                                                                               

banner(
    "QC 5 - CLASES DE SOPORTE"
)


support_class = np.zeros(
    shape,
    dtype=np.uint8
)


                                                   
support_class[
    shallow
] = 4


                                 
support_class[
    local_supported
] = 2


                                         
support_class[
    direct_incompatible
] = 3


support_class[
    direct_compatible
] = 1


for cls, label_name in [
    (
        1,
        "transicion directa compatible"
    ),
    (
        2,
        "AMSCLAE somero con soporte local"
    ),
    (
        3,
        "transicion directa incompatible"
    ),
    (
        4,
        "AMSCLAE somero sin soporte"
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
        f"[CLASS {cls}] {label_name}: "
        f"{n:,} pix | "
        f"{n*pixel_area_m2/1e6:.4f} km2"
    )


                                                                               
             
                                                                               

banner(
    "QC 6 - RESUMEN POR SECTOR"
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


for sector in range(
    1,
    13
):

    sm = (
        sector_grid
        ==
        sector
    )

    ov = (
        overlap
        &
        sm
    )

    comp = (
        direct_compatible
        &
        sm
    )

    supp = (
        local_supported
        &
        sm
    )

    shall = (
        shallow
        &
        sm
    )

    sector_rows.append(
        {
            "sector":
                sector,

            "azimuth_lo":
                (sector - 1) * 30,

            "azimuth_hi":
                sector * 30,

            "overlap_transition_pixels":
                int(
                    ov.sum()
                ),

            "direct_compatible_pixels":
                int(
                    comp.sum()
                ),

            "direct_compatible_pct":
                (
                    100.0
                    *
                    comp.sum()
                    /
                    ov.sum()
                    if ov.sum() > 0
                    else
                    np.nan
                ),

            "residual_median_abs_m":
                (
                    safe_median(
                        np.abs(
                            residual[
                                ov
                            ]
                        )
                    )
                    if ov.sum() > 0
                    else
                    np.nan
                ),

            "shallow_amsclae_pixels":
                int(
                    shall.sum()
                ),

            "local_supported_pixels":
                int(
                    supp.sum()
                ),

            "local_supported_pct_shallow":
                (
                    100.0
                    *
                    supp.sum()
                    /
                    shall.sum()
                    if shall.sum() > 0
                    else
                    np.nan
                ),
        }
    )


sector_df = pd.DataFrame(
    sector_rows
)


print(
    sector_df.to_string(
        index=False,
        formatters={
            "direct_compatible_pct":
                "{:.1f}".format,

            "residual_median_abs_m":
                "{:.2f}".format,

            "local_supported_pct_shallow":
                "{:.1f}".format,
        }
    )
)


                                                                               
                    
                                                                               

summary_df = pd.DataFrame(
    [
        {
            "metric":
                "direct_transition_pixels",
            "value":
                int(
                    direct_corridor.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "shallow_amsclae_pixels",
            "value":
                int(
                    shallow.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "direct_overlap_pixels",
            "value":
                int(
                    overlap.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "direct_compatible_pixels",
            "value":
                int(
                    direct_compatible.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "direct_compatible_pct",
            "value":
                float(
                    100.0
                    *
                    direct_compatible.sum()
                    /
                    overlap.sum()
                ),
            "unit":
                "percent",
        },
        {
            "metric":
                "direct_residual_median_abs",
            "value":
                float(
                    np.median(
                        abs_resid_overlap
                    )
                ),
            "unit":
                "m",
        },
        {
            "metric":
                "direct_residual_p90_abs",
            "value":
                float(
                    np.percentile(
                        abs_resid_overlap,
                        90
                    )
                ),
            "unit":
                "m",
        },
        {
            "metric":
                "local_supported_pixels",
            "value":
                int(
                    local_supported.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "local_supported_area",
            "value":
                float(
                    local_supported.sum()
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
                "local_supported_pct_shallow",
            "value":
                float(
                    100.0
                    *
                    local_supported.sum()
                    /
                    max(
                        int(
                            shallow.sum()
                        ),
                        1
                    )
                ),
            "unit":
                "percent",
        },
    ]
)


                                                                               
               
                                                                               

banner(
    "ESCRITURA 16B4d0"
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


with rasterio.open(
    OUT_SUPPORT,
    "w",
    **uint_profile
) as dst:

    dst.write(
        support_class,
        1
    )

    dst.set_band_description(
        1,
        "clase_soporte_extrapolacion_AMSCLAE"
    )


float_profile = profile.copy()


float_profile.update(
    dtype="float32",
    count=1,
    nodata=FLOAT_NODATA,
    compress="deflate",
)


residual_out = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


residual_out[
    overlap
] = residual[
    overlap
]


with rasterio.open(
    OUT_RESIDUAL,
    "w",
    **float_profile
) as dst:

    dst.write(
        residual_out,
        1
    )

    dst.set_band_description(
        1,
        "Z_SentinelNASA_menos_Z_AMSCLAE_central_m"
    )


for path, arr, desc in [
    (
        OUT_LOCAL_FRACTION,
        fraction_map,
        "fraccion_transiciones_compatibles_radio_200m"
    ),
    (
        OUT_LOCAL_MEDABS,
        medabs_map,
        "mediana_abs_residual_local_radio_200m_m"
    ),
    (
        OUT_DISTANCE,
        distance_map,
        "distancia_transicion_directa_m"
    ),
]:

    with rasterio.open(
        path,
        "w",
        **float_profile
    ) as dst:

        dst.write(
            arr.astype(
                np.float32
            ),
            1
        )

        dst.set_band_description(
            1,
            desc
        )


sector_df.to_csv(
    OUT_SECTORS,
    index=False,
    encoding="utf-8-sig"
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


im = ax.imshow(
    support_class,
    vmin=0,
    vmax=4
)


ax.set_title(
    "16B4d0 - Soporte para extrapolar costa con AMSCLAE\n"
    "1=directo compatible | 2=local soportado | "
    "3=directo incompatible | 4=somero sin soporte"
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
    ]
)


cb.set_label(
    "Clase"
)


plt.tight_layout()


fig.savefig(
    OUT_FIGURE,
    dpi=180,
    bbox_inches="tight"
)


plt.close(
    fig
)


                                                                               
              
                                                                               

metadata = {
    "step":
        "16B4d0",

    "status":
        "AMSCLAE_FUTURE_EXTRAPOLATION_SUPPORT_AUDIT",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "principle":
        (
            "AMSCLAE is never altered. This audit only determines where its "
            "original relative depths are locally compatible with direct "
            "Sentinel/NASA shoreline transitions and can therefore be considered "
            "for extrapolation below the directly observed level range."
        ),

    "H2014_reference": {
        "median_m":
            H2014_MED,

        "min_m":
            H2014_MIN,

        "max_m":
            H2014_MAX,
    },

    "shallow_domain": {
        "depth_min_m":
            DEPTH_MIN_M,

        "depth_max_m":
            DEPTH_MAX_M,

        "pixels":
            int(
                shallow.sum()
            ),
    },

    "direct_compatibility": {
        "comparison_tolerance_m":
            DIRECT_COMPAT_TOL_M,

        "overlap_pixels":
            int(
                overlap.sum()
            ),

        "compatible_pixels":
            int(
                direct_compatible.sum()
            ),

        "compatible_pct":
            float(
                100.0
                *
                direct_compatible.sum()
                /
                overlap.sum()
            ),

        "median_abs_residual_m":
            float(
                np.median(
                    abs_resid_overlap
                )
            ),

        "p90_abs_residual_m":
            float(
                np.percentile(
                    abs_resid_overlap,
                    90
                )
            ),
    },

    "local_support_rule": {
        "radius_m":
            LOCAL_RADIUS_M,

        "nearest_transition_max_m":
            LOCAL_NEAREST_MAX_M,

        "min_neighbors":
            LOCAL_MIN_NEIGHBORS,

        "min_compatible_fraction":
            LOCAL_MIN_COMPAT_FRACTION,

        "max_median_abs_residual_m":
            LOCAL_MAX_MEDIAN_ABS_RESIDUAL_M,

        "supported_pixels":
            int(
                local_supported.sum()
            ),

        "supported_area_km2":
            float(
                local_supported.sum()
                *
                pixel_area_m2
                /
                1e6
            ),
    },

    "classes": {
        "0":
            "outside_or_not_evaluated",

        "1":
            "direct_transition_compatible_with_AMSCLAE_2014_bracket",

        "2":
            "shallow_AMSCLAE_locally_supported_for_extrapolation",

        "3":
            "direct_transition_incompatible_with_AMSCLAE",

        "4":
            "shallow_AMSCLAE_without_sufficient_local_support",
    },

    "limitations": [
        (
            "This is not a vertical correction of AMSCLAE."
        ),
        (
            "Only shallow AMSCLAE depths between 0.5 and 30 m are evaluated."
        ),
        (
            "Direct compatibility is descriptive and includes a 1 m comparison "
            "allowance; it is not a probabilistic confidence interval."
        ),
        (
            "Unsupported AMSCLAE zones must remain unknown in future shoreline "
            "maps rather than being forced into a prediction."
        ),
    ],

    "sha256": {
        OUT_SUPPORT.name:
            sha256_file(
                OUT_SUPPORT
            ),

        OUT_RESIDUAL.name:
            sha256_file(
                OUT_RESIDUAL
            ),

        OUT_LOCAL_FRACTION.name:
            sha256_file(
                OUT_LOCAL_FRACTION
            ),

        OUT_LOCAL_MEDABS.name:
            sha256_file(
                OUT_LOCAL_MEDABS
            ),

        OUT_DISTANCE.name:
            sha256_file(
                OUT_DISTANCE
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
    "DECISION QC 16B4d0"
)


compat_pct = (
    100.0
    *
    direct_compatible.sum()
    /
    overlap.sum()
)


local_pct = (
    100.0
    *
    local_supported.sum()
    /
    max(
        int(
            shallow.sum()
        ),
        1
    )
)


print(
    "[QC] overlap directo AMSCLAE =",
    f"{overlap.sum():,} pix"
)


print(
    "[QC] compatible directo =",
    f"{direct_compatible.sum():,} pix "
    f"({compat_pct:.2f}%)"
)


print(
    "[QC] residual median abs / p90 =",
    f"{np.median(abs_resid_overlap):.3f} / "
    f"{np.percentile(abs_resid_overlap,90):.3f} m"
)


print(
    "[QC] AMSCLAE somero total =",
    f"{shallow.sum():,} pix"
)


print(
    "[QC] AMSCLAE somero con soporte local =",
    f"{local_supported.sum():,} pix "
    f"({local_pct:.2f}%)"
)


print()


if local_supported.sum() > 0:

    print(
        "[PASS PARTIAL SUPPORT]"
    )


    print(
        "Existen zonas donde AMSCLAE puede usarse de forma conservadora "
        "para extrapolar retiros por debajo de 1552.450 m."
    )


    print(
        "Las zonas sin soporte se mantendran como DESCONOCIDAS."
    )


    print()


    print(
        "[NEXT] 16B4d:"
    )


    print(
        "construir mapas de costa por nivel con procedencia pixel a pixel: "
        "OBSERVADO_DIRECTO / AMSCLAE_SOPORTADO / DESCONOCIDO."
    )


else:

    print(
        "[NO SUPPORT]"
    )


    print(
        "AMSCLAE no ofrece soporte local suficiente para extrapolacion "
        "costera bajo el minimo observado. No construir mapas futuros "
        "espaciales fuera del rango Sentinel/NASA."
    )


banner(
    "FIN PASO 16B4d0"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
