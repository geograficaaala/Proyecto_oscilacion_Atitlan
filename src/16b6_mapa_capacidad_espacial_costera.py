                       

















































































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
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

D0_DIR = (
    B4_DIR
    / "anclaje_AMSCLAE_16B4d0"
)

PATH_CURRENT_LAKE = (
    B4_DIR
    / "16B4a_mascara_lago_actual_Sentinel_50pct.tif"
)

PATH_NO_BATHY = (
    B4_DIR
    / "16B4a_agua_actual_sin_batimetria_AMSCLAE.tif"
)

PATH_DIRECT_CORRIDOR = (
    C2_DIR
    / "16B4c2_corredor_transicion_directa.tif"
)

PATH_DIRECT_THRESHOLD = (
    C2_DIR
    / "16B4c2_nivel_umbral_directo_m.tif"
)

PATH_AMS_SUPPORT = (
    D0_DIR
    / "16B4d0_soporte_extrapolacion_AMSCLAE.tif"
)

OUT_DIR = (
    B4_DIR
    / "capacidad_espacial_16B6"
)

OUT_CAPABILITY = (
    OUT_DIR
    / "16B6_capacidad_espacial_costera.tif"
)

OUT_THRESHOLD = (
    OUT_DIR
    / "16B6_nivel_umbral_con_fuente_m.tif"
)

OUT_PROVENANCE = (
    OUT_DIR
    / "16B6_procedencia_umbral.tif"
)

OUT_DOMAIN = (
    OUT_DIR
    / "16B6_dominio_costero_evaluado.tif"
)

OUT_GLOBAL = (
    OUT_DIR
    / "16B6_resumen_global.csv"
)

OUT_DEPTH = (
    OUT_DIR
    / "16B6_resumen_profundidad.csv"
)

OUT_SECTORS = (
    OUT_DIR
    / "16B6_resumen_sectores.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B6_metadata.json"
)

OUT_FIGURE = (
    OUT_DIR
    / "16B6_mapa_capacidad_espacial.png"
)


                                                                               
               
                                                                               

H2014_MED = 1557.560

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 30.0

FLOAT_NODATA = -9999.0

                 
CLASS_OUTSIDE = 0
CLASS_DIRECT = 1
CLASS_AMS_SUPPORTED = 2
CLASS_UNKNOWN = 3

                             
DEPTH_BINS = [
    0.5,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
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


def ensure_same_grid(
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


                                                                               
                 
                                                                               

banner(
    "PASO 16B6 - LOCALIZACION"
)


required = [
    PATH_AMSCLAE,
    PATH_CURRENT_LAKE,
    PATH_NO_BATHY,
    PATH_DIRECT_CORRIDOR,
    PATH_DIRECT_THRESHOLD,
    PATH_AMS_SUPPORT,
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
    PATH_NO_BATHY
) as ds:

    ensure_same_grid(
        ds,
        shape,
        transform,
        crs,
        "agua actual sin AMSCLAE"
    )

    no_bathy = (
        ds.read(1)
        > 0
    )


with rasterio.open(
    PATH_DIRECT_CORRIDOR
) as ds:

    ensure_same_grid(
        ds,
        shape,
        transform,
        crs,
        "corredor directo"
    )

    direct_corridor = (
        ds.read(1)
        > 0
    )


with rasterio.open(
    PATH_DIRECT_THRESHOLD
) as ds:

    ensure_same_grid(
        ds,
        shape,
        transform,
        crs,
        "umbral directo"
    )

    direct_threshold = (
        ds.read(1)
        .astype(
            np.float32
        )
    )

    direct_nodata = ds.nodata


direct_threshold_valid = np.isfinite(
    direct_threshold
)


if direct_nodata is not None:

    direct_threshold_valid &= (
        direct_threshold
        !=
        direct_nodata
    )


if not np.array_equal(
    direct_corridor,
    direct_threshold_valid
):

    raise RuntimeError(
        "El corredor directo no coincide exactamente con el raster de umbral."
    )


with rasterio.open(
    PATH_AMS_SUPPORT
) as ds:

    ensure_same_grid(
        ds,
        shape,
        transform,
        crs,
        "soporte AMSCLAE"
    )

    ams_support_class = (
        ds.read(1)
        .astype(
            np.uint8
        )
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
    "[DIRECT] =",
    f"{direct_corridor.sum():,} pix | "
    f"{direct_corridor.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT WATER WITHOUT AMSCLAE] =",
    f"{no_bathy.sum():,} pix | "
    f"{no_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                            
                                                                               

banner(
    "QC 2 - AMSCLAE ORIGINAL"
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

    native_valid = np.isfinite(
        bathy_native
    )

else:

    native_valid = (
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
    source=native_valid.astype(
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


z_ams = np.full(
    shape,
    np.nan,
    dtype=np.float32
)


z_ams[
    valid_20
] = (
    H2014_MED
    +
    bathy_20[
        valid_20
    ]
)


print(
    "[AMSCLAE 20m valid] =",
    f"{valid_20.sum():,} pix"
)


print(
    "[AMSCLAE shallow 0.5-30m] =",
    f"{shallow.sum():,} pix | "
    f"{shallow.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                                  
                                                                               

banner(
    "QC 3 - CAPACIDAD ESPACIAL"
)


         
                                   
                                      
                                     
                                
ams_supported = (
    ams_support_class
    ==
    2
)


                                                    
domain = (
    direct_corridor
    |
    shallow
    |
    no_bathy
)


capability = np.zeros(
    shape,
    dtype=np.uint8
)


                                         
capability[
    domain
] = CLASS_UNKNOWN


                    
capability[
    ams_supported
] = CLASS_AMS_SUPPORTED


                                      
capability[
    direct_corridor
] = CLASS_DIRECT


                                                              
if np.any(
    ams_supported
    &
    (~shallow)
):

    raise RuntimeError(
        "16B4d0 marca soporte AMSCLAE fuera del dominio somero esperado."
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
        CLASS_AMS_SUPPORTED
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
    "[DOMAIN] =",
    f"{n_domain:,} pix | "
    f"{n_domain*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CLASS 1 OBSERVADO_DIRECTO] =",
    f"{n_direct:,} pix | "
    f"{n_direct*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CLASS 2 AMSCLAE_SOPORTADO] =",
    f"{n_ams:,} pix | "
    f"{n_ams*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CLASS 3 DESCONOCIDO] =",
    f"{n_unknown:,} pix | "
    f"{n_unknown*pixel_area_m2/1e6:.4f} km2"
)


known_n = (
    n_direct
    +
    n_ams
)


known_pct = (
    100.0
    *
    known_n
    /
    max(
        n_domain,
        1
    )
)


print(
    "[KNOWN CAPABILITY] =",
    f"{known_pct:.2f}% del dominio evaluado"
)


                                                                               
                                   
                                                                               

banner(
    "QC 4 - UMBRAL DE NIVEL CON PROCEDENCIA"
)


threshold = np.full(
    shape,
    FLOAT_NODATA,
    dtype=np.float32
)


provenance = np.zeros(
    shape,
    dtype=np.uint8
)


                            
threshold[
    capability
    ==
    CLASS_AMS_SUPPORTED
] = z_ams[
    capability
    ==
    CLASS_AMS_SUPPORTED
]


provenance[
    capability
    ==
    CLASS_AMS_SUPPORTED
] = 2


                                         
threshold[
    capability
    ==
    CLASS_DIRECT
] = direct_threshold[
    capability
    ==
    CLASS_DIRECT
]


provenance[
    capability
    ==
    CLASS_DIRECT
] = 1


threshold_valid = (
    threshold
    !=
    FLOAT_NODATA
)


if int(
    threshold_valid.sum()
) != known_n:

    raise RuntimeError(
        "El numero de umbrales validos no coincide con las clases conocidas."
    )


print(
    "[THRESHOLD VALID] =",
    f"{threshold_valid.sum():,} pix"
)


print(
    "[DIRECT threshold] min/median/max =",
    f"{np.min(threshold[provenance==1]):.3f} / "
    f"{np.median(threshold[provenance==1]):.3f} / "
    f"{np.max(threshold[provenance==1]):.3f} m"
)


if n_ams > 0:

    print(
        "[AMS supported threshold] min/median/max =",
        f"{np.min(threshold[provenance==2]):.3f} / "
        f"{np.median(threshold[provenance==2]):.3f} / "
        f"{np.max(threshold[provenance==2]):.3f} m"
    )


                                                                               
                                    
                                                                               

banner(
    "QC 5 - COBERTURA POR PROFUNDIDAD"
)


depth_rows = []


for lo, hi in zip(
    DEPTH_BINS[:-1],
    DEPTH_BINS[1:]
):

    band = (
        shallow
        &
        (
            depth_abs
            >=
            lo
        )
        &
        (
            depth_abs
            <
            hi
        )
    )


    supported_band = (
        band
        &
        (
            capability
            ==
            CLASS_AMS_SUPPORTED
        )
    )


    direct_band = (
        band
        &
        (
            capability
            ==
            CLASS_DIRECT
        )
    )


    known_band = (
        band
        &
        np.isin(
            capability,
            [
                CLASS_DIRECT,
                CLASS_AMS_SUPPORTED,
            ]
        )
    )


    total_n = int(
        band.sum()
    )


    depth_rows.append(
        {
            "depth_lo_m":
                lo,

            "depth_hi_m":
                hi,

            "amsclae_shallow_pixels":
                total_n,

            "observed_direct_pixels":
                int(
                    direct_band.sum()
                ),

            "amsclae_supported_pixels":
                int(
                    supported_band.sum()
                ),

            "known_pixels":
                int(
                    known_band.sum()
                ),

            "known_pct":
                (
                    100.0
                    *
                    known_band.sum()
                    /
                    total_n
                    if total_n > 0
                    else
                    np.nan
                ),
        }
    )


depth_df = pd.DataFrame(
    depth_rows
)


print(
    depth_df.to_string(
        index=False,
        formatters={
            "depth_lo_m":
                "{:.1f}".format,

            "depth_hi_m":
                "{:.1f}".format,

            "known_pct":
                "{:.2f}".format,
        }
    )
)


                                                                               
                       
                                                                               

banner(
    "QC 6 - COBERTURA POR SECTOR"
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


    dom = (
        domain
        &
        sm
    )


    direct_m = (
        (
            capability
            ==
            CLASS_DIRECT
        )
        &
        sm
    )


    ams_m = (
        (
            capability
            ==
            CLASS_AMS_SUPPORTED
        )
        &
        sm
    )


    unk_m = (
        (
            capability
            ==
            CLASS_UNKNOWN
        )
        &
        sm
    )


    dom_n = int(
        dom.sum()
    )


    known_sector = int(
        direct_m.sum()
        +
        ams_m.sum()
    )


    sector_rows.append(
        {
            "sector":
                sector,

            "azimuth_lo":
                (sector - 1) * 30,

            "azimuth_hi":
                sector * 30,

            "domain_pixels":
                dom_n,

            "observed_direct_pixels":
                int(
                    direct_m.sum()
                ),

            "amsclae_supported_pixels":
                int(
                    ams_m.sum()
                ),

            "unknown_pixels":
                int(
                    unk_m.sum()
                ),

            "known_pct":
                (
                    100.0
                    *
                    known_sector
                    /
                    dom_n
                    if dom_n > 0
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
            "known_pct":
                "{:.2f}".format,
        }
    )
)


                                                                               
                             
                                                                               

banner(
    "QC 7 - AGUA ACTUAL SIN AMSCLAE"
)


gap_direct = (
    no_bathy
    &
    (
        capability
        ==
        CLASS_DIRECT
    )
)


gap_known = (
    no_bathy
    &
    np.isin(
        capability,
        [
            CLASS_DIRECT,
            CLASS_AMS_SUPPORTED,
        ]
    )
)


gap_unknown = (
    no_bathy
    &
    (
        capability
        ==
        CLASS_UNKNOWN
    )
)


print(
    "[CURRENT GAP TOTAL] =",
    f"{no_bathy.sum():,} pix"
)


print(
    "[CURRENT GAP OBSERVED DIRECT] =",
    f"{gap_direct.sum():,} pix"
)


print(
    "[CURRENT GAP KNOWN TOTAL] =",
    f"{gap_known.sum():,} pix"
)


print(
    "[CURRENT GAP UNKNOWN] =",
    f"{gap_unknown.sum():,} pix"
)


                                                                               
                    
                                                                               

global_df = pd.DataFrame(
    [
        {
            "metric":
                "domain_pixels",
            "value":
                n_domain,
            "unit":
                "pixels",
        },
        {
            "metric":
                "domain_area_km2",
            "value":
                float(
                    n_domain
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
                "observed_direct_pixels",
            "value":
                n_direct,
            "unit":
                "pixels",
        },
        {
            "metric":
                "observed_direct_area_km2",
            "value":
                float(
                    n_direct
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
                "amsclae_supported_pixels",
            "value":
                n_ams,
            "unit":
                "pixels",
        },
        {
            "metric":
                "amsclae_supported_area_km2",
            "value":
                float(
                    n_ams
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
                "unknown_pixels",
            "value":
                n_unknown,
            "unit":
                "pixels",
        },
        {
            "metric":
                "unknown_area_km2",
            "value":
                float(
                    n_unknown
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
                "known_capability_pct",
            "value":
                known_pct,
            "unit":
                "percent",
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
                "current_gap_known_pixels",
            "value":
                int(
                    gap_known.sum()
                ),
            "unit":
                "pixels",
        },
        {
            "metric":
                "current_gap_unknown_pixels",
            "value":
                int(
                    gap_unknown.sum()
                ),
            "unit":
                "pixels",
        },
    ]
)


                                                                               
               
                                                                               

banner(
    "ESCRITURA 16B6"
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
    OUT_CAPABILITY,
    capability,
    "capacidad_espacial_costera"
)


write_uint(
    OUT_PROVENANCE,
    provenance,
    "procedencia_nivel_umbral"
)


write_uint(
    OUT_DOMAIN,
    domain,
    "dominio_costero_evaluado"
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
        "nivel_umbral_costero_m"
    )


global_df.to_csv(
    OUT_GLOBAL,
    index=False,
    encoding="utf-8-sig"
)


depth_df.to_csv(
    OUT_DEPTH,
    index=False,
    encoding="utf-8-sig"
)


sector_df.to_csv(
    OUT_SECTORS,
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
    capability,
    vmin=0,
    vmax=3
)


ax.set_title(
    "16B6 - Capacidad espacial para modelar la orilla futura\n"
    "1=OBSERVADO_DIRECTO | 2=AMSCLAE_SOPORTADO | 3=DESCONOCIDO"
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
    OUT_FIGURE,
    dpi=180,
    bbox_inches="tight"
)


plt.close(
    fig
)


                                                                               
              
                                                                               

metadata = {
    "step":
        "16B6",

    "status":
        "SPATIAL_CAPABILITY_MAP_READY",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "purpose":
        (
            "Define pixel by pixel where future shoreline position can be "
            "supported directly, where only locally supported AMSCLAE may be "
            "used, and where shoreline position must remain unknown."
        ),

    "classes": {
        "0":
            "outside_evaluated_coastal_domain",

        "1":
            "OBSERVADO_DIRECTO",

        "2":
            "AMSCLAE_SOPORTADO",

        "3":
            "DESCONOCIDO",
    },

    "threshold_provenance": {
        "1":
            "Sentinel/NASA direct transition threshold from 16B4c2",

        "2":
            (
                "Original AMSCLAE depth converted with H2014_MED=1557.560 m, "
                "only where 16B4d0 accepted local support"
            ),
    },

    "counts": {
        "domain_pixels":
            n_domain,

        "observed_direct_pixels":
            n_direct,

        "amsclae_supported_pixels":
            n_ams,

        "unknown_pixels":
            n_unknown,

        "known_capability_pct":
            float(
                known_pct
            ),
    },

    "current_water_without_amsclae": {
        "total_pixels":
            int(
                no_bathy.sum()
            ),

        "known_pixels":
            int(
                gap_known.sum()
            ),

        "unknown_pixels":
            int(
                gap_unknown.sum()
            ),
    },

    "important_limitations": [
        (
            "This product is a capability/provenance map, not a future shoreline map."
        ),
        (
            "No unsupported coastal pixel is filled or interpolated."
        ),
        (
            "AMSCLAE is not globally corrected or shifted."
        ),
        (
            "Future shoreline maps must preserve these provenance classes and "
            "must not turn DESCONOCIDO into land or water."
        ),
        (
            "Deep lake bathymetry can remain useful for basin morphology even "
            "where the precise shallow shoreline position is unsupported."
        ),
    ],

    "sha256": {
        OUT_CAPABILITY.name:
            sha256_file(
                OUT_CAPABILITY
            ),

        OUT_THRESHOLD.name:
            sha256_file(
                OUT_THRESHOLD
            ),

        OUT_PROVENANCE.name:
            sha256_file(
                OUT_PROVENANCE
            ),

        OUT_DOMAIN.name:
            sha256_file(
                OUT_DOMAIN
            ),

        OUT_GLOBAL.name:
            sha256_file(
                OUT_GLOBAL
            ),

        OUT_DEPTH.name:
            sha256_file(
                OUT_DEPTH
            ),

        OUT_SECTORS.name:
            sha256_file(
                OUT_SECTORS
            ),
    },

    "next_step":
        (
            "Apply frozen 2030/2050 scenario lake levels to this threshold/provenance "
            "map, producing observed/supported/unknown shoreline-state products "
            "without forcing continuity across unknown zones."
        ),
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
    "DECISION QC 16B6"
)


print(
    "[QC] dominio evaluado =",
    f"{n_domain:,} pix"
)


print(
    "[QC] OBSERVADO_DIRECTO =",
    f"{n_direct:,} pix"
)


print(
    "[QC] AMSCLAE_SOPORTADO =",
    f"{n_ams:,} pix"
)


print(
    "[QC] DESCONOCIDO =",
    f"{n_unknown:,} pix"
)


print(
    "[QC] capacidad conocida =",
    f"{known_pct:.2f}%"
)


print(
    "[QC] agua actual sin AMSCLAE que sigue desconocida =",
    f"{gap_unknown.sum():,} / {no_bathy.sum():,} pix"
)


print()


print(
    "[PASS CAPABILITY MAP]"
)


print(
    "La capacidad espacial queda separada en OBSERVADO_DIRECTO, "
    "AMSCLAE_SOPORTADO y DESCONOCIDO."
)


print(
    "No se invento una costa continua."
)


print()


print(
    "[NEXT] 16B7:"
)


print(
    "aplicar los niveles futuros congelados 2030/2050 sobre este mapa "
    "y producir estados costeros con procedencia y zonas desconocidas explicitas."
)


banner(
    "FIN PASO 16B6"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
