                       

















































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import shutil
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import label
import matplotlib.pyplot as plt


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

DOWNLOADS = ROOT / "data" / "external"

                                                                              
                                               
OBSOLETE_FROZEN_DIR = (
    ROOT
    / "outputs"
    / "batimetria_hibrida_conservadora"
)

PATH_AMSCLAE = ROOT / "data" / "external" / "batimetria_amsclae.tif"

REFERENCE_NAME = (
    "atitlan_16A1_v5_costa_moderna_2024_2026_02_mascara_referencia_50pct.tif"
)

SHORE_NAME = (
    "atitlan_16A1_v5_costa_moderna_2024_2026_03_franja_litoral_20_80.tif"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
)

OUT_LAKE = (
    OUT_DIR
    / "16B4a_mascara_lago_actual_Sentinel_50pct.tif"
)

OUT_NO_BATHY = (
    OUT_DIR
    / "16B4a_agua_actual_sin_batimetria_AMSCLAE.tif"
)

OUT_SHORE = (
    OUT_DIR
    / "16B4a_corredor_litoral_Sentinel_20_80.tif"
)

OUT_CLASSES = (
    OUT_DIR
    / "16B4a_clases_cobertura_lago_actual.tif"
)

OUT_COMPONENTS = (
    OUT_DIR
    / "16B4a_componentes_agua_actual_sin_batimetria.csv"
)

OUT_SECTORS = (
    OUT_DIR
    / "16B4a_resumen_sectores.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "16B4a_resumen_global.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B4a_metadata.json"
)

OUT_FIGURE = (
    OUT_DIR
    / "16B4a_mapa_agua_actual_sin_batimetria.png"
)


                                                                               
              
                                                                               

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


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
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
    "LIMPIEZA - ELIMINAR PRODUCTOS OBSOLETOS 16B3f / 16C0"
)

print(
    "[OBSOLETE DIR]",
    OBSOLETE_FROZEN_DIR
)

if OBSOLETE_FROZEN_DIR.exists():

    shutil.rmtree(
        OBSOLETE_FROZEN_DIR
    )

    print(
        "[DELETED] carpeta obsoleta eliminada completamente."
    )

else:

    print(
        "[OK] la carpeta obsoleta no existe; no hay nada que borrar."
    )


                                                                   
protected = [
    PATH_AMSCLAE,
    DOWNLOADS,
    ROOT / "data",
    ROOT / "src",
]

for p in protected:
    print(
        "[PROTECTED]",
        p
    )


                                                                               
                 
                                                                               

banner(
    "PASO 16B4a - LOCALIZACION"
)

if not PATH_AMSCLAE.exists():
    raise FileNotFoundError(
        PATH_AMSCLAE
    )

reference_path = find_file(
    REFERENCE_NAME
)

shore_path = find_file(
    SHORE_NAME
)

for p in [
    PATH_AMSCLAE,
    reference_path,
    shore_path,
]:
    print(
        "[FILE]",
        p
    )


                                                                               
                            
                                                                               

banner(
    "QC 1 - GRILLA SENTINEL"
)

with rasterio.open(
    reference_path
) as ds:

    ref = (
        ds.read(1)
        > 0
    )

    profile_20m = ds.profile.copy()

    shape = ds.shape

    transform = ds.transform

    crs = ds.crs


with rasterio.open(
    shore_path
) as ds:

    if (
        ds.shape != shape
        or
        ds.transform != transform
        or
        ds.crs != crs
    ):
        raise RuntimeError(
            "La franja litoral no coincide con la grilla de referencia."
        )

    shore = (
        ds.read(1)
        > 0
    )


pixel_area_m2 = abs(
    transform.a
    *
    transform.e
    -
    transform.b
    *
    transform.d
)

pixel_x = abs(
    transform.a
)

pixel_y = abs(
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
    "[GRID] pixel =",
    pixel_x,
    "x",
    pixel_y,
    "m"
)

print(
    "[LAKE REF 50%] pixels =",
    f"{ref.sum():,}"
)

print(
    "[LAKE REF 50%] area =",
    f"{ref.sum() * pixel_area_m2 / 1e6:.4f} km2"
)

print(
    "[SHORE 20-80] pixels =",
    f"{shore.sum():,}"
)

print(
    "[SHORE 20-80] area =",
    f"{shore.sum() * pixel_area_m2 / 1e6:.4f} km2"
)


                                                                               
                               
                                                                               

banner(
    "QC 2 - COBERTURA AMSCLAE ORIGINAL"
)

with rasterio.open(
    PATH_AMSCLAE
) as ds:

    bathy = (
        ds.read(1)
        .astype(
            np.float32
        )
    )

    bathy_shape = ds.shape

    bathy_transform = ds.transform

    bathy_crs = ds.crs

    bathy_nodata = ds.nodata


if bathy_nodata is None:

    bathy_valid_native = np.isfinite(
        bathy
    )

else:

    bathy_valid_native = (
        np.isfinite(
            bathy
        )
        &
        (
            bathy
            !=
            bathy_nodata
        )
    )


print(
    "[AMSCLAE] native shape =",
    bathy_shape
)

print(
    "[AMSCLAE] CRS =",
    bathy_crs
)

print(
    "[AMSCLAE] valid native =",
    f"{bathy_valid_native.sum():,}"
)


                   
                                                        
                                                                                 
bathy_coverage_20 = np.zeros(
    shape,
    dtype=np.uint8
)


reproject(
    source=bathy_valid_native.astype(
        np.uint8
    ),
    destination=bathy_coverage_20,
    src_transform=bathy_transform,
    src_crs=bathy_crs,
    src_nodata=0,
    dst_transform=transform,
    dst_crs=crs,
    dst_nodata=0,
    resampling=Resampling.nearest,
)


bathy_covered = (
    bathy_coverage_20
    > 0
)


print(
    "[AMSCLAE->20m] covered pixels total =",
    f"{bathy_covered.sum():,}"
)


                                                                               
                                    
                                                                               

banner(
    "QC 3 - LAGO ACTUAL VS COBERTURA BATIMETRICA"
)


                                                  
lake_current = ref.copy()


current_with_bathy = (
    lake_current
    &
    bathy_covered
)


current_without_bathy = (
    lake_current
    &
    (~bathy_covered)
)


shore_with_bathy = (
    shore
    &
    bathy_covered
)


shore_without_bathy = (
    shore
    &
    (~bathy_covered)
)


print(
    "[CURRENT LAKE] area =",
    f"{lake_current.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT + AMSCLAE] pixels =",
    f"{current_with_bathy.sum():,}"
)


print(
    "[CURRENT + AMSCLAE] area =",
    f"{current_with_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT WITHOUT AMSCLAE] pixels =",
    f"{current_without_bathy.sum():,}"
)


print(
    "[CURRENT WITHOUT AMSCLAE] area =",
    f"{current_without_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[CURRENT WITHOUT AMSCLAE] fraction lake =",
    f"{100*current_without_bathy.sum()/max(lake_current.sum(),1):.3f}%"
)


print(
    "[SHORE 20-80 WITHOUT AMSCLAE] pixels =",
    f"{shore_without_bathy.sum():,}"
)


print(
    "[SHORE 20-80 WITHOUT AMSCLAE] area =",
    f"{shore_without_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


                                                                               
                    
                                                                               

banner(
    "QC 4 - CLASES"
)


classes = np.zeros(
    shape,
    dtype=np.uint8
)


                         
classes[
    shore_with_bathy
] = 3


classes[
    shore_without_bathy
] = 4


                                       
classes[
    current_with_bathy
] = 1


classes[
    current_without_bathy
] = 2


for cls, name in [
    (
        1,
        "lago actual con AMSCLAE"
    ),
    (
        2,
        "lago actual SIN AMSCLAE"
    ),
    (
        3,
        "franja litoral con AMSCLAE"
    ),
    (
        4,
        "franja litoral SIN AMSCLAE"
    ),
]:

    n = int(
        np.sum(
            classes
            ==
            cls
        )
    )

    print(
        f"[CLASS {cls}] {name}: "
        f"{n:,} pix | {n*pixel_area_m2/1e6:.4f} km2"
    )


                                                                               
                                               
                                                                               

banner(
    "QC 5 - COMPONENTES DE AGUA ACTUAL SIN BATIMETRIA"
)


structure = np.ones(
    (3, 3),
    dtype=np.uint8
)


labels, n_components = label(
    current_without_bathy,
    structure=structure
)


rows_out = []


if current_without_bathy.any():

    lake_rows, lake_cols = np.where(
        lake_current
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


    for component_id in range(
        1,
        n_components + 1
    ):

        m = (
            labels
            ==
            component_id
        )


        n = int(
            m.sum()
        )


        if n == 0:
            continue


        rr, cc = np.where(
            m
        )


        rmean = float(
            np.mean(
                rr
            )
        )


        cmean = float(
            np.mean(
                cc
            )
        )


                                                  
                                  
        dx = (
            cmean
            -
            cx
        )

        dy = (
            cy
            -
            rmean
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


        sector = int(
            np.floor(
                azimuth
                /
                30.0
            )
        ) + 1


        rows_out.append(
            {
                "component_id":
                    component_id,

                "pixels":
                    n,

                "area_km2":
                    n
                    *
                    pixel_area_m2
                    /
                    1e6,

                "row_centroid":
                    rmean,

                "col_centroid":
                    cmean,

                "azimuth_deg":
                    azimuth,

                "sector_30deg":
                    sector,

                "touches_shore_band":
                    bool(
                        np.any(
                            m
                            &
                            shore
                        )
                    ),
            }
        )


components_df = pd.DataFrame(
    rows_out
)


print(
    "[COMPONENTS] n =",
    n_components
)


if len(
    components_df
) > 0:

    components_df = components_df.sort_values(
        "pixels",
        ascending=False
    ).reset_index(
        drop=True
    )


    print(
        "[COMPONENTS] largest pixels =",
        int(
            components_df.iloc[0][
                "pixels"
            ]
        )
    )


    print(
        "[COMPONENTS] largest area =",
        f"{components_df.iloc[0]['area_km2']:.4f} km2"
    )


    print(
        "[COMPONENTS] top 10 sizes =",
        components_df[
            "pixels"
        ].head(
            10
        ).astype(
            int
        ).tolist()
    )


                                                                               
                       
                                                                               

banner(
    "QC 6 - RESUMEN POR SECTOR"
)


lake_rows, lake_cols = np.where(
    lake_current
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


all_rows, all_cols = np.indices(
    shape
)


dx = (
    all_cols
    -
    cx
)


dy = (
    cy
    -
    all_rows
)


azimuth_grid = (
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
        azimuth_grid
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

    lo = (
        sector
        -
        1
    ) * 30

    hi = (
        sector
        *
        30
    )


    m_lake = (
        lake_current
        &
        (
            sector_grid
            ==
            sector
        )
    )


    m_gap = (
        current_without_bathy
        &
        (
            sector_grid
            ==
            sector
        )
    )


    lake_n = int(
        m_lake.sum()
    )


    gap_n = int(
        m_gap.sum()
    )


    sector_rows.append(
        {
            "sector":
                sector,

            "azimuth_lo":
                lo,

            "azimuth_hi":
                hi,

            "lake_pixels":
                lake_n,

            "lake_area_km2":
                lake_n
                *
                pixel_area_m2
                /
                1e6,

            "missing_bathy_pixels":
                gap_n,

            "missing_bathy_area_km2":
                gap_n
                *
                pixel_area_m2
                /
                1e6,

            "missing_fraction_pct":
                (
                    100.0
                    *
                    gap_n
                    /
                    lake_n
                    if lake_n > 0
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
            "lake_area_km2":
                "{:.4f}".format,

            "missing_bathy_area_km2":
                "{:.4f}".format,

            "missing_fraction_pct":
                "{:.2f}".format,
        }
    )
)


                                                                               
                   
                                                                               

summary_df = pd.DataFrame(
    [
        {
            "metric":
                "current_lake_pixels",

            "value":
                int(
                    lake_current.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_lake_area",

            "value":
                float(
                    lake_current.sum()
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
                "current_lake_with_bathy_pixels",

            "value":
                int(
                    current_with_bathy.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_lake_without_bathy_pixels",

            "value":
                int(
                    current_without_bathy.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "current_lake_without_bathy_area",

            "value":
                float(
                    current_without_bathy.sum()
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
                "shore_without_bathy_pixels",

            "value":
                int(
                    shore_without_bathy.sum()
                ),

            "unit":
                "pixels",
        },
        {
            "metric":
                "shore_without_bathy_area",

            "value":
                float(
                    shore_without_bathy.sum()
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
                "missing_components",

            "value":
                int(
                    n_components
                ),

            "unit":
                "count",
        },
    ]
)


                                                                               
               
                                                                               

banner(
    "ESCRITURA 16B4a"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


uint_profile = profile_20m.copy()


uint_profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate",
)


def write_uint8(
    path,
    array,
    description
):

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


write_uint8(
    OUT_LAKE,
    lake_current,
    "lago_actual_referencia_Sentinel_2024_2026_50pct"
)


write_uint8(
    OUT_NO_BATHY,
    current_without_bathy,
    "agua_actual_sin_cobertura_batimetrica_AMSCLAE"
)


write_uint8(
    OUT_SHORE,
    shore,
    "corredor_litoral_Sentinel_20_80pct"
)


write_uint8(
    OUT_CLASSES,
    classes,
    "clases_cobertura_lago_actual"
)


components_df.to_csv(
    OUT_COMPONENTS,
    index=False,
    encoding="utf-8-sig"
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
    classes,
    vmin=0,
    vmax=4
)


ax.set_title(
    "16B4a - Lago actual independiente de AMSCLAE\n"
    "1=lago+AMSCLAE | 2=lago SIN AMSCLAE | "
    "3=litoral+AMSCLAE | 4=litoral SIN AMSCLAE"
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
        "16B4a",

    "status":
        "CURRENT_LAKE_MASK_AUDIT_AFTER_OBSOLETE_BRANCH_CLEANUP",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "principle":
        (
            "Current lake geometry is defined by Sentinel 2024-2026, "
            "not by the AMSCLAE bathymetry footprint. AMSCLAE NoData "
            "must never be interpreted as land."
        ),

    "current_lake_definition": {
        "source":
            str(
                reference_path
            ),

        "meaning":
            (
                "Sentinel modern reference mask at 50 percent water occurrence "
                "for 2024-2026."
            ),

        "pixels":
            int(
                lake_current.sum()
            ),

        "area_km2":
            float(
                lake_current.sum()
                *
                pixel_area_m2
                /
                1e6
            ),
    },

    "shore_corridor": {
        "source":
            str(
                shore_path
            ),

        "meaning":
            "Sentinel 20-80 percent modern littoral transition band.",

        "pixels":
            int(
                shore.sum()
            ),

        "area_km2":
            float(
                shore.sum()
                *
                pixel_area_m2
                /
                1e6
            ),
    },

    "amsclae_coverage": {
        "source":
            str(
                PATH_AMSCLAE.resolve()
            ),

        "native_valid_pixels":
            int(
                bathy_valid_native.sum()
            ),

        "coverage_reprojection":
            "nearest-neighbor reprojected validity mask only",
    },

    "current_water_without_bathymetry": {
        "pixels":
            int(
                current_without_bathy.sum()
            ),

        "area_km2":
            float(
                current_without_bathy.sum()
                *
                pixel_area_m2
                /
                1e6
            ),

        "fraction_of_current_lake_pct":
            float(
                100.0
                *
                current_without_bathy.sum()
                /
                max(
                    int(
                        lake_current.sum()
                    ),
                    1
                )
            ),

        "components":
            int(
                n_components
            ),
    },

    "classes": {
        "0":
            "outside",

        "1":
            "current lake with AMSCLAE coverage",

        "2":
            "current lake without AMSCLAE coverage",

        "3":
            "Sentinel 20-80 littoral band with AMSCLAE coverage",

        "4":
            "Sentinel 20-80 littoral band without AMSCLAE coverage",
    },

    "files_sha256": {
        OUT_LAKE.name:
            sha256_file(
                OUT_LAKE
            ),

        OUT_NO_BATHY.name:
            sha256_file(
                OUT_NO_BATHY
            ),

        OUT_SHORE.name:
            sha256_file(
                OUT_SHORE
            ),

        OUT_CLASSES.name:
            sha256_file(
                OUT_CLASSES
            ),
    },

    "next_step":
        (
            "16B4b: build the shoreline-oscillation surface only where current "
            "Sentinel water lacks AMSCLAE bathymetry, without requiring vertical "
            "continuity with the AMSCLAE edge."
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
    "DECISION QC 16B4a"
)


gap_area = (
    current_without_bathy.sum()
    *
    pixel_area_m2
    /
    1e6
)


print(
    "[QC] lago actual Sentinel =",
    f"{lake_current.sum()*pixel_area_m2/1e6:.4f} km2"
)


print(
    "[QC] agua actual sin AMSCLAE =",
    f"{gap_area:.4f} km2"
)


print(
    "[QC] componentes sin batimetria =",
    n_components
)


print(
    "[QC] franja litoral sin AMSCLAE =",
    f"{shore_without_bathy.sum()*pixel_area_m2/1e6:.4f} km2"
)


print()


if current_without_bathy.any():

    print(
        "[PASS - PROBLEMA CONFIRMADO Y DELIMITADO]"
    )


    print(
        "Existe agua actual Sentinel fuera de la cobertura AMSCLAE."
    )


    print(
        "Esa agua queda explicitamente preservada como LAGO y NO como tierra."
    )


    print()


    print(
        "[NEXT] 16B4b:"
    )


    print(
        "construir la superficie de oscilacion litoral en esas zonas "
        "sin exigir continuidad vertical con el borde AMSCLAE."
    )


else:

    print(
        "[NO GAP]"
    )


    print(
        "La mascara moderna no contiene agua fuera de AMSCLAE; "
        "revisar entonces la definicion Sentinel de lago actual."
    )


banner(
    "FIN PASO 16B4a"
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
