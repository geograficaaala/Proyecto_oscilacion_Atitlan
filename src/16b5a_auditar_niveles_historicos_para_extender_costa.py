                       






































































__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


                                                                               
          
                                                                               

ROOT = Path(__file__).resolve().parents[1]

PATH_LEVEL = (
    ROOT
    / "data"
    / "processed"
    / "nivel_canonico.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "costa_actual_maestra_16B4"
    / "auditoria_niveles_historicos_16B5a"
)

OUT_ALL = (
    OUT_DIR
    / "16B5a_todos_los_niveles_ordenados.csv"
)

OUT_CANDIDATES = (
    OUT_DIR
    / "16B5a_candidatos_imagenes_historicas.csv"
)

OUT_WINDOWS = (
    OUT_DIR
    / "16B5a_ventanas_busqueda_satelital.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "16B5a_resumen.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "16B5a_metadata.json"
)

OUT_FIGURE = (
    OUT_DIR
    / "16B5a_serie_nivel_y_rango_observado.png"
)


                                                                               
               
                                                                               

H_EXACT_MIN = 1552.450
H_EXACT_MAX = 1554.050

                                                                             
SEARCH_HALF_WINDOW_DAYS = 5

                                                             
                                                                
MIN_DAYS_BETWEEN_SELECTED = 10

                    
A_MAX = H_EXACT_MIN
B_MAX = 1552.750
C_MAX = 1553.250


                                                                               
              
                                                                               

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


def find_date_column(df):

    preferred = [
        "obs_date",
        "date",
        "fecha",
        "datetime",
        "time",
    ]

    for col in preferred:

        if col in df.columns:
            return col

    for col in df.columns:

        parsed = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        if parsed.notna().mean() >= 0.90:
            return col

    raise RuntimeError(
        "No se encontro una columna de fecha interpretable."
    )


def select_spaced(
    df,
    date_col,
    min_days
):

    if len(df) == 0:
        return df.copy()

                                  
    work = df.sort_values(
        [
            "nivel_final_m",
            date_col,
        ]
    ).copy()

    selected_rows = []

    selected_dates = []

    for idx, row in work.iterrows():

        d = row[
            date_col
        ]

        if all(
            abs(
                (
                    d
                    -
                    prev
                ).days
            )
            >=
            min_days
            for prev in selected_dates
        ):

            selected_rows.append(
                idx
            )

            selected_dates.append(
                d
            )

    return (
        work.loc[
            selected_rows
        ]
        .sort_values(
            date_col
        )
        .reset_index(
            drop=True
        )
    )


                                                                               
                 
                                                                               

banner(
    "PASO 16B5a - LOCALIZACION"
)


if not PATH_LEVEL.exists():

    raise FileNotFoundError(
        PATH_LEVEL
    )


print(
    "[FILE]",
    PATH_LEVEL
)


                                                                               
                        
                                                                               

banner(
    "QC 1 - NIVEL CANONICO COMPLETO"
)


df = pd.read_csv(
    PATH_LEVEL
)


date_col = find_date_column(
    df
)


if "nivel_final_m" not in df.columns:

    raise RuntimeError(
        "No existe la columna nivel_final_m."
    )


df[
    "_date"
] = pd.to_datetime(
    df[
        date_col
    ],
    errors="coerce"
)


df[
    "nivel_final_m"
] = pd.to_numeric(
    df[
        "nivel_final_m"
    ],
    errors="coerce"
)


df = (
    df[
        df[
            "_date"
        ].notna()
        &
        df[
            "nivel_final_m"
        ].notna()
    ]
    .copy()
    .sort_values(
        "_date"
    )
    .reset_index(
        drop=True
    )
)


if len(df) == 0:

    raise RuntimeError(
        "nivel_canonico.csv no contiene observaciones validas."
    )


print(
    "[N] =",
    len(
        df
    )
)


print(
    "[DATE RANGE] =",
    df[
        "_date"
    ].min().date(),
    "a",
    df[
        "_date"
    ].max().date()
)


print(
    "[LEVEL RANGE] =",
    f"{df['nivel_final_m'].min():.3f} - "
    f"{df['nivel_final_m'].max():.3f} m"
)


                                                                               
                       
                                                                               

banner(
    "QC 2 - MINIMOS HISTORICOS"
)


hist_min = float(
    df[
        "nivel_final_m"
    ].min()
)


idx_min = df[
    "nivel_final_m"
].idxmin()


date_min = df.loc[
    idx_min,
    "_date"
]


delta_vs_exact_min = (
    hist_min
    -
    H_EXACT_MIN
)


print(
    "[HIST MIN] =",
    f"{hist_min:.3f} m"
)


print(
    "[HIST MIN DATE] =",
    date_min.date()
)


print(
    "[HIST MIN - EXACT 2024/26 MIN] =",
    f"{delta_vs_exact_min:+.3f} m"
)


print()


print(
    "[10 LOWEST OBSERVATIONS]"
)


print(
    df[
        [
            "_date",
            "nivel_final_m",
        ]
    ]
    .nsmallest(
        10,
        "nivel_final_m"
    )
    .to_string(
        index=False,
        formatters={
            "nivel_final_m":
                "{:.3f}".format,
        }
    )
)


                                                                               
               
                                                                               

banner(
    "QC 3 - CANDIDATOS PARA COSTA HISTORICA"
)


candidate = df.copy()


candidate[
    "priority"
] = "NONE"


candidate.loc[
    candidate[
        "nivel_final_m"
    ]
    <
    A_MAX,
    "priority"
] = "A_BELOW_CURRENT_OBSERVED_MIN"


candidate.loc[
    (
        candidate[
            "nivel_final_m"
        ]
        >=
        A_MAX
    )
    &
    (
        candidate[
            "nivel_final_m"
        ]
        <=
        B_MAX
    ),
    "priority"
] = "B_LOW_BAND"


candidate.loc[
    (
        candidate[
            "nivel_final_m"
        ]
        >
        B_MAX
    )
    &
    (
        candidate[
            "nivel_final_m"
        ]
        <=
        C_MAX
    ),
    "priority"
] = "C_SECONDARY_LOW"


candidate = candidate[
    candidate[
        "priority"
    ]
    !=
    "NONE"
].copy()


counts = (
    candidate[
        "priority"
    ]
    .value_counts()
)


for label in [
    "A_BELOW_CURRENT_OBSERVED_MIN",
    "B_LOW_BAND",
    "C_SECONDARY_LOW",
]:

    print(
        f"[{label}] =",
        int(
            counts.get(
                label,
                0
            )
        )
    )


                                                                               
                                                
                                                                               

banner(
    "QC 4 - VENTANAS DE BUSQUEDA SATELITAL"
)


priority_order = [
    "A_BELOW_CURRENT_OBSERVED_MIN",
    "B_LOW_BAND",
    "C_SECONDARY_LOW",
]


selected_blocks = []


for priority in priority_order:

    sub = candidate[
        candidate[
            "priority"
        ]
        ==
        priority
    ].copy()


    spaced = select_spaced(
        sub,
        "_date",
        MIN_DAYS_BETWEEN_SELECTED
    )


    if len(spaced) > 0:

        selected_blocks.append(
            spaced
        )


if len(selected_blocks) > 0:

    selected = pd.concat(
        selected_blocks,
        ignore_index=True
    )

else:

    selected = candidate.iloc[
        0:0
    ].copy()


selected = selected.sort_values(
    [
        "nivel_final_m",
        "_date",
    ]
).reset_index(
    drop=True
)


selected[
    "search_start"
] = (
    selected[
        "_date"
    ]
    -
    pd.to_timedelta(
        SEARCH_HALF_WINDOW_DAYS,
        unit="D"
    )
)


selected[
    "search_end"
] = (
    selected[
        "_date"
    ]
    +
    pd.to_timedelta(
        SEARCH_HALF_WINDOW_DAYS,
        unit="D"
    )
)


selected[
    "delta_vs_1552_450_m"
] = (
    selected[
        "nivel_final_m"
    ]
    -
    H_EXACT_MIN
)


print(
    "[SELECTED WINDOWS] =",
    len(
        selected
    )
)


if len(selected) > 0:

    print(
        selected[
            [
                "_date",
                "nivel_final_m",
                "priority",
                "search_start",
                "search_end",
            ]
        ]
        .head(
            30
        )
        .to_string(
            index=False,
            formatters={
                "nivel_final_m":
                    "{:.3f}".format,
            }
        )
    )


                                                                               
                   
                                                                               

banner(
    "DECISION 16B5a"
)


n_below = int(
    np.sum(
        df[
            "nivel_final_m"
        ]
        <
        H_EXACT_MIN
    )
)


if hist_min < H_EXACT_MIN:

    vertical_gain = (
        H_EXACT_MIN
        -
        hist_min
    )


    decision = (
        "HISTORICAL_EXTENSION_POSSIBLE"
    )


    print(
        "[PASS HISTORICAL EXTENSION POSSIBLE]"
    )


    print(
        "El registro contiene niveles inferiores al minimo exacto 2024-2026."
    )


    print(
        "[VERTICAL GAIN AVAILABLE] =",
        f"{vertical_gain:.3f} m"
    )


    print(
        "[OBSERVATIONS BELOW 1552.450] =",
        n_below
    )


    print()


    print(
        "[NEXT] 16B5b:"
    )


    print(
        "buscar en GEE Landsat/Sentinel alrededor de esas fechas y extraer "
        "costas historicas ligadas a niveles observados."
    )


else:

    vertical_gain = 0.0


    decision = (
        "NO_HISTORICAL_EXTENSION_BELOW_CURRENT_MIN"
    )


    print(
        "[NO HISTORICAL EXTENSION BELOW 1552.450]"
    )


    print(
        "El registro canonico no contiene un nivel inferior al minimo exacto "
        "ya observado con Sentinel/NASA."
    )


    print(
        "Buscar imagenes historicas puede densificar la costa baja, pero NO "
        "extenderla verticalmente hacia los niveles futuros mas bajos."
    )


    print()


    print(
        "[IMPLICATION]"
    )


    print(
        "Para niveles por debajo de 1552.450 m, una cartografia espacial "
        "continua requerira otra fuente batimetrica/geofisica o nuevo levantamiento. "
        "No debe inventarse usando AMSCLAE donde 16B4d0 la marco sin soporte."
    )


                                                                               
              
                                                                               

banner(
    "ESCRITURA 16B5a"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


all_out = df.copy()


all_out[
    "date"
] = all_out[
    "_date"
]


all_out = (
    all_out[
        [
            "date",
            "nivel_final_m",
        ]
    ]
    .sort_values(
        "nivel_final_m"
    )
    .reset_index(
        drop=True
    )
)


all_out.to_csv(
    OUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


candidate_out = candidate.copy()


candidate_out[
    "date"
] = candidate_out[
    "_date"
]


candidate_out[
    "delta_vs_1552_450_m"
] = (
    candidate_out[
        "nivel_final_m"
    ]
    -
    H_EXACT_MIN
)


candidate_out = candidate_out[
    [
        "date",
        "nivel_final_m",
        "delta_vs_1552_450_m",
        "priority",
    ]
].sort_values(
    [
        "nivel_final_m",
        "date",
    ]
)


candidate_out.to_csv(
    OUT_CANDIDATES,
    index=False,
    encoding="utf-8-sig"
)


windows_out = pd.DataFrame(
    columns=[
        "reference_date",
        "nivel_final_m",
        "delta_vs_1552_450_m",
        "priority",
        "search_start",
        "search_end",
    ]
)


if len(selected) > 0:

    windows_out = pd.DataFrame(
        {
            "reference_date":
                selected[
                    "_date"
                ].dt.date,

            "nivel_final_m":
                selected[
                    "nivel_final_m"
                ].astype(
                    float
                ),

            "delta_vs_1552_450_m":
                selected[
                    "delta_vs_1552_450_m"
                ].astype(
                    float
                ),

            "priority":
                selected[
                    "priority"
                ].astype(
                    str
                ),

            "search_start":
                selected[
                    "search_start"
                ].dt.date,

            "search_end":
                selected[
                    "search_end"
                ].dt.date,
        }
    )


windows_out.to_csv(
    OUT_WINDOWS,
    index=False,
    encoding="utf-8-sig"
)


summary_df = pd.DataFrame(
    [
        {
            "metric":
                "n_level_observations",
            "value":
                len(
                    df
                ),
            "unit":
                "count",
        },
        {
            "metric":
                "historical_min_level",
            "value":
                hist_min,
            "unit":
                "m",
        },
        {
            "metric":
                "historical_min_date",
            "value":
                str(
                    date_min.date()
                ),
            "unit":
                "date",
        },
        {
            "metric":
                "exact_2024_2026_min",
            "value":
                H_EXACT_MIN,
            "unit":
                "m",
        },
        {
            "metric":
                "vertical_gain_below_current_exact_min",
            "value":
                vertical_gain,
            "unit":
                "m",
        },
        {
            "metric":
                "observations_below_current_exact_min",
            "value":
                n_below,
            "unit":
                "count",
        },
        {
            "metric":
                "selected_satellite_search_windows",
            "value":
                len(
                    selected
                ),
            "unit":
                "count",
        },
    ]
)


summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


                                                                               
            
                                                                               

fig = plt.figure(
    figsize=(12, 6)
)


ax = fig.add_subplot(
    111
)


ax.plot(
    df[
        "_date"
    ],
    df[
        "nivel_final_m"
    ]
)


ax.axhline(
    H_EXACT_MIN,
    linestyle="--",
    label="Min exacto Sentinel/NASA 1552.450 m"
)


ax.axhline(
    H_EXACT_MAX,
    linestyle="--",
    label="Max exacto Sentinel/NASA 1554.050 m"
)


ax.scatter(
    [
        date_min
    ],
    [
        hist_min
    ],
    marker="o",
    label="Minimo historico canonico"
)


ax.set_title(
    "16B5a - Nivel canonico y rango costero directamente observado"
)


ax.set_ylabel(
    "Nivel del lago (m)"
)


ax.set_xlabel(
    "Fecha"
)


ax.grid(
    alpha=0.25
)


ax.legend()


fig.autofmt_xdate()


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
        "16B5a",

    "status":
        decision,

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "reason":
        (
            "16B4d0 showed insufficient lake-wide spatial compatibility between "
            "shallow AMSCLAE bathymetry and directly observed Sentinel/NASA shoreline transitions. "
            "Historical direct shoreline observation is therefore audited before any future "
            "shoreline extrapolation."
        ),

    "current_direct_observed_range": {
        "min_m":
            H_EXACT_MIN,

        "max_m":
            H_EXACT_MAX,
    },

    "canonical_level_record": {
        "n":
            int(
                len(
                    df
                )
            ),

        "date_min":
            str(
                df[
                    "_date"
                ].min().date()
            ),

        "date_max":
            str(
                df[
                    "_date"
                ].max().date()
            ),

        "level_min_m":
            hist_min,

        "level_max_m":
            float(
                df[
                    "nivel_final_m"
                ].max()
            ),

        "historical_min_date":
            str(
                date_min.date()
            ),
    },

    "historical_extension": {
        "observations_below_1552_450":
            n_below,

        "vertical_gain_m":
            float(
                vertical_gain
            ),

        "selected_search_windows":
            int(
                len(
                    selected
                )
            ),
    },

    "candidate_classes": {
        "A_BELOW_CURRENT_OBSERVED_MIN":
            "level < 1552.450 m; can extend direct shoreline range downward",

        "B_LOW_BAND":
            "1552.450 <= level <= 1552.750 m; densifies lowest observed band",

        "C_SECONDARY_LOW":
            "1552.750 < level <= 1553.250 m; secondary continuity support",
    },

    "important_limitations": [
        (
            "This step only identifies level dates; it does not prove that usable satellite imagery exists."
        ),
        (
            "If no canonical level observation is below 1552.450 m, historical imagery tied to this "
            "record cannot extend the shoreline vertically below the current exact minimum."
        ),
        (
            "Unsupported AMSCLAE shoreline zones must not be silently used to create continuous future maps."
        ),
    ],

    "sha256": {
        OUT_ALL.name:
            sha256_file(
                OUT_ALL
            ),

        OUT_CANDIDATES.name:
            sha256_file(
                OUT_CANDIDATES
            ),

        OUT_WINDOWS.name:
            sha256_file(
                OUT_WINDOWS
            ),

        OUT_SUMMARY.name:
            sha256_file(
                OUT_SUMMARY
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
    "FIN PASO 16B5a"
)


print(
    "[STATUS]",
    decision
)


print(
    "[OUTPUT DIRECTORY]"
)


print(
    OUT_DIR
)
