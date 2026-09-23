from __future__ import annotations
__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VERSION = "06_cmip6_local_csv_v1.1.0"
PROYECTO = Path(__file__).resolve().parents[1]
FORZAR_REPROCESO = False

MODELOS = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2",
    "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3", "GFDL-ESM4",
    "GISS-E2-1-G", "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6",
    "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL",
]

ESCENARIOS = ["historical", "ssp245", "ssp585"]

PERIODOS = {
    "historical": ("1981-01-01", "2014-12-31"),
    "ssp245": ("2015-01-01", "2100-12-31"),
    "ssp585": ("2015-01-01", "2100-12-31"),
}

COLUMNAS = [
    "date",
    "year",
    "month",
    "day",
    "model",
    "scenario",
    "pr_mm_day",
    "tas_c",
    "tasmin_c",
    "tasmax_c",
    "rh_pct",
    "rsds_mj_m2_day",
    "rlds_mj_m2_day",
    "sfcWind_ms",
]

VARIABLES = [
    "pr_mm_day",
    "tas_c",
    "tasmin_c",
    "tasmax_c",
    "rh_pct",
    "rsds_mj_m2_day",
    "rlds_mj_m2_day",
    "sfcWind_ms",
]

UNIDADES = {
    "pr_mm_day": "mm/day",
    "tas_c": "degC",
    "tasmin_c": "degC",
    "tasmax_c": "degC",
    "rh_pct": "%",
    "rsds_mj_m2_day": "MJ/m2/day",
    "rlds_mj_m2_day": "MJ/m2/day",
    "sfcWind_ms": "m/s",
}

MODELOS_365 = {
    "CanESM5",
    "CMCC-ESM2",
    "GFDL-ESM4",
    "GISS-E2-1-G",
}

MODELOS_360 = {
    "HadGEM3-GC31-LL",
    "UKESM1-0-LL",
}

ORDEN_M = {m: i for i, m in enumerate(MODELOS)}
ORDEN_S = {s: i for i, s in enumerate(ESCENARIOS)}
ORDEN_V = {v: i for i, v in enumerate(VARIABLES)}

RAW = PROYECTO / "data" / "cmip6" / "raw"
PROCESADOS = PROYECTO / "data" / "cmip6" / "processed"
RESULTADOS = PROYECTO / "outputs" / "cmip6"
CACHE = PROYECTO / "data" / "cache"
CONFIG = PROYECTO / "config"
LOGS = PROYECTO / "logs"

ESTADO = CACHE / "estado_06_cmip6.json"
CONFIG_ARCHIVO = CONFIG / "cmip6_local.json"
LOG_ARCHIVO = LOGS / "06_preparar_cmip6.log"


class ErrorEstructural(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)

    return h.hexdigest()


def escribir_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    tmp.replace(path)


def escribir_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def logger_config() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cmip6_paso06")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(
        LOG_ARCHIVO,
        encoding="utf-8",
    )

    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)

    sh.setFormatter(fmt)

    logger.addHandler(fh)

    logger.addHandler(sh)

    return logger


def nombre_esperado(
    modelo: str,
    escenario: str,
) -> str:

    return (
        f"ATITLAN_CMIP6_"
        f"{modelo.replace('-', '_')}_"
        f"{escenario}.csv"
    )


def indice_esperado(
    escenario: str,
) -> pd.DatetimeIndex:

    inicio, fin = PERIODOS[escenario]

    return pd.date_range(
        inicio,
        fin,
        freq="D",
    )


def dias_anio(
    anio: int,
) -> int:

    return (
        366
        if pd.Timestamp(
            anio,
            12,
            31,
        ).is_leap_year
        else 365
    )


def descubrir_archivos() -> list[Path]:

    if not RAW.exists():
        raise FileNotFoundError(
            f"No existe: {RAW}"
        )

    encontrados = sorted(
        [
            p
            for p in RAW.glob("*.csv")
            if p.is_file()
        ],
        key=lambda x: x.name.lower(),
    )

    esperados = {
        nombre_esperado(m, s).lower():
        nombre_esperado(m, s)
        for m in MODELOS
        for s in ESCENARIOS
    }

    mapa = {
        p.name.lower(): p
        for p in encontrados
    }

    faltantes = [
        esperados[k]
        for k in esperados
        if k not in mapa
    ]

    extras = [
        p.name
        for k, p in mapa.items()
        if k not in esperados
    ]

    if (
        len(encontrados) != 45
        or faltantes
        or extras
    ):

        msg = [
            (
                "Se esperaban 45 CSV y se encontraron "
                f"{len(encontrados)}."
            )
        ]

        if faltantes:
            msg.append(
                "Faltantes:\n"
                + "\n".join(
                    f"  - {x}"
                    for x in faltantes
                )
            )

        if extras:
            msg.append(
                "Extras:\n"
                + "\n".join(
                    f"  - {x}"
                    for x in extras
                )
            )

        raise ErrorEstructural(
            "\n".join(msg)
        )

    return encontrados


def construir_manifest(
    archivos: list[Path],
    logger: logging.Logger,
) -> pd.DataFrame:

    filas = []

    for i, path in enumerate(
        archivos,
        start=1,
    ):

        logger.info(
            "SHA-256 %02d/45: %s",
            i,
            path.name,
        )

        filas.append(
            {
                "filename": path.name,
                "path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "mtime_ns": int(
                    path.stat().st_mtime_ns
                ),
                "sha256": sha256(path),
            }
        )

    return pd.DataFrame(filas)


def fingerprint(
    manifest: pd.DataFrame,
) -> str:

    payload = {
        "version": VERSION,
        "files": (
            manifest[
                [
                    "filename",
                    "size_bytes",
                    "sha256",
                ]
            ]
            .sort_values("filename")
            .to_dict("records")
        ),
        "models": MODELOS,
        "scenarios": ESCENARIOS,
        "periods": PERIODOS,
        "columns": COLUMNAS,
    }

    texto = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def cache_ok(
    fp: str,
) -> tuple[bool, dict | None]:

    if not ESTADO.exists():
        return False, None

    try:

        estado = json.loads(
            ESTADO.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return False, None

    if estado.get("fingerprint") != fp:
        return False, estado

    outputs = [
        Path(x)
        for x in estado.get(
            "outputs",
            [],
        )
    ]

    return (
        bool(
            outputs
            and all(
                x.exists()
                for x in outputs
            )
        ),
        estado,
    )


def registrar_anomalias(
    df,
    mask,
    variable,
    regla,
    archivo,
    salida,
):

    for idx in df.index[mask]:

        salida.append(
            {
                "source_file": archivo,
                "model": str(
                    df.at[
                        idx,
                        "model",
                    ]
                ),
                "scenario": str(
                    df.at[
                        idx,
                        "scenario",
                    ]
                ),
                "date": (
                    df.at[
                        idx,
                        "date",
                    ]
                    .date()
                    .isoformat()
                ),
                "variable": variable,
                "value": float(
                    df.at[
                        idx,
                        variable,
                    ]
                ),
                "rule": regla,
                "severity": "WARNING",
                "action": (
                    "REPORT_ONLY_"
                    "NO_AUTOMATIC_MODIFICATION"
                ),
            }
        )


def validar_archivo(
    path: Path,
    hash_archivo: str,
):

    raw = pd.read_csv(
        path,
        low_memory=False,
    )

    reales = list(
        raw.columns
    )

    faltantes = [
        c
        for c in COLUMNAS
        if c not in reales
    ]

    extras = [
        c
        for c in reales
        if c not in COLUMNAS
    ]

    if faltantes:

        raise ErrorEstructural(
            f"{path.name}: "
            f"faltan columnas {faltantes}"
        )

    df = raw[
        COLUMNAS
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    for c in (
        ["year", "month", "day"]
        + VARIABLES
    ):

        original = df[c].notna()

        conv = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        malos = int(
            (
                original
                & conv.isna()
            ).sum()
        )

        if malos:

            raise ErrorEstructural(
                f"{path.name}: "
                f"{c} tiene "
                f"{malos} valores "
                f"no numéricos"
            )

        df[c] = conv

    df["model"] = (
        df["model"]
        .astype("string")
    )

    df["scenario"] = (
        df["scenario"]
        .astype("string")
    )

    nulos = df.isna().sum()

    if int(
        nulos.sum()
    ):

        detalle = ", ".join(
            f"{c}={int(n)}"
            for c, n
            in nulos.items()
            if int(n)
        )

        raise ErrorEstructural(
            f"{path.name}: "
            f"valores faltantes: "
            f"{detalle}"
        )

    modelos = [
        str(x)
        for x
        in df[
            "model"
        ].unique()
    ]

    escenarios = [
        str(x)
        for x
        in df[
            "scenario"
        ].unique()
    ]

    if (
        len(modelos) != 1
        or len(escenarios) != 1
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "debe contener un solo "
            "modelo y un solo escenario"
        )

    modelo = modelos[0]

    escenario = escenarios[0]

    if (
        modelo not in MODELOS
        or escenario not in ESCENARIOS
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "modelo/escenario no esperado: "
            f"{modelo}/{escenario}"
        )

    if (
        path.name.lower()
        != nombre_esperado(
            modelo,
            escenario,
        ).lower()
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "nombre incompatible "
            "con contenido "
            f"{modelo}/{escenario}"
        )

    if int(
        df[
            "date"
        ].duplicated().sum()
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "fechas duplicadas"
        )

    mismatch = (
        (
            df["year"]
            != df[
                "date"
            ].dt.year
        )
        |
        (
            df["month"]
            != df[
                "date"
            ].dt.month
        )
        |
        (
            df["day"]
            != df[
                "date"
            ].dt.day
        )
    )

    if int(
        mismatch.sum()
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "date no coincide "
            "con year/month/day"
        )

    esperado = indice_esperado(
        escenario
    )

    observado = pd.DatetimeIndex(
        df["date"]
    )

    faltan = esperado.difference(
        observado
    )

    sobran = observado.difference(
        esperado
    )

    if (
        len(df)
        != len(esperado)
        or len(faltan)
        or len(sobran)
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "cobertura inválida; "
            f"filas={len(df)}, "
            f"faltantes={len(faltan)}, "
            f"fuera_periodo={len(sobran)}"
        )

    inicio, fin = PERIODOS[
        escenario
    ]

    if (
        df["date"].min()
        != pd.Timestamp(inicio)
        or
        df["date"].max()
        != pd.Timestamp(fin)
    ):

        raise ErrorEstructural(
            f"{path.name}: "
            "periodo incorrecto"
        )

    for anio, n in (
        df.groupby("year")
        .size()
        .items()
    ):

        if (
            int(n)
            != dias_anio(
                int(anio)
            )
        ):

            raise ErrorEstructural(
                f"{path.name}: "
                f"año {int(anio)} "
                f"tiene {int(n)} filas"
            )

    df["year"] = (
        df["year"]
        .astype("int16")
    )

    df["month"] = (
        df["month"]
        .astype("int8")
    )

    df["day"] = (
        df["day"]
        .astype("int8")
    )

    for c in VARIABLES:

        df[c] = (
            df[c]
            .astype("float64")
        )

    anomalias = []

    reglas = [
        (
            "pr_mm_day",
            df["pr_mm_day"] < 0,
            "precipitation_negative",
        ),
        (
            "pr_mm_day",
            df["pr_mm_day"] > 500,
            "precipitation_gt_500_mm_day",
        ),
        (
            "rh_pct",
            df["rh_pct"] < 0,
            "relative_humidity_lt_0_pct",
        ),
        (
            "rh_pct",
            df["rh_pct"] > 100,
            "relative_humidity_gt_100_pct",
        ),
        (
            "rsds_mj_m2_day",
            df[
                "rsds_mj_m2_day"
            ] < 0,
            "shortwave_negative",
        ),
        (
            "rsds_mj_m2_day",
            df[
                "rsds_mj_m2_day"
            ] > 60,
            "shortwave_gt_60_MJ_m2_day",
        ),
        (
            "rlds_mj_m2_day",
            df[
                "rlds_mj_m2_day"
            ] < 0,
            "longwave_negative",
        ),
        (
            "rlds_mj_m2_day",
            df[
                "rlds_mj_m2_day"
            ] > 60,
            "longwave_gt_60_MJ_m2_day",
        ),
        (
            "sfcWind_ms",
            df[
                "sfcWind_ms"
            ] < 0,
            "wind_speed_negative",
        ),
        (
            "sfcWind_ms",
            df[
                "sfcWind_ms"
            ] > 50,
            "wind_speed_gt_50_m_s",
        ),
    ]

    for c in [
        "tas_c",
        "tasmin_c",
        "tasmax_c",
    ]:

        reglas.extend(
            [
                (
                    c,
                    df[c] < -50,
                    "temperature_lt_minus_50_C",
                ),
                (
                    c,
                    df[c] > 60,
                    "temperature_gt_60_C",
                ),
            ]
        )

    for (
        variable,
        mask,
        regla,
    ) in reglas:

        registrar_anomalias(
            df,
            mask,
            variable,
            regla,
            path.name,
            anomalias,
        )

    for idx in df.index[
        df["tasmin_c"]
        > df["tas_c"]
    ]:

        anomalias.append(
            {
                "source_file": path.name,
                "model": modelo,
                "scenario": escenario,
                "date": (
                    df.at[
                        idx,
                        "date",
                    ]
                    .date()
                    .isoformat()
                ),
                "variable": (
                    "tasmin_c/tas_c"
                ),
                "value": (
                    f"{df.at[idx, 'tasmin_c']} "
                    f"> "
                    f"{df.at[idx, 'tas_c']}"
                ),
                "rule": "tasmin_gt_tas",
                "severity": "WARNING",
                "action": (
                    "REPORT_ONLY_"
                    "NO_AUTOMATIC_MODIFICATION"
                ),
            }
        )

    for idx in df.index[
        df["tas_c"]
        > df["tasmax_c"]
    ]:

        anomalias.append(
            {
                "source_file": path.name,
                "model": modelo,
                "scenario": escenario,
                "date": (
                    df.at[
                        idx,
                        "date",
                    ]
                    .date()
                    .isoformat()
                ),
                "variable": (
                    "tas_c/tasmax_c"
                ),
                "value": (
                    f"{df.at[idx, 'tas_c']} "
                    f"> "
                    f"{df.at[idx, 'tasmax_c']}"
                ),
                "rule": "tas_gt_tasmax",
                "severity": "WARNING",
                "action": (
                    "REPORT_ONLY_"
                    "NO_AUTOMATIC_MODIFICATION"
                ),
            }
        )

    iguales = (
        df[VARIABLES]
        .eq(
            df[
                VARIABLES
            ].shift(1)
        )
        .all(
            axis=1
        )
    )

    candidatos = []

    for idx in df.index[
        iguales
    ]:

        fecha = df.at[
            idx,
            "date",
        ]

        candidatos.append(
            {
                "source_file": path.name,
                "model": modelo,
                "scenario": escenario,
                "date": (
                    fecha
                    .date()
                    .isoformat()
                ),
                "previous_date": (
                    (
                        fecha
                        - pd.Timedelta(
                            days=1
                        )
                    )
                    .date()
                    .isoformat()
                ),
                "candidate_reason": (
                    "all_8_climate_variables_"
                    "exactly_equal_previous_day"
                ),
            }
        )

    resumen = {
        "source_file": path.name,
        "sha256": hash_archivo,
        "rows": int(len(df)),
        "columns": int(len(reales)),
        "columns_exact_expected_order": (
            reales == COLUMNAS
        ),
        "extra_columns": ";".join(
            extras
        ),
        "model": modelo,
        "scenario": escenario,
        "date_min": (
            df["date"]
            .min()
            .date()
            .isoformat()
        ),
        "date_max": (
            df["date"]
            .max()
            .date()
            .isoformat()
        ),
        "missing_values_total": 0,
        "duplicate_dates": 0,
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "missing_dates": 0,
        "extra_dates": 0,
        "ymd_mismatch": 0,
        "calendar_fill_candidates": int(
            iguales.sum()
        ),
        "value_anomalies": int(
            len(anomalias)
        ),
        "structural_status": "OK",
    }

    return (
        df,
        resumen,
        anomalias,
        candidatos,
    )


def rellenos_esperados(
    modelo: str,
    escenario: str,
) -> int:

    inicio, fin = PERIODOS[
        escenario
    ]

    anios = list(
        range(
            pd.Timestamp(
                inicio
            ).year,
            pd.Timestamp(
                fin
            ).year + 1,
        )
    )

    bisiestos = sum(
        pd.Timestamp(
            y,
            12,
            31,
        ).is_leap_year
        for y in anios
    )

    if modelo in MODELOS_365:
        return int(
            bisiestos
        )

    if modelo in MODELOS_360:
        return int(
            5 * len(anios)
            + bisiestos
        )

    return 0


def qc_calendario(
    candidatos: pd.DataFrame,
) -> pd.DataFrame:

    conteos = {}

    if not candidatos.empty:

        g = (
            candidatos.groupby(
                [
                    "model",
                    "scenario",
                ]
            )
            .size()
        )

        conteos = {
            (
                str(m),
                str(s),
            ): int(n)
            for (
                m,
                s,
            ), n
            in g.items()
        }

    filas = []

    for modelo in MODELOS:

        for escenario in ESCENARIOS:

            observado = (
                conteos.get(
                    (
                        modelo,
                        escenario,
                    ),
                    0,
                )
            )

            esperado = (
                rellenos_esperados(
                    modelo,
                    escenario,
                )
            )

            if modelo in MODELOS_365:

                patron = (
                    "365_day_fill_"
                    "leap_day"
                )

            elif modelo in MODELOS_360:

                patron = (
                    "360_day_fill_"
                    "5_or_6_days"
                )

            else:

                patron = (
                    "no_documented_"
                    "fill_pattern_for_"
                    "selected_model"
                )

            filas.append(
                {
                    "model": modelo,
                    "scenario": escenario,
                    "documented_calendar_pattern": patron,
                    "expected_fill_candidates_from_documentation": esperado,
                    "observed_exact_repeat_candidates": observado,
                    "status": (
                        "OK_COINCIDE"
                        if observado == esperado
                        else "REVISAR"
                    ),
                }
            )

    return pd.DataFrame(
        filas
    )


def qc_anual(
    df: pd.DataFrame,
) -> pd.DataFrame:

    filas = []

    for (
        modelo,
        escenario,
        anio,
    ), g in df.groupby(
        [
            "model",
            "scenario",
            "year",
        ],
        observed=True,
        sort=False,
    ):

        anio = int(
            anio
        )

        esperado = dias_anio(
            anio
        )

        filas.append(
            {
                "model": str(
                    modelo
                ),
                "scenario": str(
                    escenario
                ),
                "year": anio,
                "rows": int(
                    len(g)
                ),
                "expected_rows_gregorian": esperado,
                "rows_match_expected": (
                    len(g)
                    == esperado
                ),
                "date_min": (
                    g["date"]
                    .min()
                    .date()
                    .isoformat()
                ),
                "date_max": (
                    g["date"]
                    .max()
                    .date()
                    .isoformat()
                ),
                "missing_values_total": int(
                    g.isna()
                    .sum()
                    .sum()
                ),
                "duplicate_dates": int(
                    g["date"]
                    .duplicated()
                    .sum()
                ),
            }
        )

    out = pd.DataFrame(
        filas
    )

    out["_m"] = (
        out["model"]
        .map(
            ORDEN_M
        )
    )

    out["_s"] = (
        out["scenario"]
        .map(
            ORDEN_S
        )
    )

    return (
        out.sort_values(
            [
                "_m",
                "_s",
                "year",
            ]
        )
        .drop(
            columns=[
                "_m",
                "_s",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def qc_variable_anual(
    df: pd.DataFrame,
) -> pd.DataFrame:

    partes = []

    claves = [
        "model",
        "scenario",
        "year",
    ]

    for variable in VARIABLES:

        g = df.groupby(
            claves,
            observed=True,
            sort=False,
        )[variable]

        x = (
            g.agg(
                n="size",
                missing=lambda z: int(
                    z.isna().sum()
                ),
                mean="mean",
                std="std",
                min="min",
                median="median",
                max="max",
            )
            .reset_index()
        )

        q01 = (
            g.quantile(
                0.01
            )
            .rename(
                "q01"
            )
            .reset_index()
        )

        q99 = (
            g.quantile(
                0.99
            )
            .rename(
                "q99"
            )
            .reset_index()
        )

        x = x.merge(
            q01,
            on=claves,
            how="left",
        )

        x = x.merge(
            q99,
            on=claves,
            how="left",
        )

        x.insert(
            3,
            "variable",
            variable,
        )

        x.insert(
            4,
            "expected_unit",
            UNIDADES[
                variable
            ],
        )

        partes.append(
            x
        )

    out = pd.concat(
        partes,
        ignore_index=True,
    )

    out["_m"] = (
        out["model"]
        .map(
            ORDEN_M
        )
    )

    out["_s"] = (
        out["scenario"]
        .map(
            ORDEN_S
        )
    )

    out["_v"] = (
        out["variable"]
        .map(
            ORDEN_V
        )
    )

    return (
        out.sort_values(
            [
                "_m",
                "_s",
                "year",
                "_v",
            ]
        )
        .drop(
            columns=[
                "_m",
                "_s",
                "_v",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def qc_unidades(
    df: pd.DataFrame,
) -> pd.DataFrame:

    filas = []

    for variable in VARIABLES:

        s = df[
            variable
        ].astype(
            float
        )

        minimo = float(
            s.min()
        )

        mediana = float(
            s.median()
        )

        maximo = float(
            s.max()
        )

        q99 = float(
            s.quantile(
                0.99
            )
        )

        if variable == "pr_mm_day":

            ok = (
                minimo >= -1e-12
                and maximo < 1000
                and q99 > 1
            )

        elif variable in {
            "tas_c",
            "tasmin_c",
            "tasmax_c",
        }:

            ok = (
                -100 < minimo < 80
                and -50 < mediana < 60
                and maximo < 100
            )

        elif variable in {
            "rsds_mj_m2_day",
            "rlds_mj_m2_day",
        }:

            ok = (
                -5 < minimo
                and 0 < mediana < 60
                and maximo < 100
            )

        elif variable == "rh_pct":

            ok = (
                -50 < minimo
                and 1 < mediana < 150
                and maximo < 250
            )

        else:

            ok = (
                abs(
                    mediana
                ) < 30
                and abs(
                    maximo
                ) < 100
                and abs(
                    minimo
                ) < 100
            )

        filas.append(
            {
                "variable": variable,
                "expected_unit": UNIDADES[
                    variable
                ],
                "n": int(
                    s.notna().sum()
                ),
                "missing": int(
                    s.isna().sum()
                ),
                "min": minimo,
                "q01": float(
                    s.quantile(
                        0.01
                    )
                ),
                "median": mediana,
                "q99": q99,
                "max": maximo,
                "status": (
                    "OK_ESCALA_COMPATIBLE"
                    if ok
                    else "REVISAR_ESCALA"
                ),
                "automatic_conversion_applied": False,
            }
        )

    return pd.DataFrame(
        filas
    )


def ordenar(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["_m"] = (
        out["model"]
        .map(
            ORDEN_M
        )
    )

    out["_s"] = (
        out["scenario"]
        .map(
            ORDEN_S
        )
    )

    return (
        out.sort_values(
            [
                "_m",
                "_s",
                "date",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_m",
                "_s",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def guardar_dataset(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[Path, str]:

    parquet = (
        PROCESADOS
        / "cmip6_diario.parquet"
    )

    pickle = (
        PROCESADOS
        / "cmip6_diario.pkl.gz"
    )

    try:

        df.to_parquet(
            parquet,
            index=False,
            compression="snappy",
        )

        if pickle.exists():
            pickle.unlink()

        return (
            parquet,
            "parquet",
        )

    except Exception as exc:

        logger.warning(
            "Parquet no disponible: %s",
            exc,
        )

        df.to_pickle(
            pickle,
            compression="gzip",
            protocol=5,
        )

        return (
            pickle,
            "pickle_gzip",
        )


def main() -> None:

    for carpeta in [
        RAW,
        PROCESADOS,
        RESULTADOS,
        CACHE,
        CONFIG,
        LOGS,
    ]:

        carpeta.mkdir(
            parents=True,
            exist_ok=True,
        )

    logger = logger_config()

    logger.info(
        "PASO 06 CMIP6 LOCAL"
    )

    logger.info(
        "Proyecto: %s",
        PROYECTO,
    )

    logger.info(
        "RAW: %s",
        RAW,
    )

    config = {
        "pipeline_version": VERSION,
        "project_root": str(
            PROYECTO
        ),
        "raw_dir": str(
            RAW
        ),
        "processed_dir": str(
            PROCESADOS
        ),
        "results_dir": str(
            RESULTADOS
        ),
        "expected_models": MODELOS,
        "expected_scenarios": ESCENARIOS,
        "expected_periods": PERIODOS,
        "expected_columns": COLUMNAS,
        "automatic_climate_value_modification": False,
        "bias_correction_in_step_06": False,
        "future_level_simulation_in_step_06": False,
    }

    escribir_json(
        CONFIG_ARCHIVO,
        config,
    )

    archivos = descubrir_archivos()

    manifest = construir_manifest(
        archivos,
        logger,
    )

    fp = fingerprint(
        manifest
    )

    if not FORZAR_REPROCESO:

        ok, estado = cache_ok(
            fp
        )

        if ok:

            print()

            print(
                "SIN CAMBIOS"
            )

            print(
                f"Fingerprint: {fp}"
            )

            print(
                "Dataset: "
                f"{estado.get('dataset_principal', 'N/D')}"
            )

            return

    hash_map = dict(
        zip(
            manifest[
                "filename"
            ],
            manifest[
                "sha256"
            ],
        )
    )

    frames = []

    resumenes = []

    anomalias = []

    candidatos = []

    pares = set()

    for i, path in enumerate(
        archivos,
        start=1,
    ):

        logger.info(
            "Validando %02d/45: %s",
            i,
            path.name,
        )

        (
            df_archivo,
            resumen,
            aa,
            cc,
        ) = validar_archivo(
            path,
            str(
                hash_map[
                    path.name
                ]
            ),
        )

        par = (
            resumen[
                "model"
            ],
            resumen[
                "scenario"
            ],
        )

        if par in pares:

            raise ErrorEstructural(
                f"Par duplicado: {par}"
            )

        pares.add(
            par
        )

        frames.append(
            df_archivo
        )

        resumenes.append(
            resumen
        )

        anomalias.extend(
            aa
        )

        candidatos.extend(
            cc
        )

    pares_esperados = {
        (
            modelo,
            escenario,
        )
        for modelo in MODELOS
        for escenario in ESCENARIOS
    }

    if pares != pares_esperados:

        raise ErrorEstructural(
            "Pares incompletos. "
            f"Faltan="
            f"{sorted(pares_esperados - pares)}"
        )

    df = ordenar(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    total_esperado = (
        len(MODELOS)
        * sum(
            len(
                indice_esperado(
                    escenario
                )
            )
            for escenario
            in ESCENARIOS
        )
    )

    if len(df) != total_esperado:

        raise ErrorEstructural(
            f"Filas totales="
            f"{len(df)}; "
            f"esperadas="
            f"{total_esperado}"
        )

    resumen_archivos = pd.DataFrame(
        resumenes
    )

    df_anomalias = pd.DataFrame(
        anomalias
    )

    df_candidatos = pd.DataFrame(
        candidatos
    )

    if df_anomalias.empty:

        df_anomalias = pd.DataFrame(
            columns=[
                "source_file",
                "model",
                "scenario",
                "date",
                "variable",
                "value",
                "rule",
                "severity",
                "action",
            ]
        )

    if df_candidatos.empty:

        df_candidatos = pd.DataFrame(
            columns=[
                "source_file",
                "model",
                "scenario",
                "date",
                "previous_date",
                "candidate_reason",
            ]
        )

    qc_year = qc_anual(
        df
    )

    qc_var = qc_variable_anual(
        df
    )

    qc_units = qc_unidades(
        df
    )

    qc_cal = qc_calendario(
        df_candidatos
    )

    rutas = {
        "manifest": (
            RESULTADOS
            / "manifest_cmip6_inputs.csv"
        ),
        "archivos": (
            RESULTADOS
            / "qc_cmip6_archivos.csv"
        ),
        "anual": (
            RESULTADOS
            / "qc_cmip6_anual.csv"
        ),
        "variable_anual": (
            RESULTADOS
            / "qc_cmip6_variable_anual.csv"
        ),
        "unidades": (
            RESULTADOS
            / "qc_cmip6_unidades.csv"
        ),
        "anomalias": (
            RESULTADOS
            / "qc_cmip6_anomalias_valores.csv"
        ),
        "candidatos": (
            RESULTADOS
            / "qc_cmip6_calendar_fill_candidates.csv"
        ),
        "calendario": (
            RESULTADOS
            / "qc_cmip6_calendar_modelo_escenario.csv"
        ),
        "resumen": (
            RESULTADOS
            / "resumen_06_cmip6.json"
        ),
    }

    escribir_csv(
        manifest,
        rutas["manifest"],
    )

    escribir_csv(
        resumen_archivos,
        rutas["archivos"],
    )

    escribir_csv(
        qc_year,
        rutas["anual"],
    )

    escribir_csv(
        qc_var,
        rutas["variable_anual"],
    )

    escribir_csv(
        qc_units,
        rutas["unidades"],
    )

    escribir_csv(
        df_anomalias,
        rutas["anomalias"],
    )

    escribir_csv(
        df_candidatos,
        rutas["candidatos"],
    )

    escribir_csv(
        qc_cal,
        rutas["calendario"],
    )

    dataset_path, formato = guardar_dataset(
        df,
        logger,
    )

    resumen_final = {
        "pipeline_version": VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
        "project_root": str(
            PROYECTO
        ),
        "input_directory": str(
            RAW
        ),
        "input_fingerprint": fp,
        "csv_files": len(
            archivos
        ),
        "models": int(
            df["model"].nunique()
        ),
        "scenarios": int(
            df["scenario"].nunique()
        ),
        "model_scenario_pairs": int(
            df[
                [
                    "model",
                    "scenario",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "rows_total": int(
            len(df)
        ),
        "columns_total": int(
            len(df.columns)
        ),
        "missing_values_total": int(
            df.isna()
            .sum()
            .sum()
        ),
        "duplicate_model_scenario_date": int(
            df.duplicated(
                [
                    "model",
                    "scenario",
                    "date",
                ]
            ).sum()
        ),
        "date_min": (
            df["date"]
            .min()
            .date()
            .isoformat()
        ),
        "date_max": (
            df["date"]
            .max()
            .date()
            .isoformat()
        ),
        "value_anomalies": int(
            len(
                df_anomalias
            )
        ),
        "calendar_fill_candidates": int(
            len(
                df_candidatos
            )
        ),
        "calendar_qc_revisar": int(
            (
                qc_cal["status"]
                == "REVISAR"
            ).sum()
        ),
        "unit_qc_revisar": int(
            (
                qc_units["status"]
                != "OK_ESCALA_COMPATIBLE"
            ).sum()
        ),
        "dataset_main": str(
            dataset_path
        ),
        "dataset_format": formato,
        "bias_correction_applied": False,
        "climate_values_automatically_modified": False,
        "future_hydrology_generated": False,
        "future_lake_level_simulated": False,
        "status": (
            "OK_STEP_06_"
            "LOCAL_CMIP6_PREPARED"
        ),
    }

    escribir_json(
        rutas["resumen"],
        resumen_final,
    )

    outputs = [
        str(
            dataset_path
        ),
        str(
            CONFIG_ARCHIVO
        ),
    ] + [
        str(x)
        for x
        in rutas.values()
    ]

    estado = {
        "pipeline_version": VERSION,
        "updated_utc": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
        "fingerprint": fp,
        "dataset_principal": str(
            dataset_path
        ),
        "dataset_formato": formato,
        "outputs": outputs,
        "status": "OK",
    }

    escribir_json(
        ESTADO,
        estado,
    )

    print()

    print(
        "=" * 78
    )

    print(
        "PASO 06 COMPLETADO CORRECTAMENTE"
    )

    print(
        "=" * 78
    )

    print(
        f"CSV procesados:               "
        f"{len(archivos)}"
    )

    print(
        f"Modelos:                      "
        f"{df['model'].nunique()}"
    )

    print(
        f"Escenarios:                   "
        f"{df['scenario'].nunique()}"
    )

    print(
        f"Pares modelo/escenario:       "
        f"{df[['model', 'scenario']].drop_duplicates().shape[0]}"
    )

    print(
        f"Filas totales:                "
        f"{len(df):,}"
    )

    print(
        f"NaN totales:                  "
        f"{int(df.isna().sum().sum())}"
    )

    print(
        f"Duplicados clave:              "
        f"{int(df.duplicated(['model', 'scenario', 'date']).sum())}"
    )

    print(
        f"Anomalías de valores:         "
        f"{len(df_anomalias)}"
    )

    print(
        f"Candidatos relleno calendario:"
        f"{len(df_candidatos)}"
    )

    print(
        f"QC calendario REVISAR:        "
        f"{int((qc_cal['status'] == 'REVISAR').sum())}"
    )

    print(
        f"QC unidades REVISAR:          "
        f"{int((qc_units['status'] != 'OK_ESCALA_COMPATIBLE').sum())}"
    )

    print(
        f"Dataset principal:            "
        f"{dataset_path}"
    )

    print(
        f"Formato:                      "
        f"{formato}"
    )

    print(
        f"Resultados:                   "
        f"{RESULTADOS}"
    )

    print(
        f"Cache:                        "
        f"{ESTADO}"
    )

    print(
        f"Log:                          "
        f"{LOG_ARCHIVO}"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
