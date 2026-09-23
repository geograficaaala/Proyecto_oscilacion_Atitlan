                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "reference_1981_2014" / "raw"
OUT_DIR = ROOT / "data" / "reference_1981_2014" / "procesado"
LOG_DIR = ROOT / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ERA5_FILES = [
    "ATITLAN_ERA5_LAND_1981_1989.csv",
    "ATITLAN_ERA5_LAND_1990_1999.csv",
    "ATITLAN_ERA5_LAND_2000_2009.csv",
    "ATITLAN_ERA5_LAND_2010_2014.csv",
]

CHIRPS_FILES = [
    "ATITLAN_CHIRPS_V2_1981_1989.csv",
    "ATITLAN_CHIRPS_V2_1990_1999.csv",
    "ATITLAN_CHIRPS_V2_2000_2009.csv",
    "ATITLAN_CHIRPS_V2_2010_2014.csv",
]

EXPECTED_START = pd.Timestamp("1981-01-01")
EXPECTED_END = pd.Timestamp("2014-12-31")

ERA5_REQUIRED = [
    "date", "year", "month", "doy", "dewpoint_c", "era_precip_mm",
    "evap_mm", "pet_mm", "pressure_kpa", "runoff_mm",
    "soil_water_1", "soil_water_2", "soil_water_3", "soil_water_4",
    "solar_mj_m2", "longwave_mj_m2", "subsurface_runoff_mm",
    "surface_runoff_mm", "tmax_c", "tmean_c", "tmin_c", "wind_ms"
]

CHIRPS_REQUIRED = [
    "date", "year", "month", "doy", "chirps_precip_mm"
]


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_and_concat(files, required_cols, label):
    frames = []
    fingerprints = {}

    for name in files:
        path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Falta archivo requerido: {path}")

        fingerprints[name] = sha256_file(path)
        df = pd.read_csv(path)

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{label}: {name} no contiene columnas requeridas: {missing_cols}"
            )

        df = df[required_cols].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if df["date"].isna().any():
            bad = int(df["date"].isna().sum())
            raise ValueError(f"{label}: {name} tiene {bad} fechas inválidas.")

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)

    return out, fingerprints


def qc_dates(df, label):
    duplicate_dates = int(df["date"].duplicated().sum())

    expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="D")
    actual = pd.DatetimeIndex(df["date"])

    missing_dates = expected.difference(actual)
    extra_dates = actual.difference(expected)

    if duplicate_dates > 0:
        raise ValueError(f"{label}: hay {duplicate_dates} fechas duplicadas.")

    if len(missing_dates) > 0:
        raise ValueError(
            f"{label}: faltan {len(missing_dates)} fechas. "
            f"Primeras: {list(missing_dates[:10])}"
        )

    if len(extra_dates) > 0:
        raise ValueError(
            f"{label}: hay {len(extra_dates)} fechas fuera del periodo esperado."
        )

    if len(df) != len(expected):
        raise ValueError(
            f"{label}: número de filas inesperado: {len(df)} vs {len(expected)}"
        )

    if df["date"].iloc[0] != EXPECTED_START:
        raise ValueError(f"{label}: fecha inicial incorrecta.")

    if df["date"].iloc[-1] != EXPECTED_END:
        raise ValueError(f"{label}: fecha final incorrecta.")

    return {
        "n_rows": int(len(df)),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "duplicate_dates": duplicate_dates,
        "missing_dates": int(len(missing_dates)),
        "extra_dates": int(len(extra_dates)),
    }


def qc_numeric(df, cols, label):
    report = {}

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s

        n_nan = int(s.isna().sum())
        n_inf = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum())
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()

        report[col] = {
            "nan": n_nan,
            "inf": n_inf,
            "min": None if finite.empty else float(finite.min()),
            "max": None if finite.empty else float(finite.max()),
            "mean": None if finite.empty else float(finite.mean()),
        }

        if n_nan > 0:
            raise ValueError(f"{label}: columna {col} tiene {n_nan} NaN.")
        if n_inf > 0:
            raise ValueError(f"{label}: columna {col} tiene {n_inf} infinitos.")

    return report


def main():
    print("=" * 72)
    print("PASO 08 - PREPARAR REFERENCIA HISTORICA 1981-2014")
    print("=" * 72)
    print("Directorio RAW:")
    print(RAW_DIR)
    print()

    era5, fp_era5 = read_and_concat(ERA5_FILES, ERA5_REQUIRED, "ERA5-Land")
    chirps, fp_chirps = read_and_concat(CHIRPS_FILES, CHIRPS_REQUIRED, "CHIRPS")

    qc_era5_dates = qc_dates(era5, "ERA5-Land")
    qc_chirps_dates = qc_dates(chirps, "CHIRPS")

    era5_num_cols = [c for c in ERA5_REQUIRED if c not in ["date", "year", "month", "doy"]]
    chirps_num_cols = ["chirps_precip_mm"]

    qc_era5_num = qc_numeric(era5, era5_num_cols, "ERA5-Land")
    qc_chirps_num = qc_numeric(chirps, chirps_num_cols, "CHIRPS")

                                                     
    for df, label in [(era5, "ERA5-Land"), (chirps, "CHIRPS")]:
        calc_year = df["date"].dt.year
        calc_month = df["date"].dt.month
        calc_doy = df["date"].dt.dayofyear

        if not np.array_equal(df["year"].astype(int).to_numpy(), calc_year.to_numpy()):
            raise ValueError(f"{label}: columna year no coincide con date.")
        if not np.array_equal(df["month"].astype(int).to_numpy(), calc_month.to_numpy()):
            raise ValueError(f"{label}: columna month no coincide con date.")
        if not np.array_equal(df["doy"].astype(int).to_numpy(), calc_doy.to_numpy()):
            raise ValueError(f"{label}: columna doy no coincide con date.")

                                          
    warnings = []

    nonnegative_cols = [
        "era_precip_mm", "runoff_mm", "surface_runoff_mm",
        "subsurface_runoff_mm", "solar_mj_m2", "longwave_mj_m2",
        "wind_ms", "chirps_precip_mm"
    ]

    merged_for_checks = era5.merge(
        chirps[["date", "chirps_precip_mm"]],
        on="date",
        how="inner",
        validate="one_to_one"
    )

    for col in nonnegative_cols:
        nneg = int((merged_for_checks[col] < 0).sum())
        if nneg > 0:
            warnings.append(f"{col}: {nneg} valores negativos.")

    if (era5["tmin_c"] > era5["tmean_c"]).any():
        warnings.append("Hay días con tmin_c > tmean_c.")
    if (era5["tmean_c"] > era5["tmax_c"]).any():
        warnings.append("Hay días con tmean_c > tmax_c.")
    if (era5["tmin_c"] > era5["tmax_c"]).any():
        warnings.append("Hay días con tmin_c > tmax_c.")

                                        
    ref = era5.merge(
        chirps[["date", "chirps_precip_mm"]],
        on="date",
        how="inner",
        validate="one_to_one"
    )

    ref["precip_consensus_mm"] = (
        ref["era_precip_mm"] + ref["chirps_precip_mm"]
    ) / 2.0

                
    front = [
        "date", "year", "month", "doy",
        "era_precip_mm", "chirps_precip_mm", "precip_consensus_mm"
    ]
    rest = [c for c in ref.columns if c not in front]
    ref = ref[front + rest]

    era5_out = OUT_DIR / "atitlan_era5_land_1981_2014.csv"
    chirps_out = OUT_DIR / "atitlan_chirps_v2_1981_2014.csv"
    ref_out = OUT_DIR / "atitlan_referencia_1981_2014.csv"
    manifest_out = OUT_DIR / "manifest_paso08.json"

    era5.to_csv(era5_out, index=False, date_format="%Y-%m-%d")
    chirps.to_csv(chirps_out, index=False, date_format="%Y-%m-%d")
    ref.to_csv(ref_out, index=False, date_format="%Y-%m-%d")

    manifest = {
        "paso": 8,
        "descripcion": "Preparacion referencia historica ERA5-Land + CHIRPS v2, 1981-2014",
        "python_version": sys.version,
        "periodo": {
            "inicio": str(EXPECTED_START.date()),
            "fin": str(EXPECTED_END.date()),
            "n_dias_esperados": int(len(pd.date_range(EXPECTED_START, EXPECTED_END, freq="D"))),
        },
        "inputs_sha256": {
            "era5": fp_era5,
            "chirps": fp_chirps,
        },
        "qc": {
            "era5_dates": qc_era5_dates,
            "chirps_dates": qc_chirps_dates,
            "era5_numeric": qc_era5_num,
            "chirps_numeric": qc_chirps_num,
            "warnings": warnings,
        },
        "outputs": {
            "era5": str(era5_out),
            "chirps": str(chirps_out),
            "referencia": str(ref_out),
        },
        "nota_precip_consensus": (
            "precip_consensus_mm se calcula como promedio simple diario "
            "de era_precip_mm y chirps_precip_mm, para mantener una "
            "referencia transparente y reproducible. No se aplica bias correction."
        ),
    }

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("QC temporal ERA5:", qc_era5_dates)
    print("QC temporal CHIRPS:", qc_chirps_dates)
    print()
    print("Archivos creados:")
    print(era5_out)
    print(chirps_out)
    print(ref_out)
    print(manifest_out)
    print()

    if warnings:
        print("ADVERTENCIAS DE QC:")
        for w in warnings:
            print(" -", w)
    else:
        print("QC físico básico: sin advertencias.")

    print()
    print("PASO 08 COMPLETADO CORRECTAMENTE.")


if __name__ == "__main__":
    main()
