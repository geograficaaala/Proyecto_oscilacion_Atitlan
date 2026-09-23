__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "rutas.json"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"
LOG_DIR = ROOT / "logs"
STATE_PATH = CACHE_DIR / "estado_01_preparar_datos.json"
CATALOG_PATH = ROOT / "data" / "catalogo_fuentes.csv"
QC_PATH = ROOT / "data" / "control_calidad_preparacion.csv"
OUTPUT_CONFIG_PATH = ROOT / "config" / "datasets_procesados.json"
MANIFEST_PATH = ROOT / "data" / "manifest_preparacion.json"

for d in [PROCESSED_DIR, CACHE_DIR, LOG_DIR, ROOT / "config", ROOT / "data"]:
    d.mkdir(parents=True, exist_ok=True)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"No existe {CONFIG_PATH}. Ejecuta primero 00_crear_estructura_proyecto.py")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    ROUTES = json.load(f)

for key, value in list(ROUTES.items()):
    candidate = Path(value)
    if not candidate.is_absolute():
        ROUTES[key] = str((ROOT / candidate).resolve())


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_info(name, path, required=False):
    p = Path(path)
    info = {
        "name": name,
        "path": str(p),
        "required": bool(required),
        "exists": p.exists(),
        "size_bytes": None,
        "modified_utc": None,
        "sha256": None,
    }
    if p.exists() and p.is_file():
        stat = p.stat()
        info["size_bytes"] = int(stat.st_size)
        info["modified_utc"] = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()
        info["sha256"] = sha256_file(p)
    return info


def strip_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_date_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                return col, parsed
    return None, None


def dataframe_summary(name, df, date_col=None):
    start = None
    end = None
    if date_col is not None and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        if d.notna().any():
            start = d.min().isoformat()
            end = d.max().isoformat()
    duplicated_rows = int(df.duplicated().sum()) if len(df) else 0
    total_cells = max(1, int(df.shape[0] * df.shape[1]))
    missing_cells = int(df.isna().sum().sum()) if len(df.columns) else 0
    return {
        "dataset": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_column": date_col,
        "start": start,
        "end": end,
        "duplicated_rows": duplicated_rows,
        "missing_cells": missing_cells,
        "missing_fraction": float(missing_cells / total_cells),
    }


def save_table(df, stem, date_format="%Y-%m-%d %H:%M:%S"):
    csv_path = PROCESSED_DIR / f"{stem}.csv"
    pkl_path = PROCESSED_DIR / f"{stem}.pkl"
    df.to_csv(csv_path, index=False, date_format=date_format)
    df.to_pickle(pkl_path)
    return {"csv": str(csv_path), "pkl": str(pkl_path)}


def safe_read_csv(path):
    return strip_columns(pd.read_csv(path, low_memory=False))


def safe_read_excel(path, preferred_sheet=None):
    xls = pd.ExcelFile(path)
    if preferred_sheet in xls.sheet_names:
        sheet = preferred_sheet
    else:
        sheet = xls.sheet_names[0]
    return strip_columns(pd.read_excel(path, sheet_name=sheet)), sheet, xls.sheet_names


def canonical_river(value):
    text = normalize_text(value)
    if not isinstance(text, str):
        return np.nan
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if "quiscab" in key:
        return "Quiscab"
    if "san_francisco" in key:
        return "San_Francisco"
    if "tzununa" in key:
        return "Tzununa"
    if "catarata" in key:
        return "La_Catarata"
    if "san_buenaventura" in key:
        return "San_Buenaventura"
    return key if key else np.nan


def normalize_month(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)) and np.isfinite(value):
        m = int(value)
        return m if 1 <= m <= 12 else np.nan
    text = normalize_text(value).lower().replace(".", "")
    mapping = {
        "ene": 1, "enero": 1,
        "feb": 2, "febrero": 2,
        "mar": 3, "marzo": 3,
        "abr": 4, "abril": 4,
        "may": 5, "mayo": 5,
        "jun": 6, "junio": 6,
        "jul": 7, "julio": 7,
        "ago": 8, "agosto": 8,
        "sep": 9, "sept": 9, "septiembre": 9, "setiembre": 9,
        "oct": 10, "octubre": 10,
        "nov": 11, "noviembre": 11,
        "dic": 12, "diciembre": 12,
    }
    return mapping.get(text, np.nan)


def clean_model_matrix(df):
    out = df.copy()
    if "date_obs" not in out.columns:
        raise KeyError("La matriz autoregresiva no contiene date_obs")
    if "nivel_final_m" not in out.columns:
        raise KeyError("La matriz autoregresiva no contiene nivel_final_m")
    out["date_obs"] = pd.to_datetime(out["date_obs"], errors="coerce")
    if "obs_date" in out.columns:
        out["obs_date"] = pd.to_datetime(out["obs_date"], errors="coerce")
    out["nivel_final_m"] = numeric(out["nivel_final_m"])
    out = out.dropna(subset=["date_obs", "nivel_final_m"]).sort_values("date_obs").reset_index(drop=True)
    return out


def clean_features_table(df, matrix_columns):
    if "feature" not in df.columns:
        raise KeyError("atitlan_features_autoregresivas.csv no contiene columna feature")
    raw = df["feature"].dropna().astype(str).str.strip().tolist()
    duplicated = sorted(set(pd.Series(raw)[pd.Series(raw).duplicated(keep=False)].tolist())) if raw else []
    unique = list(dict.fromkeys([x for x in raw if x]))
    present = [x for x in unique if x in matrix_columns]
    missing = [x for x in unique if x not in matrix_columns]
    clean = pd.DataFrame({"feature": present})
    return clean, raw, duplicated, missing


def clean_rivers(df):
    date_col = "Fecha de Aforo" if "Fecha de Aforo" in df.columns else None
    place_col = "Lugar" if "Lugar" in df.columns else None
    flow_lps_col = next((c for c in df.columns if "Caudal" in c and "l/s" in c), None)
    flow_m3s_col = next((c for c in df.columns if "Caudal" in c and "m3/s" in c), None)
    if date_col is None or place_col is None or flow_lps_col is None:
        raise KeyError("No se encontraron Fecha de Aforo, Lugar y Caudal (l/s) en rivers_data.xlsx")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["river"] = df[place_col].map(canonical_river)
    out["municipality"] = df["Municipio"].map(normalize_text) if "Municipio" in df.columns else np.nan
    out["flow_lps"] = numeric(df[flow_lps_col])
    out["flow_m3s"] = out["flow_lps"] / 1000.0
    if flow_m3s_col is not None:
        out["flow_m3s_reported"] = numeric(df[flow_m3s_col])
        out["flow_m3s_difference_reported_minus_lps"] = out["flow_m3s_reported"] - out["flow_m3s"]
    else:
        out["flow_m3s_reported"] = np.nan
        out["flow_m3s_difference_reported_minus_lps"] = np.nan
    if "X" in df.columns:
        out["x"] = numeric(df["X"])
    if "Y" in df.columns:
        out["y"] = numeric(df["Y"])
    if "MSNM" in df.columns:
        out["elevation_m"] = numeric(df["MSNM"])
    out = out.dropna(subset=["date", "river", "flow_m3s"])
    out = out.loc[out["flow_m3s"] >= 0].copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["month_date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    out = out.sort_values(["date", "river"]).reset_index(drop=True)
    return out


def build_river_monthly(rivers):
    grouped = (
        rivers.groupby(["month_date", "river"], as_index=False)["flow_m3s"]
        .mean()
    )
    pivot = grouped.pivot(index="month_date", columns="river", values="flow_m3s").reset_index()
    pivot.columns.name = None
    expected = ["Quiscab", "San_Francisco", "Tzununa", "La_Catarata", "San_Buenaventura"]
    for c in expected:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot["n_rivers_observed"] = pivot[expected].notna().sum(axis=1).astype(int)
    pivot["q_sum_available_m3s"] = pivot[expected].sum(axis=1, min_count=1)
    pivot["q_total_5rivers_m3s"] = pivot[expected].sum(axis=1, min_count=5)
    pivot["complete_5rivers"] = pivot["n_rivers_observed"].eq(5)
    return pivot.sort_values("month_date").reset_index(drop=True)


def clean_weather(df):
    required = ["Año", "Estación Climática", "Mes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en weather_data.xlsx: {missing}")
    out = pd.DataFrame()
    out["year"] = numeric(df["Año"])
    out["month"] = df["Mes"].map(normalize_month)
    out["station"] = df["Estación Climática"].map(normalize_text)
    mapping = {
        "Temp. Max. (°C)": "tmax_c",
        "Temp. Min. (°C)": "tmin_c",
        "Humedad (%)": "humidity_pct",
        "Precipitación (mm)": "precip_mm",
        "Vel. Max. Viento (Km/h)": "wind_max_kmh",
        "Rad. Solar Max. (W/m2)": "solar_max_wm2",
        "Rad. UV Max.": "uv_max",
    }
    for source, target in mapping.items():
        out[target] = numeric(df[source]) if source in df.columns else np.nan
    out = out.dropna(subset=["year", "month", "station"])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["month_date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1), errors="coerce")
    out = out.dropna(subset=["month_date"]).sort_values(["month_date", "station"]).reset_index(drop=True)
    return out


def build_weather_monthly(weather):
    value_cols = [c for c in ["tmax_c", "tmin_c", "humidity_pct", "precip_mm", "wind_max_kmh", "solar_max_wm2", "uv_max"] if c in weather.columns]
    agg = weather.groupby("month_date")[value_cols].mean(numeric_only=True).reset_index()
    counts = weather.groupby("month_date")["station"].nunique().rename("n_stations").reset_index()
    out = agg.merge(counts, on="month_date", how="left")
    for c in value_cols:
        valid = weather.groupby("month_date")[c].count().rename(f"n_{c}").reset_index()
        out = out.merge(valid, on="month_date", how="left")
    return out.sort_values("month_date").reset_index(drop=True)


def clean_generic_csv(df, name):
    out = df.copy()
    date_col, parsed = parse_date_column(out, ["date", "date_obs", "obs_date", "date_clim", "Fecha", "fecha"])
    if date_col is not None:
        out[date_col] = parsed
        out = out.loc[out[date_col].notna()].sort_values(date_col).reset_index(drop=True)
    return out, date_col


def clean_lake(df):
    out = df.copy()
    unnamed = [c for c in out.columns if str(c).lower().startswith("unnamed:")]
    if unnamed:
        out = out.drop(columns=unnamed)
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce")
        out = out.loc[out["Fecha"].notna()].sort_values("Fecha").reset_index(drop=True)
    return out


def clean_limnology(df):
    out = df.copy()
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce")
    if "Profuidad (m)" in out.columns:
        out["Profuidad (m)"] = numeric(out["Profuidad (m)"])
    subset = [c for c in ["Sitio", "Fecha"] if c in out.columns]
    if subset:
        out = out.dropna(subset=subset)
    sort_cols = [c for c in ["Sitio", "Fecha", "Profuidad (m)"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


FINAL_DIR = Path(ROUTES["dataset_ml"]).parent
SOURCES = {
    "matrix_autoregressive": {"path": Path(ROUTES["dataset_ml"]), "required": True},
    "features_autoregressive": {"path": Path(ROUTES["features_autoregresivas"]), "required": True},
    "climate_daily_features": {"path": FINAL_DIR / "atitlan_clima_diario_features.csv", "required": True},
    "climate_daily_clean": {"path": FINAL_DIR / "atitlan_clima_diario_limpio.csv", "required": False},
    "level_canonical": {"path": FINAL_DIR / "atitlan_nivel_canonico.csv", "required": True},
    "level_qc": {"path": FINAL_DIR / "atitlan_nivel_qc.csv", "required": False},
    "era5_raw": {"path": Path(ROUTES["era5"]), "required": False},
    "chirps_raw": {"path": Path(ROUTES["chirps"]), "required": False},
    "nasa_level_raw": {"path": Path(ROUTES["nasa_nivel"]), "required": False},
    "rivers": {"path": Path(ROUTES["rivers"]), "required": True},
    "weather": {"path": Path(ROUTES["weather"]), "required": True},
    "lake_legacy": {"path": Path(ROUTES["lake"]), "required": False},
    "limnology": {"path": Path(ROUTES["limnology"]), "required": False},
    "sarima_notebook": {"path": Path(ROUTES["sarima_notebook"]), "required": False},
}

catalog = [file_info(name, spec["path"], spec["required"]) for name, spec in SOURCES.items()]
catalog_df = pd.DataFrame(catalog)
catalog_df.to_csv(CATALOG_PATH, index=False)

missing_required = catalog_df.loc[catalog_df["required"] & ~catalog_df["exists"], ["name", "path"]]
if len(missing_required):
    raise FileNotFoundError("Faltan fuentes obligatorias:\n" + missing_required.to_string(index=False))

fingerprint = {
    "script_version": SCRIPT_VERSION,
    "sources": {row["name"]: row["sha256"] for row in catalog if row["exists"]},
}

expected_outputs = [
    PROCESSED_DIR / "matriz_modelo_autoregresiva.pkl",
    PROCESSED_DIR / "features_autoregresivas_unicas.pkl",
    PROCESSED_DIR / "clima_diario_features.pkl",
    PROCESSED_DIR / "nivel_canonico.pkl",
    PROCESSED_DIR / "rivers_clean.pkl",
    PROCESSED_DIR / "rivers_monthly.pkl",
    PROCESSED_DIR / "weather_clean.pkl",
    PROCESSED_DIR / "weather_monthly_basin.pkl",
    MANIFEST_PATH,
    OUTPUT_CONFIG_PATH,
]

if STATE_PATH.exists() and all(p.exists() for p in expected_outputs):
    try:
        old_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        old_state = None
    if old_state == fingerprint:
        print("=" * 72)
        print("01_PREPARAR_DATOS: SIN CAMBIOS")
        print("=" * 72)
        print("Las fuentes no cambiaron y los productos procesados ya existen.")
        print("No se reproceso nada.")
        print(PROCESSED_DIR)
        sys.exit(0)

outputs = {}
qc_rows = []

matrix = clean_model_matrix(safe_read_csv(SOURCES["matrix_autoregressive"]["path"]))
outputs["matrix_autoregressive"] = save_table(matrix, "matriz_modelo_autoregresiva")
qc_rows.append(dataframe_summary("matrix_autoregressive", matrix, "date_obs"))

features_raw = safe_read_csv(SOURCES["features_autoregressive"]["path"])
features_clean, feature_raw_list, duplicated_features, missing_feature_names = clean_features_table(features_raw, matrix.columns)
outputs["features_autoregressive_unique"] = save_table(features_clean, "features_autoregresivas_unicas")
qc_rows.append(dataframe_summary("features_autoregressive_unique", features_clean, None))

climate_features_raw = safe_read_csv(SOURCES["climate_daily_features"]["path"])
climate_features, climate_date_col = clean_generic_csv(climate_features_raw, "climate_daily_features")
outputs["climate_daily_features"] = save_table(climate_features, "clima_diario_features")
qc_rows.append(dataframe_summary("climate_daily_features", climate_features, climate_date_col))

if SOURCES["climate_daily_clean"]["path"].exists():
    d = safe_read_csv(SOURCES["climate_daily_clean"]["path"])
    d, dc = clean_generic_csv(d, "climate_daily_clean")
    outputs["climate_daily_clean"] = save_table(d, "clima_diario_limpio")
    qc_rows.append(dataframe_summary("climate_daily_clean", d, dc))

level_raw = safe_read_csv(SOURCES["level_canonical"]["path"])
level, level_date_col = clean_generic_csv(level_raw, "level_canonical")
if "nivel_final_m" in level.columns:
    level["nivel_final_m"] = numeric(level["nivel_final_m"])
outputs["level_canonical"] = save_table(level, "nivel_canonico")
qc_rows.append(dataframe_summary("level_canonical", level, level_date_col))

if SOURCES["level_qc"]["path"].exists():
    d = safe_read_csv(SOURCES["level_qc"]["path"])
    d, dc = clean_generic_csv(d, "level_qc")
    outputs["level_qc"] = save_table(d, "nivel_qc")
    qc_rows.append(dataframe_summary("level_qc", d, dc))

for source_key, output_stem in [
    ("era5_raw", "era5_diario_cuenca_real"),
    ("chirps_raw", "chirps_diario_cuenca_real"),
    ("nasa_level_raw", "nasa_nivel_final_original"),
]:
    if SOURCES[source_key]["path"].exists():
        d = safe_read_csv(SOURCES[source_key]["path"])
        d, dc = clean_generic_csv(d, source_key)
        outputs[source_key] = save_table(d, output_stem)
        qc_rows.append(dataframe_summary(source_key, d, dc))

rivers_raw, rivers_sheet, rivers_sheets = safe_read_excel(SOURCES["rivers"]["path"], "Consolidado")
rivers = clean_rivers(rivers_raw)
rivers_monthly = build_river_monthly(rivers)
outputs["rivers_clean"] = save_table(rivers, "rivers_clean")
outputs["rivers_monthly"] = save_table(rivers_monthly, "rivers_monthly")
qc_rows.append(dataframe_summary("rivers_clean", rivers, "date"))
qc_rows.append(dataframe_summary("rivers_monthly", rivers_monthly, "month_date"))

weather_raw, weather_sheet, weather_sheets = safe_read_excel(SOURCES["weather"]["path"], "Datos")
weather = clean_weather(weather_raw)
weather_monthly = build_weather_monthly(weather)
outputs["weather_clean"] = save_table(weather, "weather_clean")
outputs["weather_monthly_basin"] = save_table(weather_monthly, "weather_monthly_basin")
qc_rows.append(dataframe_summary("weather_clean", weather, "month_date"))
qc_rows.append(dataframe_summary("weather_monthly_basin", weather_monthly, "month_date"))

if SOURCES["lake_legacy"]["path"].exists():
    lake_raw, lake_sheet, lake_sheets = safe_read_excel(SOURCES["lake_legacy"]["path"], None)
    lake = clean_lake(lake_raw)
    outputs["lake_legacy"] = save_table(lake, "lake_legacy")
    qc_rows.append(dataframe_summary("lake_legacy", lake, "Fecha" if "Fecha" in lake.columns else None))
else:
    lake_sheet = None
    lake_sheets = []

if SOURCES["limnology"]["path"].exists():
    lim_raw, lim_sheet, lim_sheets = safe_read_excel(SOURCES["limnology"]["path"], "2014-2024")
    lim = clean_limnology(lim_raw)
    outputs["limnology"] = save_table(lim, "limnology")
    qc_rows.append(dataframe_summary("limnology", lim, "Fecha" if "Fecha" in lim.columns else None))
else:
    lim_sheet = None
    lim_sheets = []

qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(QC_PATH, index=False)

expected_rivers = {"Quiscab", "San_Francisco", "Tzununa", "La_Catarata", "San_Buenaventura"}
actual_rivers = set(rivers["river"].dropna().unique())
missing_rivers = sorted(expected_rivers - actual_rivers)
extra_rivers = sorted(actual_rivers - expected_rivers)
complete_months = int(rivers_monthly["complete_5rivers"].sum())

manifest = {
    "script_version": SCRIPT_VERSION,
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "project_root": str(ROOT),
    "source_catalog": catalog,
    "outputs": outputs,
    "quality_control_file": str(QC_PATH),
    "feature_list": {
        "raw_count": len(feature_raw_list),
        "unique_present_count": int(len(features_clean)),
        "duplicated_names": duplicated_features,
        "names_missing_from_matrix": missing_feature_names,
    },
    "rivers": {
        "excel_sheet_used": rivers_sheet,
        "excel_sheets": rivers_sheets,
        "expected_rivers": sorted(expected_rivers),
        "actual_rivers": sorted(actual_rivers),
        "missing_expected_rivers": missing_rivers,
        "extra_rivers": extra_rivers,
        "complete_5river_months": complete_months,
        "flow_primary_source": "Caudal (l/s) / 1000",
        "no_imputation": True,
    },
    "weather": {
        "excel_sheet_used": weather_sheet,
        "excel_sheets": weather_sheets,
        "station_count": int(weather["station"].nunique()),
        "monthly_spatial_summary": "mean across available stations; station/value counts retained",
        "no_imputation": True,
    },
    "lake_legacy": {
        "excel_sheet_used": lake_sheet,
        "excel_sheets": lake_sheets,
        "role": "secondary/archive; not primary model input",
    },
    "limnology": {
        "excel_sheet_used": lim_sheet,
        "excel_sheets": lim_sheets,
        "role": "secondary/future limnology analyses; not primary level model input",
    },
}

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

with open(OUTPUT_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=2, ensure_ascii=False)

STATE_PATH.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False), encoding="utf-8")

print("=" * 72)
print("01_PREPARAR_DATOS TERMINADO")
print("=" * 72)
print("Proyecto:", ROOT)
print("Productos:", PROCESSED_DIR)
print("Fuentes inventariadas:", len(catalog_df))
print("Matriz autoregresiva:", len(matrix), "filas x", len(matrix.columns), "columnas")
print("Features autoregresivas unicas presentes:", len(features_clean))
print("Nombres repetidos en lista original:", duplicated_features if duplicated_features else "ninguno")
print("Rios limpios:", len(rivers), "aforos")
print("Tributarios encontrados:", sorted(actual_rivers))
print("Meses completos con 5 tributarios:", complete_months)
print("Clima local:", len(weather), "registros;", weather["station"].nunique(), "estaciones")
print("Control de calidad:", QC_PATH)
print("Manifest:", MANIFEST_PATH)
print("Configuracion para siguientes scripts:", OUTPUT_CONFIG_PATH)
print("=" * 72)
