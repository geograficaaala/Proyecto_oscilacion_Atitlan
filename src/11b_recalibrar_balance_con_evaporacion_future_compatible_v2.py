                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
import re
import sys

import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

                                                                       
                
                                                                       
CLIMATE_CANDIDATES = [
    ROOT / "outputs" / "caudales" / "clima_diario_features_modelo_caudal.pkl",
    ROOT / "outputs" / "caudales" / "clima_diario_features_modelo_caudal.csv",
]

LONGWAVE_FILE = (
    ROOT / "data" / "evaporation" / "procesado"
    / "atitlan_era5_longwave_1981_2026.csv"
)

EVAP_PARAMS_FILE = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "11a_parametros_candidatos.json"
)

OUT_DIR = ROOT / "outputs" / "evaporacion_future_compatible"
OUT_DIR.mkdir(parents=True, exist_ok=True)

                                                                       
                                   
                                                                       
LAKE_AREA_KM2 = 125.77
LAKE_AREA_M2 = LAKE_AREA_KM2 * 1e6

CAL_END = pd.Timestamp("2024-01-10")
TEST_START = pd.Timestamp("2024-01-20")
TEST_END = pd.Timestamp("2026-08-29")

EXPECTED_HISTORICAL_PARAMS = {
    "precip_scale": 1.278816,
    "evap_scale": 2.273196,
    "qin_scale": 0.758222,
    "net_unresolved_m_day": -0.00026467,
}

EXPECTED_HISTORICAL_TEST = {
    "mae": 0.160801,
    "rmse": 0.201679,
    "bias": -0.088605,
    "r2_nse": 0.796255,
    "pearson_r": 0.916673,
}

MAX_BASELINE_MAE_DIFFERENCE_M = 0.05

                                                                       
                          
                                                                       
SIGMA_MJ = 4.903e-9


def norm(x):
    s = str(x).strip().lower()
    for a, b in [
        ("á", "a"), ("é", "e"), ("í", "i"),
        ("ó", "o"), ("ú", "u"), ("ñ", "n"),
    ]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path):
    if path.suffix.lower() in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"{path} no contiene un DataFrame.")
        return obj.copy()

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Formato no soportado: {path}")


def read_header(path):
    try:
        if path.suffix.lower() == ".csv":
            return list(pd.read_csv(path, nrows=2).columns)

        if path.suffix.lower() in {".pkl", ".pickle"}:
            obj = pd.read_pickle(path)
            if isinstance(obj, pd.DataFrame):
                return list(obj.columns)
    except Exception:
        return None

    return None


def detect_col(df_or_cols, exact=None, contains=None):
    exact = exact or []
    contains = contains or []

    cols = (
        list(df_or_cols.columns)
        if hasattr(df_or_cols, "columns")
        else list(df_or_cols)
    )

    nmap = {norm(c): c for c in cols}

    for c in exact:
        if norm(c) in nmap:
            return nmap[norm(c)]

    for c in cols:
        nc = norm(c)
        if any(norm(token) in nc for token in contains):
            return c

    return None


def load_first_existing(candidates, label):
    for p in candidates:
        if p.exists():
            return read_table(p), p

    raise FileNotFoundError(
        f"No encontré {label}. Probé:\n"
        + "\n".join(str(p) for p in candidates)
    )


                                                                       
                  
                                                                       
def sat_vapor_pressure_kpa(temp_c):
    t = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def slope_vapor_pressure_curve_kpa_c(temp_c):
    t = np.asarray(temp_c, dtype=float)
    es = sat_vapor_pressure_kpa(t)
    return 4098.0 * es / np.square(t + 237.3)


def prepare_climate():
    climate, climate_path = load_first_existing(
        CLIMATE_CANDIDATES,
        "tabla climática histórica",
    )

    date_col = detect_col(
        climate,
        exact=["date", "fecha"],
    )
    if date_col is None:
        raise KeyError("La tabla climática no contiene date/fecha.")

    climate["date"] = pd.to_datetime(
        climate[date_col],
        errors="raise",
    ).dt.normalize()

    if climate["date"].duplicated().any():
        raise ValueError("La tabla climática tiene fechas duplicadas.")

                             
    p_col = detect_col(
        climate,
        exact=[
            "precip_consensus_mm",
            "p_consensus_mm",
            "precip_consensus",
        ],
        contains=[
            "precip_consensus",
            "p_consensus",
        ],
    )

    era_col = detect_col(
        climate,
        exact=[
            "era_precip_mm",
            "era5_precip_mm",
            "precip_era_mm",
        ],
        contains=["era_precip"],
    )

    chirps_col = detect_col(
        climate,
        exact=[
            "chirps_precip_mm",
            "precip_chirps_mm",
        ],
        contains=["chirps_precip"],
    )

    if p_col is not None:
        climate["p_consensus_mm"] = pd.to_numeric(
            climate[p_col],
            errors="coerce",
        )
        p_source = p_col

    elif era_col is not None and chirps_col is not None:
        climate["p_consensus_mm"] = (
            pd.to_numeric(climate[era_col], errors="coerce")
            + pd.to_numeric(climate[chirps_col], errors="coerce")
        ) / 2.0
        p_source = f"mean({era_col}, {chirps_col})"

    else:
        raise KeyError(
            "No encontré precip_consensus ni ERA5+CHIRPS para reconstruirla."
        )

                                     
    evap_col = detect_col(
        climate,
        exact=["evap_mm", "evaporation_mm"],
        contains=["evap_mm"],
    )

    if evap_col is None:
        raise KeyError("No encontré evap_mm en la tabla climática.")

    climate["evap_original_mm"] = pd.to_numeric(
        climate[evap_col],
        errors="coerce",
    )

                                               
    tmean_col = detect_col(
        climate,
        exact=["tmean_c", "tas_c"],
        contains=["tmean"],
    )

    solar_col = detect_col(
        climate,
        exact=[
            "solar_mj_m2",
            "rsds_mj_m2_day",
        ],
        contains=["solar_mj", "rsds_mj"],
    )

    if tmean_col is None or solar_col is None:
        raise KeyError(
            f"Faltan tmean/solar. tmean={tmean_col}, solar={solar_col}"
        )

    climate["tmean_for_evap_c"] = pd.to_numeric(
        climate[tmean_col],
        errors="coerce",
    )

    climate["solar_for_evap_mj_m2"] = pd.to_numeric(
        climate[solar_col],
        errors="coerce",
    )

                                              
    if not LONGWAVE_FILE.exists():
        raise FileNotFoundError(
            "Falta el archivo integrado del PASO 11B1:\n"
            f"{LONGWAVE_FILE}"
        )

    lw = pd.read_csv(LONGWAVE_FILE)

    if "date" not in lw.columns or "longwave_mj_m2" not in lw.columns:
        raise KeyError(
            "El archivo 11B1 debe tener date y longwave_mj_m2."
        )

    lw["date"] = pd.to_datetime(
        lw["date"],
        errors="raise",
    ).dt.normalize()

    lw["longwave_mj_m2"] = pd.to_numeric(
        lw["longwave_mj_m2"],
        errors="coerce",
    )

    if lw["date"].duplicated().any():
        raise ValueError("Longwave 11B1 tiene fechas duplicadas.")

    climate = climate.merge(
        lw[["date", "longwave_mj_m2"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

                                                                  
                                                                    
                                                                        
                                                                         
                                                           
    extra_after_test = climate.loc[
        climate["date"] > TEST_END,
        ["date", "longwave_mj_m2"],
    ].copy()

    if len(extra_after_test):
        print(
            f"Nota: se excluyen {len(extra_after_test)} días climáticos "
            f"posteriores al fin del test ({TEST_END.date()})."
        )
        print(
            "Fechas excluidas: "
            + ", ".join(
                d.strftime("%Y-%m-%d")
                for d in extra_after_test["date"]
            )
        )

    climate = climate.loc[
        climate["date"] <= TEST_END
    ].copy()

    needed = [
        "p_consensus_mm",
        "evap_original_mm",
        "tmean_for_evap_c",
        "solar_for_evap_mj_m2",
        "longwave_mj_m2",
    ]

    bad = climate[needed].isna().sum()
    bad = bad[bad > 0]

    if len(bad):
        missing_lw_dates = climate.loc[
            climate["longwave_mj_m2"].isna(),
            "date",
        ].dt.strftime("%Y-%m-%d").tolist()

        raise ValueError(
            "Hay faltantes dentro del periodo realmente usado por el modelo:\n"
            + bad.to_string()
            + (
                "\nFechas sin longwave: "
                + ", ".join(missing_lw_dates[:20])
                if missing_lw_dates
                else ""
            )
        )

    return (
        climate.sort_values("date").reset_index(drop=True),
        climate_path,
        p_source,
        evap_col,
        tmean_col,
        solar_col,
    )


def build_evap_candidate(climate):
    params11 = load_json(EVAP_PARAMS_FILE)

    pressure = float(params11["pressure_constant_kpa"])
    physical = params11.get("physical_constants", {})

    water_albedo = float(
        physical.get("water_albedo", 0.08)
    )
    water_emissivity = float(
        physical.get("water_emissivity", 0.97)
    )
    sigma = float(
        physical.get("sigma_mj_m2_day_k4", SIGMA_MJ)
    )

    eq = (
        params11["candidate_parameters"]
        ["EQUILIBRIUM_OPEN_WATER_PROXY"]
        ["AFFINE"]
    )

    if eq.get("mapping") != "AFFINE_POSITIVE":
        raise ValueError(
            "11A no contiene el mapping AFFINE_POSITIVE esperado."
        )

    intercept = float(eq["intercept"])
    slope = float(eq["slope"])

    tmean = climate["tmean_for_evap_c"].to_numpy(float)
    solar = np.maximum(
        climate["solar_for_evap_mj_m2"].to_numpy(float),
        0.0,
    )
    longwave = np.maximum(
        climate["longwave_mj_m2"].to_numpy(float),
        0.0,
    )

    delta = slope_vapor_pressure_curve_kpa_c(tmean)
    gamma = 0.000665 * pressure

    lw_up = (
        water_emissivity
        * sigma
        * np.power(tmean + 273.15, 4)
    )

    rn_water = (
        (1.0 - water_albedo) * solar
        + longwave
        - lw_up
    )

    equilibrium = (
        0.408
        * (
            delta
            / np.maximum(delta + gamma, 1e-12)
        )
        * np.maximum(rn_water, 0.0)
    )

    candidate = np.maximum(
        intercept + slope * equilibrium,
        0.0,
    )

    climate = climate.copy()
    climate["evap_candidate_mm"] = candidate

    meta = {
        "pressure_constant_kpa": pressure,
        "water_albedo": water_albedo,
        "water_emissivity": water_emissivity,
        "sigma_mj_m2_day_k4": sigma,
        "affine_intercept": intercept,
        "affine_slope": slope,
    }

    return climate, meta


                                                                       
                                               
                                                                       
RIVER_NAMES = [
    "Quiscab",
    "San_Francisco",
    "Tzununa",
    "La_Catarata",
    "San_Buenaventura",
]


def extract_flow_candidate(path):
    cols = read_header(path)
    if not cols:
        return None

    date_col = detect_col(
        cols,
        exact=["date", "fecha"],
    )
    if date_col is None:
        return None

    q_col = detect_col(
        cols,
        exact=[
            "qin_total_m3s",
            "q_total_m3s",
            "q_total_5rivers_m3s",
            "caudal_total_m3s",
            "pred_total_m3s",
            "q_pred_total_m3s",
        ],
        contains=[
            "qin_total_m3s",
            "q_total_m3s",
            "caudal_total_m3s",
        ],
    )

    nmap = {norm(c): c for c in cols}
    river_cols = {}

    for river in RIVER_NAMES:
        nr = norm(river)

        if nr in nmap:
            river_cols[river] = nmap[nr]
            continue

        matches = [
            c for c in cols
            if nr in norm(c)
            and any(
                token in norm(c)
                for token in [
                    "pred",
                    "flow",
                    "caudal",
                    "q_",
                    "m3s",
                ]
            )
        ]

        if matches:
            river_cols[river] = matches[0]

    if q_col is None and len(river_cols) != 5:
        return None

    df = read_table(path)

    df["__date"] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    ).dt.normalize()

    if q_col is not None:
        q = pd.to_numeric(
            df[q_col],
            errors="coerce",
        )
        source = q_col

    else:
        tmp = pd.DataFrame(index=df.index)

        for river, col in river_cols.items():
            tmp[river] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        q = tmp.sum(axis=1, min_count=5)
        source = river_cols

    x = pd.DataFrame({
        "date": df["__date"],
        "qin_total_m3s": q,
    }).dropna()

    if x.empty:
        return None

    x = (
        x.groupby("date", as_index=False)
        ["qin_total_m3s"]
        .mean()
        .sort_values("date")
    )

    start = x["date"].min()
    end = x["date"].max()
    n = len(x)

                                                         
                                   
    if start > pd.Timestamp("2010-01-01"):
        return None

    if end < TEST_END:
        return None

    if n < 5000:
        return None

    return {
        "path": path,
        "data": x,
        "source": source,
        "start": start,
        "end": end,
        "n": n,
    }


def find_flow():
    explicit = [
        ROOT / "outputs" / "caudales" / "caudal_diario_estimado_final.pkl",
        ROOT / "outputs" / "caudales" / "caudal_diario_estimado_final.csv",
        ROOT / "outputs" / "caudales" / "predicciones_caudal_diario_final.csv",
        ROOT / "outputs" / "nivel" / "forzantes_balance_diario.csv",
    ]

    candidates = []

    for p in explicit:
        if p.exists():
            candidates.append(p)

    for base in [
        ROOT / "outputs" / "caudales",
        ROOT / "outputs" / "nivel",
        ROOT / "data" / "processed",
    ]:
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in {".csv", ".pkl", ".pickle"}
                and p not in candidates
            ):
                candidates.append(p)

    valid = []

    for p in candidates:
        try:
            info = extract_flow_candidate(p)
            if info is not None:
                valid.append(info)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "No encontré automáticamente una serie DIARIA de caudal total "
            "que cubra aproximadamente 2009-2026.\n"
            "Necesitamos la serie histórica de caudales utilizada por PASO 03."
        )

                  
                                     
                 
                               
    def score(info):
        p = str(info["path"]).lower()
        preferred = 1 if "resultados\\caudales" in p else 0
        return (
            preferred,
            info["n"],
            -info["start"].toordinal(),
        )

    valid.sort(
        key=score,
        reverse=True,
    )

    chosen = valid[0]

    return (
        chosen["data"],
        chosen["path"],
        chosen["source"],
        valid,
    )


                                                                       
                                        
                                                                       
def extract_level_candidate(path):
    cols = read_header(path)
    if not cols:
        return None

    date_col = detect_col(
        cols,
        exact=["date", "fecha"],
    )

    level_col = detect_col(
        cols,
        exact=[
            "nivel_final_m",
            "level_m",
            "nivel_m",
            "water_level_m",
        ],
        contains=[
            "nivel_final",
            "water_level",
        ],
    )

    if date_col is None or level_col is None:
        return None

    df = read_table(path)

    x = pd.DataFrame({
        "date": pd.to_datetime(
            df[date_col],
            errors="coerce",
        ).dt.normalize(),
        "level_m": pd.to_numeric(
            df[level_col],
            errors="coerce",
        ),
    }).dropna()

    if x.empty:
        return None

    x = (
        x.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    start = x["date"].min()
    end = x["date"].max()
    n = len(x)

    if start > pd.Timestamp("2010-01-01"):
        return None

    if end < TEST_END:
        return None

    if n < 400:
        return None

    return {
        "path": path,
        "data": x,
        "date_col": date_col,
        "level_col": level_col,
        "start": start,
        "end": end,
        "n": n,
    }


def find_levels():
    explicit = [
        ROOT / "data" / "processed" / "nivel_canonico.csv",
        ROOT / "data" / "processed" / "atitlan_nivel_final.csv",
        Path.home() / "Downloads" / "atitlan_nivel_final.csv",
    ]

    candidates = []

    for p in explicit:
        if p.exists():
            candidates.append(p)

    for base in [
        ROOT / "data" / "processed",
        ROOT / "outputs" / "nivel",
    ]:
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in {".csv", ".pkl", ".pickle"}
                and p not in candidates
            ):
                candidates.append(p)

    valid = []

    for p in candidates:
        try:
            info = extract_level_candidate(p)
            if info is not None:
                valid.append(info)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "No encontré automáticamente la serie canónica de nivel "
            "con cobertura 2008/2009-2026."
        )

                                                                             
    def score(info):
        name = info["path"].name.lower()
        canonical = (
            2 if "canon" in name
            else 1 if "final" in name
            else 0
        )
        n_closeness = -abs(info["n"] - 554)

        return (
            canonical,
            n_closeness,
            info["end"].toordinal(),
        )

    valid.sort(
        key=score,
        reverse=True,
    )

    chosen = valid[0]

    return (
        chosen["data"],
        chosen["path"],
        chosen["level_col"],
        valid,
    )


                                                                       
         
                                                                       
def build_daily_forcing(climate, flow):
    daily = climate[
        [
            "date",
            "p_consensus_mm",
            "evap_original_mm",
            "evap_candidate_mm",
        ]
    ].merge(
        flow,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    daily = daily.sort_values("date").reset_index(drop=True)

    daily["p_m_day"] = (
        daily["p_consensus_mm"] / 1000.0
    )

    daily["evap_original_m_day"] = (
        daily["evap_original_mm"] / 1000.0
    )

    daily["evap_candidate_m_day"] = (
        daily["evap_candidate_mm"] / 1000.0
    )

    daily["qin_depth_m_day"] = (
        daily["qin_total_m3s"]
        * 86400.0
        / LAKE_AREA_M2
    )

    return daily


def build_intervals(levels, daily, evap_col):
    rows = []

    levels = levels.sort_values("date").reset_index(drop=True)

    for i in range(1, len(levels)):
        d0 = levels.loc[i - 1, "date"]
        d1 = levels.loc[i, "date"]

        if d1 <= d0:
            continue

        n_expected = int((d1 - d0).days)

        block = daily.loc[
            (daily["date"] > d0)
            & (daily["date"] <= d1)
        ]

        if len(block) != n_expected:
            continue

        h0 = float(levels.loc[i - 1, "level_m"])
        h1 = float(levels.loc[i, "level_m"])

        rows.append({
            "date_prev": d0,
            "date_obs": d1,
            "level_prev_m": h0,
            "level_obs_m": h1,
            "delta_obs_m": h1 - h0,
            "days": n_expected,
            "p_sum_m": float(block["p_m_day"].sum()),
            "e_sum_m": float(block[evap_col].sum()),
            "qin_sum_m": float(block["qin_depth_m_day"].sum()),
        })

    return pd.DataFrame(rows)


def fit_balance(intervals):
    fit = intervals.loc[
        intervals["date_obs"] <= CAL_END
    ].copy()

    if len(fit) < 50:
        raise ValueError(
            f"Demasiados pocos intervalos de calibración: {len(fit)}"
        )

    X = np.column_stack([
        fit["p_sum_m"].to_numpy(float),
        -fit["e_sum_m"].to_numpy(float),
        fit["qin_sum_m"].to_numpy(float),
        fit["days"].to_numpy(float),
    ])

    y = fit["delta_obs_m"].to_numpy(float)

    result = lsq_linear(
        X,
        y,
        bounds=(
            [0.0, 0.0, 0.0, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        ),
        method="trf",
        lsmr_tol="auto",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return {
        "precip_scale": float(result.x[0]),
        "evap_scale": float(result.x[1]),
        "qin_scale": float(result.x[2]),
        "net_unresolved_m_day": float(result.x[3]),
        "n_fit_intervals": int(len(fit)),
        "cost": float(result.cost),
    }


def delta_prediction(row, p):
    return (
        p["precip_scale"] * float(row["p_sum_m"])
        - p["evap_scale"] * float(row["e_sum_m"])
        + p["qin_scale"] * float(row["qin_sum_m"])
        + p["net_unresolved_m_day"] * float(row["days"])
    )


def recursive_test(intervals, params):
    test = intervals.loc[
        intervals["date_obs"].between(
            TEST_START,
            TEST_END,
        )
    ].copy()

    if test.empty:
        raise ValueError(
            "No hay intervalos en test 2024-01-20 a 2026-08-29."
        )

    first_idx = test.index[0]
    pred_level = float(
        intervals.loc[first_idx, "level_prev_m"]
    )

    rows = []

    for _, row in test.iterrows():
        d_pred = delta_prediction(row, params)
        pred_level += d_pred

        rows.append({
            "date": row["date_obs"],
            "observed_level_m": float(row["level_obs_m"]),
            "recursive_pred_level_m": float(pred_level),
            "one_step_pred_level_m": (
                float(row["level_prev_m"]) + d_pred
            ),
            "observed_delta_m": float(row["delta_obs_m"]),
            "predicted_delta_m": float(d_pred),
            "days": int(row["days"]),
        })

    return pd.DataFrame(rows)


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]

    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
        "r2_nse": float(r2_score(y, p)),
        "pearson_r": np.nan,
    }

    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson_r"] = float(
            np.corrcoef(y, p)[0, 1]
        )

    return out


def evaluate(intervals, params):
    pred = recursive_test(intervals, params)

    rec = metrics(
        pred["observed_level_m"],
        pred["recursive_pred_level_m"],
    )

    one = metrics(
        pred["observed_level_m"],
        pred["one_step_pred_level_m"],
    )

    delt = metrics(
        pred["observed_delta_m"],
        pred["predicted_delta_m"],
    )

    return pred, rec, one, delt


def main():
    print("=" * 104)
    print("PASO 11B v3 - RECALIBRAR BALANCE CON EVAPORACION FUTURE-COMPATIBLE")
    print("=" * 104)
    print()
    print("Corrección principal respecto al 11B anterior:")
    print(" - longwave se lee del archivo integrado del PASO 11B1")
    print(" - NO se espera longwave dentro de la tabla de features de caudales")
    print()
    print("Primero reconstruimos el balance con evap_mm original.")
    print("Después cambiamos SOLO evaporación y recalibramos aP, aE, aQ y c_day.")
    print()

    if not EVAP_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Falta PASO 11A:\n{EVAP_PARAMS_FILE}"
        )

    climate, climate_path, p_source, evap_source, tmean_source, solar_source = (
        prepare_climate()
    )

    climate, evap_meta = build_evap_candidate(climate)

    flow, flow_path, flow_source, flow_candidates = find_flow()
    levels, level_path, level_source, level_candidates = find_levels()

    daily = build_daily_forcing(
        climate,
        flow,
    )

    print("FUENTES SELECCIONADAS")
    print("-" * 104)
    print(f"Clima:     {climate_path}")
    print(f"Longwave:  {LONGWAVE_FILE}")
    print(f"Caudal:    {flow_path}")
    print(f"Nivel:     {level_path}")
    print()
    print(f"P:         {p_source}")
    print(f"Evap orig: {evap_source}")
    print(f"Tmean:     {tmean_source}")
    print(f"Solar:     {solar_source}")
    print(f"Q source:  {flow_source}")
    print(f"Nivel col: {level_source}")
    print()

    print(
        f"Clima:  {climate['date'].min().date()} -> "
        f"{climate['date'].max().date()} | n={len(climate):,}"
    )
    print(
        f"Caudal: {flow['date'].min().date()} -> "
        f"{flow['date'].max().date()} | n={len(flow):,}"
    )
    print(
        f"Nivel:  {levels['date'].min().date()} -> "
        f"{levels['date'].max().date()} | n={len(levels):,}"
    )
    print(
        f"Común:  {daily['date'].min().date()} -> "
        f"{daily['date'].max().date()} | n={len(daily):,}"
    )
    print()

    original_intervals = build_intervals(
        levels,
        daily,
        "evap_original_m_day",
    )

    candidate_intervals = build_intervals(
        levels,
        daily,
        "evap_candidate_m_day",
    )

    if len(original_intervals) != len(candidate_intervals):
        raise RuntimeError(
            "Original y candidato generaron diferente número de intervalos."
        )

    print(
        f"Intervalos de nivel con forzamiento diario completo: "
        f"{len(original_intervals)}"
    )
    print()

    original_params = fit_balance(
        original_intervals
    )
    candidate_params = fit_balance(
        candidate_intervals
    )

    orig_pred, orig_rec, orig_one, orig_delta = evaluate(
        original_intervals,
        original_params,
    )

    cand_pred, cand_rec, cand_one, cand_delta = evaluate(
        candidate_intervals,
        candidate_params,
    )

    baseline_mae_diff = (
        orig_rec["mae"]
        - EXPECTED_HISTORICAL_TEST["mae"]
    )

    reconstruction_ok = (
        abs(baseline_mae_diff)
        <= MAX_BASELINE_MAE_DIFFERENCE_M
    )

    change_vs_reconstructed_pct = (
        100.0
        * (cand_rec["mae"] - orig_rec["mae"])
        / orig_rec["mae"]
    )

    skill_vs_reconstructed = (
        1.0
        - cand_rec["mae"] / orig_rec["mae"]
    )

    change_vs_frozen_pct = (
        100.0
        * (
            cand_rec["mae"]
            - EXPECTED_HISTORICAL_TEST["mae"]
        )
        / EXPECTED_HISTORICAL_TEST["mae"]
    )

    accepted = (
        reconstruction_ok
        and cand_rec["mae"] <= 1.10 * orig_rec["mae"]
        and cand_rec["r2_nse"] >= 0.70
        and cand_rec["pearson_r"] >= 0.85
    )

                                                                       
             
                                                                       
    orig_pred_path = (
        OUT_DIR / "11b_v3_predicciones_test_evap_original.csv"
    )
    cand_pred_path = (
        OUT_DIR / "11b_v3_predicciones_test_evap_future_compatible.csv"
    )
    metrics_path = (
        OUT_DIR / "11b_v3_metricas_balance_evaporacion.csv"
    )
    params_path = (
        OUT_DIR / "11b_v3_parametros_balance_recalibrado.csv"
    )
    forcing_path = (
        OUT_DIR / "11b_v3_forzantes_diarios.csv"
    )
    manifest_path = (
        OUT_DIR / "manifest_paso11b_v3.json"
    )

    orig_pred.to_csv(
        orig_pred_path,
        index=False,
    )
    cand_pred.to_csv(
        cand_pred_path,
        index=False,
    )

    metric_rows = []

    for version, rec, one, delt in [
        (
            "RECONSTRUCTED_ORIGINAL_EVAP",
            orig_rec,
            orig_one,
            orig_delta,
        ),
        (
            "FUTURE_COMPATIBLE_EVAP",
            cand_rec,
            cand_one,
            cand_delta,
        ),
    ]:
        metric_rows.extend([
            {
                "version": version,
                "scope": "recursive_level",
                **rec,
            },
            {
                "version": version,
                "scope": "one_step_level",
                **one,
            },
            {
                "version": version,
                "scope": "delta",
                **delt,
            },
        ])

    pd.DataFrame(metric_rows).to_csv(
        metrics_path,
        index=False,
    )

    pd.DataFrame([
        {
            "version": "RECONSTRUCTED_ORIGINAL_EVAP",
            **original_params,
        },
        {
            "version": "FUTURE_COMPATIBLE_EVAP",
            **candidate_params,
        },
        {
            "version": "FROZEN_EXPECTED_BASELINE",
            **EXPECTED_HISTORICAL_PARAMS,
        },
    ]).to_csv(
        params_path,
        index=False,
    )

    daily[
        [
            "date",
            "p_consensus_mm",
            "evap_original_mm",
            "evap_candidate_mm",
            "qin_total_m3s",
            "qin_depth_m_day",
        ]
    ].to_csv(
        forcing_path,
        index=False,
    )

    manifest = {
        "paso": "11B_v3",
        "python_version": sys.version,
        "lake_area_km2": LAKE_AREA_KM2,
        "periods": {
            "calibration_end": str(CAL_END.date()),
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
        },
        "inputs": {
            "climate": str(climate_path),
            "climate_sha256": sha256_file(climate_path),
            "longwave": str(LONGWAVE_FILE),
            "longwave_sha256": sha256_file(LONGWAVE_FILE),
            "flow": str(flow_path),
            "flow_sha256": sha256_file(flow_path),
            "level": str(level_path),
            "level_sha256": sha256_file(level_path),
            "evap_11a_params": str(EVAP_PARAMS_FILE),
            "evap_11a_sha256": sha256_file(EVAP_PARAMS_FILE),
        },
        "selected_columns": {
            "precipitation": p_source,
            "historical_evap": evap_source,
            "tmean": tmean_source,
            "solar": solar_source,
            "flow": str(flow_source),
            "level": level_source,
        },
        "evaporation_candidate_metadata": evap_meta,
        "reconstructed_original_params": original_params,
        "future_compatible_params": candidate_params,
        "expected_frozen_params": EXPECTED_HISTORICAL_PARAMS,
        "metrics": {
            "reconstructed_original_recursive": orig_rec,
            "future_compatible_recursive": cand_rec,
            "expected_frozen_recursive": EXPECTED_HISTORICAL_TEST,
        },
        "decision": {
            "baseline_reconstruction_mae_difference_m": baseline_mae_diff,
            "baseline_reconstruction_ok": reconstruction_ok,
            "candidate_mae_change_vs_reconstructed_pct": (
                change_vs_reconstructed_pct
            ),
            "candidate_skill_vs_reconstructed": skill_vs_reconstructed,
            "candidate_mae_change_vs_frozen_pct": change_vs_frozen_pct,
            "rule": (
                "Reconstrucción baseline dentro de 0.05 m MAE del frozen; "
                "candidato <=10% peor que reconstrucción; "
                "NSE>=0.70; Pearson>=0.85."
            ),
            "accepted": accepted,
        },
        "notes": [
            "PASO 11B v3 incorpora longwave desde PASO 11B1 y recorta cualquier día climático posterior al fin real del test.",
            "Sólo cambia la representación de evaporación.",
            "Precipitación y caudal permanecen históricos para aislar evaporación.",
            "Se recalibran todos los coeficientes del balance.",
            "c_day sigue siendo término neto no resuelto; no es groundwater identificado.",
            "No autoriza aún proyección CMIP6 completa."
        ],
        "outputs": {
            "original_predictions": str(orig_pred_path),
            "candidate_predictions": str(cand_pred_path),
            "metrics": str(metrics_path),
            "parameters": str(params_path),
            "forcing": str(forcing_path),
        },
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

                                                                       
             
                                                                       
    print("=" * 104)
    print("PARAMETROS")
    print("=" * 104)

    print(
        "Original reconstruido: "
        f"aP={original_params['precip_scale']:.6f} | "
        f"aE={original_params['evap_scale']:.6f} | "
        f"aQ={original_params['qin_scale']:.6f} | "
        f"c_day={original_params['net_unresolved_m_day']:+.8f}"
    )

    print(
        "Evap future-compatible: "
        f"aP={candidate_params['precip_scale']:.6f} | "
        f"aE={candidate_params['evap_scale']:.6f} | "
        f"aQ={candidate_params['qin_scale']:.6f} | "
        f"c_day={candidate_params['net_unresolved_m_day']:+.8f}"
    )

    print(
        "Frozen esperado: "
        f"aP={EXPECTED_HISTORICAL_PARAMS['precip_scale']:.6f} | "
        f"aE={EXPECTED_HISTORICAL_PARAMS['evap_scale']:.6f} | "
        f"aQ={EXPECTED_HISTORICAL_PARAMS['qin_scale']:.6f} | "
        f"c_day={EXPECTED_HISTORICAL_PARAMS['net_unresolved_m_day']:+.8f}"
    )

    print()
    print("=" * 104)
    print("TEST RETROSPECTIVO 2024-2026")
    print("=" * 104)

    print(
        "Original reconstruido: "
        f"MAE={orig_rec['mae']:.6f} | "
        f"RMSE={orig_rec['rmse']:.6f} | "
        f"bias={orig_rec['bias']:.6f} | "
        f"NSE={orig_rec['r2_nse']:.6f} | "
        f"r={orig_rec['pearson_r']:.6f}"
    )

    print(
        "Evap future-compatible: "
        f"MAE={cand_rec['mae']:.6f} | "
        f"RMSE={cand_rec['rmse']:.6f} | "
        f"bias={cand_rec['bias']:.6f} | "
        f"NSE={cand_rec['r2_nse']:.6f} | "
        f"r={cand_rec['pearson_r']:.6f}"
    )

    print(
        "Frozen histórico esperado: "
        f"MAE={EXPECTED_HISTORICAL_TEST['mae']:.6f} | "
        f"RMSE={EXPECTED_HISTORICAL_TEST['rmse']:.6f} | "
        f"bias={EXPECTED_HISTORICAL_TEST['bias']:.6f} | "
        f"NSE={EXPECTED_HISTORICAL_TEST['r2_nse']:.6f} | "
        f"r={EXPECTED_HISTORICAL_TEST['pearson_r']:.6f}"
    )

    print()
    print(
        "Diferencia reconstrucción vs frozen (MAE): "
        f"{baseline_mae_diff:+.6f} m"
    )
    print(
        "Cambio candidato vs reconstrucción (MAE): "
        f"{change_vs_reconstructed_pct:+.2f}%"
    )
    print(
        "Skill candidato vs reconstrucción: "
        f"{skill_vs_reconstructed:+.4f}"
    )
    print(
        "Cambio candidato vs frozen histórico (MAE): "
        f"{change_vs_frozen_pct:+.2f}%"
    )

    print()
    print(
        "DECISIÓN: "
        + (
            "APROBADO"
            if accepted
            else "NO APROBADO"
        )
        + " bajo la regla conservadora de 11B."
    )

    if not reconstruction_ok:
        print()
        print(
            "ADVERTENCIA: la reconstrucción del baseline quedó demasiado "
            "lejos del PHYSICAL_ONLY congelado. Si ocurre esto, NO debemos "
            "juzgar todavía la nueva evaporación; primero revisaremos qué "
            "serie de caudal/forzantes exacta usó PASO 03."
        )

    print()
    print("Archivos creados:")
    print(orig_pred_path)
    print(cand_pred_path)
    print(metrics_path)
    print(params_path)
    print(forcing_path)
    print(manifest_path)
    print()
    print("PASO 11B v3 COMPLETADO.")


if __name__ == "__main__":
    main()
