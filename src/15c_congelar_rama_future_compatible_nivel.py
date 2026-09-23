                       






from __future__ import annotations
__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
from datetime import datetime, timezone
import calendar
import gc
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd


                                                                               
       
                                                                               

ROOT = Path(__file__).resolve().parents[1]

SCRIPT_13K = ROOT / "src" / "13k_congelar_rama_future_compatible_caudales.py"

Q_FILE = (
    ROOT / "outputs" / "caudales_future_compatible"
    / "caudal_diario_cmip6_future_compatible.pkl"
)
Q_META = (
    ROOT / "outputs" / "caudales_future_compatible"
    / "parametros_armonizacion_cmip6_future_compatible.json"
)

E_FILE = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "evaporacion_diaria_cmip6_future_compatible.pkl"
)
E_META = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "parametros_evaporacion_cmip6_future_compatible.json"
)

REFERENCE_FILE = (
    ROOT / "data" / "reference_1981_2014" / "procesado"
    / "atitlan_referencia_1981_2014.csv"
)

LEVEL_FILE = ROOT / "data" / "processed" / "nivel_canonico.csv"

OUT_DIR = ROOT / "outputs" / "nivel_future_compatible"
OUT_DAILY = OUT_DIR / "nivel_diario_cmip6_future_compatible.pkl"
OUT_HORIZONS = OUT_DIR / "resumen_horizontes_nivel_cmip6_future_compatible.csv"
OUT_ENSEMBLE = OUT_DIR / "resumen_ensemble_nivel_cmip6_future_compatible.csv"
OUT_META = OUT_DIR / "parametros_nivel_cmip6_future_compatible.json"


                                                                               
                                   
                                                                               

AP = 1.278816
AE = 2.273196
AQ = 0.758222
C_DAY = 0.0

LAKE_AREA_M2 = 125.77e6

REFERENCE_START = pd.Timestamp("1981-01-01")
REFERENCE_END = pd.Timestamp("2014-12-31")

S_LEVEL_START = pd.Timestamp("2009-02-24")
S_LEVEL_END = pd.Timestamp("2024-01-10")

SCENARIOS = ("ssp245", "ssp585")
HORIZONS = (2030, 2050, 2090)

EXPECTED_N_GCM = 15
EXPECTED_N_TRAJ = 30
EXPECTED_FIRST_DATE = pd.Timestamp("2026-08-30")
EXPECTED_LAST_DATE = pd.Timestamp("2100-12-31")
EXPECTED_ROWS_PER_TRAJ = len(
    pd.date_range(EXPECTED_FIRST_DATE, EXPECTED_LAST_DATE, freq="D")
)
EXPECTED_DAILY_ROWS = EXPECTED_N_TRAJ * EXPECTED_ROWS_PER_TRAJ

                                                        
                                                                            
EXPECTED_LEVEL_MEAN_MEDIAN = {
    ("ssp245", 2030): 1552.348610,
    ("ssp245", 2050): 1550.336502,
    ("ssp245", 2090): 1542.639156,
    ("ssp585", 2030): 1552.293235,
    ("ssp585", 2050): 1548.144740,
    ("ssp585", 2090): 1530.124355,
}
CHECKPOINT_TOL_M = 5e-5

Q_RECON_MAX_ABS_TOL = 1e-6
Q_RECON_MAE_TOL = 1e-8


                                                                               
            
                                                                               

def tag(event: str, **kwargs) -> None:
    txt = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{event}] {txt}" if txt else f"[{event}]")


def require(path: Path, file_label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{file_label}: no existe {path}")
    tag(
        "FILE",
        file_label=file_label,
        path=path,
        size_mb=f"{path.stat().st_size / 1024**2:.2f}",
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_col(df: pd.DataFrame, exact: list[str], contains: list[str]) -> str:
    low = {str(c).lower(): c for c in df.columns}
    for x in exact:
        if x.lower() in low:
            return low[x.lower()]

    hits = []
    for c in df.columns:
        lc = str(c).lower()
        score = sum(tok.lower() in lc for tok in contains)
        if score:
            hits.append((score, c))

    if not hits:
        raise KeyError(
            f"No se encontro columna: exact={exact}; contains={contains}; "
            f"columns={list(df.columns)}"
        )

    hits.sort(key=lambda z: (-z[0], str(z[1])))
    top = hits[0][0]
    ties = [c for s, c in hits if s == top]
    if len(ties) != 1:
        raise RuntimeError(f"Columna ambigua: {ties}")

    return ties[0]


def json_float_dict(d: dict[int, float]) -> dict[str, float]:
    return {str(int(k)): float(v) for k, v in d.items()}


                                                                               
                                         
                                                                               

def load_13k():
    require(SCRIPT_13K, "script_13K")

    source = SCRIPT_13K.read_text(encoding="utf-8", errors="replace")
    if "if __name__" not in source or "__main__" not in source:
        raise RuntimeError(
            "13K no tiene guard __main__; se aborta para evitar escrituras "
            "accidentales al importarlo."
        )

    spec = importlib.util.spec_from_file_location(
        "atitlan_13k_for_15c", SCRIPT_13K
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo crear spec para 13K.")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = [
        "load_reference_base",
        "load_cmip_file",
        "load_models",
        "fit_harmonization_parameters",
        "apply_harmonization",
        "build_72_features",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"13K no expone {name}")

    tag("MODULE_13K", imported=True)
    return mod


def cmip_paths(qmeta: dict) -> dict[tuple[str, str], Path]:
    out = {}
    for key, info in qmeta["source_hashes"]["cmip6"].items():
        model, scenario = key.rsplit("__", 1)
        out[(model, scenario.lower())] = ROOT / info["path"]
    return out


                                                                               
                               
                                                                               

def unique_model_features(models: dict) -> list[str]:
    return sorted(
        set(
            feature
            for info in models.values()
            for feature in info["features"]
        )
    )


def ensure_feature_date(
    feat: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    out = feat.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(
            out["date"], errors="raise"
        ).dt.normalize()
        return out

    if len(out) != len(base):
        raise RuntimeError(
            "build_72_features no devolvio date y el largo no coincide."
        )

    out.insert(
        0,
        "date",
        pd.to_datetime(
            base["date"], errors="raise"
        ).dt.normalize().to_numpy(),
    )
    return out


def predict_q_from_features(
    feat: pd.DataFrame,
    models: dict,
    intercept: float,
    slope: float,
) -> pd.DataFrame:
    required = unique_model_features(models)
    missing = [c for c in required if c not in feat.columns]
    if missing:
        raise KeyError(f"Faltan features Q: {missing}")

    numeric = feat[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    idx = np.flatnonzero(valid)

    if len(idx) == 0:
        raise RuntimeError("No hay filas validas para predecir Q.")

    qsum = np.zeros(len(idx), dtype=float)

    for river, info in models.items():
        X = feat.iloc[idx][list(info["features"])]
        pred = np.asarray(info["estimator"].predict(X), dtype=float)

        family = str(info.get("family", "")).upper()
        if family == "RF_LOG":
            pred = np.expm1(pred)
        elif family == "XGB_TWEEDIE":
            pass
        else:
            raise RuntimeError(
                f"Familia Q no reconocida: {river} -> {family}"
            )

        qsum += np.maximum(pred, 0.0)

    out = pd.DataFrame(
        {"date": pd.to_datetime(feat["date"]).dt.normalize()}
    )
    out["Q_m3s"] = np.nan
    out.loc[idx, "Q_m3s"] = np.maximum(
        intercept + slope * qsum,
        0.0,
    )
    return out


                                                                               
         
                                                                               

def load_level_and_build_s():
    require(LEVEL_FILE, "nivel_canonico")

    level = pd.read_csv(LEVEL_FILE, low_memory=False)

    dcol = find_col(
        level,
        ["date", "fecha"],
        ["date", "fecha"],
    )
    hcol = find_col(
        level,
        ["nivel_final_m"],
        ["nivel_final", "nivel", "level", "height"],
    )

    level = level[[dcol, hcol]].copy()
    level.columns = ["date", "level_m"]
    level["date"] = pd.to_datetime(
        level["date"], errors="coerce"
    ).dt.normalize()
    level["level_m"] = pd.to_numeric(
        level["level_m"], errors="coerce"
    )

    level = (
        level.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    if len(level) == 0:
        raise RuntimeError("nivel_canonico vacio.")

    anchor_date = pd.Timestamp(level.iloc[-1]["date"])
    anchor_level = float(level.iloc[-1]["level_m"])

    dev = level[
        (level["date"] >= S_LEVEL_START)
        & (level["date"] <= S_LEVEL_END)
    ].copy()

    if len(dev) < 50:
        raise RuntimeError(f"Pocas observaciones para S_month: {len(dev)}")

    parts = []
    integral = 0.0

    for i in range(1, len(dev)):
        d0 = pd.Timestamp(dev.iloc[i - 1]["date"])
        d1 = pd.Timestamp(dev.iloc[i]["date"])
        h0 = float(dev.iloc[i - 1]["level_m"])
        h1 = float(dev.iloc[i]["level_m"])

        nd = int((d1 - d0).days)
        if nd <= 0:
            continue

        rate = (h1 - h0) / nd
        dates = pd.date_range(
            d0 + pd.Timedelta(days=1),
            d1,
            freq="D",
        )
        if len(dates) != nd:
            raise RuntimeError("Asignacion diaria de S inconsistente.")

        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "obs_dh_rate_m_day": rate,
                }
            )
        )
        integral += rate * nd

    daily = pd.concat(parts, ignore_index=True)
    daily["month"] = daily["date"].dt.month

    raw = daily.groupby("month")["obs_dh_rate_m_day"].mean()

    if set(raw.index) != set(range(1, 13)):
        raise RuntimeError(
            f"S_month incompleto: meses={list(raw.index)}"
        )

    s_raw = {m: float(raw.loc[m]) for m in range(1, 13)}

    endpoint_change = float(
        dev.iloc[-1]["level_m"] - dev.iloc[0]["level_m"]
    )

    if abs(integral - endpoint_change) > 1e-12:
        raise RuntimeError(
            "S_month no reconstruye exactamente el cambio observado."
        )

    tag(
        "S_MONTH",
        n_level_obs=len(dev),
        n_daily=len(daily),
        date_min=dev["date"].min().date(),
        date_max=dev["date"].max().date(),
        integral_m=f"{integral:.12f}",
        endpoint_change_m=f"{endpoint_change:.12f}",
    )
    tag(
        "LEVEL_ANCHOR",
        date=anchor_date.date(),
        level_m=f"{anchor_level:.6f}",
    )

    return s_raw, anchor_date, anchor_level, dev


def s_for_dates(
    dates: pd.Series,
    s_raw: dict[int, float],
) -> np.ndarray:
    d = pd.to_datetime(dates)
    months = d.dt.month.to_numpy(int)
    years = d.dt.year.to_numpy(int)

    out = np.empty(len(d), dtype=float)

    for year in np.unique(years):
        year = int(year)
        ndays = 366 if calendar.isleap(year) else 365

        annual_mean = sum(
            s_raw[m] * calendar.monthrange(year, m)[1]
            for m in range(1, 13)
        ) / ndays

        mask = years == year
        out[mask] = np.asarray(
            [
                s_raw[int(m)] - annual_mean
                for m in months[mask]
            ],
            dtype=float,
        )

    return out


def audit_s_zero(
    s_raw: dict[int, float],
    first_year: int,
    last_year: int,
) -> float:
    max_abs = 0.0

    for year in range(first_year, last_year + 1):
        ndays = 366 if calendar.isleap(year) else 365
        annual_mean = sum(
            s_raw[m] * calendar.monthrange(year, m)[1]
            for m in range(1, 13)
        ) / ndays

        total = sum(
            (s_raw[m] - annual_mean)
            * calendar.monthrange(year, m)[1]
            for m in range(1, 13)
        )
        max_abs = max(max_abs, abs(total))

    return max_abs


                                                                               
               
                                                                               

def monthly_climatology(
    df: pd.DataFrame,
    value_col: str,
) -> dict[int, float]:
    x = df.copy()
    x["date"] = pd.to_datetime(
        x["date"], errors="raise"
    ).dt.normalize()

    x = x[
        (x["date"] >= REFERENCE_START)
        & (x["date"] <= REFERENCE_END)
    ].copy()

    x["month"] = x["date"].dt.month
    g = x.groupby("month")[value_col].mean()

    if set(g.index) != set(range(1, 13)):
        raise RuntimeError(
            f"Climatologia incompleta para {value_col}: "
            f"meses={list(g.index)}"
        )

    return {m: float(g.loc[m]) for m in range(1, 13)}


                                                                               
         
                                                                               

def integrate_trajectory(
    g: pd.DataFrame,
    clim: dict[str, dict[int, float]],
    s_raw: dict[int, float],
    anchor_level: float,
) -> pd.DataFrame:
    x = g.sort_values("date").copy()
    months = x["date"].dt.month

    x["P_clim_mm_day"] = months.map(clim["P"]).to_numpy(float)
    x["E_clim_mm_day"] = months.map(clim["E"]).to_numpy(float)
    x["Q_clim_m3s"] = months.map(clim["Q"]).to_numpy(float)

    x["P_anom_mm_day"] = (
        x["P_mm_day"].to_numpy(float)
        - x["P_clim_mm_day"].to_numpy(float)
    )
    x["E_anom_mm_day"] = (
        x["E_mm_day"].to_numpy(float)
        - x["E_clim_mm_day"].to_numpy(float)
    )
    x["Q_anom_m3s"] = (
        x["Q_m3s"].to_numpy(float)
        - x["Q_clim_m3s"].to_numpy(float)
    )

    x["dH_P_m"] = AP * x["P_anom_mm_day"].to_numpy(float) / 1000.0
    x["dH_E_m"] = -AE * x["E_anom_mm_day"].to_numpy(float) / 1000.0
    x["dH_Q_m"] = (
        AQ
        * x["Q_anom_m3s"].to_numpy(float)
        * 86400.0
        / LAKE_AREA_M2
    )
    x["dH_S_m"] = s_for_dates(x["date"], s_raw)

    x["dH_total_m"] = (
        x["dH_P_m"]
        + x["dH_E_m"]
        + x["dH_Q_m"]
        + x["dH_S_m"]
    )

    vals = x[
        [
            "dH_P_m",
            "dH_E_m",
            "dH_Q_m",
            "dH_S_m",
            "dH_total_m",
        ]
    ].to_numpy(float)

    if not np.isfinite(vals).all():
        raise RuntimeError(
            f"No finitos en balance de "
            f"{x['model'].iloc[0]}/{x['scenario'].iloc[0]}"
        )

    x["cum_P_m"] = x["dH_P_m"].cumsum()
    x["cum_E_m"] = x["dH_E_m"].cumsum()
    x["cum_Q_m"] = x["dH_Q_m"].cumsum()
    x["cum_S_m"] = x["dH_S_m"].cumsum()

    x["level_change_m"] = x["dH_total_m"].cumsum()
    x["level_m"] = anchor_level + x["level_change_m"]

    return x


def horizon_summary(
    daily: pd.DataFrame,
    anchor_level: float,
) -> pd.DataFrame:
    rows = []

    for (model, scenario), g in daily.groupby(
        ["model", "scenario"],
        sort=True,
    ):
        for year in HORIZONS:
            yy = g[g["date"].dt.year == year]
            if len(yy) == 0:
                raise RuntimeError(
                    f"Falta horizonte {year}: {model}/{scenario}"
                )

            end = yy.iloc[-1]

            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "horizon": year,
                    "level_mean_m": float(yy["level_m"].mean()),
                    "level_end_m": float(end["level_m"]),
                    "change_mean_vs_anchor_m": float(
                        yy["level_m"].mean() - anchor_level
                    ),
                    "change_end_vs_anchor_m": float(
                        end["level_m"] - anchor_level
                    ),
                    "cum_P_end_m": float(end["cum_P_m"]),
                    "cum_E_end_m": float(end["cum_E_m"]),
                    "cum_Q_end_m": float(end["cum_Q_m"]),
                    "cum_S_end_m": float(end["cum_S_m"]),
                }
            )

    return pd.DataFrame(rows)


def ensemble_summary(hdf: pd.DataFrame) -> pd.DataFrame:
    rows = []

    metrics = [
        "level_mean_m",
        "level_end_m",
        "change_mean_vs_anchor_m",
        "change_end_vs_anchor_m",
        "cum_P_end_m",
        "cum_E_end_m",
        "cum_Q_end_m",
        "cum_S_end_m",
    ]

    for (scenario, horizon), g in hdf.groupby(
        ["scenario", "horizon"],
        sort=True,
    ):
        row = {
            "scenario": scenario,
            "horizon": int(horizon),
            "n_models": int(g["model"].nunique()),
        }

        for metric in metrics:
            v = g[metric].to_numpy(float)
            row[f"{metric}_p05"] = float(np.quantile(v, 0.05))
            row[f"{metric}_q25"] = float(np.quantile(v, 0.25))
            row[f"{metric}_median"] = float(np.quantile(v, 0.50))
            row[f"{metric}_q75"] = float(np.quantile(v, 0.75))
            row[f"{metric}_p95"] = float(np.quantile(v, 0.95))

        rows.append(row)

    return pd.DataFrame(rows)


                                                                               
      
                                                                               

def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("=" * 128)
    print("PASO 15C - CONGELAR RAMA FUTURE-COMPATIBLE DE NIVEL")
    print("=" * 128)

    for path, label in [
        (Q_FILE, "Q_13K"),
        (Q_META, "Q_META_13K"),
        (E_FILE, "E_14D"),
        (E_META, "E_META_14D"),
        (REFERENCE_FILE, "reference_1981_2014"),
        (LEVEL_FILE, "nivel_canonico"),
    ]:
        require(path, label)

    qmeta = read_json(Q_META)
    emeta = read_json(E_META)

    qf = pd.read_pickle(Q_FILE).copy()
    ef = pd.read_pickle(E_FILE).copy()

    qf["date"] = pd.to_datetime(
        qf["date"], errors="raise"
    ).dt.normalize()
    ef["date"] = pd.to_datetime(
        ef["date"], errors="raise"
    ).dt.normalize()

    qf["model"] = qf["model"].astype(str)
    ef["model"] = ef["model"].astype(str)
    qf["scenario"] = qf["scenario"].astype(str).str.lower()
    ef["scenario"] = ef["scenario"].astype(str).str.lower()

    if qf.duplicated(["model", "scenario", "date"]).any():
        raise RuntimeError("Q_13K contiene duplicados.")
    if ef.duplicated(["model", "scenario", "date"]).any():
        raise RuntimeError("E_14D contiene duplicados.")

    models_q = sorted(qf["model"].unique())
    models_e = sorted(ef["model"].unique())

    if models_q != models_e:
        raise RuntimeError("Los conjuntos de GCM en Q y E no coinciden.")
    if len(models_q) != EXPECTED_N_GCM:
        raise RuntimeError(
            f"Esperaba {EXPECTED_N_GCM} GCM y encontre {len(models_q)}."
        )

    tag(
        "FROZEN_INPUTS",
        n_gcm=len(models_q),
        q_rows=len(qf),
        e_rows=len(ef),
        q_scenarios=sorted(qf["scenario"].unique()),
        e_scenarios=sorted(ef["scenario"].unique()),
    )

    mod = load_13k()

    reference = mod.load_reference_base()
    reference["date"] = pd.to_datetime(
        reference["date"], errors="raise"
    ).dt.normalize()

    q_models = mod.load_models()
    if len(q_models) != 5:
        raise RuntimeError(f"Esperaba 5 modelos Q; encontre {len(q_models)}.")

    intercept = float(
        qmeta["total_flow_correction_10D"]["intercept"]
    )
    slope = float(
        qmeta["total_flow_correction_10D"]["slope"]
    )

    paths = cmip_paths(qmeta)

    s_raw, anchor_date, anchor_level, s_level_dev = load_level_and_build_s()

    if anchor_date + pd.Timedelta(days=1) != EXPECTED_FIRST_DATE:
        raise RuntimeError(
            f"Ancla inesperada: {anchor_date}; "
            f"se esperaba inicio {EXPECTED_FIRST_DATE.date()}."
        )

    s_zero_max = audit_s_zero(
        s_raw,
        EXPECTED_FIRST_DATE.year,
        EXPECTED_LAST_DATE.year,
    )
    tag(
        "S_ZERO_QC",
        max_abs_annual_sum_m=f"{s_zero_max:.3e}",
    )
    if s_zero_max > 1e-12:
        raise RuntimeError("S_month no tiene suma anual cero.")

                                                                               
                                                             
                                                                               
    p_future_parts = []
    p_hist_by_model = {}
    q_hist_by_model = {}
    q_recon_rows = []

    for i, model in enumerate(models_q, 1):
        print(f"\n[MODEL_PREP] {i:02d}/{len(models_q):02d} | {model}")

        hist_path = paths[(model, "historical")]
        require(hist_path, f"CMIP_hist_{model}")

        hist_raw = mod.load_cmip_file(hist_path)
        hist_raw["date"] = pd.to_datetime(
            hist_raw["date"], errors="raise"
        ).dt.normalize()

        params = mod.fit_harmonization_parameters(
            reference,
            hist_raw,
        )

        hist_base = mod.apply_harmonization(
            hist_raw,
            params,
        )
        hist_base["date"] = pd.to_datetime(
            hist_base["date"], errors="raise"
        ).dt.normalize()

        p_hist_by_model[model] = hist_base[
            ["date", "precip_consensus_mm"]
        ].rename(
            columns={"precip_consensus_mm": "P_mm_day"}
        ).copy()

        hist_feat = ensure_feature_date(
            mod.build_72_features(hist_base),
            hist_base,
        )
        q_hist = predict_q_from_features(
            hist_feat,
            q_models,
            intercept,
            slope,
        ).dropna(subset=["Q_m3s"])

        q_hist_by_model[model] = q_hist.copy()

        hist_tail = hist_base.tail(365).copy()

        for scenario in SCENARIOS:
            fut_path = paths[(model, scenario)]
            require(fut_path, f"CMIP_{model}_{scenario}")

            fut_raw = mod.load_cmip_file(fut_path)
            fut_raw["date"] = pd.to_datetime(
                fut_raw["date"], errors="raise"
            ).dt.normalize()

            fut_base = mod.apply_harmonization(
                fut_raw,
                params,
            )
            fut_base["date"] = pd.to_datetime(
                fut_base["date"], errors="raise"
            ).dt.normalize()

            p = fut_base[
                ["date", "precip_consensus_mm"]
            ].rename(
                columns={"precip_consensus_mm": "P_mm_day"}
            ).copy()
            p["model"] = model
            p["scenario"] = scenario
            p_future_parts.append(
                p[["date", "model", "scenario", "P_mm_day"]]
            )

                                                                  
            combo = pd.concat(
                [hist_tail, fut_base],
                ignore_index=True,
            )
            feat = ensure_feature_date(
                mod.build_72_features(combo),
                combo,
            )
            qr = predict_q_from_features(
                feat,
                q_models,
                intercept,
                slope,
            )
            qr = qr[
                qr["date"] >= fut_base["date"].min()
            ].copy()

            frozen = qf[
                (qf["model"] == model)
                & (qf["scenario"] == scenario)
            ][["date", "q_total_corr_10d_m3s"]].rename(
                columns={"q_total_corr_10d_m3s": "Q_frozen"}
            )

            cmp = frozen.merge(
                qr.rename(columns={"Q_m3s": "Q_recon"}),
                on="date",
                how="outer",
                indicator=True,
            )

            if not (cmp["_merge"] == "both").all():
                raise RuntimeError(
                    f"Desalineacion al reconstruir Q: {model}/{scenario}"
                )

            diff = (
                cmp["Q_recon"].to_numpy(float)
                - cmp["Q_frozen"].to_numpy(float)
            )

            q_recon_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "mae": float(np.mean(np.abs(diff))),
                    "max_abs": float(np.max(np.abs(diff))),
                }
            )

            del fut_raw, fut_base, combo, feat, qr, frozen, cmp
            gc.collect()

        del hist_raw, hist_base, hist_feat, q_hist, hist_tail
        gc.collect()

    qrec = pd.DataFrame(q_recon_rows)
    qrec_mae_max = float(qrec["mae"].max())
    qrec_abs_max = float(qrec["max_abs"].max())

    tag(
        "Q_RECON_GUARD",
        mae_max=f"{qrec_mae_max:.3e}",
        max_abs_global=f"{qrec_abs_max:.3e}",
    )

    if (
        qrec_mae_max > Q_RECON_MAE_TOL
        or qrec_abs_max > Q_RECON_MAX_ABS_TOL
    ):
        raise RuntimeError("La reconstruccion de Q no reproduce 13K.")

                                                                               
                              
                                                                               
    e_hist = ef[ef["scenario"] == "historical"].copy()

    climatologies = {}

    for model in models_q:
        em = e_hist[e_hist["model"] == model][
            ["date", "evap_mm_day"]
        ].copy()

        climatologies[model] = {
            "P": monthly_climatology(
                p_hist_by_model[model],
                "P_mm_day",
            ),
            "E": monthly_climatology(
                em,
                "evap_mm_day",
            ),
            "Q": monthly_climatology(
                q_hist_by_model[model],
                "Q_m3s",
            ),
        }

                                                                               
                       
                                                                               
    pf = pd.concat(p_future_parts, ignore_index=True)
    pf = pf.sort_values(
        ["model", "scenario", "date"]
    ).reset_index(drop=True)

    eu = ef[
        ef["scenario"].isin(SCENARIOS)
    ][["date", "model", "scenario", "evap_mm_day"]].rename(
        columns={"evap_mm_day": "E_mm_day"}
    )

    qu = qf[
        ["date", "model", "scenario", "q_total_corr_10d_m3s"]
    ].rename(
        columns={"q_total_corr_10d_m3s": "Q_m3s"}
    )

    peq = (
        pf.merge(
            eu,
            on=["date", "model", "scenario"],
            how="outer",
            indicator="merge_PE",
        )
    )

    if not (peq["merge_PE"] == "both").all():
        raise RuntimeError(
            f"Desalineacion P/E: "
            f"{peq['merge_PE'].value_counts().to_dict()}"
        )

    peq = peq.drop(columns="merge_PE").merge(
        qu,
        on=["date", "model", "scenario"],
        how="outer",
        indicator="merge_PEQ",
    )

    if not (peq["merge_PEQ"] == "both").all():
        raise RuntimeError(
            f"Desalineacion P/E/Q: "
            f"{peq['merge_PEQ'].value_counts().to_dict()}"
        )

    peq = peq.drop(columns="merge_PEQ")
    peq["date"] = pd.to_datetime(
        peq["date"], errors="raise"
    ).dt.normalize()

    if peq.duplicated(["model", "scenario", "date"]).any():
        raise RuntimeError("Duplicados P/E/Q despues de merge.")

    if peq[["P_mm_day", "E_mm_day", "Q_m3s"]].isna().any().any():
        raise RuntimeError("NaN en P/E/Q despues de merge.")

    future = peq[peq["date"] > anchor_date].copy()

    groups = future.groupby(["model", "scenario"]).size()

    if len(groups) != EXPECTED_N_TRAJ:
        raise RuntimeError(
            f"Esperaba {EXPECTED_N_TRAJ} trayectorias; encontre {len(groups)}."
        )

    if not (groups == EXPECTED_ROWS_PER_TRAJ).all():
        raise RuntimeError(
            "No todas las trayectorias tienen el numero esperado de dias."
        )

    if future["date"].min() != EXPECTED_FIRST_DATE:
        raise RuntimeError("Fecha inicial futura inesperada.")
    if future["date"].max() != EXPECTED_LAST_DATE:
        raise RuntimeError("Fecha final futura inesperada.")

    tag(
        "PEQ_ALIGNMENT",
        rows=len(future),
        trajectories=len(groups),
        rows_per_trajectory=EXPECTED_ROWS_PER_TRAJ,
        date_min=future["date"].min().date(),
        date_max=future["date"].max().date(),
    )

                                                                               
                        
                                                                               
    parts = []

    for (model, scenario), g in future.groupby(
        ["model", "scenario"],
        sort=True,
    ):
        x = integrate_trajectory(
            g,
            climatologies[model],
            s_raw,
            anchor_level,
        )
        parts.append(x)

    daily = pd.concat(parts, ignore_index=True)
    daily = daily.sort_values(
        ["model", "scenario", "date"]
    ).reset_index(drop=True)

    if len(daily) != EXPECTED_DAILY_ROWS:
        raise RuntimeError(
            f"Filas nivel={len(daily)}; esperadas={EXPECTED_DAILY_ROWS}."
        )

    if daily.duplicated(["model", "scenario", "date"]).any():
        raise RuntimeError("Duplicados en salida diaria de nivel.")

    numeric_cols = [
        "P_mm_day",
        "E_mm_day",
        "Q_m3s",
        "P_clim_mm_day",
        "E_clim_mm_day",
        "Q_clim_m3s",
        "P_anom_mm_day",
        "E_anom_mm_day",
        "Q_anom_m3s",
        "dH_P_m",
        "dH_E_m",
        "dH_Q_m",
        "dH_S_m",
        "dH_total_m",
        "cum_P_m",
        "cum_E_m",
        "cum_Q_m",
        "cum_S_m",
        "level_change_m",
        "level_m",
    ]

    if not np.isfinite(
        daily[numeric_cols].to_numpy(float)
    ).all():
        raise RuntimeError("No finitos en salida diaria final.")

    hdf = horizon_summary(
        daily,
        anchor_level,
    )
    edf = ensemble_summary(hdf)

                                                                               
                                  
                                                                               
    checkpoint_rows = []

    for key, expected in EXPECTED_LEVEL_MEAN_MEDIAN.items():
        scenario, year = key
        got = float(
            edf.loc[
                (edf["scenario"] == scenario)
                & (edf["horizon"] == year),
                "level_mean_m_median",
            ].iloc[0]
        )
        diff = got - expected

        checkpoint_rows.append(
            {
                "scenario": scenario,
                "horizon": year,
                "expected_m": expected,
                "computed_m": got,
                "difference_m": diff,
            }
        )

        if abs(diff) > CHECKPOINT_TOL_M:
            raise RuntimeError(
                f"Checkpoint fallo {scenario}/{year}: "
                f"expected={expected}, got={got}, diff={diff}"
            )

    checkpoints = pd.DataFrame(checkpoint_rows)

    print("\n[CHECKPOINTS_15A]")
    print(checkpoints.to_string(index=False))

    compact_cols = [
        "scenario",
        "horizon",
        "n_models",
        "level_mean_m_median",
        "level_mean_m_q25",
        "level_mean_m_q75",
        "level_mean_m_p05",
        "level_mean_m_p95",
        "change_mean_vs_anchor_m_median",
        "cum_P_end_m_median",
        "cum_E_end_m_median",
        "cum_Q_end_m_median",
        "cum_S_end_m_median",
    ]

    print("\n[ENSEMBLE_FINAL]")
    print(edf[compact_cols].to_string(index=False))

                                                                               
                                                       
                                                                               
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_pickle(OUT_DAILY)
    hdf.to_csv(OUT_HORIZONS, index=False)
    edf.to_csv(OUT_ENSEMBLE, index=False)

    metadata = {
        "step": "15C",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_FUTURE_COMPATIBLE_LEVEL_BRANCH",
        "interpretation": (
            "Conditional CMIP6 lake-level scenario trajectories; "
            "not deterministic forecasts."
        ),
        "python_version": sys.version,
        "architecture": (
            "DeltaH = S_month_zero_annual + "
            "aP*(P-Pclim_month) - aE*(E-Eclim_month) + "
            "aQ*(Q-Qclim_month)"
        ),
        "coefficients": {
            "aP": AP,
            "aE": AE,
            "aQ": AQ,
            "c_day": C_DAY,
        },
        "lake_area_m2": LAKE_AREA_M2,
        "reference_period": {
            "start": str(REFERENCE_START.date()),
            "end": str(REFERENCE_END.date()),
            "role": (
                "Primary per-GCM monthly anomaly baseline for P/E/Q; "
                "same historical harmonization space used by frozen branches."
            ),
        },
        "seasonal_term": {
            "source_level_period": {
                "start": str(S_LEVEL_START.date()),
                "end": str(S_LEVEL_END.date()),
                "n_observations": int(len(s_level_dev)),
            },
            "method": (
                "Observed level differences distributed exactly over each "
                "intervening day; monthly mean daily rates; then recentered "
                "within each future Gregorian calendar year so annual sum is zero."
            ),
            "raw_monthly_m_day": json_float_dict(s_raw),
            "future_max_abs_annual_sum_m": float(s_zero_max),
        },
        "anchor": {
            "date": str(anchor_date.date()),
            "level_m": float(anchor_level),
            "role": (
                "Initial condition only. The retrospective 2024-2026 period "
                "was not used to select or tune the future architecture."
            ),
        },
        "future_period": {
            "start": str(daily["date"].min().date()),
            "end": str(daily["date"].max().date()),
            "n_gcm": int(daily["model"].nunique()),
            "scenarios": sorted(daily["scenario"].unique().tolist()),
            "n_trajectories": int(
                daily.groupby(["model", "scenario"]).ngroups
            ),
            "n_daily_rows": int(len(daily)),
            "rows_per_trajectory": int(EXPECTED_ROWS_PER_TRAJ),
        },
        "climatologies_by_model": {
            model: {
                component: json_float_dict(month_map)
                for component, month_map in climatologies[model].items()
            }
            for model in models_q
        },
        "pre_freeze_audits": {
            "15A_bridge_72f": {
                "coverage": "72/72",
                "median_nmae_iqr": 0.000242709,
                "p95_nmae_iqr": 0.00504924,
                "max_nmae_iqr": 0.0219831,
            },
            "13K_reconstruction_guard_current_run": {
                "mae_max": qrec_mae_max,
                "max_abs_global": qrec_abs_max,
            },
            "15B_evaporation_reconstruction": {
                "max_abs_mm_day": 7.993605777301127e-15,
                "rlds_nonfinite_total": 0,
                "rlds_nonpositive_total": 0,
                "Rn_nonpositive_total": 0,
            },
            "15B_temperature_order": {
                "max_frequency_pct": 1.0346693833370475,
                "max_severity_c": 2.112023343385502,
                "note": (
                    "No silent reordering/clipping is applied in production."
                ),
            },
            "15B2_nonselective_Q_temperature_order_sensitivity": {
                "repair_definition": (
                    "Tmin=min(Tmin,Tmean); Tmax=max(Tmax,Tmean); Tmean unchanged"
                ),
                "max_abs_level_effect_m_2050": 0.003009347,
                "max_abs_level_effect_m_2090": 0.008554888,
                "production_decision": (
                    "Frozen 13K retained unchanged; temperature-order issue "
                    "documented as negligible for integrated level relative "
                    "to ensemble spread."
                ),
            },
            "baseline_sensitivity_1995_2014_vs_1981_2014": {
                "role": "structural sensitivity only; not used for selection",
                "max_abs_level_difference_m_2050": 3.634526,
                "max_abs_level_difference_m_2090": 9.680396,
            },
        },
        "known_limitations": [
            (
                "Evaporation branch is radiation-dominated; air temperature "
                "is used as a water-surface-temperature proxy for outgoing longwave."
            ),
            (
                "Tree-based tributary models saturate outside their training "
                "domain and do not continuously extrapolate physical response; "
                "future temperature-domain shift grows toward 2050/2090."
            ),
            (
                "Tmin/Tmean/Tmax are corrected separately and can rarely violate "
                "physical ordering; nonselective sensitivity showed negligible "
                "effect on integrated level, so frozen 13K is not altered."
            ),
            (
                "The 1995-2014 anomaly-baseline sensitivity becomes material "
                "for some individual GCMs at long horizons; production remains "
                "on the pre-specified 1981-2014 harmonization baseline."
            ),
            (
                "2030 and 2050 are the focus horizons. 2090 is retained as a "
                "stress/tail horizon."
            ),
            (
                "Trajectories are conditional CMIP6 scenarios, not deterministic "
                "lake-level forecasts."
            ),
        ],
        "checkpoints_15A": checkpoint_rows,
        "source_hashes": {
            "q_13k_pickle": {
                "path": str(Q_FILE.relative_to(ROOT)),
                "sha256": sha256_file(Q_FILE),
            },
            "q_13k_metadata": {
                "path": str(Q_META.relative_to(ROOT)),
                "sha256": sha256_file(Q_META),
            },
            "e_14d_pickle": {
                "path": str(E_FILE.relative_to(ROOT)),
                "sha256": sha256_file(E_FILE),
            },
            "e_14d_metadata": {
                "path": str(E_META.relative_to(ROOT)),
                "sha256": sha256_file(E_META),
            },
            "reference_1981_2014": {
                "path": str(REFERENCE_FILE.relative_to(ROOT)),
                "sha256": sha256_file(REFERENCE_FILE),
            },
            "level_canonical": {
                "path": str(LEVEL_FILE.relative_to(ROOT)),
                "sha256": sha256_file(LEVEL_FILE),
            },
            "script_13k": {
                "path": str(SCRIPT_13K.relative_to(ROOT)),
                "sha256": sha256_file(SCRIPT_13K),
            },
        },
        "outputs": {
            "daily_level_pickle": str(OUT_DAILY.relative_to(ROOT)),
            "horizon_summary_csv": str(OUT_HORIZONS.relative_to(ROOT)),
            "ensemble_summary_csv": str(OUT_ENSEMBLE.relative_to(ROOT)),
            "metadata_json": str(OUT_META.relative_to(ROOT)),
        },
    }

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 128)
    print("[FROZEN] RAMA FUTURE-COMPATIBLE DE NIVEL CONGELADA")
    tag("OUTPUT", path=OUT_DAILY, rows=len(daily), sha256=sha256_file(OUT_DAILY))
    tag("OUTPUT", path=OUT_HORIZONS, rows=len(hdf), sha256=sha256_file(OUT_HORIZONS))
    tag("OUTPUT", path=OUT_ENSEMBLE, rows=len(edf), sha256=sha256_file(OUT_ENSEMBLE))
    tag("OUTPUT", path=OUT_META, sha256=sha256_file(OUT_META))
    print(
        "[NEXT] Revisar consola y artefactos. Solo despues de confirmar este "
        "freeze corresponde abrir la fase de batimetria/volumen."
    )
    print("=" * 128)


if __name__ == "__main__":
    main()
