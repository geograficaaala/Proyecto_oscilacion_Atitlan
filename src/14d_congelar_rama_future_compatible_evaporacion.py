                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CMIP_DIR = ROOT / "data" / "cmip6" / "raw"

REF_FILE = (
    ROOT / "data" / "reference_1981_2014" / "procesado"
    / "atitlan_referencia_1981_2014.csv"
)

LONGWAVE_FILE = (
    ROOT / "data" / "evaporation" / "procesado"
    / "atitlan_era5_longwave_1981_2026.csv"
)

CANDIDATES_FILE = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "11c_evaporacion_diaria_candidatos.csv"
)

PARAMS_11A_FILE = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "11a_parametros_candidatos.json"
)

OUT_DIR = ROOT / "outputs" / "evaporacion_future_compatible"
OUT_DAILY = OUT_DIR / "evaporacion_diaria_cmip6_future_compatible.pkl"
OUT_PARAMS = OUT_DIR / "parametros_evaporacion_cmip6_future_compatible.json"

HIST_START = pd.Timestamp("1981-01-01")
HIST_END = pd.Timestamp("2014-12-31")
BASE_START = pd.Timestamp("1995-01-01")
BASE_END = pd.Timestamp("2014-12-31")

HORIZONS = [
    ("2030", pd.Timestamp("2021-01-01"), pd.Timestamp("2040-12-31")),
    ("2050", pd.Timestamp("2041-01-01"), pd.Timestamp("2060-12-31")),
    ("2090", pd.Timestamp("2081-01-01"), pd.Timestamp("2100-12-31")),
]

WATER_ALBEDO_DEFAULT = 0.08
WATER_EMISSIVITY_DEFAULT = 0.97
SIGMA_DEFAULT = 4.903e-9


def norm(x):
    s = str(x).strip().lower()
    for a, b in [
        ("á", "a"), ("é", "e"), ("í", "i"),
        ("ó", "o"), ("ú", "u"), ("ñ", "n"),
    ]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def detect_col(df, aliases):
    nmap = {norm(c): c for c in df.columns}
    for alias in aliases:
        if norm(alias) in nmap:
            return nmap[norm(alias)]
    for col in df.columns:
        nc = norm(col)
        if any(norm(a) in nc for a in aliases):
            return col
    raise KeyError(
        f"No se detecto columna entre {aliases}. "
        f"Disponibles: {list(df.columns)}"
    )


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def physical_constants():
    albedo = WATER_ALBEDO_DEFAULT
    emissivity = WATER_EMISSIVITY_DEFAULT
    sigma = SIGMA_DEFAULT

    if PARAMS_11A_FILE.exists():
        with open(PARAMS_11A_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        physical = data.get("physical_constants", {})
        albedo = float(physical.get("water_albedo", albedo))
        emissivity = float(physical.get("water_emissivity", emissivity))
        sigma = float(physical.get("sigma_mj_m2_day_k4", sigma))

    return albedo, emissivity, sigma


def radiation_scale():
    df = pd.read_csv(CANDIDATES_FILE, low_memory=False)
    raw_col = detect_col(df, ["RADIATION_ENERGY_PROXY__RAW"])
    scale_col = detect_col(df, ["RADIATION_ENERGY_PROXY__SCALE"])

    raw = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(float)
    scaled = pd.to_numeric(df[scale_col], errors="coerce").to_numpy(float)

    valid = np.isfinite(raw) & np.isfinite(scaled) & (np.abs(raw) > 1e-12)
    ratio = scaled[valid] / raw[valid]

    if len(ratio) < 1000:
        raise RuntimeError("Cobertura insuficiente para recuperar SCALE.")

    factor = float(np.median(ratio))
    max_dev = float(np.max(np.abs(ratio - factor)))

    if max_dev > 1e-9:
        raise RuntimeError("RADIATION_ENERGY_PROXY__SCALE no es constante.")

    return factor


def load_reference():
    base = pd.read_csv(REF_FILE, low_memory=False)
    lw = pd.read_csv(LONGWAVE_FILE, low_memory=False)

    base_date = detect_col(base, ["date", "fecha"])
    tmean_col = detect_col(base, ["tmean_c", "tas_c", "tmean"])
    solar_col = detect_col(base, ["solar_mj_m2", "rsds_mj_m2_day", "solar"])

    lw_date = detect_col(lw, ["date", "fecha"])
    lw_col = detect_col(lw, ["longwave_mj_m2", "rlds_mj_m2_day", "longwave"])

    ref = pd.DataFrame({
        "date": pd.to_datetime(base[base_date], errors="raise").dt.normalize(),
        "tmean_ref": pd.to_numeric(base[tmean_col], errors="coerce"),
        "solar_ref": pd.to_numeric(base[solar_col], errors="coerce"),
    })

    lwd = pd.DataFrame({
        "date": pd.to_datetime(lw[lw_date], errors="raise").dt.normalize(),
        "rlds_ref": pd.to_numeric(lw[lw_col], errors="coerce"),
    })

    ref = (
        ref.merge(lwd, on="date", how="inner", validate="one_to_one")
        .loc[lambda x: x["date"].between(HIST_START, HIST_END)]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(ref) != 12418:
        raise RuntimeError(f"Referencia 1981-2014: {len(ref)} filas; esperadas 12418.")

    return ref


def read_cmip(path):
    df = pd.read_csv(path, low_memory=False)

    required = [
        "date", "model", "scenario",
        "tas_c", "rsds_mj_m2_day", "rlds_mj_m2_day",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: faltan {missing}")

    out = df[required].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()

    for col in ["tas_c", "rsds_mj_m2_day", "rlds_mj_m2_day"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[["tas_c", "rsds_mj_m2_day", "rlds_mj_m2_day"]].isna().any().any():
        raise ValueError(f"{path.name}: NaN en variables requeridas.")

    if out["date"].duplicated().any():
        raise ValueError(f"{path.name}: fechas duplicadas.")

    return out.sort_values("date").reset_index(drop=True)


def fit_harmonization(hist, ref):
    m = hist.merge(ref, on="date", how="inner", validate="one_to_one")
    m = m.loc[m["date"].between(HIST_START, HIST_END)].copy()

    if len(m) != 12418:
        raise RuntimeError(
            f"Historical-reference overlap: {len(m)} filas; esperadas 12418."
        )

    t_shift = {}
    solar_scale = {}
    rlds_shift = {}

    for month in range(1, 13):
        sub = m.loc[m["date"].dt.month == month]

        t_shift[month] = float(np.mean(sub["tmean_ref"] - sub["tas_c"]))

        solar_model = float(np.mean(sub["rsds_mj_m2_day"]))
        if solar_model <= 0:
            raise ValueError(f"Solar media invalida en mes {month}.")

        solar_scale[month] = float(np.mean(sub["solar_ref"]) / solar_model)
        rlds_shift[month] = float(np.mean(sub["rlds_ref"] - sub["rlds_mj_m2_day"]))

    return {
        "tmean_monthly_shift": t_shift,
        "solar_monthly_scale": solar_scale,
        "rlds_monthly_shift": rlds_shift,
    }


def apply_harmonization(df, params):
    out = df.copy()
    months = out["date"].dt.month.to_numpy(int)

    t_shift = np.asarray(
        [params["tmean_monthly_shift"][int(m)] for m in months],
        dtype=float,
    )
    solar_scale = np.asarray(
        [params["solar_monthly_scale"][int(m)] for m in months],
        dtype=float,
    )
    rlds_shift = np.asarray(
        [params["rlds_monthly_shift"][int(m)] for m in months],
        dtype=float,
    )

    out["tmean_corr_c"] = out["tas_c"].to_numpy(float) + t_shift
    out["solar_corr_mj_m2_day"] = (
        out["rsds_mj_m2_day"].to_numpy(float) * solar_scale
    )
    out["rlds_corr_mj_m2_day"] = (
        out["rlds_mj_m2_day"].to_numpy(float) + rlds_shift
    )

    return out


def compute_evap(df, scale_factor, albedo, emissivity, sigma):
    t = df["tmean_corr_c"].to_numpy(float)
    solar = np.maximum(df["solar_corr_mj_m2_day"].to_numpy(float), 0.0)
    rlds = np.maximum(df["rlds_corr_mj_m2_day"].to_numpy(float), 0.0)

    lw_up = emissivity * sigma * np.power(t + 273.15, 4)
    rn_water = (1.0 - albedo) * solar + rlds - lw_up
    evap = scale_factor * 0.408 * np.maximum(rn_water, 0.0)

    if (~np.isfinite(evap)).any():
        raise RuntimeError("Evaporacion no finita.")
    if (evap < 0).any():
        raise RuntimeError("Evaporacion negativa.")

    return evap


def main():
    print("=" * 128)
    print("PASO 14D - CONGELAR RAMA FUTURE-COMPATIBLE DE EVAPORACION")
    print("=" * 128)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_reference()
    scale_factor = radiation_scale()
    albedo, emissivity, sigma = physical_constants()

    historical_files = sorted(CMIP_DIR.glob("*historical*.csv"))
    future_files = sorted(
        list(CMIP_DIR.glob("*ssp245*.csv"))
        + list(CMIP_DIR.glob("*ssp585*.csv"))
    )

    if len(historical_files) != 15:
        raise RuntimeError(f"Historical: {len(historical_files)}; esperados 15.")
    if len(future_files) != 30:
        raise RuntimeError(f"SSP: {len(future_files)}; esperados 30.")

    params_by_model = {}
    baseline = {}
    frames = []

    for path in historical_files:
        hist = read_cmip(path)
        model = str(hist["model"].iloc[0])

        params = fit_harmonization(hist, ref)
        params_by_model[model] = params

        h = apply_harmonization(hist, params)
        evap = compute_evap(h, scale_factor, albedo, emissivity, sigma)

        frames.append(pd.DataFrame({
            "date": h["date"],
            "model": model,
            "scenario": "historical",
            "evap_mm_day": evap,
        }))

        mask = h["date"].between(BASE_START, BASE_END)
        ebase = evap[mask.to_numpy()]

        baseline[model] = {
            "mean_mm_day": float(np.mean(ebase)),
            "p99_mm_day": float(np.quantile(ebase, 0.99)),
        }

    horizon_rows = []

    for path in future_files:
        fut = read_cmip(path)
        model = str(fut["model"].iloc[0])
        scenario = str(fut["scenario"].iloc[0]).lower()

        if model not in params_by_model:
            raise KeyError(f"Sin parametros historical para {model}.")

        f = apply_harmonization(fut, params_by_model[model])
        evap = compute_evap(f, scale_factor, albedo, emissivity, sigma)

        max_ratio = float(
            np.max(evap) / baseline[model]["p99_mm_day"]
        )

        if max_ratio > 2.0:
            raise RuntimeError(
                f"{model}/{scenario}: max/p99 historical={max_ratio:.6f} > 2."
            )

        frames.append(pd.DataFrame({
            "date": f["date"],
            "model": model,
            "scenario": scenario,
            "evap_mm_day": evap,
        }))

        for horizon, start, end in HORIZONS:
            mask = f["date"].between(start, end)
            e = evap[mask.to_numpy()]

            if len(e) < 6500:
                raise RuntimeError(
                    f"{model}/{scenario}/{horizon}: ventana incompleta."
                )

            mean_e = float(np.mean(e))

            horizon_rows.append({
                "model": model,
                "scenario": scenario,
                "horizon": horizon,
                "mean_mm_day": mean_e,
                "change_pct": (
                    100.0
                    * (
                        mean_e / baseline[model]["mean_mm_day"]
                        - 1.0
                    )
                ),
                "max_mm_day": float(np.max(e)),
                "max_over_hist_p99": max_ratio,
            })

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily = (
        daily.sort_values(["model", "scenario", "date"])
        .reset_index(drop=True)
    )

    dup = int(
        daily.duplicated(["model", "scenario", "date"]).sum()
    )

    if dup != 0:
        raise RuntimeError(f"Duplicados model/scenario/date: {dup}")

    if len(daily) != 1128600:
        raise RuntimeError(
            f"Filas: {len(daily)}; esperadas 1128600."
        )

    daily.to_pickle(OUT_DAILY)

    horizon_df = pd.DataFrame(horizon_rows)

    summary = (
        horizon_df.groupby(["scenario", "horizon"])
        .agg(
            n_models=("model", "nunique"),
            evap_mean_mm_day_median=("mean_mm_day", "median"),
            evap_mean_mm_day_q25=(
                "mean_mm_day",
                lambda x: float(np.quantile(x, 0.25)),
            ),
            evap_mean_mm_day_q75=(
                "mean_mm_day",
                lambda x: float(np.quantile(x, 0.75)),
            ),
            change_pct_median=("change_pct", "median"),
            change_pct_q25=(
                "change_pct",
                lambda x: float(np.quantile(x, 0.25)),
            ),
            change_pct_q75=(
                "change_pct",
                lambda x: float(np.quantile(x, 0.75)),
            ),
            max_mm_day=("max_mm_day", "max"),
            max_over_hist_p99=("max_over_hist_p99", "max"),
        )
        .reset_index()
    )

    source_files = [
        REF_FILE,
        LONGWAVE_FILE,
        CANDIDATES_FILE,
        *historical_files,
        *future_files,
    ]

    if PARAMS_11A_FILE.exists():
        source_files.append(PARAMS_11A_FILE)

    metadata = {
        "paso": "14D",
        "branch": "future_compatible_evaporation",
        "python_version": sys.version,
        "selected_proxy": "RADIATION_ENERGY_PROXY__SCALE",
        "radiation_scale": scale_factor,
        "physical_constants": {
            "water_albedo": albedo,
            "water_emissivity": emissivity,
            "sigma_mj_m2_day_k4": sigma,
        },
        "harmonization": {
            "tmean": "MONTHLY_SHIFT",
            "solar": "MONTHLY_SCALE",
            "rlds": "MONTHLY_SHIFT",
            "fit_period": ["1981-01-01", "2014-12-31"],
            "per_model_parameters": params_by_model,
        },
        "historical_level_audit": {
            "architecture": (
                "monthly anomalies, zero annual drift, "
                "fixed historical balance coefficients"
            ),
            "retrospective_test": {
                "mae_m": 0.183785,
                "wmae_m": 0.183270,
                "rmse_m": 0.233928,
                "bias_m": -0.044463,
                "nse": 0.725888,
                "pearson_r": 0.860727,
            },
        },
        "future_horizon_summary": summary.to_dict(orient="records"),
        "known_limitations": [
            "Radiation-dominated evaporation proxy.",
            "Air temperature is used as a surface-temperature proxy for outgoing longwave.",
            "CMIP6 trajectories are conditional scenarios, not deterministic forecasts.",
            "The 2024-2026 retrospective period is not a pristine prospective test.",
        ],
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in source_files
        },
        "daily_output_sha256": sha256_file(OUT_DAILY),
    }

    with open(OUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("Archivos guardados:")
    print(OUT_DAILY)
    print(OUT_PARAMS)

    print()
    print(f"Filas: {len(daily)}")
    print(f"Modelos: {daily['model'].nunique()}")
    print(f"Escenarios: {sorted(daily['scenario'].unique().tolist())}")
    print(f"Duplicados model/scenario/date: {dup}")
    print(f"Evap min: {daily['evap_mm_day'].min():.12f} mm/d")
    print(f"Evap max: {daily['evap_mm_day'].max():.12f} mm/d")

    print()
    print("Resumen futuro:")
    with pd.option_context(
        "display.max_columns", 30,
        "display.width", 300,
    ):
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print()
    print("PASO 14D COMPLETADO.")


if __name__ == "__main__":
    main()
