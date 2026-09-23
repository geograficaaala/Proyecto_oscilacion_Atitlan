                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

REF_FILE = (
    ROOT / "data" / "reference_1981_2014" / "procesado"
    / "atitlan_referencia_1981_2014.csv"
)

OUT_DIR = ROOT / "outputs" / "evaporacion_future_compatible"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAL_START = pd.Timestamp("1981-01-01")
CAL_END = pd.Timestamp("2004-12-31")
VAL_START = pd.Timestamp("2005-01-01")
VAL_END = pd.Timestamp("2014-12-31")

                                                
                                                                                 
WIND_MEASUREMENT_HEIGHT_M = 10.0
WATER_ALBEDO = 0.08
WATER_EMISSIVITY = 0.97
REFERENCE_ALBEDO = 0.23
REFERENCE_EMISSIVITY = 0.98
PRIESTLEY_TAYLOR_ALPHA = 1.26

                                      
SIGMA_MJ = 4.903e-9

REQUIRED_COLUMNS = [
    "date",
    "evap_mm",
    "pet_mm",
    "pressure_kpa",
    "tmean_c",
    "tmin_c",
    "tmax_c",
    "dewpoint_c",
    "wind_ms",
    "solar_mj_m2",
    "longwave_mj_m2",
]


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sat_vapor_pressure_kpa(temp_c):
    t = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def slope_vapor_pressure_curve_kpa_c(temp_c):
    t = np.asarray(temp_c, dtype=float)
    es = sat_vapor_pressure_kpa(t)
    return 4098.0 * es / np.square(t + 237.3)


def wind_to_2m(u_z, z_m=WIND_MEASUREMENT_HEIGHT_M):
    u = np.asarray(u_z, dtype=float)
    factor = 4.87 / np.log(67.8 * z_m - 5.42)
    return np.maximum(u * factor, 0.0)


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "r2_nse": np.nan,
            "pearson_r": np.nan,
        }

    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
        "r2_nse": (
            float(r2_score(y, p))
            if len(y) > 1 else np.nan
        ),
        "pearson_r": np.nan,
    }

    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson_r"] = float(np.corrcoef(y, p)[0, 1])

    return out


def aggregate_metrics(df, obs_col, pred_col):
    out = {}

    daily = metrics(df[obs_col], df[pred_col])

    for k, v in daily.items():
        out[f"daily_{k}"] = v

    monthly = (
        df.set_index("date")[[obs_col, pred_col]]
        .resample("MS")
        .sum(min_count=1)
        .dropna()
    )

    m = metrics(monthly[obs_col], monthly[pred_col])

    for k, v in m.items():
        out[f"monthly_total_{k}"] = v

    annual = (
        df.set_index("date")[[obs_col, pred_col]]
        .resample("YS")
        .sum(min_count=1)
        .dropna()
    )

    a = metrics(annual[obs_col], annual[pred_col])

    for k, v in a.items():
        out[f"annual_total_{k}"] = v

    monthly["month_num"] = monthly.index.month

    clim = (
        monthly.groupby("month_num")[[obs_col, pred_col]]
        .mean()
    )

    c = metrics(clim[obs_col], clim[pred_col])

    for k, v in c.items():
        out[f"monthly_climatology_{k}"] = v

    obs_ann_mean = float(annual[obs_col].mean())
    pred_ann_mean = float(annual[pred_col].mean())

    out["annual_mean_obs_mm"] = obs_ann_mean
    out["annual_mean_pred_mm"] = pred_ann_mean
    out["annual_mean_bias_mm"] = pred_ann_mean - obs_ann_mean

    if abs(obs_ann_mean) > 1e-12:
        out["annual_mean_relative_bias_pct"] = (
            100.0
            * (pred_ann_mean - obs_ann_mean)
            / obs_ann_mean
        )
    else:
        out["annual_mean_relative_bias_pct"] = np.nan

    if len(annual) >= 3:
        years = annual.index.year.to_numpy(dtype=float)

        out["obs_trend_mm_decade"] = float(
            np.polyfit(
                years,
                annual[obs_col].to_numpy(float),
                1,
            )[0] * 10.0
        )

        out["pred_trend_mm_decade"] = float(
            np.polyfit(
                years,
                annual[pred_col].to_numpy(float),
                1,
            )[0] * 10.0
        )

        out["trend_difference_mm_decade"] = (
            out["pred_trend_mm_decade"]
            - out["obs_trend_mm_decade"]
        )
    else:
        out["obs_trend_mm_decade"] = np.nan
        out["pred_trend_mm_decade"] = np.nan
        out["trend_difference_mm_decade"] = np.nan

    rolling_source = (
        df.set_index("date")[[obs_col, pred_col]]
        .sort_index()
    )

    for window in [30, 90, 365]:
        roll = rolling_source.rolling(
            window=window,
            min_periods=window,
        ).sum().dropna()

        if len(roll):
            out[f"rolling_{window}d_mae_mm"] = float(
                mean_absolute_error(
                    roll[obs_col],
                    roll[pred_col],
                )
            )
        else:
            out[f"rolling_{window}d_mae_mm"] = np.nan

    return out


def fit_scale_through_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    denom = float(np.dot(x, x))

    if denom <= 1e-12:
        raise ValueError("No se puede ajustar escala: predictor degenerado.")

    scale = float(np.dot(x, y) / denom)
    scale = max(scale, 0.0)

    return {
        "mapping": "SCALE",
        "scale": scale,
    }


def fit_affine_positive_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid].reshape(-1, 1)
    y = y[valid]

    if len(y) < 10:
        raise ValueError("Muy pocos datos para ajuste afín.")

    model = LinearRegression(
        fit_intercept=True,
        positive=True,
    )

    model.fit(x, y)

    return {
        "mapping": "AFFINE_POSITIVE",
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
    }


def apply_mapping(x, params):
    x = np.asarray(x, dtype=float)

    if params["mapping"] == "IDENTITY":
        y = x.copy()

    elif params["mapping"] == "SCALE":
        y = float(params["scale"]) * x

    elif params["mapping"] == "AFFINE_POSITIVE":
        y = (
            float(params["intercept"])
            + float(params["slope"]) * x
        )

    else:
        raise ValueError(params["mapping"])

    return np.maximum(y, 0.0)


def build_physical_indices(df, pressure_constant_kpa):
    out = df.copy()

    tmean = out["tmean_c"].to_numpy(float)
    tmin = out["tmin_c"].to_numpy(float)
    tmax = out["tmax_c"].to_numpy(float)
    tdew = out["dewpoint_c"].to_numpy(float)

    solar = np.maximum(
        out["solar_mj_m2"].to_numpy(float),
        0.0,
    )

    lw_down = np.maximum(
        out["longwave_mj_m2"].to_numpy(float),
        0.0,
    )

    wind10 = np.maximum(
        out["wind_ms"].to_numpy(float),
        0.0,
    )

    wind2 = wind_to_2m(
        wind10,
        WIND_MEASUREMENT_HEIGHT_M,
    )

    es_tmin = sat_vapor_pressure_kpa(tmin)
    es_tmax = sat_vapor_pressure_kpa(tmax)

    es_mean = (
        es_tmin + es_tmax
    ) / 2.0

    ea = sat_vapor_pressure_kpa(tdew)

    rh = (
        100.0
        * ea
        / np.maximum(
            sat_vapor_pressure_kpa(tmean),
            1e-9,
        )
    )

    rh = np.clip(
        rh,
        0.0,
        100.0,
    )

    vpd = np.maximum(
        es_mean - ea,
        0.0,
    )

    delta = slope_vapor_pressure_curve_kpa_c(
        tmean
    )

    gamma = (
        0.000665
        * float(pressure_constant_kpa)
    )

    t_kelvin = tmean + 273.15

    lw_up_water = (
        WATER_EMISSIVITY
        * SIGMA_MJ
        * np.power(t_kelvin, 4)
    )

    lw_up_reference = (
        REFERENCE_EMISSIVITY
        * SIGMA_MJ
        * np.power(t_kelvin, 4)
    )

    rn_water = (
        (1.0 - WATER_ALBEDO) * solar
        + lw_down
        - lw_up_water
    )

    rn_reference = (
        (1.0 - REFERENCE_ALBEDO) * solar
        + lw_down
        - lw_up_reference
    )

    rn_water_pos = np.maximum(
        rn_water,
        0.0,
    )

    rn_reference_pos = np.maximum(
        rn_reference,
        0.0,
    )

                                                           
                                  
    equilibrium = (
        0.408
        * (
            delta
            / np.maximum(
                delta + gamma,
                1e-9,
            )
        )
        * rn_water_pos
    )

    priestley_taylor = (
        PRIESTLEY_TAYLOR_ALPHA
        * equilibrium
    )

                                                                      
                                                                    
                                                      
    numerator = (
        0.408
        * delta
        * rn_reference_pos
        + gamma
        * (900.0 / np.maximum(tmean + 273.0, 1e-9))
        * wind2
        * vpd
    )

    denominator = (
        delta
        + gamma
        * (1.0 + 0.34 * wind2)
    )

    pm_reference_proxy = (
        numerator
        / np.maximum(
            denominator,
            1e-9,
        )
    )

    pm_reference_proxy = np.maximum(
        pm_reference_proxy,
        0.0,
    )

                                                             
    aero_vpd_wind = (
        (1.0 + 0.536 * wind2)
        * vpd
    )

                                                                
    radiation_energy_proxy = (
        0.408
        * rn_water_pos
    )

    out["rh_derived_pct"] = rh
    out["vpd_kpa"] = vpd
    out["wind_2m_ms"] = wind2
    out["rn_water_proxy_mj_m2_day"] = rn_water
    out["rn_reference_proxy_mj_m2_day"] = rn_reference
    out["evap_equilibrium_proxy_mm_day"] = equilibrium
    out["evap_priestley_taylor_proxy_mm_day"] = priestley_taylor
    out["eto_pm_reference_proxy_mm_day"] = pm_reference_proxy
    out["radiation_energy_proxy_mm_day"] = radiation_energy_proxy
    out["aero_vpd_wind_index"] = aero_vpd_wind

    return out


def fit_combination_transparent(cal):
    features = [
        "evap_priestley_taylor_proxy_mm_day",
        "aero_vpd_wind_index",
    ]

    X = cal[features].to_numpy(float)
    y = cal["evap_mm"].to_numpy(float)

    valid = (
        np.isfinite(y)
        & np.isfinite(X).all(axis=1)
    )

    X = X[valid]
    y = y[valid]

    model = LinearRegression(
        fit_intercept=True,
        positive=True,
    )

    model.fit(X, y)

    params = {
        "mapping": "TRANSPARENT_COMBINATION",
        "intercept": float(model.intercept_),
        "coef_priestley_taylor": float(model.coef_[0]),
        "coef_aero_vpd_wind": float(model.coef_[1]),
    }

    return params


def apply_combination_transparent(df, params):
    pred = (
        float(params["intercept"])
        + float(params["coef_priestley_taylor"])
        * df["evap_priestley_taylor_proxy_mm_day"].to_numpy(float)
        + float(params["coef_aero_vpd_wind"])
        * df["aero_vpd_wind_index"].to_numpy(float)
    )

    return np.maximum(
        pred,
        0.0,
    )


def main():
    print("=" * 100)
    print("PASO 11A - DIAGNOSTICO DE EVAPORACION FUTURE-COMPATIBLE")
    print("=" * 100)
    print()
    print("Este paso NO modifica PHYSICAL_ONLY y NO recalibra todavía el balance del lago.")
    print("Compara índices físicamente interpretables reproducibles con variables disponibles en CMIP6.")
    print()

    if not REF_FILE.exists():
        raise FileNotFoundError(
            f"No existe la referencia del PASO 08:\n{REF_FILE}"
        )

    ref = pd.read_csv(
        REF_FILE
    )

    missing = [
        c
        for c in REQUIRED_COLUMNS
        if c not in ref.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}"
        )

    ref["date"] = pd.to_datetime(
        ref["date"],
        errors="raise",
    )

    ref = (
        ref.sort_values("date")
        .reset_index(drop=True)
    )

    for c in REQUIRED_COLUMNS:
        if c == "date":
            continue

        ref[c] = pd.to_numeric(
            ref[c],
            errors="coerce",
        )

    nan_counts = (
        ref[REQUIRED_COLUMNS[1:]]
        .isna()
        .sum()
    )

    if int(nan_counts.sum()) > 0:
        bad = nan_counts[
            nan_counts > 0
        ].to_dict()

        raise ValueError(
            f"Hay NaN en la referencia: {bad}"
        )

    cal_mask = ref["date"].between(
        CAL_START,
        CAL_END,
    )

    val_mask = ref["date"].between(
        VAL_START,
        VAL_END,
    )

    if not cal_mask.any() or not val_mask.any():
        raise ValueError(
            "No se pudo construir el split 1981-2004 / 2005-2014."
        )

                                                         
    pressure_constant_kpa = float(
        ref.loc[
            cal_mask,
            "pressure_kpa",
        ].median()
    )

    work = build_physical_indices(
        ref,
        pressure_constant_kpa,
    )

    cal = work.loc[
        cal_mask
    ].copy()

    val = work.loc[
        val_mask
    ].copy()

    print(f"Calibración: {CAL_START.date()} a {CAL_END.date()} | n={len(cal)}")
    print(f"Validación:  {VAL_START.date()} a {VAL_END.date()} | n={len(val)}")
    print(f"Presión fija histórica mediana: {pressure_constant_kpa:.4f} kPa")
    print()

    n_ref_negative = int(
        (work["evap_mm"] < 0).sum()
    )

    n_ref_zero = int(
        np.isclose(
            work["evap_mm"],
            0.0,
        ).sum()
    )

    print(f"QC evap_mm ERA5-Land: negativos={n_ref_negative}, ceros={n_ref_zero}")
    print()

    base_candidates = {
        "EQUILIBRIUM_OPEN_WATER_PROXY":
            "evap_equilibrium_proxy_mm_day",

        "PRIESTLEY_TAYLOR_OPEN_WATER_PROXY":
            "evap_priestley_taylor_proxy_mm_day",

        "PM_REFERENCE_DEMAND_PROXY":
            "eto_pm_reference_proxy_mm_day",

        "RADIATION_ENERGY_PROXY":
            "radiation_energy_proxy_mm_day",

        "AERO_VPD_WIND_INDEX":
            "aero_vpd_wind_index",
    }

    ranking_rows = []
    parameter_store = {}
    validation_predictions = pd.DataFrame({
        "date": val["date"].to_numpy(),
        "evap_reference_mm": val["evap_mm"].to_numpy(float),
    })

    full_predictions = pd.DataFrame({
        "date": work["date"].to_numpy(),
        "evap_reference_mm": work["evap_mm"].to_numpy(float),
        "era5_pet_benchmark_mm": work["pet_mm"].to_numpy(float),
        "rh_derived_pct": work["rh_derived_pct"].to_numpy(float),
        "vpd_kpa": work["vpd_kpa"].to_numpy(float),
        "wind_2m_ms": work["wind_2m_ms"].to_numpy(float),
        "rn_water_proxy_mj_m2_day": work["rn_water_proxy_mj_m2_day"].to_numpy(float),
        "rn_reference_proxy_mj_m2_day": work["rn_reference_proxy_mj_m2_day"].to_numpy(float),
    })

    for name, col in base_candidates.items():
        x_cal = cal[col].to_numpy(float)
        y_cal = cal["evap_mm"].to_numpy(float)

        mappings = {
            "RAW": {
                "mapping": "IDENTITY",
            },
            "SCALE": fit_scale_through_origin(
                x_cal,
                y_cal,
            ),
            "AFFINE": fit_affine_positive_slope(
                x_cal,
                y_cal,
            ),
        }

        parameter_store[name] = mappings

        for mapping_name, params in mappings.items():
            pred_val = apply_mapping(
                val[col].to_numpy(float),
                params,
            )

            pred_full = apply_mapping(
                work[col].to_numpy(float),
                params,
            )

            candidate_name = (
                f"{name}__{mapping_name}"
            )

            validation_predictions[
                candidate_name
            ] = pred_val

            full_predictions[
                candidate_name
            ] = pred_full

            eval_df = pd.DataFrame({
                "date": val["date"].to_numpy(),
                "obs": val["evap_mm"].to_numpy(float),
                "pred": pred_val,
            })

            met = aggregate_metrics(
                eval_df,
                obs_col="obs",
                pred_col="pred",
            )

            ranking_rows.append({
                "candidate": name,
                "mapping": mapping_name,
                "future_compatible": True,
                "formula_column": col,
                **met,
            })

                                                               
    combination_params = fit_combination_transparent(
        cal
    )

    pred_val_combination = apply_combination_transparent(
        val,
        combination_params,
    )

    pred_full_combination = apply_combination_transparent(
        work,
        combination_params,
    )

    validation_predictions[
        "TRANSPARENT_COMBINATION_PT_PLUS_AERO"
    ] = pred_val_combination

    full_predictions[
        "TRANSPARENT_COMBINATION_PT_PLUS_AERO"
    ] = pred_full_combination

    parameter_store[
        "TRANSPARENT_COMBINATION_PT_PLUS_AERO"
    ] = combination_params

    eval_comb = pd.DataFrame({
        "date": val["date"].to_numpy(),
        "obs": val["evap_mm"].to_numpy(float),
        "pred": pred_val_combination,
    })

    met_comb = aggregate_metrics(
        eval_comb,
        obs_col="obs",
        pred_col="pred",
    )

    ranking_rows.append({
        "candidate": "TRANSPARENT_COMBINATION_PT_PLUS_AERO",
        "mapping": "CALIBRATED_ON_1981_2004",
        "future_compatible": True,
        "formula_column": (
            "intercept + a*PriestleyTaylor + b*AeroVPDWind"
        ),
        **met_comb,
    })

                                               
    eval_pet = pd.DataFrame({
        "date": val["date"].to_numpy(),
        "obs": val["evap_mm"].to_numpy(float),
        "pred": np.maximum(
            val["pet_mm"].to_numpy(float),
            0.0,
        ),
    })

    pet_met = aggregate_metrics(
        eval_pet,
        obs_col="obs",
        pred_col="pred",
    )

    ranking_rows.append({
        "candidate": "ERA5_PET_HISTORICAL_BENCHMARK",
        "mapping": "RAW",
        "future_compatible": False,
        "formula_column": "pet_mm",
        **pet_met,
    })

    ranking = pd.DataFrame(
        ranking_rows
    )

                                                                 
                                                              
    ranking["abs_annual_mean_relative_bias_pct"] = (
        ranking[
            "annual_mean_relative_bias_pct"
        ].abs()
    )

    ranking = ranking.sort_values(
        by=[
            "future_compatible",
            "monthly_total_mae",
            "rolling_365d_mae_mm",
            "abs_annual_mean_relative_bias_pct",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    fc = ranking.loc[
        ranking["future_compatible"]
    ].copy()

    if fc.empty:
        raise RuntimeError(
            "No quedaron candidatos future-compatible."
        )

    best = fc.iloc[0].to_dict()

    ranking_path = (
        OUT_DIR
        / "11a_ranking_evaporacion_future_compatible.csv"
    )

    validation_path = (
        OUT_DIR
        / "11a_predicciones_validacion_2005_2014.csv"
    )

    full_path = (
        OUT_DIR
        / "11a_indices_evaporativos_1981_2014.csv"
    )

    params_path = (
        OUT_DIR
        / "11a_parametros_candidatos.json"
    )

    manifest_path = (
        OUT_DIR
        / "manifest_paso11a.json"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    validation_predictions.to_csv(
        validation_path,
        index=False,
    )

    full_predictions.to_csv(
        full_path,
        index=False,
    )

    with open(
        params_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "pressure_constant_kpa": pressure_constant_kpa,
                "physical_constants": {
                    "wind_measurement_height_m": WIND_MEASUREMENT_HEIGHT_M,
                    "water_albedo": WATER_ALBEDO,
                    "water_emissivity": WATER_EMISSIVITY,
                    "reference_albedo": REFERENCE_ALBEDO,
                    "reference_emissivity": REFERENCE_EMISSIVITY,
                    "priestley_taylor_alpha": PRIESTLEY_TAYLOR_ALPHA,
                    "sigma_mj_m2_day_k4": SIGMA_MJ,
                },
                "candidate_parameters": parameter_store,
                "best_by_validation_monthly_total_mae": {
                    "candidate": best["candidate"],
                    "mapping": best["mapping"],
                    "monthly_total_mae": best["monthly_total_mae"],
                    "rolling_365d_mae_mm": best["rolling_365d_mae_mm"],
                    "annual_mean_relative_bias_pct": best[
                        "annual_mean_relative_bias_pct"
                    ],
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    manifest = {
        "paso": "11A",
        "descripcion": (
            "Diagnóstico de formulaciones evaporativas reproducibles con "
            "variables disponibles también en NEX-GDDP-CMIP6."
        ),
        "python_version": sys.version,
        "reference_file": str(REF_FILE),
        "reference_sha256": sha256_file(REF_FILE),
        "calibration_period": {
            "start": str(CAL_START.date()),
            "end": str(CAL_END.date()),
            "n_days": int(len(cal)),
        },
        "validation_period": {
            "start": str(VAL_START.date()),
            "end": str(VAL_END.date()),
            "n_days": int(len(val)),
        },
        "reference_qc": {
            "evap_mm_negative_days": n_ref_negative,
            "evap_mm_zero_days": n_ref_zero,
        },
        "pressure_constant_kpa": pressure_constant_kpa,
        "future_compatible_inputs_required": [
            "tas/tmean",
            "tasmin",
            "tasmax",
            "relative humidity OR vapor pressure information",
            "sfcWind",
            "rsds",
            "rlds",
        ],
        "candidate_descriptions": {
            "EQUILIBRIUM_OPEN_WATER_PROXY": (
                "Equilibrium evaporation from positive open-water net-radiation proxy."
            ),
            "PRIESTLEY_TAYLOR_OPEN_WATER_PROXY": (
                "Priestley-Taylor alpha=1.26 applied to equilibrium open-water proxy."
            ),
            "PM_REFERENCE_DEMAND_PROXY": (
                "FAO Penman-Monteith atmospheric-demand form used only as a climatic "
                "index; it is NOT interpreted as actual lake evaporation."
            ),
            "RADIATION_ENERGY_PROXY": (
                "Available open-water radiative-energy proxy converted to equivalent water depth."
            ),
            "AERO_VPD_WIND_INDEX": (
                "Simple aerodynamic vapor-pressure-deficit and wind index; requires calibration."
            ),
            "TRANSPARENT_COMBINATION_PT_PLUS_AERO": (
                "Positive-coefficient linear combination of Priestley-Taylor energy "
                "term and aerodynamic VPD-wind term, calibrated on 1981-2004."
            ),
            "ERA5_PET_HISTORICAL_BENCHMARK": (
                "Historical benchmark only; not considered future-compatible."
            ),
        },
        "important_method_notes": [
            "No se modifica evap_mm.",
            "No se sustituye evap_mm dentro de PHYSICAL_ONLY en este paso.",
            "ET0/PM reference proxy is treated only as an atmospheric-demand index, not lake evaporation.",
            "Net radiation is approximated using incoming shortwave, incoming longwave, albedo, emissivity, and air temperature as a surface-temperature proxy.",
            "The air-temperature approximation for outgoing longwave is a structural limitation and must remain explicit.",
            "Pressure is fixed to the calibration-period median so future CMIP6 does not require a pressure predictor.",
            "All calibration mappings are fitted only on 1981-2004 and evaluated on untouched 2005-2014.",
            "The automatic ranking prioritizes monthly-total MAE because the lake balance integrates evaporation over time.",
            "The top-ranked candidate is diagnostic only; it is not accepted until PASO 11B recalibrates and revalidates the lake balance."
        ],
        "best_diagnostic_candidate": {
            "candidate": best["candidate"],
            "mapping": best["mapping"],
            "monthly_total_mae": best["monthly_total_mae"],
            "rolling_365d_mae_mm": best["rolling_365d_mae_mm"],
            "annual_mean_relative_bias_pct": best[
                "annual_mean_relative_bias_pct"
            ],
            "daily_pearson_r": best["daily_pearson_r"],
        },
        "outputs": {
            "ranking": str(ranking_path),
            "validation_predictions": str(validation_path),
            "full_indices": str(full_path),
            "candidate_parameters": str(params_path),
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

    print("=" * 100)
    print("RANKING FUTURE-COMPATIBLE EN VALIDACION 2005-2014")
    print("=" * 100)

    show_cols = [
        "candidate",
        "mapping",
        "daily_mae",
        "daily_rmse",
        "daily_bias",
        "daily_pearson_r",
        "monthly_total_mae",
        "rolling_365d_mae_mm",
        "annual_mean_relative_bias_pct",
        "trend_difference_mm_decade",
    ]

    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 30,
        "display.width", 240,
        "display.float_format", lambda x: f"{x:,.4f}",
    ):
        print(
            fc[show_cols]
            .head(15)
            .to_string(index=False)
        )

    print()
    print("=" * 100)
    print("MEJOR CANDIDATO DIAGNOSTICO")
    print("=" * 100)
    print(
        f"Candidato: {best['candidate']}"
    )
    print(
        f"Mapping:   {best['mapping']}"
    )
    print(
        f"MAE diario: {best['daily_mae']:.4f} mm/día"
    )
    print(
        f"r diario:   {best['daily_pearson_r']:.4f}"
    )
    print(
        f"MAE mensual acumulado: {best['monthly_total_mae']:.4f} mm/mes"
    )
    print(
        f"MAE rolling 365d:      {best['rolling_365d_mae_mm']:.4f} mm"
    )
    print(
        f"Sesgo anual medio:      {best['annual_mean_relative_bias_pct']:+.2f}%"
    )

    print()
    print("IMPORTANTE:")
    print("Este resultado NO reemplaza todavía evap_mm en el balance del lago.")
    print("El candidato sólo pasa a 11B para recalibrar y revalidar el balance completo.")
    print()
    print("Archivos creados:")
    print(ranking_path)
    print(validation_path)
    print(full_path)
    print(params_path)
    print(manifest_path)
    print()
    print("PASO 11A COMPLETADO.")


if __name__ == "__main__":
    main()
