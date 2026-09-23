__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels>=0.14"])
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_VERSION = "1.0.0"
RANDOM_STATE = 42

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_BLOCK_LENGTHS = [5, 10, 15]
MONTE_CARLO_SENSITIVITY_DRAWS = 2000
MONTE_CARLO_PARAMETER_FRACTION = 0.10

ROLLING_HORIZONS = [
    ("1obs", None),
    ("1m", 30),
    ("3m", 90),
    ("6m", 180),
    ("12m", 365),
    ("24m", 730),
]

ROOT = Path(__file__).resolve().parents[1]
BALANCE_CONFIG = ROOT / "config" / "balance_hidrico.json"
ML_CONFIG = ROOT / "config" / "ml_residual.json"

RESULT_DIR = ROOT / "outputs" / "validacion"
FIG_DIR = ROOT / "outputs" / "figuras"
CACHE_DIR = ROOT / "data" / "cache"

STATE_PATH = CACHE_DIR / "estado_05_validar_modelo.json"
REPORT_PATH = RESULT_DIR / "reporte_validacion_final.md"
CONFIG_OUT = ROOT / "config" / "validacion_final.json"

for d in [RESULT_DIR, FIG_DIR, CACHE_DIR, ROOT / "config"]:
    d.mkdir(parents=True, exist_ok=True)

if not BALANCE_CONFIG.exists():
    raise FileNotFoundError(
        "Falta config/balance_hidrico.json. Ejecuta primero 03_balance_hidrico.py"
    )

with open(BALANCE_CONFIG, "r", encoding="utf-8") as f:
    BALANCE_CFG = json.load(f)

ML_CFG = {}
if ML_CONFIG.exists():
    with open(ML_CONFIG, "r", encoding="utf-8") as f:
        ML_CFG = json.load(f)


def sha256_file(path, block_size=1024 * 1024):
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path, low_memory=False)


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_mae(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not valid.any():
        return np.nan
    return float(
        np.average(
            np.abs(y_true[valid] - y_pred[valid]),
            weights=weights[valid],
        )
    )


def weighted_rmse(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not valid.any():
        return np.nan
    return float(
        np.sqrt(
            np.average(
                (y_true[valid] - y_pred[valid]) ** 2,
                weights=weights[valid],
            )
        )
    )


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if (
        len(y_true) < 3
        or np.std(y_true, ddof=1) <= 0
        or np.std(y_pred, ddof=1) <= 0
    ):
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def kge_2012(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    obs = y_true[valid]
    sim = y_pred[valid]

    if len(obs) < 3:
        return {
            "KGE2012": np.nan,
            "KGE_r": np.nan,
            "KGE_beta": np.nan,
            "KGE_gamma": np.nan,
        }

    mean_obs = float(np.mean(obs))
    mean_sim = float(np.mean(sim))
    sd_obs = float(np.std(obs, ddof=1))
    sd_sim = float(np.std(sim, ddof=1))
    r = safe_pearson(obs, sim)

    if (
        not np.isfinite(r)
        or abs(mean_obs) < 1e-12
        or abs(mean_sim) < 1e-12
        or sd_obs <= 0
        or sd_sim <= 0
    ):
        return {
            "KGE2012": np.nan,
            "KGE_r": r,
            "KGE_beta": np.nan,
            "KGE_gamma": np.nan,
        }

    beta = mean_sim / mean_obs
    cv_obs = sd_obs / mean_obs
    cv_sim = sd_sim / mean_sim

    if abs(cv_obs) < 1e-12:
        gamma = np.nan
        kge = np.nan
    else:
        gamma = cv_sim / cv_obs
        kge = 1.0 - np.sqrt(
            (r - 1.0) ** 2
            + (beta - 1.0) ** 2
            + (gamma - 1.0) ** 2
        )

    return {
        "KGE2012": float(kge) if np.isfinite(kge) else np.nan,
        "KGE_r": float(r) if np.isfinite(r) else np.nan,
        "KGE_beta": float(beta) if np.isfinite(beta) else np.nan,
        "KGE_gamma": float(gamma) if np.isfinite(gamma) else np.nan,
    }


def full_metrics(model_name, y_true, y_pred, weights=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if weights is None:
        weights = np.ones(len(y_true), dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(weights)
        & (weights > 0)
    )

    obs = y_true[valid]
    sim = y_pred[valid]
    w = weights[valid]

    if len(obs) == 0:
        return {"model": model_name, "n": 0}

    mae = float(mean_absolute_error(obs, sim))
    rmse = float(np.sqrt(mean_squared_error(obs, sim)))
    wmae = weighted_mae(obs, sim, w)
    wrmse = weighted_rmse(obs, sim, w)

    residual = sim - obs
    bias = float(np.mean(residual))
    wbias = weighted_mean(residual, w)

    r2 = float(r2_score(obs, sim)) if len(obs) >= 2 else np.nan
    r = safe_pearson(obs, sim)

    obs_range = float(np.max(obs) - np.min(obs))
    obs_sd = float(np.std(obs, ddof=1)) if len(obs) > 1 else np.nan
    sim_sd = float(np.std(sim, ddof=1)) if len(sim) > 1 else np.nan

    crmse = float(
        np.sqrt(
            np.mean(
                (
                    (sim - np.mean(sim))
                    - (obs - np.mean(obs))
                ) ** 2
            )
        )
    )

    variability_ratio = (
        sim_sd / obs_sd
        if np.isfinite(obs_sd) and obs_sd > 0
        else np.nan
    )

    kge = kge_2012(obs, sim)

    out = {
        "model": model_name,
        "n": int(len(obs)),
        "MAE_m": mae,
        "weighted_MAE_m": wmae,
        "RMSE_m": rmse,
        "weighted_RMSE_m": wrmse,
        "bias_m": bias,
        "weighted_bias_m": wbias,
        "R2": r2,
        "NSE": r2,
        "Pearson_r": r,
        "centered_RMSE_m": crmse,
        "variability_ratio": variability_ratio,
        "nRMSE_range": (
            rmse / obs_range
            if obs_range > 0
            else np.nan
        ),
        "nRMSE_sd": (
            rmse / obs_sd
            if np.isfinite(obs_sd) and obs_sd > 0
            else np.nan
        ),
        **kge,
    }

    return out


def trajectory_from_delta(start_level, delta):
    current = float(start_level)
    out = []

    for value in np.asarray(delta, dtype=float):
        current += float(value)
        out.append(current)

    return np.asarray(out, dtype=float)


def linear_slope_per_year(dates, values):
    dates = pd.to_datetime(pd.Series(dates))
    values = np.asarray(values, dtype=float)

    valid = dates.notna().to_numpy() & np.isfinite(values)

    if valid.sum() < 3:
        return np.nan

    d = dates.loc[valid].reset_index(drop=True)
    v = values[valid]

    x_days = (
        d - d.iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float) / 86400.0

    slope_day = np.polyfit(x_days, v, 1)[0]

    return float(slope_day * 365.2425)


def moving_block_indices(n, block_length, rng):
    if n <= 0:
        return np.array([], dtype=int)

    block_length = int(max(1, min(block_length, n)))
    n_blocks = int(math.ceil(n / block_length))
    max_start = n - block_length

    if max_start <= 0:
        return np.arange(n, dtype=int)

    starts = rng.integers(
        0,
        max_start + 1,
        size=n_blocks,
    )

    indices = np.concatenate(
        [
            np.arange(
                start,
                start + block_length,
                dtype=int,
            )
            for start in starts
        ]
    )

    return indices[:n]


def bootstrap_skill(
    y_true,
    model_pred,
    baseline_pred,
    weights,
    block_length,
    n_resamples,
    seed,
):
    y_true = np.asarray(y_true, dtype=float)
    model_pred = np.asarray(model_pred, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n = len(y_true)
    rng = np.random.default_rng(seed)

    skill_values = []
    mae_diff_values = []

    for _ in range(n_resamples):
        idx = moving_block_indices(
            n,
            block_length,
            rng,
        )

        mae_model = weighted_mae(
            y_true[idx],
            model_pred[idx],
            weights[idx],
        )

        mae_base = weighted_mae(
            y_true[idx],
            baseline_pred[idx],
            weights[idx],
        )

        if (
            np.isfinite(mae_model)
            and np.isfinite(mae_base)
            and mae_base > 0
        ):
            skill_values.append(
                1.0 - mae_model / mae_base
            )
            mae_diff_values.append(
                mae_base - mae_model
            )

    skill_values = np.asarray(skill_values, dtype=float)
    mae_diff_values = np.asarray(mae_diff_values, dtype=float)

    if len(skill_values) == 0:
        return {
            "block_length": block_length,
            "n_resamples_valid": 0,
            "skill_median": np.nan,
            "skill_ci_low": np.nan,
            "skill_ci_high": np.nan,
            "mae_improvement_median_m": np.nan,
            "mae_improvement_ci_low_m": np.nan,
            "mae_improvement_ci_high_m": np.nan,
            "probability_skill_positive": np.nan,
        }

    return {
        "block_length": int(block_length),
        "n_resamples_valid": int(len(skill_values)),
        "skill_median": float(np.median(skill_values)),
        "skill_ci_low": float(np.quantile(skill_values, 0.025)),
        "skill_ci_high": float(np.quantile(skill_values, 0.975)),
        "mae_improvement_median_m": float(np.median(mae_diff_values)),
        "mae_improvement_ci_low_m": float(
            np.quantile(mae_diff_values, 0.025)
        ),
        "mae_improvement_ci_high_m": float(
            np.quantile(mae_diff_values, 0.975)
        ),
        "probability_skill_positive": float(
            np.mean(skill_values > 0)
        ),
    }


def choose_horizon_target(
    test_df,
    origin_pos,
    horizon_name,
    horizon_days,
):
    if origin_pos >= len(test_df):
        return None

    origin_date = pd.Timestamp(
        test_df.iloc[origin_pos]["date_prev"]
    )

    if horizon_name == "1obs":
        return origin_pos

    candidates = test_df.iloc[origin_pos:].copy()

    lead_days = (
        pd.to_datetime(candidates["date_obs"])
        - origin_date
    ).dt.total_seconds() / 86400.0

    if len(lead_days) == 0:
        return None

    target_local = int(
        np.argmin(
            np.abs(
                lead_days.to_numpy(dtype=float)
                - float(horizon_days)
            )
        )
    )

    target_pos = origin_pos + target_local
    actual_lead = float(
        (
            pd.Timestamp(
                test_df.iloc[target_pos]["date_obs"]
            )
            - origin_date
        ).total_seconds()
        / 86400.0
    )

    tolerance = max(
        15.0,
        float(horizon_days) * 0.05,
    )

    if abs(actual_lead - float(horizon_days)) > tolerance:
        return None

    return target_pos


def rolling_horizon_predictions(test_df):
    rows = []

    delta_model = test_df[
        "delta_balance_calibrated_final_m"
    ].to_numpy(dtype=float)

    for horizon_name, horizon_days in ROLLING_HORIZONS:
        for origin_pos in range(len(test_df)):
            target_pos = choose_horizon_target(
                test_df,
                origin_pos,
                horizon_name,
                horizon_days,
            )

            if target_pos is None:
                continue

            origin_level = float(
                test_df.iloc[origin_pos]["level_prev_m"]
            )
            origin_date = pd.Timestamp(
                test_df.iloc[origin_pos]["date_prev"]
            )

            model_level = (
                origin_level
                + float(
                    np.sum(
                        delta_model[
                            origin_pos:target_pos + 1
                        ]
                    )
                )
            )

            observed_level = float(
                test_df.iloc[target_pos]["level_obs_m"]
            )

            target_date = pd.Timestamp(
                test_df.iloc[target_pos]["date_obs"]
            )

            actual_lead_days = float(
                (
                    target_date - origin_date
                ).total_seconds()
                / 86400.0
            )

            target_weight = float(
                test_df.iloc[target_pos][
                    "interval_weight"
                ]
            )

            rows.append(
                {
                    "horizon": horizon_name,
                    "requested_lead_days": (
                        np.nan
                        if horizon_days is None
                        else float(horizon_days)
                    ),
                    "origin_position": int(origin_pos),
                    "target_position": int(target_pos),
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "actual_lead_days": actual_lead_days,
                    "origin_level_m": origin_level,
                    "observed_level_m": observed_level,
                    "physical_level_m": model_level,
                    "frozen_level_m": origin_level,
                    "interval_weight": target_weight,
                }
            )

    return pd.DataFrame(rows)


def classify_season(month):
    month = int(month)

    if 5 <= month <= 10:
        return "lluviosa_May-Oct"

    return "seca_Nov-Apr"


def recompute_delta(
    table,
    precip_scale,
    evap_scale,
    qin_scale,
    c_day,
):
    return (
        float(precip_scale)
        * table["p_consensus_m"].to_numpy(dtype=float)
        - float(evap_scale)
        * table["evap_m"].to_numpy(dtype=float)
        + float(qin_scale)
        * table["q_in_final_equiv_m"].to_numpy(dtype=float)
        + float(c_day)
        * table["duration_days"].to_numpy(dtype=float)
    )


INTERVALS_PATH = Path(
    BALANCE_CFG["outputs"]["intervals"]
)
FINAL_PARAMS_PATH = Path(
    BALANCE_CFG["outputs"]["final_parameters"]
)

for path in [INTERVALS_PATH, FINAL_PARAMS_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

with open(FINAL_PARAMS_PATH, "r", encoding="utf-8") as f:
    FINAL_PARAMS = json.load(f)

fingerprint = {
    "script_version": SCRIPT_VERSION,
    "intervals_sha256": sha256_file(INTERVALS_PATH),
    "final_params_sha256": sha256_file(FINAL_PARAMS_PATH),
    "balance_config_sha256": sha256_file(BALANCE_CONFIG),
    "ml_config_sha256": (
        sha256_file(ML_CONFIG)
        if ML_CONFIG.exists()
        else None
    ),
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_block_lengths": BOOTSTRAP_BLOCK_LENGTHS,
    "monte_carlo_draws": MONTE_CARLO_SENSITIVITY_DRAWS,
}

expected_outputs = [
    RESULT_DIR / "metricas_globales_test.csv",
    RESULT_DIR / "metricas_por_horizonte.csv",
    RESULT_DIR / "bootstrap_skill.csv",
    RESULT_DIR / "diagnostico_residuos.csv",
    RESULT_DIR / "sensibilidad_parametros.csv",
    RESULT_DIR / "resumen_validacion.csv",
    REPORT_PATH,
    CONFIG_OUT,
]

if (
    STATE_PATH.exists()
    and all(path.exists() for path in expected_outputs)
):
    try:
        old_state = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        old_state = None

    if old_state == fingerprint:
        print("=" * 72)
        print("05_VALIDAR_MODELO: SIN CAMBIOS")
        print("=" * 72)
        print(
            "Los productos fisicos y la configuracion no cambiaron."
        )
        print(
            "No se repitieron bootstrap ni diagnosticos."
        )
        print(RESULT_DIR)
        sys.exit(0)

intervals = read_table(INTERVALS_PATH).copy()

for col in [
    "date_prev",
    "date_obs",
]:
    intervals[col] = pd.to_datetime(
        intervals[col],
        errors="coerce",
    )

numeric_cols = [
    "duration_days",
    "level_prev_m",
    "level_obs_m",
    "delta_h_obs_m",
    "p_consensus_m",
    "evap_m",
    "q_in_final_equiv_m",
    "delta_balance_calibrated_final_m",
    "interval_weight",
]

for col in numeric_cols:
    intervals[col] = pd.to_numeric(
        intervals[col],
        errors="coerce",
    )

test = (
    intervals.loc[
        intervals["partition"].eq("test")
    ]
    .copy()
    .sort_values("date_obs")
    .reset_index(drop=True)
)

train_cal = (
    intervals.loc[
        intervals["partition"].isin(
            ["train", "calibration"]
        )
    ]
    .copy()
    .sort_values("date_obs")
    .reset_index(drop=True)
)

if len(test) < 40:
    raise ValueError(
        "El TEST tiene muy pocas observaciones para la validacion final."
    )

required_test = [
    "date_prev",
    "date_obs",
    "level_prev_m",
    "level_obs_m",
    "delta_h_obs_m",
    "p_consensus_m",
    "evap_m",
    "q_in_final_equiv_m",
    "delta_balance_calibrated_final_m",
    "interval_weight",
]

if test[required_test].isna().any().any():
    bad = test[
        required_test
    ].columns[
        test[
            required_test
        ].isna().any()
    ].tolist()

    raise ValueError(
        "Hay NaN en variables necesarias del TEST: "
        + str(bad)
    )

selected_model_from_04 = ML_CFG.get(
    "selected_model",
    "UNKNOWN",
)

test_start_level = float(
    test.iloc[0]["level_prev_m"]
)

observed_level = test[
    "level_obs_m"
].to_numpy(dtype=float)

observed_delta = test[
    "delta_h_obs_m"
].to_numpy(dtype=float)

physical_delta = test[
    "delta_balance_calibrated_final_m"
].to_numpy(dtype=float)

weights = test[
    "interval_weight"
].to_numpy(dtype=float)

physical_level = trajectory_from_delta(
    test_start_level,
    physical_delta,
)

frozen_level = np.full(
    len(test),
    test_start_level,
    dtype=float,
)

one_step_physical_level = (
    test["level_prev_m"].to_numpy(dtype=float)
    + physical_delta
)

one_step_persistence_level = test[
    "level_prev_m"
].to_numpy(dtype=float)

global_rows = []

global_rows.append(
    {
        "evaluation": "open_loop_test",
        **full_metrics(
            "PhysicalBalance",
            observed_level,
            physical_level,
            weights,
        ),
    }
)

global_rows.append(
    {
        "evaluation": "open_loop_test",
        **full_metrics(
            "Frozen_Last_Level",
            observed_level,
            frozen_level,
            weights,
        ),
    }
)

global_rows.append(
    {
        "evaluation": "one_step_test",
        **full_metrics(
            "PhysicalBalance_OneStep",
            observed_level,
            one_step_physical_level,
            weights,
        ),
    }
)

global_rows.append(
    {
        "evaluation": "one_step_test",
        **full_metrics(
            "Persistence_1obs",
            observed_level,
            one_step_persistence_level,
            weights,
        ),
    }
)

global_df = pd.DataFrame(global_rows)

open_loop_base_wmae = float(
    global_df.loc[
        (
            global_df["evaluation"].eq(
                "open_loop_test"
            )
        )
        & (
            global_df["model"].eq(
                "Frozen_Last_Level"
            )
        ),
        "weighted_MAE_m",
    ].iloc[0]
)

one_step_base_wmae = float(
    global_df.loc[
        (
            global_df["evaluation"].eq(
                "one_step_test"
            )
        )
        & (
            global_df["model"].eq(
                "Persistence_1obs"
            )
        ),
        "weighted_MAE_m",
    ].iloc[0]
)

global_df[
    "weighted_MAE_skill_vs_compatible_baseline"
] = np.nan

mask = (
    global_df["evaluation"].eq(
        "open_loop_test"
    )
)
global_df.loc[
    mask,
    "weighted_MAE_skill_vs_compatible_baseline",
] = (
    1.0
    - global_df.loc[
        mask,
        "weighted_MAE_m",
    ]
    / open_loop_base_wmae
)

mask = (
    global_df["evaluation"].eq(
        "one_step_test"
    )
)
global_df.loc[
    mask,
    "weighted_MAE_skill_vs_compatible_baseline",
] = (
    1.0
    - global_df.loc[
        mask,
        "weighted_MAE_m",
    ]
    / one_step_base_wmae
)

global_df.to_csv(
    RESULT_DIR / "metricas_globales_test.csv",
    index=False,
)

predictions = pd.DataFrame(
    {
        "date": test["date_obs"],
        "observed_level_m": observed_level,
        "physical_open_loop_level_m": physical_level,
        "frozen_level_m": frozen_level,
        "physical_one_step_level_m": one_step_physical_level,
        "persistence_one_step_level_m": one_step_persistence_level,
        "observed_delta_m": observed_delta,
        "physical_delta_m": physical_delta,
        "level_residual_m": (
            physical_level - observed_level
        ),
        "delta_residual_m": (
            physical_delta - observed_delta
        ),
        "interval_weight": weights,
    }
)

predictions.to_csv(
    RESULT_DIR / "predicciones_validacion_test.csv",
    index=False,
)

rolling = rolling_horizon_predictions(test)
rolling.to_csv(
    RESULT_DIR / "predicciones_rolling_horizontes.csv",
    index=False,
)

horizon_rows = []

for horizon_name, _ in ROLLING_HORIZONS:
    sub = rolling.loc[
        rolling["horizon"].eq(
            horizon_name
        )
    ].copy()

    if len(sub) == 0:
        continue

    model_metrics = full_metrics(
        "PhysicalBalance",
        sub["observed_level_m"],
        sub["physical_level_m"],
        sub["interval_weight"],
    )

    base_metrics = full_metrics(
        "Frozen_Last_Level",
        sub["observed_level_m"],
        sub["frozen_level_m"],
        sub["interval_weight"],
    )

    skill = (
        1.0
        - model_metrics["weighted_MAE_m"]
        / base_metrics["weighted_MAE_m"]
        if base_metrics["weighted_MAE_m"] > 0
        else np.nan
    )

    horizon_rows.append(
        {
            "horizon": horizon_name,
            "n": int(len(sub)),
            "mean_actual_lead_days": float(
                sub["actual_lead_days"].mean()
            ),
            "physical_MAE_m": model_metrics["MAE_m"],
            "physical_weighted_MAE_m": model_metrics["weighted_MAE_m"],
            "physical_RMSE_m": model_metrics["RMSE_m"],
            "physical_bias_m": model_metrics["bias_m"],
            "physical_R2": model_metrics["R2"],
            "physical_Pearson_r": model_metrics["Pearson_r"],
            "frozen_MAE_m": base_metrics["MAE_m"],
            "frozen_weighted_MAE_m": base_metrics["weighted_MAE_m"],
            "frozen_RMSE_m": base_metrics["RMSE_m"],
            "weighted_MAE_skill_vs_frozen": skill,
        }
    )

horizon_df = pd.DataFrame(horizon_rows)

horizon_df.to_csv(
    RESULT_DIR / "metricas_por_horizonte.csv",
    index=False,
)

year_rows = []

pred_year = predictions.copy()
pred_year["year"] = pd.to_datetime(
    pred_year["date"]
).dt.year

for year, sub in pred_year.groupby("year"):
    metric_model = full_metrics(
        "PhysicalBalance",
        sub["observed_level_m"],
        sub["physical_open_loop_level_m"],
        sub["interval_weight"],
    )

    metric_base = full_metrics(
        "Frozen_Last_Level",
        sub["observed_level_m"],
        sub["frozen_level_m"],
        sub["interval_weight"],
    )

    year_rows.append(
        {
            "year": int(year),
            "n": int(len(sub)),
            "physical_MAE_m": metric_model["MAE_m"],
            "physical_weighted_MAE_m": metric_model["weighted_MAE_m"],
            "physical_bias_m": metric_model["bias_m"],
            "physical_R2": metric_model["R2"],
            "physical_Pearson_r": metric_model["Pearson_r"],
            "frozen_weighted_MAE_m": metric_base["weighted_MAE_m"],
            "weighted_MAE_skill_vs_frozen": (
                1.0
                - metric_model["weighted_MAE_m"]
                / metric_base["weighted_MAE_m"]
                if metric_base["weighted_MAE_m"] > 0
                else np.nan
            ),
        }
    )

year_df = pd.DataFrame(year_rows)
year_df.to_csv(
    RESULT_DIR / "metricas_por_ano.csv",
    index=False,
)

season_table = test.copy()
season_table["physical_level_m"] = physical_level
season_table["frozen_level_m"] = frozen_level
season_table["season"] = (
    season_table["date_obs"]
    .dt.month
    .map(classify_season)
)

season_rows = []

for season, sub in season_table.groupby("season"):
    model_metric = full_metrics(
        "PhysicalBalance",
        sub["level_obs_m"],
        sub["physical_level_m"],
        sub["interval_weight"],
    )

    base_metric = full_metrics(
        "Frozen_Last_Level",
        sub["level_obs_m"],
        sub["frozen_level_m"],
        sub["interval_weight"],
    )

    delta_metric = full_metrics(
        "PhysicalBalance_Delta",
        sub["delta_h_obs_m"],
        sub["delta_balance_calibrated_final_m"],
        sub["interval_weight"],
    )

    season_rows.append(
        {
            "season": season,
            "n": int(len(sub)),
            "level_weighted_MAE_m": model_metric["weighted_MAE_m"],
            "level_bias_m": model_metric["bias_m"],
            "level_R2": model_metric["R2"],
            "delta_weighted_MAE_m": delta_metric["weighted_MAE_m"],
            "delta_bias_m": delta_metric["bias_m"],
            "frozen_weighted_MAE_m": base_metric["weighted_MAE_m"],
            "level_skill_vs_frozen": (
                1.0
                - model_metric["weighted_MAE_m"]
                / base_metric["weighted_MAE_m"]
                if base_metric["weighted_MAE_m"] > 0
                else np.nan
            ),
        }
    )

season_df = pd.DataFrame(season_rows)
season_df.to_csv(
    RESULT_DIR / "metricas_por_estacion.csv",
    index=False,
)

p_q33, p_q67 = np.nanquantile(
    train_cal["p_consensus_m"],
    [1 / 3, 2 / 3],
)

q_q33, q_q67 = np.nanquantile(
    train_cal["q_in_final_equiv_m"],
    [1 / 3, 2 / 3],
)


def tercile_label(value, q33, q67):
    if value <= q33:
        return "low"
    if value >= q67:
        return "high"
    return "normal"


regime_table = test.copy()
regime_table["physical_level_m"] = physical_level
regime_table["frozen_level_m"] = frozen_level

regime_table["precip_regime"] = [
    tercile_label(v, p_q33, p_q67)
    for v in regime_table["p_consensus_m"]
]

regime_table["qin_regime"] = [
    tercile_label(v, q_q33, q_q67)
    for v in regime_table["q_in_final_equiv_m"]
]

regime_rows = []

for regime_type in [
    "precip_regime",
    "qin_regime",
]:
    for regime, sub in regime_table.groupby(
        regime_type
    ):
        model_metric = full_metrics(
            "PhysicalBalance",
            sub["level_obs_m"],
            sub["physical_level_m"],
            sub["interval_weight"],
        )

        delta_metric = full_metrics(
            "PhysicalBalance_Delta",
            sub["delta_h_obs_m"],
            sub["delta_balance_calibrated_final_m"],
            sub["interval_weight"],
        )

        regime_rows.append(
            {
                "regime_type": regime_type,
                "regime": regime,
                "n": int(len(sub)),
                "level_weighted_MAE_m": model_metric["weighted_MAE_m"],
                "level_bias_m": model_metric["bias_m"],
                "delta_weighted_MAE_m": delta_metric["weighted_MAE_m"],
                "delta_bias_m": delta_metric["bias_m"],
            }
        )

regime_df = pd.DataFrame(regime_rows)

regime_df.to_csv(
    RESULT_DIR / "metricas_por_regimen.csv",
    index=False,
)

bootstrap_rows = []

for i, block_length in enumerate(
    BOOTSTRAP_BLOCK_LENGTHS
):
    result = bootstrap_skill(
        y_true=observed_level,
        model_pred=physical_level,
        baseline_pred=frozen_level,
        weights=weights,
        block_length=block_length,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=RANDOM_STATE + i,
    )

    bootstrap_rows.append(result)

bootstrap_df = pd.DataFrame(bootstrap_rows)

bootstrap_df.to_csv(
    RESULT_DIR / "bootstrap_skill.csv",
    index=False,
)

level_residual = physical_level - observed_level
delta_residual = physical_delta - observed_delta

diagnostic_rows = []

for residual_name, residual_values in [
    ("level_open_loop_residual", level_residual),
    ("delta_residual", delta_residual),
]:
    residual_values = np.asarray(
        residual_values,
        dtype=float,
    )

    diagnostic_rows.append(
        {
            "diagnostic": residual_name,
            "lag": 0,
            "statistic": "durbin_watson",
            "value": float(
                durbin_watson(
                    residual_values
                )
            ),
            "pvalue": np.nan,
        }
    )

    valid_lags = [
        lag
        for lag in [1, 3, 6, 12]
        if lag < len(residual_values) / 2
    ]

    if valid_lags:
        lb = acorr_ljungbox(
            residual_values,
            lags=valid_lags,
            return_df=True,
        )

        for lag in valid_lags:
            diagnostic_rows.append(
                {
                    "diagnostic": residual_name,
                    "lag": int(lag),
                    "statistic": "ljung_box",
                    "value": float(
                        lb.loc[
                            lag,
                            "lb_stat",
                        ]
                    ),
                    "pvalue": float(
                        lb.loc[
                            lag,
                            "lb_pvalue",
                        ]
                    ),
                }
            )

diagnostic_df = pd.DataFrame(
    diagnostic_rows
)

diagnostic_df.to_csv(
    RESULT_DIR / "diagnostico_residuos.csv",
    index=False,
)

observed_slope = linear_slope_per_year(
    test["date_obs"],
    observed_level,
)

physical_slope = linear_slope_per_year(
    test["date_obs"],
    physical_level,
)

frozen_slope = linear_slope_per_year(
    test["date_obs"],
    frozen_level,
)

trend_df = pd.DataFrame(
    [
        {
            "series": "Observed",
            "slope_m_per_year": observed_slope,
        },
        {
            "series": "PhysicalBalance",
            "slope_m_per_year": physical_slope,
        },
        {
            "series": "Frozen_Last_Level",
            "slope_m_per_year": frozen_slope,
        },
    ]
)

trend_df["absolute_gap_vs_observed_m_per_year"] = (
    np.abs(
        trend_df["slope_m_per_year"]
        - observed_slope
    )
)

trend_df.to_csv(
    RESULT_DIR / "tendencias_test.csv",
    index=False,
)

precip_scale = float(
    FINAL_PARAMS["precip_scale"]
)
evap_scale = float(
    FINAL_PARAMS["evap_scale"]
)
qin_scale = float(
    FINAL_PARAMS["qin_scale"]
)
c_day = float(
    FINAL_PARAMS["net_unresolved_m_per_day"]
)

base_params = {
    "precip_scale": precip_scale,
    "evap_scale": evap_scale,
    "qin_scale": qin_scale,
    "net_unresolved_m_per_day": c_day,
}

sensitivity_rows = []

for parameter_name in base_params:
    for factor in [
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
    ]:
        params = dict(base_params)
        params[parameter_name] = (
            params[parameter_name] * factor
        )

        delta = recompute_delta(
            test,
            params["precip_scale"],
            params["evap_scale"],
            params["qin_scale"],
            params["net_unresolved_m_per_day"],
        )

        level = trajectory_from_delta(
            test_start_level,
            delta,
        )

        model_metric = full_metrics(
            "PhysicalBalance",
            observed_level,
            level,
            weights,
        )

        skill = (
            1.0
            - model_metric["weighted_MAE_m"]
            / open_loop_base_wmae
        )

        sensitivity_rows.append(
            {
                "parameter": parameter_name,
                "factor": factor,
                "parameter_value": params[
                    parameter_name
                ],
                "weighted_MAE_m": model_metric[
                    "weighted_MAE_m"
                ],
                "weighted_bias_m": model_metric[
                    "weighted_bias_m"
                ],
                "R2": model_metric["R2"],
                "skill_vs_frozen": skill,
                "final_level_m": float(
                    level[-1]
                ),
            }
        )

sensitivity_df = pd.DataFrame(
    sensitivity_rows
)

sensitivity_df.to_csv(
    RESULT_DIR / "sensibilidad_parametros.csv",
    index=False,
)

rng = np.random.default_rng(
    RANDOM_STATE + 1000
)

mc_rows = []

for draw in range(
    MONTE_CARLO_SENSITIVITY_DRAWS
):
    factors = rng.uniform(
        1.0 - MONTE_CARLO_PARAMETER_FRACTION,
        1.0 + MONTE_CARLO_PARAMETER_FRACTION,
        size=4,
    )

    params = {
        "precip_scale": (
            precip_scale * factors[0]
        ),
        "evap_scale": (
            evap_scale * factors[1]
        ),
        "qin_scale": (
            qin_scale * factors[2]
        ),
        "net_unresolved_m_per_day": (
            c_day * factors[3]
        ),
    }

    delta = recompute_delta(
        test,
        params["precip_scale"],
        params["evap_scale"],
        params["qin_scale"],
        params["net_unresolved_m_per_day"],
    )

    level = trajectory_from_delta(
        test_start_level,
        delta,
    )

    wmae = weighted_mae(
        observed_level,
        level,
        weights,
    )

    mc_rows.append(
        {
            "draw": int(draw),
            **params,
            "weighted_MAE_m": wmae,
            "skill_vs_frozen": (
                1.0
                - wmae
                / open_loop_base_wmae
            ),
            "final_level_m": float(
                level[-1]
            ),
        }
    )

mc_df = pd.DataFrame(mc_rows)

mc_df.to_csv(
    RESULT_DIR / "sensibilidad_monte_carlo.csv",
    index=False,
)

mc_summary = pd.DataFrame(
    [
        {
            "metric": "weighted_MAE_m",
            "q025": float(
                mc_df[
                    "weighted_MAE_m"
                ].quantile(0.025)
            ),
            "median": float(
                mc_df[
                    "weighted_MAE_m"
                ].median()
            ),
            "q975": float(
                mc_df[
                    "weighted_MAE_m"
                ].quantile(0.975)
            ),
        },
        {
            "metric": "skill_vs_frozen",
            "q025": float(
                mc_df[
                    "skill_vs_frozen"
                ].quantile(0.025)
            ),
            "median": float(
                mc_df[
                    "skill_vs_frozen"
                ].median()
            ),
            "q975": float(
                mc_df[
                    "skill_vs_frozen"
                ].quantile(0.975)
            ),
        },
        {
            "metric": "final_level_m",
            "q025": float(
                mc_df[
                    "final_level_m"
                ].quantile(0.025)
            ),
            "median": float(
                mc_df[
                    "final_level_m"
                ].median()
            ),
            "q975": float(
                mc_df[
                    "final_level_m"
                ].quantile(0.975)
            ),
        },
    ]
)

mc_summary.to_csv(
    RESULT_DIR / "sensibilidad_monte_carlo_resumen.csv",
    index=False,
)

physical_global = global_df.loc[
    (
        global_df["evaluation"].eq(
            "open_loop_test"
        )
    )
    & (
        global_df["model"].eq(
            "PhysicalBalance"
        )
    )
].iloc[0]

physical_one_step = global_df.loc[
    (
        global_df["evaluation"].eq(
            "one_step_test"
        )
    )
    & (
        global_df["model"].eq(
            "PhysicalBalance_OneStep"
        )
    )
].iloc[0]

zero_change_delta_wmae = weighted_mae(
    observed_delta,
    np.zeros(len(observed_delta)),
    weights,
)

physical_delta_wmae = weighted_mae(
    observed_delta,
    physical_delta,
    weights,
)

delta_skill_vs_zero = (
    1.0
    - physical_delta_wmae
    / zero_change_delta_wmae
    if zero_change_delta_wmae > 0
    else np.nan
)

bootstrap_reference = bootstrap_df.loc[
    bootstrap_df["block_length"].eq(10)
]

if len(bootstrap_reference):
    bootstrap_ci_low = float(
        bootstrap_reference[
            "skill_ci_low"
        ].iloc[0]
    )
else:
    bootstrap_ci_low = np.nan

delta_ljung = diagnostic_df.loc[
    (
        diagnostic_df[
            "diagnostic"
        ].eq(
            "delta_residual"
        )
    )
    & (
        diagnostic_df[
            "statistic"
        ].eq(
            "ljung_box"
        )
    )
]

if len(delta_ljung):
    lag_choice = int(
        delta_ljung[
            "lag"
        ].max()
    )

    delta_ljung_p = float(
        delta_ljung.loc[
            delta_ljung[
                "lag"
            ].eq(
                lag_choice
            ),
            "pvalue",
        ].iloc[0]
    )
else:
    lag_choice = np.nan
    delta_ljung_p = np.nan

global_skill = float(
    physical_global[
        "weighted_MAE_skill_vs_compatible_baseline"
    ]
)

nse_value = float(
    physical_global["NSE"]
)

trend_same_sign = bool(
    np.sign(observed_slope)
    == np.sign(physical_slope)
)

criteria = [
    {
        "criterion": "open_loop_skill_vs_frozen_positive",
        "value": global_skill,
        "pass": bool(
            np.isfinite(global_skill)
            and global_skill > 0
        ),
        "interpretation": "El balance debe superar mantener fijo el nivel inicial del TEST.",
    },
    {
        "criterion": "open_loop_NSE_positive",
        "value": nse_value,
        "pass": bool(
            np.isfinite(nse_value)
            and nse_value > 0
        ),
        "interpretation": "NSE positivo indica mejora frente a usar la media observada como referencia; no se usa de forma aislada.",
    },
    {
        "criterion": "bootstrap_95pct_skill_lower_bound_positive",
        "value": bootstrap_ci_low,
        "pass": bool(
            np.isfinite(bootstrap_ci_low)
            and bootstrap_ci_low > 0
        ),
        "interpretation": "La mejora frente al baseline permanece positiva bajo bootstrap temporal en bloques.",
    },
    {
        "criterion": "trend_direction_correct",
        "value": (
            physical_slope
            - observed_slope
        ),
        "pass": trend_same_sign,
        "interpretation": "La tendencia simulada debe tener la misma direccion que la observada.",
    },
    {
        "criterion": "delta_skill_vs_zero_change_positive",
        "value": delta_skill_vs_zero,
        "pass": bool(
            np.isfinite(delta_skill_vs_zero)
            and delta_skill_vs_zero > 0
        ),
        "interpretation": "Los incrementos fisicos deben mejorar asumir cambio cero entre observaciones.",
    },
    {
        "criterion": "delta_residual_no_strong_autocorrelation",
        "value": delta_ljung_p,
        "pass": bool(
            np.isfinite(delta_ljung_p)
            and delta_ljung_p >= 0.05
        ),
        "interpretation": (
            "Ljung-Box no rechaza ausencia de autocorrelacion hasta el mayor lag diagnostico. "
            "Un fallo identifica estructura temporal aun no explicada, no invalida automaticamente el modelo."
        ),
    },
]

summary_df = pd.DataFrame(criteria)

n_core_pass = int(
    summary_df.iloc[:5]["pass"].sum()
)

if n_core_pass >= 5:
    diagnostic_label = "ROBUSTO_EN_TEST_RETROSPECTIVO"
elif n_core_pass >= 4:
    diagnostic_label = "PROMETEDOR_CON_RESERVAS"
else:
    diagnostic_label = "NO_ROBUSTO_PARA_PROYECCION"

summary_df["diagnostic_label"] = diagnostic_label

summary_df.to_csv(
    RESULT_DIR / "resumen_validacion.csv",
    index=False,
)

fig = plt.figure(
    figsize=(13, 6)
)

plt.plot(
    predictions["date"],
    predictions["observed_level_m"],
    label="Observado",
    linewidth=2,
)

plt.plot(
    predictions["date"],
    predictions["physical_open_loop_level_m"],
    label="Balance fisico",
    linewidth=1.8,
)

plt.plot(
    predictions["date"],
    predictions["frozen_level_m"],
    label="Nivel inicial congelado",
    linestyle="--",
)

plt.xlabel("Fecha")
plt.ylabel("Nivel EGM2008 (m)")
plt.title(
    "Lago Atitlan - validacion final del balance fisico"
)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIG_DIR / "validacion_final_trayectoria_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

fig = plt.figure(
    figsize=(13, 5)
)

plt.plot(
    predictions["date"],
    predictions["level_residual_m"],
)

plt.axhline(0)

plt.xlabel("Fecha")
plt.ylabel("Predicho - observado (m)")
plt.title(
    "Residual de nivel - trayectoria open-loop"
)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "validacion_final_residuos_nivel.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

if len(horizon_df):
    fig = plt.figure(
        figsize=(10, 6)
    )

    x = np.arange(
        len(horizon_df)
    )

    plt.plot(
        x,
        horizon_df[
            "physical_weighted_MAE_m"
        ],
        marker="o",
        label="Balance fisico",
    )

    plt.plot(
        x,
        horizon_df[
            "frozen_weighted_MAE_m"
        ],
        marker="o",
        label="Nivel congelado",
    )

    plt.xticks(
        x,
        horizon_df["horizon"],
    )

    plt.xlabel("Horizonte")
    plt.ylabel("Weighted MAE (m)")
    plt.title(
        "Validacion rolling-origin por horizonte"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "validacion_final_horizontes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

if len(year_df):
    fig = plt.figure(
        figsize=(9, 5)
    )

    x = np.arange(
        len(year_df)
    )

    plt.bar(
        x - 0.18,
        year_df[
            "physical_weighted_MAE_m"
        ],
        width=0.36,
        label="Balance fisico",
    )

    plt.bar(
        x + 0.18,
        year_df[
            "frozen_weighted_MAE_m"
        ],
        width=0.36,
        label="Nivel congelado",
    )

    plt.xticks(
        x,
        year_df["year"].astype(str),
    )

    plt.xlabel("Año")
    plt.ylabel("Weighted MAE (m)")
    plt.title(
        "Error anual en TEST"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "validacion_final_error_anual.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

central_sensitivity = sensitivity_df.loc[
    sensitivity_df["factor"].eq(1.0)
]

fig = plt.figure(
    figsize=(11, 6)
)

for parameter_name in sensitivity_df[
    "parameter"
].unique():
    sub = sensitivity_df.loc[
        sensitivity_df[
            "parameter"
        ].eq(
            parameter_name
        )
    ].sort_values("factor")

    plt.plot(
        sub["factor"],
        sub["weighted_MAE_m"],
        marker="o",
        label=parameter_name,
    )

plt.xlabel("Factor aplicado al parametro")
plt.ylabel("Weighted MAE TEST (m)")
plt.title(
    "Sensibilidad local de parametros del balance"
)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIG_DIR / "validacion_final_sensibilidad_parametros.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

report_lines = []

report_lines.append(
    "# Validacion final del balance hidrico - Lago de Atitlan"
)
report_lines.append("")
report_lines.append(
    f"Fecha de generacion: {pd.Timestamp.now(tz='UTC').isoformat()}"
)
report_lines.append("")
report_lines.append(
    "## Estado metodologico"
)
report_lines.append("")
report_lines.append(
    "Este modulo no calibra ni entrena modelos. Evalua exclusivamente el balance fisico ya congelado."
)
report_lines.append(
    "El periodo 2024-2026 debe tratarse como holdout retrospectivo de desarrollo y no como un test prospectivo completamente virgen, porque sus resultados ya influyeron en iteraciones previas del proyecto."
)
report_lines.append("")
report_lines.append(
    f"Modelo seleccionado en el paso 04: `{selected_model_from_04}`."
)
report_lines.append("")
report_lines.append(
    "## Resultado global"
)
report_lines.append("")
report_lines.append(
    f"- Weighted MAE balance fisico: {float(physical_global['weighted_MAE_m']):.4f} m"
)
report_lines.append(
    f"- Weighted MAE nivel congelado: {open_loop_base_wmae:.4f} m"
)
report_lines.append(
    f"- Skill MAE frente a nivel congelado: {global_skill:.3f}"
)
report_lines.append(
    f"- NSE/R2: {nse_value:.3f}"
)
report_lines.append(
    f"- Pearson r: {float(physical_global['Pearson_r']):.3f}"
)
report_lines.append(
    f"- Bias ponderado: {float(physical_global['weighted_bias_m']):.4f} m"
)
report_lines.append(
    f"- KGE 2012: {float(physical_global['KGE2012']):.3f}"
)
report_lines.append("")
report_lines.append(
    "KGE y NSE se reportan como diagnosticos complementarios, no como criterios unicos de aceptacion."
)
report_lines.append("")
report_lines.append(
    "## Tendencia"
)
report_lines.append("")
report_lines.append(
    f"- Observada: {observed_slope:.4f} m/año"
)
report_lines.append(
    f"- Simulada: {physical_slope:.4f} m/año"
)
report_lines.append(
    f"- Diferencia absoluta: {abs(physical_slope - observed_slope):.4f} m/año"
)
report_lines.append("")
report_lines.append(
    "## Bootstrap temporal"
)
report_lines.append("")

for _, row in bootstrap_df.iterrows():
    report_lines.append(
        f"- Bloque {int(row['block_length'])}: skill mediano {row['skill_median']:.3f}, IC95% [{row['skill_ci_low']:.3f}, {row['skill_ci_high']:.3f}], P(skill>0)={row['probability_skill_positive']:.3f}"
    )

report_lines.append("")
report_lines.append(
    "## Diagnostico automatico"
)
report_lines.append("")
report_lines.append(
    f"**{diagnostic_label}**"
)
report_lines.append("")

for _, row in summary_df.iterrows():
    report_lines.append(
        f"- {row['criterion']}: {'PASS' if row['pass'] else 'REVIEW'}"
    )

report_lines.append("")
report_lines.append(
    "## Nota"
)
report_lines.append("")
report_lines.append(
    "La sensibilidad Monte Carlo de este modulo es una perturbacion local de parametros y no representa todavia una distribucion probabilistica calibrada de incertidumbre."
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

configuration = {
    "script_version": SCRIPT_VERSION,
    "created_utc": pd.Timestamp.now(
        tz="UTC"
    ).isoformat(),
    "project_root": str(ROOT),
    "validated_model": "PHYSICAL_ONLY",
    "selected_model_from_stage04": selected_model_from_04,
    "test_period_start": str(
        test["date_obs"].min()
    ),
    "test_period_end": str(
        test["date_obs"].max()
    ),
    "n_test": int(len(test)),
    "diagnostic_label": diagnostic_label,
    "bootstrap": {
        "resamples": BOOTSTRAP_RESAMPLES,
        "block_lengths": BOOTSTRAP_BLOCK_LENGTHS,
        "method": "moving block bootstrap paired model/baseline",
    },
    "rolling_horizons": ROLLING_HORIZONS,
    "season_definition": {
        "rainy": "May-Oct",
        "dry": "Nov-Apr",
        "note": (
            "Clasificacion diagnostica simplificada; el inicio real de la epoca lluviosa varia espacial e interanualmente."
        ),
    },
    "sensitivity": {
        "one_at_a_time_factors": [
            0.80,
            0.90,
            1.00,
            1.10,
            1.20,
        ],
        "monte_carlo_draws": MONTE_CARLO_SENSITIVITY_DRAWS,
        "monte_carlo_uniform_fraction": MONTE_CARLO_PARAMETER_FRACTION,
        "interpretation": "sensibilidad local, no incertidumbre probabilistica calibrada",
    },
    "outputs": {
        "global_metrics": str(
            RESULT_DIR / "metricas_globales_test.csv"
        ),
        "horizon_metrics": str(
            RESULT_DIR / "metricas_por_horizonte.csv"
        ),
        "annual_metrics": str(
            RESULT_DIR / "metricas_por_ano.csv"
        ),
        "season_metrics": str(
            RESULT_DIR / "metricas_por_estacion.csv"
        ),
        "regime_metrics": str(
            RESULT_DIR / "metricas_por_regimen.csv"
        ),
        "bootstrap": str(
            RESULT_DIR / "bootstrap_skill.csv"
        ),
        "residual_diagnostics": str(
            RESULT_DIR / "diagnostico_residuos.csv"
        ),
        "parameter_sensitivity": str(
            RESULT_DIR / "sensibilidad_parametros.csv"
        ),
        "monte_carlo": str(
            RESULT_DIR / "sensibilidad_monte_carlo.csv"
        ),
        "summary": str(
            RESULT_DIR / "resumen_validacion.csv"
        ),
        "report": str(
            REPORT_PATH
        ),
    },
}

CONFIG_OUT.write_text(
    json.dumps(
        configuration,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)

STATE_PATH.write_text(
    json.dumps(
        fingerprint,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("05_VALIDAR_MODELO TERMINADO")
print("=" * 72)
print(
    "Modelo validado: PHYSICAL_ONLY"
)
print(
    "Modelo seleccionado en paso 04:",
    selected_model_from_04,
)
print()
print(
    "Metricas globales TEST:"
)
print(
    global_df.to_string(
        index=False
    )
)
print()
print(
    "Validacion rolling por horizonte:"
)
print(
    horizon_df.to_string(
        index=False
    )
)
print()
print(
    "Bootstrap temporal:"
)
print(
    bootstrap_df.to_string(
        index=False
    )
)
print()
print(
    "Tendencias:"
)
print(
    trend_df.to_string(
        index=False
    )
)
print()
print(
    "Resumen diagnostico:"
)
print(
    summary_df[
        [
            "criterion",
            "value",
            "pass",
        ]
    ].to_string(
        index=False
    )
)
print()
print(
    "CLASIFICACION:",
    diagnostic_label,
)
print()
print(
    "Resultados:",
    RESULT_DIR,
)
print(
    "Reporte:",
    REPORT_PATH,
)
print(
    "Configuracion:",
    CONFIG_OUT,
)
print("=" * 72)
