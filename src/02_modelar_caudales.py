__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

try:
    import optuna
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna>=4.0"])
    import optuna

try:
    import xgboost as xgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost>=3.0"])
    import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_VERSION = "2.0.0"
RANDOM_STATE = 42
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, CPU_COUNT - 1)
FAST_TEST = os.environ.get("ATITLAN_FAST_TEST", "0") == "1"
N_CV_SPLITS = 3 if not FAST_TEST else 2
TARGET_TRIALS = {
    "XGB_TWEEDIE": 10 if not FAST_TEST else 1,
    "XGB_LOGHUBER": 10 if not FAST_TEST else 1,
    "RF_LOG": 6 if not FAST_TEST else 1,
}
EXPECTED_RIVERS = ["Quiscab", "San_Francisco", "Tzununa", "La_Catarata", "San_Buenaventura"]
FAMILIES = ["XGB_TWEEDIE", "XGB_LOGHUBER", "RF_LOG"]
TRAIN_TIME_FRACTION = 0.70
CAL_TIME_FRACTION = 0.15

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DATASETS = ROOT / "config" / "datasets_procesados.json"
MANIFEST_01 = ROOT / "data" / "manifest_preparacion.json"
RESULT_DIR = ROOT / "outputs" / "caudales"
FIG_DIR = RESULT_DIR / "figuras"
OPTUNA_EXPORT_DIR = RESULT_DIR / "optuna"
MODEL_DIR = ROOT / "models" / "caudales"
CACHE_DIR = ROOT / "data" / "cache"
STATE_PATH = CACHE_DIR / "estado_02_modelar_caudales.json"
DB_PATH = ROOT / "studies" / "caudales.sqlite3"
CONFIG_OUT = ROOT / "config" / "modelo_caudales.json"

for d in [RESULT_DIR, FIG_DIR, OPTUNA_EXPORT_DIR, MODEL_DIR, CACHE_DIR, DB_PATH.parent, ROOT / "config"]:
    d.mkdir(parents=True, exist_ok=True)

if not CONFIG_DATASETS.exists():
    raise FileNotFoundError(f"No existe {CONFIG_DATASETS}. Ejecuta primero scripts/01_preparar_datos.py")

with open(CONFIG_DATASETS, "r", encoding="utf-8") as f:
    DATASETS = json.load(f)


def project_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def resolve_dataset(key):
    if key not in DATASETS:
        raise KeyError(f"No existe '{key}' en {CONFIG_DATASETS}")
    item = DATASETS[key]
    pkl = project_path(item.get("pkl", "")) if isinstance(item, dict) else Path("")
    csv = project_path(item.get("csv", "")) if isinstance(item, dict) else Path("")
    if pkl.exists():
        return pkl
    if csv.exists():
        return csv
    raise FileNotFoundError(f"No existe producto procesado para {key}: {item}")


RIVERS_PATH = resolve_dataset("rivers_clean")
CLIMATE_PATH = resolve_dataset("climate_daily_features")
MATRIX_PATH = resolve_dataset("matrix_autoregressive")


def read_table(path):
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path, low_memory=False)


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj):
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


SEARCH_CONFIG = {
    "script_version": SCRIPT_VERSION,
    "target_trials": TARGET_TRIALS,
    "n_cv_splits": N_CV_SPLITS,
    "families": FAMILIES,
    "event_weight": "1 + 1.5 * percentile_rank^2",
    "objective_loss": "0.55*MAE + 0.20*abs_bias + 0.25*highflow_MAE",
    "xgb_tree_method": "hist",
    "feature_windows_days": [3, 7, 14, 30, 60, 90],
    "api_half_lives_days": [7, 15, 30, 60],
    "climate_inputs": ["precip", "tas", "tasmin", "tasmax", "hurs", "sfcWind", "rsds"],
}

FINGERPRINT = {
    "script_version": SCRIPT_VERSION,
    "search_config_hash": stable_hash(SEARCH_CONFIG),
    "rivers_sha256": sha256_file(RIVERS_PATH),
    "climate_sha256": sha256_file(CLIMATE_PATH),
    "matrix_sha256": sha256_file(MATRIX_PATH),
}
FINGERPRINT["study_suffix"] = stable_hash(FINGERPRINT)[:12]

EXPECTED_OUTPUTS = [
    RESULT_DIR / "caudal_diario_estimado_desarrollo.csv",
    RESULT_DIR / "caudal_diario_estimado_final.csv",
    RESULT_DIR / "caudal_estimado_mensual.csv",
    RESULT_DIR / "metricas_caudal_por_rio_validacion_temporal.csv",
    RESULT_DIR / "metricas_caudal_total_validacion_temporal.csv",
    RESULT_DIR / "predicciones_caudal_por_rio_validacion_temporal.csv",
    RESULT_DIR / "predicciones_caudal_total_validacion_temporal.csv",
    RESULT_DIR / "seleccion_modelos_por_tributario.csv",
    CONFIG_OUT,
]

if STATE_PATH.exists() and all(p.exists() for p in EXPECTED_OUTPUTS):
    try:
        old_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        old_state = None
    if old_state == FINGERPRINT:
        print("=" * 72)
        print("02_MODELAR_CAUDALES: SIN CAMBIOS")
        print("=" * 72)
        print("Los datos, la configuracion y los productos no cambiaron.")
        print("No se repitio Optuna ni se reentrenaron modelos.")
        print(RESULT_DIR)
        sys.exit(0)


def temporal_partition(dates):
    dates = pd.to_datetime(dates)
    start = dates.min()
    end = dates.max()
    span = end - start
    train_end = start + span * TRAIN_TIME_FRACTION
    calibration_end = start + span * (TRAIN_TIME_FRACTION + CAL_TIME_FRACTION)
    train_idx = np.where(dates <= train_end)[0]
    cal_idx = np.where((dates > train_end) & (dates <= calibration_end))[0]
    test_idx = np.where(dates > calibration_end)[0]
    if len(train_idx) < 200 or len(cal_idx) < 40 or len(test_idx) < 40:
        n = len(dates)
        n_train = int(np.floor(n * TRAIN_TIME_FRACTION))
        n_cal = int(np.floor(n * CAL_TIME_FRACTION))
        train_idx = np.arange(0, n_train)
        cal_idx = np.arange(n_train, n_train + n_cal)
        test_idx = np.arange(n_train + n_cal, n)
    return train_idx, cal_idx, test_idx


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 2 or np.std(y_true) <= 0 or np.std(y_pred) <= 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def build_flow_daily_features(climate):
    daily = climate.copy()
    daily.columns = daily.columns.astype(str).str.strip()
    if "date" not in daily.columns:
        raise KeyError("clima_diario_features no contiene columna date")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    required = ["era_precip_mm", "chirps_precip_mm", "tmean_c", "tmin_c", "tmax_c", "rh_pct", "wind_ms", "solar_mj_m2"]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise KeyError("Faltan variables climaticas necesarias para caudal: " + str(missing))
    for col in required:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily["precip_consensus_mm"] = daily[["era_precip_mm", "chirps_precip_mm"]].mean(axis=1)
    daily["date_month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    doy = daily["date"].dt.dayofyear.astype(float)
    daily["sin_doy_h1_q"] = np.sin(2.0 * np.pi * doy / 365.25)
    daily["cos_doy_h1_q"] = np.cos(2.0 * np.pi * doy / 365.25)
    daily["sin_doy_h2_q"] = np.sin(4.0 * np.pi * doy / 365.25)
    daily["cos_doy_h2_q"] = np.cos(4.0 * np.pi * doy / 365.25)
    p = daily["precip_consensus_mm"].shift(1)
    daily["precip_prev1d_mm"] = p
    for window in [3, 7, 14, 30, 60, 90]:
        minp = max(1, window // 2)
        daily[f"precip_sum_{window}d"] = p.rolling(window, min_periods=minp).sum()
        daily[f"precip_max1d_{window}d"] = p.rolling(window, min_periods=minp).max()
        daily[f"wet_days_{window}d"] = (p >= 1.0).astype(float).rolling(window, min_periods=minp).sum()
        daily[f"heavy10_days_{window}d"] = (p >= 10.0).astype(float).rolling(window, min_periods=minp).sum()
        daily[f"heavy25_days_{window}d"] = (p >= 25.0).astype(float).rolling(window, min_periods=minp).sum()
    for half_life in [7, 15, 30, 60]:
        daily[f"precip_api_hl{half_life}d"] = p.fillna(0.0).ewm(halflife=half_life, adjust=False).mean()
    for col in ["tmean_c", "tmin_c", "tmax_c", "rh_pct", "wind_ms", "solar_mj_m2"]:
        shifted = daily[col].shift(1)
        for window in [7, 30, 60]:
            minp = max(1, window // 2)
            if col == "solar_mj_m2":
                daily[f"{col}_sum_{window}d"] = shifted.rolling(window, min_periods=minp).sum()
            else:
                daily[f"{col}_mean_{window}d"] = shifted.rolling(window, min_periods=minp).mean()
    daily["precip_recent_ratio_7_30"] = daily["precip_sum_7d"] / (daily["precip_sum_30d"] + 1.0)
    daily["precip_recent_ratio_14_60"] = daily["precip_sum_14d"] / (daily["precip_sum_60d"] + 1.0)
    features = [
        c for c in daily.columns
        if c.startswith("precip_")
        or c.startswith("wet_days_")
        or c.startswith("heavy10_days_")
        or c.startswith("heavy25_days_")
        or c.startswith("tmean_c_mean_")
        or c.startswith("tmin_c_mean_")
        or c.startswith("tmax_c_mean_")
        or c.startswith("rh_pct_mean_")
        or c.startswith("wind_ms_mean_")
        or c.startswith("solar_mj_m2_sum_")
        or c in ["sin_doy_h1_q", "cos_doy_h1_q", "sin_doy_h2_q", "cos_doy_h2_q"]
    ]
    return daily, list(dict.fromkeys(features))


@dataclass
class QPreprocessor:
    features: list
    medians: pd.Series


def fit_preprocessor(table, feature_cols, train_rows):
    train_rows = np.asarray(train_rows, dtype=int)
    data = table.iloc[train_rows][feature_cols].copy()
    missing = data.isna().mean()
    features = [c for c in feature_cols if missing.get(c, 1.0) <= 0.20]
    if not features:
        raise ValueError("No quedaron variables de caudal despues del filtro de faltantes")
    variance = data[features].var(skipna=True)
    features = [c for c in features if np.isfinite(variance.get(c, np.nan)) and variance.get(c, 0.0) > 1e-12]
    med0 = data[features].median(numeric_only=True)
    features = [c for c in features if c in med0.index and np.isfinite(med0[c])]
    if not features:
        raise ValueError("No quedaron variables de caudal despues del filtro de varianza")
    temp = data[features].copy()
    for col in features:
        temp[col] = temp[col].fillna(med0[col])
    corr = temp.corr(method="spearman").abs()
    kept = []
    for col in features:
        if all(not (np.isfinite(corr.loc[col, prev]) and corr.loc[col, prev] > 0.995) for prev in kept):
            kept.append(col)
    medians = data[kept].median(numeric_only=True)
    return QPreprocessor(features=kept, medians=medians)


def transform(table, rows, prep):
    rows = np.asarray(rows, dtype=int)
    X = table.iloc[rows][prep.features].copy()
    for col in prep.features:
        X[col] = X[col].fillna(prep.medians[col])
    return X.to_numpy(dtype=np.float32)


def calendar_expanding_splits(table, n_splits):
    dates = pd.to_datetime(table["date"]).reset_index(drop=True)
    start = dates.min()
    end = dates.max()
    span = end - start
    initial_end = start + span * 0.50
    remaining = end - initial_end
    fold_span = remaining / n_splits
    splits = []
    for fold in range(n_splits):
        valid_start = initial_end + fold_span * fold
        valid_end = end + pd.Timedelta(days=1) if fold == n_splits - 1 else initial_end + fold_span * (fold + 1)
        tr = np.where(dates < valid_start)[0]
        va = np.where((dates >= valid_start) & (dates < valid_end))[0]
        if len(tr) >= 24 and len(va) >= 5:
            splits.append((tr, va))
    if len(splits) >= 2:
        return splits
    n = len(table)
    initial = max(24, int(np.floor(n * 0.50)))
    remaining_n = n - initial
    fold_size = max(5, remaining_n // n_splits)
    splits = []
    for fold in range(n_splits):
        train_end = initial + fold * fold_size
        valid_end = n if fold == n_splits - 1 else min(n, train_end + fold_size)
        tr = np.arange(0, train_end)
        va = np.arange(train_end, valid_end)
        if len(tr) >= 24 and len(va) >= 5:
            splits.append((tr, va))
    return splits


def event_weights(y):
    y = np.asarray(y, dtype=float)
    ranks = pd.Series(y).rank(method="average", pct=True).to_numpy(dtype=float)
    return 1.0 + 1.5 * ranks ** 2


def suggest_params(trial, family):
    if FAST_TEST:
        if family == "XGB_TWEEDIE":
            return {
                "n_estimators": trial.suggest_categorical("n_estimators", [80]),
                "max_depth": trial.suggest_categorical("max_depth", [3]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.05]),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [1.0]),
                "subsample": trial.suggest_categorical("subsample", [0.85]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.75]),
                "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-4]),
                "reg_lambda": trial.suggest_categorical("reg_lambda", [1.0]),
                "gamma": trial.suggest_categorical("gamma", [0.0]),
                "tweedie_variance_power": trial.suggest_categorical("tweedie_variance_power", [1.5]),
            }
        if family == "XGB_LOGHUBER":
            return {
                "n_estimators": trial.suggest_categorical("n_estimators", [80]),
                "max_depth": trial.suggest_categorical("max_depth", [3]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.05]),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [1.0]),
                "subsample": trial.suggest_categorical("subsample", [0.85]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.75]),
                "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-4]),
                "reg_lambda": trial.suggest_categorical("reg_lambda", [1.0]),
                "gamma": trial.suggest_categorical("gamma", [0.0]),
                "huber_slope": trial.suggest_categorical("huber_slope", [1.0]),
            }
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [80]),
            "max_depth": trial.suggest_categorical("max_depth", [8]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [4]),
            "max_features": trial.suggest_categorical("max_features", [0.7]),
            "max_samples": trial.suggest_categorical("max_samples", [0.85]),
        }
    if family == "XGB_TWEEDIE":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.10, 1.90),
        }
    if family == "XGB_LOGHUBER":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "huber_slope": trial.suggest_float("huber_slope", 0.05, 1.5, log=True),
        }
    if family == "RF_LOG":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 10, 14, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 14),
            "max_features": trial.suggest_float("max_features", 0.35, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.65, 1.0),
        }
    raise ValueError(family)


def make_model(family, params):
    if family == "XGB_TWEEDIE":
        return xgb.XGBRegressor(
            objective="reg:tweedie",
            eval_metric="mae",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
            **params,
        )
    if family == "XGB_LOGHUBER":
        return xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            eval_metric="mae",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
            **params,
        )
    if family == "RF_LOG":
        return RandomForestRegressor(
            criterion="squared_error",
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            **params,
        )
    raise ValueError(family)


def fit_model(family, params, table, feature_cols, fit_rows):
    prep = fit_preprocessor(table, feature_cols, fit_rows)
    X = transform(table, fit_rows, prep)
    y = table.iloc[np.asarray(fit_rows, dtype=int)]["flow_m3s"].to_numpy(dtype=float)
    w = event_weights(y)
    target = y if family == "XGB_TWEEDIE" else np.log1p(np.maximum(y, 0.0))
    model = make_model(family, params)
    if family.startswith("XGB"):
        model.fit(X, target, sample_weight=w, verbose=False)
    else:
        model.fit(X, target, sample_weight=w)
    return model, prep


def predict_model(model, family, prep, table, rows):
    X = transform(table, rows, prep)
    raw = np.asarray(model.predict(X), dtype=float)
    pred = raw if family == "XGB_TWEEDIE" else np.expm1(raw)
    return np.maximum(pred, 0.0)


def score_flow(y_true, y_pred, training_y):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    training_y = np.asarray(training_y, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    threshold = float(np.quantile(training_y, 0.75)) if len(training_y) else np.inf
    high = y_true >= threshold
    high_mae = float(mean_absolute_error(y_true[high], y_pred[high])) if np.any(high) else mae
    return 0.55 * mae + 0.20 * abs(bias) + 0.25 * high_mae


def metrics_flow(model_name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    q75 = float(np.quantile(y_true, 0.75)) if len(y_true) else np.nan
    high = y_true >= q75 if np.isfinite(q75) else np.zeros(len(y_true), dtype=bool)
    return {
        "model": model_name,
        "n": int(len(y_true)),
        "MAE_m3s": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_m3s": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias_m3s": float(np.mean(y_pred - y_true)),
        "highflow_MAE_m3s": float(mean_absolute_error(y_true[high], y_pred[high])) if np.any(high) else np.nan,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "NSE": float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else np.nan,
        "Pearson_r": safe_pearson(y_true, y_pred),
    }


def monthly_climatology(train_table, pred_dates):
    medians = train_table.groupby(train_table["date"].dt.month)["flow_m3s"].median().to_dict()
    global_median = float(train_table["flow_m3s"].median())
    return np.array([medians.get(pd.Timestamp(d).month, global_median) for d in pred_dates], dtype=float)


def objective_factory(family, table, feature_cols, splits):
    def objective(trial):
        params = suggest_params(trial, family)
        losses = []
        for fold, (tr, va) in enumerate(splits):
            model, prep = fit_model(family, params, table, feature_cols, tr)
            pred = predict_model(model, family, prep, table, va)
            true = table.iloc[va]["flow_m3s"].to_numpy(dtype=float)
            train_y = table.iloc[tr]["flow_m3s"].to_numpy(dtype=float)
            losses.append(score_flow(true, pred, train_y))
            trial.report(float(np.mean(losses)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))
    return objective


def save_model_bundle(model, family, prep, river, stage, params):
    prefix = MODEL_DIR / f"{river.lower()}_{stage}_{family.lower()}"
    prep_path = Path(str(prefix) + "_preprocessor.joblib")
    joblib.dump(prep, prep_path)
    if family.startswith("XGB"):
        model_path = Path(str(prefix) + ".json")
        model.save_model(str(model_path))
    else:
        model_path = Path(str(prefix) + ".joblib")
        joblib.dump(model, model_path)
    meta_path = Path(str(prefix) + "_meta.json")
    meta = {
        "river": river,
        "stage": stage,
        "family": family,
        "model_path": str(model_path),
        "preprocessor_path": str(prep_path),
        "features": prep.features,
        "medians": {k: float(v) for k, v in prep.medians.items()},
        "params": params,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {"model": str(model_path), "preprocessor": str(prep_path), "meta": str(meta_path)}


matrix = read_table(MATRIX_PATH).copy()
matrix["date_obs"] = pd.to_datetime(matrix["date_obs"], errors="coerce")
matrix = matrix.dropna(subset=["date_obs", "nivel_final_m"]).sort_values("date_obs").reset_index(drop=True)
idx_train, idx_cal, idx_test = temporal_partition(matrix["date_obs"])
train_end = pd.Timestamp(matrix.loc[idx_train, "date_obs"].max()).normalize()
cal_end = pd.Timestamp(matrix.loc[idx_cal, "date_obs"].max()).normalize()
q_dev_cutoff = train_end - pd.Timedelta(days=1)
q_final_cutoff = cal_end - pd.Timedelta(days=1)

rivers = read_table(RIVERS_PATH).copy()
rivers["date"] = pd.to_datetime(rivers["date"], errors="coerce").dt.normalize()
rivers["flow_m3s"] = pd.to_numeric(rivers["flow_m3s"], errors="coerce")
rivers = rivers.dropna(subset=["date", "river", "flow_m3s"])
rivers = rivers.loc[rivers["flow_m3s"] >= 0].copy()
rivers = (
    rivers.groupby(["date", "river"], as_index=False)
    .agg(flow_m3s=("flow_m3s", "median"), n_measurements=("flow_m3s", "size"))
    .sort_values(["river", "date"])
    .reset_index(drop=True)
)
rivers["date_month"] = rivers["date"].dt.to_period("M").dt.to_timestamp()

actual_rivers = sorted(rivers["river"].dropna().unique().tolist())
missing_rivers = [r for r in EXPECTED_RIVERS if r not in actual_rivers]
if missing_rivers:
    raise ValueError("Faltan tributarios esperados: " + str(missing_rivers))

climate = read_table(CLIMATE_PATH)
flow_daily, feature_cols = build_flow_daily_features(climate)
flow_daily.to_pickle(RESULT_DIR / "clima_diario_features_modelo_caudal.pkl")
flow_daily[["date"] + feature_cols].to_csv(RESULT_DIR / "clima_diario_features_modelo_caudal.csv", index=False)

obs = rivers.merge(flow_daily[["date"] + feature_cols], on="date", how="left")
obs = obs.sort_values(["river", "date"]).reset_index(drop=True)

storage_url = "sqlite:///" + DB_PATH.as_posix()
storage = optuna.storages.RDBStorage(
    url=storage_url,
    engine_kwargs={"connect_args": {"timeout": 60}},
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

selection_rows = []
holdout_rows = []
holdout_metric_rows = []
model_config = {}
q_daily_dev = flow_daily[["date"] + feature_cols].copy()
q_daily_final = flow_daily[["date"] + feature_cols].copy()

print("=" * 72)
print("02_MODELAR_CAUDALES - MODELOS POR TRIBUTARIO")
print("=" * 72)
print("Cutoff desarrollo:", q_dev_cutoff.date())
print("Cutoff final:", q_final_cutoff.date())
print("Base Optuna:", DB_PATH)
print()

for river_number, river in enumerate(EXPECTED_RIVERS):
    river_table = obs.loc[obs["river"] == river].sort_values("date").reset_index(drop=True)
    dev_table = river_table.loc[river_table["date"] <= q_dev_cutoff].copy().reset_index(drop=True)
    final_table = river_table.loc[river_table["date"] <= q_final_cutoff].copy().reset_index(drop=True)
    holdout = river_table.loc[(river_table["date"] > q_dev_cutoff) & (river_table["date"] <= q_final_cutoff)].copy().reset_index(drop=True)
    if len(dev_table) < 45:
        raise ValueError(f"Muy pocos aforos de desarrollo para {river}: {len(dev_table)}")
    splits = calendar_expanding_splits(dev_table, N_CV_SPLITS)
    if len(splits) < 2:
        raise ValueError(f"No fue posible construir CV temporal para {river}")
    family_results = []
    print("Tributario:", river)
    for family_number, family in enumerate(FAMILIES):
        startup = 4 if TARGET_TRIALS[family] >= 4 else 1
        sampler = optuna.samplers.TPESampler(
            seed=RANDOM_STATE + 200 + river_number * 10 + family_number,
            multivariate=True,
            n_startup_trials=startup,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=startup,
            n_warmup_steps=1,
        )
        study_name = f"Atitlan_Q_{FINGERPRINT['study_suffix']}_{river}_{family}"
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        finished_states = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
        finished = sum(t.state in finished_states for t in study.trials)
        remaining = max(0, TARGET_TRIALS[family] - finished)
        if remaining > 0:
            print(f"  {family}: {finished}/{TARGET_TRIALS[family]} trials guardados; ejecutando {remaining}")
            study.optimize(
                objective_factory(family, dev_table, feature_cols, splits),
                n_trials=remaining,
                gc_after_trial=True,
                show_progress_bar=True,
            )
        else:
            print(f"  {family}: reutilizando {finished} trials existentes")
        trials_df = study.trials_dataframe()
        trials_df.to_csv(OPTUNA_EXPORT_DIR / f"optuna_{river.lower()}_{family.lower()}_trials.csv", index=False)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError(f"No hay trials completos para {river} {family}")
        family_results.append((family, float(study.best_value), dict(study.best_params), study_name))
    family_results.sort(key=lambda x: x[1])
    best_family, best_loss, best_params, best_study_name = family_results[0]
    dev_model, dev_prep = fit_model(best_family, best_params, dev_table, feature_cols, np.arange(len(dev_table)))
    final_model, final_prep = fit_model(best_family, best_params, final_table, feature_cols, np.arange(len(final_table)))
    q_daily_dev[river] = predict_model(dev_model, best_family, dev_prep, q_daily_dev, np.arange(len(q_daily_dev)))
    q_daily_final[river] = predict_model(final_model, best_family, final_prep, q_daily_final, np.arange(len(q_daily_final)))
    dev_paths = save_model_bundle(dev_model, best_family, dev_prep, river, "desarrollo", best_params)
    final_paths = save_model_bundle(final_model, best_family, final_prep, river, "final", best_params)
    if len(holdout):
        pred = predict_model(dev_model, best_family, dev_prep, holdout, np.arange(len(holdout)))
        clim = monthly_climatology(dev_table, holdout["date"])
        h = holdout[["date", "date_month", "river", "flow_m3s"]].copy()
        h["pred_m3s"] = pred
        h["climatology_m3s"] = clim
        h["model_family"] = best_family
        holdout_rows.append(h)
        met_model = metrics_flow(f"{river}_{best_family}", holdout["flow_m3s"], pred)
        met_base = metrics_flow(f"{river}_MonthlyClimatology", holdout["flow_m3s"], clim)
        met_model["river"] = river
        met_base["river"] = river
        met_model["MAE_skill_vs_monthly_climatology"] = 1.0 - met_model["MAE_m3s"] / met_base["MAE_m3s"] if met_base["MAE_m3s"] > 0 else np.nan
        met_base["MAE_skill_vs_monthly_climatology"] = 0.0
        holdout_metric_rows.extend([met_model, met_base])
    selection_rows.append({
        "river": river,
        "selected_family": best_family,
        "best_cv_loss": best_loss,
        "study_name": best_study_name,
        "n_dev": int(len(dev_table)),
        "n_holdout": int(len(holdout)),
        "n_final": int(len(final_table)),
        "dev_last_date": dev_table["date"].max(),
        "final_last_date": final_table["date"].max(),
        "n_features_dev": len(dev_prep.features),
        "n_features_final": len(final_prep.features),
    })
    model_config[river] = {
        "family": best_family,
        "best_cv_loss": best_loss,
        "best_params": best_params,
        "study_name": best_study_name,
        "n_dev": int(len(dev_table)),
        "n_holdout": int(len(holdout)),
        "n_final": int(len(final_table)),
        "dev_model": dev_paths,
        "final_model": final_paths,
        "dev_features": dev_prep.features,
        "final_features": final_prep.features,
    }
    print("  Seleccionado:", best_family, "CV loss", round(best_loss, 6))
    print()

q_daily_dev["q_pred_dev_m3s"] = q_daily_dev[EXPECTED_RIVERS].sum(axis=1)
q_daily_final["q_pred_final_m3s"] = q_daily_final[EXPECTED_RIVERS].sum(axis=1)

daily_dev_out = q_daily_dev[["date"] + EXPECTED_RIVERS + ["q_pred_dev_m3s"]].copy()
daily_final_out = q_daily_final[["date"] + EXPECTED_RIVERS + ["q_pred_final_m3s"]].copy()
daily_dev_out.to_csv(RESULT_DIR / "caudal_diario_estimado_desarrollo.csv", index=False)
daily_dev_out.to_pickle(RESULT_DIR / "caudal_diario_estimado_desarrollo.pkl")
daily_final_out.to_csv(RESULT_DIR / "caudal_diario_estimado_final.csv", index=False)
daily_final_out.to_pickle(RESULT_DIR / "caudal_diario_estimado_final.pkl")

selection_df = pd.DataFrame(selection_rows)
selection_df.to_csv(RESULT_DIR / "seleccion_modelos_por_tributario.csv", index=False)

holdout_all = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()
holdout_all.to_csv(RESULT_DIR / "predicciones_caudal_por_rio_validacion_temporal.csv", index=False)
metrics_river_df = pd.DataFrame(holdout_metric_rows)
metrics_river_df.to_csv(RESULT_DIR / "metricas_caudal_por_rio_validacion_temporal.csv", index=False)

monthly_obs = (
    rivers.groupby(["date_month", "river"], as_index=False)["flow_m3s"].median()
    .pivot(index="date_month", columns="river", values="flow_m3s")
    .reset_index()
)
for river in EXPECTED_RIVERS:
    if river not in monthly_obs.columns:
        monthly_obs[river] = np.nan
monthly_obs["n_rivers"] = monthly_obs[EXPECTED_RIVERS].notna().sum(axis=1)
monthly_obs["q_total_obs_m3s"] = monthly_obs[EXPECTED_RIVERS].sum(axis=1, min_count=len(EXPECTED_RIVERS))

monthly_dev = daily_dev_out.set_index("date")[EXPECTED_RIVERS + ["q_pred_dev_m3s"]].resample("MS").mean().reset_index().rename(columns={"date": "date_month"})
monthly_final = daily_final_out.set_index("date")[EXPECTED_RIVERS + ["q_pred_final_m3s"]].resample("MS").mean().reset_index().rename(columns={"date": "date_month"})
monthly_out = monthly_dev.merge(monthly_final[["date_month", "q_pred_final_m3s"]], on="date_month", how="outer")
monthly_out = monthly_out.merge(monthly_obs[["date_month", "q_total_obs_m3s", "n_rivers"]], on="date_month", how="left")
monthly_out = monthly_out.sort_values("date_month")
monthly_out.to_csv(RESULT_DIR / "caudal_estimado_mensual.csv", index=False)
monthly_out.to_pickle(RESULT_DIR / "caudal_estimado_mensual.pkl")

metrics_total_rows = []
q_total_holdout = pd.DataFrame()
if not holdout_all.empty:
    q_total_holdout = (
        holdout_all.groupby("date_month", as_index=False)
        .agg(
            n_rivers=("river", "nunique"),
            q_total_obs_m3s=("flow_m3s", "sum"),
            q_total_pred_m3s=("pred_m3s", "sum"),
            q_total_climatology_m3s=("climatology_m3s", "sum"),
        )
    )
    q_total_holdout = q_total_holdout.loc[q_total_holdout["n_rivers"] == len(EXPECTED_RIVERS)].copy()
    if len(q_total_holdout):
        met_model = metrics_flow("SUM_PER_RIVER_MODELS", q_total_holdout["q_total_obs_m3s"], q_total_holdout["q_total_pred_m3s"])
        met_base = metrics_flow("SUM_PER_RIVER_MONTHLY_CLIMATOLOGY", q_total_holdout["q_total_obs_m3s"], q_total_holdout["q_total_climatology_m3s"])
        met_model["MAE_skill_vs_monthly_climatology"] = 1.0 - met_model["MAE_m3s"] / met_base["MAE_m3s"] if met_base["MAE_m3s"] > 0 else np.nan
        met_base["MAE_skill_vs_monthly_climatology"] = 0.0
        metrics_total_rows = [met_model, met_base]
q_total_holdout.to_csv(RESULT_DIR / "predicciones_caudal_total_validacion_temporal.csv", index=False)
metrics_total_df = pd.DataFrame(metrics_total_rows)
metrics_total_df.to_csv(RESULT_DIR / "metricas_caudal_total_validacion_temporal.csv", index=False)

if len(q_total_holdout):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(q_total_holdout["date_month"], q_total_holdout["q_total_obs_m3s"], marker="o", label="Observado")
    plt.plot(q_total_holdout["date_month"], q_total_holdout["q_total_pred_m3s"], marker="o", label="Modelos por tributario")
    plt.plot(q_total_holdout["date_month"], q_total_holdout["q_total_climatology_m3s"], label="Climatologia mensual")
    plt.xlabel("Fecha")
    plt.ylabel("Caudal total 5 tributarios (m3/s)")
    plt.title("Validacion temporal del submodelo de caudales")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "validacion_caudal_total.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

configuration = {
    "script_version": SCRIPT_VERSION,
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "project_root": str(ROOT),
    "optuna_database": str(DB_PATH),
    "study_suffix": FINGERPRINT["study_suffix"],
    "development_cutoff": str(q_dev_cutoff),
    "final_cutoff": str(q_final_cutoff),
    "lake_train_end": str(train_end),
    "lake_calibration_end": str(cal_end),
    "future_compatibility": "Variables de caudal restringidas a predictores derivables de NEX-GDDP-CMIP6: precipitacion, temperatura, humedad relativa, viento, radiacion y estacionalidad.",
    "aforo_alignment": "Todas las ventanas climaticas terminan el dia anterior al aforo.",
    "optuna_persistence": "Los estudios usan SQLite y load_if_exists=True; en reanudaciones solo se ejecutan trials faltantes hasta TARGET_TRIALS.",
    "search_config": SEARCH_CONFIG,
    "river_models": model_config,
    "outputs": {
        "daily_development": str(RESULT_DIR / "caudal_diario_estimado_desarrollo.pkl"),
        "daily_final": str(RESULT_DIR / "caudal_diario_estimado_final.pkl"),
        "monthly": str(RESULT_DIR / "caudal_estimado_mensual.pkl"),
        "metrics_per_river": str(RESULT_DIR / "metricas_caudal_por_rio_validacion_temporal.csv"),
        "metrics_total": str(RESULT_DIR / "metricas_caudal_total_validacion_temporal.csv"),
        "selection": str(RESULT_DIR / "seleccion_modelos_por_tributario.csv"),
    },
}
CONFIG_OUT.write_text(json.dumps(configuration, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
STATE_PATH.write_text(json.dumps(FINGERPRINT, indent=2, ensure_ascii=False), encoding="utf-8")

print("=" * 72)
print("02_MODELAR_CAUDALES TERMINADO")
print("=" * 72)
print("Proyecto:", ROOT)
print("Aforos:", len(rivers))
print("Tributarios:", EXPECTED_RIVERS)
print("Predictores climaticos de caudal:", len(feature_cols))
print("Cutoff desarrollo:", q_dev_cutoff.date())
print("Cutoff final:", q_final_cutoff.date())
print()
print("Modelos seleccionados:")
for _, row in selection_df.iterrows():
    print(row["river"], ":", row["selected_family"], "CV loss", round(float(row["best_cv_loss"]), 6))
if len(metrics_total_df):
    print()
    print("Metricas de caudal total en validacion temporal:")
    print(metrics_total_df.to_string(index=False))
print()
print("SQLite Optuna:", DB_PATH)
print("Resultados:", RESULT_DIR)
print("Configuracion:", CONFIG_OUT)
print("=" * 72)
