__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "xgboost>=3.0"])
    import xgboost as xgb

try:
    import optuna
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "optuna>=4.0"])
    import optuna

try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_VERSION = "1.0.0"
RANDOM_STATE = 42
N_CV_SPLITS = 4
MAX_MISSING_TRAIN = 0.10
CORR_THRESHOLD = 0.995
TARGET_TRIALS_XGB = 30
TARGET_TRIALS_RF = 20
MIN_CALIBRATION_SCORE_IMPROVEMENT = 0.05
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, CPU_COUNT - 1)

ROOT = Path(__file__).resolve().parents[1]
DATASETS_CONFIG = ROOT / "config" / "datasets_procesados.json"
BALANCE_CONFIG = ROOT / "config" / "balance_hidrico.json"
RESULT_DIR = ROOT / "outputs" / "ml_residual"
MODEL_DIR = ROOT / "models" / "nivel"
FIG_DIR = ROOT / "outputs" / "figuras"
CACHE_DIR = ROOT / "data" / "cache"
DB_PATH = ROOT / "studies" / "ml_residual.sqlite3"
STATE_PATH = CACHE_DIR / "estado_04_ml_residual.json"
CONFIG_OUT = ROOT / "config" / "ml_residual.json"

for d in [RESULT_DIR, MODEL_DIR, FIG_DIR, CACHE_DIR, DB_PATH.parent, ROOT / "config"]:
    d.mkdir(parents=True, exist_ok=True)

if not DATASETS_CONFIG.exists():
    raise FileNotFoundError("Falta config/datasets_procesados.json. Ejecuta primero 01_preparar_datos.py")
if not BALANCE_CONFIG.exists():
    raise FileNotFoundError("Falta config/balance_hidrico.json. Ejecuta primero 03_balance_hidrico.py")

with open(DATASETS_CONFIG, "r", encoding="utf-8") as f:
    DATASETS = json.load(f)

with open(BALANCE_CONFIG, "r", encoding="utf-8") as f:
    BALANCE_CFG = json.load(f)


def resolve_dataset(name):
    item = DATASETS.get(name)
    if item is None:
        raise KeyError(f"No existe dataset procesado: {name}")
    if isinstance(item, dict):
        pkl = item.get("pkl")
        csv = item.get("csv")
        if pkl and Path(pkl).exists():
            return Path(pkl)
        if csv and Path(csv).exists():
            return Path(csv)
    elif Path(item).exists():
        return Path(item)
    raise FileNotFoundError(f"No se encontro el dataset procesado {name}")


def read_table(path):
    path = Path(path)
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


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 2 or np.std(y_true) <= 0 or np.std(y_pred) <= 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metrics(name, y_true, y_pred, weights=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if weights is None:
        w = np.ones(len(y_true), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[valid]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    mae = float(mean_absolute_error(y_true, y_pred))
    wmae = float(np.average(np.abs(y_true - y_pred), weights=w))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wrmse = float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))
    bias = float(np.mean(y_pred - y_true))
    wbias = float(np.average(y_pred - y_true, weights=w))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return {
        "model": name,
        "n": int(len(y_true)),
        "MAE_m": mae,
        "weighted_MAE_m": wmae,
        "RMSE_m": rmse,
        "weighted_RMSE_m": wrmse,
        "bias_m": bias,
        "weighted_bias_m": wbias,
        "R2": r2,
        "NSE": r2,
        "Pearson_r": safe_pearson(y_true, y_pred),
    }


def trajectory_from_delta(start_level, delta):
    current = float(start_level)
    out = []
    for d in np.asarray(delta, dtype=float):
        current += float(d)
        out.append(current)
    return np.asarray(out, dtype=float)


def composite_trajectory_loss(y_true_level, y_pred_level, y_true_delta, y_pred_delta, weights):
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    level_wmae = float(np.average(np.abs(y_true_level - y_pred_level), weights=w))
    level_wbias = float(np.average(y_pred_level - y_true_level, weights=w))
    delta_wmae = float(np.average(np.abs(y_true_delta - y_pred_delta), weights=w))
    return level_wmae + 0.50 * abs(level_wbias) + 0.25 * delta_wmae


def future_compatible_feature(name):
    n = str(name).lower()
    if n.startswith("nivel_"):
        return False
    forbidden = [
        "pressure",
        "soil_water",
        "runoff",
        "surface_runoff",
        "subsurface_runoff",
        "source_ratio",
        "difference",
        "ratio_chirps_era",
        "abs_difference",
        "sample_weight",
        "estimated_error",
        "mission",
        "cycle",
        "target",
    ]
    if any(token in n for token in forbidden):
        return False
    allowed_prefixes = [
        "chirps_precip",
        "tmean_c",
        "tmin_c",
        "tmax_c",
        "dewpoint_c",
        "rh_pct",
        "vpd_kpa",
        "wind_ms",
        "solar_mj_m2",
        "sin_doy",
        "cos_doy",
        "days_since_previous_obs",
    ]
    return any(n.startswith(prefix) for prefix in allowed_prefixes)


def calendar_expanding_splits(dates, n_splits=4):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    start = dates.min()
    end = dates.max()
    duration = end - start
    initial_end = start + duration * 0.45
    remaining = end - initial_end
    fold_span = remaining / n_splits
    splits = []

    for fold in range(n_splits):
        valid_start = initial_end + fold_span * fold
        valid_end = end + pd.Timedelta(days=1) if fold == n_splits - 1 else initial_end + fold_span * (fold + 1)
        tr = np.where(dates < valid_start)[0]
        va = np.where((dates >= valid_start) & (dates < valid_end))[0]
        if len(tr) >= 100 and len(va) >= 15:
            splits.append((tr, va))

    if len(splits) < 3:
        n = len(dates)
        initial = int(np.floor(n * 0.45))
        remaining_n = n - initial
        test_size = max(15, remaining_n // n_splits)
        splits = []
        for fold in range(n_splits):
            train_end = initial + fold * test_size
            valid_start = train_end
            valid_end = n if fold == n_splits - 1 else min(n, valid_start + test_size)
            tr = np.arange(0, train_end)
            va = np.arange(valid_start, valid_end)
            if len(tr) >= 100 and len(va) >= 10:
                splits.append((tr, va))
    return splits


def fit_preprocessor(X_raw, train_rows, candidate_features):
    tr = X_raw.iloc[train_rows][candidate_features].copy()

    missing = tr.isna().mean()
    features = missing[missing <= MAX_MISSING_TRAIN].index.tolist()

    if not features:
        raise ValueError("No quedaron predictores tras filtro de missing")

    medians = tr[features].median(numeric_only=True)

    for col in features:
        tr[col] = tr[col].fillna(medians[col])

    variance = tr[features].var(skipna=True)
    features = variance[variance > 1e-12].index.tolist()

    if not features:
        raise ValueError("No quedaron predictores tras filtro de varianza")

    tr2 = tr[features].copy()
    corr = tr2.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
    features = [c for c in features if c not in drop_corr]

    medians = X_raw.iloc[train_rows][features].median(numeric_only=True)

    return features, medians


def transform_matrix(X_raw, rows, features, medians):
    X = X_raw.iloc[rows][features].copy()
    for col in features:
        X[col] = X[col].fillna(medians[col])
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError("Persisten NaN tras imputacion en: " + str(bad))
    return X.to_numpy(dtype=np.float32)


def make_xgb(params):
    return xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        eval_metric="mae",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=0,
        **params,
    )


def make_rf(params):
    return RandomForestRegressor(
        criterion="squared_error",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        **params,
    )


def xgb_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1400, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        "huber_slope": trial.suggest_float("huber_slope", 0.02, 0.8, log=True),
    }


def rf_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1400, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 10, 14, 18, None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 4, 24),
        "max_features": trial.suggest_float("max_features", 0.25, 0.85),
        "max_samples": trial.suggest_float("max_samples", 0.60, 1.0),
    }


def completed_or_terminal_trials(study):
    return len([
        t for t in study.trials
        if t.state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        )
    ])


MATRIX_PATH = resolve_dataset("matrix_autoregressive")
FEATURES_PATH = resolve_dataset("features_autoregressive_unique")
BALANCE_INTERVALS_PATH = Path(BALANCE_CFG["outputs"]["intervals"])

for p in [MATRIX_PATH, FEATURES_PATH, BALANCE_INTERVALS_PATH]:
    if not p.exists():
        raise FileNotFoundError(p)

fingerprint_base = {
    "script_version": SCRIPT_VERSION,
    "matrix_sha256": sha256_file(MATRIX_PATH),
    "features_sha256": sha256_file(FEATURES_PATH),
    "balance_sha256": sha256_file(BALANCE_INTERVALS_PATH),
    "target_trials_xgb": TARGET_TRIALS_XGB,
    "target_trials_rf": TARGET_TRIALS_RF,
    "corr_threshold": CORR_THRESHOLD,
    "max_missing_train": MAX_MISSING_TRAIN,
    "minimum_calibration_score_improvement": MIN_CALIBRATION_SCORE_IMPROVEMENT,
}
study_suffix = hashlib.sha256(
    json.dumps(fingerprint_base, sort_keys=True).encode("utf-8")
).hexdigest()[:12]
fingerprint = {**fingerprint_base, "study_suffix": study_suffix}

expected_outputs = [
    RESULT_DIR / "seleccion_corrector_residual.csv",
    RESULT_DIR / "metricas_test_ml_residual.csv",
    RESULT_DIR / "predicciones_test_ml_residual.csv",
    RESULT_DIR / "features_residuales_train.csv",
    RESULT_DIR / "features_residuales_train_cal.csv",
    CONFIG_OUT,
]

if STATE_PATH.exists() and all(p.exists() for p in expected_outputs):
    try:
        old_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        old_state = None
    if old_state == fingerprint:
        print("=" * 72)
        print("04_ML_RESIDUAL: SIN CAMBIOS")
        print("=" * 72)
        print("Los datos, balance y configuracion no cambiaron.")
        print("No se repitio Optuna ni el entrenamiento.")
        print(RESULT_DIR)
        sys.exit(0)

matrix = read_table(MATRIX_PATH).copy()
matrix.columns = matrix.columns.astype(str).str.strip()
matrix["date_obs"] = pd.to_datetime(matrix["date_obs"], errors="coerce")
matrix["nivel_final_m"] = pd.to_numeric(matrix["nivel_final_m"], errors="coerce")
matrix = matrix.dropna(subset=["date_obs", "nivel_final_m"]).sort_values("date_obs").reset_index(drop=True)

feature_table = read_table(FEATURES_PATH).copy()
feature_table.columns = feature_table.columns.astype(str).str.strip()
if "feature" not in feature_table.columns:
    raise KeyError("features_autoregresivas_unicas no contiene columna feature")

raw_feature_list = (
    feature_table["feature"]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

candidate_exog = [
    c for c in dict.fromkeys(raw_feature_list)
    if c in matrix.columns and future_compatible_feature(c)
]

intervals = read_table(BALANCE_INTERVALS_PATH).copy()
intervals["date_obs"] = pd.to_datetime(intervals["date_obs"], errors="coerce")
intervals["row_index"] = pd.to_numeric(intervals["row_index"], errors="coerce").astype(int)
intervals = intervals.sort_values("date_obs").reset_index(drop=True)

matrix_features = matrix.loc[intervals["row_index"], candidate_exog].reset_index(drop=True)
for col in candidate_exog:
    matrix_features[col] = pd.to_numeric(matrix_features[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

physical_feature_names_dev = [
    "duration_days",
    "p_consensus_m",
    "evap_m",
    "q_in_dev_equiv_m",
    "delta_balance_calibrated_dev_m",
]
physical_feature_names_final = [
    "duration_days",
    "p_consensus_m",
    "evap_m",
    "q_in_final_equiv_m",
    "delta_balance_calibrated_final_m",
]

X_dev = matrix_features.copy()
X_final = matrix_features.copy()

for col in physical_feature_names_dev:
    X_dev[f"PHYS_{col}"] = pd.to_numeric(intervals[col], errors="coerce")

for col in physical_feature_names_final:
    X_final[f"PHYS_{col}"] = pd.to_numeric(intervals[col], errors="coerce")

candidate_dev = candidate_exog + [f"PHYS_{c}" for c in physical_feature_names_dev]
candidate_final = candidate_exog + [f"PHYS_{c}" for c in physical_feature_names_final]

train_rows = np.where(intervals["partition"].eq("train").to_numpy())[0]
cal_rows = np.where(intervals["partition"].eq("calibration").to_numpy())[0]
test_rows = np.where(intervals["partition"].eq("test").to_numpy())[0]
train_cal_rows = np.where(intervals["partition"].isin(["train", "calibration"]).to_numpy())[0]

if min(len(train_rows), len(cal_rows), len(test_rows)) == 0:
    raise ValueError("Particiones vacias en balance_componentes_intervalos")

y_resid_dev = pd.to_numeric(intervals["residual_balance_dev_m"], errors="coerce").to_numpy(dtype=float)
y_resid_final = pd.to_numeric(intervals["residual_balance_final_m"], errors="coerce").to_numpy(dtype=float)
w = pd.to_numeric(intervals["interval_weight"], errors="coerce").fillna(1.0).clip(0.10, 10.0).to_numpy(dtype=float)
y_level = pd.to_numeric(intervals["level_obs_m"], errors="coerce").to_numpy(dtype=float)
y_delta = pd.to_numeric(intervals["delta_h_obs_m"], errors="coerce").to_numpy(dtype=float)
delta_phys_dev = pd.to_numeric(intervals["delta_balance_calibrated_dev_m"], errors="coerce").to_numpy(dtype=float)
delta_phys_final = pd.to_numeric(intervals["delta_balance_calibrated_final_m"], errors="coerce").to_numpy(dtype=float)

train_dates = intervals.iloc[train_rows]["date_obs"].reset_index(drop=True)
local_splits = calendar_expanding_splits(train_dates, n_splits=N_CV_SPLITS)

if len(local_splits) < 2:
    raise ValueError("No fue posible construir suficientes folds temporales para ML residual")

fold_cache = []
for fold_no, (tr_local, va_local) in enumerate(local_splits):
    tr_global = train_rows[tr_local]
    va_global = train_rows[va_local]
    features, medians = fit_preprocessor(X_dev, tr_global, candidate_dev)
    fold_cache.append({
        "fold": fold_no,
        "tr_global": tr_global,
        "va_global": va_global,
        "features": features,
        "medians": medians,
        "X_train": transform_matrix(X_dev, tr_global, features, medians),
        "X_valid": transform_matrix(X_dev, va_global, features, medians),
    })

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = f"sqlite:///{DB_PATH.as_posix()}"

sampler_xgb = optuna.samplers.TPESampler(
    seed=RANDOM_STATE,
    multivariate=True,
    n_startup_trials=7,
)
pruner_xgb = optuna.pruners.MedianPruner(
    n_startup_trials=7,
    n_warmup_steps=1,
)

sampler_rf = optuna.samplers.TPESampler(
    seed=RANDOM_STATE + 1,
    multivariate=True,
    n_startup_trials=5,
)
pruner_rf = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=1,
)

study_xgb = optuna.create_study(
    direction="minimize",
    study_name=f"Atitlan_Residual_XGB_{study_suffix}",
    storage=storage,
    load_if_exists=True,
    sampler=sampler_xgb,
    pruner=pruner_xgb,
)

study_rf = optuna.create_study(
    direction="minimize",
    study_name=f"Atitlan_Residual_RF_{study_suffix}",
    storage=storage,
    load_if_exists=True,
    sampler=sampler_rf,
    pruner=pruner_rf,
)


def objective_xgb(trial):
    params = xgb_search_space(trial)
    losses = []

    for fold_info in fold_cache:
        tr = fold_info["tr_global"]
        va = fold_info["va_global"]

        model = make_xgb(params)
        model.fit(
            fold_info["X_train"],
            y_resid_dev[tr],
            sample_weight=w[tr],
            verbose=False,
        )

        resid_pred = model.predict(fold_info["X_valid"])
        delta_pred = delta_phys_dev[va] + resid_pred

        start_level = float(intervals.iloc[va[0]]["level_prev_m"])
        level_pred = trajectory_from_delta(start_level, delta_pred)

        loss = composite_trajectory_loss(
            y_level[va],
            level_pred,
            y_delta[va],
            delta_pred,
            w[va],
        )
        losses.append(float(loss))

        trial.report(float(np.mean(losses)), step=fold_info["fold"])
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(losses))


def objective_rf(trial):
    params = rf_search_space(trial)
    losses = []

    for fold_info in fold_cache:
        tr = fold_info["tr_global"]
        va = fold_info["va_global"]

        model = make_rf(params)
        model.fit(
            fold_info["X_train"],
            y_resid_dev[tr],
            sample_weight=w[tr],
        )

        resid_pred = model.predict(fold_info["X_valid"])
        delta_pred = delta_phys_dev[va] + resid_pred

        start_level = float(intervals.iloc[va[0]]["level_prev_m"])
        level_pred = trajectory_from_delta(start_level, delta_pred)

        loss = composite_trajectory_loss(
            y_level[va],
            level_pred,
            y_delta[va],
            delta_pred,
            w[va],
        )
        losses.append(float(loss))

        trial.report(float(np.mean(losses)), step=fold_info["fold"])
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(losses))


print("=" * 72)
print("04_ML_RESIDUAL - CORRECTOR DEL BALANCE FISICO")
print("=" * 72)
print("SQLite Optuna:", DB_PATH)
print("Predictores exogenos compatibles con futuro:", len(candidate_exog))
print()

done_xgb = completed_or_terminal_trials(study_xgb)
remaining_xgb = max(0, TARGET_TRIALS_XGB - done_xgb)
print(f"XGBoost residual: {done_xgb}/{TARGET_TRIALS_XGB} trials guardados; ejecutando {remaining_xgb}")
if remaining_xgb > 0:
    study_xgb.optimize(
        objective_xgb,
        n_trials=remaining_xgb,
        gc_after_trial=True,
        show_progress_bar=True,
    )

done_rf = completed_or_terminal_trials(study_rf)
remaining_rf = max(0, TARGET_TRIALS_RF - done_rf)
print(f"Random Forest residual: {done_rf}/{TARGET_TRIALS_RF} trials guardados; ejecutando {remaining_rf}")
if remaining_rf > 0:
    study_rf.optimize(
        objective_rf,
        n_trials=remaining_rf,
        gc_after_trial=True,
        show_progress_bar=True,
    )

best_xgb = dict(study_xgb.best_params)
best_rf = dict(study_rf.best_params)

features_train, medians_train = fit_preprocessor(X_dev, train_rows, candidate_dev)
X_train_np = transform_matrix(X_dev, train_rows, features_train, medians_train)
X_cal_np = transform_matrix(X_dev, cal_rows, features_train, medians_train)

xgb_dev = make_xgb(best_xgb)
xgb_dev.fit(
    X_train_np,
    y_resid_dev[train_rows],
    sample_weight=w[train_rows],
    verbose=False,
)

rf_dev = make_rf(best_rf)
rf_dev.fit(
    X_train_np,
    y_resid_dev[train_rows],
    sample_weight=w[train_rows],
)

resid_xgb_cal = xgb_dev.predict(X_cal_np)
resid_rf_cal = rf_dev.predict(X_cal_np)

delta_phys_cal = delta_phys_dev[cal_rows]
delta_xgb_cal = delta_phys_cal + resid_xgb_cal
delta_rf_cal = delta_phys_cal + resid_rf_cal

cal_start_level = float(intervals.iloc[cal_rows[0]]["level_prev_m"])
level_phys_cal = trajectory_from_delta(cal_start_level, delta_phys_cal)
level_xgb_cal = trajectory_from_delta(cal_start_level, delta_xgb_cal)
level_rf_cal = trajectory_from_delta(cal_start_level, delta_rf_cal)
level_obs_cal = y_level[cal_rows]

selection_rows = []

for model_name, level_pred, delta_pred in [
    ("PHYSICAL_ONLY", level_phys_cal, delta_phys_cal),
    ("XGB_RESIDUAL", level_xgb_cal, delta_xgb_cal),
    ("RF_RESIDUAL", level_rf_cal, delta_rf_cal),
]:
    m = metrics(model_name, level_obs_cal, level_pred, w[cal_rows])
    delta_m = metrics(model_name, y_delta[cal_rows], delta_pred, w[cal_rows])
    composite = (
        m["weighted_MAE_m"]
        + 0.50 * abs(m["weighted_bias_m"])
        + 0.25 * delta_m["weighted_MAE_m"]
    )
    selection_rows.append({
        "model": model_name,
        "calibration_composite_score": composite,
        "level_weighted_MAE_m": m["weighted_MAE_m"],
        "level_weighted_bias_m": m["weighted_bias_m"],
        "level_R2": m["R2"],
        "level_Pearson_r": m["Pearson_r"],
        "delta_weighted_MAE_m": delta_m["weighted_MAE_m"],
        "delta_weighted_bias_m": delta_m["weighted_bias_m"],
    })

selection_df = pd.DataFrame(selection_rows).sort_values("calibration_composite_score").reset_index(drop=True)
physical_score = float(
    selection_df.loc[selection_df["model"] == "PHYSICAL_ONLY", "calibration_composite_score"].iloc[0]
)

best_candidate_row = selection_df.loc[selection_df["model"] != "PHYSICAL_ONLY"].iloc[0]
candidate_score = float(best_candidate_row["calibration_composite_score"])
candidate_improvement = 1.0 - candidate_score / physical_score

if candidate_improvement >= MIN_CALIBRATION_SCORE_IMPROVEMENT:
    selected_model = str(best_candidate_row["model"])
else:
    selected_model = "PHYSICAL_ONLY"

selection_df["selected_for_test"] = selection_df["model"].eq(selected_model)
selection_df["score_improvement_vs_physical"] = 1.0 - selection_df["calibration_composite_score"] / physical_score
selection_df.to_csv(RESULT_DIR / "seleccion_corrector_residual.csv", index=False)

feature_train_df = pd.DataFrame({
    "feature": features_train,
    "training_median": [float(medians_train[c]) for c in features_train],
})
feature_train_df.to_csv(RESULT_DIR / "features_residuales_train.csv", index=False)

features_train_cal, medians_train_cal = fit_preprocessor(X_final, train_cal_rows, candidate_final)
X_train_cal_np = transform_matrix(X_final, train_cal_rows, features_train_cal, medians_train_cal)
X_test_np = transform_matrix(X_final, test_rows, features_train_cal, medians_train_cal)

xgb_final = make_xgb(best_xgb)
xgb_final.fit(
    X_train_cal_np,
    y_resid_final[train_cal_rows],
    sample_weight=w[train_cal_rows],
    verbose=False,
)

rf_final = make_rf(best_rf)
rf_final.fit(
    X_train_cal_np,
    y_resid_final[train_cal_rows],
    sample_weight=w[train_cal_rows],
)

xgb_final.save_model(str(MODEL_DIR / "xgboost_residual_final.json"))
joblib.dump(rf_final, MODEL_DIR / "random_forest_residual_final.joblib")

resid_xgb_test = xgb_final.predict(X_test_np)
resid_rf_test = rf_final.predict(X_test_np)

delta_phys_test = delta_phys_final[test_rows]
delta_xgb_test = delta_phys_test + resid_xgb_test
delta_rf_test = delta_phys_test + resid_rf_test

test_start_level = float(intervals.iloc[test_rows[0]]["level_prev_m"])
level_phys_test = trajectory_from_delta(test_start_level, delta_phys_test)
level_xgb_test = trajectory_from_delta(test_start_level, delta_xgb_test)
level_rf_test = trajectory_from_delta(test_start_level, delta_rf_test)
level_obs_test = y_level[test_rows]

if selected_model == "XGB_RESIDUAL":
    level_selected_test = level_xgb_test
    delta_selected_test = delta_xgb_test
elif selected_model == "RF_RESIDUAL":
    level_selected_test = level_rf_test
    delta_selected_test = delta_rf_test
else:
    level_selected_test = level_phys_test
    delta_selected_test = delta_phys_test

test_metric_rows = []
for model_name, level_pred, delta_pred in [
    ("PHYSICAL_ONLY", level_phys_test, delta_phys_test),
    ("XGB_RESIDUAL", level_xgb_test, delta_xgb_test),
    ("RF_RESIDUAL", level_rf_test, delta_rf_test),
    ("SELECTED_MODEL", level_selected_test, delta_selected_test),
]:
    m = metrics(model_name, level_obs_test, level_pred, w[test_rows])
    dm = metrics(model_name, y_delta[test_rows], delta_pred, w[test_rows])
    m["delta_MAE_m"] = dm["MAE_m"]
    m["delta_weighted_MAE_m"] = dm["weighted_MAE_m"]
    m["delta_bias_m"] = dm["bias_m"]
    m["delta_weighted_bias_m"] = dm["weighted_bias_m"]
    if model_name == "SELECTED_MODEL":
        m["selected_identity"] = selected_model
    else:
        m["selected_identity"] = ""
    test_metric_rows.append(m)

test_metrics_df = pd.DataFrame(test_metric_rows)
physical_test_wmae = float(
    test_metrics_df.loc[test_metrics_df["model"] == "PHYSICAL_ONLY", "weighted_MAE_m"].iloc[0]
)
test_metrics_df["weighted_MAE_skill_vs_physical"] = (
    1.0 - test_metrics_df["weighted_MAE_m"] / physical_test_wmae
)
test_metrics_df.to_csv(RESULT_DIR / "metricas_test_ml_residual.csv", index=False)

predictions = pd.DataFrame({
    "date": intervals.iloc[test_rows]["date_obs"].to_numpy(),
    "observed_level_m": level_obs_test,
    "physical_level_m": level_phys_test,
    "xgb_residual_level_m": level_xgb_test,
    "rf_residual_level_m": level_rf_test,
    "selected_level_m": level_selected_test,
    "physical_delta_m": delta_phys_test,
    "xgb_residual_prediction_m": resid_xgb_test,
    "rf_residual_prediction_m": resid_rf_test,
    "xgb_hybrid_delta_m": delta_xgb_test,
    "rf_hybrid_delta_m": delta_rf_test,
    "selected_delta_m": delta_selected_test,
    "observed_delta_m": y_delta[test_rows],
    "interval_weight": w[test_rows],
})
predictions.to_csv(RESULT_DIR / "predicciones_test_ml_residual.csv", index=False)

feature_train_cal_df = pd.DataFrame({
    "feature": features_train_cal,
    "training_calibration_median": [float(medians_train_cal[c]) for c in features_train_cal],
})
feature_train_cal_df.to_csv(RESULT_DIR / "features_residuales_train_cal.csv", index=False)

xgb_booster = xgb_final.get_booster()
dtest = xgb.DMatrix(X_test_np)
shap_matrix = xgb_booster.predict(dtest, pred_contribs=True)
shap_values = shap_matrix[:, :-1]
xgb_importance = pd.DataFrame({
    "feature": features_train_cal,
    "mean_abs_shap_m": np.mean(np.abs(shap_values), axis=0),
    "mean_shap_m": np.mean(shap_values, axis=0),
}).sort_values("mean_abs_shap_m", ascending=False)
xgb_importance.to_csv(RESULT_DIR / "importancia_xgb_residual.csv", index=False)

rf_importance = pd.DataFrame({
    "feature": features_train_cal,
    "importance": rf_final.feature_importances_,
}).sort_values("importance", ascending=False)
rf_importance.to_csv(RESULT_DIR / "importancia_rf_residual.csv", index=False)

study_xgb.trials_dataframe().to_csv(
    RESULT_DIR / "optuna_xgb_residual_trials.csv",
    index=False,
)
study_rf.trials_dataframe().to_csv(
    RESULT_DIR / "optuna_rf_residual_trials.csv",
    index=False,
)

fig = plt.figure(figsize=(13, 6))
plt.plot(predictions["date"], predictions["observed_level_m"], label="Observado", linewidth=2)
plt.plot(predictions["date"], predictions["physical_level_m"], label="Balance fisico", linewidth=1.8)
plt.plot(predictions["date"], predictions["selected_level_m"], label=f"Seleccionado: {selected_model}", linewidth=1.8)
plt.xlabel("Fecha")
plt.ylabel("Nivel EGM2008 (m)")
plt.title("Lago Atitlan - balance fisico vs corrector ML residual")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "ml_residual_test_comparacion.png", dpi=300, bbox_inches="tight")
plt.close(fig)

top_imp = xgb_importance.head(25).sort_values("mean_abs_shap_m", ascending=True)
fig = plt.figure(figsize=(11, 9))
plt.barh(top_imp["feature"], top_imp["mean_abs_shap_m"])
plt.xlabel("Media absoluta SHAP del residual (m)")
plt.ylabel("Variable")
plt.title("XGBoost residual - importancia en TEST")
plt.tight_layout()
plt.savefig(FIG_DIR / "ml_residual_xgb_importancia_top25.png", dpi=300, bbox_inches="tight")
plt.close(fig)

configuration = {
    "script_version": SCRIPT_VERSION,
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "project_root": str(ROOT),
    "method": "Modelo hibrido: balance fisico + ML sobre residual de delta de nivel.",
    "future_compatibility": "Los predictores ML se restringen a variables climaticas reproducibles/derivables con escenarios futuros y componentes fisicos del balance. Se excluyen nivel_lag, presion, humedad del suelo y runoff del corrector residual.",
    "selection_rule": {
        "selection_partition": "calibration",
        "minimum_composite_score_improvement_vs_physical": MIN_CALIBRATION_SCORE_IMPROVEMENT,
        "fallback": "PHYSICAL_ONLY",
        "test_not_used_for_selection": True,
    },
    "cv": {
        "type": "expanding temporal",
        "n_splits_requested": N_CV_SPLITS,
        "n_splits_used": len(local_splits),
        "objective": "trajectory_weighted_MAE + 0.50*abs(weighted_bias) + 0.25*delta_weighted_MAE",
    },
    "optuna": {
        "database": str(DB_PATH),
        "study_suffix": study_suffix,
        "xgb_target_trials": TARGET_TRIALS_XGB,
        "rf_target_trials": TARGET_TRIALS_RF,
        "persistence": "SQLite + load_if_exists=True",
    },
    "best_xgb_parameters": best_xgb,
    "best_rf_parameters": best_rf,
    "selected_model": selected_model,
    "calibration_candidate_improvement_vs_physical": float(candidate_improvement),
    "outputs": {
        "selection": str(RESULT_DIR / "seleccion_corrector_residual.csv"),
        "test_metrics": str(RESULT_DIR / "metricas_test_ml_residual.csv"),
        "test_predictions": str(RESULT_DIR / "predicciones_test_ml_residual.csv"),
        "xgb_importance": str(RESULT_DIR / "importancia_xgb_residual.csv"),
        "rf_importance": str(RESULT_DIR / "importancia_rf_residual.csv"),
        "features_train": str(RESULT_DIR / "features_residuales_train.csv"),
        "features_train_cal": str(RESULT_DIR / "features_residuales_train_cal.csv"),
        "xgb_model": str(MODEL_DIR / "xgboost_residual_final.json"),
        "rf_model": str(MODEL_DIR / "random_forest_residual_final.joblib"),
    },
}

CONFIG_OUT.write_text(
    json.dumps(configuration, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
STATE_PATH.write_text(
    json.dumps(fingerprint, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("=" * 72)
print("04_ML_RESIDUAL TERMINADO")
print("=" * 72)
print("Predictores climaticos/exogenos candidatos:", len(candidate_exog))
print("Features finales TRAIN:", len(features_train))
print("Features finales TRAIN+CAL:", len(features_train_cal))
print()
print("Mejor CV:")
print("XGB residual:", round(float(study_xgb.best_value), 6))
print("RF residual :", round(float(study_rf.best_value), 6))
print()
print("Seleccion en CALIBRACION:")
print(selection_df.to_string(index=False))
print()
print("MODELO SELECCIONADO:", selected_model)
print("Mejora de score del mejor corrector vs balance:", round(float(candidate_improvement), 6))
print()
print("Metricas finales TEST:")
print(test_metrics_df.to_string(index=False))
print()
print("SQLite Optuna:", DB_PATH)
print("Resultados:", RESULT_DIR)
print("Configuracion:", CONFIG_OUT)
print("=" * 72)
