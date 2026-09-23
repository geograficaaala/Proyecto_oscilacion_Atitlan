__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy.optimize import least_squares
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.optimize import least_squares

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_VERSION = "1.0.0"
RANDOM_STATE = 42
TRAIN_TIME_FRACTION = 0.70
CAL_TIME_FRACTION = 0.15
LAKE_AREA_KM2 = 125.77
LAKE_AREA_M2 = LAKE_AREA_KM2 * 1_000_000.0
SECONDS_PER_DAY = 86400.0
F_SCALE_M = 0.15

ROOT = Path(__file__).resolve().parents[1]
DATASETS_CONFIG = ROOT / "config" / "datasets_procesados.json"
FLOW_CONFIG = ROOT / "config" / "modelo_caudales.json"
RESULT_DIR = ROOT / "outputs" / "balance_hidrico"
FIG_DIR = ROOT / "outputs" / "figuras"
CACHE_DIR = ROOT / "data" / "cache"
STATE_PATH = CACHE_DIR / "estado_03_balance_hidrico.json"
CONFIG_OUT = ROOT / "config" / "balance_hidrico.json"

for d in [RESULT_DIR, FIG_DIR, CACHE_DIR, ROOT / "config"]:
    d.mkdir(parents=True, exist_ok=True)

if not DATASETS_CONFIG.exists():
    raise FileNotFoundError("Falta config/datasets_procesados.json. Ejecuta primero 01_preparar_datos.py")
if not FLOW_CONFIG.exists():
    raise FileNotFoundError("Falta config/modelo_caudales.json. Ejecuta primero 02_modelar_caudales.py")

with open(DATASETS_CONFIG, "r", encoding="utf-8") as f:
    DATASETS = json.load(f)

with open(FLOW_CONFIG, "r", encoding="utf-8") as f:
    FLOW_CFG = json.load(f)


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
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wmae = float(np.average(np.abs(y_true - y_pred), weights=w))
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


def interval_weight_from_errors(err_prev, err_curr, fallback_weight=1.0):
    if np.isfinite(err_prev) and err_prev > 0 and np.isfinite(err_curr) and err_curr > 0:
        sigma = np.sqrt(err_prev ** 2 + err_curr ** 2)
        return 1.0 / (sigma ** 2)
    return float(fallback_weight) if np.isfinite(fallback_weight) and fallback_weight > 0 else 1.0


def overlap_fraction(interval_start, interval_end, day):
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    start = max(interval_start, day_start)
    end = min(interval_end, day_end)
    if end <= start:
        return 0.0
    return float((end - start).total_seconds() / SECONDS_PER_DAY)


def build_interval_components(matrix, climate, q_dev, q_final, idx_train, idx_cal, idx_test):
    climate = climate.copy()
    climate.columns = climate.columns.astype(str).str.strip()
    climate["date"] = pd.to_datetime(climate["date"], errors="coerce").dt.normalize()
    climate = climate.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").set_index("date")

    required_climate = ["era_precip_mm", "chirps_precip_mm", "evap_mm", "pet_mm"]
    missing = [c for c in required_climate if c not in climate.columns]
    if missing:
        raise KeyError("Faltan variables climaticas para el balance: " + str(missing))

    for col in required_climate:
        climate[col] = pd.to_numeric(climate[col], errors="coerce")

    q_dev = q_dev.copy()
    q_final = q_final.copy()
    q_dev["date"] = pd.to_datetime(q_dev["date"], errors="coerce").dt.normalize()
    q_final["date"] = pd.to_datetime(q_final["date"], errors="coerce").dt.normalize()

    if "q_pred_dev_m3s" not in q_dev.columns:
        raise KeyError("caudal_diario_estimado_desarrollo no contiene q_pred_dev_m3s")
    if "q_pred_final_m3s" not in q_final.columns:
        raise KeyError("caudal_diario_estimado_final no contiene q_pred_final_m3s")

    q_dev["q_pred_dev_m3s"] = pd.to_numeric(q_dev["q_pred_dev_m3s"], errors="coerce")
    q_final["q_pred_final_m3s"] = pd.to_numeric(q_final["q_pred_final_m3s"], errors="coerce")
    q_dev = q_dev.dropna(subset=["date"]).drop_duplicates("date").set_index("date")
    q_final = q_final.dropna(subset=["date"]).drop_duplicates("date").set_index("date")

    train_set = set(np.asarray(idx_train, dtype=int).tolist())
    cal_set = set(np.asarray(idx_cal, dtype=int).tolist())
    test_set = set(np.asarray(idx_test, dtype=int).tolist())

    rows = []

    for i in range(1, len(matrix)):
        prev_dt = pd.Timestamp(matrix.loc[i - 1, "date_obs"])
        curr_dt = pd.Timestamp(matrix.loc[i, "date_obs"])
        if curr_dt <= prev_dt:
            continue

        partition = "train" if i in train_set else "calibration" if i in cal_set else "test" if i in test_set else "unknown"
        duration_days = float((curr_dt - prev_dt).total_seconds() / SECONDS_PER_DAY)

        p_era_mm = 0.0
        p_chirps_mm = 0.0
        evap_mm = 0.0
        pet_mm = 0.0
        q_dev_volume_m3 = 0.0
        q_final_volume_m3 = 0.0
        coverage_days_climate = 0.0
        coverage_days_q_dev = 0.0
        coverage_days_q_final = 0.0

        for day in pd.date_range(prev_dt.normalize(), curr_dt.normalize(), freq="D"):
            frac = overlap_fraction(prev_dt, curr_dt, day)
            if frac <= 0:
                continue

            if day in climate.index:
                row_c = climate.loc[day]
                vals = [row_c[c] for c in required_climate]
                if all(np.isfinite(v) for v in vals):
                    p_era_mm += float(row_c["era_precip_mm"]) * frac
                    p_chirps_mm += float(row_c["chirps_precip_mm"]) * frac
                    evap_mm += float(row_c["evap_mm"]) * frac
                    pet_mm += float(row_c["pet_mm"]) * frac
                    coverage_days_climate += frac

            if day in q_dev.index:
                qv = q_dev.loc[day, "q_pred_dev_m3s"]
                if np.isfinite(qv):
                    q_dev_volume_m3 += float(qv) * SECONDS_PER_DAY * frac
                    coverage_days_q_dev += frac

            if day in q_final.index:
                qv = q_final.loc[day, "q_pred_final_m3s"]
                if np.isfinite(qv):
                    q_final_volume_m3 += float(qv) * SECONDS_PER_DAY * frac
                    coverage_days_q_final += frac

        p_consensus_mm = 0.5 * (p_era_mm + p_chirps_mm)
        q_dev_equiv_m = q_dev_volume_m3 / LAKE_AREA_M2
        q_final_equiv_m = q_final_volume_m3 / LAKE_AREA_M2

        level_prev = float(matrix.loc[i - 1, "nivel_final_m"])
        level_curr = float(matrix.loc[i, "nivel_final_m"])
        delta_h = level_curr - level_prev

        err_prev = pd.to_numeric(pd.Series([matrix.loc[i - 1, "estimated_error_m"]]), errors="coerce").iloc[0] if "estimated_error_m" in matrix.columns else np.nan
        err_curr = pd.to_numeric(pd.Series([matrix.loc[i, "estimated_error_m"]]), errors="coerce").iloc[0] if "estimated_error_m" in matrix.columns else np.nan
        fallback = float(matrix.loc[i, "sample_weight"]) if "sample_weight" in matrix.columns and np.isfinite(matrix.loc[i, "sample_weight"]) else 1.0
        raw_weight = interval_weight_from_errors(err_prev, err_curr, fallback)

        rows.append({
            "row_index": i,
            "date_prev": prev_dt,
            "date_obs": curr_dt,
            "partition": partition,
            "duration_days": duration_days,
            "level_prev_m": level_prev,
            "level_obs_m": level_curr,
            "delta_h_obs_m": delta_h,
            "p_era_m": p_era_mm / 1000.0,
            "p_chirps_m": p_chirps_mm / 1000.0,
            "p_consensus_m": p_consensus_mm / 1000.0,
            "evap_m": evap_mm / 1000.0,
            "pet_m": pet_mm / 1000.0,
            "q_in_dev_volume_m3": q_dev_volume_m3,
            "q_in_final_volume_m3": q_final_volume_m3,
            "q_in_dev_equiv_m": q_dev_equiv_m,
            "q_in_final_equiv_m": q_final_equiv_m,
            "coverage_climate_fraction": coverage_days_climate / duration_days if duration_days > 0 else np.nan,
            "coverage_q_dev_fraction": coverage_days_q_dev / duration_days if duration_days > 0 else np.nan,
            "coverage_q_final_fraction": coverage_days_q_final / duration_days if duration_days > 0 else np.nan,
            "interval_weight_raw": raw_weight,
        })

    out = pd.DataFrame(rows)
    med = out["interval_weight_raw"].median()
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    out["interval_weight"] = (out["interval_weight_raw"] / med).clip(0.10, 10.0)
    return out


def predict_balance(params, table, q_col):
    ap, ae, aq, cday = [float(x) for x in params]
    return (
        ap * table["p_consensus_m"].to_numpy(dtype=float)
        - ae * table["evap_m"].to_numpy(dtype=float)
        + aq * table[q_col].to_numpy(dtype=float)
        + cday * table["duration_days"].to_numpy(dtype=float)
    )


def fit_balance(table, q_col):
    y = table["delta_h_obs_m"].to_numpy(dtype=float)
    w = table["interval_weight"].to_numpy(dtype=float)
    p = table["p_consensus_m"].to_numpy(dtype=float)
    e = table["evap_m"].to_numpy(dtype=float)
    q = table[q_col].to_numpy(dtype=float)
    dt = table["duration_days"].to_numpy(dtype=float)

    valid = np.isfinite(y) & np.isfinite(w) & np.isfinite(p) & np.isfinite(e) & np.isfinite(q) & np.isfinite(dt)
    y = y[valid]
    w = w[valid]
    p = p[valid]
    e = e[valid]
    q = q[valid]
    dt = dt[valid]

    if len(y) < 50:
        raise ValueError("Muy pocos intervalos para calibrar el balance hidrico")

    def residual(theta):
        ap, ae, aq, cday = theta
        pred = ap * p - ae * e + aq * q + cday * dt
        return np.sqrt(w) * (pred - y)

    result = least_squares(
        residual,
        x0=np.array([1.0, 1.0, 1.0, 0.0], dtype=float),
        bounds=(
            np.array([0.0, 0.0, 0.0, -0.01], dtype=float),
            np.array([3.0, 3.0, 10.0, 0.01], dtype=float),
        ),
        loss="soft_l1",
        f_scale=F_SCALE_M,
        max_nfev=20000,
    )

    return {
        "precip_scale": float(result.x[0]),
        "evap_scale": float(result.x[1]),
        "qin_scale": float(result.x[2]),
        "net_unresolved_m_per_day": float(result.x[3]),
        "success": bool(result.success),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "n_fit": int(len(y)),
    }


def trajectory_from_deltas(table, delta_col, start_level):
    levels = []
    current = float(start_level)
    for value in table[delta_col].to_numpy(dtype=float):
        current += float(value)
        levels.append(current)
    return np.asarray(levels, dtype=float)


MATRIX_PATH = resolve_dataset("matrix_autoregressive")
CLIMATE_PATH = resolve_dataset("climate_daily_features")

Q_DEV_PATH = Path(FLOW_CFG["outputs"]["daily_development"])
Q_FINAL_PATH = Path(FLOW_CFG["outputs"]["daily_final"])

for p in [MATRIX_PATH, CLIMATE_PATH, Q_DEV_PATH, Q_FINAL_PATH]:
    if not p.exists():
        raise FileNotFoundError(p)

fingerprint = {
    "script_version": SCRIPT_VERSION,
    "lake_area_km2": LAKE_AREA_KM2,
    "matrix_sha256": sha256_file(MATRIX_PATH),
    "climate_sha256": sha256_file(CLIMATE_PATH),
    "q_dev_sha256": sha256_file(Q_DEV_PATH),
    "q_final_sha256": sha256_file(Q_FINAL_PATH),
}

expected_outputs = [
    RESULT_DIR / "balance_componentes_intervalos.csv",
    RESULT_DIR / "metricas_balance_hidrico.csv",
    RESULT_DIR / "trayectorias_balance.csv",
    RESULT_DIR / "parametros_balance_desarrollo.json",
    RESULT_DIR / "parametros_balance_final.json",
    CONFIG_OUT,
]

if STATE_PATH.exists() and all(p.exists() for p in expected_outputs):
    try:
        old_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        old_state = None
    if old_state == fingerprint:
        print("=" * 72)
        print("03_BALANCE_HIDRICO: SIN CAMBIOS")
        print("=" * 72)
        print("Los datos y productos no cambiaron.")
        print("No se recalibro el balance.")
        print(RESULT_DIR)
        sys.exit(0)

matrix = read_table(MATRIX_PATH).copy()
matrix.columns = matrix.columns.astype(str).str.strip()
matrix["date_obs"] = pd.to_datetime(matrix["date_obs"], errors="coerce")
matrix["nivel_final_m"] = pd.to_numeric(matrix["nivel_final_m"], errors="coerce")
matrix = matrix.dropna(subset=["date_obs", "nivel_final_m"]).sort_values("date_obs").reset_index(drop=True)

climate = read_table(CLIMATE_PATH)
q_dev = read_table(Q_DEV_PATH)
q_final = read_table(Q_FINAL_PATH)

idx_train, idx_cal, idx_test = temporal_partition(matrix["date_obs"])

intervals = build_interval_components(
    matrix=matrix,
    climate=climate,
    q_dev=q_dev,
    q_final=q_final,
    idx_train=idx_train,
    idx_cal=idx_cal,
    idx_test=idx_test,
)

if (intervals["coverage_climate_fraction"] < 0.999).any():
    bad = intervals.loc[intervals["coverage_climate_fraction"] < 0.999, ["date_prev", "date_obs", "coverage_climate_fraction"]]
    raise ValueError("Cobertura climatica incompleta en intervalos:\n" + bad.head(20).to_string(index=False))

if (intervals["coverage_q_dev_fraction"] < 0.999).any():
    bad = intervals.loc[intervals["coverage_q_dev_fraction"] < 0.999, ["date_prev", "date_obs", "coverage_q_dev_fraction"]]
    raise ValueError("Cobertura incompleta del caudal de desarrollo:\n" + bad.head(20).to_string(index=False))

if (intervals["coverage_q_final_fraction"] < 0.999).any():
    bad = intervals.loc[intervals["coverage_q_final_fraction"] < 0.999, ["date_prev", "date_obs", "coverage_q_final_fraction"]]
    raise ValueError("Cobertura incompleta del caudal final:\n" + bad.head(20).to_string(index=False))

train_intervals = intervals.loc[intervals["partition"] == "train"].copy()
train_cal_intervals = intervals.loc[intervals["partition"].isin(["train", "calibration"])].copy()

dev_params = fit_balance(train_intervals, "q_in_dev_equiv_m")
final_params = fit_balance(train_cal_intervals, "q_in_final_equiv_m")

theta_dev = [
    dev_params["precip_scale"],
    dev_params["evap_scale"],
    dev_params["qin_scale"],
    dev_params["net_unresolved_m_per_day"],
]
theta_final = [
    final_params["precip_scale"],
    final_params["evap_scale"],
    final_params["qin_scale"],
    final_params["net_unresolved_m_per_day"],
]

intervals["delta_balance_raw_dev_m"] = (
    intervals["p_consensus_m"]
    - intervals["evap_m"]
    + intervals["q_in_dev_equiv_m"]
)
intervals["delta_balance_calibrated_dev_m"] = predict_balance(theta_dev, intervals, "q_in_dev_equiv_m")
intervals["delta_balance_raw_final_m"] = (
    intervals["p_consensus_m"]
    - intervals["evap_m"]
    + intervals["q_in_final_equiv_m"]
)
intervals["delta_balance_calibrated_final_m"] = predict_balance(theta_final, intervals, "q_in_final_equiv_m")

intervals["residual_balance_dev_m"] = intervals["delta_h_obs_m"] - intervals["delta_balance_calibrated_dev_m"]
intervals["residual_balance_final_m"] = intervals["delta_h_obs_m"] - intervals["delta_balance_calibrated_final_m"]

intervals["delta_balance_for_stage04_m"] = np.where(
    intervals["partition"].eq("test"),
    intervals["delta_balance_calibrated_final_m"],
    intervals["delta_balance_calibrated_dev_m"],
)
intervals["residual_for_stage04_m"] = intervals["delta_h_obs_m"] - intervals["delta_balance_for_stage04_m"]

intervals.to_csv(RESULT_DIR / "balance_componentes_intervalos.csv", index=False)
intervals.to_pickle(RESULT_DIR / "balance_componentes_intervalos.pkl")

dev_params["lake_area_km2"] = LAKE_AREA_KM2
dev_params["fit_partition"] = "train"
dev_params["q_source"] = "caudal_diario_estimado_desarrollo"
final_params["lake_area_km2"] = LAKE_AREA_KM2
final_params["fit_partition"] = "train+calibration"
final_params["q_source"] = "caudal_diario_estimado_final"

(RESULT_DIR / "parametros_balance_desarrollo.json").write_text(
    json.dumps(dev_params, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(RESULT_DIR / "parametros_balance_final.json").write_text(
    json.dumps(final_params, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

metric_rows = []

for partition, pred_col, raw_col in [
    ("calibration", "delta_balance_calibrated_dev_m", "delta_balance_raw_dev_m"),
    ("test", "delta_balance_calibrated_final_m", "delta_balance_raw_final_m"),
]:
    sub = intervals.loc[intervals["partition"] == partition].copy()
    if len(sub):
        metric_rows.append({
            "evaluation": f"delta_{partition}",
            **metrics("PhysicalBalance_Calibrated", sub["delta_h_obs_m"], sub[pred_col], sub["interval_weight"]),
        })
        metric_rows.append({
            "evaluation": f"delta_{partition}",
            **metrics("PhysicalBalance_Raw", sub["delta_h_obs_m"], sub[raw_col], sub["interval_weight"]),
        })
        metric_rows.append({
            "evaluation": f"delta_{partition}",
            **metrics("ZeroChange_Persistence", sub["delta_h_obs_m"], np.zeros(len(sub)), sub["interval_weight"]),
        })

trajectory_frames = []

cal_rows = intervals.loc[intervals["partition"] == "calibration"].copy()
if len(cal_rows):
    first_cal_index = int(cal_rows.iloc[0]["row_index"])
    cal_start_level = float(matrix.loc[first_cal_index - 1, "nivel_final_m"])
    cal_rows["level_balance_m"] = trajectory_from_deltas(cal_rows, "delta_balance_calibrated_dev_m", cal_start_level)
    cal_rows["level_frozen_m"] = cal_start_level
    cal_rows["evaluation"] = "calibration"
    trajectory_frames.append(cal_rows[["evaluation", "date_obs", "level_obs_m", "level_balance_m", "level_frozen_m"]])
    metric_rows.append({
        "evaluation": "level_recursive_calibration",
        **metrics("PhysicalBalance_Calibrated", cal_rows["level_obs_m"], cal_rows["level_balance_m"]),
    })
    metric_rows.append({
        "evaluation": "level_recursive_calibration",
        **metrics("Frozen_Last_Level", cal_rows["level_obs_m"], cal_rows["level_frozen_m"]),
    })

test_rows = intervals.loc[intervals["partition"] == "test"].copy()
if len(test_rows):
    first_test_index = int(test_rows.iloc[0]["row_index"])
    test_start_level = float(matrix.loc[first_test_index - 1, "nivel_final_m"])
    test_rows["level_balance_m"] = trajectory_from_deltas(test_rows, "delta_balance_calibrated_final_m", test_start_level)
    test_rows["level_frozen_m"] = test_start_level
    test_rows["evaluation"] = "test"
    trajectory_frames.append(test_rows[["evaluation", "date_obs", "level_obs_m", "level_balance_m", "level_frozen_m"]])
    metric_rows.append({
        "evaluation": "level_recursive_test",
        **metrics("PhysicalBalance_Calibrated", test_rows["level_obs_m"], test_rows["level_balance_m"]),
    })
    metric_rows.append({
        "evaluation": "level_recursive_test",
        **metrics("Frozen_Last_Level", test_rows["level_obs_m"], test_rows["level_frozen_m"]),
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(RESULT_DIR / "metricas_balance_hidrico.csv", index=False)

trajectories = pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
trajectories.to_csv(RESULT_DIR / "trayectorias_balance.csv", index=False)

if len(trajectories):
    for evaluation in ["calibration", "test"]:
        sub = trajectories.loc[trajectories["evaluation"] == evaluation].copy()
        if not len(sub):
            continue
        fig = plt.figure(figsize=(12, 6))
        plt.plot(sub["date_obs"], sub["level_obs_m"], label="Observado", linewidth=2)
        plt.plot(sub["date_obs"], sub["level_balance_m"], label="Balance hidrico", linewidth=1.8)
        plt.plot(sub["date_obs"], sub["level_frozen_m"], label="Nivel inicial congelado", linestyle="--")
        plt.xlabel("Fecha")
        plt.ylabel("Nivel EGM2008 (m)")
        plt.title(f"Balance hidrico recursivo - {evaluation}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"balance_hidrico_{evaluation}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

configuration = {
    "script_version": SCRIPT_VERSION,
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "project_root": str(ROOT),
    "lake_area_km2_reference": LAKE_AREA_KM2,
    "lake_area_note": "Area fija de referencia temporal para la fase hidrologica. La relacion nivel-area dinamica se incorporara posteriormente con batimetria/DEM.",
    "formula": "deltaH = aP*P_consensus - aE*Evap + aQ*Qin_equiv + c_day*delta_t",
    "precipitation_note": "P_consensus es el promedio ERA5-Land/CHIRPS sobre la cuenca disponible; aP absorbe parcialmente diferencias de representatividad respecto a precipitacion directa sobre el lago.",
    "evaporation_note": "Evap corresponde a ERA5-Land procesado. Para escenarios CMIP6 se requerira derivar evaporacion/PET compatible a partir de variables climaticas futuras.",
    "qin_note": "Qin es la suma modelada de los cinco tributarios monitoreados; aQ permite compensar parcialmente tributarios no monitoreados y sesgo del submodelo, sin interpretarse como caudal fisico observado total.",
    "unresolved_note": "c_day representa flujo neto no resuelto y sesgo medio del balance, incluyendo potencial intercambio subterraneo/percolacion y errores de componentes. No debe interpretarse aisladamente como medicion de agua subterranea.",
    "calibration_method": "scipy.optimize.least_squares con loss soft_l1, restricciones de signo para P, E y Qin y pesos derivados de la incertidumbre de dos observaciones consecutivas.",
    "development_parameters": dev_params,
    "final_parameters": final_params,
    "outputs": {
        "intervals": str(RESULT_DIR / "balance_componentes_intervalos.pkl"),
        "metrics": str(RESULT_DIR / "metricas_balance_hidrico.csv"),
        "trajectories": str(RESULT_DIR / "trayectorias_balance.csv"),
        "development_parameters": str(RESULT_DIR / "parametros_balance_desarrollo.json"),
        "final_parameters": str(RESULT_DIR / "parametros_balance_final.json"),
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

print("=" * 72)
print("03_BALANCE_HIDRICO TERMINADO")
print("=" * 72)
print("Area de referencia del lago:", LAKE_AREA_KM2, "km2")
print()
print("Parametros DESARROLLO (ajuste solo TRAIN):")
print("  Escala precipitacion:", round(dev_params["precip_scale"], 6))
print("  Escala evaporacion:", round(dev_params["evap_scale"], 6))
print("  Escala caudal:", round(dev_params["qin_scale"], 6))
print("  Flujo neto no resuelto:", round(dev_params["net_unresolved_m_per_day"], 8), "m/dia")
print()
print("Parametros FINAL (ajuste TRAIN+CALIBRATION):")
print("  Escala precipitacion:", round(final_params["precip_scale"], 6))
print("  Escala evaporacion:", round(final_params["evap_scale"], 6))
print("  Escala caudal:", round(final_params["qin_scale"], 6))
print("  Flujo neto no resuelto:", round(final_params["net_unresolved_m_per_day"], 8), "m/dia")
print()
print("Metricas:")
print(metrics_df.to_string(index=False))
print()
print("Resultados:", RESULT_DIR)
print("Configuracion:", CONFIG_OUT)
print("=" * 72)
