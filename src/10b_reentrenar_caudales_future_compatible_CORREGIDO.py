                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import re
import hashlib
import sys
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

ROOT = Path(__file__).resolve().parents[1]

FEATURES_CSV = ROOT / "outputs" / "caudales" / "clima_diario_features_modelo_caudal.csv"
FEATURES_PKL = ROOT / "outputs" / "caudales" / "clima_diario_features_modelo_caudal.pkl"

RIVERS_CSV = ROOT / "data" / "processed" / "rivers_clean.csv"
RIVERS_PKL = ROOT / "data" / "processed" / "rivers_clean.pkl"

CONFIG_FILE = ROOT / "config" / "modelo_caudales.json"
FEATURE_LIST_FILE = (
    ROOT / "outputs" / "caudales_future_compatible"
    / "10a_features_future_compatible_strict.json"
)

BASELINE_RIVER_METRICS = (
    ROOT / "outputs" / "caudales"
    / "metricas_caudal_por_rio_validacion_temporal.csv"
)

OUT_DIR = ROOT / "outputs" / "caudales_future_compatible"
MODEL_DIR = ROOT / "models" / "caudales_future_compatible"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RIVERS = [
    "Quiscab",
    "San_Francisco",
    "Tzununa",
    "La_Catarata",
    "San_Buenaventura",
]

EXPECTED_FEATURES = 72
RANDOM_STATE = 42


def norm(x):
    s = str(x).strip().lower()
    for a, b in [
        ("á", "a"), ("é", "e"), ("í", "i"),
        ("ó", "o"), ("ú", "u"), ("ñ", "n")
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


def load_dataframe(pkl_path, csv_path):
    if pkl_path.exists():
        obj = pd.read_pickle(pkl_path)
        if isinstance(obj, pd.DataFrame):
            return obj.copy(), pkl_path
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path
    raise FileNotFoundError(f"No existe {pkl_path} ni {csv_path}")


def river_key_from_config(config, river):
    models = config.get("river_models", {})
    nr = norm(river)

    for key in models:
        if norm(key) == nr:
            return key

    aliases = {
        "quiscab": ["quiscab"],
        "san_francisco": ["san_francisco", "sanfrancisco"],
        "tzununa": ["tzununa"],
        "la_catarata": ["la_catarata", "catarata"],
        "san_buenaventura": ["san_buenaventura", "sanbuenaventura", "buenaventura"],
    }

    for key in models:
        nk = norm(key)
        if any(a in nk for a in aliases[nr]):
            return key

    raise KeyError(f"No encontré {river} dentro de config['river_models'].")


def extract_family_and_params(config, river):
    key = river_key_from_config(config, river)
    info = config["river_models"][key]

    family = None
    for k in ["family", "best_family", "model_family", "modelo", "model"]:
        if k in info and isinstance(info[k], str):
            family = info[k]
            break

    if family is None:
        txt = json.dumps(info, ensure_ascii=False).upper()
        if "XGB_TWEEDIE" in txt:
            family = "XGB_TWEEDIE"
        elif "RF_LOG" in txt:
            family = "RF_LOG"

    if family is None:
        raise ValueError(f"No pude identificar la familia de {river}.")

    family = family.upper()

    params = info.get("best_params")
    if not isinstance(params, dict):
        raise ValueError(f"No encontré best_params para {river}.")

    return family, dict(params), key


def clean_params(params):
    out = {}
    for k, v in params.items():
                                                      
        if v is None:
            continue
        out[k] = v
    return out


def build_estimator(family, params):
    params = clean_params(params)

    if family == "RF_LOG":
        base = {
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        base.update(params)
        return RandomForestRegressor(**base)

    if family == "XGB_TWEEDIE":
        if XGBRegressor is None:
            raise ImportError("XGBoost no está disponible en este entorno.")

        base = {
            "objective": "reg:tweedie",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        base.update(params)
        return XGBRegressor(**base)

    raise ValueError(f"Familia no soportada: {family}")


def normalize_river_value(x):
    n = norm(x)

    if "quiscab" in n:
        return "Quiscab"
    if "san_francisco" in n or "sanfrancisco" in n:
        return "San_Francisco"
    if "tzununa" in n:
        return "Tzununa"
    if "catarata" in n:
        return "La_Catarata"
    if "buenaventura" in n:
        return "San_Buenaventura"

    return None


def resolve_feature_columns(df, features):
    cmap = {norm(c): c for c in df.columns}
    resolved = {}
    missing = []

    for f in features:
        nf = norm(f)
        if nf in cmap:
            resolved[f] = cmap[nf]
        else:
            missing.append(f)

    return resolved, missing


def prepare_X(df, features, medians=None):
    mapping, missing = resolve_feature_columns(df, features)
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} features. Primeras: {missing[:15]}"
        )

    X = pd.DataFrame(index=df.index)

    for f in features:
        X[f] = pd.to_numeric(df[mapping[f]], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = X.median(axis=0, numeric_only=True)

    bad_medians = medians[medians.isna()]
    if len(bad_medians):
        raise ValueError(
            "Hay features sin mediana válida: "
            + ", ".join(map(str, bad_medians.index[:15]))
        )

    X = X.fillna(medians)

    if X.isna().any().any():
        cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Persisten NaN en X: {cols[:15]}")

    return X.astype(float), medians.astype(float)


def metrics(y_true, y_pred, training_y):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    tr = np.asarray(training_y, dtype=float)

    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        raise ValueError("No hay observaciones válidas para métricas.")

    q75 = float(np.quantile(tr[np.isfinite(tr)], 0.75))
    high = y >= q75

    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
        "r2_nse": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "pearson_r": np.nan,
        "highflow_threshold_train_q75": q75,
        "highflow_mae": np.nan,
    }

    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson_r"] = float(np.corrcoef(y, p)[0, 1])

    if high.any():
        out["highflow_mae"] = float(mean_absolute_error(y[high], p[high]))

    return out


def fit_model(family, estimator, X, y):
    y = np.asarray(y, dtype=float)

    if family == "RF_LOG":
        if np.any(y < 0):
            raise ValueError("RF_LOG encontró caudales negativos.")
        estimator.fit(X, np.log1p(y))
        return estimator

    if family == "XGB_TWEEDIE":
        estimator.fit(X, np.maximum(y, 0.0))
        return estimator

    raise ValueError(f"Familia no soportada: {family}")


def predict_model(family, estimator, X):
    pred = np.asarray(estimator.predict(X), dtype=float)

    if family == "RF_LOG":
        pred = np.expm1(pred)

    return np.maximum(pred, 0.0)


def baseline_metrics_by_river():
    if not BASELINE_RIVER_METRICS.exists():
        return {}

    df = pd.read_csv(BASELINE_RIVER_METRICS)
    nmap = {norm(c): c for c in df.columns}

    river_col = nmap.get("river")
    if river_col is None:
        return {}

    result = {}

    for _, row in df.iterrows():
        river = normalize_river_value(row[river_col])
        if river is None:
            continue

        vals = {}
        for canonical, candidates in {
            "baseline_mae": ["mae_m3s", "mae"],
            "baseline_rmse": ["rmse_m3s", "rmse"],
            "baseline_bias": ["bias_m3s", "bias"],
            "baseline_r2_nse": ["r2", "nse", "r2_nse"],
            "baseline_pearson_r": ["pearson_r", "r"],
            "baseline_highflow_mae": ["highflow_mae_m3s", "highflow_mae"],
        }.items():
            for cand in candidates:
                if cand in nmap:
                    try:
                        vals[canonical] = float(row[nmap[cand]])
                    except Exception:
                        vals[canonical] = np.nan
                    break

        result[river] = vals

    return result


def main():
    print("=" * 92)
    print("PASO 10B - CAUDALES FUTURE-COMPATIBLE, VERSION CORREGIDA")
    print("=" * 92)
    print()
    print("Usa exactamente los hallazgos del diagnóstico 10B0:")
    print(" - target: datos/procesados/rivers_clean -> flow_m3s")
    print(" - unión por fecha con clima_diario_features_modelo_caudal")
    print(" - development_cutoff y final_cutoff desde modelo_caudales.json")
    print(" - 72 predictores del PASO 10A")
    print(" - mismas familias y best_params")
    print(" - NO Optuna")
    print(" - NO sobrescribe baseline")
    print()

    for p in [CONFIG_FILE, FEATURE_LIST_FILE]:
        if not p.exists():
            raise FileNotFoundError(p)

    config = load_json(CONFIG_FILE)
    feature_sets = load_json(FEATURE_LIST_FILE)

    dev_cutoff = pd.Timestamp(config["development_cutoff"]).normalize()
    final_cutoff = pd.Timestamp(config["final_cutoff"]).normalize()

    print(f"Cutoff desarrollo: {dev_cutoff.date()}")
    print(f"Cutoff final:      {final_cutoff.date()}")
    print()

    climate, climate_path = load_dataframe(FEATURES_PKL, FEATURES_CSV)
    rivers, rivers_path = load_dataframe(RIVERS_PKL, RIVERS_CSV)

    if "date" not in climate.columns:
        raise KeyError("La tabla climática no contiene columna 'date'.")
    if "date" not in rivers.columns:
        raise KeyError("rivers_clean no contiene columna 'date'.")
    if "river" not in rivers.columns:
        raise KeyError("rivers_clean no contiene columna 'river'.")
    if "flow_m3s" not in rivers.columns:
        raise KeyError("rivers_clean no contiene columna 'flow_m3s'.")

    climate["date"] = pd.to_datetime(climate["date"], errors="raise").dt.normalize()
    rivers["date"] = pd.to_datetime(rivers["date"], errors="raise").dt.normalize()
    rivers["river_fc"] = rivers["river"].apply(normalize_river_value)

    unknown = rivers["river_fc"].isna()
    if unknown.any():
        bad = sorted(rivers.loc[unknown, "river"].astype(str).unique())
        warnings.warn(
            f"Se ignorarán {int(unknown.sum())} filas de ríos no reconocidos: {bad}"
        )

    rivers = rivers.loc[~unknown].copy()

    if climate["date"].duplicated().any():
        raise ValueError("La tabla climática tiene fechas duplicadas.")

                                                                                   
    table = rivers.merge(
        climate,
        on="date",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_clim"),
    )

    if table.empty:
        raise RuntimeError("El merge entre aforos y clima quedó vacío.")

    n_lost = len(rivers) - len(table)
    print(f"Clima:  {len(climate):,} filas")
    print(f"Aforos: {len(rivers):,} filas reconocidas")
    print(f"Merge:  {len(table):,} filas")
    print(f"Aforos sin match climático: {n_lost}")
    print()

    baseline = baseline_metrics_by_river()

    metric_rows = []
    prediction_rows = []
    manifest_models = {}

    for i, river in enumerate(RIVERS, 1):
        print(f"[{i}/5] {river}")

        features = feature_sets.get(river)
        if features is None:
            raise KeyError(f"No existe lista future-compatible para {river}.")
        if len(features) != EXPECTED_FEATURES:
            raise ValueError(
                f"{river}: {len(features)} features; se esperaban {EXPECTED_FEATURES}."
            )

        family, best_params, config_key = extract_family_and_params(config, river)

        rt = (
            table.loc[table["river_fc"].eq(river)]
            .sort_values("date")
            .reset_index(drop=True)
        )

        rt["flow_m3s"] = pd.to_numeric(rt["flow_m3s"], errors="coerce")
        rt = rt.loc[np.isfinite(rt["flow_m3s"])].copy()

        dev = rt.loc[rt["date"] <= dev_cutoff].copy()
        hold = rt.loc[
            (rt["date"] > dev_cutoff) &
            (rt["date"] <= final_cutoff)
        ].copy()
        final = rt.loc[rt["date"] <= final_cutoff].copy()

        if len(dev) < 10 or len(hold) < 5:
            raise ValueError(
                f"{river}: datos insuficientes. dev={len(dev)}, holdout={len(hold)}"
            )

                                                                   
        X_dev, med_dev = prepare_X(dev, features, medians=None)
        X_hold, _ = prepare_X(hold, features, medians=med_dev)

        y_dev = dev["flow_m3s"].to_numpy(float)
        y_hold = hold["flow_m3s"].to_numpy(float)

        val_est = build_estimator(family, best_params)
        val_est = fit_model(family, val_est, X_dev, y_dev)
        pred_hold = predict_model(family, val_est, X_hold)

        met = metrics(y_hold, pred_hold, y_dev)

                                                                                  
        X_final, med_final = prepare_X(final, features, medians=None)
        y_final = final["flow_m3s"].to_numpy(float)

        final_est = build_estimator(family, best_params)
        final_est = fit_model(family, final_est, X_final, y_final)

        bundle = {
            "river": river,
            "family": family,
            "best_params": best_params,
            "features": features,
            "feature_medians": med_final.to_dict(),
            "development_cutoff": str(dev_cutoff.date()),
            "final_cutoff": str(final_cutoff.date()),
            "target": "flow_m3s",
            "target_transform": "log1p/expm1" if family == "RF_LOG" else "none",
            "estimator": final_est,
            "training_n": int(len(final)),
        }

        model_path = MODEL_DIR / f"{river}_future_compatible_72f.joblib"
        joblib.dump(bundle, model_path)

        base = baseline.get(river, {})

        row = {
            "river": river,
            "family": family,
            "n_features": len(features),
            "n_development": int(len(dev)),
            "n_holdout": int(len(hold)),
            "n_final_fit": int(len(final)),
            **met,
            **base,
            "model_path": str(model_path),
        }

        if "baseline_mae" in row and np.isfinite(row["baseline_mae"]):
            row["mae_change_vs_baseline_pct"] = (
                100.0 * (row["mae"] - row["baseline_mae"]) / row["baseline_mae"]
            )
            row["mae_skill_vs_baseline"] = (
                1.0 - row["mae"] / row["baseline_mae"]
            )
        else:
            row["mae_change_vs_baseline_pct"] = np.nan
            row["mae_skill_vs_baseline"] = np.nan

        metric_rows.append(row)

        p = pd.DataFrame({
            "date": hold["date"].dt.strftime("%Y-%m-%d"),
            "river": river,
            "observed_m3s": y_hold,
            "predicted_future_compatible_m3s": pred_hold,
            "residual_m3s": pred_hold - y_hold,
        })
        prediction_rows.append(p)

        manifest_models[river] = {
            "config_key": config_key,
            "family": family,
            "best_params": best_params,
            "features": features,
            "n_features": len(features),
            "n_development": int(len(dev)),
            "n_holdout": int(len(hold)),
            "n_final_fit": int(len(final)),
            "validation_metrics": met,
            "baseline_metrics_detected": base,
            "model_file": str(model_path),
            "model_sha256": sha256_file(model_path),
        }

        print(
            f"    {family:12s} dev={len(dev):3d} holdout={len(hold):3d} "
            f"final={len(final):3d} | MAE={met['mae']:.4f} "
            f"RMSE={met['rmse']:.4f} R2={met['r2_nse']:.4f} "
            f"r={met['pearson_r']:.4f}"
        )

        if "baseline_mae" in base and np.isfinite(base["baseline_mae"]):
            delta = 100.0 * (met["mae"] - base["baseline_mae"]) / base["baseline_mae"]
            print(
                f"    baseline MAE={base['baseline_mae']:.4f} | "
                f"cambio MAE={delta:+.2f}%"
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    metrics_path = OUT_DIR / "10b_metricas_validacion_72f.csv"
    preds_path = OUT_DIR / "10b_predicciones_holdout_72f.csv"
    manifest_path = OUT_DIR / "manifest_paso10b.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(preds_path, index=False)

    manifest = {
        "paso": "10B",
        "descripcion": (
            "Reentrenamiento de los cinco submodelos de caudal usando exclusivamente "
            "72 predictores reproducibles con CMIP6, conservando familia y best_params "
            "del PASO 02."
        ),
        "python_version": sys.version,
        "development_cutoff": str(dev_cutoff.date()),
        "final_cutoff": str(final_cutoff.date()),
        "climate_table": str(climate_path),
        "climate_sha256": sha256_file(climate_path),
        "rivers_table": str(rivers_path),
        "rivers_sha256": sha256_file(rivers_path),
        "config": str(CONFIG_FILE),
        "config_sha256": sha256_file(CONFIG_FILE),
        "feature_list": str(FEATURE_LIST_FILE),
        "feature_list_sha256": sha256_file(FEATURE_LIST_FILE),
        "models": manifest_models,
        "rules": [
            "No se ejecuta Optuna.",
            "No se sobrescriben modelos históricos.",
            "Validación: development <= development_cutoff; holdout > development_cutoff y <= final_cutoff.",
            "Modelo final: ajuste con todas las observaciones <= final_cutoff.",
            "Imputación por mediana se ajusta únicamente dentro del conjunto de entrenamiento correspondiente.",
            "RF_LOG usa log1p/expm1; XGB_TWEEDIE usa objetivo reg:tweedie.",
        ],
        "outputs": {
            "metrics": str(metrics_path),
            "predictions": str(preds_path),
            "models_dir": str(MODEL_DIR),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 92)
    print("RESUMEN")
    print("=" * 92)

    cols = [
        "river", "family", "n_development", "n_holdout",
        "mae", "rmse", "bias", "r2_nse", "pearson_r",
        "highflow_mae", "baseline_mae",
        "mae_change_vs_baseline_pct"
    ]
    cols = [c for c in cols if c in metrics_df.columns]

    with pd.option_context(
        "display.max_columns", 30,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:,.4f}"
    ):
        print(metrics_df[cols].to_string(index=False))

    print()
    print("Archivos:")
    print(metrics_path)
    print(preds_path)
    print(manifest_path)
    print(MODEL_DIR)
    print()
    print("PASO 10B COMPLETADO.")
    print("Los modelos históricos originales siguen intactos.")


if __name__ == "__main__":
    main()
