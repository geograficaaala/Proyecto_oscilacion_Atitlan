                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import re
import hashlib
import sys
import warnings

import numpy as np
import pandas as pd

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

HOLDOUT_72F_FILE = (
    ROOT / "outputs" / "caudales_future_compatible"
    / "10b_predicciones_holdout_72f.csv"
)

BASELINE_TOTAL_FILE = (
    ROOT / "outputs" / "caudales"
    / "predicciones_caudal_total_validacion_temporal.csv"
)

OUT_DIR = ROOT / "outputs" / "caudales_future_compatible"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIVERS = [
    "Quiscab",
    "San_Francisco",
    "Tzununa",
    "La_Catarata",
    "San_Buenaventura",
]

EXPECTED_FEATURES = 72
RANDOM_STATE = 42

                                                   
INITIAL_TRAIN_FRACTION = 0.50
N_OOF_FOLDS = 3

                                                 
MIN_INTERNAL_IMPROVEMENT = 0.02       

EXPECTED_BASELINE = {
    "mae": 1.649707,
    "mae_skill_vs_monthly_climatology": 0.145223,
}


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

    raise FileNotFoundError(
        f"No existe ni {pkl_path} ni {csv_path}"
    )


def canonical_river(x):
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
        "san_buenaventura": [
            "san_buenaventura", "sanbuenaventura", "buenaventura"
        ],
    }

    for key in models:
        nk = norm(key)
        if any(a in nk for a in aliases[nr]):
            return key

    raise KeyError(f"No encontré {river} en config['river_models'].")


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
        raise ValueError(f"No pude identificar familia para {river}.")

    params = info.get("best_params")

    if not isinstance(params, dict):
        raise ValueError(f"No encontré best_params para {river}.")

    return family.upper(), dict(params)


def clean_params(params):
    return {k: v for k, v in params.items() if v is not None}


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
            raise ImportError("XGBoost no está disponible.")

        base = {
            "objective": "reg:tweedie",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        base.update(params)
        return XGBRegressor(**base)

    raise ValueError(f"Familia no soportada: {family}")


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
        X[f] = pd.to_numeric(
            df[mapping[f]],
            errors="coerce"
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = X.median(axis=0, numeric_only=True)

    if medians.isna().any():
        bad = medians.index[medians.isna()].tolist()
        raise ValueError(
            f"Features sin mediana válida: {bad[:15]}"
        )

    X = X.fillna(medians)

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Persisten NaN: {bad[:15]}")

    return X.astype(float), medians.astype(float)


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


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        raise ValueError("No hay pares válidos.")

    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
        "r2_nse": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "pearson_r": np.nan,
    }

    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson_r"] = float(np.corrcoef(y, p)[0, 1])

    return out


def make_monthly_total(df, pred_col, obs_col):
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(
        tmp["date"], errors="raise"
    ).dt.to_period("M").dt.to_timestamp()

    monthly_river = (
        tmp.groupby(["month", "river"], as_index=False)
        .agg(
            observed_m3s=(obs_col, "mean"),
            predicted_m3s=(pred_col, "mean"),
        )
    )

    obs = monthly_river.pivot(
        index="month",
        columns="river",
        values="observed_m3s",
    )

    pred = monthly_river.pivot(
        index="month",
        columns="river",
        values="predicted_m3s",
    )

    complete = (
        obs.dropna(subset=RIVERS).index
        .intersection(
            pred.dropna(subset=RIVERS).index
        )
    )

    obs = obs.loc[complete, RIVERS]
    pred = pred.loc[complete, RIVERS]

    total = pd.DataFrame({
        "month": complete,
        "observed_total_m3s": obs.sum(axis=1).to_numpy(float),
        "predicted_total_m3s": pred.sum(axis=1).to_numpy(float),
    })

    return total.sort_values("month").reset_index(drop=True)


def make_oof_boundaries(dev_start, dev_end):
    start = pd.Timestamp(dev_start).normalize()
    end = pd.Timestamp(dev_end).normalize()

    total_days = (end - start).days + 1

    initial_days = max(
        1,
        int(np.floor(total_days * INITIAL_TRAIN_FRACTION))
    )

    initial_end = start + pd.Timedelta(days=initial_days)

    remaining_start = initial_end + pd.Timedelta(days=1)
    remaining_days = (end - remaining_start).days + 1

    if remaining_days <= N_OOF_FOLDS:
        raise ValueError("Development demasiado corto para OOF temporal.")

    fold_days = remaining_days / N_OOF_FOLDS

    folds = []

    for fold in range(N_OOF_FOLDS):
        valid_start = (
            remaining_start
            + pd.Timedelta(days=int(np.floor(fold * fold_days)))
        )

        if fold == N_OOF_FOLDS - 1:
            valid_end = end
        else:
            valid_end = (
                remaining_start
                + pd.Timedelta(
                    days=int(np.floor((fold + 1) * fold_days)) - 1
                )
            )

        folds.append({
            "fold": fold + 1,
            "train_end": valid_start - pd.Timedelta(days=1),
            "valid_start": valid_start,
            "valid_end": valid_end,
        })

    return folds


def generate_development_oof(
    table,
    config,
    feature_sets,
    development_cutoff,
):
    dev_table = table.loc[
        table["date"] <= development_cutoff
    ].copy()

    dev_start = dev_table["date"].min()
    folds = make_oof_boundaries(
        dev_start,
        development_cutoff,
    )

    all_predictions = []

    print("OOF temporal dentro de DEVELOPMENT:")
    for f in folds:
        print(
            f" fold {f['fold']}: "
            f"train <= {f['train_end'].date()} | "
            f"valid {f['valid_start'].date()} a {f['valid_end'].date()}"
        )
    print()

    for river in RIVERS:
        rt = (
            dev_table.loc[
                dev_table["river"].eq(river)
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        features = feature_sets[river]

        if len(features) != EXPECTED_FEATURES:
            raise ValueError(
                f"{river}: {len(features)} features, se esperaban {EXPECTED_FEATURES}."
            )

        family, params = extract_family_and_params(
            config,
            river,
        )

        for f in folds:
            train = rt.loc[
                rt["date"] <= f["train_end"]
            ].copy()

            valid = rt.loc[
                (rt["date"] >= f["valid_start"])
                & (rt["date"] <= f["valid_end"])
            ].copy()

            train = train.loc[
                np.isfinite(
                    pd.to_numeric(
                        train["flow_m3s"],
                        errors="coerce"
                    )
                )
            ].copy()

            valid = valid.loc[
                np.isfinite(
                    pd.to_numeric(
                        valid["flow_m3s"],
                        errors="coerce"
                    )
                )
            ].copy()

            if len(train) < 20 or len(valid) == 0:
                continue

            X_train, medians = prepare_X(
                train,
                features,
                medians=None,
            )

            X_valid, _ = prepare_X(
                valid,
                features,
                medians=medians,
            )

            y_train = train["flow_m3s"].to_numpy(float)

            estimator = build_estimator(
                family,
                params,
            )

            estimator = fit_model(
                family,
                estimator,
                X_train,
                y_train,
            )

            pred = predict_model(
                family,
                estimator,
                X_valid,
            )

            piece = pd.DataFrame({
                "date": valid["date"].to_numpy(),
                "river": river,
                "fold": f["fold"],
                "observed_m3s": valid["flow_m3s"].to_numpy(float),
                "predicted_m3s": pred,
            })

            all_predictions.append(piece)

    if not all_predictions:
        raise RuntimeError(
            "No fue posible generar predicciones OOF."
        )

    oof = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    oof_total = make_monthly_total(
        oof,
        pred_col="predicted_m3s",
        obs_col="observed_m3s",
    )

    return oof, oof_total, folds


def fit_multiplicative(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = (
        np.isfinite(y)
        & np.isfinite(p)
        & (p > 1e-9)
    )

    if valid.sum() < 3:
        raise ValueError(
            "No hay suficientes pares para corrección multiplicativa."
        )

    ratios = y[valid] / p[valid]

    factor = float(np.median(ratios))

    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(
            "Factor multiplicativo inválido."
        )

    return {
        "method": "MULTIPLICATIVE",
        "factor": factor,
    }


def fit_linear(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = np.isfinite(y) & np.isfinite(p)

    y = y[valid]
    p = p[valid]

    if len(y) < 4 or np.std(p) <= 0:
        raise ValueError(
            "No hay datos suficientes para corrección lineal."
        )

    slope, intercept = np.polyfit(
        p,
        y,
        1,
    )

    slope = float(slope)
    intercept = float(intercept)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("Corrección lineal inválida.")

    if slope < 0:
        raise ValueError(
            "Pendiente lineal negativa; se rechaza por falta de plausibilidad."
        )

    return {
        "method": "LINEAR",
        "slope": slope,
        "intercept": intercept,
    }


def apply_correction(pred, params):
    p = np.asarray(pred, dtype=float)
    method = params["method"]

    if method == "IDENTITY":
        out = p.copy()

    elif method == "MULTIPLICATIVE":
        out = p * float(params["factor"])

    elif method == "LINEAR":
        out = (
            float(params["intercept"])
            + float(params["slope"]) * p
        )

    else:
        raise ValueError(f"Método desconocido: {method}")

    return np.maximum(out, 0.0)


def candidate_fit(y, p):
    candidates = [
        {"method": "IDENTITY"},
    ]

    try:
        candidates.append(
            fit_multiplicative(y, p)
        )
    except Exception as e:
        warnings.warn(
            f"No se pudo ajustar multiplicativa: {e}"
        )

    try:
        candidates.append(
            fit_linear(y, p)
        )
    except Exception as e:
        warnings.warn(
            f"No se pudo ajustar lineal: {e}"
        )

    return candidates


def choose_method(oof_total):
    if len(oof_total) < 9:
        raise ValueError(
            f"Sólo hay {len(oof_total)} meses OOF completos; "
            "se necesitan al menos 9."
        )

    n_fit = int(
        np.floor(len(oof_total) * 2 / 3)
    )

    n_fit = max(5, n_fit)

    if len(oof_total) - n_fit < 3:
        n_fit = len(oof_total) - 3

    fit_df = oof_total.iloc[:n_fit].copy()
    val_df = oof_total.iloc[n_fit:].copy()

    y_fit = fit_df["observed_total_m3s"].to_numpy(float)
    p_fit = fit_df["predicted_total_m3s"].to_numpy(float)

    y_val = val_df["observed_total_m3s"].to_numpy(float)
    p_val = val_df["predicted_total_m3s"].to_numpy(float)

    candidates = candidate_fit(
        y_fit,
        p_fit,
    )

    rows = []

    for params in candidates:
        corrected = apply_correction(
            p_val,
            params,
        )

        m = metrics(
            y_val,
            corrected,
        )

        rows.append({
            "method": params["method"],
            "params_json": json.dumps(
                params,
                ensure_ascii=False
            ),
            **m,
        })

    comp = pd.DataFrame(rows)

    identity_mae = float(
        comp.loc[
            comp["method"].eq("IDENTITY"),
            "mae",
        ].iloc[0]
    )

    best_idx = comp["mae"].idxmin()
    best = comp.loc[best_idx]

    improvement = (
        1.0
        - float(best["mae"]) / identity_mae
    )

    if (
        best["method"] != "IDENTITY"
        and improvement >= MIN_INTERNAL_IMPROVEMENT
    ):
        selected_method = str(best["method"])
    else:
        selected_method = "IDENTITY"

                                                                 
    y_all = oof_total["observed_total_m3s"].to_numpy(float)
    p_all = oof_total["predicted_total_m3s"].to_numpy(float)

    if selected_method == "IDENTITY":
        final_params = {
            "method": "IDENTITY",
        }

    elif selected_method == "MULTIPLICATIVE":
        final_params = fit_multiplicative(
            y_all,
            p_all,
        )

    elif selected_method == "LINEAR":
        final_params = fit_linear(
            y_all,
            p_all,
        )

    else:
        raise ValueError(selected_method)

    selection_info = {
        "n_oof_months": int(len(oof_total)),
        "n_internal_fit_months": int(len(fit_df)),
        "n_internal_validation_months": int(len(val_df)),
        "identity_validation_mae": identity_mae,
        "best_raw_method": str(best["method"]),
        "best_raw_validation_mae": float(best["mae"]),
        "best_raw_improvement_vs_identity": float(improvement),
        "minimum_improvement_required": MIN_INTERNAL_IMPROVEMENT,
        "selected_method": selected_method,
        "final_params_refit_all_oof": final_params,
    }

    return comp, selection_info, fit_df, val_df


def load_holdout_72f_total():
    if not HOLDOUT_72F_FILE.exists():
        raise FileNotFoundError(
            HOLDOUT_72F_FILE
        )

    df = pd.read_csv(
        HOLDOUT_72F_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    df["river"] = df["river"].apply(
        canonical_river
    )

    if df["river"].isna().any():
        raise ValueError(
            "Hay ríos no reconocidos en 10B."
        )

    return make_monthly_total(
        df,
        pred_col="predicted_future_compatible_m3s",
        obs_col="observed_m3s",
    )


def detect_col(df, candidates):
    nmap = {
        norm(c): c
        for c in df.columns
    }

    for cand in candidates:
        if norm(cand) in nmap:
            return nmap[norm(cand)]

    for c in df.columns:
        nc = norm(c)

        if any(norm(k) in nc for k in candidates):
            return c

    return None


def load_baseline_total():
    if not BASELINE_TOTAL_FILE.exists():
        return None

    df = pd.read_csv(
        BASELINE_TOTAL_FILE
    )

    date_col = detect_col(
        df,
        ["date", "month", "fecha", "periodo"]
    )

    obs_col = detect_col(
        df,
        [
            "observed_total_m3s",
            "q_total_obs_m3s",
            "observed_m3s",
            "observed",
        ]
    )

    pred_col = detect_col(
        df,
        [
            "predicted_total_m3s",
            "q_total_pred_m3s",
            "predicted_m3s",
            "predicted",
        ]
    )

    clim_col = detect_col(
        df,
        [
            "climatology_m3s",
            "monthly_climatology_m3s",
            "climatology",
            "climatologia",
        ]
    )

    if date_col is None or obs_col is None or pred_col is None:
        return None

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="raise"
    )

    df["month"] = (
        df[date_col]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    keep = [
        "month",
        obs_col,
        pred_col,
    ]

    if clim_col is not None:
        keep.append(clim_col)

    out = df[keep].copy()

    ren = {
        obs_col: "baseline_observed_m3s",
        pred_col: "baseline_predicted_m3s",
    }

    if clim_col is not None:
        ren[clim_col] = "baseline_climatology_m3s"

    out = out.rename(
        columns=ren
    )

    agg = {
        "baseline_observed_m3s": "mean",
        "baseline_predicted_m3s": "mean",
    }

    if "baseline_climatology_m3s" in out.columns:
        agg["baseline_climatology_m3s"] = "mean"

    return (
        out.groupby("month", as_index=False)
        .agg(agg)
    )


def main():
    print("=" * 96)
    print("PASO 10D - CALIBRACION CONSERVADORA DEL SESGO DE CAUDAL TOTAL")
    print("=" * 96)
    print()
    print("Reglas:")
    print(" - corrección calibrada SOLO con DEVELOPMENT")
    print(" - selección mediante OOF temporal interno dentro de DEVELOPMENT")
    print(" - el holdout de 29 meses NO participa en la selección")
    print(" - se prueban IDENTITY, MULTIPLICATIVE y LINEAR")
    print(" - corrección no trivial exige >= 2% de mejora interna")
    print(" - NO Optuna")
    print(" - NO se modifican los modelos 10B ni el baseline")
    print()

    for p in [
        CONFIG_FILE,
        FEATURE_LIST_FILE,
        HOLDOUT_72F_FILE,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    config = load_json(
        CONFIG_FILE
    )

    feature_sets = load_json(
        FEATURE_LIST_FILE
    )

    development_cutoff = pd.Timestamp(
        config["development_cutoff"]
    ).normalize()

    final_cutoff = pd.Timestamp(
        config["final_cutoff"]
    ).normalize()

    climate, climate_path = load_dataframe(
        FEATURES_PKL,
        FEATURES_CSV,
    )

    rivers, rivers_path = load_dataframe(
        RIVERS_PKL,
        RIVERS_CSV,
    )

    climate["date"] = pd.to_datetime(
        climate["date"],
        errors="raise"
    ).dt.normalize()

    rivers["date"] = pd.to_datetime(
        rivers["date"],
        errors="raise"
    ).dt.normalize()

    rivers["river"] = rivers["river"].apply(
        canonical_river
    )

    rivers["flow_m3s"] = pd.to_numeric(
        rivers["flow_m3s"],
        errors="coerce"
    )

    rivers = rivers.loc[
        rivers["river"].notna()
        & np.isfinite(rivers["flow_m3s"])
    ].copy()

    if climate["date"].duplicated().any():
        raise ValueError(
            "La tabla climática tiene fechas duplicadas."
        )

    table = rivers.merge(
        climate,
        on="date",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_clim"),
    )

    print(f"Development cutoff: {development_cutoff.date()}")
    print(f"Final cutoff:       {final_cutoff.date()}")
    print()

    oof_river, oof_total, folds = generate_development_oof(
        table=table,
        config=config,
        feature_sets=feature_sets,
        development_cutoff=development_cutoff,
    )

    print(f"Meses OOF completos con los 5 ríos: {len(oof_total)}")
    print()

    comp_internal, selection, fit_df, val_df = choose_method(
        oof_total
    )

    print("VALIDACIÓN INTERNA DE CORRECCIÓN:")
    with pd.option_context(
        "display.max_columns", 20,
        "display.width", 180,
        "display.float_format", lambda x: f"{x:,.5f}"
    ):
        print(
            comp_internal[
                [
                    "method",
                    "n",
                    "mae",
                    "rmse",
                    "bias",
                    "r2_nse",
                    "pearson_r",
                ]
            ].to_string(index=False)
        )

    print()
    print("Método seleccionado:", selection["selected_method"])
    print(
        "Parámetros finales:",
        selection["final_params_refit_all_oof"]
    )
    print()

    holdout = load_holdout_72f_total()

    final_params = selection[
        "final_params_refit_all_oof"
    ]

    holdout["predicted_corrected_m3s"] = apply_correction(
        holdout["predicted_total_m3s"].to_numpy(float),
        final_params,
    )

    raw_met = metrics(
        holdout["observed_total_m3s"],
        holdout["predicted_total_m3s"],
    )

    corrected_met = metrics(
        holdout["observed_total_m3s"],
        holdout["predicted_corrected_m3s"],
    )

    baseline = load_baseline_total()

    baseline_met = None
    climatology_mae = None

    if baseline is not None:
        common = holdout.merge(
            baseline,
            on="month",
            how="inner",
            validate="one_to_one",
        )

        if len(common):
            baseline_met = metrics(
                common["baseline_observed_m3s"],
                common["baseline_predicted_m3s"],
            )

            if "baseline_climatology_m3s" in common.columns:
                clim_met = metrics(
                    common["baseline_observed_m3s"],
                    common["baseline_climatology_m3s"],
                )
                climatology_mae = float(
                    clim_met["mae"]
                )

    if baseline_met is None:
        baseline_mae = EXPECTED_BASELINE["mae"]
    else:
        baseline_mae = float(
            baseline_met["mae"]
        )

    if climatology_mae is None:
        climatology_mae = (
            EXPECTED_BASELINE["mae"]
            / (
                1.0
                - EXPECTED_BASELINE[
                    "mae_skill_vs_monthly_climatology"
                ]
            )
        )

    corrected_mae = float(
        corrected_met["mae"]
    )

    skill_vs_baseline = (
        1.0
        - corrected_mae / baseline_mae
    )

    skill_vs_climatology = (
        1.0
        - corrected_mae / climatology_mae
    )

    mae_change_vs_baseline_pct = (
        100.0
        * (corrected_mae - baseline_mae)
        / baseline_mae
    )

                                             
    accepted = (
        corrected_mae <= baseline_mae * 1.05
        and skill_vs_climatology > 0.0
    )

    oof_river_path = (
        OUT_DIR
        / "10d_oof_development_por_rio.csv"
    )

    oof_total_path = (
        OUT_DIR
        / "10d_oof_development_total.csv"
    )

    internal_path = (
        OUT_DIR
        / "10d_comparacion_correcciones_interna.csv"
    )

    holdout_path = (
        OUT_DIR
        / "10d_holdout_total_corregido.csv"
    )

    summary_path = (
        OUT_DIR
        / "10d_resumen_calibracion_sesgo.csv"
    )

    params_path = (
        OUT_DIR
        / "10d_parametros_correccion_caudal_total.json"
    )

    manifest_path = (
        OUT_DIR
        / "manifest_paso10d.json"
    )

    oof_river.to_csv(
        oof_river_path,
        index=False,
    )

    oof_total.to_csv(
        oof_total_path,
        index=False,
    )

    comp_internal.to_csv(
        internal_path,
        index=False,
    )

    holdout.to_csv(
        holdout_path,
        index=False,
    )

    summary_df = pd.DataFrame([
        {
            "version": "72F_RAW",
            **raw_met,
            "skill_vs_baseline_mae": (
                1.0
                - raw_met["mae"] / baseline_mae
            ),
            "skill_vs_monthly_climatology": (
                1.0
                - raw_met["mae"] / climatology_mae
            ),
        },
        {
            "version": (
                "72F_CORRECTED_"
                + selection["selected_method"]
            ),
            **corrected_met,
            "skill_vs_baseline_mae": skill_vs_baseline,
            "skill_vs_monthly_climatology": skill_vs_climatology,
        },
        {
            "version": "BASELINE_91F",
            **(
                baseline_met
                if baseline_met is not None
                else {
                    "n": 29,
                    "mae": baseline_mae,
                    "rmse": np.nan,
                    "bias": np.nan,
                    "r2_nse": np.nan,
                    "pearson_r": np.nan,
                }
            ),
            "skill_vs_baseline_mae": 0.0,
            "skill_vs_monthly_climatology": (
                1.0
                - baseline_mae / climatology_mae
            ),
        },
    ])

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    with open(
        params_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selected_method": selection["selected_method"],
                "params": final_params,
                "selection": selection,
                "development_cutoff": str(
                    development_cutoff.date()
                ),
                "final_cutoff": str(
                    final_cutoff.date()
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest = {
        "paso": "10D",
        "descripcion": (
            "Calibración conservadora del sesgo del caudal total "
            "future-compatible usando únicamente predicciones OOF "
            "temporales dentro del periodo DEVELOPMENT."
        ),
        "python_version": sys.version,
        "development_cutoff": str(
            development_cutoff.date()
        ),
        "final_cutoff": str(
            final_cutoff.date()
        ),
        "oof_settings": {
            "initial_train_fraction": INITIAL_TRAIN_FRACTION,
            "n_folds": N_OOF_FOLDS,
            "folds": [
                {
                    k: (
                        str(v.date())
                        if isinstance(v, pd.Timestamp)
                        else v
                    )
                    for k, v in fold.items()
                }
                for fold in folds
            ],
        },
        "selection": selection,
        "holdout_metrics_raw": raw_met,
        "holdout_metrics_corrected": corrected_met,
        "baseline_metrics": baseline_met,
        "baseline_mae_used": baseline_mae,
        "climatology_mae_used": climatology_mae,
        "mae_change_vs_baseline_pct": mae_change_vs_baseline_pct,
        "skill_vs_baseline_mae": skill_vs_baseline,
        "skill_vs_monthly_climatology": skill_vs_climatology,
        "acceptance_rule": (
            "MAE corregido <= 1.05 * MAE baseline y "
            "skill vs climatología > 0."
        ),
        "accepted": accepted,
        "inputs": {
            "climate": str(climate_path),
            "climate_sha256": sha256_file(climate_path),
            "rivers": str(rivers_path),
            "rivers_sha256": sha256_file(rivers_path),
            "config": str(CONFIG_FILE),
            "config_sha256": sha256_file(CONFIG_FILE),
            "feature_list": str(FEATURE_LIST_FILE),
            "feature_list_sha256": sha256_file(FEATURE_LIST_FILE),
            "holdout_72f": str(HOLDOUT_72F_FILE),
            "holdout_72f_sha256": sha256_file(HOLDOUT_72F_FILE),
        },
        "outputs": {
            "oof_river": str(oof_river_path),
            "oof_total": str(oof_total_path),
            "internal_comparison": str(internal_path),
            "holdout_corrected": str(holdout_path),
            "summary": str(summary_path),
            "correction_params": str(params_path),
        },
        "notes": [
            "El holdout 2021-05-24 a 2024-01-09 no participa en el ajuste ni selección de la corrección.",
            "IDENTITY forma parte de los candidatos para evitar forzar una corrección innecesaria.",
            "MULTIPLICATIVE usa la mediana de observado/predicho sobre el bloque de ajuste interno.",
            "LINEAR usa y = intercept + slope*x y rechaza pendiente negativa.",
            "Una corrección no trivial sólo se selecciona si mejora al menos 2% el MAE interno respecto a IDENTITY.",
            "Tras seleccionar método, sus parámetros se reajustan usando todo el OOF de DEVELOPMENT.",
            "No se modifica ningún modelo del PASO 10B."
        ],
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

    print("=" * 96)
    print("RESULTADO HOLDOUT 29 MESES")
    print("=" * 96)
    print(
        f"RAW 72f       MAE={raw_met['mae']:.6f} | "
        f"bias={raw_met['bias']:.6f} | "
        f"R2/NSE={raw_met['r2_nse']:.6f} | "
        f"r={raw_met['pearson_r']:.6f}"
    )
    print(
        f"CORREGIDO     MAE={corrected_met['mae']:.6f} | "
        f"bias={corrected_met['bias']:.6f} | "
        f"R2/NSE={corrected_met['r2_nse']:.6f} | "
        f"r={corrected_met['pearson_r']:.6f}"
    )
    print(
        f"BASELINE MAE  = {baseline_mae:.6f}"
    )
    print(
        f"Cambio vs baseline = "
        f"{mae_change_vs_baseline_pct:+.2f}%"
    )
    print(
        f"Skill vs baseline  = "
        f"{skill_vs_baseline:+.4f}"
    )
    print(
        f"Skill vs climatol. = "
        f"{skill_vs_climatology:+.4f}"
    )
    print()
    print(
        "DECISIÓN: "
        + (
            "APROBADO"
            if accepted
            else "NO APROBADO"
        )
        + " bajo la misma regla conservadora del PASO 10C."
    )
    print()
    print("Archivos creados:")
    print(oof_river_path)
    print(oof_total_path)
    print(internal_path)
    print(holdout_path)
    print(summary_path)
    print(params_path)
    print(manifest_path)
    print()
    print("PASO 10D COMPLETADO.")


if __name__ == "__main__":
    main()
