                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import re
import hashlib

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

NEW_PRED = (
    ROOT / "outputs" / "caudales_future_compatible"
    / "10b_predicciones_holdout_72f.csv"
)

BASE_TOTAL_PRED = (
    ROOT / "outputs" / "caudales"
    / "predicciones_caudal_total_validacion_temporal.csv"
)

BASE_TOTAL_METRICS = (
    ROOT / "outputs" / "caudales"
    / "metricas_caudal_total_validacion_temporal.csv"
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

EXPECTED_BASELINE = {
    "n": 29,
    "mae": 1.649707,
    "rmse": 2.917370,
    "bias": -1.626990,
    "r2_nse": 0.095947,
    "pearson_r": 0.857637,
    "highflow_mae": 4.736170,
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


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        raise ValueError("No hay pares válidos para métricas.")

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

                                                                           
    q75 = float(np.quantile(y, 0.75))
    high = y >= q75
    out["highflow_threshold_q75_validation"] = q75
    out["highflow_mae"] = (
        float(mean_absolute_error(y[high], p[high]))
        if high.any() else np.nan
    )

    return out


def detect_col(df, candidates, contains=False):
    nmap = {norm(c): c for c in df.columns}

    for cand in candidates:
        nc = norm(cand)
        if nc in nmap:
            return nmap[nc]

    if contains:
        for c in df.columns:
            nc = norm(c)
            if any(norm(k) in nc for k in candidates):
                return c

    return None


def prepare_new_monthly():
    if not NEW_PRED.exists():
        raise FileNotFoundError(f"Falta: {NEW_PRED}")

    df = pd.read_csv(NEW_PRED)

    required = {
        "date",
        "river",
        "observed_m3s",
        "predicted_future_compatible_m3s",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"10B no contiene columnas requeridas: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["river"] = df["river"].apply(canonical_river)

    if df["river"].isna().any():
        bad = df.loc[df["river"].isna(), "river"].unique()
        raise ValueError(f"Ríos no reconocidos en 10B: {bad}")

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

                                                              
                                                                        
    monthly_river = (
        df.groupby(["month", "river"], as_index=False)
        .agg(
            observed_m3s=("observed_m3s", "mean"),
            predicted_m3s=("predicted_future_compatible_m3s", "mean"),
            n_obs=("observed_m3s", "size"),
        )
    )

    obs = monthly_river.pivot(
        index="month", columns="river", values="observed_m3s"
    )
    pred = monthly_river.pivot(
        index="month", columns="river", values="predicted_m3s"
    )

    counts = monthly_river.pivot(
        index="month", columns="river", values="n_obs"
    )

    complete_idx = (
        obs.dropna(subset=RIVERS).index
        .intersection(pred.dropna(subset=RIVERS).index)
    )

    obs = obs.loc[complete_idx, RIVERS]
    pred = pred.loc[complete_idx, RIVERS]
    counts = counts.loc[complete_idx, RIVERS]

    total = pd.DataFrame({
        "month": complete_idx,
        "observed_total_72f_m3s": obs.sum(axis=1).to_numpy(float),
        "predicted_total_72f_m3s": pred.sum(axis=1).to_numpy(float),
        "n_rivers": len(RIVERS),
        "n_measurements_in_month": counts.sum(axis=1).to_numpy(float),
    })

    return total, monthly_river


def load_baseline_total():
    if not BASE_TOTAL_PRED.exists():
        return None, None

    df = pd.read_csv(BASE_TOTAL_PRED)

    date_col = detect_col(
        df,
        [
            "date", "month", "fecha", "month_start",
            "date_month", "period", "periodo"
        ],
        contains=True,
    )

    obs_col = detect_col(
        df,
        [
            "observed_total_m3s",
            "q_total_obs_m3s",
            "q_total_observed_m3s",
            "observed_m3s",
            "flow_m3s_obs",
            "q_obs",
            "observed",
        ],
        contains=True,
    )

    pred_col = detect_col(
        df,
        [
            "predicted_total_m3s",
            "q_total_pred_m3s",
            "q_total_predicted_m3s",
            "predicted_m3s",
            "flow_m3s_pred",
            "q_pred",
            "predicted",
        ],
        contains=True,
    )

    clim_col = detect_col(
        df,
        [
            "climatology_m3s",
            "monthly_climatology_m3s",
            "climatology",
            "climatologia",
        ],
        contains=True,
    )

    if date_col is None:
        ycol = detect_col(df, ["year"], contains=False)
        mcol = detect_col(df, ["month"], contains=False)

        if ycol is not None and mcol is not None:
            df["_date_rebuilt"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(df[ycol], errors="raise").astype(int),
                    month=pd.to_numeric(df[mcol], errors="raise").astype(int),
                    day=1,
                )
            )
            date_col = "_date_rebuilt"

    if date_col is None or obs_col is None or pred_col is None:
        return df, {
            "date_col": date_col,
            "obs_col": obs_col,
            "pred_col": pred_col,
            "clim_col": clim_col,
            "usable": False,
        }

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    keep = ["month", obs_col, pred_col]
    if clim_col is not None:
        keep.append(clim_col)

    out = df[keep].copy()

    rename = {
        obs_col: "baseline_observed_total_m3s",
        pred_col: "baseline_predicted_total_m3s",
    }

    if clim_col is not None:
        rename[clim_col] = "baseline_climatology_m3s"

    out = out.rename(columns=rename)

                                                        
    agg = {
        "baseline_observed_total_m3s": "mean",
        "baseline_predicted_total_m3s": "mean",
    }

    if "baseline_climatology_m3s" in out.columns:
        agg["baseline_climatology_m3s"] = "mean"

    out = out.groupby("month", as_index=False).agg(agg)

    info = {
        "date_col": date_col,
        "obs_col": obs_col,
        "pred_col": pred_col,
        "clim_col": clim_col,
        "usable": True,
    }

    return out, info


def parse_baseline_metrics():
    if not BASE_TOTAL_METRICS.exists():
        return None

    try:
        df = pd.read_csv(BASE_TOTAL_METRICS)
    except Exception:
        return None

    return {
        "columns": list(df.columns),
        "rows": df.to_dict("records"),
    }


def main():
    print("=" * 92)
    print("PASO 10C - VALIDAR CAUDAL TOTAL FUTURE-COMPATIBLE")
    print("=" * 92)
    print()
    print("Este paso NO entrena modelos y NO modifica el baseline.")
    print("Suma los cinco tributarios de la validación 10B a escala mensual")
    print("y compara contra la validación total histórica del PASO 02.")
    print()

    new_total, monthly_river = prepare_new_monthly()

    print(f"Meses completos con los 5 tributarios en 10B: {len(new_total)}")

    new_met = metrics(
        new_total["observed_total_72f_m3s"],
        new_total["predicted_total_72f_m3s"],
    )

    base_total, base_info = load_baseline_total()
    baseline_metrics_file = parse_baseline_metrics()

    comparison = new_total.copy()
    base_met_recomputed = None
    clim_met = None
    common_met_new = None
    common_met_base = None

    if base_total is not None and base_info and base_info.get("usable"):
        comparison = comparison.merge(
            base_total,
            on="month",
            how="inner",
            validate="one_to_one",
        )

        if len(comparison):
                                                                                       
            obs_diff = (
                comparison["observed_total_72f_m3s"]
                - comparison["baseline_observed_total_m3s"]
            )

            comparison["observed_difference_72f_minus_baseline_m3s"] = obs_diff

            common_met_new = metrics(
                comparison["baseline_observed_total_m3s"],
                comparison["predicted_total_72f_m3s"],
            )

            common_met_base = metrics(
                comparison["baseline_observed_total_m3s"],
                comparison["baseline_predicted_total_m3s"],
            )

            base_met_recomputed = common_met_base

            if "baseline_climatology_m3s" in comparison.columns:
                clim_met = metrics(
                    comparison["baseline_observed_total_m3s"],
                    comparison["baseline_climatology_m3s"],
                )

                                                                                    
    new_eval = common_met_new if common_met_new is not None else new_met
    base_eval = common_met_base if common_met_base is not None else EXPECTED_BASELINE

    baseline_mae = float(base_eval["mae"])
    new_mae = float(new_eval["mae"])

    skill_vs_baseline = 1.0 - new_mae / baseline_mae
    mae_change_pct = 100.0 * (new_mae - baseline_mae) / baseline_mae

                                        
    if clim_met is not None:
        climatology_mae = float(clim_met["mae"])
    else:
                                                                       
        s = EXPECTED_BASELINE["mae_skill_vs_monthly_climatology"]
        climatology_mae = EXPECTED_BASELINE["mae"] / (1.0 - s)

    skill_vs_climatology = 1.0 - new_mae / climatology_mae

                                       
                                                                    
    accepted = (
        new_mae <= baseline_mae * 1.05
        and skill_vs_climatology > 0.0
    )

    comparison_path = OUT_DIR / "10c_comparacion_caudal_total.csv"
    monthly_river_path = OUT_DIR / "10c_validacion_mensual_por_rio.csv"
    summary_path = OUT_DIR / "10c_resumen_validacion_total.csv"
    manifest_path = OUT_DIR / "manifest_paso10c.json"

    comparison.to_csv(comparison_path, index=False)
    monthly_river.to_csv(monthly_river_path, index=False)

    summary = pd.DataFrame([
        {
            "model": "FUTURE_COMPATIBLE_72F",
            **new_eval,
            "skill_vs_baseline_mae": skill_vs_baseline,
            "mae_change_vs_baseline_pct": mae_change_pct,
            "climatology_mae_reference": climatology_mae,
            "skill_vs_monthly_climatology": skill_vs_climatology,
            "accepted_under_rule": accepted,
        },
        {
            "model": "BASELINE_HISTORICO_91F",
            **base_eval,
            "skill_vs_baseline_mae": 0.0,
            "mae_change_vs_baseline_pct": 0.0,
            "climatology_mae_reference": climatology_mae,
            "skill_vs_monthly_climatology": (
                1.0 - baseline_mae / climatology_mae
            ),
            "accepted_under_rule": True,
        },
    ])

    summary.to_csv(summary_path, index=False)

    manifest = {
        "paso": "10C",
        "descripcion": (
            "Validación del caudal total de los cinco tributarios para la versión "
            "future-compatible de 72 predictores, comparada con el baseline histórico."
        ),
        "inputs": {
            "new_predictions": str(NEW_PRED),
            "new_predictions_sha256": sha256_file(NEW_PRED),
            "baseline_total_predictions": (
                str(BASE_TOTAL_PRED) if BASE_TOTAL_PRED.exists() else None
            ),
            "baseline_total_metrics": (
                str(BASE_TOTAL_METRICS) if BASE_TOTAL_METRICS.exists() else None
            ),
        },
        "baseline_detection": base_info,
        "baseline_metrics_file_content": baseline_metrics_file,
        "expected_baseline_from_project": EXPECTED_BASELINE,
        "new_metrics_all_complete_months": new_met,
        "new_metrics_common_months": common_met_new,
        "baseline_metrics_recomputed_common_months": base_met_recomputed,
        "climatology_metrics_recomputed_common_months": clim_met,
        "comparison": {
            "baseline_mae_used": baseline_mae,
            "new_mae_used": new_mae,
            "mae_change_vs_baseline_pct": mae_change_pct,
            "skill_vs_baseline_mae": skill_vs_baseline,
            "climatology_mae_reference": climatology_mae,
            "skill_vs_monthly_climatology": skill_vs_climatology,
            "acceptance_rule": (
                "Aceptar si MAE <= 1.05 * baseline_MAE y "
                "skill_vs_monthly_climatology > 0."
            ),
            "accepted": accepted,
        },
        "outputs": {
            "comparison_csv": str(comparison_path),
            "monthly_river_csv": str(monthly_river_path),
            "summary_csv": str(summary_path),
        },
        "notes": [
            "No se reentrena ningún modelo.",
            "La agregación por tributario es mensual; si hay múltiples mediciones de un río en un mes se usa su media.",
            "Sólo se suman meses con observaciones de los cinco tributarios.",
            "Cuando es posible, la comparación se restringe a los meses comunes con el archivo de validación total del PASO 02.",
            "Si la climatología no está explícita en el archivo baseline, se reconstruye su MAE a partir del skill histórico reportado."
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 92)
    print("RESULTADO PRINCIPAL")
    print("=" * 92)

    print(f"Meses usados nueva validación: {new_eval['n']}")
    print(f"MAE 72f:       {new_eval['mae']:.6f} m3/s")
    print(f"RMSE 72f:      {new_eval['rmse']:.6f} m3/s")
    print(f"Bias 72f:      {new_eval['bias']:.6f} m3/s")
    print(f"R2/NSE 72f:    {new_eval['r2_nse']:.6f}")
    print(f"Pearson 72f:   {new_eval['pearson_r']:.6f}")
    print(f"Highflow MAE:  {new_eval['highflow_mae']:.6f} m3/s")
    print()

    print(f"MAE baseline usado: {baseline_mae:.6f} m3/s")
    print(f"Cambio MAE:         {mae_change_pct:+.2f}%")
    print(f"Skill vs baseline:  {skill_vs_baseline:+.4f}")
    print(f"Skill vs climatol.: {skill_vs_climatology:+.4f}")
    print()

    if len(comparison) and "observed_difference_72f_minus_baseline_m3s" in comparison:
        max_obs_diff = float(
            comparison["observed_difference_72f_minus_baseline_m3s"]
            .abs()
            .max()
        )
        print(f"Máxima diferencia en observado vs baseline: {max_obs_diff:.10f} m3/s")
        print()

    print(
        "DECISIÓN: "
        + ("APROBADO" if accepted else "NO APROBADO")
        + " bajo la regla conservadora del PASO 10C."
    )

    print()
    print("Archivos creados:")
    print(comparison_path)
    print(monthly_river_path)
    print(summary_path)
    print(manifest_path)
    print()
    print("PASO 10C COMPLETADO.")


if __name__ == "__main__":
    main()
