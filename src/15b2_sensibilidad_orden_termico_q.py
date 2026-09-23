from __future__ import annotations
__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"

                                                                               
                                                                     
                                                                               
                                                   
 
           
                                                                            
                                                                         
                                                                    
 
                                                         
                                                                             
                                                       
 
                                  
                                  
                    
 
                                                                       
                                                                              
                                                                            
                                                       
 
                                                     
                                                                               

from pathlib import Path
import gc
import importlib.util
import json

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

LEVEL_FILE = ROOT / "data" / "processed" / "nivel_canonico.csv"

AQ = 0.758222
AREA_M2 = 125.77e6

PRIMARY_START = pd.Timestamp("1981-01-01")
PRIMARY_END = pd.Timestamp("2014-12-31")

SCENARIOS = ("ssp245", "ssp585")
HORIZONS = (2030, 2050, 2090)

EPS_CHANGE = 1e-12


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


def load_13k():
    require(SCRIPT_13K, "script_13K")
    source = SCRIPT_13K.read_text(encoding="utf-8", errors="replace")
    if "if __name__" not in source or "__main__" not in source:
        raise RuntimeError(
            "13K no tiene guard __main__; se aborta para evitar escrituras."
        )

    spec = importlib.util.spec_from_file_location(
        "atitlan_13k_for_15b2", SCRIPT_13K
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar 13K.")

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


def model_features(models: dict) -> list[str]:
    return sorted(
        set(c for info in models.values() for c in info["features"])
    )


def ensure_feature_date(feat: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if "date" in feat.columns:
        out = feat.copy()
        out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
        return out

    if len(feat) != len(base):
        raise RuntimeError(
            "build_72_features no devolvio date y el largo no coincide."
        )

    out = feat.copy()
    out.insert(
        0,
        "date",
        pd.to_datetime(base["date"], errors="raise").dt.normalize().to_numpy(),
    )
    return out


def predict_q(
    feat: pd.DataFrame,
    models: dict,
    intercept: float,
    slope: float,
) -> pd.DataFrame:
    feats = model_features(models)
    missing = [c for c in feats if c not in feat.columns]
    if missing:
        raise KeyError(f"Faltan features: {missing}")

    numeric = feat[feats].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    idx = np.flatnonzero(valid)

    if len(idx) == 0:
        raise RuntimeError("No hay filas validas para Q.")

    q_sum = np.zeros(len(idx), dtype=float)

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
                f"Familia no reconocida: {river} -> {family}"
            )

        q_sum += np.maximum(pred, 0.0)

    out = pd.DataFrame(
        {"date": pd.to_datetime(feat["date"]).dt.normalize()}
    )
    out["Q_m3s"] = np.nan
    out.loc[idx, "Q_m3s"] = np.maximum(
        intercept + slope * q_sum, 0.0
    )
    return out


def repair_temperature_order(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = base.copy()

    tmin = pd.to_numeric(out["tmin_c"], errors="coerce").to_numpy(float)
    tmean = pd.to_numeric(out["tmean_c"], errors="coerce").to_numpy(float)
    tmax = pd.to_numeric(out["tmax_c"], errors="coerce").to_numpy(float)

    bad_min = tmin > tmean
    bad_max = tmax < tmean
    bad = bad_min | bad_max

    tmin_new = np.minimum(tmin, tmean)
    tmax_new = np.maximum(tmax, tmean)

    dmin = tmin_new - tmin
    dmax = tmax_new - tmax

    out["tmin_c"] = tmin_new
    out["tmax_c"] = tmax_new

    severity = np.maximum(
        np.maximum(tmin - tmean, tmean - tmax),
        0.0,
    )
    positive = severity[bad]

    rec = {
        "rows": int(len(out)),
        "bad_n": int(bad.sum()),
        "bad_pct": float(100.0 * bad.mean()),
        "max_violation_c": (
            float(positive.max()) if len(positive) else 0.0
        ),
        "mean_abs_tmin_change_all_c": float(np.mean(np.abs(dmin))),
        "mean_abs_tmax_change_all_c": float(np.mean(np.abs(dmax))),
        "max_abs_tmin_change_c": float(np.max(np.abs(dmin))),
        "max_abs_tmax_change_c": float(np.max(np.abs(dmax))),
    }

                                                               
    post_bad = (
        (pd.to_numeric(out["tmin_c"]).to_numpy(float) > tmean)
        | (tmean > pd.to_numeric(out["tmax_c"]).to_numpy(float))
    )
    if post_bad.any():
        raise RuntimeError("La reparacion minima no logro orden termico.")

    return out, rec


def monthly_clim(
    qhist: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[int, float]:
    x = qhist.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.normalize()
    x = x[
        (x["date"] >= start)
        & (x["date"] <= end)
    ].copy()
    x["month"] = x["date"].dt.month

    g = x.groupby("month")["Q_m3s"].mean()

    if set(g.index) != set(range(1, 13)):
        raise RuntimeError(
            f"Climatologia Q incompleta: meses={list(g.index)}"
        )

    return {int(m): float(g.loc[m]) for m in range(1, 13)}


def level_anchor_date() -> pd.Timestamp:
    require(LEVEL_FILE, "nivel_canonico")
    x = pd.read_csv(LEVEL_FILE, low_memory=False)

    date_col = None
    for c in ["date", "fecha"]:
        if c in x.columns:
            date_col = c
            break

    if date_col is None:
        raise KeyError(f"nivel_canonico sin date/fecha: {list(x.columns)}")

    d = pd.to_datetime(x[date_col], errors="coerce").dropna()
    if len(d) == 0:
        raise RuntimeError("nivel_canonico sin fechas validas.")

    return pd.Timestamp(d.max()).normalize()


def q_difference_stats(
    q_frozen: pd.DataFrame,
    q_repaired: pd.DataFrame,
) -> dict:
    z = q_frozen.merge(
        q_repaired,
        on="date",
        how="outer",
        suffixes=("_frozen", "_repair"),
        indicator=True,
    )

    if not (z["_merge"] == "both").all():
        raise RuntimeError(
            f"Desalineacion Q: {z['_merge'].value_counts().to_dict()}"
        )

    d = (
        z["Q_m3s_repair"].to_numpy(float)
        - z["Q_m3s_frozen"].to_numpy(float)
    )
    ad = np.abs(d)
    changed = ad > EPS_CHANGE

    return {
        "n": int(len(z)),
        "changed_n": int(changed.sum()),
        "changed_pct": float(100.0 * changed.mean()),
        "mae_m3s": float(np.mean(ad)),
        "p95_abs_m3s": float(np.quantile(ad, 0.95)),
        "p99_abs_m3s": float(np.quantile(ad, 0.99)),
        "max_abs_m3s": float(np.max(ad)),
        "bias_repair_minus_frozen_m3s": float(np.mean(d)),
    }


def level_q_sensitivity(
    q_frozen: pd.DataFrame,
    q_repaired: pd.DataFrame,
    clim_frozen: dict[int, float],
    clim_repaired: dict[int, float],
    anchor_date: pd.Timestamp,
    model: str,
    scenario: str,
) -> list[dict]:
    z = q_frozen.merge(
        q_repaired,
        on="date",
        how="inner",
        suffixes=("_frozen", "_repair"),
    ).sort_values("date")

    z = z[z["date"] > anchor_date].copy()
    if len(z) == 0:
        raise RuntimeError(f"Sin Q posterior al ancla: {model}/{scenario}")

    month = z["date"].dt.month

    q0_anom = (
        z["Q_m3s_frozen"].to_numpy(float)
        - month.map(clim_frozen).to_numpy(float)
    )
    q1_anom = (
        z["Q_m3s_repair"].to_numpy(float)
        - month.map(clim_repaired).to_numpy(float)
    )

    delta_dQ_m = (
        AQ
        * (q1_anom - q0_anom)
        * 86400.0
        / AREA_M2
    )

    z["delta_level_Q_m"] = np.cumsum(delta_dQ_m)

    rows = []
    for year in HORIZONS:
        yy = z[z["date"].dt.year == year]
        if len(yy) == 0:
            continue

        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "horizon": year,
                "delta_level_mean_repair_minus_frozen_m": float(
                    yy["delta_level_Q_m"].mean()
                ),
                "delta_level_end_repair_minus_frozen_m": float(
                    yy["delta_level_Q_m"].iloc[-1]
                ),
            }
        )

    return rows


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.float_format", lambda x: f"{x:.9f}")

    print("=" * 120)
    print("PASO 15B2 - SENSIBILIDAD DE Q AL ORDEN TERMICO")
    print("=" * 120)

    require(Q_FILE, "Q_13K")
    require(Q_META, "Q_META_13K")

    qmeta = read_json(Q_META)
    qf = pd.read_pickle(Q_FILE).copy()
    qf["date"] = pd.to_datetime(
        qf["date"], errors="raise"
    ).dt.normalize()
    qf["scenario"] = qf["scenario"].astype(str).str.lower()

    if qf.duplicated(["model", "scenario", "date"]).any():
        raise RuntimeError("Q_13K tiene duplicados inesperados.")

    mod = load_13k()
    reference = mod.load_reference_base()
    reference["date"] = pd.to_datetime(
        reference["date"], errors="raise"
    ).dt.normalize()

    models = mod.load_models()
    paths = cmip_paths(qmeta)

    intercept = float(
        qmeta["total_flow_correction_10D"]["intercept"]
    )
    slope = float(
        qmeta["total_flow_correction_10D"]["slope"]
    )

    anchor_date = level_anchor_date()
    tag("ANCHOR_DATE", date=anchor_date.date())

    gcm_names = sorted(qf["model"].astype(str).unique())

    repair_rows = []
    qdiff_rows = []
    level_rows = []
    recon_guard_rows = []

    for i, model in enumerate(gcm_names, 1):
        print(
            f"\n[MODEL_START] {i:02d}/{len(gcm_names):02d} | {model}"
        )

        hist_path = paths[(model, "historical")]
        require(hist_path, f"CMIP_hist_{model}")

        hist_raw = mod.load_cmip_file(hist_path)
        hist_raw["date"] = pd.to_datetime(
            hist_raw["date"], errors="raise"
        ).dt.normalize()

        params = mod.fit_harmonization_parameters(
            reference, hist_raw
        )

        hist_base = mod.apply_harmonization(
            hist_raw, params
        )
        hist_base["date"] = pd.to_datetime(
            hist_base["date"], errors="raise"
        ).dt.normalize()

        hist_repair, rrec = repair_temperature_order(
            hist_base
        )
        repair_rows.append(
            {
                "model": model,
                "scenario": "historical",
                **rrec,
            }
        )

        fh0 = ensure_feature_date(
            mod.build_72_features(hist_base),
            hist_base,
        )
        fh1 = ensure_feature_date(
            mod.build_72_features(hist_repair),
            hist_repair,
        )

        qh0 = predict_q(
            fh0, models, intercept, slope
        ).dropna(subset=["Q_m3s"])
        qh1 = predict_q(
            fh1, models, intercept, slope
        ).dropna(subset=["Q_m3s"])

        clim0 = monthly_clim(
            qh0, PRIMARY_START, PRIMARY_END
        )
        clim1 = monthly_clim(
            qh1, PRIMARY_START, PRIMARY_END
        )

        hist_tail0 = hist_base.tail(365).copy()
        hist_tail1 = hist_repair.tail(365).copy()

        for scenario in SCENARIOS:
            fut_path = paths[(model, scenario)]
            require(fut_path, f"CMIP_{model}_{scenario}")

            fut_raw = mod.load_cmip_file(fut_path)
            fut_raw["date"] = pd.to_datetime(
                fut_raw["date"], errors="raise"
            ).dt.normalize()

            fut_base = mod.apply_harmonization(
                fut_raw, params
            )
            fut_base["date"] = pd.to_datetime(
                fut_base["date"], errors="raise"
            ).dt.normalize()

            fut_repair, frec = repair_temperature_order(
                fut_base
            )
            repair_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    **frec,
                }
            )

                                                                           
            combo0 = pd.concat(
                [hist_tail0, fut_base],
                ignore_index=True,
            )
            ff0 = ensure_feature_date(
                mod.build_72_features(combo0),
                combo0,
            )
            qr0 = predict_q(
                ff0, models, intercept, slope
            )
            qr0 = qr0[
                qr0["date"] >= fut_base["date"].min()
            ].copy()

            frozen = qf[
                (qf["model"].astype(str) == model)
                & (qf["scenario"] == scenario)
            ][["date", "q_total_corr_10d_m3s"]].rename(
                columns={
                    "q_total_corr_10d_m3s": "Q_m3s_frozen"
                }
            )

            guard = frozen.merge(
                qr0.rename(
                    columns={"Q_m3s": "Q_m3s_recon"}
                ),
                on="date",
                how="outer",
                indicator=True,
            )
            if not (guard["_merge"] == "both").all():
                raise RuntimeError(
                    f"Guard Q date mismatch {model}/{scenario}"
                )

            gd = (
                guard["Q_m3s_recon"].to_numpy(float)
                - guard["Q_m3s_frozen"].to_numpy(float)
            )
            recon_guard_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "mae": float(np.mean(np.abs(gd))),
                    "max_abs": float(np.max(np.abs(gd))),
                }
            )

                                    
            combo1 = pd.concat(
                [hist_tail1, fut_repair],
                ignore_index=True,
            )
            ff1 = ensure_feature_date(
                mod.build_72_features(combo1),
                combo1,
            )
            qr1 = predict_q(
                ff1, models, intercept, slope
            )
            qr1 = qr1[
                qr1["date"] >= fut_repair["date"].min()
            ].copy()

            q0 = frozen.rename(
                columns={"Q_m3s_frozen": "Q_m3s"}
            )
            q1 = qr1[["date", "Q_m3s"]].copy()

            qdiff_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    **q_difference_stats(q0, q1),
                }
            )

            level_rows.extend(
                level_q_sensitivity(
                    q0.rename(
                        columns={"Q_m3s": "Q_m3s_frozen"}
                    ),
                    q1.rename(
                        columns={"Q_m3s": "Q_m3s_repair"}
                    ),
                    clim0,
                    clim1,
                    anchor_date,
                    model,
                    scenario,
                )
            )

            del (
                fut_raw,
                fut_base,
                fut_repair,
                combo0,
                combo1,
                ff0,
                ff1,
                qr0,
                qr1,
                frozen,
                guard,
                q0,
                q1,
            )
            gc.collect()

        del (
            hist_raw,
            hist_base,
            hist_repair,
            fh0,
            fh1,
            qh0,
            qh1,
            hist_tail0,
            hist_tail1,
        )
        gc.collect()

    repair_df = pd.DataFrame(repair_rows)
    qdiff_df = pd.DataFrame(qdiff_rows)
    lev_df = pd.DataFrame(level_rows)
    guard_df = pd.DataFrame(recon_guard_rows)

    print("\n[Q_RECON_GUARD]")
    print(
        guard_df.sort_values(
            ["scenario", "model"]
        ).to_string(index=False)
    )
    tag(
        "Q_RECON_GUARD_GLOBAL",
        mae_max=f"{guard_df['mae'].max():.3e}",
        max_abs_global=f"{guard_df['max_abs'].max():.3e}",
    )

    print("\n[TEMP_REPAIR_COUNTS]")
    print(
        repair_df.sort_values(
            ["scenario", "bad_pct"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\n[Q_SENSITIVITY_BY_MODEL]")
    print(
        qdiff_df.sort_values(
            ["scenario", "mae_m3s"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    qsum = (
        qdiff_df.groupby("scenario")
        .agg(
            models=("model", "nunique"),
            changed_pct_median=("changed_pct", "median"),
            changed_pct_max=("changed_pct", "max"),
            mae_m3s_median=("mae_m3s", "median"),
            mae_m3s_max=("mae_m3s", "max"),
            p99_abs_m3s_max=("p99_abs_m3s", "max"),
            max_abs_m3s_global=("max_abs_m3s", "max"),
            abs_bias_m3s_max=(
                "bias_repair_minus_frozen_m3s",
                lambda x: float(np.max(np.abs(x))),
            ),
        )
        .reset_index()
    )
    print("\n[Q_SENSITIVITY_SUMMARY]")
    print(qsum.to_string(index=False))

    rows = []
    for (scenario, horizon), g in lev_df.groupby(
        ["scenario", "horizon"]
    ):
        for metric in [
            "delta_level_mean_repair_minus_frozen_m",
            "delta_level_end_repair_minus_frozen_m",
        ]:
            v = g[metric].to_numpy(float)
            rows.append(
                {
                    "scenario": scenario,
                    "horizon": int(horizon),
                    "metric": metric,
                    "median_m": float(np.median(v)),
                    "q25_m": float(np.quantile(v, 0.25)),
                    "q75_m": float(np.quantile(v, 0.75)),
                    "p05_m": float(np.quantile(v, 0.05)),
                    "p95_m": float(np.quantile(v, 0.95)),
                    "max_abs_m": float(np.max(np.abs(v))),
                }
            )

    lev_summary = pd.DataFrame(rows)
    print("\n[LEVEL_Q_ORDER_SENSITIVITY]")
    print(
        lev_summary.sort_values(
            ["scenario", "horizon", "metric"]
        ).to_string(index=False)
    )

    print("\n" + "=" * 120)
    print(
        "[END] PASO 15B2 terminado. No se escribio ningun archivo."
    )
    print(
        "[NEXT_REQUIRED] Pegar salida completa. Esta es una sensibilidad "
        "estructural no selectiva; no reemplaza ni modifica 13K."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
