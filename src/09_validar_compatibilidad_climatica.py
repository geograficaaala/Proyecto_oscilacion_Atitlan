                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_FILE = (
    ROOT / "data" / "reference_1981_2014" / "procesado"
    / "atitlan_referencia_1981_2014.csv"
)

OUT_DIR = ROOT / "outputs" / "compatibilidad_cmip6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_START = pd.Timestamp("1981-01-01")
EXPECTED_END = pd.Timestamp("2014-12-31")
EXPECTED_MODELS = 15

CMIP6_REQUIRED = {
    "date", "year", "month", "day", "model", "scenario",
    "pr_mm_day", "tas_c", "tasmin_c", "tasmax_c", "rh_pct",
    "rsds_mj_m2_day", "rlds_mj_m2_day", "sfcWind_ms"
}

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_cmip6_csvs():
    candidates = []
    datos_dir = ROOT / "data"

    if not datos_dir.exists():
        raise FileNotFoundError(f"No existe: {datos_dir}")

    for path in datos_dir.rglob("*.csv"):
        s = str(path).lower()
        if "reference_1981_2014" in s:
            continue

        try:
            head = pd.read_csv(path, nrows=3)
        except Exception:
            continue

        if CMIP6_REQUIRED.issubset(set(head.columns)):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No encontré CSV CMIP6 dentro de la carpeta datos del proyecto.\n"
            "El PASO 09 busca automáticamente archivos que contengan las columnas "
            "date, model, scenario, pr_mm_day, tas_c, tasmin_c, tasmax_c, rh_pct, "
            "rsds_mj_m2_day, rlds_mj_m2_day y sfcWind_ms.\n"
            "Copia primero los CSV CMIP6 procesados del PASO 06 dentro de datos."
        )

    return sorted(set(candidates))


def load_cmip6_historical(paths):
    frames = []
    fingerprints = {}

    for path in paths:
        fingerprints[str(path.relative_to(ROOT))] = sha256_file(path)

        df = pd.read_csv(path)

        if not CMIP6_REQUIRED.issubset(df.columns):
            continue

        df["scenario"] = df["scenario"].astype(str).str.lower()

        hist = df.loc[df["scenario"].eq("historical")].copy()
        if hist.empty:
            continue

        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.loc[
            hist["date"].between(EXPECTED_START, EXPECTED_END)
        ].copy()

        if not hist.empty:
            frames.append(hist)

    if not frames:
        raise ValueError(
            "Encontré archivos con estructura CMIP6, pero ninguno contiene "
            "scenario='historical' para 1981-2014."
        )

    cmip = pd.concat(frames, ignore_index=True)

    cmip = cmip.sort_values(["model", "date"]).reset_index(drop=True)

    dup = cmip.duplicated(["model", "scenario", "date"])
    if dup.any():
        examples = cmip.loc[
            dup, ["model", "scenario", "date"]
        ].head(10).to_dict("records")
        raise ValueError(
            f"Hay {int(dup.sum())} filas CMIP6 duplicadas por modelo/escenario/fecha. "
            f"Ejemplos: {examples}"
        )

    return cmip, fingerprints


def load_reference():
    if not REF_FILE.exists():
        raise FileNotFoundError(
            f"No existe la referencia del PASO 08:\n{REF_FILE}"
        )

    ref = pd.read_csv(REF_FILE)
    ref["date"] = pd.to_datetime(ref["date"], errors="raise")
    ref = ref.sort_values("date").reset_index(drop=True)

    expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="D")

    if len(ref) != len(expected):
        raise ValueError(
            f"Referencia PASO 08 tiene {len(ref)} filas; se esperaban {len(expected)}."
        )

    if ref["date"].duplicated().any():
        raise ValueError("La referencia PASO 08 contiene fechas duplicadas.")

    missing = expected.difference(pd.DatetimeIndex(ref["date"]))
    if len(missing):
        raise ValueError(
            f"La referencia PASO 08 tiene {len(missing)} fechas faltantes."
        )

                                                              
                                                                           
    a = 17.625
    b = 243.04
    gamma_t = (a * ref["tmean_c"]) / (b + ref["tmean_c"])
    gamma_td = (a * ref["dewpoint_c"]) / (b + ref["dewpoint_c"])
    ref["rh_derived_pct"] = 100.0 * np.exp(gamma_td - gamma_t)
    ref["rh_derived_pct"] = ref["rh_derived_pct"].clip(0.0, 100.0)

    return ref


def linear_trend_per_decade(df, value_col):
    x = df["year"].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan

    slope = np.polyfit(x[mask], y[mask], 1)[0]
    return float(slope * 10.0)


def annual_series(df, value_col, precip=False):
    tmp = df[["date", value_col]].copy()
    tmp["year"] = tmp["date"].dt.year

    if precip:
        out = tmp.groupby("year", as_index=False)[value_col].sum()
    else:
        out = tmp.groupby("year", as_index=False)[value_col].mean()

    return out


def basic_pair_metrics(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o = np.asarray(obs)[mask]
    s = np.asarray(sim)[mask]

    if len(o) == 0:
        return {
            "n": 0, "obs_mean": np.nan, "sim_mean": np.nan,
            "bias": np.nan, "relative_bias_pct": np.nan,
            "mae": np.nan, "rmse": np.nan, "pearson_r": np.nan
        }

    bias = float(np.mean(s - o))
    obs_mean = float(np.mean(o))
    sim_mean = float(np.mean(s))

    rel = np.nan
    if abs(obs_mean) > 1e-12:
        rel = float(100.0 * bias / obs_mean)

    r = np.nan
    if len(o) > 2 and np.std(o) > 0 and np.std(s) > 0:
        r = float(np.corrcoef(o, s)[0, 1])

    return {
        "n": int(len(o)),
        "obs_mean": obs_mean,
        "sim_mean": sim_mean,
        "bias": bias,
        "relative_bias_pct": rel,
        "mae": float(np.mean(np.abs(s - o))),
        "rmse": float(np.sqrt(np.mean((s - o) ** 2))),
        "pearson_r": r
    }


def make_long_comparison(model_df, ref):
    merged = model_df.merge(ref, on="date", how="inner", validate="one_to_one")

    mapping = {
        "precip_vs_era5": ("era_precip_mm", "pr_mm_day", True),
        "precip_vs_chirps": ("chirps_precip_mm", "pr_mm_day", True),
        "precip_vs_consensus": ("precip_consensus_mm", "pr_mm_day", True),
        "tas": ("tmean_c", "tas_c", False),
        "tasmin": ("tmin_c", "tasmin_c", False),
        "tasmax": ("tmax_c", "tasmax_c", False),
        "rh": ("rh_derived_pct", "rh_pct", False),
        "shortwave": ("solar_mj_m2", "rsds_mj_m2_day", False),
        "longwave": ("longwave_mj_m2", "rlds_mj_m2_day", False),
        "wind": ("wind_ms", "sfcWind_ms", False),
    }

    rows = []

    for variable, (obs_col, sim_col, is_precip) in mapping.items():
        m = basic_pair_metrics(
            merged[obs_col].to_numpy(float),
            merged[sim_col].to_numpy(float)
        )

        obs_ann = annual_series(merged, obs_col, precip=is_precip)
        sim_ann = annual_series(merged, sim_col, precip=is_precip)

        m["variable"] = variable
        m["obs_column"] = obs_col
        m["cmip_column"] = sim_col
        m["obs_trend_per_decade"] = linear_trend_per_decade(
            obs_ann, obs_col
        )
        m["cmip_trend_per_decade"] = linear_trend_per_decade(
            sim_ann, sim_col
        )
        m["trend_difference_per_decade"] = (
            m["cmip_trend_per_decade"] - m["obs_trend_per_decade"]
        )

        rows.append(m)

    return merged, pd.DataFrame(rows)


def monthly_metrics(merged, model):
    mapping = {
        "precip_consensus": ("precip_consensus_mm", "pr_mm_day"),
        "tas": ("tmean_c", "tas_c"),
        "tasmin": ("tmin_c", "tasmin_c"),
        "tasmax": ("tmax_c", "tasmax_c"),
        "rh": ("rh_derived_pct", "rh_pct"),
        "shortwave": ("solar_mj_m2", "rsds_mj_m2_day"),
        "longwave": ("longwave_mj_m2", "rlds_mj_m2_day"),
        "wind": ("wind_ms", "sfcWind_ms"),
    }

    out = []

    merged = merged.copy()
    merged["month_num"] = merged["date"].dt.month

    for variable, (obs_col, sim_col) in mapping.items():
        for month, g in merged.groupby("month_num"):
            m = basic_pair_metrics(
                g[obs_col].to_numpy(float),
                g[sim_col].to_numpy(float)
            )
            m.update({
                "model": model,
                "variable": variable,
                "month": int(month)
            })
            out.append(m)

    return pd.DataFrame(out)


def quantile_metrics(merged, model):
    mapping = {
        "precip_consensus": ("precip_consensus_mm", "pr_mm_day"),
        "tas": ("tmean_c", "tas_c"),
        "tasmin": ("tmin_c", "tasmin_c"),
        "tasmax": ("tmax_c", "tasmax_c"),
        "rh": ("rh_derived_pct", "rh_pct"),
        "shortwave": ("solar_mj_m2", "rsds_mj_m2_day"),
        "longwave": ("longwave_mj_m2", "rlds_mj_m2_day"),
        "wind": ("wind_ms", "sfcWind_ms"),
    }

    rows = []

    for variable, (obs_col, sim_col) in mapping.items():
        obs = merged[obs_col].to_numpy(float)
        sim = merged[sim_col].to_numpy(float)

        for q in QUANTILES:
            oq = float(np.nanquantile(obs, q))
            sq = float(np.nanquantile(sim, q))

            rows.append({
                "model": model,
                "variable": variable,
                "quantile": q,
                "obs_value": oq,
                "cmip_value": sq,
                "difference": sq - oq,
                "relative_difference_pct": (
                    np.nan if abs(oq) < 1e-12 else 100.0 * (sq - oq) / oq
                )
            })

    return pd.DataFrame(rows)


def precip_diagnostics(merged, model):
    obs = merged["precip_consensus_mm"].to_numpy(float)
    sim = merged["pr_mm_day"].to_numpy(float)

    def pct(cond):
        return 100.0 * float(np.mean(cond))

    thresholds = [0.1, 1.0, 10.0, 20.0, 50.0]

    rows = []

    for t in thresholds:
        rows.append({
            "model": model,
            "metric": f"days_precip_ge_{t:g}mm_pct",
            "reference": pct(obs >= t),
            "cmip6": pct(sim >= t)
        })

    rows.append({
        "model": model,
        "metric": "dry_days_lt_1mm_pct",
        "reference": pct(obs < 1.0),
        "cmip6": pct(sim < 1.0)
    })

    wet_obs = obs[obs >= 1.0]
    wet_sim = sim[sim >= 1.0]

    rows.append({
        "model": model,
        "metric": "wet_day_mean_mm",
        "reference": float(np.mean(wet_obs)) if len(wet_obs) else np.nan,
        "cmip6": float(np.mean(wet_sim)) if len(wet_sim) else np.nan
    })

    rows.append({
        "model": model,
        "metric": "annual_precip_mean_mm",
        "reference": float(
            merged.assign(year=merged["date"].dt.year)
            .groupby("year")["precip_consensus_mm"].sum().mean()
        ),
        "cmip6": float(
            merged.assign(year=merged["date"].dt.year)
            .groupby("year")["pr_mm_day"].sum().mean()
        )
    })

    out = pd.DataFrame(rows)
    out["difference"] = out["cmip6"] - out["reference"]
    out["relative_difference_pct"] = np.where(
        np.abs(out["reference"]) > 1e-12,
        100.0 * out["difference"] / out["reference"],
        np.nan
    )

    return out


def physical_anomalies(model_df, model):
    rh = model_df["rh_pct"]
    wind = model_df["sfcWind_ms"]

    return {
        "model": model,
        "n_rows": int(len(model_df)),
        "rh_below_0": int((rh < 0).sum()),
        "rh_above_100": int((rh > 100).sum()),
        "rh_min": float(rh.min()),
        "rh_max": float(rh.max()),
        "wind_below_0": int((wind < 0).sum()),
        "wind_min": float(wind.min()),
        "wind_max": float(wind.max()),
    }


def validate_model_dates(model_df, model):
    expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="D")
    actual = pd.DatetimeIndex(model_df["date"])

    missing = expected.difference(actual)
    extra = actual.difference(expected)

    return {
        "model": model,
        "n_rows": int(len(model_df)),
        "start": str(model_df["date"].min().date()),
        "end": str(model_df["date"].max().date()),
        "missing_dates": int(len(missing)),
        "extra_dates": int(len(extra)),
        "duplicate_dates": int(model_df["date"].duplicated().sum()),
    }


def main():
    print("=" * 78)
    print("PASO 09 - VALIDAR COMPATIBILIDAD CLIMATICA NEX-GDDP vs REFERENCIA")
    print("=" * 78)
    print()

    ref = load_reference()
    cmip_paths = detect_cmip6_csvs()

    print(f"CSV CMIP6 detectados: {len(cmip_paths)}")
    for p in cmip_paths:
        print(" -", p.relative_to(ROOT))
    print()

    cmip, cmip_fingerprints = load_cmip6_historical(cmip_paths)

    models = sorted(cmip["model"].dropna().astype(str).unique())
    print(f"Modelos historical encontrados: {len(models)}")
    print(", ".join(models))
    print()

    if len(models) != EXPECTED_MODELS:
        warnings.warn(
            f"Se esperaban {EXPECTED_MODELS} GCM y se encontraron {len(models)}."
        )

    date_qc_rows = []
    anomaly_rows = []
    overall_tables = []
    monthly_tables = []
    quantile_tables = []
    precip_tables = []

    for i, model in enumerate(models, 1):
        print(f"[{i:02d}/{len(models):02d}] {model}")

        g = cmip.loc[cmip["model"].astype(str).eq(model)].copy()

        date_qc_rows.append(validate_model_dates(g, model))
        anomaly_rows.append(physical_anomalies(g, model))

        merged, overall = make_long_comparison(g, ref)
        overall.insert(0, "model", model)
        overall_tables.append(overall)

        monthly_tables.append(monthly_metrics(merged, model))
        quantile_tables.append(quantile_metrics(merged, model))
        precip_tables.append(precip_diagnostics(merged, model))

    date_qc = pd.DataFrame(date_qc_rows)
    anomalies = pd.DataFrame(anomaly_rows)
    overall = pd.concat(overall_tables, ignore_index=True)
    monthly = pd.concat(monthly_tables, ignore_index=True)
    quantiles = pd.concat(quantile_tables, ignore_index=True)
    precip = pd.concat(precip_tables, ignore_index=True)

                                                                      
    ensemble_summary = (
        overall
        .groupby("variable", as_index=False)
        .agg(
            n_models=("model", "nunique"),
            median_relative_bias_pct=("relative_bias_pct", "median"),
            p25_relative_bias_pct=("relative_bias_pct", lambda x: x.quantile(0.25)),
            p75_relative_bias_pct=("relative_bias_pct", lambda x: x.quantile(0.75)),
            median_bias=("bias", "median"),
            median_mae=("mae", "median"),
            median_rmse=("rmse", "median"),
            median_pearson_r=("pearson_r", "median"),
            median_trend_difference_per_decade=(
                "trend_difference_per_decade", "median"
            )
        )
    )

                                                                         
    key_vars = [
        "precip_vs_consensus", "tas", "tasmin", "tasmax",
        "rh", "shortwave", "longwave", "wind"
    ]
    concise = ensemble_summary.loc[
        ensemble_summary["variable"].isin(key_vars)
    ].copy()

              
    paths = {
        "date_qc": OUT_DIR / "09_qc_fechas_por_modelo.csv",
        "anomalies": OUT_DIR / "09_anomalias_fisicas_cmip6.csv",
        "overall": OUT_DIR / "09_metricas_globales_por_modelo.csv",
        "monthly": OUT_DIR / "09_metricas_mensuales_por_modelo.csv",
        "quantiles": OUT_DIR / "09_cuantiles_por_modelo.csv",
        "precip": OUT_DIR / "09_diagnostico_precipitacion.csv",
        "ensemble": OUT_DIR / "09_resumen_ensemble.csv",
        "manifest": OUT_DIR / "manifest_paso09.json",
    }

    date_qc.to_csv(paths["date_qc"], index=False)
    anomalies.to_csv(paths["anomalies"], index=False)
    overall.to_csv(paths["overall"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    quantiles.to_csv(paths["quantiles"], index=False)
    precip.to_csv(paths["precip"], index=False)
    ensemble_summary.to_csv(paths["ensemble"], index=False)

    manifest = {
        "paso": 9,
        "descripcion": (
            "Diagnostico de compatibilidad climatica entre NEX-GDDP-CMIP6 "
            "historical y referencia local ERA5-Land + CHIRPS v2, 1981-2014. "
            "No aplica bias correction."
        ),
        "periodo": {
            "inicio": str(EXPECTED_START.date()),
            "fin": str(EXPECTED_END.date()),
        },
        "reference_file": str(REF_FILE),
        "reference_sha256": sha256_file(REF_FILE),
        "cmip6_files_detected": [
            str(p.relative_to(ROOT)) for p in cmip_paths
        ],
        "cmip6_sha256": cmip_fingerprints,
        "models": models,
        "n_models": len(models),
        "outputs": {k: str(v) for k, v in paths.items() if k != "manifest"},
        "method_notes": [
            "No se modifica ningun dato CMIP6.",
            "No se aplica QM, QDM ni clipping.",
            "La humedad relativa de referencia se deriva de tmean_c y dewpoint_c "
            "solo para diagnostico.",
            "La precipitacion principal se compara contra precip_consensus_mm.",
            "Las tendencias se calculan sobre agregados anuales y se expresan por decada.",
            "Los dias secos se definen adicionalmente como precip < 1 mm/dia.",
        ],
    }

    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 78)
    print("RESUMEN DEL ENSEMBLE - DIAGNOSTICO, SIN CORRECCION")
    print("=" * 78)

    display_cols = [
        "variable",
        "median_relative_bias_pct",
        "median_bias",
        "median_mae",
        "median_pearson_r",
        "median_trend_difference_per_decade",
    ]

    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 20,
        "display.width", 180,
        "display.float_format", lambda x: f"{x:,.4f}"
    ):
        print(concise[display_cols].to_string(index=False))

    print()
    print("QC de fechas:")
    print(
        date_qc[
            ["model", "n_rows", "missing_dates", "extra_dates", "duplicate_dates"]
        ].to_string(index=False)
    )

    print()
    print("Anomalias fisicas totales:")
    print(f" hurs < 0:   {int(anomalies['rh_below_0'].sum())}")
    print(f" hurs > 100: {int(anomalies['rh_above_100'].sum())}")
    print(f" wind < 0:   {int(anomalies['wind_below_0'].sum())}")

    print()
    print("Archivos creados en:")
    print(OUT_DIR)
    for name, path in paths.items():
        print(f" - {name}: {path.name}")

    print()
    print("PASO 09 COMPLETADO.")
    print("IMPORTANTE: este paso SOLO diagnostica compatibilidad.")
    print("No decide ni aplica ninguna correccion de sesgo.")


if __name__ == "__main__":
    main()
