from __future__ import annotations
__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"

                                                                               
                                                               
                                                                               
                                                  
                                         
 
         
                                                                            
                                                                         
 
                                                                
                                                     
                                           
                                                                        
                                           
 
             
                                    
                             
                                                
                                          
                                                                               

from pathlib import Path
import importlib.util
import json
import re
import gc

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_13K = ROOT / "src" / "13k_congelar_rama_future_compatible_caudales.py"

Q_META = (
    ROOT / "outputs" / "caudales_future_compatible"
    / "parametros_armonizacion_cmip6_future_compatible.json"
)

E_FILE = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "evaporacion_diaria_cmip6_future_compatible.pkl"
)

E_META = (
    ROOT / "outputs" / "evaporacion_future_compatible"
    / "parametros_evaporacion_cmip6_future_compatible.json"
)

SCENARIOS = ["historical", "ssp245", "ssp585"]
HORIZON_YEARS = [2030, 2050, 2090]


def tag(event: str, **kwargs) -> None:
    txt = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{event}] {txt}" if txt else f"[{event}]")


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label}: no existe {path}")
    tag(
        "FILE",
        label=label,
        path=path,
        size_mb=f"{path.stat().st_size / 1024**2:.2f}",
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_13k():
    require(SCRIPT_13K, "script_13K")
    spec = importlib.util.spec_from_file_location("atitlan_13k_for_15b", SCRIPT_13K)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar 13K.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for fn in [
        "load_reference_base",
        "load_cmip_file",
        "fit_harmonization_parameters",
        "apply_harmonization",
    ]:
        if not hasattr(mod, fn):
            raise RuntimeError(f"13K no expone {fn}")

    tag("MODULE_13K", imported=True)
    return mod


def cmip_paths(qmeta: dict) -> dict[tuple[str, str], Path]:
    src = qmeta["source_hashes"]["cmip6"]
    out = {}
    for key, info in src.items():
        model, scenario = key.rsplit("__", 1)
        out[(model, scenario.lower())] = ROOT / info["path"]
    return out


def detect_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "fecha"]:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontro date/fecha. columns={list(df.columns)}")


def detect_rlds_col(df: pd.DataFrame) -> str:
    preferred = [
        "rlds_mj_m2_day",
        "rlds",
        "longwave_mj_m2_day",
        "longwave_down_mj_m2_day",
    ]
    low = {str(c).lower(): c for c in df.columns}

    for p in preferred:
        if p.lower() in low:
            return low[p.lower()]

    hits = [
        c for c in df.columns
        if "rlds" in str(c).lower()
        or ("longwave" in str(c).lower() and "down" in str(c).lower())
    ]
    if len(hits) == 1:
        return hits[0]

    raise KeyError(f"No se identifico rlds. candidates={hits}; columns={list(df.columns)}")


def extract_monthly_map(model_meta: dict, token: str) -> dict[int, float]:
                                                            
    candidates = []

    for k, v in model_meta.items():
        lk = str(k).lower()
        if token.lower() in lk and isinstance(v, dict):
            candidates.append((k, v))

                                        
    if token in model_meta and isinstance(model_meta[token], dict):
        candidates.append((token, model_meta[token]))

    def normalize_map(x):
        if not isinstance(x, dict):
            return None

                                              
        for kk in ["monthly", "monthly_shift", "monthly_scale"]:
            if kk in x and isinstance(x[kk], dict):
                x = x[kk]
                break

        out = {}
        for m in range(1, 13):
            if str(m) in x:
                out[m] = float(x[str(m)])
            elif m in x:
                out[m] = float(x[m])
            else:
                return None
        return out

    normalized = []
    for k, v in candidates:
        nm = normalize_map(v)
        if nm is not None:
            normalized.append((k, nm))

                                                
    monthly = [x for x in normalized if "monthly" in str(x[0]).lower()]
    if len(monthly) == 1:
        return monthly[0][1]
    if len(monthly) > 1:
                                                       
        vals = [tuple(x[1][m] for m in range(1, 13)) for x in monthly]
        if all(v == vals[0] for v in vals[1:]):
            return monthly[0][1]

    if len(normalized) == 1:
        return normalized[0][1]

    raise RuntimeError(
        f"No se pudo resolver mapa mensual '{token}'. "
        f"keys={list(model_meta.keys())}; candidates={[x[0] for x in normalized]}"
    )


def radiation_scale(emeta: dict) -> float:
    x = emeta["radiation_scale"]
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, dict):
        for k in ["value", "scale", "factor", "radiation_scale"]:
            if k in x and isinstance(x[k], (int, float)):
                return float(x[k])

    raise RuntimeError(f"No se pudo leer radiation_scale: {x}")


def corrected_rlds(
    raw_csv: pd.DataFrame,
    model_meta: dict,
) -> pd.DataFrame:
    dcol = detect_date_col(raw_csv)
    rcol = detect_rlds_col(raw_csv)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw_csv[dcol], errors="raise").dt.normalize()
    raw = pd.to_numeric(raw_csv[rcol], errors="coerce").to_numpy(float)

    rmap = extract_monthly_map(model_meta, "rlds")
    months = out["date"].dt.month.to_numpy(int)
    shift = np.asarray([rmap[int(m)] for m in months], dtype=float)

    out["rlds_raw"] = raw
    out["rlds_corr"] = raw + shift
    return out


def severity_stats(base: pd.DataFrame) -> dict:
    tmin = pd.to_numeric(base["tmin_c"], errors="coerce").to_numpy(float)
    tmean = pd.to_numeric(base["tmean_c"], errors="coerce").to_numpy(float)
    tmax = pd.to_numeric(base["tmax_c"], errors="coerce").to_numpy(float)

    low = tmin - tmean                          
    high = tmean - tmax                         

    bad_low = low > 0
    bad_high = high > 0
    bad = bad_low | bad_high

    sev = np.maximum(np.maximum(low, high), 0.0)
    pos = sev[bad]

    def qq(q):
        return float(np.quantile(pos, q)) if len(pos) else 0.0

    return {
        "n": int(len(base)),
        "bad_any_n": int(bad.sum()),
        "bad_any_pct": float(100.0 * bad.mean()),
        "tmin_gt_tmean_n": int(bad_low.sum()),
        "tmean_gt_tmax_n": int(bad_high.sum()),
        "severity_median_c": qq(0.50),
        "severity_p95_c": qq(0.95),
        "severity_p99_c": qq(0.99),
        "severity_max_c": float(pos.max()) if len(pos) else 0.0,
    }


def rlds_stats(x: pd.Series) -> dict:
    v = pd.to_numeric(x, errors="coerce").to_numpy(float)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return {
            "rlds_nonfinite_n": int((~np.isfinite(v)).sum()),
            "rlds_nonpositive_n": 0,
            "rlds_min": np.nan,
            "rlds_p01": np.nan,
            "rlds_median": np.nan,
            "rlds_p99": np.nan,
            "rlds_max": np.nan,
        }

    return {
        "rlds_nonfinite_n": int((~np.isfinite(v)).sum()),
        "rlds_nonpositive_n": int((finite <= 0).sum()),
        "rlds_min": float(np.min(finite)),
        "rlds_p01": float(np.quantile(finite, 0.01)),
        "rlds_median": float(np.median(finite)),
        "rlds_p99": float(np.quantile(finite, 0.99)),
        "rlds_max": float(np.max(finite)),
    }


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.float_format", lambda x: f"{x:.9f}")

    print("=" * 120)
    print("PASO 15B - AUDITORIA TERMICA + RLDS + RECONSTRUCCION EXACTA DE E_14D")
    print("=" * 120)

    for p, label in [
        (Q_META, "Q_META_13K"),
        (E_FILE, "E_14D"),
        (E_META, "E_META_14D"),
    ]:
        require(p, label)

    qmeta = read_json(Q_META)
    emeta = read_json(E_META)
    e14 = pd.read_pickle(E_FILE).copy()

    e14["date"] = pd.to_datetime(e14["date"], errors="raise").dt.normalize()
    e14["scenario"] = e14["scenario"].astype(str).str.lower()

    mod = load_13k()
    ref = mod.load_reference_base()
    ref["date"] = pd.to_datetime(ref["date"], errors="raise").dt.normalize()

    paths = cmip_paths(qmeta)
    models = sorted(e14["model"].astype(str).unique())

    scale = radiation_scale(emeta)
    pc = emeta["physical_constants"]
    albedo = float(pc["water_albedo"])
    emiss = float(pc["water_emissivity"])

    sigma = None
    for k in [
        "sigma_mj_m2_day_k4",
        "sigma",
        "stefan_boltzmann_mj_m2_day_k4",
    ]:
        if k in pc:
            sigma = float(pc[k])
            break
    if sigma is None:
        raise KeyError(f"No se encontro sigma. physical_constants={pc}")

    tag(
        "E_FORMULA",
        radiation_scale=scale,
        water_albedo=albedo,
        water_emissivity=emiss,
        sigma=sigma,
    )

    per_model = emeta["harmonization"]["per_model_parameters"]

    temp_rows = []
    rlds_rows = []
    ecmp_rows = []

    for i, model in enumerate(models, 1):
        print(f"\n[MODEL_START] {i:02d}/{len(models):02d} | {model}")

        hist_raw_for_fit = mod.load_cmip_file(paths[(model, "historical")])
        hist_raw_for_fit["date"] = pd.to_datetime(
            hist_raw_for_fit["date"], errors="raise"
        ).dt.normalize()

        params = mod.fit_harmonization_parameters(ref, hist_raw_for_fit)

        if model not in per_model:
            raise KeyError(f"14D metadata sin parametros para {model}")
        emodel = per_model[model]

        for scenario in SCENARIOS:
            path = paths[(model, scenario)]
            require(path, f"CMIP_{model}_{scenario}")

            raw13 = mod.load_cmip_file(path)
            raw13["date"] = pd.to_datetime(
                raw13["date"], errors="raise"
            ).dt.normalize()

            base = mod.apply_harmonization(raw13, params)
            base["date"] = pd.to_datetime(
                base["date"], errors="raise"
            ).dt.normalize()

                                                                           
                                                      
                                                                           
            rec = {
                "model": model,
                "scenario": scenario,
                "window": "full",
                **severity_stats(base),
            }
            temp_rows.append(rec)

            if scenario != "historical":
                for year in HORIZON_YEARS:
                    yy = base[base["date"].dt.year == year]
                    if len(yy):
                        temp_rows.append(
                            {
                                "model": model,
                                "scenario": scenario,
                                "window": str(year),
                                **severity_stats(yy),
                            }
                        )

                                                                           
                                                              
                                                                           
            raw_csv = pd.read_csv(path, low_memory=False)
            lr = corrected_rlds(raw_csv, emodel)

            rrec = {
                "model": model,
                "scenario": scenario,
                "window": "full",
                **rlds_stats(lr["rlds_corr"]),
            }
            rlds_rows.append(rrec)

            if scenario != "historical":
                for year in HORIZON_YEARS:
                    yy = lr[lr["date"].dt.year == year]
                    if len(yy):
                        rlds_rows.append(
                            {
                                "model": model,
                                "scenario": scenario,
                                "window": str(year),
                                **rlds_stats(yy["rlds_corr"]),
                            }
                        )

                                                                           
                                                
                                                                           
            m = (
                base[["date", "tmean_c", "solar_mj_m2"]]
                .merge(lr[["date", "rlds_corr"]], on="date", how="inner")
                .sort_values("date")
                .reset_index(drop=True)
            )

            if len(m) != len(base) or len(m) != len(lr):
                raise RuntimeError(
                    f"Desalineacion base/rlds {model}/{scenario}: "
                    f"base={len(base)}, rlds={len(lr)}, merge={len(m)}"
                )

            T = pd.to_numeric(m["tmean_c"], errors="coerce").to_numpy(float)
            solar = pd.to_numeric(
                m["solar_mj_m2"], errors="coerce"
            ).to_numpy(float)
            rlds = pd.to_numeric(
                m["rlds_corr"], errors="coerce"
            ).to_numpy(float)

            rn = (
                (1.0 - albedo) * solar
                + rlds
                - emiss * sigma * (T + 273.15) ** 4
            )

            ecalc = scale * 0.408 * np.maximum(rn, 0.0)

            target = e14[
                (e14["model"].astype(str) == model)
                & (e14["scenario"] == scenario)
            ][["date", "evap_mm_day"]].copy()

            cmp = m[["date"]].copy()
            cmp["E_recon"] = ecalc
            cmp["Rn"] = rn
            cmp = cmp.merge(target, on="date", how="outer", indicator=True)

            if not (cmp["_merge"] == "both").all():
                raise RuntimeError(
                    f"Desalineacion E14D {model}/{scenario}: "
                    f"{cmp['_merge'].value_counts().to_dict()}"
                )

            diff = (
                cmp["E_recon"].to_numpy(float)
                - cmp["evap_mm_day"].to_numpy(float)
            )

            ecmp_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "n": len(cmp),
                    "mae_mm_day": float(np.mean(np.abs(diff))),
                    "max_abs_mm_day": float(np.max(np.abs(diff))),
                    "bias_mm_day": float(np.mean(diff)),
                    "Rn_nonpositive_n": int((cmp["Rn"] <= 0).sum()),
                    "Rn_min": float(cmp["Rn"].min()),
                    "Rn_p01": float(cmp["Rn"].quantile(0.01)),
                    "Rn_median": float(cmp["Rn"].median()),
                    "Rn_p99": float(cmp["Rn"].quantile(0.99)),
                    "Rn_max": float(cmp["Rn"].max()),
                }
            )

            del raw13, base, raw_csv, lr, m, target, cmp
            gc.collect()

        del hist_raw_for_fit
        gc.collect()

                                                                               
                        
                                                                               
    tdf = pd.DataFrame(temp_rows)
    rdf = pd.DataFrame(rlds_rows)
    edf = pd.DataFrame(ecmp_rows)

    print("\n[TEMP_ORDER_FULL_BY_MODEL_SCENARIO]")
    print(
        tdf[tdf["window"] == "full"]
        .sort_values(["scenario", "bad_any_pct"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\n[TEMP_ORDER_HORIZONS_SUMMARY]")
    th = tdf[tdf["window"] != "full"].copy()
    if len(th):
        tsum = (
            th.groupby(["scenario", "window"])
            .agg(
                models=("model", "nunique"),
                bad_total=("bad_any_n", "sum"),
                pct_median=("bad_any_pct", "median"),
                pct_max=("bad_any_pct", "max"),
                severity_p95_max_c=("severity_p95_c", "max"),
                severity_max_c=("severity_max_c", "max"),
            )
            .reset_index()
        )
        print(tsum.to_string(index=False))

    tag(
        "TEMP_ORDER_GLOBAL",
        bad_total=int(tdf[tdf["window"] == "full"]["bad_any_n"].sum()),
        max_pct=float(tdf[tdf["window"] == "full"]["bad_any_pct"].max()),
        max_severity_c=float(
            tdf[tdf["window"] == "full"]["severity_max_c"].max()
        ),
        max_p95_severity_c=float(
            tdf[tdf["window"] == "full"]["severity_p95_c"].max()
        ),
    )

    print("\n[RLDS_FULL_BY_MODEL_SCENARIO]")
    print(
        rdf[rdf["window"] == "full"]
        .sort_values(["scenario", "model"])
        .to_string(index=False)
    )

    tag(
        "RLDS_GLOBAL",
        nonfinite_total=int(
            rdf[rdf["window"] == "full"]["rlds_nonfinite_n"].sum()
        ),
        nonpositive_total=int(
            rdf[rdf["window"] == "full"]["rlds_nonpositive_n"].sum()
        ),
        min_global=float(
            rdf[rdf["window"] == "full"]["rlds_min"].min()
        ),
        max_global=float(
            rdf[rdf["window"] == "full"]["rlds_max"].max()
        ),
    )

    print("\n[E14D_RECONSTRUCTION]")
    print(edf.sort_values(["scenario", "model"]).to_string(index=False))

    tag(
        "E14D_RECON_GLOBAL",
        mae_max=float(edf["mae_mm_day"].max()),
        max_abs_global=float(edf["max_abs_mm_day"].max()),
        abs_bias_max=float(edf["bias_mm_day"].abs().max()),
        Rn_nonpositive_total=int(edf["Rn_nonpositive_n"].sum()),
        Rn_min_global=float(edf["Rn_min"].min()),
        Rn_max_global=float(edf["Rn_max"].max()),
    )

    print("\n" + "=" * 120)
    print("[END] PASO 15B terminado. No se escribio ningun archivo.")
    print(
        "[NEXT_REQUIRED] Pegar salida completa. Con esta auditoria se decide "
        "si la rama de nivel puede congelarse sin modificar 13K/14D."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
