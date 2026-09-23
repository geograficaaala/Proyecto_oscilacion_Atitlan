from __future__ import annotations
__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"

from pathlib import Path
import json
import re
import unicodedata
import gc

import numpy as np
import pandas as pd

                                                              
                                                        
                                                
                                         
 
            
                                                                     
                                                        
                                                                             
                                                               
                                                                       
                                                              

ROOT = Path(__file__).resolve().parents[1]

Q_FILE = ROOT / "outputs" / "caudales_future_compatible" / "caudal_diario_cmip6_future_compatible.pkl"
Q_META = ROOT / "outputs" / "caudales_future_compatible" / "parametros_armonizacion_cmip6_future_compatible.json"
E_FILE = ROOT / "outputs" / "evaporacion_future_compatible" / "evaporacion_diaria_cmip6_future_compatible.pkl"
E_META = ROOT / "outputs" / "evaporacion_future_compatible" / "parametros_evaporacion_cmip6_future_compatible.json"

CLIMA_FEATURES = ROOT / "data" / "processed" / "clima_diario_features.pkl"
CLIMA_Q = ROOT / "outputs" / "caudales" / "clima_diario_features_modelo_caudal.pkl"

DATA_EXTS = {".pkl", ".pickle", ".parquet", ".csv"}


def tag(name: str, **kwargs) -> None:
    txt = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{name}] {txt}" if txt else f"[{name}]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def exists(path: Path, label: str) -> bool:
    ok = path.exists()
    if ok:
        tag("FILE", label=label, path=path, size_mb=f"{path.stat().st_size / 1024**2:.2f}")
    else:
        tag("MISSING", label=label, path=path)
    return ok


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pretty_section(meta: dict, label: str, keys: list[str]) -> None:
    for k in keys:
        if k in meta:
            print(f"\n[META_SECTION] {label}.{k}")
            txt = json.dumps(meta[k], ensure_ascii=False, indent=2, default=str)
                                                                    
            if len(txt) > 30000:
                print(txt[:30000])
                print(f"... [TRUNCATED] chars_total={len(txt)}")
            else:
                print(txt)


def dataframe_schema(path: Path, max_pickle_mb: float = 250.0):




    suf = path.suffix.lower()
    size_mb = path.stat().st_size / 1024**2

    try:
        if suf == ".csv":
            x = pd.read_csv(path, nrows=50)
            return list(x.columns), None, x, "csv_nrows50"

        if suf in {".pkl", ".pickle"}:
            if size_mb > max_pickle_mb:
                return [], None, None, f"pickle_skip_size>{max_pickle_mb:.0f}MB"
            x = pd.read_pickle(path)
            if isinstance(x, pd.DataFrame):
                sample = x.head(50).copy()
                rows = len(x)
                cols = list(x.columns)
                del x
                gc.collect()
                return cols, rows, sample, "pickle_loaded"
            return [], None, None, f"pickle_type={type(x).__name__}"

        if suf == ".parquet":
                                                                              
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                cols = list(pf.schema.names)
                rows = pf.metadata.num_rows
                x = pd.read_parquet(path, columns=cols[: min(len(cols), 30)]).head(50)
                return cols, rows, x, "parquet_schema"
            except Exception:
                x = pd.read_parquet(path)
                if isinstance(x, pd.DataFrame):
                    sample = x.head(50).copy()
                    rows = len(x)
                    cols = list(x.columns)
                    del x
                    gc.collect()
                    return cols, rows, sample, "parquet_loaded"
    except Exception as exc:
        return [], None, None, f"ERROR={type(exc).__name__}:{exc}"

    return [], None, None, "unsupported"


def find_named_col(cols: list[str], exact: list[str], tokens: list[str]) -> str | None:
    cmap = {norm(c): c for c in cols}
    for x in exact:
        if norm(x) in cmap:
            return cmap[norm(x)]

    scores = []
    for c in cols:
        n = norm(c)
        score = sum(1 for t in tokens if t in n)
        if score:
            scores.append((score, c))
    if not scores:
        return None
    scores.sort(key=lambda z: (-z[0], z[1]))
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return None
    return scores[0][1]


def classify_daily_p(path: Path) -> dict:
    cols, rows, sample, note = dataframe_schema(path)
    out = {
        "path": str(path),
        "size_mb": path.stat().st_size / 1024**2,
        "rows": rows,
        "note": note,
        "columns": cols,
        "date_col": None,
        "model_col": None,
        "scenario_col": None,
        "p_col": None,
        "daily_plausible": False,
    }

    if not cols:
        return out

    dcol = find_named_col(cols, ["date", "fecha"], ["date", "fecha", "time"])
    mcol = find_named_col(cols, ["model", "gcm", "modelo"], ["model", "gcm", "modelo"])
    scol = find_named_col(cols, ["scenario", "escenario"], ["scenario", "escenario", "ssp"])
    pcol = find_named_col(
        cols,
        [
            "pr_corr_mm_day",
            "pr_mm_day_corr",
            "precipitacion_corr_mm_day",
            "precip_mm_day_corr",
            "pr_mm_day",
            "precipitacion_mm_day",
        ],
        ["precip", "lluv", "pr_mm", "pr_corr", "mm_day"],
    )

    out.update(date_col=dcol, model_col=mcol, scenario_col=scol, p_col=pcol)

    if not all([dcol, mcol, scol, pcol]):
        return out

                                                
    try:
        ss = sample[[dcol, mcol, scol, pcol]].copy()
        ss[dcol] = pd.to_datetime(ss[dcol], errors="coerce")
        ss[pcol] = pd.to_numeric(ss[pcol], errors="coerce")
        parse_ok = ss[dcol].notna().mean() >= 0.8 and ss[pcol].notna().mean() >= 0.8
    except Exception:
        parse_ok = False

                                                            
    rows_ok = rows is None or rows >= 10000
    size_ok = out["size_mb"] >= 0.5

    out["daily_plausible"] = bool(parse_ok and rows_ok and size_ok)
    return out


def candidate_files() -> list[Path]:
    roots = [
        ROOT / "outputs",
        ROOT / "data" / "processed",
        ROOT / "data" / "cache",
    ]

    items = []
    seen = set()

    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in DATA_EXTS:
                continue

            full = norm(str(p))
                                                                                 
                                                                      
            relevant = (
                "cmip6" in full
                or "precip" in full
                or "lluv" in full
                or "clima" in full
                or "armon" in full
                or "future" in full
                or "12d" in full
            )
            if not relevant:
                continue

            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                items.append(p)

                                                                                 
    def score(p: Path):
        s = norm(str(p))
        v = 0
        v += 8 if "precip" in s or "lluv" in s else 0
        v += 7 if "cmip6" in s else 0
        v += 6 if "future" in s else 0
        v += 5 if "armon" in s or "corr" in s else 0
        v += 4 if "diari" in s or "daily" in s else 0
        v += 2 if "clima" in s else 0
        v -= 5 if "diagnostico" in s or "summary" in s or "resumen" in s else 0
        return (-v, str(p).lower())

    return sorted(items, key=score)[:120]


def scan_p_candidates() -> list[dict]:
    files = candidate_files()
    tag("P_SCAN_START", files_considered=len(files))

    results = []
    for i, p in enumerate(files, 1):
        info = classify_daily_p(p)
        cols_short = info["columns"][:18]
        print(
            f"[P_SCAN] {i:03d} | plausible={info['daily_plausible']} | "
            f"size_mb={info['size_mb']:.2f} | rows={info['rows']} | "
            f"date={info['date_col']} | model={info['model_col']} | "
            f"scenario={info['scenario_col']} | P={info['p_col']} | "
            f"note={info['note']} | path={info['path']}"
        )
        if info["p_col"] or "precip" in norm(info["path"]) or "lluv" in norm(info["path"]):
            print(f"[P_SCAN_COLUMNS] {cols_short}")
        results.append(info)

    good = [r for r in results if r["daily_plausible"]]
    print(f"\n[P_DAILY_CANDIDATES] n={len(good)}")
    for r in good:
        print(
            f"  path={r['path']}\n"
            f"  rows={r['rows']} | size_mb={r['size_mb']:.2f} | "
            f"date={r['date_col']} | model={r['model_col']} | "
            f"scenario={r['scenario_col']} | P={r['p_col']}\n"
        )

    if len(good) == 1:
        print(f"[P_RESOLVED_UNIQUE] {good[0]['path']}")
    elif len(good) == 0:
        print("[P_RESOLVED_UNIQUE] NONE")
    else:
        print("[P_RESOLVED_UNIQUE] MULTIPLE")

    return good


def inspect_core_dataframe(path: Path, label: str) -> None:
    if not exists(path, label):
        return

    try:
        x = pd.read_pickle(path)
    except Exception as exc:
        tag("READ_ERROR", label=label, error=f"{type(exc).__name__}:{exc}")
        return

    if not isinstance(x, pd.DataFrame):
        tag("TYPE", label=label, type=type(x).__name__)
        return

    tag("SHAPE", label=label, rows=len(x), cols=len(x.columns))
    print(f"[SCHEMA] {label} columns={list(x.columns)}")

    dcol = find_named_col(list(x.columns), ["date", "fecha"], ["date", "fecha", "time"])
    mcol = find_named_col(list(x.columns), ["model", "gcm", "modelo"], ["model", "gcm", "modelo"])
    scol = find_named_col(list(x.columns), ["scenario", "escenario"], ["scenario", "escenario", "ssp"])

    if dcol:
        d = pd.to_datetime(x[dcol], errors="coerce")
        tag(
            "DATE_RANGE",
            label=label,
            valid=int(d.notna().sum()),
            date_min=d.min(),
            date_max=d.max(),
        )

    if mcol:
        tag("MODELS", label=label, n=x[mcol].astype(str).nunique())

    if scol:
        vals = sorted(x[scol].astype(str).str.lower().dropna().unique().tolist())
        tag("SCENARIOS", label=label, values=vals[:30])

    pcols = [c for c in x.columns if any(t in norm(c) for t in ("precip", "lluv", "pr_mm", "pr_corr"))]
    climcols = [
        c for c in x.columns
        if any(t in norm(c) for t in ("tas", "tmin", "tmax", "tmean", "rh", "wind", "rsds", "rlds", "solar"))
    ]
    tag("EMBEDDED_P_COLUMNS", label=label, columns=pcols)
    tag("EMBEDDED_CLIMATE_COLUMNS", label=label, n=len(climcols), columns=climcols[:80])

    del x
    gc.collect()


def inspect_feature_table(path: Path, label: str) -> None:
    if not exists(path, label):
        return
    try:
        x = pd.read_pickle(path)
    except Exception as exc:
        tag("READ_ERROR", label=label, error=f"{type(exc).__name__}:{exc}")
        return

    if not isinstance(x, pd.DataFrame):
        tag("TYPE", label=label, type=type(x).__name__)
        return

    tag("SHAPE", label=label, rows=len(x), cols=len(x.columns))
    print(f"[SCHEMA_HEAD] {label} first_columns={list(x.columns)[:80]}")

    dcol = find_named_col(list(x.columns), ["date", "fecha"], ["date", "fecha", "time"])
    if dcol:
        d = pd.to_datetime(x[dcol], errors="coerce")
        tag("DATE_RANGE", label=label, min=d.min(), max=d.max(), valid=int(d.notna().sum()))

    pcols = [c for c in x.columns if any(t in norm(c) for t in ("precip", "lluv", "pr_", "chirps", "era5"))]
    tag("PRECIP_FEATURES", label=label, n=len(pcols), columns=pcols[:120])

                                                                   
    for c in pcols[:12]:
        try:
            v = pd.to_numeric(x[c], errors="coerce")
            if v.notna().any():
                q = v.quantile([0.01, 0.5, 0.99]).to_dict()
                tag("PRECIP_FEATURE_STATS", label=label, column=c, n=int(v.notna().sum()), q=q)
        except Exception:
            pass

    del x
    gc.collect()


def relevant_scripts() -> list[Path]:
    scripts_dir = ROOT / "src"
    if not scripts_dir.exists():
        return []

    out = []
    for p in scripts_dir.glob("*.py"):
        n = norm(p.name)
        if any(t in n for t in ("12", "13k", "14c", "14d", "precip", "cmip6", "future")):
            out.append(p)
    return sorted(out, key=lambda p: p.name.lower())


def inspect_script(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        tag("SCRIPT_READ_ERROR", path=path, error=f"{type(exc).__name__}:{exc}")
        return

    keys = [
        "shrunk_monthly_75",
        "precip",
        "pr_mm_day",
        "historical_reference",
        "reference_period",
        "baseline",
        "climat",
        "1981",
        "2008",
        "2014",
        "2015",
        "to_pickle",
        "to_parquet",
        "to_csv",
        "output",
        "future_compatible",
    ]

    hits = []
    for i, line in enumerate(lines):
        n = norm(line)
        if any(k in n for k in keys):
            hits.append(i)

    if not hits:
        return

    print(f"\n[SCRIPT] {path}")
                                               
    idx = set()
    for i in hits:
        for j in range(max(0, i - 2), min(len(lines), i + 3)):
            idx.add(j)

    idx = sorted(idx)
    if len(idx) > 260:
        idx = idx[:260]
        truncated = True
    else:
        truncated = False

    prev = None
    for j in idx:
        if prev is not None and j > prev + 1:
            print("  ...")
        print(f"  L{j+1:04d}: {lines[j]}")
        prev = j

    if truncated:
        print("  ... [SCRIPT_OUTPUT_TRUNCATED]")


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 100)

    tag("CONFIG", root=ROOT, mode="READ_ONLY_NO_OUTPUT_FILES")

                                                              
                                         
                                                              
    inspect_core_dataframe(Q_FILE, "Q_13K")
    inspect_core_dataframe(E_FILE, "E_14D")

    if exists(Q_META, "Q_META_13K"):
        qmeta = read_json(Q_META)
        print(f"[META_KEYS] Q_META_13K keys={list(qmeta.keys())}")
        pretty_section(
            qmeta,
            "Q_META_13K",
            [
                "historical_reference_period",
                "future_output_period",
                "harmonization_methods",
                "precip_shrink_lambda",
                "feature_recipe",
                "total_flow_correction_10D",
                "validation_notes",
                "source_hashes",
                "outputs",
            ],
        )

                                                                      
        if "gcm_parameters" in qmeta:
            gp = qmeta["gcm_parameters"]
            print("\n[META_SECTION] Q_META_13K.gcm_parameters")
            txt = json.dumps(gp, ensure_ascii=False, indent=2, default=str)
            print(txt[:30000])
            if len(txt) > 30000:
                print(f"... [TRUNCATED] chars_total={len(txt)}")

    if exists(E_META, "E_META_14D"):
        emeta = read_json(E_META)
        print(f"[META_KEYS] E_META_14D keys={list(emeta.keys())}")
        pretty_section(
            emeta,
            "E_META_14D",
            [
                "selected_proxy",
                "radiation_scale",
                "physical_constants",
                "harmonization",
                "historical_level_audit",
                "known_limitations",
                "source_sha256",
                "daily_output_sha256",
            ],
        )

                                                              
                                                       
                                                              
    inspect_feature_table(CLIMA_FEATURES, "clima_diario_features")
    inspect_feature_table(CLIMA_Q, "clima_diario_features_modelo_caudal")

                                                              
                                                  
                                                           
                                                              
    _ = scan_p_candidates()

                                                              
                                                             
                                                                              
                                                              
    scripts = relevant_scripts()
    tag("SCRIPTS_RELEVANT", n=len(scripts), names=[p.name for p in scripts])
    for p in scripts:
        inspect_script(p)

    print("\n[END] PASO 15A0 terminado. No se escribio ningun archivo.")
    print("[NEXT_REQUIRED] Pegar la salida completa de consola. Con ella se fija la fuente diaria P y el baseline compatible antes de integrar nivel.")


if __name__ == "__main__":
    main()
