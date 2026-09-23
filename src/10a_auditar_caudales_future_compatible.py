                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import json
import re
import csv
import sys
import hashlib
import pickle
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "modelo_caudales.json"
MODELS_DIR = ROOT / "models" / "caudales"
OUT_DIR = ROOT / "outputs" / "caudales_future_compatible"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIBUTARIOS = [
    "Quiscab",
    "San_Francisco",
    "Tzununa",
    "La_Catarata",
    "San_Buenaventura",
]

                                                       
BLOQUEADAS_REGEX = [
    r"precip_difference",
    r"precip_abs_difference",
    r"precip_ratio_chirps_era",
    r"precip_ratio_era_chirps",
    r"precip_source_ratio",
    r"chirps.*era",
    r"era.*chirps",
]

                                                                  
                                                                        
AMBIGUAS_REGEX = [
    r"chirps",
    r"era5",
    r"era_",
    r"_era",
    r"precip_era",
]

                                                                      
                                                    
COMPATIBLES_REGEX = [
    r"^pr($|_)",
    r"precip",
    r"rain",
    r"tas",
    r"temp",
    r"tmean",
    r"tmin",
    r"tmax",
    r"dew",
    r"rh",
    r"humid",
    r"vpd",
    r"wind",
    r"solar",
    r"shortwave",
    r"longwave",
    r"radiat",
    r"rsds",
    r"rlds",
    r"month",
    r"doy",
    r"sin_",
    r"cos_",
    r"harmonic",
    r"api",
    r"wet",
    r"heavy",
    r"lag",
    r"rolling",
    r"mean",
    r"sum",
    r"std",
    r"min",
    r"max",
]


def norm(s):
    s = str(s).strip().lower()
    s = s.replace("á", "a").replace("é", "e").replace("í", "i")
    s = s.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def recursive_string_lists(obj, path="root"):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(recursive_string_lists(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        if len(obj) >= 3 and all(isinstance(x, str) for x in obj):
            found.append((path, obj))
        else:
            for i, v in enumerate(obj):
                found.extend(recursive_string_lists(v, f"{path}[{i}]"))
    return found


def tributary_from_text(text):
    nt = norm(text)
    aliases = {
        "Quiscab": ["quiscab"],
        "San_Francisco": ["san_francisco", "sanfrancisco"],
        "Tzununa": ["tzununa", "tzununa"],
        "La_Catarata": ["la_catarata", "catarata"],
        "San_Buenaventura": ["san_buenaventura", "sanbuenaventura", "buenaventura"],
    }
    for trib, vals in aliases.items():
        if any(v in nt for v in vals):
            return trib
    return None


def extract_feature_lists_from_config(cfg):
    lists = recursive_string_lists(cfg)
    by_trib = defaultdict(list)
    generic = []

    feature_key_tokens = (
        "feature", "predict", "variable", "column", "input", "x_cols", "xcol"
    )

    for path, values in lists:
        pnorm = norm(path)
                                                               
        score = sum(tok in pnorm for tok in feature_key_tokens)
                                                                                 
        if score == 0 and len(values) < 20:
            continue

        trib = tributary_from_text(path)
        if trib:
            by_trib[trib].append((path, values))
        else:
            generic.append((path, values))

                                                                                                
    generic_sorted = sorted(generic, key=lambda x: len(x[1]), reverse=True)
    generic_best = generic_sorted[0] if generic_sorted else None

    resolved = {}
    sources = {}

    for trib in TRIBUTARIOS:
        candidates = by_trib.get(trib, [])
        if candidates:
                                                                   
            path, vals = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[0]
            resolved[trib] = list(vals)
            sources[trib] = f"config:{path}"
        elif generic_best is not None:
            path, vals = generic_best
            resolved[trib] = list(vals)
            sources[trib] = f"config_generico:{path}"

    return resolved, sources, lists


def try_load_model(path):
    errors = []

            
    try:
        import joblib
        model = joblib.load(path)
        return model, "joblib", None
    except Exception as e:
        errors.append(f"joblib={type(e).__name__}: {e}")

            
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, "pickle", None
    except Exception as e:
        errors.append(f"pickle={type(e).__name__}: {e}")

    return None, None, " | ".join(errors)


def model_feature_names(model):
    if model is None:
        return None

    if hasattr(model, "feature_names_in_"):
        vals = getattr(model, "feature_names_in_")
        try:
            return [str(x) for x in vals]
        except Exception:
            pass

                             
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster.feature_names:
                return [str(x) for x in booster.feature_names]
        except Exception:
            pass

                     
    if hasattr(model, "feature_names"):
        try:
            vals = model.feature_names
            if vals:
                return [str(x) for x in vals]
        except Exception:
            pass

    return None


def model_feature_importance(model, feature_names):
    if model is None or not feature_names:
        return None

    imp = None

    if hasattr(model, "feature_importances_"):
        try:
            imp = list(map(float, model.feature_importances_))
        except Exception:
            imp = None

    if imp is not None and len(imp) == len(feature_names):
        total = sum(x for x in imp if x >= 0)
        if total > 0:
            imp = [max(x, 0.0) / total for x in imp]
        return dict(zip(feature_names, imp))

    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            scores = booster.get_score(importance_type="gain")
            arr = [float(scores.get(f, 0.0)) for f in feature_names]
            total = sum(arr)
            if total > 0:
                arr = [x / total for x in arr]
            return dict(zip(feature_names, arr))
        except Exception:
            pass

    return None


def inspect_saved_models():
    results = {}
    if not MODELS_DIR.exists():
        return results

    candidates = [
        p for p in MODELS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".joblib", ".pkl", ".pickle", ".sav"}
    ]

    for path in candidates:
        trib = tributary_from_text(path.name)
        if trib is None:
            continue

        model, loader, error = try_load_model(path)
        feats = model_feature_names(model)
        importance = model_feature_importance(model, feats)

        record = {
            "path": str(path),
            "loader": loader,
            "error": error,
            "features": feats,
            "importance": importance,
        }

                                                                          
        if trib not in results or (
            results[trib].get("features") is None and feats is not None
        ):
            results[trib] = record

    return results


def matches_any(name, patterns):
    n = norm(name)
    return any(re.search(p, n, flags=re.I) for p in patterns)


def classify_feature(name):
    n = norm(name)

    if matches_any(n, BLOQUEADAS_REGEX):
        return "BLOQUEADA_CROSS_SOURCE"

    if matches_any(n, AMBIGUAS_REGEX):
        return "AMBIGUA_SOURCE_SPECIFIC"

    if matches_any(n, COMPATIBLES_REGEX):
        return "COMPATIBLE_PROBABLE"

    return "REVISAR_MANUAL"


def blocked_importance(features, importance):
    if not importance:
        return None

    total = 0.0
    for f in features:
        if classify_feature(f) == "BLOQUEADA_CROSS_SOURCE":
            total += float(importance.get(f, 0.0))
    return total


def main():
    print("=" * 80)
    print("PASO 10A - AUDITAR PREDICTORES DE CAUDAL FUTURE-COMPATIBLE")
    print("=" * 80)
    print()
    print("Este script NO reentrena, NO modifica modelos y NO ejecuta Optuna.")
    print()

    if not CONFIG.exists():
        raise FileNotFoundError(
            f"No existe el archivo esperado:\n{CONFIG}\n"
            "No se ha modificado nada."
        )

    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    config_features, config_sources, all_lists = extract_feature_lists_from_config(cfg)
    saved_models = inspect_saved_models()

    audit_rows = []
    summary_rows = []
    manifest_models = {}

    for trib in TRIBUTARIOS:
        feats_cfg = config_features.get(trib)
        mdl = saved_models.get(trib, {})
        feats_model = mdl.get("features")

                                                                               
        if feats_model:
            features = feats_model
            feature_source = f"modelo:{mdl.get('path')}"
        elif feats_cfg:
            features = feats_cfg
            feature_source = config_sources.get(trib, "config")
        else:
            print(f"[ADVERTENCIA] {trib}: no pude resolver lista de predictores.")
            manifest_models[trib] = {
                "status": "NO_FEATURE_LIST",
                "config_source": config_sources.get(trib),
                "model_path": mdl.get("path"),
                "model_load_error": mdl.get("error"),
            }
            continue

                                               
        features = list(dict.fromkeys(map(str, features)))

        classifications = {f: classify_feature(f) for f in features}

        blocked = [f for f in features if classifications[f] == "BLOQUEADA_CROSS_SOURCE"]
        ambiguous = [f for f in features if classifications[f] == "AMBIGUA_SOURCE_SPECIFIC"]
        compatible = [f for f in features if classifications[f] == "COMPATIBLE_PROBABLE"]
        review = [f for f in features if classifications[f] == "REVISAR_MANUAL"]

        strict = [
            f for f in features
            if classifications[f] == "COMPATIBLE_PROBABLE"
        ]

        importance = mdl.get("importance")
        imp_blocked = blocked_importance(features, importance)

        for f in features:
            imp = None
            if importance is not None:
                imp = importance.get(f)

            audit_rows.append({
                "tributario": trib,
                "feature": f,
                "clasificacion": classifications[f],
                "importance_normalized": imp,
                "feature_source": feature_source,
            })

        summary_rows.append({
            "tributario": trib,
            "n_features_total": len(features),
            "n_blocked_cross_source": len(blocked),
            "n_ambiguous_source_specific": len(ambiguous),
            "n_compatible_probable": len(compatible),
            "n_review_manual": len(review),
            "n_strict_future_compatible": len(strict),
            "blocked_importance_fraction": imp_blocked,
            "blocked_importance_pct": (
                None if imp_blocked is None else 100.0 * imp_blocked
            ),
            "feature_source": feature_source,
            "model_path": mdl.get("path"),
        })

        manifest_models[trib] = {
            "status": "OK",
            "feature_source": feature_source,
            "model_path": mdl.get("path"),
            "model_loader": mdl.get("loader"),
            "model_load_error": mdl.get("error"),
            "n_features_total": len(features),
            "features_all": features,
            "features_blocked_cross_source": blocked,
            "features_ambiguous_source_specific": ambiguous,
            "features_compatible_probable": compatible,
            "features_review_manual": review,
            "features_strict_future_compatible": strict,
            "blocked_importance_fraction": imp_blocked,
        }

        print(
            f"{trib:18s} total={len(features):3d} | "
            f"bloqueadas={len(blocked):2d} | "
            f"ambiguas={len(ambiguous):2d} | "
            f"compatibles={len(compatible):2d} | "
            f"revisar={len(review):2d}"
        )
        if imp_blocked is not None:
            print(f"  importancia bloqueada ≈ {100.0 * imp_blocked:.2f}%")

    audit_csv = OUT_DIR / "10a_auditoria_predictores.csv"
    summary_csv = OUT_DIR / "10a_resumen_por_tributario.csv"
    strict_json = OUT_DIR / "10a_features_future_compatible_strict.json"
    manifest_json = OUT_DIR / "manifest_paso10a.json"

                 
    fieldnames = [
        "tributario", "feature", "clasificacion",
        "importance_normalized", "feature_source"
    ]
    with open(audit_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

                 
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_csv.write_text("", encoding="utf-8")

    strict_payload = {
        trib: info.get("features_strict_future_compatible", [])
        for trib, info in manifest_models.items()
        if info.get("status") == "OK"
    }
    with open(strict_json, "w", encoding="utf-8") as f:
        json.dump(strict_payload, f, indent=2, ensure_ascii=False)

    manifest = {
        "paso": "10A",
        "descripcion": (
            "Auditoria reproducible de predictores del modelo historico de caudales "
            "para identificar variables no reproducibles con un unico GCM."
        ),
        "python_version": sys.version,
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "models_dir": str(MODELS_DIR),
        "rules": {
            "blocked_regex": BLOQUEADAS_REGEX,
            "ambiguous_regex": AMBIGUAS_REGEX,
            "compatible_regex": COMPATIBLES_REGEX,
        },
        "models": manifest_models,
        "outputs": {
            "audit_csv": str(audit_csv),
            "summary_csv": str(summary_csv),
            "strict_features_json": str(strict_json),
        },
        "important_note": (
            "La lista strict NO debe usarse automaticamente para reentrenar hasta "
            "revisar variables AMBIGUA_SOURCE_SPECIFIC y REVISAR_MANUAL. "
            "Este paso no modifica el baseline historico."
        ),
    }

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("ARCHIVOS CREADOS")
    print("=" * 80)
    print(audit_csv)
    print(summary_csv)
    print(strict_json)
    print(manifest_json)
    print()
    print("PASO 10A COMPLETADO.")
    print("IMPORTANTE: todavía NO se reentrenó ningún modelo.")


if __name__ == "__main__":
    main()
