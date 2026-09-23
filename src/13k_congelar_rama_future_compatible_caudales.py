                       






__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILE = ROOT / 'data' / 'reference_1981_2014' / 'procesado' / 'atitlan_referencia_1981_2014.csv'
CMIP_RAW_DIR = ROOT / 'data' / 'cmip6' / 'raw'
MODEL_DIR = ROOT / 'models' / 'caudales_future_compatible'
RESULT_DIR = ROOT / 'outputs' / 'caudales_future_compatible'
CORRECTION_JSON = RESULT_DIR / '10d_parametros_correccion_caudal_total.json'
OUT_Q = RESULT_DIR / 'caudal_diario_cmip6_future_compatible.pkl'
OUT_PARAMS = RESULT_DIR / 'parametros_armonizacion_cmip6_future_compatible.json'
FALLBACK_INTERCEPT = 0.05682983145593207
FALLBACK_SLOPE = 1.1885852684704885
PRECIP_WINDOWS = [3, 7, 14, 30, 60, 90]
METEO_WINDOWS = [7, 10, 30, 60, 90]
API_HALF_LIVES = [7, 15, 30, 60]
SHRINK_75 = 0.75
EXPECTED_MODELS = 15
EXPECTED_FUTURE_TRAJECTORIES = 30

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def norm(x):
    s = str(x).strip().lower()
    for a, b in [('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ñ', 'n')]:
        s = s.replace(a, b)
    return re.sub('[^a-z0-9]+', '_', s).strip('_')

def find_best_column(df, aliases):
    ncols = {c: norm(c) for c in df.columns}
    aliases_n = [norm(a) for a in aliases]
    for alias in aliases_n:
        for c, nc in ncols.items():
            if nc == alias:
                return c
    candidates = []
    for alias in aliases_n:
        for c, nc in ncols.items():
            if nc.endswith('_' + alias):
                candidates.append((len(nc), c))
    if candidates:
        return sorted(candidates)[0][1]
    return None

def saturation_vapor_pressure_kpa(temp_c):
    t = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))

def rh_from_t_dewpoint_pct(tmean_c, dewpoint_c):
    es_t = saturation_vapor_pressure_kpa(tmean_c)
    es_td = saturation_vapor_pressure_kpa(dewpoint_c)
    rh = 100.0 * es_td / np.maximum(es_t, 1e-12)
    return np.clip(rh, 0.0, 100.0)

def load_reference_base():
    df = pd.read_csv(REFERENCE_FILE, low_memory=False)
    mapping = {'date': ['date', 'fecha'], 'precip_consensus_mm': ['precip_consensus_mm'], 'tmean_c': ['tmean_c'], 'tmin_c': ['tmin_c'], 'tmax_c': ['tmax_c'], 'wind_ms': ['wind_ms'], 'solar_mj_m2': ['solar_mj_m2'], 'dewpoint_c': ['dewpoint_c', 'dew_point_c']}
    detected = {}
    for key, aliases in mapping.items():
        col = find_best_column(df, aliases)
        if col is None:
            raise KeyError(f'No detecté {key} en referencia.')
        detected[key] = col
    out = pd.DataFrame({'date': pd.to_datetime(df[detected['date']], errors='raise').dt.normalize()})
    for key in ['precip_consensus_mm', 'tmean_c', 'tmin_c', 'tmax_c', 'wind_ms', 'solar_mj_m2', 'dewpoint_c']:
        out[key] = pd.to_numeric(df[detected[key]], errors='coerce')
    out['rh_pct'] = rh_from_t_dewpoint_pct(out['tmean_c'], out['dewpoint_c'])
    out = out.drop(columns=['dewpoint_c'])
    out['month'] = out['date'].dt.month
    if out.isna().any().any():
        raise ValueError('Referencia contiene NaN.')
    return out

def inventory_cmip():
    rows = []
    for path in sorted(CMIP_RAW_DIR.glob('*.csv')):
        head = pd.read_csv(path, nrows=20, low_memory=False)
        model_col = find_best_column(head, ['model', 'modelo', 'source_id'])
        scenario_col = find_best_column(head, ['scenario', 'escenario', 'experiment'])
        if model_col is None or scenario_col is None:
            raise KeyError(f'{path.name}: no detecté model/scenario.')
        models = head[model_col].dropna().astype(str).unique()
        scenarios = head[scenario_col].dropna().astype(str).str.lower().unique()
        if len(models) != 1 or len(scenarios) != 1:
            raise ValueError(f'{path.name}: inventario ambiguo.')
        rows.append({'path': path, 'model': models[0], 'scenario': scenarios[0]})
    return pd.DataFrame(rows)

def load_cmip_file(path):
    df = pd.read_csv(path, low_memory=False)
    date_col = find_best_column(df, ['date', 'fecha'])
    required = ['pr_mm_day', 'tas_c', 'tasmin_c', 'tasmax_c', 'rh_pct', 'sfcWind_ms', 'rsds_mj_m2_day']
    if date_col is None:
        raise KeyError(f'{path.name}: no detecté date.')
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='raise').dt.normalize()})
    for col in required:
        source = find_best_column(df, [col])
        if source is None:
            raise KeyError(f'{path.name}: falta {col}.')
        out[col] = pd.to_numeric(df[source], errors='coerce')
    out['month'] = out['date'].dt.month
    if out.isna().any().any():
        raise ValueError(f'{path.name}: contiene NaN.')
    if out['date'].duplicated().any():
        raise ValueError(f'{path.name}: fechas duplicadas.')
    return out.sort_values('date').reset_index(drop=True)

def fit_additive(ref, gcm, ref_col, gcm_col):
    global_shift = float(ref[ref_col].mean() - gcm[gcm_col].mean())
    monthly = {}
    for month in range(1, 13):
        r = ref.loc[ref['month'] == month, ref_col]
        g = gcm.loc[gcm['month'] == month, gcm_col]
        monthly[month] = float(r.mean() - g.mean())
    return (global_shift, monthly)

def fit_multiplicative(ref, gcm, ref_col, gcm_col):
    gm = float(gcm[gcm_col].mean())
    rm = float(ref[ref_col].mean())
    if gm <= 1e-12:
        raise ValueError(f'Media inválida {gcm_col}.')
    global_scale = rm / gm
    monthly = {}
    for month in range(1, 13):
        r = ref.loc[ref['month'] == month, ref_col]
        g = gcm.loc[gcm['month'] == month, gcm_col]
        gmm = float(g.mean())
        if gmm <= 1e-12:
            raise ValueError(f'Media mensual inválida {gcm_col}, mes {month}.')
        monthly[month] = float(r.mean() / gmm)
    return (global_scale, monthly)

def fit_harmonization_parameters(ref, hist):
    common = ref[['date']].merge(hist[['date']], on='date', how='inner')['date']
    r = ref.loc[ref['date'].isin(common)].sort_values('date').reset_index(drop=True)
    g = hist.loc[hist['date'].isin(common)].sort_values('date').reset_index(drop=True)
    if not r['date'].equals(g['date']):
        raise RuntimeError('Desalineación referencia/GCM historical.')
    return {'precip': fit_multiplicative(r, g, 'precip_consensus_mm', 'pr_mm_day'), 'tmean': fit_additive(r, g, 'tmean_c', 'tas_c'), 'tmin': fit_additive(r, g, 'tmin_c', 'tasmin_c'), 'tmax': fit_additive(r, g, 'tmax_c', 'tasmax_c'), 'rh': fit_additive(r, g, 'rh_pct', 'rh_pct'), 'wind': fit_multiplicative(r, g, 'wind_ms', 'sfcWind_ms'), 'solar': fit_multiplicative(r, g, 'solar_mj_m2', 'rsds_mj_m2_day')}

def month_map(months, mapping):
    return np.array([mapping[int(m)] for m in months], dtype=float)

def apply_harmonization(df, params):
    months = df['month'].to_numpy(int)
    out = pd.DataFrame({'date': df['date'].copy()})
    global_scale, monthly_scale = params['precip']
    factor = global_scale + SHRINK_75 * (month_map(months, monthly_scale) - global_scale)
    out['precip_consensus_mm'] = np.maximum(df['pr_mm_day'].to_numpy(float) * factor, 0.0)
    _, monthly_shift = params['tmean']
    out['tmean_c'] = df['tas_c'].to_numpy(float) + month_map(months, monthly_shift)
    _, monthly_shift = params['tmin']
    out['tmin_c'] = df['tasmin_c'].to_numpy(float) + month_map(months, monthly_shift)
    global_shift, monthly_shift = params['tmax']
    shift = global_shift + SHRINK_75 * (month_map(months, monthly_shift) - global_shift)
    out['tmax_c'] = df['tasmax_c'].to_numpy(float) + shift
    _, monthly_shift = params['rh']
    out['rh_pct'] = np.clip(df['rh_pct'].to_numpy(float) + month_map(months, monthly_shift), 0.0, 100.0)
    _, monthly_scale = params['wind']
    out['wind_ms'] = np.maximum(df['sfcWind_ms'].to_numpy(float) * month_map(months, monthly_scale), 0.0)
    _, monthly_scale = params['solar']
    out['solar_mj_m2'] = np.maximum(df['rsds_mj_m2_day'].to_numpy(float) * month_map(months, monthly_scale), 0.0)
    return out

def build_72_features(base):
    x = base.sort_values('date').reset_index(drop=True).copy()
    date = pd.to_datetime(x['date'], errors='raise')
    out = pd.DataFrame(index=x.index)
    out['precip_consensus_mm'] = pd.to_numeric(x['precip_consensus_mm'], errors='coerce')
    p = out['precip_consensus_mm'].shift(1)
    out['precip_prev1d_mm'] = p
    for window in PRECIP_WINDOWS:
        minp = max(1, window // 2)
        roll = p.rolling(window, min_periods=minp)
        out[f'precip_sum_{window}d'] = roll.sum()
        out[f'precip_max1d_{window}d'] = roll.max()
        out[f'wet_days_{window}d'] = (p >= 1.0).astype(float).rolling(window, min_periods=minp).sum()
        out[f'heavy10_days_{window}d'] = (p >= 10.0).astype(float).rolling(window, min_periods=minp).sum()
        out[f'heavy25_days_{window}d'] = (p >= 25.0).astype(float).rolling(window, min_periods=minp).sum()
    for half_life in API_HALF_LIVES:
        out[f'precip_api_hl{half_life}d'] = p.fillna(0.0).ewm(halflife=half_life, adjust=False).mean()
    out['precip_recent_ratio_7_30'] = out['precip_sum_7d'] / (out['precip_sum_30d'] + 1.0)
    out['precip_recent_ratio_14_60'] = out['precip_sum_14d'] / (out['precip_sum_60d'] + 1.0)
    for col in ['tmean_c', 'tmin_c', 'tmax_c', 'rh_pct', 'wind_ms', 'solar_mj_m2']:
        shifted = pd.to_numeric(x[col], errors='coerce').shift(1)
        for window in METEO_WINDOWS:
            if window in [10, 90]:
                minp = int(np.ceil(0.9 * window))
            else:
                minp = max(1, window // 2)
            roll = shifted.rolling(window, min_periods=minp)
            if col == 'solar_mj_m2':
                out[f'{col}_sum_{window}d'] = roll.sum()
            else:
                out[f'{col}_mean_{window}d'] = roll.mean()
    doy = date.dt.dayofyear.astype(float)
    for h in [1, 2]:
        angle = 2.0 * np.pi * h * doy / 365.25
        out[f'sin_doy_h{h}_q'] = np.sin(angle)
        out[f'cos_doy_h{h}_q'] = np.cos(angle)
    out.insert(0, 'date', date)
    return out

def find_estimator(obj, depth=0, seen=None):
    if seen is None:
        seen = set()
    if obj is None or depth > 8:
        return None
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)
    if hasattr(obj, 'predict') and (not isinstance(obj, (dict, list, tuple))):
        return obj
    if isinstance(obj, dict):
        for key in ['model', 'estimator', 'regressor', 'best_model', 'fitted_model']:
            if key in obj:
                candidate = find_estimator(obj[key], depth + 1, seen)
                if candidate is not None:
                    return candidate
        for value in obj.values():
            candidate = find_estimator(value, depth + 1, seen)
            if candidate is not None:
                return candidate
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            candidate = find_estimator(value, depth + 1, seen)
            if candidate is not None:
                return candidate
    return None

def find_features(obj, depth=0, seen=None):
    if seen is None:
        seen = set()
    if obj is None or depth > 8:
        return []
    oid = id(obj)
    if oid in seen:
        return []
    seen.add(oid)
    found = []
    if hasattr(obj, 'feature_names_in_'):
        try:
            found.extend((str(x) for x in obj.feature_names_in_))
        except Exception:
            pass
    if hasattr(obj, 'get_booster'):
        try:
            booster = obj.get_booster()
            if booster.feature_names:
                found.extend((str(x) for x in booster.feature_names))
        except Exception:
            pass
    if isinstance(obj, dict):
        for key in ['features', 'feature_names', 'selected_features', 'predictors', 'columns', 'model_features', 'x_columns']:
            if key in obj:
                value = obj[key]
                if isinstance(value, (list, tuple, np.ndarray, pd.Index)) and len(value) > 0 and all((isinstance(x, (str, np.str_)) for x in value)):
                    found.extend((str(x) for x in value))
        for value in obj.values():
            found.extend(find_features(value, depth + 1, seen))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(find_features(value, depth + 1, seen))
    out = []
    used = set()
    for feature in found:
        if feature not in used:
            out.append(feature)
            used.add(feature)
    return out

def infer_family(bundle, path):
    text = (str(bundle) + ' ' + path.name).lower()
    if 'rf_log' in text:
        return 'RF_LOG'
    if 'tweedie' in text:
        return 'XGB_TWEEDIE'
    estimator = find_estimator(bundle)
    if estimator is not None:
        cls = estimator.__class__.__name__.lower()
        if 'randomforest' in cls:
            return 'RF_LOG'
        if 'xgb' in cls:
            return 'XGB_TWEEDIE'
    return 'UNKNOWN'

def load_models():
    paths = sorted(MODEL_DIR.glob('*_future_compatible_72f.joblib'))
    if len(paths) != 5:
        raise RuntimeError(f'Esperaba 5 modelos y encontré {len(paths)}.')
    models = {}
    for path in paths:
        bundle = joblib.load(path)
        estimator = find_estimator(bundle)
        features = find_features(bundle)
        family = infer_family(bundle, path)
        if estimator is None:
            raise RuntimeError(f'No encontré estimador en {path.name}.')
        if len(features) != 72:
            raise RuntimeError(f'{path.name}: {len(features)} features, esperaba 72.')
        if family == 'UNKNOWN':
            raise RuntimeError(f'No identifiqué familia de {path.name}.')
        river = path.name.split('_future_compatible_72f')[0]
        models[river] = {'path': path, 'estimator': estimator, 'features': features, 'family': family}
    return models

def predict_one(info, X):
    pred = np.asarray(info['estimator'].predict(X[info['features']]), dtype=float)
    if info['family'] == 'RF_LOG':
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0)

def recursive_numeric_items(obj, prefix=''):
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            rows.extend(recursive_numeric_items(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            rows.extend(recursive_numeric_items(v, f'{prefix}[{i}]'))
    elif isinstance(obj, (int, float)) and np.isfinite(obj):
        rows.append((prefix.lower(), float(obj)))
    return rows

def load_10d():
    intercept = None
    slope = None
    if CORRECTION_JSON.exists():
        data = json.loads(CORRECTION_JSON.read_text(encoding='utf-8'))
        for key, value in recursive_numeric_items(data):
            if intercept is None and ('intercept' in key or 'intercepto' in key):
                intercept = value
            if slope is None and ('slope' in key or 'pendiente' in key or 'coef' in key) and (0.2 < abs(value) < 5.0):
                slope = value
    if intercept is None:
        intercept = FALLBACK_INTERCEPT
    if slope is None:
        slope = FALLBACK_SLOPE
    return (float(intercept), float(slope))

def params_to_jsonable(params):
    out = {}
    for variable, (global_value, monthly) in params.items():
        out[variable] = {'global': float(global_value), 'monthly': {str(month): float(value) for month, value in monthly.items()}}
    return out

def main():
    if not RESULT_DIR.exists():
        raise FileNotFoundError(f'No existe el directorio esperado:\n{RESULT_DIR}')
    ref = load_reference_base()
    inventory = inventory_cmip()
    models = load_models()
    intercept, slope = load_10d()
    hist_inv = inventory.loc[inventory['scenario'] == 'historical'].copy()
    future_inv = inventory.loc[inventory['scenario'].isin(['ssp245', 'ssp585'])].copy()
    if len(hist_inv) != EXPECTED_MODELS:
        raise RuntimeError(f'Esperaba {EXPECTED_MODELS} historical; encontré {len(hist_inv)}.')
    if len(future_inv) != EXPECTED_FUTURE_TRAJECTORIES:
        raise RuntimeError(f'Esperaba {EXPECTED_FUTURE_TRAJECTORIES} trayectorias futuras; encontré {len(future_inv)}.')
    params_cache = {}
    hist_tail_cache = {}
    params_json = {}
    for i, row in enumerate(hist_inv.itertuples(index=False), start=1):
        hist_raw = load_cmip_file(row.path)
        params = fit_harmonization_parameters(ref, hist_raw)
        hist_base = apply_harmonization(hist_raw, params)
        params_cache[row.model] = params
        hist_tail_cache[row.model] = hist_base.tail(365).copy()
        params_json[row.model] = params_to_jsonable(params)
    output_parts = []
    for i, row in enumerate(future_inv.itertuples(index=False), start=1):
        fut_raw = load_cmip_file(row.path)
        fut_base = apply_harmonization(fut_raw, params_cache[row.model])
        combined = pd.concat([hist_tail_cache[row.model], fut_base], ignore_index=True)
        features = build_72_features(combined)
        features = features.loc[features['date'] >= fut_base['date'].min()].reset_index(drop=True)
        out = pd.DataFrame({'date': features['date'], 'model': row.model, 'scenario': row.scenario})
        tributary_cols = []
        for river, info in models.items():
            X = features[info['features']]
            if X.isna().any().any():
                raise ValueError(f'{row.model} {row.scenario} {river}: NaN en predictores.')
            q = predict_one(info, features)
            col = 'q_' + norm(river) + '_m3s'
            out[col] = q
            tributary_cols.append(col)
        out['q_total_raw_m3s'] = out[tributary_cols].sum(axis=1)
        out['q_total_corr_10d_m3s'] = np.maximum(intercept + slope * out['q_total_raw_m3s'].to_numpy(float), 0.0)
        if (~np.isfinite(out['q_total_corr_10d_m3s'])).any():
            raise ValueError(f'{row.model} {row.scenario}: Q no finito.')
        if (out['q_total_corr_10d_m3s'] < 0).any():
            raise ValueError(f'{row.model} {row.scenario}: Q negativo.')
        output_parts.append(out)
    final_q = pd.concat(output_parts, ignore_index=True)
    final_q = final_q.sort_values(['model', 'scenario', 'date']).reset_index(drop=True)
    dupes = int(final_q.duplicated(['model', 'scenario', 'date']).sum())
    if dupes:
        raise RuntimeError(f'Hay {dupes} duplicados model-scenario-date.')
    groups = final_q.groupby(['model', 'scenario'])['date'].agg(['min', 'max', 'count']).reset_index()
    if len(groups) != EXPECTED_FUTURE_TRAJECTORIES:
        raise RuntimeError('Número inesperado de trayectorias en salida final.')
    final_q.to_pickle(OUT_Q)
    model_meta = {}
    for river, info in models.items():
        model_meta[river] = {'family': info['family'], 'feature_count': len(info['features']), 'model_file': str(info['path'].relative_to(ROOT)), 'sha256': sha256_file(info['path'])}
    source_hashes = {'reference': {'path': str(REFERENCE_FILE.relative_to(ROOT)), 'sha256': sha256_file(REFERENCE_FILE)}, 'cmip6': {row.model + '__' + row.scenario: {'path': str(row.path.relative_to(ROOT)), 'sha256': sha256_file(row.path)} for row in inventory.itertuples(index=False)}}
    metadata = {'script_step': '13K', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'FROZEN_FUTURE_COMPATIBLE_FLOW_BRANCH', 'interpretation': 'CMIP6 outputs are climate scenario trajectories, not deterministic weather or flow forecasts.', 'historical_reference_period': '1981-01-01 to 2014-12-31', 'future_output_period': {'min': str(final_q['date'].min().date()), 'max': str(final_q['date'].max().date())}, 'n_gcm': int(hist_inv['model'].nunique()), 'scenarios': sorted(future_inv['scenario'].unique().tolist()), 'n_trajectories': int(len(groups)), 'n_daily_rows': int(len(final_q)), 'harmonization_methods': {'precip_consensus_mm': 'SHRUNK_MONTHLY_75 multiplicative; selected in PASO 12C, final parameters fitted on 1981-2014', 'tmean_c': 'MONTHLY_SHIFT additive; selected by temporal block CV in PASO 13F', 'tmin_c': 'MONTHLY_SHIFT additive; selected by temporal block CV in PASO 13F', 'tmax_c': 'SHRUNK_MONTHLY_75 additive; selected by temporal block CV in PASO 13F', 'rh_pct': 'MONTHLY_SHIFT additive with clipping to [0,100]; selected by temporal block CV in PASO 13F', 'wind_ms': 'MONTHLY_SCALE multiplicative; selected by temporal block CV in PASO 13F', 'solar_mj_m2': 'MONTHLY_SCALE multiplicative; selected by temporal block CV in PASO 13F'}, 'precip_shrink_lambda': SHRINK_75, 'feature_recipe': {'count': 72, 'precip_lag': 'p = precip_consensus_mm.shift(1)', 'precip_windows_days': PRECIP_WINDOWS, 'precip_min_periods': 'max(1, window//2)', 'api_half_lives_days': API_HALF_LIVES, 'api_formula': 'p.fillna(0).ewm(halflife=HL, adjust=False).mean()', 'recent_ratio_7_30': 'precip_sum_7d / (precip_sum_30d + 1.0)', 'recent_ratio_14_60': 'precip_sum_14d / (precip_sum_60d + 1.0)', 'meteo_windows_days': METEO_WINDOWS, 'meteo_min_periods': {'7_30_60': 'max(1, window//2)', '10_90': 'ceil(0.9*window)'}, 'seasonal_harmonics': 'sin/cos(2*pi*h*doy/365.25), h=1,2'}, 'flow_models': model_meta, 'total_flow_correction_10D': {'formula': 'Q_corr = intercept + slope * Q_sum_5_tributaries', 'intercept': intercept, 'slope': slope, 'source_file': str(CORRECTION_JSON.relative_to(ROOT)) if CORRECTION_JSON.exists() else 'fallback constants from approved PASO 10D'}, 'validation_notes': {'13F': 'base-variable harmonization approved by temporal block cross-validation', '13G': 'historical 72-feature domain shift substantially reduced', '13H': 'future extrapolation diagnosed; temperature leaves training domain increasingly toward 2050/2090', '13I': 'tree-model flow predictions were effectively invariant to min/max clipping; saturation/extrapolation limitation retained explicitly', '13J': 'future flow trajectories numerically stable, non-negative, and without extreme blow-up'}, 'gcm_parameters': params_json, 'source_hashes': source_hashes, 'outputs': {'daily_flow_pickle': str(OUT_Q.relative_to(ROOT)), 'parameters_json': str(OUT_PARAMS.relative_to(ROOT))}}
    OUT_PARAMS.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(len(groups))
    print(len(final_q))
    print(final_q['q_total_corr_10d_m3s'].min())
    print(final_q['q_total_corr_10d_m3s'].max())
    print(dupes)
if __name__ == '__main__':
    main()
