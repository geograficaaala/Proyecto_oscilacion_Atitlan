__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import ast
import hashlib
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "data" / "processed" / "nivel_canonico.csv",
    ROOT / "data" / "external" / "batimetria_amsclae.tif",
    ROOT / "outputs" / "nivel_future_compatible" / "nivel_diario_cmip6_future_compatible.pkl",
    ROOT / "outputs" / "costa_future_compatible" / "mascara_lago_actual_2024_2026.tif",
    ROOT / "outputs" / "area_futura_acotada_16C4" / "16C4_resumen_p50_area_volumen.csv",
    ROOT / "outputs" / "hidromorfologia_future_compatible" / "resumen_hidromorfologico_final.csv",
    ROOT / "outputs" / "productos_qgis_anuales_final_18D" / "tablas" / "18D_resumen_hidromorfologico_anual_2027_2050.csv",
]
missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    raise FileNotFoundError("\n".join(missing))
for folder in [ROOT / "src", ROOT / "qgis", ROOT / "tools"]:
    for path in folder.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        for line in text.splitlines():
            if re.match(r"^\s*#", line) or re.search(r"\s+#", line):
                raise RuntimeError(f"Comentario detectado: {path.name}")
        local_pattern = "C:" + chr(92) + "Users" + chr(92)
        if path.name != "verify_repository.py" and (local_pattern in text or "LAPTOP2024" in text):
            raise RuntimeError(f"Ruta local detectada: {path.name}")
annual = pd.read_csv(REQUIRED[-1])
checks = {
    ("ssp245", 2030): 1552.348610,
    ("ssp245", 2050): 1550.336502,
    ("ssp585", 2030): 1552.293235,
    ("ssp585", 2050): 1548.144740,
}
scenario_col = next(c for c in annual.columns if c.lower() == "scenario")
year_col = next(c for c in annual.columns if c.lower() in {"year", "anio", "año"})
level_col = next(c for c in annual.columns if c.lower() in {"p50_m", "level_p50_m", "nivel_p50_m", "p50"})
for key, expected in checks.items():
    scenario, year = key
    row = annual[(annual[scenario_col].astype(str).str.lower() == scenario) & (annual[year_col].astype(int) == year)]
    if len(row) != 1:
        raise RuntimeError(f"Horizonte no unico: {scenario} {year}")
    value = float(row.iloc[0][level_col])
    if abs(value - expected) > 1e-5:
        raise RuntimeError(f"Nivel congelado alterado: {scenario} {year}: {value}")
print("REPOSITORY_VERIFIED")
print(f"PYTHON={sys.version.split()[0]}")
print(f"SCRIPTS={len(list((ROOT / 'src').glob('*.py')))}")
