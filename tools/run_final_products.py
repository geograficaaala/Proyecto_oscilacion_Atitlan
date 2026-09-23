__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "16c1_hipsometria_total_completa.py",
    "16c2_congelar_hipsometria_future_compatible.py",
    "16c3_auditar_consistencia_costa_vs_hipsometria.py",
    "16c4_area_futura_acotada_y_volumen_retenido.py",
    "16c5_integrar_y_congelar_hidromorfologia_final.py",
    "18d_productos_qgis_anuales_finales_validados.py",
]
for name in SCRIPTS:
    path = ROOT / "src" / name
    print(f"RUN {name}", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)
print("FINAL_PRODUCTS_COMPLETE")
