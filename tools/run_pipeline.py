__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "02_modelar_caudales.py",
    "03_balance_hidrico.py",
    "04_ml_residual.py",
    "05_validar_modelo.py",
    "06_preparar_cmip6.py",
    "08_preparar_referencia_1981_2014.py",
    "09_validar_compatibilidad_climatica.py",
    "10a_auditar_caudales_future_compatible.py",
    "10b_reentrenar_caudales_future_compatible_CORREGIDO.py",
    "10c_validar_caudal_total_future_compatible.py",
    "10d_calibrar_sesgo_caudal_total.py",
    "11a_diagnosticar_evaporacion_future_compatible.py",
    "11b_recalibrar_balance_con_evaporacion_future_compatible_v2.py",
    "13k_congelar_rama_future_compatible_caudales.py",
    "14d_congelar_rama_future_compatible_evaporacion.py",
    "15c_congelar_rama_future_compatible_nivel.py",
    "16b4a_mascara_maestra_lago_actual_CON_LIMPIEZA.py",
    "16b4b_superficie_litoral_observacional.py",
    "16b4c2_corredor_directo_observado.py",
    "16b4d0_auditar_soporte_AMSCLAE_extrapolacion.py",
    "16b5a_auditar_niveles_historicos_para_extender_costa.py",
    "16b6_mapa_capacidad_espacial_costera.py",
    "16b7_aplicar_niveles_futuros_2030_2050.py",
    "16b8_zonas_nueva_exposicion_litoral.py",
    "16b9_congelar_rama_costera_future_compatible.py",
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
print("PIPELINE_COMPLETE")
