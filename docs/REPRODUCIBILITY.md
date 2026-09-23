# Reproducibilidad

**Autoría institucional:** Asociación Amigos del Lago de Atitlán en colaboración con la Autoridad para el Manejo Sustentable de la Cuenca del Lago de Atitlán y su Entorno (AMSCLAE).

## Cadena principal

La producción se divide en preparación histórica, modelación hidrológica, compatibilidad climática, ramas futuras, análisis costero, hipsometría, integración hidromorfológica y cartografía anual.

Los datos históricos alimentan los modelos de caudal y el balance de nivel. La referencia climática 1981–2014 permite construir anomalías comparables entre el periodo histórico y CMIP6. Las ramas future-compatible generan caudal, evaporación y nivel. La rama costera separa evidencia observada, soporte local AMSCLAE e incertidumbre espacial. La hipsometría cuantifica la geometría agregada y es sometida a una auditoría de consistencia antes de utilizarse en resultados espaciales. La etapa 16C4 produce límites de área compatibles con la evidencia costera. La etapa 18D calcula primero la media anual de cada GCM y posteriormente los cuantiles del ensamble.

## Orden de ejecución

1. `02_modelar_caudales.py`
2. `03_balance_hidrico.py`
3. `04_ml_residual.py`
4. `05_validar_modelo.py`
5. `06_preparar_cmip6.py`
6. `08_preparar_referencia_1981_2014.py`
7. `09_validar_compatibilidad_climatica.py`
8. `10a_auditar_caudales_future_compatible.py`
9. `10b_reentrenar_caudales_future_compatible_CORREGIDO.py`
10. `10c_validar_caudal_total_future_compatible.py`
11. `10d_calibrar_sesgo_caudal_total.py`
12. `11a_diagnosticar_evaporacion_future_compatible.py`
13. `11b_recalibrar_balance_con_evaporacion_future_compatible_v2.py`
14. `13k_congelar_rama_future_compatible_caudales.py`
15. `14d_congelar_rama_future_compatible_evaporacion.py`
16. `15c_congelar_rama_future_compatible_nivel.py`
17. `16b4a_mascara_maestra_lago_actual_CON_LIMPIEZA.py`
18. `16b4b_superficie_litoral_observacional.py`
19. `16b4c2_corredor_directo_observado.py`
20. `16b4d0_auditar_soporte_AMSCLAE_extrapolacion.py`
21. `16b5a_auditar_niveles_historicos_para_extender_costa.py`
22. `16b6_mapa_capacidad_espacial_costera.py`
23. `16b7_aplicar_niveles_futuros_2030_2050.py`
24. `16b8_zonas_nueva_exposicion_litoral.py`
25. `16b9_congelar_rama_costera_future_compatible.py`
26. `16c1_hipsometria_total_completa.py`
27. `16c2_congelar_hipsometria_future_compatible.py`
28. `16c3_auditar_consistencia_costa_vs_hipsometria.py`
29. `16c4_area_futura_acotada_y_volumen_retenido.py`
30. `16c5_integrar_y_congelar_hidromorfologia_final.py`
31. `18d_productos_qgis_anuales_finales_validados.py`

`01_preparar_datos.py` se conserva para reconstrucción a partir de fuentes originales. Los scripts `15a0_preflight_fuentes_climatologias.py`, `15b_auditoria_termica_rlds_e14d_v2.py` y `15b2_sensibilidad_orden_termico_q.py` corresponden a auditorías auxiliares.

## Verificación

`python tools/verify_repository.py` comprueba la presencia de los productos esenciales, la sintaxis de los scripts, la ausencia de rutas locales de la estación de trabajo y la conservación de los niveles de control de 2030 y 2050.

Los valores verificados son:

- SSP2-4.5, 2030: 1552.348610 m.
- SSP2-4.5, 2050: 1550.336502 m.
- SSP5-8.5, 2030: 1552.293235 m.
- SSP5-8.5, 2050: 1548.144740 m.

## Reglas científicas

La selección de modelos históricos se realiza con información histórica. Los escenarios SSP se mantienen como trayectorias condicionadas. Los cuantiles p05, p50 y p95 resumen la dispersión del ensamble. La representación costera se limita a las zonas con soporte espacial explícito. Las zonas sin soporte permanecen clasificadas como incertidumbre espacial. El horizonte 2090 se conserva como análisis de estrés.
