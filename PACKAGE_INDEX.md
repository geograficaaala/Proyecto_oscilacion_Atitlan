# Paquetes de distribución

**Autoría institucional:** Asociación Amigos del Lago de Atitlán en colaboración con la Autoridad para el Manejo Sustentable de la Cuenca del Lago de Atitlán y su Entorno (AMSCLAE).

El repositorio completo se distribuye en varios archivos ZIP que comparten la misma carpeta raíz `Atitlan_Hydrological_Model_GitHub/`. Todos los paquetes deben extraerse en el mismo directorio para reconstruir la estructura completa.

- `01_CORE`: código, configuración, documentación, informe y herramientas.
- `02A_DATA_BASE`: datos históricos, climatologías, evaporación e insumos externos.
- `02B_CMIP6_SSP245`: archivos CMIP6 SSP2-4.5.
- `02C_CMIP6_SSP585`: archivos CMIP6 SSP5-8.5.
- `02D_CMIP6_PROCESSED`: productos CMIP6 procesados.
- `03_MODELS_STUDIES`: modelos ajustados y estudios de optimización.
- `04A_OUTPUTS_HISTORICAL`: resultados históricos, validación y compatibilidad climática.
- `04B_OUTPUTS_FLOW_FUTURE`: rama future-compatible de caudales.
- `04C_OUTPUTS_EVAP_FUTURE`: rama future-compatible de evaporación.
- `04D_OUTPUTS_LEVEL_FUTURE`: rama future-compatible de nivel.
- `05_OUTPUTS_SPATIAL`: costa, hipsometría, hidromorfología y productos QGIS.

Después de extraer todos los paquetes, ejecutar `python tools/verify_repository.py`.
