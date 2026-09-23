# Modelación de la oscilación del nivel del Lago de Atitlán

**Autoría institucional:** Asociación Amigos del Lago de Atitlán en colaboración con la Autoridad para el Manejo Sustentable de la Cuenca del Lago de Atitlán y su Entorno (AMSCLAE).

Repositorio científico para la modelación matemática, hidrológica, climática y geoespacial de la oscilación del nivel del Lago de Atitlán, Guatemala.

## Objetivo

El proyecto representa la evolución temporal del nivel del lago y su expresión espacial bajo forzamientos climáticos CMIP6. Las trayectorias futuras son resultados condicionados por escenario y por el ensamble de modelos climáticos. Los horizontes 2030 y 2050 constituyen los horizontes principales de análisis; 2090 se conserva como horizonte de estrés debido al incremento de la dispersión del ensamble y de la exigencia de extrapolación.

La condición inicial de la rama futura es el nivel observado de **1552.770 m** del **29 de agosto de 2026**. El balance de nivel utiliza anomalías de precipitación, evaporación y caudal, junto con un término estacional. Los coeficientes de la formulación final son **aP = 1.278816**, **aE = 2.273196** y **aQ = 0.758222**.

## Arquitectura del repositorio

```text
Atitlan_Hydrological_Model_GitHub/
├── src/                         Modelación y procesamiento científico
├── qgis/                        Carga y estilización de productos cartográficos
├── config/                      Configuración de modelos, datos y rutas
├── data/
│   ├── processed/               Series históricas procesadas
│   ├── cmip6/                   Datos climáticos CMIP6
│   ├── reference_1981_2014/     Climatologías de referencia
│   ├── evaporation/             Insumos para evaporación
│   ├── external/                Batimetría y productos espaciales de referencia
│   └── source/                  Inventario de fuentes originales
├── models/                      Modelos ajustados y preprocesadores
├── outputs/                     Resultados numéricos, espaciales y cartográficos
├── studies/                     Estudios de optimización y diagnóstico
├── docs/                        Informe, metodología y decisiones científicas
└── tools/                       Ejecución y verificación del repositorio
```

La descripción detallada de cada bloque se encuentra en `docs/ARCHITECTURE.md`.

## Cadena científica

La cadena de trabajo integra observaciones históricas de nivel, precipitación, evaporación y caudal; climatologías 1981–2014; simulaciones CMIP6 bajo SSP2-4.5 y SSP5-8.5; observación de agua mediante Sentinel-2; batimetría AMSCLAE; auditorías de consistencia costera e hipsométrica; y productos cartográficos anuales 2027–2050.

La rama temporal final se encuentra en `outputs/nivel_future_compatible`. La autoridad espacial costera se encuentra en `outputs/costa_future_compatible`. Los límites conservadores de área futura están en `outputs/area_futura_acotada_16C4`. La integración hidromorfológica final está en `outputs/hidromorfologia_future_compatible`. Los productos cartográficos anuales validados están en `outputs/productos_qgis_anuales_final_18D`.

## Resultados de referencia

| Escenario | Horizonte | p50 del nivel (m) | Cambio respecto de 2026 (m) |
|---|---:|---:|---:|
| SSP2-4.5 | 2030 | 1552.349 | -0.421 |
| SSP2-4.5 | 2050 | 1550.337 | -2.433 |
| SSP5-8.5 | 2030 | 1552.293 | -0.477 |
| SSP5-8.5 | 2050 | 1548.145 | -4.625 |

Los cuantiles p05, p50 y p95 resumen la dispersión entre modelos del ensamble. Los escenarios SSP se mantienen separados y no reciben probabilidades dentro del estudio.

## Tratamiento espacial

La máscara moderna del lago se construyó con observaciones Sentinel-2 2024–2026. El dominio costero evaluado distingue evidencia directa, soporte local AMSCLAE y zonas sin soporte suficiente. La curva hipsométrica agregada fue auditada contra la evidencia espacial. Las estimaciones puntuales de área de la etapa 16C2 fueron descartadas al detectarse una contradicción con la exposición costera demostrada. La etapa 16C4 reemplazó esas estimaciones por límites inferior y superior de área futura.

## Reproducción

Se recomienda crear el entorno Conda definido en `environment.yml`:

```bash
conda env create -f environment.yml
conda activate atitlan-hydrology
python tools/verify_repository.py
```

La cadena principal puede ejecutarse con:

```bash
python tools/run_pipeline.py
```

Para regenerar los productos finales a partir de las ramas ya disponibles:

```bash
python tools/run_final_products.py
```

La secuencia completa y las dependencias entre etapas se documentan en `docs/REPRODUCIBILITY.md`.

## QGIS

El archivo `qgis/18e_qgis_cargar_y_estilizar_tiffs_18D.py` carga y organiza los productos anuales 18D dentro de QGIS. Los GeoTIFF finales se encuentran en `outputs/productos_qgis_anuales_final_18D`.

## Datos y archivos grandes

Los archivos científicos binarios se gestionan mediante Git LFS según `.gitattributes`. Antes de incorporar los archivos al repositorio remoto:

```bash
git lfs install
git lfs track
```

La procedencia y función de las fuentes principales se documentan en `docs/DATA_SOURCES.md`. La redistribución de datos de terceros debe respetar las condiciones de sus instituciones proveedoras.

## Informe técnico-científico

El informe final del proyecto se encuentra en:

`docs/report/Oscilacion_del_lago_modelamiento.docx`

El documento desarrolla la formulación matemática, el tratamiento estadístico del ensamble, las auditorías metodológicas, las limitaciones, los resultados temporales y la interpretación cartográfica.

## Citación y autoría

La autoría institucional del proyecto corresponde a **Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE**. La información estructurada para citación se encuentra en `CITATION.cff` y `codemeta.json`.
