# Fuentes de datos

**Autoría del proyecto:** Asociación Amigos del Lago de Atitlán en colaboración con la Autoridad para el Manejo Sustentable de la Cuenca del Lago de Atitlán y su Entorno (AMSCLAE).

## CMIP6

El Coupled Model Intercomparison Project Phase 6 proporciona simulaciones climáticas coordinadas de múltiples modelos globales. El proyecto utiliza SSP2-4.5 y SSP5-8.5 para construir forzamientos futuros condicionados por escenario.

Consulta institucional: https://wcrp-cmip.org/cmip6/

Distribución de datos: https://esgf-node.llnl.gov/projects/cmip6/

## ERA5 y ERA5-Land

El reanálisis del ECMWF se utiliza como fuente meteorológica de referencia y para variables requeridas por la estimación de evaporación.

Consulta: https://cds.climate.copernicus.eu/

## CHIRPS

Climate Hazards InfraRed Precipitation with Station data se utiliza como referencia de precipitación observacional.

Consulta: https://www.chc.ucsb.edu/data/chirps

## Sentinel-2

Las observaciones Sentinel-2 2024–2026 sustentan la máscara moderna del lago, la ocurrencia de agua y el corredor costero de transición.

Consulta: https://dataspace.copernicus.eu/

## AMSCLAE

La batimetría utilizada corresponde al levantamiento institucional disponible para el Lago de Atitlán. El archivo empleado por la cadena se conserva en `data/external/batimetria_amsclae.tif`. Su atribución corresponde a AMSCLAE y su redistribución pública debe ajustarse a las condiciones de la institución fuente.

## Datos hidrológicos históricos

Las series procesadas utilizadas por el modelo se encuentran en `data/processed`. El inventario de fuentes originales se mantiene en `data/source/README.md`.

## Trazabilidad

`MANIFEST_SHA256.csv` registra ruta, tamaño y suma SHA-256 de los archivos del repositorio para facilitar la comprobación de integridad de los paquetes de distribución.
