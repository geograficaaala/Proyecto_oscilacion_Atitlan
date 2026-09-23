# Arquitectura del proyecto

**Autoría institucional:** Asociación Amigos del Lago de Atitlán en colaboración con la Autoridad para el Manejo Sustentable de la Cuenca del Lago de Atitlán y su Entorno (AMSCLAE).

## Organización funcional

`src/` contiene la cadena científica ejecutable. Los scripts están ordenados según las etapas históricas, de compatibilidad climática, proyección futura, análisis costero, hipsometría e integración final.

`config/` concentra configuraciones de balance hídrico, modelos de caudal, corrección residual, CMIP6, rutas y validación. Las rutas del repositorio se resuelven de forma relativa a la raíz del proyecto.

`data/` contiene los insumos requeridos por la cadena reproducible. Los datos históricos procesados se separan de CMIP6, climatologías de referencia, insumos de evaporación, archivos externos y fuentes originales.

`models/` conserva modelos ajustados y objetos de preprocesamiento necesarios para reproducir las ramas históricas y futuras.

`outputs/` conserva los resultados por etapa. Las carpetas `nivel_future_compatible`, `costa_future_compatible`, `area_futura_acotada_16C4`, `hidromorfologia_future_compatible` y `productos_qgis_anuales_final_18D` contienen los productos de cierre utilizados en el informe.

`qgis/` contiene la automatización de carga y estilización de la cartografía final.

`tools/` contiene la verificación estructural del repositorio y los lanzadores de ejecución.

`docs/` contiene el informe técnico-científico, las fuentes de datos, las decisiones científicas, la secuencia de reproducción y los criterios de publicación.

`studies/` conserva bases de estudios de optimización y diagnóstico asociadas a la selección y ajuste de modelos.

## Separación de responsabilidades

La arquitectura mantiene separados los datos de entrada, el código, los modelos ajustados y los resultados. Esta separación permite auditar la procedencia de cada producto y evita que la ejecución sobrescriba silenciosamente los insumos originales.

La rama temporal y la rama espacial se integran únicamente después de sus respectivas validaciones. El nivel futuro se obtiene a partir del balance hidrológico condicionado por CMIP6. La traducción espacial utiliza Sentinel-2 y batimetría AMSCLAE bajo controles explícitos de soporte y extrapolación.

## Productos de cierre

La rama de nivel diario future-compatible contiene las trayectorias hasta 2100. La rama costera congelada conserva el dominio evaluado, los umbrales y la incertidumbre espacial. La etapa 16C4 expresa el área futura mediante cotas conservadoras. La etapa 16C5 integra los resultados hidromorfológicos. La etapa 18D genera la serie cartográfica anual 2027–2050 y reproduce los horizontes de control 2030 y 2050.
