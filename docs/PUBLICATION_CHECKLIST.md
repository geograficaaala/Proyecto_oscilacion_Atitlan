# Lista de comprobación para publicación

**Responsable institucional del proyecto:** Asociación Amigos del Lago de Atitlán en colaboración con la Autoridad para el Manejo Sustentable de la Cuenca del Lago de Atitlán y su Entorno (AMSCLAE).

- Confirmar la licencia aplicable al código fuente.
- Confirmar las condiciones de redistribución de la batimetría AMSCLAE.
- Confirmar las condiciones de redistribución de los productos derivados de Sentinel-2, ERA5, CHIRPS y CMIP6.
- Instalar Git LFS antes de incorporar archivos binarios grandes.
- Ejecutar `python tools/verify_repository.py` antes de crear una versión pública.
- Comprobar `MANIFEST_SHA256.csv` después de transferir los paquetes.
- Mantener `CITATION.cff`, `codemeta.json` y `AUTHORS.md` junto con la versión publicada.
- Publicar el informe técnico-científico contenido en `docs/report/` como documento de referencia metodológica.
- Mantener identificados los productos 16C4, 16C5 y 18D como productos de cierre para área, integración hidromorfológica y cartografía anual.
