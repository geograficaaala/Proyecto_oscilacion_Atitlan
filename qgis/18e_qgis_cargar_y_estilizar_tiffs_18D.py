__author__ = "Asociación Amigos del Lago de Atitlán en colaboración con AMSCLAE"
from pathlib import Path
import csv
import re

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsColorRampShader,
    QgsSingleBandPseudoColorRenderer,
    QgsPalettedRasterRenderer,
)
from qgis.PyQt.QtGui import QColor


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs" / "productos_qgis_anuales_final_18D"
RASTERS = PACKAGE / "rasters_anuales"
STACKS = PACKAGE / "stacks_multibanda"
TABLE = PACKAGE / "tablas" / "18D_resumen_hidromorfologico_anual_2027_2050.csv"
PROJECT_COPY = PACKAGE / "18D_cartografia_anual_estilizada.qgz"
ROOT_GROUP_NAME = "18D - Oscilacion anual Lago de Atitlan"
SAVE_PROJECT_COPY = True


def banner(text):
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


def read_ranges():
    level_vals = []
    delta_vals = []
    uncertainty_vals = []

    with open(TABLE, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            level_vals.append(float(row["level_p50_m"]))
            delta_vals.append(float(row["delta_p50_vs_2026_m"]))
            uncertainty_vals.append(float(row["uncertainty_width_p95_p05_m"]))

    return {
        "level_min": min(level_vals),
        "level_max": max(level_vals),
        "delta_min": min(delta_vals),
        "delta_max": max(0.0, max(delta_vals)),
        "uncertainty_min": 0.0,
        "uncertainty_max": max(uncertainty_vals),
    }


def categorical_renderer(layer, classes, band=1):
    items = [
        QgsPalettedRasterRenderer.Class(value, QColor(color), label)
        for value, color, label in classes
    ]
    renderer = QgsPalettedRasterRenderer(layer.dataProvider(), band, items)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


def continuous_renderer(layer, items, band=1):
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(value, QColor(color), label)
            for value, color, label in items
        ]
    )
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band, shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


def clean_old():
    project = QgsProject.instance()
    ids = [
        layer.id()
        for layer in project.mapLayers().values()
        if bool(layer.customProperty("atitlan_18d", False))
    ]
    if ids:
        project.removeMapLayers(ids)

    root = project.layerTreeRoot()
    old = root.findGroup(ROOT_GROUP_NAME)
    if old is not None:
        root.removeChildNode(old)


def year_from_name(name):
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None


def scenario_from_name(name):
    low = name.lower()
    if "ssp245" in low:
        return "SSP245"
    if "ssp585" in low:
        return "SSP585"
    if "consenso" in low:
        return "CONSENSO"
    return "OTRO"


def product_from_name(name):
    low = name.lower()

    if "estado_p50" in low:
        return "estado"
    if "robustez_p05_p95" in low:
        return "robustez"
    if "nivel_p50_m" in low:
        return "nivel"
    if "delta_nivel_vs_2026_m" in low:
        return "delta"
    if "ancho_incertidumbre_m" in low:
        return "incertidumbre"
    if "consenso" in low:
        return "consenso"

    return "otro"


def nice_name(path, is_stack):
    name = path.stem
    scenario = scenario_from_name(name)
    product = product_from_name(name)
    year = year_from_name(name)

    labels = {
        "estado": "Estado p50",
        "robustez": "Robustez p05-p95",
        "nivel": "Nivel p50",
        "delta": "Delta nivel vs 2026",
        "incertidumbre": "Ancho incertidumbre p95-p05",
        "consenso": "Consenso SSP245-SSP585",
        "otro": "Raster",
    }

    if is_stack:
        return f"STACK {scenario} - {labels[product]} - 2027-2050"

    if scenario == "CONSENSO":
        return f"{year} - {labels[product]}"

    return f"{year} - {scenario} - {labels[product]}"


def apply_style(layer, product, ranges, band=1):
    if product == "estado":
        categorical_renderer(
            layer,
            [
                (0, "#ffffff", "Fuera"),
                (1, "#2c7fb8", "Agua cierta - evidencia costera"),
                (2, "#f03b20", "Expuesto cierto - evidencia costera"),
                (3, "#225ea8", "Agua cierta - interior AMSCLAE"),
                (4, "#bd0026", "Expuesto cierto - interior AMSCLAE"),
                (5, "#bdbdbd", "Incierto / sin soporte suficiente"),
            ],
            band,
        )
        return

    if product == "robustez":
        categorical_renderer(
            layer,
            [
                (0, "#ffffff", "Fuera"),
                (1, "#2c7fb8", "Agua robusta"),
                (2, "#fdae61", "Sensible al ensemble / intervalo vertical"),
                (3, "#d73027", "Exposicion robusta"),
                (4, "#bdbdbd", "Incertidumbre espacial"),
            ],
            band,
        )
        return

    if product == "consenso":
        categorical_renderer(
            layer,
            [
                (0, "#ffffff", "Fuera"),
                (1, "#2c7fb8", "Agua cierta en ambos escenarios"),
                (2, "#d73027", "Expuesto cierto en ambos escenarios"),
                (3, "#fdae61", "Diferencia entre escenarios"),
                (4, "#bdbdbd", "Incierto"),
            ],
            band,
        )
        return

    if product == "nivel":
        lo = ranges["level_min"]
        hi = ranges["level_max"]
        mid = (lo + hi) / 2.0

        continuous_renderer(
            layer,
            [
                (lo, "#a50026", f"{lo:.2f} m"),
                (mid, "#fee08b", f"{mid:.2f} m"),
                (hi, "#2c7bb6", f"{hi:.2f} m"),
            ],
            band,
        )
        return

    if product == "delta":
        lo = ranges["delta_min"]
        hi = ranges["delta_max"]
        mid = lo / 2.0

        continuous_renderer(
            layer,
            [
                (lo, "#a50026", f"{lo:.2f} m"),
                (mid, "#f46d43", f"{mid:.2f} m"),
                (hi, "#f7fbff", f"{hi:.2f} m"),
            ],
            band,
        )
        return

    if product == "incertidumbre":
        lo = ranges["uncertainty_min"]
        hi = ranges["uncertainty_max"]
        mid = hi / 2.0

        continuous_renderer(
            layer,
            [
                (lo, "#1a9850", f"{lo:.2f} m"),
                (mid, "#fee08b", f"{mid:.2f} m"),
                (hi, "#d73027", f"{hi:.2f} m"),
            ],
            band,
        )


def add_layer(path, group, ranges, visible=False, is_stack=False):
    layer = QgsRasterLayer(str(path), nice_name(path, is_stack))

    if not layer.isValid():
        print(f"[INVALID] {path}")
        return None

    layer.setCustomProperty("atitlan_18d", True)
    layer.setCustomProperty("atitlan_source", str(path))
    layer.setCustomProperty("atitlan_product", product_from_name(path.name))
    layer.setCustomProperty("atitlan_scenario", scenario_from_name(path.name))
    layer.setCustomProperty("atitlan_year", year_from_name(path.name) or "")

    QgsProject.instance().addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setItemVisibilityChecked(visible)

    product = product_from_name(path.name)
    apply_style(layer, product, ranges, 1)

    return layer


def create_group(parent, name, expanded=False):
    g = parent.addGroup(name)
    g.setExpanded(expanded)
    return g


def main():
    if not PACKAGE.exists():
        raise FileNotFoundError(PACKAGE)

    if not TABLE.exists():
        raise FileNotFoundError(TABLE)

    clean_old()
    ranges = read_ranges()

    banner("18E - CARGA Y ESTILIZACION AUTOMATICA QGIS")
    print(f"[PACKAGE] {PACKAGE}")
    print(f"[LEVEL RANGE] {ranges['level_min']:.3f} a {ranges['level_max']:.3f} m")
    print(f"[DELTA RANGE] {ranges['delta_min']:.3f} a {ranges['delta_max']:.3f} m")
    print(f"[UNCERTAINTY RANGE] {ranges['uncertainty_min']:.3f} a {ranges['uncertainty_max']:.3f} m")

    project = QgsProject.instance()
    root = project.layerTreeRoot()
    master = create_group(root, ROOT_GROUP_NAME, True)

    products = {
        "estado": create_group(master, "01 - Estado hidromorfologico p50", True),
        "robustez": create_group(master, "02 - Robustez p05-p95", False),
        "nivel": create_group(master, "03 - Nivel p50", False),
        "delta": create_group(master, "04 - Delta de nivel vs 2026", False),
        "incertidumbre": create_group(master, "05 - Incertidumbre p95-p05", False),
        "consenso": create_group(master, "06 - Consenso SSP245 vs SSP585", True),
        "otro": create_group(master, "07 - Otros rasters", False),
    }

    scenario_groups = {}

    for key in ["estado", "robustez", "nivel", "delta", "incertidumbre"]:
        scenario_groups[(key, "SSP245")] = create_group(products[key], "SSP245", False)
        scenario_groups[(key, "SSP585")] = create_group(products[key], "SSP585", False)

    annual_files = sorted(RASTERS.rglob("*.tif"))

    annual_loaded = 0

    for path in annual_files:
        product = product_from_name(path.name)
        scenario = scenario_from_name(path.name)
        year = year_from_name(path.name)

        if product == "consenso":
            group = products["consenso"]
            visible = year == 2050
        elif (product, scenario) in scenario_groups:
            group = scenario_groups[(product, scenario)]
            visible = False
        else:
            group = products["otro"]
            visible = False

        if add_layer(path, group, ranges, visible=visible, is_stack=False):
            annual_loaded += 1

    stack_master = create_group(master, "08 - Stacks multibanda 2027-2050", False)
    stack_groups = {
        "SSP245": create_group(stack_master, "SSP245", False),
        "SSP585": create_group(stack_master, "SSP585", False),
        "CONSENSO": create_group(stack_master, "Consenso", False),
        "OTRO": create_group(stack_master, "Otros", False),
    }

    stack_files = sorted(STACKS.rglob("*.tif"))
    stack_loaded = 0

    for path in stack_files:
        scenario = scenario_from_name(path.name)
        group = stack_groups.get(scenario, stack_groups["OTRO"])

        if add_layer(path, group, ranges, visible=False, is_stack=True):
            stack_loaded += 1

    products["estado"].setItemVisibilityChecked(True)
    products["consenso"].setItemVisibilityChecked(True)

    for key in ["robustez", "nivel", "delta", "incertidumbre", "otro"]:
        products[key].setItemVisibilityChecked(True)

    stack_master.setItemVisibilityChecked(True)

    master.setItemVisibilityChecked(True)

    project.setCustomVariables(
        {
            **project.customVariables(),
            "atitlan_18d_package": str(PACKAGE),
            "atitlan_anchor_2026_m": 1552.770,
            "atitlan_primary_period": "2027-2050",
        }
    )

    if SAVE_PROJECT_COPY:
        ok = project.write(str(PROJECT_COPY))
        print(f"[PROJECT COPY] {PROJECT_COPY} | saved={ok}")

    banner("FIN 18E")
    print(f"[ANNUAL TIFF LOADED] {annual_loaded}")
    print(f"[STACK TIFF LOADED] {stack_loaded}")
    print(f"[TOTAL TIFF LOADED] {annual_loaded + stack_loaded}")
    print("[VISIBLE BY DEFAULT] Consenso 2050")
    print("[STYLE CRITERION] escalas continuas fijas 2027-2050 para comparabilidad")
    print("[STATUS] QGIS_18D_STYLED_AND_LOADED")


main()
