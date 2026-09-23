# Validacion final del balance hidrico - Lago de Atitlan

Fecha de generacion: 2026-09-14T06:55:50.500999+00:00

## Estado metodologico

Este modulo no calibra ni entrena modelos. Evalua exclusivamente el balance fisico ya congelado.
El periodo 2024-2026 debe tratarse como holdout retrospectivo de desarrollo y no como un test prospectivo completamente virgen, porque sus resultados ya influyeron en iteraciones previas del proyecto.

Modelo seleccionado en el paso 04: `PHYSICAL_ONLY`.

## Resultado global

- Weighted MAE balance fisico: 0.1628 m
- Weighted MAE nivel congelado: 0.5047 m
- Skill MAE frente a nivel congelado: 0.677
- NSE/R2: 0.796
- Pearson r: 0.917
- Bias ponderado: -0.1085 m
- KGE 2012: 0.915

KGE y NSE se reportan como diagnosticos complementarios, no como criterios unicos de aceptacion.

## Tendencia

- Observada: -0.4769 m/año
- Simulada: -0.4460 m/año
- Diferencia absoluta: 0.0309 m/año

## Bootstrap temporal

- Bloque 5: skill mediano 0.688, IC95% [0.511, 0.789], P(skill>0)=1.000
- Bloque 10: skill mediano 0.689, IC95% [0.454, 0.798], P(skill>0)=1.000
- Bloque 15: skill mediano 0.679, IC95% [0.382, 0.805], P(skill>0)=1.000

## Diagnostico automatico

**ROBUSTO_EN_TEST_RETROSPECTIVO**

- open_loop_skill_vs_frozen_positive: PASS
- open_loop_NSE_positive: PASS
- bootstrap_95pct_skill_lower_bound_positive: PASS
- trend_direction_correct: PASS
- delta_skill_vs_zero_change_positive: PASS
- delta_residual_no_strong_autocorrelation: REVIEW

## Nota

La sensibilidad Monte Carlo de este modulo es una perturbacion local de parametros y no representa todavia una distribucion probabilistica calibrada de incertidumbre.