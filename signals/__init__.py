"""Signal library. Every family exposes compute(panel, params) -> month-end DataFrame (dates x assets).

Normalisation differs by family on purpose:
  * carry, value       : relative signals -> cross-sectional rank within asset class
  * tsmom, surprise,
    cot, enso          : directional signals -> time-series scaling (a cross-sectional rank would
                         throw away the "everything is trending up" information)
"""
from . import carry, cot, enso, surprise, tsmom, value

FAMILIES = {
    "tsmom": tsmom.compute,
    "carry": carry.compute,
    "value": value.compute,
    "surprise": surprise.compute,
    "cot": cot.compute,
    "enso": enso.compute,
}
BENCHMARK = ("tsmom", "carry", "value")
NEW = ("surprise", "cot", "enso")
