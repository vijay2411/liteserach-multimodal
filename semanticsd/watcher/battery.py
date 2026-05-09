"""AC / battery detection via psutil, with graceful fallbacks."""
from __future__ import annotations
import logging
from typing import Literal

log = logging.getLogger(__name__)

PowerSource = Literal["ac", "battery", "unknown"]


def power_source() -> PowerSource:
    """Return "ac" if plugged in, "battery" if running on battery,
    or "unknown" if we can't tell (no battery sensor / desktop / error)."""
    try:
        import psutil
        bat = psutil.sensors_battery()
    except Exception as e:
        log.debug("battery probe failed: %s", e)
        return "unknown"
    if bat is None:
        return "unknown"
    if bat.power_plugged is None:
        return "unknown"
    return "ac" if bat.power_plugged else "battery"


def is_on_battery() -> bool:
    """True only when we're confidently on battery. Unknown → False so the
    daemon doesn't flip into saver mode just because the sensor is missing."""
    return power_source() == "battery"
