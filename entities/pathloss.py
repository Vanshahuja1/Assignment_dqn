# entities/pathloss.py
import math

def cost231_hata(frequency_mhz, distance_m, bs_height, ue_height, dense_urban=True):
    """
    COST-231 Hata path-loss model.
    distance_m is in meters; internally converted to km.
    """
    if distance_m < 1.0:
        distance_m = 1.0

    dnd_km = distance_m / 1000.0

    a_h = (1.1 * math.log10(frequency_mhz) - 0.7) * ue_height \
          - (1.56 * math.log10(frequency_mhz) - 0.8)

    C = 3 if dense_urban else 0

    path_loss_db = (
        46.3
        + 33.9 * math.log10(frequency_mhz)
        - 13.82 * math.log10(bs_height)
        - a_h
        + (44.9 - 6.55 * math.log10(bs_height)) * math.log10(dnd_km)  # FIXED HERE
        + C
    )

    return path_loss_db
