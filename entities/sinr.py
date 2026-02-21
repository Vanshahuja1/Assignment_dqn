# entities/sinr.py
import math
from math import log10

from params import (
    BS_HEIGHT, UE_HEIGHT, NOISE_POWER_DBM,
    OPERATING_FREQUENCY, ASSOC_OWN_OPERATOR_ONLY
)
from entities.pathloss import cost231_hata


def watts_to_dbm(power_watt: float) -> float:
    return 10.0 * math.log10(power_watt * 1e3)


def bs_frequency(bs) -> float:
    # operator-wise frequency list
    return OPERATING_FREQUENCY[bs.operator_id]


def compute_sinr_for_all(users, base_stations):
    """
    Fills user.sinr_table = {bs_id: {...}} for each user.

    IMPORTANT change vs your old file:
    - If ASSOC_OWN_OPERATOR_ONLY is True, we only compute SINR to BSs of the
      user's operator (so user won't accidentally attach to other operator BS).
    """
    # Pre-group BS by operator for speed
    bs_by_op = {}
    for bs in base_stations:
        bs_by_op.setdefault(bs.operator_id, []).append(bs)

    for user in users:
        sinr_table = {}

        # Candidate BSs
        if ASSOC_OWN_OPERATOR_ONLY:
            candidate_bs_list = bs_by_op.get(user.operator_id, [])
        else:
            candidate_bs_list = base_stations

        for bs in candidate_bs_list:
            # distance user->bs
            dist = math.hypot(
                user.location[0] - bs.location[0],
                user.location[1] - bs.location[1]
            )

            # Path loss and received signal power
            path_loss_db = cost231_hata(bs_frequency(bs), dist, BS_HEIGHT, UE_HEIGHT)
            channel_gain = 10 ** (-path_loss_db / 10.0)

            rx_power_dbm = watts_to_dbm(bs.tx_power) - path_loss_db
            signal_linear_w = 10 ** ((rx_power_dbm - 30.0) / 10.0)  # Watts

            # Interference: same operator BSs only (matches your original approach)
            interference_linear_w = 0.0
            for interferer in bs_by_op.get(bs.operator_id, []):
                if interferer.id == bs.id:
                    continue

                d_int = math.hypot(
                    user.location[0] - interferer.location[0],
                    user.location[1] - interferer.location[1]
                )
                int_path_loss = cost231_hata(bs_frequency(interferer), d_int, BS_HEIGHT, UE_HEIGHT)
                int_rx_dbm = watts_to_dbm(interferer.tx_power) - int_path_loss
                interference_linear_w += 10 ** ((int_rx_dbm - 30.0) / 10.0)

            noise_linear_w = 10 ** ((NOISE_POWER_DBM - 30.0) / 10.0)

            denom = interference_linear_w + noise_linear_w
            if denom <= 0:
                sinr_linear = float("inf")
                sinr_db = 1000.0
            else:
                sinr_linear = signal_linear_w / denom
                sinr_db = 10.0 * log10(sinr_linear) if sinr_linear > 0 else -1000.0

            sinr_table[bs.id] = {
                "bs": bs,
                "sinr": sinr_db,
                "signal_power": signal_linear_w,
                "interference": interference_linear_w,
                "path_loss": path_loss_db,
                "gain": channel_gain,
            }

        user.sinr_table = sinr_table
