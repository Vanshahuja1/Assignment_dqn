# association.py

from entities.sinr import compute_sinr_for_all
from params import ASSOC_OWN_OPERATOR_ONLY


def associate_users_to_bs(users, base_stations):
    """
    Fill each user's:
      - sinr_table (done inside compute_sinr_for_all)
      - associated_bs (bs_id)
      - associated_bs_obj (BS object)

    Return:
      user_to_bs[user_id] = best_bs_id (or None if no SINR entries)
    """
    compute_sinr_for_all(users, base_stations)

    user_to_bs = {}

    for user in users:
        # safety: if SINR table missing/empty
        if not getattr(user, "sinr_table", None):
            user.associated_bs = None
            user.associated_bs_obj = None
            user_to_bs[user.id] = None
            continue

        if ASSOC_OWN_OPERATOR_ONLY:
            # consider only same-operator BSs
            candidates = {
                bs_id: info
                for bs_id, info in user.sinr_table.items()
                if info["bs"].operator_id == user.operator_id
            }
        else:
            candidates = user.sinr_table

        if not candidates:
            user.associated_bs = None
            user.associated_bs_obj = None
            user_to_bs[user.id] = None
            continue

        # pick max SINR
        best_bs_id, best_info = max(candidates.items(), key=lambda kv: kv[1]["sinr"])

        user.associated_bs = best_bs_id
        user.associated_bs_obj = best_info["bs"]

        # IMPORTANT: return mapping
        user_to_bs[user.id] = best_bs_id

    return user_to_bs
