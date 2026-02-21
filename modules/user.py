# modules/user.py

class UserEquipment:
    def __init__(self, id, operator_id, location):
        self.id = id
        self.operator_id = operator_id
        self.location = location  # (x, y) in meters

        # Filled during association step
        self.associated_bs = None          # best BS id (highest SINR)
        self.associated_bs_obj = None      # best BS object
        self.sinr_table = {}               # {bs_id: {"bs": bs_obj, "sinr": sinr_db, ...}}

    def __str__(self):
        return (
            f"User {self.id} | Op {self.operator_id} | "
            f"Loc {self.location} | AssocBS {self.associated_bs}"
        )
