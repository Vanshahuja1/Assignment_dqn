# modules/bs.py

class BaseStation:
    def __init__(self, id, operator_id, location, bs_type, tx_power):
        self.id = id
        self.operator_id = operator_id
        self.location = location          # (x, y) in meters
        self.bs_type = bs_type            # "MACRO"/"MICRO"/...
        self.tx_power = tx_power          # in Watt

        # Will be filled after edge servers are generated:
        # BS -> nearest edge server (same operator)
        self.mapped_edge_id = None

    def __str__(self):
        return (
            f"BS {self.id} | Op {self.operator_id} | Type {self.bs_type} | "
            f"Loc {self.location} | Ptx {self.tx_power} W | Edge {self.mapped_edge_id}"
        )
