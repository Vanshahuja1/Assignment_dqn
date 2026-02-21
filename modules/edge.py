# modules/edge.py

class EdgeServer:
    def __init__(self, id, operator_id, location, cpu_budget_mc, mem_budget_mb):
        self.id = id
        self.operator_id = operator_id
        self.location = location  # (x, y) in meters

        self.cpu_budget_mc = cpu_budget_mc
        self.mem_budget_mb = mem_budget_mb

    def __str__(self):
        return (
            f"Edge {self.id} | Op {self.operator_id} | Loc {self.location} | "
            f"CPUbud {self.cpu_budget_mc} mc | MEMbud {self.mem_budget_mb} MB"
        )
