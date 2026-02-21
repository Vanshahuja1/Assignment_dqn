# initializer.py

import random
import math
from params import *

from modules.bs import BaseStation
from modules.user import UserEquipment
from modules.edge import EdgeServer

def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def _quota_sizes(n, m):
    base = n // m
    rem = n % m
    return [base + (1 if i < rem else 0) for i in range(m)]  # e.g., 18,5 -> [4,4,4,3,3]

def generate_base_stations(rng: random.Random):
    """
    BS placement: random uniform across the area, per operator.
    Fixed across runs because rng_topo is seeded with RANDOM_SEED.
    """
    bs_list = []
    bs_id = 0

    area_km2 = (AREA_WIDTH * AREA_HEIGHT) / 1e6

    for operator_id in range(NUM_OPERATORS):
        num_bs = int(BS_DENSITY_PER_KM2[operator_id] * area_km2)

        for _ in range(num_bs):
            x = rng.uniform(0, AREA_WIDTH)
            y = rng.uniform(0, AREA_HEIGHT)

            bs_type = rng.choices(BS_TYPES, weights=BS_TYPE_RATIO[operator_id], k=1)[0]
            tx_power = BS_TX_POWER_WATT[bs_type]

            bs_list.append(BaseStation(
                id=bs_id,
                operator_id=operator_id,
                location=(x, y),
                bs_type=bs_type,
                tx_power=tx_power
            ))
            bs_id += 1

    bs_list.sort(key=lambda b: b.id)
    return bs_list


def generate_edge_servers(rng: random.Random):
    """
    Edge placement: random uniform across the area, per operator.
    Fixed across runs because rng_topo is seeded with RANDOM_SEED.
    """
    edge_list = []
    edge_id = 0

    for operator_id in range(NUM_OPERATORS):
        num_edges = EDGE_SERVERS_PER_OPERATOR[operator_id]

        for _ in range(num_edges):
            x = rng.uniform(0, AREA_WIDTH)
            y = rng.uniform(0, AREA_HEIGHT)

            edge_list.append(EdgeServer(
                id=edge_id,
                operator_id=operator_id,
                location=(x, y),
                cpu_budget_mc=EDGE_CPU_BUDGET_MC,
                mem_budget_mb=EDGE_MEM_BUDGET_MB
            ))
            edge_id += 1

    edge_list.sort(key=lambda e: e.id)
    return edge_list


def map_bs_to_edges(bs_list, edge_list):
    """
    Balanced-nearest SAME-operator mapping:
    - enforce equal BS per edge (quota)
    - within that, choose nearest possible edge
    """
    # group by operator
    bs_by_op = {op: [] for op in range(NUM_OPERATORS)}
    edges_by_op = {op: [] for op in range(NUM_OPERATORS)}
    for bs in bs_list:
        bs_by_op[bs.operator_id].append(bs)
    for e in edge_list:
        edges_by_op[e.operator_id].append(e)

    bs_to_edge = {}

    for op in range(NUM_OPERATORS):
        bss = bs_by_op[op]
        edges = sorted(edges_by_op[op], key=lambda e: e.id)
        if not edges:
            raise ValueError(f"No edges for operator {op}")

        # quotas per edge
        quotas = _quota_sizes(len(bss), len(edges))          # sizes in edge-id order
        remaining = {edges[i].id: quotas[i] for i in range(len(edges))}

        # stable BS order (geography-aware, reduces long assignments)
        bss = sorted(bss, key=lambda b: (b.location[0], b.location[1]))

        for bs in bss:
            # candidate edges = same operator edges that still have quota
            candidates = [e for e in edges if remaining[e.id] > 0]
            if not candidates:
                raise RuntimeError("Quota assignment failed (should not happen).")

            best_edge = min(candidates, key=lambda e: _dist(bs.location, e.location))

            bs.mapped_edge_id = best_edge.id
            bs_to_edge[bs.id] = best_edge.id
            remaining[best_edge.id] -= 1

    return bs_to_edge


def generate_users(num_users, ratio=USER_OPERATOR_RATIO):
    """
    Users should CHANGE every run:
    - uses global random (do NOT call random.seed(RANDOM_SEED) in your runner)
    """
    user_list = []
    total_ratio = sum(ratio)

    num_users_op0 = int(num_users * ratio[0] / total_ratio)
    num_users_op1 = num_users - num_users_op0

    user_id = 0

    for _ in range(num_users_op0):
        x = random.uniform(0, AREA_WIDTH)
        y = random.uniform(0, AREA_HEIGHT)
        user_list.append(UserEquipment(id=user_id, operator_id=0, location=(x, y)))
        user_id += 1

    for _ in range(num_users_op1):
        x = random.uniform(0, AREA_WIDTH)
        y = random.uniform(0, AREA_HEIGHT)
        user_list.append(UserEquipment(id=user_id, operator_id=1, location=(x, y)))
        user_id += 1

    user_list.sort(key=lambda u: u.id)
    return user_list


def initialize_topology(num_users, ratio=USER_OPERATOR_RATIO):
    """
    Final behavior:
    - BS fixed across runs (seeded rng_topo)
    - Edges fixed across runs (seeded rng_topo)
    - BS->Edge mapping balanced (equal BS per edge)
    - Users change every run
    """
    rng_topo = random.Random(RANDOM_SEED)

    bs_list = generate_base_stations(rng_topo)
    edge_list = generate_edge_servers(rng_topo)
    bs_to_edge = map_bs_to_edges(bs_list, edge_list)

    users = generate_users(num_users, ratio=ratio)

    return bs_list, edge_list, bs_to_edge, users
