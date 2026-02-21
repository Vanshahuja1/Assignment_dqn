# simulation.py
# Topology sanity checks + topology plot (BS + Edges + BS->Edge mapping)

from collections import defaultdict
import os
from datetime import datetime
import matplotlib.pyplot as plt

from params import NUM_OPERATORS, RANDOM_SEED
from initializer import initialize_topology


def _count_by_operator(items):
    d = defaultdict(int)
    for x in items:
        d[x.operator_id] += 1
    return dict(d)


def sanity_checks(bs_list, edge_list, bs_to_edge):
    print("\n==============================")
    print("TOPOLOGY SANITY CHECKS")
    print("==============================")

    # basic counts
    print(f"Total BS   : {len(bs_list)}")
    print(f"Total Edges: {len(edge_list)}")

    bs_by_op = _count_by_operator(bs_list)
    edge_by_op = _count_by_operator(edge_list)
    print(f"BS per operator   : {bs_by_op}")
    print(f"Edges per operator: {edge_by_op}")

    # BS per edge
    bs_per_edge = defaultdict(int)
    for bs in bs_list:
        bs_per_edge[bs.mapped_edge_id] += 1

    print("\nBS per edge (edge_id -> count):")
    for eid in sorted(bs_per_edge.keys()):
        print(f"  Edge {eid:2d}: {bs_per_edge[eid]}")

    # mapping consistency checks
    edge_by_id = {e.id: e for e in edge_list}

    # 1) every BS has a mapped_edge_id and it exists
    missing = [bs.id for bs in bs_list if bs.mapped_edge_id is None]
    if missing:
        print(f"\n❌ ERROR: {len(missing)} BS have mapped_edge_id=None. Example: {missing[:10]}")
    else:
        print("\n✅ All BS have mapped_edge_id set.")

    bad_edge_ref = [bs.id for bs in bs_list if bs.mapped_edge_id not in edge_by_id]
    if bad_edge_ref:
        print(f"❌ ERROR: {len(bad_edge_ref)} BS map to a non-existent edge. Example: {bad_edge_ref[:10]}")
    else:
        print("✅ All BS mapped edges exist.")

    # 2) BS operator == mapped edge operator (your requirement)
    wrong_op = []
    for bs in bs_list:
        e = edge_by_id.get(bs.mapped_edge_id, None)
        if e is None:
            continue
        if e.operator_id != bs.operator_id:
            wrong_op.append((bs.id, bs.operator_id, e.id, e.operator_id))

    if wrong_op:
        print(f"❌ ERROR: Found {len(wrong_op)} BS mapped to edge of different operator.")
        print("   Example (bs_id, bs_op, edge_id, edge_op):", wrong_op[:5])
    else:
        print("✅ All BS mapped to same-operator edges.")

    # 3) bs_to_edge dict matches bs.mapped_edge_id
    mismatch = []
    for bs in bs_list:
        if bs_to_edge.get(bs.id, None) != bs.mapped_edge_id:
            mismatch.append((bs.id, bs_to_edge.get(bs.id, None), bs.mapped_edge_id))
    if mismatch:
        print(f"❌ ERROR: bs_to_edge mismatch for {len(mismatch)} BS.")
        print("   Example (bs_id, dict_edge, obj_edge):", mismatch[:5])
    else:
        print("✅ bs_to_edge dict matches BS objects.")

    print("\n(Info) RANDOM_SEED used for FIXED topology:", RANDOM_SEED)
    print("(Info) Users are expected to change every run (unless you globally seed random in main runner).")
    print("==============================\n")


def plot_topology(bs_list, edge_list, out_dir="plots", draw_links=True):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"topology_bs_edges_{ts}.png")

    # group by operator
    bs_by_op = defaultdict(list)
    for bs in bs_list:
        bs_by_op[bs.operator_id].append(bs)

    edge_by_op = defaultdict(list)
    for e in edge_list:
        edge_by_op[e.operator_id].append(e)

    edge_by_id = {e.id: e for e in edge_list}

    plt.figure()
    plt.title("Fixed Topology: BS and Edge Locations (per operator)")

    # plot BS and edges per operator using different markers
    # (matplotlib default colors will differentiate; we avoid specifying colors explicitly)
    for op in range(NUM_OPERATORS):
        bss = bs_by_op.get(op, [])
        xs = [b.location[0] for b in bss]
        ys = [b.location[1] for b in bss]
        plt.scatter(xs, ys, marker="o", label=f"BS (Op{op})", alpha=0.8)

        eds = edge_by_op.get(op, [])
        ex = [e.location[0] for e in eds]
        ey = [e.location[1] for e in eds]
        plt.scatter(ex, ey, marker="X", s=120, label=f"Edge (Op{op})", alpha=1.0)

    # optional BS -> edge links (shows "coverage" visually)
    if draw_links:
        for bs in bs_list:
            e = edge_by_id.get(bs.mapped_edge_id, None)
            if e is None:
                continue
            plt.plot([bs.location[0], e.location[0]], [bs.location[1], e.location[1]], linewidth=0.5, alpha=0.3)

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"✅ Topology plot saved to: {out_path}")


def main():
    # pick any num_users; topology checks don't depend on users
    num_users = 500

    bs_list, edge_list, bs_to_edge, users = initialize_topology(num_users=num_users)

    sanity_checks(bs_list, edge_list, bs_to_edge)
    plot_topology(bs_list, edge_list, out_dir="plots", draw_links=True)


if __name__ == "__main__":
    main()
