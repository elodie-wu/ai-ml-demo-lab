"""
AIML431 — Project 1: Resource Allocation
==============================================

Instructions:
  1. Read and understand the problem description in the assignment PDF.
  2. Implement the TODO sections below.
  3. Run:  python3 main.py
  4. The program reads containers.txt, vm_types.txt, pm_capacity.txt
     and outputs the solution obtained by the FF&BF/FF heuristic.
"""

import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Provided functions — DO NOT modify                          ║
# ╚═══════════════════════════════════════════════════════════════╝


def read_containers(path=None):
    """Read container sizes from a text file.

    Returns: list of ints, in the order they appear in the file.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "containers.txt")
    with open(path) as f:
        sizes = [int(x.strip()) for x in f.readline().strip().split(",") if x.strip()]
    return sizes


def read_vm_types(path=None):
    """Read VM type names and capacities from a text file.

    Returns: list of (type_name, capacity) tuples.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "vm_types.txt")
    types = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, cap = line.split(",")
            types.append((name.strip(), int(cap.strip())))
    return types


def read_pm_capacity(path=None):
    """Read the PM capacity from a text file.

    Returns: int.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "pm_capacity.txt")
    with open(path) as f:
        return int(f.readline().strip())


# containers: cid, size
# vm: vname,vcap,used,c_list
# pm: used,vm_list


def first_fit_heuristic(containers, vm_types, pm_cap):
    vms = []
    pms = []
    # to achieve best-fit we need order the vm_types firstly
    vm_types = sorted(
        [vm_type for vm_type in vm_types if vm_type[1] <= pm_cap],
        key=lambda vm_type: vm_type[1],
    )
    print(f"[first_fit_heuristic]Valid VM types: {vm_types}")
    for cid, size in enumerate(containers, 1):
        print(f"[first_fit_heuristic]Processing containers: {cid}, size={size}")
        flag = True  # whether create new vm
        for item in vms:
            if size <= (item[1] - item[2]):
                # put container into bin.
                item[2] += size
                item[3].append((cid, size))
                flag = False
                break
        if flag:
            # create a new bin
            for name, cap in vm_types:
                if cap >= size:
                    vm_item = [name, cap, size, [(cid, size)]]
                    vms.append(vm_item)
                    break
    for vname, vcap, used, c_list in vms:
        print(f"[first_fit_heuristic]Processing vms: {vname}, vcap={vcap},used={used}")
        vm_item = [vname, vcap, c_list]
        flag = True  # whether create new pm
        for pm in pms:
            if vcap <= (pm_cap - pm[0]):
                # put container into bin
                pm[0] += vcap
                pm[1].append(vm_item)
                flag = False
                break
        if flag:
            # create a new bin
            pm = [vcap, [vm_item]]
            pms.append(pm)
    return (vms, pms)


def best_fit_heuristic(containers, vm_types, pm_cap):
    vms = []
    pms = []

    vm_types = sorted(
        [vm_type for vm_type in vm_types if vm_type[1] <= pm_cap],
        key=lambda vm_type: vm_type[1],
    )
    print(f"[best_fit_heuristic]Valid VM types: {vm_types}")
    for cid, size in enumerate(containers, 1):
        print(f"[best_fit_heuristic]Processing containers: {cid}, size={size}")
        candidates = [vm for vm in vms if vm[1] - vm[2] >= size]
        best_vm = (
            min(candidates, key=lambda vm: vm[1] - vm[2] - size) if candidates else None
        )
        if best_vm is not None:
            best_vm[2] += size
            best_vm[3].append((cid, size))
        else:
            # create a new bin
            for name, cap in vm_types:
                if cap >= size:
                    vm_item = [name, cap, size, [(cid, size)]]
                    vms.append(vm_item)
                    break
    for vname, vcap, used, c_list in vms:
        print(f"[best_fit_heuristic]Processing vms: {vname}, vcap={vcap},used={used}")
        vm_item = [vname, vcap, c_list]

        pm_candidates = [pm for pm in pms if pm_cap - pm[0] >= vcap]
        best_pm = (
            min(pm_candidates, key=lambda pm: pm_cap - pm[0] - vcap)
            if pm_candidates
            else None
        )
        if best_pm is not None:
            best_pm[0] += vcap
            best_pm[1].append(vm_item)
        else:
            pm = [vcap, [vm_item]]
            pms.append(pm)
    return (vms, pms)


def format_vm_list(vms):
    """Format Level 1 VM allocation for display."""
    lines = []
    for i, (name, cap, _, items) in enumerate(vms, 1):
        items_str = ", ".join(f"c{cid}:{size}" for cid, size in items)
        lines.append(f"    v{i}: ({name}, Cap={cap})  [{items_str}]")
    lines.append(f"    ==> Total VMs used: {len(vms)}")
    return "\n".join(lines)


def format_pm_list(pms, pm_cap):
    """Format Level 2 PM allocation for display."""
    lines = []
    total_containers = 0
    for pi, (used, vm_list) in enumerate(pms, 1):
        parts = []
        for vname, vcap, clist in vm_list:
            total_containers += len(clist)
            inner = " + ".join(f"c{cid}:{size}" for cid, size in clist)
            parts.append(f"({vname}, Cap={vcap})[{inner}]")
        lines.append(f"    p{pi}: [used={used}/{pm_cap}]  " + ", ".join(parts))
    lines.append(f"    ==> Total PMs used: {len(pms)}")
    lines.append(f"    ==> Containers allocated: {total_containers}")
    return "\n".join(lines)


def main():
    """Main driver — reads data, runs heuristic, prints output."""
    containers = read_containers()
    vm_types = read_vm_types()
    pm_capacity = read_pm_capacity()

    print(f"  Containers: {containers}")
    print(f"  VM types:   {vm_types}")
    print(f"  PM capacity: {pm_capacity}")
    print()

    # -------------------------
    # FF&BF/FF
    # -------------------------
    print("=" * 65)
    print("  Question 1 — FF&BF/FF Heuristic")
    print("=" * 65)

    vms, pms = first_fit_heuristic(containers, vm_types, pm_capacity)

    print("\nLevel 1:")
    print(format_vm_list(vms))
    print("\nLevel 2:")
    print(format_pm_list(pms, pm_capacity))

    # -------------------------
    # BF&BF/BF
    # -------------------------
    print()
    print("=" * 65)
    print("  Question 1 — BF&BF/BF Heuristic")
    print("=" * 65)

    best_vms, best_pms = best_fit_heuristic(containers, vm_types, pm_capacity)

    print("\nLevel 1:")
    print(format_vm_list(best_vms))
    print("\nLevel 2:")
    print(format_pm_list(best_pms, pm_capacity))


if __name__ == "__main__":
    main()
