# Resource Allocation Heuristics

A two-level resource allocation demo using First Fit and Best Fit heuristics to allocate containers to virtual machines (VMs) and VMs to physical machines (PMs).

## Features

- Two-level resource allocation
- Container-to-VM allocation
- VM-to-PM allocation
- First Fit heuristic
- Best Fit heuristic
- Capacity constraint handling
- Allocation result comparison

## Requirements

- Python 3.10+

No external packages are required.

## Run

From the repository root:

```bash
cd 09-resource-allocation
python resource_allocation.py
```

The program reads:

```text
containers.txt
vm_types.txt
pm_capacity.txt
```

## Experiments

The allocation process contains two levels:

```text
Level 1: Containers → Virtual Machines
Level 2: Virtual Machines → Physical Machines
```

Two allocation strategies are compared:

```text
First Fit
Best Fit
```

VM types that exceed the physical machine capacity are excluded before allocation.

## Results

| Method    | VMs Used | PMs Used |
| --------- | -------: | -------: |
| First Fit |        4 |        3 |
| Best Fit  |        4 |        2 |

Both methods allocate all eight containers using four virtual machines.

Best Fit produces a more compact VM-to-PM allocation and reduces the number of physical machines from three to two.

## Project Structure

```text
09-resource-allocation/
├── containers.txt
├── vm_types.txt
├── pm_capacity.txt
├── sampleoutput.txt
├── resource_allocation.py
└── README.md
```

## Notes

The physical machine capacity is `60`.

VM type `V4` has capacity `70`, so it cannot be allocated to a physical machine and is excluded from the available VM types.
