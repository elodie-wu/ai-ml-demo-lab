# Job Scheduling Heuristics

A job scheduling demo comparing common dispatching rules under static and dynamic scheduling environments.

## Features

- First Come First Served (FCFS)
- Shortest Processing Time (SPT)
- Earliest Due Date (EDD)
- Static and dynamic scheduling
- Job arrival and processing times
- Flow time calculation
- Tardiness calculation
- Scheduling performance comparison

## Requirements

- Python 3.10+

No external packages are required.

## Run

From the repository root:

```bash
cd 10-job-scheduling
python scheduling.py
```

## Experiments

Three dispatching rules are compared:

```text
FCFS — First Come First Served
SPT  — Shortest Processing Time
EDD  — Earliest Due Date
```

The dynamic experiments consider job arrival times when selecting the next available job.

A static EDD experiment is also included, where all jobs are available for scheduling from the beginning.

## Results

Each scheduling strategy reports:

```text
Finish time
Flow time
Tardiness
Average flow time
Average tardiness
```

The experiments demonstrate how different dispatching rules affect scheduling efficiency and deadline performance.

## Project Structure

```text
10-job-scheduling/
├── scheduling.py
└── README.md
```

## Notes

Flow time is calculated as:

```text
Flow Time = Finish Time - Arrival Time
```

Tardiness is calculated as:

```text
Tardiness = max(Finish Time - Due Date, 0)
```

Lower average flow time and average tardiness indicate better scheduling performance for the corresponding objective.
