class Job:
    def __init__(self, name, arrival_time, processing_time, due_date):
        self.name = name
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.due_date = due_date
        self.flow_time = 0
        self.finish_time = 0
        self.tardiness = 0


def dispatch(jobs, rule, isStatic=False):
    print(f"Begin: make a schedule by {rule}, is static:{isStatic}\n")
    current_time = 0
    unfinished_jobs = jobs.copy()
    total_flow_time = 0
    total_tardiness = 0

    # [] = false
    while unfinished_jobs:
        available_jobs = []

        if isStatic == False:
            # When arrival_time is reached, add them to available_jobs
            for job in unfinished_jobs:
                if job.arrival_time <= current_time:
                    available_jobs.append(job)
        else:
            available_jobs = unfinished_jobs.copy()

        if available_jobs:
            next_job = None
            if rule == "FCFS":
                next_job = min(available_jobs, key=lambda job: job.arrival_time)

            elif rule == "SPT":
                next_job = min(available_jobs, key=lambda job: job.processing_time)

            elif rule == "EDD":
                next_job = min(available_jobs, key=lambda job: job.due_date)

            else:
                print("Error: Unknown dispatching rule")
                return

            current_time += next_job.processing_time
            unfinished_jobs.remove(next_job)

            next_job.finish_time = current_time
            next_job.flow_time = next_job.finish_time - next_job.arrival_time
            next_job.tardiness = max(next_job.finish_time - next_job.due_date, 0)
            total_flow_time += next_job.flow_time
            total_tardiness += next_job.tardiness
            print(
                f"Process: job: {next_job.name}, flow_time:{next_job.flow_time}, finish_time:{next_job.finish_time}, tardiness:{next_job.tardiness}"
            )
        else:
            current_time += 1
            print(f"Process: current_time + 1 ={current_time}")

    average_flow_time = total_flow_time / len(jobs)
    average_tardiness = total_tardiness / len(jobs)
    print("\n")
    print(
        f"Result: rule:{rule} average_flow_time:{average_flow_time},average_tardiness:{average_tardiness}\n"
    )
    return average_flow_time, average_tardiness


def main():
    jobs = [
        Job("A", 0, 11, 61),
        Job("B", 1, 29, 45),
        Job("C", 2, 31, 31),
        Job("D", 3, 1, 33),
        Job("E", 4, 2, 32),
    ]

    print("*" * 30)
    dispatch(jobs, "FCFS")
    print("*" * 30)
    dispatch(jobs, "SPT")
    print("*" * 30)
    dispatch(jobs, "EDD")
    print("*" * 30)
    dispatch(jobs, "EDD", isStatic=True)


if __name__ == "__main__":
    main()
