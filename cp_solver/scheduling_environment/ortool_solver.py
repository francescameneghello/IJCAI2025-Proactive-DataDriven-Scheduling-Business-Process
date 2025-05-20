"""Minimal jobshop example."""
import collections
import math

from ortools.sat.python import cp_model
from helper_functions import load_parameters, load_job_shop_env
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

TIME_LIMIT = 180


def print_scheduling(all_machines, assigned_jobs, objective_value):
    # Create per machine output lines.
    output = ""
    for machine in all_machines:
        # Sort by starting time.
        assigned_jobs[machine].sort()
        sol_line_tasks = "Machine " + str(machine) + ": "
        sol_line = "           "

        for assigned_task in assigned_jobs[machine]:
            name = f"job_{assigned_task.job}_task_{assigned_task.index}"
            # add spaces to output to align columns.
            sol_line_tasks += f"{name:15}"

            start = assigned_task.start
            duration = assigned_task.duration
            sol_tmp = f"[{start},{start + duration}]"
            # add spaces to output to align columns.
            sol_line += f"{sol_tmp:15}"

        sol_line += "\n"
        sol_line_tasks += "\n"
        output += sol_line_tasks
        output += sol_line

    # Finally print the solution found.
    print(f"Optimal Schedule Length: {objective_value}")
    #print(output)
    plot_schedule(assigned_jobs, all_machines)
    return output


def plot_schedule(assigned_jobs, all_machines):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set up colors for different jobs
    colors = plt.cm.get_cmap('tab20', len(assigned_jobs))

    # Plot each machine's schedule
    for machine_idx, machine in enumerate(all_machines):
        # Sort the tasks by their start time
        assigned_jobs[machine].sort(key=lambda x: x.start)

        for assigned_task in assigned_jobs[machine]:
            job_id = assigned_task.job
            task_id = assigned_task.index
            start = assigned_task.start
            duration = assigned_task.duration
            end = start + duration

            # Create a rectangle for each task (for the Gantt chart)
            rect = patches.Rectangle((start, machine_idx), duration, 0.8, facecolor=colors(job_id), edgecolor="black")

            # Add the rectangle to the plot
            ax.add_patch(rect)

            # Label the task with job and task index f'Job {job_id} Task {task_id}'
            ax.text(start + duration / 2, machine_idx + 0.4, '',
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    # Set limits and labels
    ax.set_xlim(0, max(assigned_task.start + assigned_task.duration for machine in all_machines for assigned_task in
                       assigned_jobs[machine]))
    ax.set_ylim(-0.5, len(all_machines) - 0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(len(all_machines)))
    ax.set_yticklabels([f'Machine {machine}' for machine in all_machines])

    plt.title('Job Schedule Gantt Chart')
    plt.grid(True)
    plt.show()


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solution_methods."""

    def __init__(self, vars, jobs_data):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__vars = vars
        self.__jobs_data = jobs_data
        self.__solutions = {}

    def on_solution_callback(self):
        """Called at each new solution."""
        print(
            "Solution %i, time = %f s, objective = %i"
            % (self.__solution_count, self.WallTime(), self.ObjectiveValue())
        )

        # Create one list of assigned tasks per machine.
        machines_count = 1 + max(task[0] for job in self.__jobs_data for task in job)
        all_machines = range(machines_count)
        assigned_jobs = collections.defaultdict(list)
        assigned_task_type = collections.namedtuple(
            "assigned_task_type", "start job index duration"
        )
        all_tasks = self.__vars

        for job_id, job in enumerate(self.__jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=self.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )
                )

        schedule_simulation = {}
        for machine in all_machines:
            assigned_jobs[machine].sort()
            schedule_simulation[str(machine)] = []
            for assigned_task in assigned_jobs[machine]:
                schedule_simulation[str(machine)].append(str(assigned_task.job))

        print('################################ Schedule solution ', self.__solution_count)
        self.__solutions[self.__solution_count] = {'makespan': self.ObjectiveValue(), 'solution': schedule_simulation, 'time': self.WallTime()}
        print(self.__solutions[self.__solution_count])
        self.__solution_count += 1

        #print_scheduling(all_machines, assigned_jobs, self.ObjectiveValue())

    def solution_count(self):
        return self.__solution_count

    def solutions(self):
        return self.__solutions


def create_calendar(machine, horizon):
    intervals = []
    #### number of hours per shift
    shift_duration = 480 #8ore di lavoro
    #start_shift = random.choice(range(0, 1380 - shift_duration, 1))
    start_shift = random.choice([i * 60 for i in range(0, 23)])
    end_shift = start_shift + shift_duration

    t = 0
    horizon = horizon*2
    days_week = 1
    while (start_shift + end_shift + t) < horizon:
        if days_week % 7 > 0 and days_week % 7 < 6:
            interval = (start_shift + t, end_shift + t)
            intervals.append(interval)
        t += 1440
        days_week += 1
    return intervals


def mySolver(jobShopEnv=None, calendar=None, use_calendar=False) -> None:
    """Minimal jobshop problem."""

    jobs_data = []
    for i in range(0, jobShopEnv.nr_of_jobs):
        operation = jobShopEnv.get_job(i).operations
        job = []
        for op in operation:
            duration = list(op.processing_times.items())[0][1]
            job.append([op.optional_machines_id[0], duration])
        jobs_data.append(job)

    all_machines = list(set(task[0] for job in jobs_data for task in job))

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)
    horizon = horizon*2
    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    ### ipotizziamo turni di 8 e 5 ore
    allowed_intervals = {}
    calendars = {}
    for machine in all_machines:
        if use_calendar:
            allowed_intervals[machine] = calendar[str(machine)]
        else:
            calendars[str(machine)] = create_calendar(machine, horizon)
            allowed_intervals[machine] = [(0, horizon)]

    if calendar:
        calendars = calendar

    #for i in range(0, jobShopEnv.nr_of_machines):
    #    machine = jobShopEnv.get_machine(i)
    #    allowed_intervals[i] = machine._calendar_intervals

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)  ### start time for activity
            end_var = model.NewIntVar(0, horizon, "end" + suffix) ### end time for activity
            interval_var = model.NewIntervalVar(    ### IntervalVar variable represents an integer interval variable (starting time, duration, ending time)
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )

            machine_to_intervals[machine].append(interval_var)

            # Add constraints that the task must start in one of the allowed intervals
            interval_constraints = []

            for (start_all, end_all) in allowed_intervals[machine]:
                interval_constraints.append(model.NewBoolVar(f'interval_{start_all}_{end_all}'))
                model.AddLinearConstraint(start_var, start_all, end_all - duration).OnlyEnforceIf(
                    interval_constraints[-1])

            ### add last interval
            start_all = allowed_intervals[0][-1][1]
            end_all = allowed_intervals[0][-1][1]*10
            interval_constraints.append(model.NewBoolVar(f'interval_{start_all}_{end_all}'))
            model.AddLinearConstraint(start_var, start_all, end_all - duration).OnlyEnforceIf(
                interval_constraints[-1])

            # Enforce that the task starts in one of the allowed intervals
            model.AddBoolOr(interval_constraints)

    machine_capacity = {}
    for i in range(0, jobShopEnv.nr_of_machines):
        machine = jobShopEnv.get_machine(i)
        machine_capacity[i] = int(machine._capacity)

    for machine, intervals in machine_to_intervals.items():
        if machine in machine_capacity:
            ### [1] number of resources requires for each job
            model.AddCumulative(intervals, [1] * len(intervals), machine_capacity[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    print("################################  Starting run CP Solver  ################################")
    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(all_tasks, jobs_data)
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    status = solver.Solve(model, solution_printer)
    if status == cp_model.OPTIMAL:
        print("Optimal solution found!")
    elif status == cp_model.FEASIBLE:
        print("Feasible solution found, but optimality not guaranteed.")
    elif status == cp_model.INFEASIBLE:
        print("No feasible solution exists.")
    elif status == cp_model.MODEL_INVALID:
        print("The model is invalid.")
    else:
        print("The search was stopped before finding a solution.")
    solution_count = solution_printer.solution_count()
    solutions = solution_printer.solutions()

    return solutions, solution_count, calendars


DEFINE_PROBLEM = False
CALENDAR = False
calendars = None

problem_instance = "../jsp/cscmax_20_20_2.txt" ### if DEFINE_PROBLE is True
name_instance = "cscmax_20_20_2_cp_solver_0.1"
#### prepare json file for simulation #####
parameters_sim = {"start_timestamp":  "", "jobs": {}, "machines": [],
                  "name_beanchmark": problem_instance, "calendars_res": {}}

if DEFINE_PROBLEM:
    jobShopEnv = load_job_shop_env(problem_instance, 1)
    parameters_sim['machines'] = [str(i) for i in range(jobShopEnv.nr_of_machines)]
    calendars = None
else:
    path = 'problem_instance/simulation_settings_' + name_instance + '.json'
    jobShopEnv = load_job_shop_env(problem_instance, from_absolute_path=path, q_value=0.726)
    with open(path, "r") as f:
        data = json.load(f)
        calendars = data["calendars"]
        parameters_sim['calendars'] = calendars


solutions, solution_count, calendars= mySolver(jobShopEnv, calendars, use_calendar=CALENDAR)
parameters_sim['machines'] = [str(i) for i in range(jobShopEnv.nr_of_machines)]
########## ATTENZIONE PER DATI SINTETICI ###############
if DEFINE_PROBLEM:
    parameters_sim['calendars'] = calendars

if DEFINE_PROBLEM:
    parameters_sim['solutions_q1'] = solutions
    parameters_sim['n_solutions_q1'] = len(solutions)
else:
    if CALENDAR:
        parameters_sim['solutions'] = solutions
        parameters_sim['n_solutions'] = len(solutions)
    else:
        parameters_sim['solutions'] = solutions
        parameters_sim['n_solutions'] = len(solutions)

for idx, job in enumerate(jobShopEnv.jobs):
    machine_seq = []
    times = []
    activity_seq = []
    for operation in job.operations:
        for k, v in operation.processing_times.items():
            machine_seq.append(str(k))
            if DEFINE_PROBLEM:
                times.append(operation.simulation_parameters[str(k)])
            else:
                time, activity = operation.simulation_parameters
                times.append(time[k])
                activity_seq.append(activity)
    parameters_sim['jobs'][idx] = {"machine_seq": machine_seq, "times": times, "activity_seq": activity_seq}

if DEFINE_PROBLEM:
    with open('simulation_settings_' + name_instance + '.json', 'w') as outfile:
        json.dump(parameters_sim, outfile, indent=2)
else:
    if CALENDAR:
        name_instance = name_instance + '_cal'
    with open('simulation_settings_solutions_' + name_instance + '.json', 'w') as outfile:
        json.dump(parameters_sim, outfile, indent=2)
