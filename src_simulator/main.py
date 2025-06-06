
import pandas as pd
from run_simulation import main
import math
import numpy as np

import json
import time


def define_job_problem(path, type, calendar=False):
    TASKS = {}
    with open(path) as file:
        data = json.load(file)
        jobs = data['jobs']
        MACHINES = data['machines']
        SCHEDULES = None if "solutions_q1" in data else data['solutions']
        for job, details in jobs.items():
            if type == 'fsjp':
                last = data["n_solutions"] - 1
                machine_seq_solution = data['solutions'][last]['machine_seq']
            machine_seq = details["machine_seq"]
            times = details["times"]

            for i, machine in enumerate(machine_seq):
                mean, std = times[i]
                if i == 0:
                    prec = None
                else:
                    prec = (job, machine_seq[i - 1])

                TASKS[(job, machine)] = {
                    "mean": mean,
                    "std": math.sqrt(float(std)),
                    "prec": prec
                }
    return TASKS, MACHINES, SCHEDULES


def simulate_schedule(s, N, MACHINES, path, D_start=None, NAME_EXP= None):
    makespan_simulation = main(s, N, path, D_start, NAME_EXP)
    return makespan_simulation

def define_Q3(path, start_time):
    TASK, MACHINES, SCHEDULES = define_job_problem(path, type)
    length_path, stds_list, n_activities = simulate_schedule(None, 1000, MACHINES, path)
    stds_2 = [s*s for s in stds_list]
    Q3 = (1.645/math.sqrt(n_activities))*(math.sqrt(np.mean(stds_2))/np.mean(stds_list))

    write_path_actual = '/results/results_simulation_Q3_' + NAME_EXP + '.json'
    with open(write_path_actual, 'w') as outfile:
        results = {"name_instance": NAME_EXP, 'Q3': Q3, 'time': (time.time() - start_time)}
        json.dump(results, outfile, indent=2)
    return Q3


########## CALENDAR ################
#final_path = path + "/new_experiments/simulation_settings_abz6_cp_solver_0.1_q3_CAL.json"


def define_top_ten(final_path, results_iteration):
    N = 100
    TASK, MACHINES, SCHEDULES = define_job_problem(final_path, type)
    makespans = []
    s = 0
    number_best = 0
    s_star = SCHEDULES[str(s)]["solution"]
    D_star = math.inf
    while s < len(SCHEDULES):
        schedule = SCHEDULES[str(s)]["solution"]
        makespans = simulate_schedule(schedule, N, MACHINES, final_path)  ## D'
        D_alpha = np.percentile(makespans, 95)
        print('Solution n ', s, 'D_alpha', D_alpha)
        print("Mean of makespans", np.mean(makespans), "Std of makespans", np.std(makespans))
        print("********************************************************************************")
        results_iteration["first_step"][s] = {"D_alpha": D_alpha, "mean": np.mean(makespans), "std": np.std(makespans),
                                "DET_makespan": SCHEDULES[str(s)]["makespan"], "n_simulation": 100}
        if D_alpha < D_star:
            number_best = s
            s_star = SCHEDULES[str(s)]["solution"]
            D_star = D_alpha

        s += 1

    ### find best 10 solution
    sort_results = {k: v for k, v in sorted(results_iteration['first_step'].items(), key=lambda item: item[1]["D_alpha"], reverse=True)}
    top_ten = [k for k in sort_results]
    if len(top_ten) > 10:
        top_ten = top_ten[-10:]
    results_iteration["top_ten"] = top_ten

    print('Best solution', D_star, number_best)
    results_iteration["Best_solution_1_step"] = {"solution": number_best, "D_alpha": D_star, "schedule": s_star}

    return results_iteration, top_ten


def define_k_solutions(results_path):
    with open(results_path, "r") as f:
        data = json.load(f)
        makespan_det = []
        for s in data["solutions"]:
            makespan_det.append(data["solutions"][s]["makespan"])

    improvements = []
    for i in range(0, len(makespan_det)-1):
        imp = ((makespan_det[i] - makespan_det[i+1]) /makespan_det[i])*100
        improvements.append(round(imp, 3))

    search_list = improvements
    start_point_research = -1
    while start_point_research < 0:
        try:
            jump = next(i for i in reversed(search_list) if i >= 0.20)
            index_jump = improvements.index(jump)
        except StopIteration:
            index_jump = 0
        count = 0
        for i in range(index_jump, len(improvements)):
            if improvements[i] < 0.6:
                count += 1
        if count >= 5:
            start_point_research = index_jump
        search_list = improvements[:index_jump]
        if not search_list:
            start_point_research = 0

    return range(start_point_research, len(data["solutions"]), 1)


def final_simulation(final_path, results_iteration, top_k):
    print("TOP K solutions", top_k)
    N = 1000
    TASK, MACHINES, SCHEDULES = define_job_problem(final_path, type)
    number_best = 0
    s_star = None
    D_star = math.inf
    for s in top_k:
        schedule = SCHEDULES[str(s)]["solution"]
        makespans = simulate_schedule(schedule, N, MACHINES, final_path)  ## D'
        D_alpha = np.percentile(makespans, 95)
        print('Solution n ', s, 'D_alpha', D_alpha)
        print("Mean of makespans", np.mean(makespans), "Std of makespans", np.std(makespans))
        print("********************************************************************************")
        results_iteration["k_step"][s] = {"D_alpha": D_alpha, "mean": np.mean(makespans), "std": np.std(makespans),
                                "DET_makespan": SCHEDULES[str(s)]["makespan"], "n_simulation": 1000, "all_makespan": makespans}
        if D_alpha < D_star:
            number_best = s
            s_star = SCHEDULES[str(number_best)]["solution"]
            D_star = D_alpha

        s += 1

    print('Best solution', D_star, number_best)
    results_iteration["Best_solution"] = {"solution": number_best, "D_alpha": D_star, "schedule": s_star}

    return results_iteration


def simulate_actual_scheduling(final_path, NAME_EXP = None):
    N = 1000
    with open(final_path) as file:
        data = json.load(file)
        schedule = data["actual_scheduling"]
        TASK, MACHINES, SCHEDULES = define_job_problem(final_path, type)
        makespans = simulate_schedule(schedule, N, MACHINES, final_path, NAME_EXP=NAME_EXP)  ## D'
        D_alpha = np.percentile(makespans, 95)
        print('Actual scheduling D_alpha: ', D_alpha)
        print("Mean of makespans", np.mean(makespans), "Std of makespans", np.std(makespans))
        print("********************************************************************************")

    write_path_actual = '/Users/francescameneghello/Documents/GitHub/RIMS_tool/core_jsp/example/results/results_simulation_actual_scheduling' + NAME_EXP + '_prediction.json'
    with open(write_path_actual, 'w') as outfile:
        results = {"name_instance": NAME_EXP, "D_alpha": D_alpha, "Mean": np.mean(makespans), "Std": np.std(makespans),
                   "makespans": makespans}
        json.dump(results, outfile, indent=2)
    return D_alpha, np.mean(makespans), np.std(makespans)


##### call for beanchmarks #####
CRITICAL_PATH_DISCOVERY = False
start_time = time.time()
NAME_EXP = 'cscmax_20_20_2_cp_solver_0.1_q3'
final_path = '../syn_data/cscmax_20_20_2/simulation_settings_cscmax_20_20_2_cp_solver_0.1_q3.json'
if not CRITICAL_PATH_DISCOVERY:
    TASK, MACHINES, SCHEDULES = define_job_problem(final_path, type)

    top_k_solutions = define_k_solutions(final_path)

    results_iteration = {"top_k": list(top_k_solutions), "k_step": {}, "Best_solution": {}}

    results_iteration = final_simulation(final_path, results_iteration, top_k_solutions)

    results_iteration['total_time'] = (time.time() - start_time)

    write_path = 'results/results_simulation_' + NAME_EXP + '.json'
    with open(write_path, 'w') as outfile:
        json.dump(results_iteration, outfile, indent=2)
else:
    Q3 = define_Q3(final_path, start_time)
    print('---------------- Q3: ', Q3, ' ----------------------')

print("--- Execution time %s seconds ---" % (time.time() - start_time))
