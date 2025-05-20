import math
import re
from pathlib import Path
from job import Job
from machine import Machine
from operation import Operation
import numpy as np
import json

def parse(JobShop, instance, u_level, from_absolute_path=False):
    if not from_absolute_path:
        base_path = Path(__file__).parent.parent.absolute()
        data_path = base_path.joinpath('data' + instance)
    else:
        data_path = instance

    with open(data_path, "r") as data:
        total_jobs, total_machines = re.findall('\S+', data.readline())
        number_total_jobs, number_total_machines = int(
            total_jobs), int(total_machines)

        JobShop.set_nr_of_jobs(number_total_jobs)
        JobShop.set_nr_of_machines(number_total_machines)
        precedence_relations = {}
        job_id = 0
        operation_id = 0

        for key, line in enumerate(data):
            if key >= number_total_jobs:
                break
            # Split data with multiple spaces as separator
            parsed_line = re.findall('\S+', line)

            # Current item of the parsed line
            i = 0

            job = Job(job_id)
            while i < len(parsed_line):
                # Current operation
                operation = Operation(job, job_id, operation_id)
                operation_options = 1
                for operation_option_id in range(operation_options):
                    mean = int(parsed_line[i + 1])
                    std = round(np.random.uniform(0, u_level*mean, 1)[0], 3)
                    n = int(total_jobs)
                    q = 1.645/(math.sqrt(2*n))
                    duration = int(mean + q*std)
                    #operation.add_operation_option(int(parsed_line[i]), int(parsed_line[i + 1]))
                    operation.add_operation_option(int(parsed_line[i]), duration)
                    operation.set_simulation_parameters(parsed_line[i], mean, std)
                job.add_operation(operation)
                JobShop.add_operation(operation)
                if i != 0:
                    precedence_relations[operation_id] = [
                        JobShop.get_operation(operation_id - 1)]
                i += 2
                operation_id += 1

            JobShop.add_job(job)
            job_id += 1

    # Precedence Relations
    for operation in JobShop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    sequence_dependent_setup_times = [[[0 for r in range(len(JobShop.operations))] for t in range(len(JobShop.operations))] for
                                      m in range(number_total_machines)]

    # Precedence Relations
    JobShop.add_precedence_relations_operations(precedence_relations)
    JobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    # Machines
    for id_machine in range(0, number_total_machines):
        JobShop.add_machine((Machine(id_machine)))

    return JobShop

def parse_json(JobShop, from_absolute_path, q_value):
    print("################################  PARSING JSON FILE  ################################")
    data_path = from_absolute_path
    with open(data_path, "r") as f:
        data = json.load(f)
        number_total_jobs = len(data["jobs"])
        number_total_machines = len(data["machines"])
        JobShop.set_nr_of_jobs(number_total_jobs)
        JobShop.set_nr_of_machines(number_total_machines)
        precedence_relations = {}

        for job_id in data["jobs"]:
            job = Job(int(job_id))
            for idx, m in enumerate(data["jobs"][job_id]["machine_seq"]):
                operation = Operation(job, int(job_id), idx)
                operation_options = 1

                mean = round(data["jobs"][job_id]["times"][idx][0])
                std = round(data["jobs"][job_id]["times"][idx][1], 3)
                #n = data["n_tasks"] ## max number of activities in all the jobs
                #q = 1.645 / (math.sqrt(n))
                duration = int(mean + q_value * std)
                #duration = int(data["jobs"][job_id]["times_fixed"][idx])
                machine = int(data["jobs"][job_id]["machine_seq"][idx])
                #machine = data["jobs"][job_id]["machine_seq"][idx]
                operation.add_operation_option(machine, duration)
                #operation.set_simulation_parameters(machine, mean, std, data["jobs"][job_id]["activity_seq"][idx])
                operation.set_simulation_parameters(machine, mean, std)
                job.add_operation(operation)
                JobShop.add_operation(operation)
                if idx != 0:
                    precedence_relations[idx] = [
                        JobShop.get_operation(idx - 1)]

            JobShop.add_job(job)

    # Precedence Relations
    for operation in JobShop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    sequence_dependent_setup_times = [[[0 for r in range(len(JobShop.operations))] for t in range(len(JobShop.operations))] for
                                      m in range(number_total_machines)]

    # Precedence Relations
    JobShop.add_precedence_relations_operations(precedence_relations)
    JobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    # Machines
    for id_machine in range(0, number_total_machines):
        JobShop.add_machine((Machine(id_machine, capacity=1,
                                     intervals=None)))

        #JobShop.add_machine((Machine(id_machine, capacity=data["machines_capacity"][str(id_machine)],
        #                             intervals=data["calendars_intervals"][str(id_machine)])))

    return JobShop