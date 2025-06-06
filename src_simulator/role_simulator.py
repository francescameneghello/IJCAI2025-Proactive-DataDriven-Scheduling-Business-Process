'''
Class to define the role object within the simulation.
The role is defined by its name (*name*), the amount of resources available (*capacity*),
and the schedule assigned to it (*calendar*).

Each role needs to be defined within the json file in the following format.
The example describes *Role 2*, Ellen and Sue, who work Monday
through Saturday, from 8 a.m. to 7 p.m.
```shell
    "resource": {
        ........
        "name_of_role": {
            "resources": #List of resources,
            "calendar": {
                "days": #List of working days of the role,
                        #given as integer 0 means Sunday and 6 means Saturday.
                "hour_min": #the hour of the start of the working day,
                "hour_max": #the hour of the end of the working day
            }
        }
        ........
    }
```

```json
    "Role 2": {
        "resources": ["Ellen", "Sue"],
        "calendar": {
            "days": [1, 2, 3, 4, 5, 6],
            "hour_min": 8,
            "hour_max": 19
        }
    }
```
Finally, in the simulation parameters file, we have to indicate the role assigned to the execution of each activity.
(Here to view complete examples)

```json
    "resource_table": [
        {
            "role": "Role 1",
            "task": "A_SUBMITTED"
        },
        {
            "role": "Role 2",
            "task": "A_PARTLYSUBMITTED"
        }
    ]
```

'''
from datetime import timedelta
import simpy
import random
from random import randint


class RoleSimulator(object):

    def __init__(self, env: simpy.Environment, name: str, capacity, calendar: dict, schedule=None, n_jobs=None):
        self._env = env
        self._name = name
        self._resources_name = capacity
        self._capacity = capacity
        self._calendar = calendar
        #self._resource_simpy = simpy.Resource(env, self._capacity) ### old version
        self._queue = []
        self._queue_jobs = []
        self._schedule = schedule
        self._n_jobs = n_jobs
        if self._schedule:
            self._define_dict_resource_schedule()
            self._schedule_active = True
        else:
            self._resource_simpy = simpy.PriorityResource(self._env, self._capacity)
            self._schedule_active = False

    def request_no_schedule(self):
        prio = randint(0, 50)
        return self._resource_simpy.request(priority=prio)

    def _release_no_schedule(self, request):
        self._resource_simpy.release(request)

    def _define_dict_resource_schedule(self):
        self._dict_res = dict()
        jobs = list(self._schedule)
        capacity = self._capacity
        i = 0
        for j in jobs:
            if self._schedule[i] == j and capacity > 0:
                res = simpy.Container(self._env, init=self._schedule.count(j), capacity=100)
                i += 1
                capacity -= 1
            else:
                res = simpy.Container(self._env, init=0, capacity=100)
            if j not in self._dict_res:
                self._dict_res[j] = res
        del self._schedule[0: self._capacity]

        '''for j in jobs:
            if self._schedule[0] == j:
                res = simpy.Container(self._env, init=1, capacity=100)
            else:
                res = simpy.Container(self._env, init=0, capacity=100)
            self._dict_res[j] = res
        del self._schedule[0]'''

    def _release(self):
        if self._schedule:
            next_job = self._schedule[0]
            del self._schedule[0]
            yield self._dict_res[next_job].put(1)

    def request(self, job_id):
        return self._dict_res[job_id]

    '''def release(self, request):
        """
        Method to release the role resource that was used to perform the activity.
        """
        self._resource_simpy.release(request)

    def request(self, job_id):
        """
        Method to require a resource of the role needed to perform the activity.
        """
        self._queue.append(self._resource_simpy.queue)
        if self._schedule:
            del self._schedule[0]
        return self._resource_simpy.request()'''


    def _get_name(self):
        return self._name

    def _get_capacity(self):
        return self._capacity

    def _get_resource(self):
        return self._resource_simpy

    def _get_calendar(self):
        return self._calendar

    def _check_day_work(self, timestamp):
        return True if (timestamp.weekday() in self._calendar['days']) else False

    def _check_hour_work(self, timestamp):
        return True if (self._calendar['hour_min'] <= timestamp.hour < self._calendar['hour_max']) else False

    def _define_stop_weekend(self, timestamp):
        if min(self._calendar['days']) > timestamp.weekday():
            monday = min(self._calendar['days']) - timestamp.weekday()
        else:
            monday = 7 - timestamp.weekday()
        new_start = timestamp.replace(hour=self._calendar['hour_min'], minute=0, second=0) + timedelta(days=monday)
        return (new_start-timestamp).total_seconds()

    def _define_stop_week(self, timestamp):
        if timestamp.hour < self._calendar['hour_min']:
            stop = timestamp.replace(hour=self._calendar['hour_min'], minute=0, second=0) - timestamp
        else:
            new_day = timestamp.replace(hour=self._calendar['hour_min'], minute=0, second=0) + timedelta(days=1)
            stop = (new_day - timestamp).total_seconds()
            if new_day.weekday() not in self._calendar['days']:
                stop = stop + self._define_stop_weekend(new_day)
        return stop

    def to_time_schedule(self, timestamp):
        """
            Method to check the schedule of the requested resource and
            eventually it returns the time to wait before executing the activity.
        """
        if not self._check_day_work(timestamp):
            stop = self._define_stop_weekend(timestamp)
        elif not self._check_hour_work(timestamp):
            stop = self._define_stop_week(timestamp)
        else:
            stop = 0
        return int(stop)

    def _split_week(self, timestamp, duration):
        before = (timestamp.replace(hour=self._calendar['hour_max'], minute=0, second=0) - timestamp).seconds
        stop = self._define_stop_week(timestamp.replace(hour=self._calendar['hour_max'], minute=0, second=0))
        if not self._check_day_work(timestamp + timedelta(seconds=before) + timedelta(seconds=stop)):
            stop += self._define_stop_weekend(timestamp + timedelta(seconds=before + stop))
        after = duration - before
        return before, stop, after

    def _split_weekend(self, timestamp, duration):
        before = (timestamp.replace(hour=self._calendar['hour_max'], minute=0, second=0) - timestamp).total_seconds
        stop = self._define_stop_weekend(timestamp.replace(hour=self._calendar['hour_max'], minute=0, second=0))
        after = duration - before
        return before, stop, after

    def _check_duration(self, timestamp, duration):
        time_to_complete = timestamp + timedelta(seconds=duration)
        before = duration
        stop = after = 0
        if not self._check_hour_work(time_to_complete):
            if self._check_day_work(time_to_complete) in self._calendar['days']:
                before, stop, after = self._split_week(timestamp, duration)
            else:
                before, stop, after = self._split_weekend(timestamp, duration)
        return before, stop, after

    def _define_timework(self, timestamp, duration):
        stop_pre = self.to_time_schedule(timestamp)
        before, stop, after = self._check_duration(timestamp + timedelta(seconds=stop_pre), duration)
        return stop_pre, before + stop + after

    def _get_resources_name(self):
        choiced = self._resources_name[0]
        self._resources_name.remove(choiced)
        return choiced

    def _release_resource_name(self, resource):
        self._resources_name.append(resource)