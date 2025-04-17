from datetime import datetime, timedelta
import random
import os
import json
import pandas as pd

data_dict = {}

class VirtualScenario(object):
    def __init__(self, available_time, testcases=[], solutions={}, name_suffix='vrt', schedule_date=datetime.today()):
        self.available_time = available_time
        self.gen_testcases = testcases
        self.solutions = solutions
        self.no_testcases = len(testcases)
        self.name = name_suffix
        self.scheduled_testcases = []
        self.schedule_date = schedule_date

    def testcases(self):
        return iter(self.gen_testcases)

    def submit(self):
        sorted_tc = sorted(self.gen_testcases, key=lambda x: (
            x['CalcPrio'],
            -(x['LastRun'].timestamp()),
            -x['Duration'],
        ))
        unexecuted = sorted_tc.copy()

        scheduled_time = 0
        detection_ranks = []
        undetected_failures = 0
        rank_counter = 1  #
        index_xsq = -1
        counter = 0
        while sorted_tc:
            cur_tc = sorted_tc.pop()
            counter += 1

            if scheduled_time + cur_tc['Duration'] <= self.available_time:
                unexecuted.pop()

                if self.solutions[cur_tc['Id']]:
                    detection_ranks.append(rank_counter)

                scheduled_time += cur_tc['Duration']
                self.scheduled_testcases.append(cur_tc)
                rank_counter += 1

            else:
                undetected_failures += self.solutions[cur_tc['Id']]
                if index_xsq == -1 and self.solutions[cur_tc['Id']]:
                    index_xsq = counter

        detected_failures = len(detection_ranks)
        total_failure_count = sum(self.solutions.values())

        assert undetected_failures + detected_failures == total_failure_count
        flag = False

        if total_failure_count > 0:
            ttf = detection_ranks[0] if detection_ranks else index_xsq

            if undetected_failures > 0:
                p = (detected_failures / total_failure_count)
            else:
                p = 1
            napfd = p - sum(detection_ranks) / (total_failure_count * self.no_testcases) + p / (2 * self.no_testcases)

            recall = detected_failures / total_failure_count
            avg_precision = 123
        else:
            ttf = 0
            napfd = 1
            recall = 1
            avg_precision = 1
            flag = True

        return [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks], unexecuted, flag


    def get_ta_metadata(self):
        execTimes, durations = zip(*[(tc['LastRun'], tc['Duration']) for tc in self.testcases()])

        metadata = {
            'availAgents': 1,
            'totalTime': self.available_time,
            'minExecTime': min(execTimes),
            'maxExecTime': max(execTimes),
            'scheduleDate': self.schedule_date,
            'minDuration': min(durations),
            'maxDuration': max(durations)
        }

        return metadata

    def set_testcase_prio(self, prio, tcid=-1):
        self.gen_testcases[tcid]['CalcPrio'] = prio

    def clean(self):
        for tc in self.testcases():
            self.set_testcase_prio(0, tc['Id'] - 1)

        self.scheduled_testcases = []


class IndustrialDatasetScenarioProvider():
    """
    Scenario provider to process CSV files for experimental evaluation of RETECS.

    Required columns are `self.tc_fieldnames` plus ['Verdict', 'Cycle']
    """

    def __init__(self, name, sched_time_ratio=0.5):

        super(IndustrialDatasetScenarioProvider, self).__init__()

        self.name = name

        self.tcdf = data_dict[name]

        self.tcdf['LastResults'] = self.tcdf['LastResults'].apply(json.loads)

        self.solutions = dict(zip(self.tcdf['Id'].tolist(), self.tcdf['Verdict'].tolist()))

        self.cycle = 0

        self.maxtime = min(self.tcdf.LastRun)

        self.max_cycles = max(self.tcdf.Cycle)

        self.scenario = None

        self.avail_time_ratio = sched_time_ratio

        self.tc_fieldnames = ['Id', 'Name', 'Duration', 'LastRun', 'LastResults']

    def get(self, name_suffix=None):
        self.cycle += 1

        if self.cycle > self.max_cycles:
            self.scenario = None
            return None

        cycledf = self.tcdf.loc[self.tcdf.Cycle == self.cycle]

        seltc = cycledf[self.tc_fieldnames].to_dict(orient='records')

        if name_suffix is None:
            name_suffix = (self.maxtime + timedelta(days=1)).isoformat()

        req_time = sum([tc['Duration'] for tc in seltc])
        total_time = req_time * self.avail_time_ratio

        selsol = dict(zip(cycledf['Id'].tolist(), cycledf['Verdict'].tolist()))

        self.scenario = VirtualScenario(testcases=seltc, solutions=selsol, name_suffix=name_suffix,
                                        available_time=total_time, schedule_date=self.maxtime + timedelta(days=1))

        self.maxtime = seltc[-1]['LastRun']

        return self.scenario

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


def get_scenario(name):
    compat = '../dataset/' + name + ".csv"
    if data_dict.get(name) is None:
        data_dict[name] = pd.read_csv(compat, sep=';', parse_dates=['LastRun'])
    return IndustrialDatasetScenarioProvider(name=name)
