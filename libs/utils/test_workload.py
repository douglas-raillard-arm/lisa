import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import isnan

# TODO: killme.jpeg
from executor import Executor

from perf_analysis import PerfAnalysis
from serialize import YAMLSerializable
from wlgen.rta import RTA

# from wa import Metric

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! FIXME !!! this is from workload-automation you bloody thief!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class Metric(object):
    """
    This is a single metric collected from executing a workload.

    :param name: the name of the metric. Uniquely identifies the metric
                 within the results.
    :param value: The numerical value of the metric for this execution of a
                  workload. This can be either an int or a float.
    :param units: Units for the collected value. Can be None if the value
                  has no units (e.g. it's a count or a standardised score).
    :param lower_is_better: Boolean flag indicating where lower values are
                            better than higher ones. Defaults to False.
    :param classifiers: A set of key-value pairs to further classify this
                        metric beyond current iteration (e.g. this can be used
                        to identify sub-tests).

    """

    __slots__ = ['name', 'value', 'units', 'lower_is_better', 'classifiers']

    @staticmethod
    def from_pod(pod):
        return Metric(**pod)

    def __init__(self, name, value, units=None, lower_is_better=False,
                 classifiers=None):
        self.name = name
        self.value = value
        self.units = units
        self.lower_is_better = lower_is_better
        self.classifiers = classifiers or {}

    def to_pod(self):
        return dict(
            name=self.name,
            value=self.value,
            units=self.units,
            lower_is_better=self.lower_is_better,
            classifiers=self.classifiers,
        )

    def __str__(self):
        result = '{}: {}'.format(self.name, self.value)
        if self.units:
            result += ' ' + self.units
        result += ' ({})'.format('-' if self.lower_is_better else '+')
        return result

    def __repr__(self):
        text = self.__str__()
        if self.classifiers:
            return '<{} {}>'.format(text, self.classifiers)
        else:
            return '<{}>'.format(text)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! FIXME !!! this is from workload-automation you bloody thief!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class ResultBundle(object):
    """
    Bundle for storing test results

    :param passed: Indicates whether the associated test passed.
        It will also be used as the truth-value of a ResultBundle.
    :type passed: boolean
    """
    def __init__(self, passed):
        self.passed = passed
        self.metrics = []

    def __nonzero__(self):
        return self.passed

    def add_metric(self, metric):
        """
        Lets you append several test :class:`Metric` to the bundle.

        :param metric: Metric to add to the bundle
        :type metric: Metric
        """
        self.metrics.append(metric)

import shutil

class TestBundle(YAMLSerializable):
    """
    A LISA test bundle.

    :param res_dir: Directory from where the target execution artifacts reside.
        This will also be used to dump any artifact generated in the test code.
    :type res_dir: str

    :Design notes:
        * :meth:`from_target` will collect whatever artifacts are required
          from a given target, and will then create a :class:`TestBundle`.
        * :meth:`from_path` will use whatever artifacts are available in a
          given directory, and will then create a :class:`TestBundle`.

       .. warning:: Those artifacts must be created by :meth:`from_target` to allow
          "offline" replaying, that is, we must be able to re-run tests without
          having access to a target.


    Thus, creating a Bundle from a live target would be done as follows:

    >>> bundle = TestBundle.from_target(test_env, "my/res/dir")

    And you could run some test on the collected data like so:

    >>> res_bundle = bundle.test_foo()

    Offline replaying would be done like so:

    >>> bundle = TestBundle.from_path("/my/res/dir")
    >>> res_bundle = bundle.test_foo()
    """

    def __init__(self, res_dir):
        self.res_dir = res_dir

    @classmethod
    def _from_target(cls, te, res_dir):
        """
        Internals of the target factory method.
        """
        raise NotImplementedError()

    @classmethod
    def from_target(cls, te, res_dir=None, **kwargs):
        """
        Factory method to create a bundle using a live target

        This is mostly boiler-plate code around :meth:`_from_target`,
        have a look at that method for additionnal optionnal parameters.
        """
        if not res_dir:
            res_dir = te.get_res_dir()

        # Logger stuff?

        bundle = cls._from_target(te, res_dir, **kwargs)

        # We've created the bundle from the target, and have all of
        # the information we need to execute the test code. However,
        # we enforce the use of the offline reloading path to ensure
        # it does not get broken.
        bundle.to_path(res_dir)
        return cls.from_path(res_dir)

    @classmethod
    def _filepath(cls, res_dir):
        return os.path.join(res_dir, "{}.yaml".format(cls.__name__))

    @classmethod
    def from_path(cls, res_dir):
        """
        See :meth:`YAMLSerializable.from_path`
        """
        bundle = YAMLSerializable.from_path(cls._filepath(res_dir))
        # We need to update the res_dir to the one we were given
        bundle.res_dir = res_dir

        return bundle

    def to_path(self, res_dir):
        """
        See :meth:`YAMLSerializable.to_path`
        """
        super(TestBundle, self).to_path(self._filepath(res_dir))

from bart.common.Utils import area_under_curve
from trace import Trace
from energy_model import EnergyModel
class GenericTestBundle(TestBundle):
    """
    "Abstract" class for generic synthetic tests.

    :param res_dir: See :class:`TestBundle`
    :type res_dir: str

    :param nrg_model: The energy model of the platform the synthetic workload
      was run on
    :type nrg_model: EnergyModel

    :param rtapp_params: The rtapp parameters used to create the synthetic
      workload. That happens to be what is returned by :meth:`create_rtapp_params`
    :type rtapp_params: dict

    This class provides :meth:`test_slack` and :meth:`test_task_placement` to
    validate the basic behaviour of EAS.
    """

    ftrace_conf = {
        "events" : ["sched_switch"],
    }
    """
    The FTrace configuration used to record a trace while the synthetic workload
    is being run.
    """

    @property
    def trace(self):
        """

        :returns: a Trace

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`to_path`
        """
        if not self._trace:
            self._trace = Trace(self.res_dir, events=self.ftrace_conf["events"])

        return self._trace

    def __init__(self, res_dir, nrg_model, rtapp_params):
        super(GenericTestBundle, self).__init__(res_dir)

        # self.trace = Trace(res_dir, events=self.ftrace_conf["events"])
        #EnergyModel.from_path(os.path.join(res_dir, "nrg_model.yaml"))
        self._trace = None
        self.nrg_model = nrg_model
        self.rtapp_params = rtapp_params

    @classmethod
    def create_rtapp_params(cls, te):
        """
        :returns: a :class:`dict` with task names as keys and :class:`RTATask` as values

        This is the method you want to override
        """
        raise NotImplementedError()

    @classmethod
    def _from_target(cls, te, res_dir):
        rtapp_params = cls.create_rtapp_params(te)

        wload = RTA(te.target, "rta_{}".format(cls.__name__.lower()), te.calibration())
        wload.conf(kind='profile', params=rtapp_params,
                   run_dir=Executor.get_run_dir(te.target))

        trace_path = os.path.join(res_dir, "trace.dat")
        te.ftrace_conf(cls.ftrace_conf)

        with te.record_ftrace(trace_path):
            with te.freeze_userspace():
                wload.run(out_dir=res_dir)

        return cls(res_dir, te.nrg_model, rtapp_params)

    @classmethod
    def min_cpu_capacity(cls, te):
        return min(te.target.sched.get_capacities().values())

    @classmethod
    def max_cpu_capacity(cls, te):
        return max(te.target.sched.get_capacities().values())

    def test_slack(self, negative_slack_allowed_pct=15):
        """
        Assert that the RTApp workload was given enough performance

        :param out_dir: Output directory for test artefacts
        :type out_dir: str

        :param negative_slack_allowed_pct: Allowed percentage of RT-app task
            activations with negative slack.
        :type negative_slack_allowed_pct: int

        Use :class:`PerfAnalysis` to find instances where the RT-App workload
        wasn't able to complete its activations (i.e. its reported "slack"
        was negative). Assert that this happened less than
        :attr:`negative_slack_allowed_pct` percent of the time.
        """
        pa = PerfAnalysis(self.res_dir)

        slacks = {}
        passed = True

        # Data is only collected for rt-app tasks, so it's safe to iterate over
        # all of them
        for task in pa.tasks():
            slack = pa.df(task)["Slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100. / len(slack)
            if bad_activations_pct > negative_slack_allowed_pct:
                passed = False

            slacks[task] = bad_activations_pct

        res = ResultBundle(passed)
        for task, slack in slacks.iteritems():
            res.add_metric(Metric("slack_{}".format(task), slack,
                                  units='%', lower_is_better=True))

        return res

    def _get_start_time(self):
        """
        Get the time where the first task spawned
        """
        tasks = self.rtapp_params.keys()
        sdf = self.trace.data_frame.trace_event('sched_switch')
        start_time = self.trace.start_time + self.trace.time_range

        for task in tasks:
            pid = self.trace.getTaskByName(task)
            assert len(pid) == 1, "getTaskByName returned more than one PID"
            pid = pid[0]
            start_time = min(start_time, sdf[sdf.next_pid == pid].index[0])

        return start_time

    def _get_expected_task_utils_df(self):
        """
        Get a DataFrame with the *expected* utilization of each task over time

        :returns: A Pandas DataFrame with a column for each task, showing how
                  the utilization of that task varies over time
        """
        util_scale = self.nrg_model.capacity_scale

        transitions = {}
        def add_transition(time, task, util):
            if time not in transitions:
                transitions[time] = {task: util}
            else:
                transitions[time][task] = util

        # First we'll build a dict D {time: {task_name: util}} where D[t][n] is
        # the expected utilization of task n from time t.
        for task, params in self.rtapp_params.iteritems():
            # time = self.get_start_time(experiment) + params.get('delay', 0)
            time = params.delay_s
            add_transition(time, task, 0)
            for _ in range(params.loops):
                for phase in params.phases:
                    util = (phase.duty_cycle_pct * util_scale / 100.)
                    add_transition(time, task, util)
                    time += phase.duration_s
            add_transition(time, task, 0)

        index = sorted(transitions.keys())
        df = pd.DataFrame([transitions[k] for k in index], index=index)
        return df.fillna(method='ffill')

    def _get_task_cpu_df(self):
        """
        Get a DataFrame mapping task names to the CPU they ran on

        Use the sched_switch trace event to find which CPU each task ran
        on. Does not reflect idleness - tasks not running are shown as running
        on the last CPU they woke on.

        :returns: A Pandas DataFrame with a column for each task, showing the
                  CPU that the task was "on" at each moment in time
        """
        tasks = self.rtapp_params.keys()
        trace = self.trace

        df = trace.ftrace.sched_switch.data_frame[['next_comm', '__cpu']]
        df = df[df['next_comm'].isin(tasks)]
        df = df.pivot(index=df.index, columns='next_comm').fillna(method='ffill')
        cpu_df = df['__cpu']
        # Drop consecutive duplicates
        cpu_df = cpu_df[(cpu_df.shift(+1) != cpu_df).any(axis=1)]
        return cpu_df

    def _sort_power_df_columns(self, df):
        """
        Helper method to re-order the columns of a power DataFrame

        This has no significance for code, but when examining DataFrames by hand
        they are easier to understand if the columns are in a logical order.
        """
        node_cpus = [node.cpus for node in self.nrg_model.root.iter_nodes()]
        return pd.DataFrame(df, columns=[c for c in node_cpus if c in df])

    def _get_expected_power_df(self):
        """
        Estimate *optimal* power usage over time

        Examine a trace and use :meth:get_optimal_placements and
        :meth:EnergyModel.estimate_from_cpu_util to get a DataFrame showing the
        estimated power usage over time under ideal EAS behaviour.

        :meth:get_optimal_placements returns several optimal placements. They
        are usually equivalent, but can be drastically different in some cases.
        Currently only one of those placements is used (the first in the list).

        :returns: A Pandas DataFrame with a column each node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) and a
                  "power" column with the sum of other columns. Shows the
                  estimated *optimal* power over time.
        """
        task_utils_df = self._get_expected_task_utils_df()

        data = []
        index = []

        def exp_power(row):
            task_utils = row.to_dict()
            expected_utils = self.nrg_model.get_optimal_placements(task_utils)[0]
            power = self.nrg_model.estimate_from_cpu_util(expected_utils)
            columns = power.keys()

            # Assemble a dataframe to plot the expected utilization
            data.append(expected_utils)
            index.append(row.name)

            return pd.Series([power[c] for c in columns], index=columns)

        res_df = self._sort_power_df_columns(
            task_utils_df.apply(exp_power, axis=1))

        #self._plot_expected_util(pd.DataFrame(data, index=index))

        return res_df

    def _get_estimated_power_df(self):
        """
        Considering only the task placement, estimate power usage over time

        Examine a trace and use :meth:EnergyModel.estimate_from_cpu_util to get
        a DataFrame showing the estimated power usage over time. This assumes
        perfect cpuidle and cpufreq behaviour.

        :returns: A Pandas DataFrame with a column node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) Shows
                  the estimated power over time.
        """
        task_cpu_df = self._get_task_cpu_df()
        task_utils_df = self._get_expected_task_utils_df()
        task_utils_df.index = [time + self._get_start_time() for time in task_utils_df.index]
        tasks = self.rtapp_params.keys()

        # Create a combined DataFrame with the utilization of a task and the CPU
        # it was running on at each moment. Looks like:
        #                       utils                  cpus
        #          task_wmig0 task_wmig1 task_wmig0 task_wmig1
        # 2.375056      102.4      102.4        NaN        NaN
        # 2.375105      102.4      102.4        2.0        NaN

        df = pd.concat([task_utils_df, task_cpu_df],
                       axis=1, keys=['utils', 'cpus'])
        df = df.sort_index().fillna(method='ffill')

        # Now make a DataFrame with the estimated power at each moment.
        def est_power(row):
            cpu_utils = [0 for cpu in self.nrg_model.cpus]
            for task in tasks:
                cpu = row['cpus'][task]
                util = row['utils'][task]
                if not isnan(cpu):
                    cpu_utils[int(cpu)] += util
            power = self.nrg_model.estimate_from_cpu_util(cpu_utils)
            columns = power.keys()
            return pd.Series([power[c] for c in columns], index=columns)
        return self._sort_power_df_columns(df.apply(est_power, axis=1))


    def test_task_placement(self, energy_est_threshold_pct=5):
        """
        Test that task placement was energy-efficient

        :param energy_est_threshold_pct: Allowed margin for estimated vs
            optimal task placement energy cost
        :type energy_est_threshold_pct: int

        Compute optimal energy consumption (energy-optimal task placement)
        and compare to energy consumption estimated from the trace.
        Check that the estimated energy does not exceed the optimal energy by
        more than :attr:`energy_est_threshold_pct` percents.
        """
        exp_power = self._get_expected_power_df()
        est_power = self._get_estimated_power_df()

        exp_energy = area_under_curve(exp_power.sum(axis=1), method='rect')
        est_energy = area_under_curve(est_power.sum(axis=1), method='rect')

        msg = 'Estimated {} bogo-Joules to run workload, expected {}'.format(
            est_energy, exp_energy)
        threshold = exp_energy * (1 + (energy_est_threshold_pct / 100.))

        passed = est_energy < threshold
        res = ResultBundle(passed)
        res.add_metric(Metric("estimated_energy", est_energy, units='bogo-joules',
                              lower_is_better=True))
        res.add_metric(Metric("energy_threshold", threshold, units='bogo-joules',
                              lower_is_better=True))
        return res


from wlgen.rta import Periodic, Ramp, Step

# TODO: factorize this crap out of these classes
class OneSmallTask(GenericTestBundle):
    """
    A single 'small' (fits in a LITTLE) task
    """

    task_name = "small"

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        duty = int((cls.min_cpu_capacity(te) / 1024.) * 50)

        rtapp_params = {}
        rtapp_params[cls.task_name] = Periodic(
            duty_cycle_pct=duty,
            duration_s=1,
            period_ms=16
        )

        return rtapp_params

class ThreeSmallTasks(GenericTestBundle):
    """
    Three 'small' (fits in a LITTLE) tasks
    """
    task_prefix = "small"

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        duty = int((cls.min_cpu_capacity(te) / 1024.) * 50)

        rtapp_params = {}
        for i in range(3):
            rtapp_params["{}_{}".format(cls.task_prefix, i)] = Periodic(
                duty_cycle_pct=duty,
                duration_s=1,
                period_ms=16
            )

        return rtapp_params

class TwoBigTasks(GenericTestBundle):
    """
    """

    task_prefix = "big"

    @classmethod
    def create_rtapp_params(cls, te):
        # 80% of the biggest CPU's capacity
        duty = int((cls.max_cpu_capacity(te) / 1024.) * 80)

        rtapp_params = {}
        for i in range(2):
            rtapp_params["{}_{}".format(cls.task_prefix, i)] = Periodic(
                duty_cycle_pct=duty,
                duration_s=1,
                period_ms=16
            )

        return rtapp_params

class TwoBigThreeSmall(GenericTestBundle):
    """
    """

    small_prefix = "small"
    big_prefix = "big"

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        small_duty = int((cls.min_cpu_capacity(te) / 1024.) * 50)
        # 80% of the biggest CPU's capacity
        big_duty = int((cls.max_cpu_capacity(te) / 1024.) * 80)

        rtapp_params = {}

        for i in range(3):
            rtapp_params["{}_{}".format(cls.small_prefix, i)] = Periodic(
                duty_cycle_pct=small_duty,
                duration_s=1,
                period_ms=16
            )

        for i in range(2):
            rtapp_params["{}_{}".format(cls.big_prefix, i)] = Periodic(
                duty_cycle_pct=big_duty,
                duration_s=1,
                period_ms=16
            )

        return rtapp_params

class RampUp(GenericTestBundle):
    """
    """
    task_name = "ramp_up"

    @classmethod
    def create_rtapp_params(cls, te):
        rtapp_params = {}
        rtapp_params[cls.task_name] = Ramp(
            start_pct=5,
            end_pct=70,
            delta_pct=5,
            time_s=.5,
            period_ms=16
        )

        return rtapp_params

class RampDown(GenericTestBundle):
    """
    """
    task_name = "ramp_down"

    @classmethod
    def create_rtapp_params(cls, te):
        rtapp_params = {}
        rtapp_params[cls.task_name] = Ramp(
            start_pct=70,
            end_pct=5,
            delta_pct=5,
            time_s=.5,
            period_ms=16
        )

        return rtapp_params

class EnergyModelWakeMigration(GenericTestBundle):
    """
    """
    task_prefix = "emwm"

    @classmethod
    def create_rtapp_params(cls, te):
        rtapp_params = {}
        capacities = te.target.sched.get_capacities()
        max_capa = cls.max_cpu_capacity(te)
        bigs = [cpu for cpu, capacity in capacities.items() if capacity == max_capa]

        for i in range(len(bigs)):
            rtapp_params["{}_{}".format(cls.task_prefix, i)] = Step(
                start_pct=10,
                end_pct=70,
                time_s=2,
                loops=2,
                period_ms=16
            )

        return rtapp_params

################################################################################
################################################################################

import sys
import random
import json
import os.path

from target_script import TargetScript
from devlib.module.hotplug import HotplugModule
from devlib.exception import TimeoutError

class HotplugTestBundle(TestBundle):

    def __init__(self, target_alive, hotpluggable_cpus, live_cpus):
        self.target_alive = target_alive
        self.hotpluggable_cpus = hotpluggable_cpus
        self.live_cpus = live_cpus

    @classmethod
    def _random_cpuhp_seq(cls, seed, nr_operations,
                          hotpluggable_cpus, max_cpus_off):
        """
        Yield a consistent random sequence of CPU hotplug operations

        :param seed: Seed of the RNG
        :param nr_operations: Number of operations in the sequence
            <= 0 will encode 'no sleep'
        :param max_cpus_off: Max number of CPUs plugged-off

        "Consistent" means that a CPU will be plugged-in only if it was
        plugged-off before (and vice versa). Moreover the state of the CPUs
        once the sequence has completed should the same as it was before.

        The actual length of the sequence might differ from the requested one
        by 1 because it's easier to implement and it shouldn't be an issue for
        most test cases.
        """
        cur_on_cpus = hotpluggable_cpus[:]
        cur_off_cpus = []
        i = 0
        while i < nr_operations - len(cur_off_cpus):
            if len(cur_on_cpus)<=1 or len(cur_off_cpus)>=max_cpus_off:
                # Force plug IN when only 1 CPU is on or too many are off
                plug_way = 1
            elif not cur_off_cpus:
                # Force plug OFF if all CPUs are on
                plug_way = 0 # Plug OFF
            else:
                plug_way = random.randint(0,1)

            src = cur_off_cpus if plug_way else cur_on_cpus
            dst = cur_on_cpus if plug_way else cur_off_cpus
            cpu = random.choice(src)
            src.remove(cpu)
            dst.append(cpu)
            i += 1
            yield cpu, plug_way

        # Re-plug offline cpus to come back to original state
        for cpu in cur_off_cpus:
            yield cpu, 1

    @classmethod
    def _random_cpuhp_script(cls, te, sequence, sleep_min_ms, sleep_max_ms, timeout_s):
        """
        :param sleep_min_ms: Min sleep duration between hotplugs
        :param sleep_max_ms: Max sleep duration between hotplugs
        """

        shift = '    '
        script = TargetScript(te, 'random_cpuhp.sh')

        # Record configuration
        # script.append('# File generated automatically')
        # script.append('# Configuration:')
        # script.append('# {}'.format(cls.hp_stress))
        # script.append('# Hotpluggable CPUs:')
        # script.append('# {}'.format(cls.hotpluggable_cpus))

        script.append('while true')
        script.append('do')
        for cpu, plug_way in sequence:
            # Write in sysfs entry
            cmd = 'echo {} > {}'.format(plug_way, HotplugModule._cpu_path(te.target, cpu))
            script.append(shift + cmd)
            # Sleep if necessary
            if sleep_max_ms > 0:
                sleep_dur_sec = random.randint(sleep_min_ms, sleep_max_ms)/1000.0
                script.append(shift + 'sleep {}'.format(sleep_dur_sec))
        script.append('done &')

        # Make sure to stop the hotplug stress after timeout_s seconds
        script.append('LOOP_PID=$!')
        script.append('sleep {}'.format(timeout_s))
        script.append('kill -9 $LOOP_PID')

        return script

    @classmethod
    def _from_target(cls, te, res_dir=None, seed=None, nr_operations=100,
            sleep_min_ms=10, sleep_max_ms=100, duration_s=10,
            max_cpus_off=sys.maxint):

        if not seed:
            random.seed()
            seed = random.randint(0, sys.maxint)

        te.target.hotplug.online_all()
        hotpluggable_cpus = te.target.hotplug.list_hotpluggable_cpus()

        sequence = cls._random_cpuhp_seq(
            seed, nr_operations, hotpluggable_cpus, max_cpus_off
        )

        script = cls._random_cpuhp_script(
            te, sequence, sleep_min_ms, sleep_max_ms, duration_s
        )
        script.push(res_dir)

        target_alive = True
        timeout = duration_s + 60

        try:
            script.run(as_root=True, timeout=timeout)
            te.target.hotplug.online_all()
        except TimeoutError:
            #msg = 'Target not responding after {} seconds ...'
            #cls._log.info(msg.format(timeout))
            target_alive = False

        live_cpus = te.target.list_online_cpus() if target_alive else []

        return cls(target_alive, hotpluggable_cpus, live_cpus)

    def test_target_alive(self):
        """
        Test that the hotplugs didn't leave the target in an unusable state
        """
        return self.target_alive

    def test_cpus_alive(self):
        """
        Test that all CPUs came back online after the hotplug operations
        """
        return self.hotpluggable_cpus == self.live_cpus

################################################################################
################################################################################

# class AndroidWorkload(LisaWorkload):

#     def _setup_wload(self):
#         self.target.set_auto_brightness(0)
#         self.target.set_brightness(0)

#         self.target.ensure_screen_is_on()
#         self.target.swipe_to_unlock()

#         self.target.set_auto_rotation(0)
#         self.target.set_rotation(1)

#     def _run_wload(self):
#         pass

#     def _teardown_wload(self):
#         self.target.set_auto_rotation(1)
#         self.target.set_auto_brightness(1)

#     def run(self, trace_tool):
#         if trace_tool == "ftrace":
#             pass
#         elif trace_tool == "systrace":
#             pass

#         self._setup_wload()

#         with self.te.record_ftrace():
#             self._run_wload()

#         self._teardown_wload()

# from target_script import TargetScript
# from devlib.target import AndroidTarget

# class GmapsWorkload(AndroidWorkload):

#     def _setup_wload(self):
#         super(GmapsWorkload, self)._setup_wload()

#         self.script = TargetScript(self.te, "gmaps_swiper.sh")

#         for i in range(self.swipe_count):
#             # Swipe right
#             self.script.input_swipe_pct(40, 50, 60, 60)
#             #AndroidTarget.input_swipe_pct(self.script, 40, 50, 60, 60)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe down
#             AndroidTarget.input_swipe_pct(self.script, 50, 60, 50, 40)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe left
#             AndroidTarget.input_swipe_pct(self.script, 60, 50, 40, 50)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe up
#             AndroidTarget.input_swipe_pct(self.script, 50, 40, 50, 60)
#             AndroidTarget.sleep(self.script, 1)

#         # Push script to the target
#         self.script.push()

#     def _run_wload(self):
#         self.script.run()

#     def run(self, swipe_count=10):
#         self.swipe_count = swipe_count

#         super(GmapsWorkload, self).run("ftrace")
