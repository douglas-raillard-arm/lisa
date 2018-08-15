import os
from executor import Executor

from perf_analysis import PerfAnalysis
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
    Bundle for storing test results such as metrics or pass/fail
    """
    def __init__(self, passed):
        self.passed = passed
        self.metrics = []

    def __nonzero__(self):
        return self.passed

    def add_metric(self, metric):
        self.metrics.append(metric)

import shutil

class TestBundle(object):
    """
    Description of a Lisa test bundle

    :meth:`from_target` will use :meth:`from_disk` to create the actual
    instance, which guarantees that there is no discrepancy between the target
    and disk methods. If it makes no sense for the use-case to save any kind of
    state on the disk (e.g. the end result of the target execution is just a
    boolean value), that can be overriden.
    """

    def __init__(self, res_dir):
        self.res_dir = res_dir

    @classmethod
    def _from_target(cls, te, res_dir):
        """
        Internals of the target factory method.

        TODO: Please give me a proper name
        """
        raise NotImplementedError()

    @classmethod
    def from_target(cls, te, res_dir=None, **kwargs):
        """
        Factory method to create a bundle using a live target

        This is mostly boiler-plate code around :meth:`_from_target`
        """
        if not res_dir:
            res_dir = te.get_res_dir()

        # Logger stuff?

        cls._from_target(te, res_dir, **kwargs)
        return cls.from_disk(res_dir)

    @classmethod
    def from_disk(cls, res_dir):
        """
        Factory method to create the bundle from a disk location
        """
        raise NotImplementedError()

class TransientTestBundle(TestBundle):
    """
    A TestBundle that cannot be saved to the disk, or for which it does not make
    sense to do so.
    """

    def __init__(self):
        raise RuntimeError("yadda yadda resbundle")

    @classmethod
    def from_target(cls, te, res_dir=None, **kwargs):
        """
        """
        if not res_dir:
            res_dir = te.get_res_dir()

        # Logger stuff?

        return cls._from_target(te, res_dir, **kwargs)

    @classmethod
    def from_disk(cls, res_dir):
        raise RuntimeError("{} can't be saved to disk".format(cls.__name__))


from bart.common.Utils import area_under_curve
#from energy_model import save_em_yaml, load_em_yaml
class GenericTestBundle(TestBundle):
    """
    Yadda yadda, EAS placement and energy test cases
    """

    ftrace_conf = {
        "events" : ["sched_switch"],
    }

    def __init__(self, res_dir):
        super(GenericTestBundle, self).__init__(res_dir)

        #self.nrg_model = load_em_yaml(res_dir)

        with open(os.path.join(res_dir, "{}.json".format(self.__class__.__name__)), "r") as fh:
            self.rtapp_params = json.load(fh)

    @classmethod
    def create_rtapp_params(cls, te):
        """
        :returns: dict
        """
        raise NotImplementedError()

    @classmethod
    def _from_target(cls, te, res_dir):
        rtapp_params = cls.create_rtapp_params(te)

        wload = RTA(te.target, cls.__name__, te.calibration())
        wload.conf(kind='profile', params=rtapp_params,
                   run_dir=Executor.get_run_dir(te.target))

        trace_path = os.path.join(res_dir, "trace.dat")
        te.ftrace_conf(cls.ftrace_conf)

        with te.record_ftrace(trace_path):
            with te.freeze_userspace():
                wload.run(out_dir=res_dir)

        # Serialize the energy model to the res_dir

        return cls.from_disk(res_dir)

    @classmethod
    def from_disk(cls, res_dir):
        return cls(res_dir)

    def test_slack(self, negative_slack_allowed_pct=15):
        """
        Assert that the RTApp workload was given enough performance

        :param out_dir: Output directory for test artefacts
        :type out_dir: str

        :param negative_slack_allowed_pct: Allowed percentage of RT-app task
            activations with negative slack.
        :type negative_slack_allowed_pct: int

        Use :class:PerfAnalysis to find instances where the RT-App workload
        wasn't able to complete its activations (i.e. its reported "slack"
        was negative). Assert that this happened less than
        `negative_slack_allowed_pct` percent of the time.
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

    def get_task_utils_df(self):
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
            time = self.get_start_time(experiment) + params['delay']
            add_transition(time, task, 0)
            for _ in range(params.get('loops', 1)):
                for phase in params['phases']:
                    util = (phase.duty_cycle_pct * util_scale / 100.)
                    add_transition(time, task, util)
                    time += phase.duration_s
            add_transition(time, task, 0)

        index = sorted(transitions.keys())
        df = pd.DataFrame([transitions[k] for k in index], index=index)
        return df.fillna(method='ffill')

    def get_expected_power_df(self):
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
        task_utils_df = self.get_task_utils_df(experiment)

        data = []
        index = []

        def exp_power(row):
            task_utils = row.to_dict()
            expected_utils = nrg_model.get_optimal_placements(task_utils)[0]
            power = nrg_model.estimate_from_cpu_util(expected_utils)
            columns = power.keys()

            # Assemble a dataframe to plot the expected utilization
            data.append(expected_utils)
            index.append(row.name)

            return pd.Series([power[c] for c in columns], index=columns)

        res_df = self._sort_power_df_columns(
            task_utils_df.apply(exp_power, axis=1))

        self.plot_expected_util(experiment, pd.DataFrame(data, index=index))

        return res_df

    def test_task_placement(self, energy_est_threshold_pct=5):
        """
        Test that task placement was energy-efficient

        :param energy_est_threshold_pct: Allowed margin for estimated vs
            optimal task placement energy cost
        :type energy_est_threshold_pct: int

        Use :meth:`get_expected_power_df` and :meth:`get_power_df` to estimate
        optimal and observed power usage for task placements of the experiment's
        workload. Assert that the observed power does not exceed the optimal
        power by more than :attr:energy_est_threshold_pct percents.
        """
        exp_power = self.get_expected_power_df(experiment)
        est_power = self.get_power_df(experiment)

        exp_energy = area_under_curve(exp_power.sum(axis=1), method='rect')
        est_energy = area_under_curve(est_power.sum(axis=1), method='rect')

        msg = 'Estimated {} bogo-Joules to run workload, expected {}'.format(
            est_energy, exp_energy)
        threshold = exp_energy * (1 + (energy_est_threshold_pct / 100.))
        self.assertLess(est_energy, threshold, msg=msg)

from wlgen.rta import Periodic

class OneSmallTask(GenericTestBundle):
    """
    A single 'small' (fits in a LITTLE) task
    """

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        duty = int((min(te.target.sched.get_capacities().values()) / 1024.) * 50)

        rtapp_params = {}
        rtapp_params["small"] = Periodic(
            duty_cycle_pct=duty,
            duration_s=1,
            period_ms=16
        ).get()

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

class HotplugTestBundle(TransientTestBundle):

    @classmethod
    def _cpuhp_path(cls, te, cpu):
        cpu = 'cpu{}'.format(cpu)
        return te.target.path.join(HotplugModule.base_path, cpu, 'online')

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
            cmd = 'echo {} > {}'.format(plug_way, cls._cpuhp_path(te, cpu))
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
    def _from_target(cls, te,res_dir=None, seed=None, nr_operations=100,
            sleep_min_ms=10, sleep_max_ms=100, duration_s=10,
            max_cpus_off=sys.maxint, max_duration_s=10):

        if not seed:
            random.seed()
            seed = random.randint(0, sys.maxint)

        hotpluggable_cpus = filter(
            lambda cpu: te.target.file_exists(cls._cpuhp_path(te, cpu)),
            te.target.list_online_cpus()
        )

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

        return ResultBundle(target_alive)

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
