import os
from utils.executor import Executor

from utils.perf_analysis import PerfAnalysis
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

    def __bool__(self):
        return self.passed

    def add_metric(self, metric):
        self.metrics.append(metric)

class TestBundle(object):

    def __init__(self, res_dir):
        self.res_dir = res_dir


class LisaWorkload(object):
    """
    A class meant to describe "workloads" to execute on a target
    """
    def __init__(self, te, res_dir=None):
        self.te = te
        self.target = te.target
        # Logger stuff ?

        # If no res dir is given, ask TestEnv for one
        self.res_dir = res_dir or te.get_res_dir()

    def run(self):
        """
        The main method of the :class:`LisaWorkload` class
        """
        raise NotImplementedError()

class TestableWorkload(LisaWorkload):
    """
    A Workload that will generate some :class:`TestBundle`
    """
    @property
    def test_bundle(self):
        if not self._test_bundle:
            raise RuntimeError(
                "No test bundle generated, call run() first"
            )
        return self._test_bundle

    def __init__(self, te, res_dir=None):
        super(TestableWorkload, self).__init__(te, res_dir)
        self._test_bundle = None

    def create_test_bundle(self):
        """
        Serialize the workload artefacts into a TestBundle

        :returns: a :class:`TestBundle`
        """
        raise NotImplementedError(
            "Testable workloads require an implementation of this method"
        )

    def collect_test(self):
        self.run()
        self._test_bundle = self.create_test_bundle()
        return self._test_bundle

class GenericTestBundle(TestBundle):
    """
    Yadda yadda, EAS placement and energy test cases
    """

    energy_est_threshold_pct = 5
    """Allowed margin for estimated vs optimal task placement energy cost"""

    def __init__(self, res_dir):
        super(GenericTestBundle, self).__init__(res_dir)

    def test_slack(self, out_dir=None, negative_slack_allowed_pct=15):
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
        for task, slack in slacks.items():
            res.add_metric(Metric("slack_{}".format(task), slack,
                                  units='%', lower_is_better=True))

        return res

    def test_task_placement(self, out_dir=None, energy_est_threshold_pct=5):
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
        raise NotImplementedError("lol 2bad nice try")

class SyntheticWorkload(TestableWorkload):
    """
    Description of a synthetic test

    Synthetic tests are based on rt-app workloads that attempt to stress a
    specific scheduler behaviour. The resulting trace can then be post-processed
    and used as scheduler unit tests.
    """

    ftrace_conf = {
        "events" : ["sched_switch"],
    }

    def run(self, rtapp_params):
        self.rtapp_params = rtapp_params

        self.te.ftrace_conf(self.ftrace_conf)

        self.wload = RTA(self.target, self.__class__.__name__, self.te.calibration())
        self.wload.conf(kind='profile', params=rtapp_params,
                        run_dir=Executor.get_run_dir(self.target))

        trace_path = os.path.join(self.res_dir, "trace.dat")

        with self.te.record_ftrace(trace_path):
            with self.te.freeze_userspace():
                self.wload.run(out_dir=self.res_dir)

    def create_test_bundle(self):
        return GenericTestBundle(self.res_dir)


from wlgen.rta import Periodic

class OneSmallTask(SyntheticWorkload):

    def run(self):
        # 50% of the smallest CPU's capacity
        duty = int((min(self.target.sched.get_capacities().values()) / 1024.) * 50)

        rtapp_params = {}
        rtapp_params["small"] = Periodic(
            duty_cycle_pct=duty,
            duration_s=1,
            period_ms=16
        ).get()

        super(OneSmallTask, self).run(rtapp_params)

################################################################################
################################################################################

import sys
import random
import json
import os.path

from utils.target_script import TargetScript
from devlib.module.hotplug import HotplugModule
from devlib.exception import TimeoutError

class HotplugTestBundle(TestBundle):
    def __init__(self, res_dir, target_alive):
        super(HotplugTestBundle, self).__init__(res_dir)
        self.target_alive = target_alive

    def test_system_alive(self):
        return ResultBundle(self.target_alive)

class CpuHotplugTorture(TestableWorkload):

    def run(self, seed=None, nr_operations=100,
            sleep_min_ms=10, sleep_max_ms=100, duration_s=10,
            max_cpus_off=sys.maxsize, max_duration_s=10):

        if not seed:
            random.seed()
            seed = random.randint(0, sys.maxsize)

        hotpluggable_cpus = [cpu for cpu in self.target.list_online_cpus() if self.target.file_exists(self._cpuhp_path(cpu))]

        sequence = self._random_cpuhp_seq(
            seed, nr_operations, hotpluggable_cpus, max_cpus_off
        )

        script = self._random_cpuhp_script(
            sequence, sleep_min_ms, sleep_max_ms, duration_s
        )
        script.push(self.res_dir)

        target_alive = True
        timeout = duration_s + 60

        try:
            script.run(as_root=True, timeout=timeout)
            self.target.hotplug.online_all()
        except TimeoutError:
            #msg = 'Target not responding after {} seconds ...'
            #cls._log.info(msg.format(timeout))
            target_alive = False

        #return HotplugTestBundle(self.res_dir, target_alive)

    def create_test_bundle(self):
        pass

    def _cpuhp_path(self, cpu):
        cpu = 'cpu{}'.format(cpu)
        return self.target.path.join(HotplugModule.base_path, cpu, 'online')

    def _random_cpuhp_seq(self, seed, nr_operations,
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

    def _random_cpuhp_script(self, sequence, sleep_min_ms, sleep_max_ms, timeout_s):
        """
        :param sleep_min_ms: Min sleep duration between hotplugs
        :param sleep_max_ms: Max sleep duration between hotplugs
        """

        shift = '    '
        script = TargetScript(self.te, 'random_cpuhp.sh')

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
            cmd = 'echo {} > {}'.format(plug_way, self._cpuhp_path(cpu))
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

################################################################################
################################################################################

class AndroidWorkload(LisaWorkload):

    def _setup_wload(self):
        self.target.set_auto_brightness(0)
        self.target.set_brightness(0)

        self.target.ensure_screen_is_on()
        self.target.swipe_to_unlock()

        self.target.set_auto_rotation(0)
        self.target.set_rotation(1)

    def _run_wload(self):
        pass

    def _teardown_wload(self):
        self.target.set_auto_rotation(1)
        self.target.set_auto_brightness(1)

    def run(self, trace_tool):
        if trace_tool == "ftrace":
            pass
        elif trace_tool == "systrace":
            pass

        self._setup_wload()

        with self.te.record_ftrace():
            self._run_wload()

        self._teardown_wload()

from utils.target_script import TargetScript
from devlib.target import AndroidTarget

class GmapsWorkload(AndroidWorkload):

    def _setup_wload(self):
        super(GmapsWorkload, self)._setup_wload()

        self.script = TargetScript(self.te, "gmaps_swiper.sh")

        for i in range(self.swipe_count):
            # Swipe right
            self.script.input_swipe_pct(40, 50, 60, 60)
            #AndroidTarget.input_swipe_pct(self.script, 40, 50, 60, 60)
            AndroidTarget.sleep(self.script, 1)
            # Swipe down
            AndroidTarget.input_swipe_pct(self.script, 50, 60, 50, 40)
            AndroidTarget.sleep(self.script, 1)
            # Swipe left
            AndroidTarget.input_swipe_pct(self.script, 60, 50, 40, 50)
            AndroidTarget.sleep(self.script, 1)
            # Swipe up
            AndroidTarget.input_swipe_pct(self.script, 50, 40, 50, 60)
            AndroidTarget.sleep(self.script, 1)

        # Push script to the target
        self.script.push()

    def _run_wload(self):
        self.script.run()

    def run(self, swipe_count=10):
        self.swipe_count = swipe_count

        super(GmapsWorkload, self).run("ftrace")
