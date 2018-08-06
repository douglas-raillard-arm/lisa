import os
from executor import Executor

from perf_analysis import PerfAnalysis
from wlgen.rta import RTA


class ResultBundle(object):
    """
    Bundle for storing test results such as metrics or pass/fail
    """
    def __init__(self, passed):
        self.passed = passed

    def __nonzero__(self):
        return self.passed


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

    def test(self):
        self.run()
        self._test_bundle = self.create_test_bundle()
        return self._test_bundle

class GenericTestBundle(TestBundle):
    """
    Yadda yadda, EAS placement and energy test cases
    """

    negative_slack_allowed_pct = 15
    """Percentage of RT-App task activations with negative slack allowed"""

    energy_est_threshold_pct = 5
    """Allowed margin for estimated vs optimal task placement energy cost"""

    def test_slack(self, out_dir=None, negative_slack_allowed_pct=None):
        """
        Assert that the RTApp workload was given enough performance

        :param out_dir: Output directory for test artefacts
        :type out_dir: str

        Use :class:PerfAnalysis to find instances where the RT-App workload
        wasn't able to complete its activations (i.e. its reported "slack"
        was negative). Assert that this happened less than
        `negative_slack_allowed_pct` percent of the time.
        """
        if not negative_slack_allowed_pct:
            negative_slack_allowed_pct = self.negative_slack_allowed_pct

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
        res.__str__ = lambda x : "42"

    def test_task_placement(self, out_dir=None, energy_est_threshold_pct=None):
        """
        Test that task placement was energy-efficient

        Use :meth:`get_expected_power_df` and :meth:`get_power_df` to estimate
        optimal and observed power usage for task placements of the experiment's
        workload. Assert that the observed power does not exceed the optimal
        power by more than :attr:energy_est_threshold_pct percents.
        """
        if not energy_est_threshold_pct:
            energy_est_threshold_pct = self.energy_est_threshold_pct

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

from wlgen.rta import Periodic

class OneSmallTask(SyntheticWorkload):

    def run(self):
        # 50% of the smallest CPU's capacity
        duty = int((min(self.target.sched.get_capacities().values()) / 1024.) * 50)
        print duty

        rtapp_params = {}
        rtapp_params["small"] = Periodic(
            duty_cycle_pct=duty,
            duration_s=1,
            period_ms=16
        ).get()

        super(OneSmallTask, self).run(rtapp_params)


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

from target_script import TargetScript
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
