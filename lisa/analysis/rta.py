# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import re
import glob
from collections import namedtuple

import pandas as pd

from lisa.analysis.base import AnalysisHelpers, TraceAnalysisBase
from lisa.datautils import df_filter_task_ids
from lisa.trace import TaskID, requires_events
from lisa.utils import memoized, deprecate


RefTime = namedtuple("RefTime", ['kernel', 'user'])
"""
Named tuple to synchronize kernel and userspace (``rt-app``) timestamps.
"""


PhaseWindow = namedtuple("PhaseWindow", ['id', 'start', 'end'])
"""
Named tuple with fields:

    * ``id``: integer ID of the phase
    * ``start``: timestamp of the start of the phase
    * ``end``: timestamp of the end of the phase

"""


class RTAEventsAnalysis(TraceAnalysisBase):
    """
    Support for RTA events analysis.

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'rta'

    def _task_filtered(self, df, task=None):
        if not task:
            return df

        task = self.trace.get_task_id(task)

        if task not in self.rtapp_tasks:
            raise ValueError("Task [{}] is not an rt-app task: {}"
                             .format(task, self.rtapp_tasks))

        return df_filter_task_ids(df, [task],
                                  pid_col='__pid', comm_col='__comm')

    @memoized
    def _get_rtapp_tasks(self):
        task_ids = set()
        for evt in self.trace.available_events:
            if not evt.startswith('rtapp_'):
                continue
            df = self.trace.df_events(evt)
            for pid, name in df[['__pid', '__comm']].drop_duplicates().values:
                task_ids.add(TaskID(pid, name))
        return sorted(task_ids)

    @property
    def rtapp_tasks(self):
        """
        List of :class:`lisa.trace.TaskID` of the ``rt-app`` tasks present in
        the trace.
        """
        return self._get_rtapp_tasks()


###############################################################################
# DataFrame Getter Methods
###############################################################################

    ###########################################################################
    # rtapp_main events related methods
    ###########################################################################

    @requires_events('rtapp_main')
    def df_rtapp_main(self):
        """
        Dataframe of events generated by the rt-app main task.

        :returns: a :class:`pandas.DataFrame` with:

          * A ``__comm`` column: the actual rt-app trace task name
          * A ``__cpu``  column: the CPU on which the task was running at event
                                 generation time
          * A ``__line`` column: the ftrace line numer
          * A ``__pid``  column: the PID of the task
          * A ``data``   column: the data corresponding to the reported event
          * An ``event`` column: the event generated

        The ``event`` column can report these events:

          * ``start``: the start of the rt-app main thread execution
          * ``end``: the end of the rt-app main thread execution
          * ``clock_ref``: the time rt-app gets the clock to be used for logfile entries

        The ``data`` column reports:

          * the base timestamp used for logfile generated event for the ``clock_ref`` event
          * ``NaN`` for all the other events

        """
        return self.trace.df_events('rtapp_main')

    @property
    @df_rtapp_main.used_events
    def rtapp_window(self):
        """
        Return the time range the rt-app main thread executed.

        :returns: a tuple(start_time, end_time)
        """
        df = self.df_rtapp_main()
        return (
            df[df.event == 'start'].index.values[0],
            df[df.event == 'end'].index.values[0])

    @property
    @df_rtapp_main.used_events
    def rtapp_reftime(self):
        """
        Return the tuple representing the ``kernel`` and ``user`` timestamp.

        RTApp log events timestamps are generated by the kernel ftrace
        infrastructure. This method allows to know which trace timestamp
        corresponds to the rt-app generated timestamps stored in log files.

        :returns: a :class:`RefTime` reporting ``kernel`` and ``user``
                  timestamps.
        """
        df = self.df_rtapp_main()
        df = df[df['event'] == 'clock_ref']
        return RefTime(df.index.values[0], df.data.values[0])

    ###########################################################################
    # rtapp_task events related methods
    ###########################################################################

    @requires_events('rtapp_task')
    def df_rtapp_task(self, task=None):
        """
        Dataframe of events generated by each rt-app generated task.

        :param task: the (optional) rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: a :class:`pandas.DataFrame` with:

          * A ``__comm`` column: the actual rt-app trace task name
          * A ``__cpu``  column: the CPU on which the task was running at event
                                 generation time
          * A ``__line`` column: the ftrace line numer
          * A ``__pid``  column: the PID of the task
          * An ``event`` column: the event generated

        The ``event`` column can report these events:

          * ``start``: the start of the ``__pid``:``__comm`` task execution
          * ``end``: the end of the ``__pid``:``__comm`` task execution

        """
        df = self.trace.df_events('rtapp_task')
        return self._task_filtered(df, task)

    ###########################################################################
    # rtapp_loop events related methods
    ###########################################################################

    @requires_events('rtapp_loop')
    def df_rtapp_loop(self, task=None):
        """
        Dataframe of events generated by each rt-app generated task.

        :param task: the (optional) rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: a :class:`pandas.DataFrame` with:

          * A  ``__comm`` column: the actual rt-app trace task name
          * A  ``__cpu``  column: the CPU on which the task was running at event
                                 generation time
          * A  ``__line`` column: the ftrace line numer
          * A  ``__pid``  column: the PID of the task
          * An ``event``  column: the generated event
          * A  ``phase``  column: the phases counter for each ``__pid``:``__comm`` task
          * A  ``phase_loop``  colum: the phase_loops's counter
          * A  ``thread_loop`` column: the thread_loop's counter

        The ``event`` column can report these events:

          * ``start``: the start of the ``__pid``:``__comm`` related event
          * ``end``: the end of the ``__pid``:``__comm`` related event

        """
        df = self.trace.df_events('rtapp_loop')
        return self._task_filtered(df, task)

    @df_rtapp_loop.used_events
    def _get_rtapp_phases(self, event, task):
        df = self.df_rtapp_loop(task)
        df = df[df.event == event]

        # Sort START/END phase loop event from newers/older and...
        if event == 'start':
            df = df[df.phase_loop == 0]
        elif event == 'end':
            df = df.sort_values(by=['phase_loop', 'thread_loop'],
                                ascending=False)
        # ... keep only the newest/oldest event
        df = df.groupby(['__comm', '__pid', 'phase', 'event']).head(1)

        # Reorder the index and keep only required cols
        df = (df.sort_index()[['__comm', '__pid', 'phase']]
              .reset_index()
              .set_index(['__comm', '__pid', 'phase']))

        return df

    @memoized
    @_get_rtapp_phases.used_events
    def df_rtapp_phases_start(self, task=None):
        """
        Dataframe of phases start times.

        :param task: the (optional) rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: a :class:`pandas.DataFrame` with:

          * A  ``__comm`` column: the actual rt-app trace task name
          * A  ``__pid``  column: the PID of the task
          * A  ``phase``  column: the phases counter for each ``__pid``:``__comm`` task

        The ``index`` represents the timestamp of a phase start event.
        """
        return self._get_rtapp_phases('start', task)

    @memoized
    @_get_rtapp_phases.used_events
    def df_rtapp_phases_end(self, task=None):
        """
        Dataframe of phases end times.

        :param task: the (optional) rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: a :class:`pandas.DataFrame` with:

          * A  ``__comm`` column: the actual rt-app trace task name
          * A  ``__pid``  column: the PID of the task
          * A  ``phase``  column: the phases counter for each ``__pid``:``__comm`` task

        The ``index`` represents the timestamp of a phase end event.
        """
        return self._get_rtapp_phases('end', task)

    @df_rtapp_phases_start.used_events
    def _get_task_phase(self, event, task, phase):
        task = self.trace.get_task_id(task)
        if event == 'start':
            df = self.df_rtapp_phases_start(task)
        elif event == 'end':
            df = self.df_rtapp_phases_end(task)
        if phase and phase < 0:
            phase += len(df)
        phase += 1 # because of the followig "head().tail()" filter
        return df.loc[task.comm].head(phase).tail(1).Time.values[0]

    @_get_task_phase.used_events
    def df_rtapp_phase_start(self, task, phase=0):
        """
        Start of the specified phase for a given task.

        A negative phase value can be used to count from the oldest, e.g. -1
        represents the last phase.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :param phase: the ID of the phase start to return (default 0)
        :type phase: int

        :returns: the requires task's phase start timestamp
        """
        return self._get_task_phase('start', task, phase)

    @_get_task_phase.used_events
    def df_rtapp_phase_end(self, task, phase=-1):
        """
        End of the specified phase for a given task.

        A negative phase value can be used to count from the oldest, e.g. -1
        represents the last phase.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :param phase: the ID of the phase end to return (default -1)
        :type phase: int

        :returns: the requires task's phase end timestamp
        """
        return self._get_task_phase('end', task, phase)

    @df_rtapp_task.used_events
    def tasks_window(self, task):
        """
        Return the start end end time for the specified task.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID
        """
        task_df = self.df_rtapp_task(task)
        start_time = task_df[task_df.event == "start"].index[0]
        end_time = task_df[task_df.event == "end"].index[0]

        return (start_time, end_time)

    @df_rtapp_phases_start.used_events
    def task_phase_window(self, task, phase):
        """
        Return the window of a requested task phase.

        For the specified ``task`` it returns a tuple with the (start, end)
        time of the requested ``phase``. A negative phase number can be used to
        count phases backward from the last (-1) toward the first.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :param phase: The ID of the phase to consider
        :type phase: int

        :rtype: PhaseWindow
        """
        phase_start = self.df_rtapp_phase_start(task, phase)
        phase_end = self.df_rtapp_phase_end(task, phase)

        return PhaseWindow(phase, phase_start, phase_end)

    @task_phase_window.used_events
    def task_phase_at(self, task, timestamp):
        """
        Return the :class:`PhaseWindow` for the specified
        task and timestamp.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :param timestamp: the timestamp to get the phase for
        :type timestamp: int or float

        :returns: the ID of the phase corresponding to the specified timestamp.
        """
        # Last phase is special, compute end time as start + duration
        last_phase_end = self.df_phases(task).index[-1]
        last_phase_end += float(self.df_phases(task).iloc[-1].values)
        if timestamp > last_phase_end:
            raise ValueError('Passed timestamp ({}) is after last phase end ({})'.format(
                timestamp, last_phase_end))

        phase_id = len(self.df_phases(task)) - \
                   len(self.df_phases(task)[timestamp:]) - 1
        if phase_id < 0:
            raise ValueError('negative phase ID')

        return self.task_phase_window(task, phase_id)


    ###########################################################################
    # rtapp_phase events related methods
    ###########################################################################

    @requires_events('rtapp_event')
    def df_rtapp_event(self, task=None):
        """
        Returns a :class:`pandas.DataFrame` of all the rt-app generated events.

        :param task: the (optional) rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: a :class:`pandas.DataFrame` with:

          * A  ``__comm`` column: the actual rt-app trace task name
          * A  ``__pid``  column: the PID of the task
          * A ``__cpu``  column: the CPU on which the task was running at event
                                 generation time
          * A ``__line`` column: the ftrace line numer
          * A ``type`` column: the type of the generated event
          * A ``desc`` column: the mnemonic type of the generated event
          * A ``id`` column: the ID of the resource associated to the event,
                             e.g. the ID of the fired timer

        The ``index`` represents the timestamp of the event.
        """
        df = self.trace.df_events('rtapp_event')
        return self._task_filtered(df, task)

    ###########################################################################
    # rtapp_stats events related methods
    ###########################################################################

    @memoized
    @requires_events('rtapp_stats')
    def _get_stats(self):
        df = self.trace.df_events('rtapp_stats').copy(deep=True)
        # Add performance metrics column, performance is defined as:
        #             slack
        #   perf = -------------
        #          period - run
        df['perf_index'] = df['slack'] / (df['c_period'] - df['c_run'])

        return df

    @_get_stats.used_events
    def df_rtapp_stats(self, task=None):
        """
        Returns a :class:`pandas.DataFrame` of all the rt-app generated stats.

        :param task: the (optional) rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: a :class:`pandas.DataFrame` with a set of colums representing
            the stats generated by rt-app after each loop.


        .. seealso:: the rt-app provided documentation:
            https://github.com/scheduler-tools/rt-app/blob/master/doc/tutorial.txt

            * A  ``__comm`` column: the actual rt-app trace task name
            * A  ``__pid``  column: the PID of the task
            * A ``__cpu``  column: the CPU on which the task was running at event
                                    generation time
            * A ``__line`` column: the ftrace line numer
            * A ``type`` column: the type of the generated event
            * A ``desc`` column: the mnemonic type of the generated event
            * A ``id`` column: the ID of the resource associated to the event,
                                e.g. the ID of the fired timer

        The ``index`` represents the timestamp of the event.
        """
        df = self._get_stats()
        return self._task_filtered(df, task)

    @memoized
    @df_rtapp_loop.used_events
    def df_phases(self, task):
        """
        Get phases actual start times and durations

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :returns: A :class:`pandas.DataFrame` with index representing the
            start time of a phase and these column:

                * ``duration``: the measured phase duration.
        """
        # Mark for removal all the events that are not the first 'start'
        def keep_first_start(raw):
            if raw.phase_loop:
                return -1
            if raw.event == 'end':
                return -1
            return 0

        loops_df = self.df_rtapp_loop(task)

        # Keep only the first 'start' and the last 'end' event
        # Do that by first setting -1 the 'phase_loop' of all entries which are
        # not the first 'start' event. Then drop the 'event' column so that we
        # can drop all duplicates thus keeping only the last 'end' even for
        # each phase.
        phases_df = loops_df[['event', 'phase', 'phase_loop']].copy()
        phases_df['phase_loop'] = phases_df.apply(keep_first_start, axis=1)
        phases_df = phases_df[['phase', 'phase_loop']]
        phases_df.drop_duplicates(keep='last', inplace=True)

        # Compute deltas and keep only [start..end] intervals, by dropping
        # instead the [end..start] internals
        durations = phases_df.index[1:] - phases_df.index[:-1]
        durations = durations[::2]

        # Drop all 'end' events thus keeping only the first 'start' event
        phases_df = phases_df[::2][['phase']]

        # Append the duration column
        phases_df['duration'] = durations

        return phases_df[['duration']]

    @df_phases.used_events
    def task_phase_windows(self, task):
        """
        Yield the phases of the specified task.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        Yield :class: `namedtuple` reporting:

            * `id` : the iteration ID
            * `start` : the iteration start time
            * `end` : the iteration end time

        :return: Generator yielding :class:`PhaseWindow` with
            start end end timestamps.
        """
        for idx, phase in enumerate(self.df_phases(task).itertuples()):
            yield PhaseWindow(idx, phase.Index,
                                    phase.Index+phase.duration)

###############################################################################
# Plotting Methods
###############################################################################

    @AnalysisHelpers.plot_method()
    @df_phases.used_events
    def plot_phases(self, task, axis, local_fig):
        """
        Draw the task's phases colored bands

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID
        """
        phases_df = self.df_phases(task)

        # Compute phases intervals
        bands = [(t, t + phases_df['duration'][t]) for t in phases_df.index]
        for idx, (start, end) in enumerate(bands):
            color = self.get_next_color(axis)
            label = 'Phase_{:02d}'.format(idx)
            axis.axvspan(start, end, alpha=0.1, facecolor=color, label=label)
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2,), ncol=8)

        if local_fig:
            task = self.trace.get_task_id(task)
            axis.set_title("Task [{}] phases".format(task))

    @AnalysisHelpers.plot_method()
    @df_rtapp_stats.used_events
    def plot_perf(self, task, axis, local_fig):
        r"""
        Plot the performance index.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        The perf index is defined as:

        .. math::

            perf_index = \frac{slack}{c_period - c_run}

        where

            - ``c_period``: is the configured period for an activation
            - ``c_run``: is the configured run time for an activation, assuming to
                        run at the maximum frequency and on the maximum capacity
                        CPU.
            - ``slack``: is the measured slack for an activation

        The slack is defined as the different among the activation deadline
        and the actual completion time of the activation.

        The deadline defines also the start of the next activation, thus in
        normal conditions a task activation is always required to complete
        before its deadline.

        The slack is thus a positive value if a task complete before its
        deadline. It's zero when a task complete an activation right at its
        eadline. It's negative when the completion is over the deadline.

        Thus, a performance index in [0..1] range represents activations
        completed within their deadlines. While, the more the performance index
        is negative the more the task is late with respect to its deadline.
        """
        task = self.trace.get_task_id(task)
        axis.set_title('Task [{}] Performance Index'.format(task))
        data = self.df_rtapp_stats(task)[['perf_index',]]
        data.plot(ax=axis, drawstyle='steps-post')
        axis.set_ylim(0, 2)


    @AnalysisHelpers.plot_method()
    @df_rtapp_stats.used_events
    def plot_latency(self, task, axis, local_fig):
        """
        Plot the Latency/Slack and Performance data for the specified task.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        .. seealso:: :meth:`plot_perf` for metrics definition.
        """
        task = self.trace.get_task_id(task)
        axis.set_title('Task [{}] (start) Latency and (completion) Slack'
                       .format(task))
        data = self.df_rtapp_stats(task)[['slack', 'wu_lat']]
        data.plot(ax=axis, drawstyle='steps-post')

    @AnalysisHelpers.plot_method()
    @df_rtapp_stats.used_events
    def plot_slack_histogram(self, task, axis, local_fig, bins=30):
        """
        Plot the slack histogram.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :param bins: number of bins for the histogram.
        :type bins: int

        .. seealso:: :meth:`plot_perf` for the slack definition.
        """
        task = self.trace.get_task_id(task)
        ylabel = 'slack of "{}"'.format(task)
        series = self.df_rtapp_stats(task)['slack']
        series.hist(bins=bins, ax=axis, alpha=0.4, label=ylabel, figure=axis.get_figure())
        axis.axvline(series.mean(), linestyle='--', linewidth=2, label='mean')
        axis.legend()

        if local_fig:
            axis.set_title(ylabel)

    @AnalysisHelpers.plot_method()
    @df_rtapp_stats.used_events
    def plot_perf_index_histogram(self, task, axis, local_fig, bins=30):
        """
        Plot the perf index histogram.

        :param task: the rt-app task to filter for
        :type task: int or str or lisa.trace.TaskID

        :param bins: number of bins for the histogram.
        :type bins: int

        .. seealso:: :meth:`plot_perf` for the perf index definition.

        """
        task = self.trace.get_task_id(task)
        ylabel = 'perf index of "{}"'.format(task)
        series = self.df_rtapp_stats(task)['perf_index']
        mean = series.mean()
        self.logger.info('perf index of task "{}": avg={:.2f} std={:.2f}'
                               .format(task, mean, series.std()))

        series.hist(bins=bins, ax=axis, alpha=0.4, label=ylabel, figure=axis.get_figure())
        axis.axvline(mean, linestyle='--', linewidth=2, label='mean')
        axis.legend()

        if local_fig:
            axis.set_title(ylabel)

@deprecate('Log-file based analysis has been replaced by ftrace-based analysis',
    deprecated_in='2.0',
    replaced_by=RTAEventsAnalysis,
)
class PerfAnalysis(AnalysisHelpers):
    """
    Parse and analyse a set of RTApp log files

    :param task_log_map: Mapping of task names to log files
    :type task_log_map: dict

    .. note:: That is not a subclass of
        :class:`lisa.analysis.base.TraceAnalysisBase` since it does not uses traces.
    """

    name = 'rta_logs'

    RTA_LOG_PATTERN = 'rt-app-{task}.log'
    "Filename pattern matching RTApp log files"

    def __init__(self, task_log_map):
        """
        Load peformance data of an rt-app workload
        """
        logger = self.logger

        if not task_log_map:
            raise ValueError('No tasks in the task log mapping')

        for task_name, logfile in task_log_map.items():
            logger.debug('rt-app task [{}] logfile: {}'.format(
                task_name, logfile
            ))

        self.perf_data = {
            task_name: {
                'logfile': logfile,
                'df': self._parse_df(logfile),
            }
            for task_name, logfile in task_log_map.items()
        }

    @classmethod
    def from_log_files(cls, rta_logs):
        """
        Build a :class:`PerfAnalysis` from a sequence of RTApp log files

        :param rta_logs: sequence of path to log files
        :type rta_logs: list(str)
        """

        def find_task_name(logfile):
            logfile = os.path.basename(logfile)
            regex = cls.RTA_LOG_PATTERN.format(task=r'(.+)-[0-9]+')
            match = re.search(regex, logfile)
            if not match:
                raise ValueError('The logfile [{}] is not from rt-app'.format(logfile))
            return match.group(1)

        task_log_map = {
            find_task_name(logfile): logfile
            for logfile in rta_logs
        }
        return cls(task_log_map)

    @classmethod
    def from_dir(cls, log_dir):
        """
        Build a :class:`PerfAnalysis` from a folder path

        :param log_dir: Folder containing RTApp log files
        :type log_dir: str
        """
        rta_logs = glob.glob(os.path.join(
            log_dir, cls.RTA_LOG_PATTERN.format(task='*'),
        ))
        return cls.from_log_files(rta_logs)

    @classmethod
    def from_task_names(cls, task_names, log_dir):
        """
        Build a :class:`PerfAnalysis` from a list of task names

        :param task_names: List of task names to look for
        :type task_names: list(str)

        :param log_dir: Folder containing RTApp log files
        :type log_dir: str
        """
        def find_log_file(task_name, log_dir):
            log_file = os.path.join(log_dir, cls.RTA_LOG_PATTERN.format(task_name))
            if not os.path.isfile(log_file):
                raise ValueError('No rt-app logfile found for task [{}]'.format(
                    task_name
                ))
            return log_file

        task_log_map = {
            task_name: find_log_file(task_name, log_dir)
            for task_name in tasks_names
        }
        return cls(task_log_map)

    @staticmethod
    def _parse_df(logfile):
        df = pd.read_csv(logfile,
                sep='\s+',
                header=0,
                usecols=[1,2,3,4,7,8,9,10],
                names=[
                    'Cycles', 'Run' ,'Period', 'Timestamp',
                    'Slack', 'CRun', 'CPeriod', 'WKPLatency'
                ])
        # Normalize time to [s] with origin on the first event
        start_time = df['Timestamp'][0]/1e6
        df['Time'] = df['Timestamp']/1e6 - start_time
        df.set_index(['Time'], inplace=True)
        # Add performance metrics column, performance is defined as:
        #             slack
        #   perf = -------------
        #          period - run
        df['PerfIndex'] = df['Slack'] / (df['CPeriod'] - df['CRun'])

        return df

    @property
    def tasks(self):
        """
        List of tasks for which performance data have been loaded
        """
        return sorted(self.perf_data.keys())

    def get_log_file(self, task):
        """
        Return the logfile for the specified task

        :param task: Name of the task that we want the logfile of.
        :type task: str
        """
        return self.perf_data[task]['logfile']

    def get_df(self, task):
        """
        Return the pandas dataframe with the performance data for the
        specified task

        :param task: Name of the task that we want the performance dataframe of.
        :type task: str
        """
        return self.perf_data[task]['df']

    def get_default_plot_path(self, **kwargs):
        # If all logfiles are located in the same folder, use that folder
        # and the default_filename
        dirnames = {
            os.path.realpath(os.path.dirname(perf_data['logfile']))
            for perf_data in self.perf_data.values()
        }
        if len(dirnames) != 1:
            raise ValueError('A default folder cannot be inferred from logfiles location unambiguously: {}'.format(dirnames))

        default_dir = dirnames.pop()

        return super().get_default_plot_path(
            default_dir=default_dir,
            **kwargs,
        )

    @AnalysisHelpers.plot_method()
    def plot_perf(self, task, axis, local_fig):
        """
        Plot the performance Index
        """
        axis.set_title('Task [{}] Performance Index'.format(task))
        data = self.get_df(task)[['PerfIndex',]]
        data.plot(ax=axis, drawstyle='steps-post')
        axis.set_ylim(0, 2)


    @AnalysisHelpers.plot_method()
    def plot_latency(self, task, axis, local_fig):
        """
        Plot the Latency/Slack and Performance data for the specified task.
        """
        axis.set_title('Task [{}] (start) Latency and (completion) Slack'\
                .format(task))
        data = self.get_df(task)[['Slack', 'WKPLatency']]
        data.plot(ax=axis, drawstyle='steps-post')

    @AnalysisHelpers.plot_method()
    def plot_slack_histogram(self, task, axis, local_fig, bins=30):
        """
        Plot the slack histogram.

        :param task: rt-app task name to plot
        :type task: str

        :param bins: number of bins for the histogram.
        :type bins: int

        .. seealso:: :meth:`plot_perf_index_histogram`
        """
        ylabel = 'slack of "{}"'.format(task)
        series = self.get_df(task)['Slack']
        series.hist(bins=bins, ax=axis, alpha=0.4, label=ylabel)
        axis.axvline(series.mean(), linestyle='--', linewidth=2, label='mean')
        axis.legend()

        if local_fig:
            axis.set_title(ylabel)

    @AnalysisHelpers.plot_method()
    def plot_perf_index_histogram(self, task, axis, local_fig, bins=30):
        r"""
        Plot the perf index histogram.

        :param task: rt-app task name to plot
        :type task: str

        :param bins: number of bins for the histogram.
        :type bins: int

        The perf index is defined as:

        .. math::

            perfIndex = \frac{slack}{period - runtime}

        """
        ylabel = 'perf index of "{}"'.format(task)
        series = self.get_df(task)['PerfIndex']
        mean = series.mean()
        self.logger.info('perf index of task "{}": avg={:.2f} std={:.2f}'.format(
            task, mean, series.std()))

        series.hist(bins=bins, ax=axis, alpha=0.4, label=ylabel)
        axis.axvline(mean, linestyle='--', linewidth=2, label='mean')
        axis.legend()

        if local_fig:
            axis.set_title(ylabel)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
