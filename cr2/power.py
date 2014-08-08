#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current
directory's trace.dat"""

from matplotlib import pyplot as plt
import pandas as pd

from thermal import BaseThermal
from plot_utils import normalize_title, pre_plot_setup, post_plot_setup

def pivot_with_labels(dfr, data_col_name, new_col_name, mapping_label):
    """Pivot a DataFrame row into columns

    dfr is the DataFrame to operate on.  data_col_name is the name of
    the column in the DataFrame which contains the values.
    new_col_name is the name of the column in the DataFrame that will
    became the new columns.  mapping_label is a dictionary whose keys
    are the values in new_col_name and whose values are their
    corresponding name in the DataFrame to be returned.

    There has to be a more "pandas" way of doing this.

    Example: XXX

    In [8]: dfr_in = pd.DataFrame({'cpus': ["000000f0", "0000000f", "000000f0", "0000000f"], 'freq': [1, 3, 2, 6]})

    In [9]: dfr_in
    Out[9]:
           cpus  freq
    0  000000f0     1
    1  0000000f     3
    2  000000f0     2
    3  0000000f     6

    [4 rows x 2 columns]

    In [10]: map_label = {"000000f0": "A15", "0000000f": "A7"}

    In [11]: power.pivot_with_labels(dfr_in, "freq", "cpus", map_label)
    Out[11]:
       A15  A7
    0    1 NaN
    1    1   3
    2    2   3
    3    2   6

    [4 rows x 2 columns]
    """

    col_set = set(dfr[new_col_name])

    ret_series = {}
    for col in col_set:
        label = mapping_label[col]
        data = dfr[dfr[new_col_name] == col][data_col_name]

        ret_series[label] = data

    return pd.DataFrame(ret_series).fillna(method="pad")

def get_all_freqs_data(in_power, out_power, map_label):
    """get a dict of DataFrames suitable for the allfreqs plot"""

    in_freqs = in_power.get_all_freqs(map_label)
    out_freqs = out_power.get_all_freqs(map_label)

    ret_dict = {}
    for label in map_label.values():
        in_label = label + "_freq_in"
        out_label = label + "_freq_out"

        inout_freq_dict = {in_label: in_freqs[label], out_label: out_freqs[label]}
        ret_dict[label] = pd.DataFrame(inout_freq_dict).fillna(method="pad")

    return ret_dict

class OutPower(BaseThermal):
    """Process the cpufreq cooling power actor data in a ftrace dump"""

    def __init__(self, path=None):
        super(OutPower, self).__init__(
            basepath=path,
            unique_word="thermal_power_actor_cpu_limit",
        )

    def get_all_freqs(self, mapping_label):
        """get a DataFrame with the maximum frequencies allowed by the governor

        mapping_label must be a dictionary that maps cpumasks to name
        of the cpu.  Returned freqs are in KHz
        """

        dfr = self.data_frame

        return pivot_with_labels(dfr, "freq", "cpus", mapping_label) / 1000

class InPower(BaseThermal):
    """Process the cpufreq cooling power actor data in a ftrace dump"""

    def __init__(self, path=None):
        super(InPower, self).__init__(
            basepath=path,
            unique_word="thermal_power_actor_cpu_get_dyn",
        )

    def get_load_data(self, mapping_label):
        """return a dataframe suitable for plot_load()

        mapping_label is a dictionary mapping cluster numbers to labels."""

        dfr = self.data_frame
        load_cols = [s for s in dfr.columns if s.startswith("load")]

        load_series = dfr[load_cols[0]]
        for col in load_cols[1:]:
            load_series += dfr[col]

        load_dfr = pd.DataFrame({"cpus": dfr["cpus"], "load": load_series})
        cluster_numbers = set(dfr["cpus"])

        return pivot_with_labels(load_dfr, "load", "cpus", mapping_label)

    def get_all_freqs(self, mapping_label):
        """get a DataFrame with the "in" frequencies as seen by the governor

        Frequencies are in KHz
        """

        dfr = self.data_frame

        return pivot_with_labels(dfr, "freq", "cpus", mapping_label) / 1000

    def plot_load(self, mapping_label, title="", width=None, height=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """

        load_data = self.get_load_data(mapping_label)
        title = normalize_title("Utilisation", title)

        ax = pre_plot_setup(width=width, height=height)
        load_data.plot(ax=ax)
        post_plot_setup(ax, title=title)
