#! /usr/bin/env python3

from utils.env import TestEnv
from utils.test_workload import OneSmallTask
from utils.conf import LisaLogging

LisaLogging.setup()


target_conf = {
        "platform" : "linux",
        "board" : "hikey620",
        "host": "pwrsft-hikey620-1",
        "modules" :
        ["sched", "cgroups", "hotplug"],"tools" : ["taskset", "rt-app"],
        "rtapp-calib" : {"0": 509, "1": 465, "2": 466, "3": 466, "4": 465, "5": 466, "6": 466, "7": 466}
}


te = TestEnv(target_conf)
wload = OneSmallTask(te)
wload.run()
bundle = wload.collect_test()
res = bundle.test_slack()

print(dir(res))
print(res)
print(res.metrics)
print(res.passed)

