from __future__ import annotations

import pytest

from pylab.core import timeseries
from pylab.core import loader
from pylab.core import testing
from pylab.simulink import simulink

from pylab.simulink import _engine

_engine.import_matlab_engine("R2021b")


def test_experiment():
    info = loader.load_test("test.yml")
    details = simulink.load_details("matlab_detail.yml")
    experiment = simulink.create(info, details)
    report = experiment.execute()
    result = report.results["PT2Regler.Alpha"]
    s = result.pretty_string()
    print(s)
