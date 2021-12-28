import pytest
import random as random

from learning.learning import do_learning


def test_qlearning_traces(pytestconfig):
    """Test qlearning_traces"""

    configfile = pytestconfig.rootpath / ".." / "data" / "qlearning_traces.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    do_learning(configfile, do_azure, do_logging, random_seed, do_profile, pickle_file)
