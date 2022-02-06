import pytest
import random as random

from learning.learning import do_learning


def test_qlearning(pytestconfig):
    """Test qlearning"""

    configfile = pytestconfig.rootpath / ".." / "data" / "qlearning.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    search_statistics = do_learning(
        configfile, do_azure, do_logging, random_seed, do_profile, pickle_file, False, None
    )

    assert search_statistics["episode"] == 1000
    assert search_statistics["avg_reward"] < 0.0
    assert search_statistics["states_visited"] > 1000


def test_qlearning_traces(pytestconfig):
    """Test qlearning_traces"""

    configfile = pytestconfig.rootpath / ".." / "data" / "qlearning_traces.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    search_statistics = do_learning(
        configfile, do_azure, do_logging, random_seed, do_profile, pickle_file, False, None
    )

    assert search_statistics["episode"] == 1000
    assert search_statistics["avg_reward"] < 0.0
    assert search_statistics["states_visited"] > 1000
