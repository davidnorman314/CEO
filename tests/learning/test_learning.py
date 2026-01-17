import random as random

from ceo.learning.learning import do_learning


def get_data_dir(pytestconfig):
    # Find the CEO/data dir whether pytest is run in CEO or CEO/src
    rootpath = pytestconfig.rootpath
    datadir = rootpath / "data"

    if datadir.is_dir():
        return datadir

    datadir = rootpath / ".." / "data"
    if datadir.is_dir():
        return datadir

    msg = "Can't find data dir from rootpath " + str(rootpath)
    assert msg == ""


def test_qlearning(pytestconfig):
    """Test qlearning"""

    configfile = get_data_dir(pytestconfig) / "qlearning.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    search_statistics = do_learning(
        configfile,
        do_azure,
        do_logging,
        random_seed,
        do_profile,
        pickle_file,
        False,
        None,
    )

    assert search_statistics["episode"] == 1000
    assert search_statistics["avg_reward"] < 0.0
    assert search_statistics["states_visited"] > 1000


def test_qlearning_traces(pytestconfig):
    """Test qlearning_traces"""

    configfile = get_data_dir(pytestconfig) / "qlearning_traces.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    search_statistics = do_learning(
        configfile,
        do_azure,
        do_logging,
        random_seed,
        do_profile,
        pickle_file,
        False,
        None,
    )

    assert search_statistics["episode"] == 1000
    assert search_statistics["avg_reward"] < 0.0
    assert search_statistics["states_visited"] > 1000


def test_qlearning_afterstates(pytestconfig):
    """Test qlearning_afterstates"""

    configfile = get_data_dir(pytestconfig) / "qlearning_afterstates.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    search_statistics = do_learning(
        configfile,
        do_azure,
        do_logging,
        random_seed,
        do_profile,
        pickle_file,
        False,
        None,
    )

    assert search_statistics["episode"] == 1000
    assert search_statistics["avg_reward"] < 0.0
    assert search_statistics["states_visited"] > 1000


def test_qlearning_afterstates_handsummary(pytestconfig):
    """Test qlearning_afterstates with the HandSummary feature"""

    configfile = get_data_dir(pytestconfig) / "qlearning_afterstates_handsummary.json"
    do_azure = False
    do_logging = False
    random_seed = 0
    do_profile = False
    pickle_file = None

    search_statistics = do_learning(
        configfile,
        do_azure,
        do_logging,
        random_seed,
        do_profile,
        pickle_file,
        False,
        None,
    )

    assert search_statistics["episode"] == 1000
    assert search_statistics["avg_reward"] != 0
    assert search_statistics["avg_reward"] < 0.1
    assert search_statistics["states_visited"] > 1000
