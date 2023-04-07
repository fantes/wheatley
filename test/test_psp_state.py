import sys

sys.path.append(".")

import pytest
import numpy as np
from env.psp_state import PSPState
from problem.psp_description import PSPDescription

from args import args, exp_name, path
from env.psp_env_specification import PSPEnvSpecification
from utils.loaders import PSPLoader


@pytest.fixture
def small_pb():
    loader = PSPLoader()
    return loader.load_single("instances/psp/small/small.sm")


@pytest.fixture
def problem_description_small(small_pb):
    return PSPDescription(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        train_psps=[small_pb],
        test_psps=[small_pb],
    )


@pytest.fixture
def env_specification_small(problem_description_small):
    if args.conflicts == "clique" and args.precompute_cliques:
        observe_clique = True
    else:
        observe_clique = False
    return PSPEnvSpecification(
        problems=problem_description_small,
        normalize_input=True,
        input_list=args.features,
        max_edges_factor=args.max_edges_upper_bound_factor,
        sample_n_jobs=args.sample_n_jobs,
        chunk_n_jobs=args.chunk_n_jobs,
        observe_conflicts_as_cliques=observe_clique,
        observe_real_duration_when_affect=False,
        do_not_observe_updated_bounds=args.do_not_observe_updated_bounds,
    )


@pytest.fixture
def state_small(problem_description_small, env_specification_small):
    return PSPState(
        env_specification_small,
        problem_description_small,
        problem_description_small.train_psps[0],
        deterministic=True,
        observe_conflicts_as_cliques=False,
    )


def test_state(state_small, capsys):
    s = state_small
    # no node is affected
    assert np.all(s.features[:, 0] == 0)
    # only node 0 is selectable
    assert s.features[0, 1] == 1
    assert np.all(s.features[1:, 1] == 0)

    s.compute_dates_on_affectation(0)
    s.affect_node(0)
    s.update_completion_times(0)
    # only node 0 is affected
    assert s.features[0, 0] == 1
    assert np.all(s.features[1:, 0] == 0)
    # node 1 and 2 are the only selectable
    assert s.features[0, 1] == 0
    assert s.features[1, 1] == 1
    assert s.features[2, 1] == 1
    assert np.all(s.features[3:, 1] == 0)
    with capsys.disabled():
        print("tl1", s.resource_timelines[0][0].timepoints)
        print("tl2", s.resource_timelines[1][0].timepoints)
        print("tct", s.get_task_completion_times(0))
        print("tct_r", s.get_task_completion_time_real(0))

    s.compute_dates_on_affectation(1)
    s.affect_node(1)
    s.update_completion_times(1)
    assert np.all(s.features[:2, 0] == 1)
    assert np.all(s.features[2:, 0] == 0)
    assert np.all(s.features[:2, 1] == 0)
    assert np.all(s.features[2, 1] == 1)
    assert np.all(s.features[3:, 1] == 0)
    with capsys.disabled():
        print("tl1", s.resource_timelines[0][0].timepoints)
        print("tl2", s.resource_timelines[1][0].timepoints)
        print("tct", s.get_task_completion_times(1))
        print("tct_r", s.get_task_completion_time_real(1))

    s.compute_dates_on_affectation(2)
    s.affect_node(2)
    s.update_completion_times(2)
    assert np.all(s.features[:3, 0] == 1)
    assert np.all(s.features[3:, 0] == 0)
    assert np.all(s.features[:3, 1] == 0)
    assert np.all(s.features[3:5, 1] == 1)
    assert np.all(s.features[5:, 1] == 0)
    with capsys.disabled():
        print("tl1", s.resource_timelines[0][0].timepoints)
        print("tl2", s.resource_timelines[1][0].timepoints)
        print("tct", s.get_task_completion_times(2))
        print("tct_r", s.get_task_completion_time_real(2))

    s.compute_dates_on_affectation(3)
    s.affect_node(3)
    s.update_completion_times(3)
    assert s.get_task_completion_time_real(3) == 10.0

    s.compute_dates_on_affectation(4)
    s.affect_node(4)
    s.update_completion_times(4)
    print(s.get_task_completion_time_real(4) == 6.0)

    s.compute_dates_on_affectation(5)
    s.affect_node(5)
    s.update_completion_times(5)
    print(s.get_task_completion_time_real(5) == 11.0)

    s.compute_dates_on_affectation(6)
    s.affect_node(6)
    s.update_completion_times(6)
    print(s.get_task_completion_time_real(6) == 15.0)

    s.compute_dates_on_affectation(7)
    s.affect_node(7)
    s.update_completion_times(7)
    print(s.get_task_completion_time_real(6) == 15.0)
