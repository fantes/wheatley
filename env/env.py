import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import sys
import traceback

from env.transition_models.l2d_transition_model import L2DTransitionModel
from env.transition_models.slot_locking_transition_model import SlotLockingTransitionModel
from env.reward_models.intrinsic_reward_model import IntrinsicRewardModel
from env.reward_models.l2d_reward_model import L2DRewardModel
from env.reward_models.meta_reward_model import MetaRewardModel
from env.reward_models.sparse_reward_model import SparseRewardModel
from env.reward_models.tassel_reward_model import TasselRewardModel
from env.reward_models.uncertain_reward_model import UncertainRewardModel
from utils.env_observation import EnvObservation
from utils.utils import get_n_features
from env.state import State


class Env(gym.Env):
    def __init__(
        self,
        problem_description,
        env_specification,
    ):
        self.problem_description = problem_description
        self.env_specification = env_specification

        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config
        self.n_jobs = problem_description.n_jobs
        # adjust n_jobs if we are going to sample
        if env_specification.sample_n_jobs != -1:
            self.n_jobs = env_specification.sample_n_jobs
        self.n_machines = problem_description.n_machines
        self.n_nodes = self.n_machines * self.n_jobs
        self.deterministic = problem_description.deterministic

        self.n_features = get_n_features(
            self.env_specification.input_list, self.env_specification.max_n_jobs, self.env_specification.max_n_machines
        )
        self.action_space = Discrete(self.env_specification.max_n_nodes * (2 if self.env_specification.add_boolean else 1))
        if self.env_specification.max_edges_factor > 0:
            shape = (2, self.env_specification.max_edges_factor * self.env_specification.max_n_nodes)
        else:
            shape = (2, self.env_specification.max_n_edges)
        self.observation_space = Dict(
            {
                "n_jobs": Discrete(self.env_specification.max_n_jobs + 1),
                "n_machines": Discrete(self.env_specification.max_n_machines + 1),
                "n_nodes": Discrete(self.env_specification.max_n_nodes + 1),
                "n_edges": Discrete(self.env_specification.max_n_edges + 1),
                "features": Box(
                    low=0,
                    high=1000 * self.env_specification.max_n_machines,
                    shape=(self.env_specification.max_n_nodes, self.n_features),
                ),
                "edge_index": Box(
                    low=0,
                    high=self.env_specification.max_n_nodes,
                    shape=shape,
                    dtype=np.int64,
                ),
            }
        )

        self.transition_model = None
        self.reward_model = None

        self.n_steps = 0

        self._create_reward_model()

        self.reset()

    def step(self, action):
        # Getting current observation
        obs = self.observe()

        # Running the transition model on the current action
        node_id, boolean = self._convert_action_to_node_id(action)
        if self.env_specification.insertion_mode == "no_forced_insertion":
            self.transition_model.run(self.state, node_id, force_insert=False)
        elif self.env_specification.insertion_mode == "full_forced_insertion":
            self.transition_model.run(self.state, node_id, force_insert=True)
        elif self.env_specification.insertion_mode == "choose_forced_insertion":
            self.transition_model.run(self.state, node_id, force_insert=boolean)
        elif self.env_specification.insertion_mode == "slot_locking":
            self.transition_model.run(self.state, node_id, lock_slot=boolean)

        # Getting next observation
        next_obs = self.observe()

        # Getting the reward associated with the current action
        reward = self.reward_model.evaluate(
            obs,
            action,
            next_obs,
        )

        # Getting final necessary information
        done = self.done()
        gym_observation = next_obs.to_gym_observation()
        info = {"episode": {"r": reward, "l": 1 + self.n_steps * 2}}
        self.n_steps += 1

        return gym_observation, reward, done, info

    def _convert_action_to_node_id(self, action):
        boolean = True
        if self.env_specification.add_boolean:
            boolean = True if action >= self.env_specification.max_n_nodes else False
        node_id = action % self.env_specification.max_n_nodes
        return node_id, boolean

    def reset(self, soft=False):
        # Reset the internal state, but do not sample a new problem
        if soft:
            self.state.reset()

        # Reset the state by creating a new one
        else:
            self._create_state()

        # Reset the transition model by creating a new one
        self._create_transition_model()

        # Get the new observation
        observation = self.observe()

        self.n_steps = 0

        return observation.to_gym_observation()

    def get_solution(self):
        return self.state.get_solution()

    def render_solution(self, schedule, scaling=1.0):
        return self.state.render_solution(schedule, scaling)

    def sample_jobs(self, input_affectations, input_durations):
        sample = self.env_specification.sample_n_jobs
        if sample == -1:
            return input_affectations, input_durations
        assert sample <= len(input_affectations)
        assert sample <= len(input_durations)
        ids = list(range(len(input_durations)))
        samples = np.random.choice(ids, sample, replace=False)
        affectations = np.array([ input_affectations[i] for i in samples ])
        durations = np.array([ input_durations[i] for i in samples ])
        return affectations, durations

    def _create_state(self):
        affectations, durations = self.problem_description.sample_problem()
        affectations, durations = self.sample_jobs(affectations, durations)
        self.state = State(
            affectations,
            durations,
            self.env_specification.max_n_jobs,
            self.env_specification.max_n_machines,
            self.deterministic,
            feature_list=self.env_specification.input_list,
        )

    def _create_transition_model(self):
        if self.transition_model_config == "L2D" and self.env_specification.insertion_mode != "slot_locking":
            self.transition_model = L2DTransitionModel(
                self.state.affectations,
                self.state.durations,
                self.env_specification.max_n_jobs,
                self.env_specification.max_n_machines,
            )
        elif self.transition_model_config == "L2D" and self.env_specification.insertion_mode == "slot_locking":
            self.transition_model = SlotLockingTransitionModel(
                self.state.affectations,
                self.state.durations,
                self.env_specification.max_n_jobs,
                self.env_specification.max_n_machines,
            )
        else:
            raise Exception("Transition model not recognized")

    def _create_reward_model(self):
        # For deterministic problems, there are a few rewards available
        if self.deterministic:
            if self.reward_model_config == "L2D":
                self.reward_model = L2DRewardModel()
            elif self.reward_model_config == "Sparse":
                self.reward_model = SparseRewardModel()
            elif self.reward_model_config == "Tassel":
                self.reward_model = TasselRewardModel(
                    self.affectations, self.durations, self.env_specification.normalize_input
                )
            elif self.reward_model_config == "Intrinsic":
                self.reward_model = IntrinsicRewardModel(self.n_features * self.n_nodes, self.n_nodes)
            else:
                raise Exception("Reward model not recognized")

        # If the problem_description is stochastic, only Sparse and Uncertain reward models are accepted
        else:
            if self.reward_model_config in ["realistic", "optimistic", "pessimistic", "averagistic"]:
                self.reward_model = UncertainRewardModel(self.reward_model_config)
            elif self.reward_model_config == "Sparse":
                self.reward_model = SparseRewardModel()
            else:
                raise Exception("Reward model not recognized")

    def observe(self):
        features, edge_index = self.state.to_features_and_edge_index(
            self.env_specification.normalize_input,
            self.env_specification.input_list,
        )
        return EnvObservation(
            self.n_jobs,
            self.n_machines,
            features,
            edge_index,
            self.env_specification.max_n_jobs,
            self.env_specification.max_n_machines,
            self.env_specification.max_edges_factor,
        )

    def done(self):
        return self.state.done()

    def is_uncertain(self):
        return self.state.durations.shape[2] > 1

    def action_masks(self):
        return self.transition_model.get_mask(self.state, self.env_specification.add_boolean)
