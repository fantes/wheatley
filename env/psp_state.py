import numpy as np

# import matplotlib.pyplot as plt
import networkx as nx
import time
from utils.utils import compute_resources_graph


class PSPState:
    # TODO
    def __init__(
        self,
        env_specification,
        problem_description,
        problem,
        deterministic=True,
        observe_conflicts_as_cliques=True,
    ):
        self.problem = problem
        self.problem_description = problem_description
        self.n_features = env_specification.n_features
        self.n_nodes = self.problem["n_modes"]
        self.features = np.zeros((self.n_nodes, self.n_features), dtype=float)
        self.deterministic = deterministic
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques

        # features :
        # 0: is_affected
        # 1: is selectable
        # 2: job_id (in case of several mode per job)
        # 3,4,5: durations (min max mode)
        # 6,7,8 : tct (min max mode)
        # 9.. 8+max_n_resources : level of resource i used by this mode (normalized)

        job_info = problem["job_info"]
        self.job_modes = []
        m = 0
        for n, j in enumerate(job_info):
            self.job_modes.append(list(range(m, m + j[0])))
            for mi in range(m, m + j[0]):
                self.features[mi, 2] = n
            m += j[0]
        self.problem_edges = []
        for n, j in enumerate(job_info):
            for succ_job in j[1]:
                for orig_mode in self.job_modes[n]:
                    for dest_mode in self.job_modes[succ_job - 1]:
                        self.problem_edges.append((orig_mode, dest_mode))
        self.problem_graph = nx.DiGraph(self.problem_edges)
        # nx.draw_networkx(self.graph)
        # plt.show()
        self.numpy_problem_graph = np.array(self.problem_edges)
        self.priority_edges = []

        self.resources_users = [[] * self.problem["n_resources"]]
        self.resources_used = [[] * self.problem["n_resources"]]

        self.reset_durations()
        self.reset_task_completion_times()
        self.reset_is_affected()
        self.reset_resources()
        self.reset_selectable()

        if self.observe_conflicts_as_cliques:
            (
                self.resource_edges,
                self.resources_id,
                self.resources_val,
            ) = compute_resources_graph(self.features[:, 9:])

        self.add_resource_priority(1)
        exit()

    def reset_durations(self, redraw_real=True):
        # at init, draw real durations from distrib if not determinisitc
        flat_dur = [item for sublist in self.problem["durations"] for item in sublist]
        self.features[:, 3:6] = np.expand_dims(np.array(flat_dur), 1)
        if redraw_real:
            self.real_durations = self.draw_real_durations(self.features[:, 3:6])

    def reset_is_affected(self):
        self.features[:, 0] = 0

    def draw_real_durations(self, durs):
        if self.deterministic:
            return durs[:, 0]
        else:
            raise RuntimeError("stochastic durs not yet implemented")

    def reset(self):
        self.edges = []
        self.graph = nx.DiGraph(self.problem_edges)
        self.reset_task_completion_times()

    def reset_resources(self):
        flat_res = np.array(
            [item for sublist in self.problem["resources"] for item in sublist],
            dtype=float,
        )
        for i in range(self.problem["n_resources"]):
            flat_res[:, i] /= self.problem["resource_availability"][i]
        self.features[:, 9 : 9 + self.problem["n_resources"]] = flat_res

    def get_task_completion_times(self, nodeid):
        return self.features[nodeid, 6:9]

    def set_task_completion_times(self, nodeid, ct):
        self.features[nodeid, 6:9] = ct

    def get_durations(self, nodeid):
        return self.features[nodeid, 3:6]

    def reset_selectable(self):
        no_parents = np.where(np.array(self.problem_graph.in_degree())[:, 1] == 0)[0]
        self.features[no_parents, 1] = 1

    def reset_task_completion_times(self):
        open_nodes = np.where(np.array(self.problem_graph.in_degree())[:, 1] == 0)[
            0
        ].tolist()
        while open_nodes:
            cur_node_id = open_nodes.pop(0)

            if self.problem_graph.in_degree(cur_node_id) == 0:
                max_tct_predecessors = np.zeros((3))
            else:
                task_comp_time_pred = np.stack(
                    [
                        self.get_task_completion_times(p)
                        for p in self.problem_graph.predecessors(cur_node_id)
                    ]
                )
                max_tct_predecessors = np.max(task_comp_time_pred, 0)[0]

            new_completion_time = max_tct_predecessors + self.get_durations(cur_node_id)
            self.set_task_completion_times(cur_node_id, new_completion_time)
            for successor in self.problem_graph.successors(cur_node_id):
                to_open = True
                for p in self.problem_graph.predecessors(successor):
                    if p in open_nodes:
                        to_open = False
                        break
                if to_open:
                    open_nodes.append(successor)

    def to_features_and_edge_index(self, normalize, input_list):
        if not self.edges:
            edge_index = np.transpose(self.numpy_edges)
        else:
            edge_index = np.transpose(
                np.concatenate([self.numpy_edges, np.array(self.edges)])
            )

        return self.features, edge_index

    def get_features_wo_real_dur(self):
        return self.features

    def get_selectable(self):
        return self.features[:, 1]

    def affect_node(self, nodeid):
        # should be selectable
        assert self.features[nodeid, 1] == 1
        # all modes become unselectable
        mm = self.job_modes[self.features[nodeid, 2].astype(int)]
        self.features[mm, 1] = 0
        # mark as affected
        self.features[nodeid, 0] = 1
        # make sucessor selectable, if other parents *jobs* are affected
        for successor in self.graph.successors(nodeid):
            parents_jobs = set(
                [
                    self.features[pm, 2].astype(int)
                    for pm in self.graph.predecessors(successor)
                ]
            )
            # no need to test job from currently affected node
            parents_jobs.remove(self.features[nodeid, 2])
            # check if one mode per job is affected
            all_parent_jobs_affected = True
            for pj in parents_jobs:
                pjm = self.job_modes[pj]
                job_is_affected = False
                for pjmm in pjm:
                    if self.features[pjmm, 0]:
                        job_is_affected = True
                        break
                if not job_is_affected:
                    all_parent_jobs_affected = False
                    break
            if all_parent_jobs_affected:
                # make selectable, at last
                self.features[successor, 1] = 1

    def add_resource_priority(self, node_id):
        ru = np.where(self.features[node_id, 9:] != 0)[0]
        for r in ru:
            self.resources_users[r].append(node_id)
            self.resources_used[r].append(self.features[node_id, 9 + r])

    def done(self):
        return np.sum(self.features[:, 0]) == self.problem["n_jobs"]

    def tct():
        return self.features[:, 6:9]

    def render_solution(self, schedule, scaling):
        pass

    def get_solution(self):
        pass
