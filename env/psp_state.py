import numpy as np

# import matplotlib.pyplot as plt
import networkx as nx
import time
from utils.utils import compute_resources_graph
from utils.resource_timeline import ResourceTimeline


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
        self.affected_nodes = set()

        print('self.problem["n_resources"]', self.problem["n_resources"])
        self.resource_timelines = []
        print("resource_timelines", self.resource_timelines)
        for r in range(self.problem["n_renewable_resources"]):
            self.resource_timelines.append([])
            for i in range(4):
                self.resource_timelines[r].append(
                    ResourceTimeline(
                        max_level=1.0,
                        renewable=True,
                    )
                )
        for r in range(
            self.problem["n_renewable_resources"],
            self.problem["n_renewable_resources"]
            + self.problem["n_nonrenewable_resources"],
        ):
            self.resource_timelines.append([])
            for i in range(4):
                self.resource_timelines[r].append(
                    ResourceTimeline(
                        max_level=1.0,
                        renewable=False,
                    )
                )

        self.real_durations = None
        self.real_tct = np.zeros((self.n_nodes), dtype=float)

        self.reset_durations()
        self.reset_task_completion_times()
        self.reset_is_affected()
        self.reset_resources()
        self.reset_selectable()

        if self.observe_conflicts_as_cliques:
            (
                self.static_resource_edges,
                self.static_resources_id,
                self.static_resources_val,
            ) = compute_resources_graph(self.features[:, 9:])
        else:
            self.static_resource_edges = None
            self.static_resources_id = None
            self.static_resources_val = None

        self.resources_edges = []
        self.resources_edges_att = []
        # self.resources_edges.append((prec, succ))
        # self.resources_edges_att.append((on_start, critical))

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

    def get_task_completion_time_real(self, nodeid):
        print("nodeid", nodeid)
        print("self.real_tct", self.real_tct)
        return self.real_tct[nodeid]

    def set_task_completion_times(self, nodeid, ct):
        self.features[nodeid, 6:9] = ct

    def set_task_completion_time_real(self, nodeid, ct):
        self.real_tct[nodeid] = ct

    def get_durations(self, nodeid):
        return self.features[nodeid, 3:6]

    def get_durations_real(self, nodeid):
        return self.real_durations[nodeid]

    def reset_selectable(self):
        no_parents = np.where(np.array(self.problem_graph.in_degree())[:, 1] == 0)[0]
        self.features[no_parents, 1] = 1

    def reset_task_completion_times(self):
        self.update_completion_times(None)

    def to_features_and_edge_index(self, normalize, input_list):
        return (
            self.normalize_features(),
            self.numpy_problem_graph,
            self.static_resource_edges,
            self.static_resource_id,
            self.static_resource_val,
            np.transpose(self.resources_edges),
            self.resources_edges_att,
        )

    def normalize_features(self):
        # TODO: normalize features
        return self.features

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
        for successor in self.problem_graph.successors(nodeid):
            parents_jobs = set(
                [
                    self.features[pm, 2].astype(int)
                    for pm in self.problem_graph.predecessors(successor)
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

    def get_last_finishing_dates(self, jobs):
        if len(jobs) == 0:
            return np.zeros((3))
        tct = self.get_task_completion_times(jobs)
        return np.amax(tct, axis=0)

    def get_last_finishing_dates_real(self, jobs):
        if len(jobs) == 0:
            return np.array([0])
        tct = self.get_task_completion_time_real(jobs)
        print("tct", tct)
        print("amax", np.amax(tct, axis=0))
        return np.array([np.amax(tct, axis=0)])

    def compute_dates_on_affectation(self, node_id):
        job_parents = list(self.problem_graph.predecessors(node_id))
        print("job_parents", job_parents)
        affected_parents = np.where(self.features[job_parents, 0] == 1)[0]
        print("affected_parents", affected_parents)
        last_parent_finish_date = self.get_last_finishing_dates(affected_parents)
        print("last parent finish date", last_parent_finish_date)
        last_parent_finish_date_real = self.get_last_finishing_dates_real(
            affected_parents
        )

        resources_used = np.where(self.features[node_id, 9:] != 0)[0]
        max_r_date = np.array([0.0, 0.0, 0.0, 0.0])
        pred_on_resource = [None] * 4
        pred_on_resource_is_start = [None] * 4
        constraining_resource = [None] * 4

        for r in resources_used:
            rad = self.get_resource_available_date(r, self.features[node_id, 9 + r])
            # rad is 4-long list of (date, jobid, start_tp)
            for i in range(4):
                if rad[i][0] > max_r_date[i]:
                    max_r_date[i] = rad[i][0]
                    pred_on_resource[i] = rad[i][1]
                    pred_on_resource_is_start[i] = rad[i][2]
                    constraining_resource[i] = r
                    if i != 0:
                        self.add_resource_precedence(
                            pred_on_resource[i],
                            node_id,
                            pred_on_resource_is_start[i],
                            True,
                            i,
                        )
                else:
                    if i != 0:
                        self.add_resource_precedence(
                            pred_on_resource[i],
                            node_id,
                            pred_on_resource_is_start[i],
                            False,
                            i,
                        )

        # do a min per coord
        print("max_r_date", max_r_date)
        print("last_parent_finish_date_real", last_parent_finish_date_real)
        print("last_parent_finish_date", last_parent_finish_date)
        start = np.maximum(
            max_r_date,
            np.concatenate([last_parent_finish_date_real, last_parent_finish_date]),
        )
        print("start", start)
        self.features[node_id, 6:9] = start[1:] + self.get_durations(node_id)
        self.real_tct[node_id] = start[0] + self.get_durations_real(node_id)

        self.insert_timepoints_in_resources(
            0, node_id, start[0], self.get_task_completion_time_real(node_id)
        )
        for i in range(3):
            self.insert_timepoints_in_resources(
                i + 1, node_id, start[i + 1], self.get_task_completion_times(node_id)[i]
            )

        self.update_completion_times(node_id)

    def add_resource_precedence(self, prec, succ, on_start, critical, timetype):
        self.resources_edges.append((prec, succ))
        self.resources_edges_att.append((on_start, critical, timetype))

    def get_resource_available_date(self, rid, level):
        return [self.resource_timelines[rid][i].availability(level) for i in range(4)]

    def insert_timepoints_in_resources(self, timeindex, node_id, start, end):
        resources_used = np.where(self.features[node_id, 9:] != 0)[0]
        for r in resources_used:
            self.resource_timelines[r][timeindex].consume(
                node_id, self.features[node_id, 9 + r], start, end
            )

    def update_completion_times(self, node_id):
        if node_id is None:
            open_nodes = np.where(np.array(self.problem_graph.in_degree())[:, 1] == 0)[
                0
            ].tolist()
        else:
            open_nodes = [node_id]
        while open_nodes:
            cur_node_id = open_nodes.pop(0)

            if self.problem_graph.in_degree(cur_node_id) == 0:
                max_tct_predecessors = np.zeros((3))
                max_tct_predecessors_real = 0
            else:
                task_comp_time_pred = np.stack(
                    [
                        self.get_task_completion_times(p)
                        for p in self.problem_graph.predecessors(cur_node_id)
                    ]
                )
                max_tct_predecessors = np.max(task_comp_time_pred, 0)[0]
                max_tct_predecessors_real = max(
                    [
                        self.get_task_completion_time_real(p)
                        for p in self.problem_graph.predecessors(cur_node_id)
                    ]
                )

            new_completion_time = max_tct_predecessors + self.get_durations(cur_node_id)
            new_completion_time_real = (
                max_tct_predecessors_real + self.get_durations_real(cur_node_id)
            )

            self.set_task_completion_times(cur_node_id, new_completion_time)
            self.set_task_completion_time_real(cur_node_id, new_completion_time_real)

            for successor in self.problem_graph.successors(cur_node_id):
                to_open = True
                for p in self.problem_graph.predecessors(successor):
                    if p in open_nodes:
                        to_open = False
                        break
                if to_open:
                    open_nodes.append(successor)

    def done(self):
        return np.sum(self.features[:, 0]) == self.problem["n_jobs"]

    def tct():
        return self.features[:, 6:9]

    def render_solution(self, schedule, scaling):
        pass

    def get_solution(self):
        pass
