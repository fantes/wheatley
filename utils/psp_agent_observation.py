#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import torch
import numpy as np
from models.tokengt.utils import get_laplacian_pe_simple
from .utils import (
    compute_conflicts_cliques,
    put_back_one_hot_encoding_unbatched,
    compute_resources_graph,
)
import dgl
import time


class PSPAgentObservation:
    def __init__(self, graphs, glist=False):
        self.graphs = graphs
        self.glist = glist

    def get_batch_size(self):
        if self.glist:
            return len(self.graphs)
        return self.graphs.batch_size

    def get_n_nodes(self):
        if self.glist:
            return int(sum([g.num_nodes() for g in self.graphs]) / len(self.graphs))
        else:
            return int(self.graphs.num_nodes() / self.graphs.batch_size)

    @classmethod
    def build_graph(cls, n_edges, edges, nnodes, feats, bidir):
        edges0 = edges[0]
        edges1 = edges[1]
        type0 = [1] * n_edges
        if bidir:
            type1 = [2] * n_edges

            gnew = dgl.graph(
                (torch.cat([edges0, edges1]), torch.cat([edges1, edges0])),
                num_nodes=nnodes,
            )
        else:
            gnew = dgl.graph((edges0, edges1), num_nodes=nnodes)

        gnew.ndata["feat"] = feats
        if bidir:
            type0.extend(type1)
        gnew.edata["type"] = torch.LongTensor(type0)
        return gnew

    @classmethod
    def get_machine_id(cls, machine_one_hot):
        # print("machine_one_hot", machine_one_hot)

        return torch.max(machine_one_hot, dim=1)

    @classmethod
    def add_conflicts_cliques2(cls, g, cedges, mid):
        g.add_edges(cedges[0], cedges[1], data={"type": mid[0] + 5})
        return g

    @classmethod
    def np_to_torch(cls, obs):
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs)
        elif isinstance(obs, dict):
            has_batch_dim = obs["features"].ndim == 3
            max_n_nodes = np.max(obs["n_nodes"])
            max_n_jobs = np.max(obs["n_jobs"])
            max_pr_edges = np.max(obs["n_pr_edges"])
            max_rp_edges = np.max(obs["n_rp_edges"])
            if "rc_edges" in obs:
                max_rc_edges = np.max(obs["n_rc_edges"])
            newobs = {}
            if has_batch_dim:
                for key, _obs in obs.items():
                    if key == "features":
                        newobs[key] = torch.tensor(_obs[:, :max_n_nodes, :])
                    elif key == "pr_edges":
                        newobs[key] = torch.tensor(_obs[:, :, :max_pr_edges])
                    elif key == "rp_edges":
                        newobs[key] = torch.tensor(_obs[:, :, :max_rp_edges])
                    elif key == "rc_edges":
                        newobs[key] = torch.tensor(_obs[:, :, :max_rc_edges])
                    elif key == "rc_att":
                        newobs[key] = torch.tensor(_obs[:, :max_rc_edges, :])
                    elif key == "rp_att":
                        newobs[key] = torch.tensor(_obs[:, :max_rp_edges, :])
                    else:
                        newobs[key] = torch.tensor(_obs)
            else:
                for key, _obs in obs.items():
                    if key == "features":
                        newobs[key] = torch.tensor(_obs[:max_n_nodes, :]).unsqueeze(0)
                    elif key == "pr_edges":
                        newobs[key] = torch.tensor(_obs[:, :max_pr_edges]).unsqueeze(0)
                    elif key == "rp_edges":
                        newobs[key] = torch.tensor(_obs[:, :max_rp_edges]).unsqueeze(0)
                    elif key == "rc_edges":
                        newobs[key] = torch.tensor(_obs[:, :max_rc_edges]).unsqueeze(0)
                    elif key == "rc_att":
                        newobs[key] = torch.tensor(_obs[:max_rc_edges, :]).unsqueeze(0)
                    elif key == "rp_att":
                        newobs[key] = torch.tensor(_obs[:max_rp_edges, :]).unsqueeze(0)
                    else:
                        newobs[key] = torch.tensor(_obs)

            return newobs
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @classmethod
    def from_gym_observation(
        cls,
        gym_observation,
        conflicts="att",
        add_self_loops=True,
        device=None,
        do_batch=True,
        compute_laplacian_pe=False,
        laplacian_pe_cache=None,
        n_laplacian_eigv=50,
        bidir=True,
    ):

        # batching on CPU for performance reasons...
        n_nodes = gym_observation["n_nodes"].long().to(torch.device("cpu"))

        n_pr_edges = gym_observation["n_pr_edges"].long().to(torch.device("cpu"))

        pr_edge_index = gym_observation["pr_edges"].long().to(torch.device("cpu"))

        orig_feat = gym_observation["features"]
        # .to(torch.device("cpu"))

        if conflicts != "clique":
            # orig_feat = put_back_one_hot_encoding_unbatched(orig_feat, max_n_machines)
            orig_feat = orig_feat.to(torch.device("cpu"))
        else:
            if "rc_edges" in gym_observation:  # precomputed cliques
                n_rc_edges = (
                    gym_observation["n_rc_edges"].long().to(torch.device("cpu"))
                )

                rc_edges = gym_observation["rc_edges"].long().to(torch.device("cpu"))
                rc_att = gym_observation["rc_att"].to(torch.device("cpu"))
                orig_feat = orig_feat.to(torch.device("cpu"))
            else:  # compute cliques
                rc_edges = []
                rc_att = []
                all_nce = []

                for i in range(orig_feat.shape[0]):
                    one_rc_edges, rid, rval = compute_resources_graph(
                        orig_feat, n_nodes
                    )
                    all_rc_edges.append(one_rc_edges)
                    att = np.concatenate([rid, rval], axis=-1)
                    all_att.append(att)
                    nce = torch.LongTensor([rc_edges.shape[1]])
                    all_nce.append(nce)

                n_conflict_edges = torch.cat(all_nce)
                orig_feat = orig_feat.to(torch.device("cpu"))

        graphs = []
        if do_batch:
            batch_num_nodes = []
            batch_num_edges = []

        for i, nnodes in enumerate(n_nodes):
            features = orig_feat[i, :nnodes, :]
            gnew = cls.build_graph(
                n_edges[i],
                edge_index[i, :, : n_edges[i].item()],
                nnodes.item(),
                orig_feat[i, : nnodes.item(), :],
                bidir,
            )

            if conflicts == "clique":
                gnew = AgentObservation.add_conflicts_cliques2(
                    gnew,
                    conflicts_edges[i][:, : n_conflict_edges[i].item()],
                    conflicts_edges_machineid[i][:, : n_conflict_edges[i].item()],
                )
                # gnew = AgentObservation.add_conflicts_cliques(gnew, features, nnodes.item(), max_n_machines)

            if add_self_loops:
                gnew = dgl.add_self_loop(gnew, edge_feat_names=["type"], fill_data=0)
            if compute_laplacian_pe:
                gnew.ndata["laplacian_pe"] = get_laplacian_pe_simple(
                    gnew, laplacian_pe_cache, n_laplacian_eigv
                )
            gnew = gnew.to(device)
            graphs.append(gnew)
            if do_batch:
                batch_num_nodes.append(gnew.num_nodes())
                batch_num_edges.append(gnew.num_edges())

        if do_batch:
            graph = dgl.batch(graphs)
            graph.set_batch_num_nodes(torch.tensor(batch_num_nodes))
            graph.set_batch_num_edges(torch.tensor(batch_num_edges))

            return cls(graph, glist=False)
        else:
            return cls(graphs, glist=True)

    def to_graph(self):
        """
        Returns the batched graph associated with the observation.
        """
        return self.graphs
