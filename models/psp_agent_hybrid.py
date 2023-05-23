#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

from models.agent import Agent
from models.psp_gnn_dgl import PSPGnnDGL as GnnDGL
from models.psp_gnn_tokengt import PSPGnnTokenGT as GnnTokenGT
from models.mlp import MLP
from torch.distributions.categorical import Categorical
from functools import partial
import numpy as np
from utils.psp_agent_observation import PSPAgentObservation


class PSPAgentHybrid(Agent):
    def __init__(
        self,
        env_specification,
        gnn=None,
        value_net=None,
        action_net=None,
        agent_specification=None,
    ):
        """
        There are 2 ways to init an Agent:
         - Either provide a valid env_specification and agent_specification
         - Or use the load method, to load an already saved Agent
        """
        super().__init__(
            env_specification, gnn, value_net, action_net, agent_specification
        )

        # If a model is provided, we simply load the existing model.
        if gnn is not None and value_net is not None and action_net is not None:
            self.gnn = gnn
            self.value_net = value_net
            self.action_net = action_net
            return

        self.gnn_dgl = GnnDGL(
            input_dim_features_extractor=env_specification.n_features,
            graph_pooling=agent_specification.graph_pooling,
            max_n_nodes=env_specification.max_n_nodes,
            max_n_resources=env_specification.max_n_resources,
            n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
            activation_features_extractor=agent_specification.activation_fn_graph,
            n_layers_features_extractor=agent_specification.n_layers_features_extractor,
            hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
            n_attention_heads=agent_specification.n_attention_heads,
            residual=agent_specification.residual_gnn,
            normalize=agent_specification.normalize_gnn,
            conflicts="clique",
            edge_embedding_flavor=agent_specification.edge_embedding_flavor,
            layer_pooling=agent_specification.layer_pooling,
        )
        self.gnn_tokengt = GnnTokenGT(
            input_dim_features_extractor=env_specification.n_features,
            max_n_nodes=env_specification.max_n_nodes,
            max_n_resources=env_specification.max_n_resources,
            conflicts="att",
            encoder_layers=4,
            encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
            encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
            encoder_attention_heads=agent_specification.n_attention_heads,
            activation_fn=agent_specification.activation_fn_graph,
            lap_node_id=True,
            lap_node_id_k=agent_specification.lap_node_id_k,
            lap_node_id_sign_flip=True,
            type_id=True,
            transformer_flavor=agent_specification.transformer_flavor,
            layer_pooling=agent_specification.layer_pooling,
            dropout=agent_specification.dropout,
            attention_dropout=agent_specification.dropout,
            act_dropout=agent_specification.dropout,
            cache_lap_node_id=agent_specification.cache_lap_node_id,
            performer_nb_features=agent_specification.performer_nb_features,
            performer_feature_redraw_interval=agent_specification.performer_feature_redraw_interval,
            performer_generalized_attention=agent_specification.performer_generalized_attention,
            performer_auto_check_redraw=agent_specification.performer_auto_check_redraw,
        )

        self.value_net = MLP(
            len(self.agent_specification.net_arch["vf"]),
            # self.gnn.features_dim // 2,
            self.gnn_dgl.features_dim // 2 + self.gnn_tokengt.features_dim // 2,
            self.agent_specification.net_arch["vf"][0],
            1,
            False,
            self.agent_specification.activation_fn,
        )

        # # action
        self.action_net = MLP(
            len(self.agent_specification.net_arch["pi"]),
            # self.gnn.features_dim,
            self.gnn_dgl.features_dim + self.gnn_tokengt.features_dim,
            self.agent_specification.net_arch["pi"][0],
            1,
            False,
            self.agent_specification.activation_fn,
        )
        # usually ppo use gain = np.sqrt(2) here
        # best so far below
        self.gnn_dgl.apply(partial(self.init_weights, gain=1.0, zero_bias=False))
        self.gnn_tokengt.apply(partial(self.init_weights, gain=1.0, zero_bias=False))
        # usually ppo use gain = 0.01 here
        self.action_net.apply(partial(self.init_weights, gain=1.0, zero_bias=True))
        # usually ppo use gain = 1 here
        self.value_net.apply(partial(self.init_weights, gain=1.0, zero_bias=True))

    @classmethod
    def load(cls, path):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        save_data = torch.load(path + "agent.pkl")
        agent_specification = save_data["agent_specification"]
        env_specification = save_data["env_specification"]
        if agent_specification.fe_type == "dgl":
            gnn = GnnDGL(
                input_dim_features_extractor=env_specification.n_features,
                gconv_type=agent_specification.gconv_type,
                graph_pooling=agent_specification.graph_pooling,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_resources=env_specification.max_n_resources,
                n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
                activation_features_extractor=agent_specification.activation_fn_graph,
                n_layers_features_extractor=agent_specification.n_layers_features_extractor,
                hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
                n_attention_heads=agent_specification.n_attention_heads,
                residual=agent_specification.residual_gnn,
                normalize=agent_specification.normalize_gnn,
                conflicts=agent_specification.conflicts,
                edge_embedding_flavor=agent_specification.edge_embedding_flavor,
                layer_pooling=agent_specification.layer_pooling,
            )
        elif agent_specification.fe_type == "tokengt":
            gnn = GnnTokenGT(
                input_dim_features_extractor=env_specification.n_features,
                device=agent_specification.device,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_resources=env_specification.max_n_resources,
                conflicts=agent_specification.conflicts,
                encoder_layers=agent_specification.n_layers_features_extractor,
                encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_attention_heads=agent_specification.n_attention_heads,
                activation_fn=agent_specification.activation_fn_graph,
                lap_node_id=True,
                lap_node_id_k=agent_specification.lap_node_id_k,
                lap_node_id_sign_flip=True,
                type_id=True,
                transformer_flavor=agent_specification.transformer_flavor,
                layer_pooling=agent_specification.layer_pooling,
                dropout=agent_specification.dropout,
                attention_dropout=agent_specification.dropout,
                act_dropout=agent_specification.dropout,
                cache_lap_node_id=agent_specification.cache_lap_node_id,
                performer_nb_features=agent_specification.performer_nb_features,
                performer_feature_redraw_interval=agent_specification.performer_feature_redraw_interval,
                performer_redraw_interval=agent_specification.performer_redraw_interval,
            )
        value_net = MLP(
            len(agent_specification.net_arch["vf"]),
            gnn.features_dim // 2,
            agent_specification.net_arch["vf"][0],
            1,
            False,
            agent_specification.activation_fn,
        )

        # # action
        action_net = MLP(
            len(agent_specification.net_arch["pi"]),
            gnn.features_dim,
            agent_specification.net_arch["pi"][0],
            1,
            False,
            agent_specification.activation_fn,
        )

        agent = cls(env_specification, gnn, value_net, action_net, agent_specification)
        # constructors init weight!!!
        agent.gnn.load_state_dict(save_data["gnn"])
        agent.action_net.load_state_dict(save_data["action_net"])
        agent.value_net.load_state_dict(save_data["value_net"])
        return agent

    def obs_as_tensor_add_batch_dim(self, obs):
        return PSPAgentObservation.np_to_torch(obs)

    def obs_as_tensor(self, obs):
        return PSPAgentObservation.np_to_torch(obs)

    def rebatch_obs(self, obs):
        return PSPAgentObservation.rebatch_obs(obs)

    def get_value(self, x):
        features_dgl = self.gnn_dgl(x)
        features_tokengt = self.gnn_tokengt(x)

        # filter out node specific features
        features_graph_dgl = features_dgl[:, 0, features_dgl.shape[2] // 2 :]
        features_graph_tokengt = features_tokengt[
            :, 0, features_tokengt.shape[2] // 2 :
        ]
        features_graph = torch.cat([features_graph_dgl, features_graph_tokengt], dim=-1)
        return self.value_net(features_graph)

    def get_action_and_value(
        self, x, action=None, action_masks=None, deterministic=False
    ):
        # features = self.gnn(x)
        features_dgl = self.gnn_dgl(x)
        features_tokengt = self.gnn_tokengt(x)
        features_graph_dgl = features_dgl[:, 0, features_dgl.shape[2] // 2 :]
        features_graph_tokengt = features_tokengt[
            :, 0, features_tokengt.shape[2] // 2 :
        ]

        # value = self.value_net(features[:, 0, features.shape[2] // 2 :])
        value = self.value_net(
            torch.cat([features_graph_dgl, features_graph_tokengt], dim=-1)
        )
        features = torch.cat([features_dgl, features_tokengt], dim=2)
        logits = self.action_net(features).squeeze(-1)
        if action_masks is not None:
            mask = torch.as_tensor(
                action_masks, dtype=torch.bool, device=features.device
            )
            HUGE_NEG = torch.tensor(-1e12, dtype=logits.dtype, device=features.device)
            logits = torch.where(mask, logits, HUGE_NEG)
        distrib = Categorical(logits=logits)
        if action is None:
            if deterministic == False:
                action = distrib.sample()
            else:
                action = torch.argmax(distrib.probs, dim=1)
        if action_masks is not None:
            p_log_p = distrib.logits * distrib.probs
            p_log_p = torch.where(mask, p_log_p, torch.tensor(0.0).to(features.device))
            entropy = -p_log_p.sum(-1)
        else:
            entropy = distrib.entropy()
        return action, distrib.log_prob(action), entropy, value

    def get_action_probs(self, x, action_masks):
        # features = self.gnn(x)
        features_dgl = self.gnn_dgl(x)
        features_tokengt = self.gnn_tokengt(x)
        features = torch.cat([features_dgl, features_tokengt], dim=2)

        action_logits = self.action_net(features).squeeze(-1)
        if action_masks is not None:
            mask = torch.as_tensor(
                action_masks, dtype=torch.bool, device=features.device
            )
            HUGE_NEG = torch.tensor(
                -1e12, dtype=action_logits.dtype, device=features.device
            )
            logits = torch.where(mask, action_logits, HUGE_NEG)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs
        # distrib = Categorical(logits=action_logits)
        # return distrib.probs
        probs = torch.nn.functional.softmax(action_logits, dim=-1)
        return probs

    def predict(self, observation, deterministic, action_masks):
        with torch.no_grad():
            # features = self.gnn(observation)

            features_dgl = self.gnn_dgl(observation)
            features_tokengt = self.gnn_tokengt(observation)
            features = torch.cat([features_dgl, features_tokengt], dim=2)
            logits = self.action_net(features)
            if action_masks is not None:
                mask = torch.as_tensor(
                    action_masks, dtype=torch.bool, device=features.device
                ).reshape(logits.shape)
                HUGE_NEG = torch.tensor(
                    -1e12, dtype=logits.dtype, device=features.device
                )
                logits = torch.where(mask, logits, HUGE_NEG)
            distrib = Categorical(logits=logits.squeeze(-1))
            if deterministic == False:
                action = distrib.sample()
            else:
                action = torch.argmax(distrib.probs, dim=1)
            return action

    def solve(self, problem_description):
        # Creating an environment on which we will run the inference
        env = Env(problem_description, self.env_specification)

        # Running the inference loop
        observation, info = env.reset()
        action_masks = info["mask"]
        done = False
        while not done:
            action_masks = get_action_masks(env)
            action = self.predict(
                observation, deterministic=True, action_masks=action_masks
            )
            observation, reward, done, _, info = env.step(action)
            mask = info["mask"]

        return env.get_solution()

    def save(self, path):
        """Saving an agent corresponds to saving his model and a few args to specify how the model is working"""
        device = next(self.gnn_dgl.parameters()).device
        self.to(torch.device("cpu"))
        torch.save(
            {
                "env_specification": self.env_specification,
                "agent_specification": self.agent_specification,
                "gnn_dgl": self.gnn_dgl.state_dict(),
                "gnn_tokengt": self.gnn_tokengt.state_dict(),
                "value_net": self.value_net.state_dict(),
                "action_net": self.action_net.state_dict(),
            },
            path,
        )
        self.to(device)
