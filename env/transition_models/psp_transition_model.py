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

from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node, node_to_job_and_task


class PSPTransitionModel:
    def __init__(
        self,
        env_specification,
        problem,
    ):


    def run(self, state, node_id):  # noqa
        resources = problem.get_resources(node_id)
        state.observe_real_duration(node_id)
        for r in resources:
            last_mode_on_resource = state.get_last_mode_on_resource()
            if last_mode_on_resource is not None:
                state.set_priority(last_mode_on_resource, node_id)
            state.cache_last_mode_on_resource(r, node_id)
        state.update_completion_times(node_id)
        state.affect_node(node_id)


    def get_mask(self, state):
        return state.get_selectable() == 1
