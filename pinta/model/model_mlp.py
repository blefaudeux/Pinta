from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
from pinta.model.model_base import NN


class Mlp(NN):
    """
    Classic multi layer perceptron
    """

    def __init__(
        self,
        logdir: str,
        input_size: int,
        hidden_size: int,
        number_hidden_layers: int,
        output_size: int = 1,
        filename=None,
    ):
        super().__init__(logdir, "MLP")
        self.output_size = output_size

        module_list: List[Tuple[str, nn.Module]] = []

        module_list.append(("input_layer", nn.Linear(input_size, hidden_size)))
        module_list.append(("relu_0", nn.ReLU()))

        for i in range(number_hidden_layers):
            module_list.append((f"inner_layer_{i}", nn.Linear(hidden_size, hidden_size)))
            module_list.append((f"relu_{i+1}", nn.ReLU()))

        module_list.append(("output_layer", nn.Linear(hidden_size, output_size)))

        self.mlp = nn.Sequential(OrderedDict(module_list))

        # Load from trained NN if required
        try:
            if filename is not None and self.load(filename):
                self._valid = True
                return
        except RuntimeError:
            pass

    def get_layer_weights(self):
        def is_linear(module):
            return "layer" in module[0]

        # Select the linear layers, return the weights
        return map(lambda x: x[1].weight, filter(is_linear, self.mlp.named_modules()))

    def forward(self, inputs: torch.Tensor, tuning_inputs: torch.Tensor):
        inputs_avg = torch.mean(inputs, dim=2)
        return self.mlp(inputs_avg), None
