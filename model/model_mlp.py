from collections import OrderedDict
from typing import List, Tuple

import torch.nn as nn

from model.model_base import NN


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

        module_list: List[Tuple[str, nn.Module]] = []

        module_list.append(("input_layer", nn.Linear(input_size, hidden_size)))
        module_list.append(("relu_0", nn.ReLU()))

        for i in range(number_hidden_layers):
            module_list.append(
                (f"inner_layer_{i}", nn.Linear(hidden_size, hidden_size))
            )
            module_list.append((f"relu_{i+1}", nn.ReLU()))

        module_list.append(("output_layer", nn.Linear(hidden_size, output_size)))

        self.mlp = nn.Sequential(OrderedDict(module_list))

    def get_layer_weights(self):
        # TODO: ben, only return the linear weights
        return None

    def forward(self, x):
        # Do not use time, for now at least, only use the latest sample
        return self.mlp(x[:, :, -1]), None
