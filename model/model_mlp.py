import torch.nn as nn
from model.model_base import NN


class MLP(NN):
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

        module_list = []

        module_list.append(["input_layer", nn.Linear(input_size, hidden_size)])
        module_list.append(["relu_0", nn.ReLU()])

        for i in range(number_hidden_layers):
            module_list.append(
                [f"inner_layer_{i}", nn.Linear(hidden_size, hidden_size)]
            )
            module_list.append([f"relu_{i+1}", nn.ReLU()])

        module_list.append(["output_layer", nn.Linear(hidden_size, output_size)])
        self.model = nn.ModuleList(module_list)

    def get_layer_weights(self):
        # TODO: ben
        return None

    def forward(self, x):
        return self.model(x), None
