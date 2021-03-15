import torch
from typing import Dict, Any, List
from pinta.model.model_base import NN
from pinta.model.encoder import TuningEncoder


class Mixer(NN):
    def __init__(self, logdir, trunk: torch.nn.Module, tuning_encoder: TuningEncoder, params: Dict[str, Any]):
        super().__init__(logdir)

        self.trunk = trunk
        self.tuning_encoder = tuning_encoder

        # A bit naive, just use yet another MLP for now
        layers: List[torch.nn.Module] = [
            torch.nn.Linear(
                in_features=trunk.output_size + tuning_encoder.output_size,
                out_features=params["mixer"]["hidden_size"],
            )
        ]
        for _ in range(params["mixer"]["hidden_layers"]):
            layers.append(
                torch.nn.Linear(in_features=params["mixer"]["hidden_size"], out_features=params["mixer"]["hidden_size"])
            )
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(in_features=params["mixer"]["hidden_size"], out_features=len(params["outputs"])))

        self.mixer = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, tuning_inputs: torch.Tensor) -> torch.Tensor:
        temporal_signal, _ = self.trunk(inputs)
        tuning_signal = self.tuning_encoder(tuning_inputs)

        return self.mixer(torch.cat((temporal_signal.squeeze(), tuning_signal), dim=1))

    def get_layer_weights(self):
        import logging

        logging.warning("Only returning the trunk's linear layers weights, would need to be fixed")

        def is_linear(module):
            return "layer" in module[0]

        return []

        # Select the linear layers, return the weights
        # return map(lambda x: x[1].weight, filter(is_linear, self.trunk.named_modules()))
