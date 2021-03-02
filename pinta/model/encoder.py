import torch


"""
Encode slow varying variables, for instance boat tuning parameters
"""


class TuningEncoder(torch.nn.Module):
    def __init__(self, inputs: int, hidden: int, out_features: int):
        super().__init__()

        # FIXME: dumb testing, just a MLP, could probably do with something smarter
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=inputs, out_features=hidden, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_features),
        )
        self.output_size = out_features

    def forward(self, inputs: torch.Tensor):
        return self.mlp(inputs)
