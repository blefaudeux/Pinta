import torch.nn as nn
from pinta.model.model_base import NN
from collections import OrderedDict
import torch


class Transformer(NN):
    """
    Vanilla Transformer model, no prior convolutions
    """

    def __init__(
        self,
        logdir,
        input_size: int,
        output_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        context_len: int = 512,
        dropout: float = 0.1,
        filename="",
    ):
        super().__init__(
            logdir,
            "TransformerModel",
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )

        # FIXME: MLP on the way in ?
        self.model = nn.Sequential(
            OrderedDict(
                **{
                    "proj_in": nn.Linear(input_size, hidden_size),
                    "non_lin_in": nn.SiLU(),
                    "transformer": nn.TransformerEncoder(
                        encoder_layer=encoder_layer,
                        num_layers=num_layers,
                        enable_nested_tensor=False,
                        # norm=nn.RMSNorm(normalized_shape=hidden_size),
                    ),
                    "non_lin_out": nn.SiLU(),
                    "proj_out": nn.Linear(hidden_size, output_size),
                }
            )
        )
        self._valid = False

        self.log.info(
            "Model created. Receptive field is {} samples".format(context_len)
        )
        self.log.info(self.model)

        # Load from trained NN if required
        try:
            if filename is not None and self.load(filename):
                self._valid = True
                return
        except RuntimeError:
            pass

        self.log.warning(
            "Could not load the specified model," " needs to be computed from scratch"
        )

    def forward(self, x: torch.Tensor, *_, **__):  # type:ignore
        # We're getting BDN, we should get BND here, some weird choices in the past
        x = x.movedim(-1, -2)
        pred = self.model(x)

        # Instead of pooling, we return the last token in the sequence
        return pred[..., -1, :], None
