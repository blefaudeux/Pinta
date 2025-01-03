from typing import List

import torch.nn as nn
from pinta.model.model_base import NN


"""
Code from https://github.com/facebookresearch/VideoPose3D
Dilated convolutions over time, very elegant architecture.
Minor modifications to fit our usecase


"Dario Pavllo, Christoph Feichtenhofer,"
"David Grangier, and Michael Auli. "
"3D human pose estimation in video with temporal convolutions
and semi-supervised training.
In Conference on Computer Vision and Pattern Recognition (CVPR),
2019."
"""


class TemporalModelBase(NN):
    """
    Do not instantiate this class.
    """

    def __init__(
        self,
        log_dir,
        num_input_channels: int,
        num_output_channels: int,
        filter_widths: List[int],
        dropout: float,
        channels: int,
        bn_momentum: float = 0.1,
    ):
        super().__init__(log_dir, "TemporalDilatedConv")

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, "Only odd filter widths are supported"

        self.num_input_channels = num_input_channels
        self.output_size = num_output_channels
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=bn_momentum)
        self.shrink = nn.Conv1d(channels, self.output_size, 1)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 3

        assert (
            x.shape[1] == self.num_input_channels
        ), f"Input shape {x.shape} - expected {self.num_input_channels} inputs"

        sz = x.shape[:3]

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.output_size)

        return x, None


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(
        self,
        logdir,
        num_input_channels,
        num_output_channels,
        filter_widths,
        dropout: float = 0.25,
        channels: int = 1024,
        bn_momentum: float = 0.1,
        filename="",
    ):
        """
        Initialize this model.

        Arguments:
        num_input_channels
        num_output_channels
        filter_widths -- list of convolution widths,
         which also determines the # of blocks and receptive field



        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(
            logdir,
            num_input_channels,
            num_output_channels,
            filter_widths,
            dropout,
            channels,
            bn_momentum,
        )

        self.expand_conv = nn.Conv1d(
            num_input_channels, channels, filter_widths[0], bias=False
        )

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation))

            layers_conv.append(
                nn.Conv1d(
                    channels,
                    channels,
                    filter_widths[i],
                    dilation=next_dilation,
                    bias=False,
                )
            )
            layers_bn.append(nn.BatchNorm1d(channels, momentum=bn_momentum))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=bn_momentum))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self._valid = False

        self.log.info(
            "Model created. Receptive field is {} samples".format(
                self.receptive_field()
            )
        )

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

    def _forward_blocks(self, x):
        x = self.expand_bn(self.expand_conv(x))
        x = self.drop(self.relu(x))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(
                self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x)))
            )

        return self.shrink(x)
