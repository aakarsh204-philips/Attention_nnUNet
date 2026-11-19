from typing import Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU


class AttentionGate(nn.Module):
	"""
	Attention U-Net style gate that suppresses irrelevant skip features.
	"""

	def __init__(
		self,
		conv_op: Type[_ConvNd],
		gating_channels: int,
		skip_channels: int,
		inter_channels: int = None,
		conv_bias: bool = True,
		norm_op: Union[None, Type[nn.Module]] = None,
		norm_op_kwargs: dict = None,
	):
		super().__init__()
		if inter_channels is None:
			inter_channels = max(1, skip_channels // 2)
		if norm_op_kwargs is None:
			norm_op_kwargs = {}

		self.W_g = ConvDropoutNormReLU(
			conv_op=conv_op,
			input_channels=gating_channels,
			output_channels=inter_channels,
			kernel_size=1,
			stride=1,
			conv_bias=conv_bias,
			norm_op=norm_op,
			norm_op_kwargs=norm_op_kwargs,
			dropout_op=None,
			dropout_op_kwargs=None,
			nonlin=None,
			nonlin_kwargs=None,
		)

		self.W_x = ConvDropoutNormReLU(
			conv_op=conv_op,
			input_channels=skip_channels,
			output_channels=inter_channels,
			kernel_size=1,
			stride=1,
			conv_bias=conv_bias,
			norm_op=norm_op,
			norm_op_kwargs=norm_op_kwargs,
			dropout_op=None,
			dropout_op_kwargs=None,
			nonlin=None,
			nonlin_kwargs=None,
		)

		self.psi = ConvDropoutNormReLU(
			conv_op=conv_op,
			input_channels=inter_channels,
			output_channels=1,
			kernel_size=1,
			stride=1,
			conv_bias=True,
			norm_op=norm_op,
			norm_op_kwargs=norm_op_kwargs,
			dropout_op=None,
			dropout_op_kwargs=None,
			nonlin=nn.Sigmoid,
			nonlin_kwargs={},
		)
		self.relu = nn.ReLU(inplace=True)

		self.inter_channels = inter_channels

	def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
		return x * alpha

	def compute_conv_feature_map_size(self, input_size):
		output = np.int64(0)
		output += self.W_g.compute_conv_feature_map_size(input_size)
		output += self.W_x.compute_conv_feature_map_size(input_size)
		output += self.psi.compute_conv_feature_map_size(input_size)
		return output
