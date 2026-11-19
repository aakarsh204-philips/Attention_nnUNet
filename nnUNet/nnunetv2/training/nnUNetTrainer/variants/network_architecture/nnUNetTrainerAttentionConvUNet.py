from typing import List, Tuple, Union

from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerAttentionConvUNet(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        arch_init_kwargs = dict(arch_init_kwargs)
        arch_init_kwargs.setdefault('squeeze_excitation', False)
        arch_init_kwargs.setdefault('squeeze_excitation_rd_ratio', 1.0 / 16)
        arch_init_kwargs.setdefault('squeeze_excitation_reduction_ratio', None)

        architecture_class_name = 'dynamic_network_architectures.architectures.unet.AttentionConvUNet'

        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )