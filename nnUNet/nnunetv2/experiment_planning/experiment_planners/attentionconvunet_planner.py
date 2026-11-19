from dynamic_network_architectures.architectures.unet import AttentionConvUNet

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class AttentionConvUNetPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetAttentionConvUNetPlans',
                 overwrite_target_spacing=None, suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = AttentionConvUNet