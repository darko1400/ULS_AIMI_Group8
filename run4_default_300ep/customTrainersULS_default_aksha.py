import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_default_300_aksha(nnUNetTrainer):
    """Default nnUNetTrainer with num_epochs overridden to 300.
    Uses standard nnUNet augmentation pipeline (no StrongAug additions).
    Created for baseline comparison vs StrongAug variants."""
    def __init__(self, plans, configuration, fold, dataset_json, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2.5e-3
        self.num_epochs = 300
