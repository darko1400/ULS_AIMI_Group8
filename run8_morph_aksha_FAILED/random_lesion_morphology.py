"""
RandomLesionMorphologyTransform — Custom augmentation transform that randomly
dilates or erodes the lesion segmentation mask.

NOTE ON DESIGN: This transform modifies the segmentation mask ONLY, not the
image. It intentionally injects label-noise as a form of regularization
(teaching the model that lesion boundaries are noisy). This differs from the
other StrongAug transforms (rotation, scaling, elastic) which transform
image+seg together.

Authors: Nuno Silva (original idea) + cleaned up for cluster deployment.
"""
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class RandomLesionMorphologyTransform(BasicTransform):
    def __init__(
        self,
        lesion_label=1,
        p_per_sample=0.3,
        radius_range=(1, 3),
    ):
        super().__init__()
        self.lesion_label = lesion_label
        self.p_per_sample = p_per_sample
        self.radius_range = radius_range

    def apply(self, data_dict, **params):
        # Apply with probability p_per_sample, otherwise pass through unchanged
        if np.random.rand() > self.p_per_sample:
            return data_dict

        seg = data_dict["segmentation"]

        # batchgeneratorsv2 may pass torch tensors; convert to numpy for scipy ops
        was_torch = isinstance(seg, torch.Tensor)
        if was_torch:
            seg_device = seg.device
            seg_np = seg.cpu().numpy()
        else:
            seg_np = seg

        # seg shape is typically (C, D, H, W) for 3D; we operate on channel 0
        lesion = (seg_np[0] == self.lesion_label)

        # Skip if lesion is too small (degenerate cases)
        if lesion.sum() < 20:
            return data_dict

        radius = np.random.randint(self.radius_range[0], self.radius_range[1] + 1)
        structure = generate_binary_structure(lesion.ndim, 1)

        # 50/50 coin flip: dilate or erode
        if np.random.rand() < 0.5:
            for _ in range(radius):
                lesion = binary_dilation(lesion, structure=structure)
        else:
            for _ in range(radius):
                lesion = binary_erosion(lesion, structure=structure)

        # Update segmentation: clear old lesion, set new lesion
        seg_np[0][seg_np[0] == self.lesion_label] = 0
        seg_np[0][lesion] = self.lesion_label

        # Convert back to torch if we started with torch
        if was_torch:
            seg = torch.from_numpy(seg_np).to(seg_device)
        else:
            seg = seg_np

        data_dict["segmentation"] = seg
        return data_dict
