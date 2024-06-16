import torch
from depth_anything_v2.dpt import DepthAnythingV2

import utils

class DepthAnythingV2Pytorch:
    def __init__(self, model_type="vits", device="cpu"):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        self.device = device
        self.weights_path = f"weights/depth_anything_v2_{model_type}.pth"
        self.model = DepthAnythingV2(**self.model_configs[model_type]).eval()
        self.model.load_state_dict(torch.load(self.weights_path, map_location=device))

    def predict(self, image):
        """depth estimation prediction method from a RGB Image.
        Args:
            image (numpy): RGB Image of shape (height, width, 3)
        """
        input_tensor, image_size = utils.image_preprocess(image)
        out = self.model(torch.from_numpy(input_tensor))
        depth = utils.postprocess(out.cpu().detach().numpy(), image_size)
        return depth