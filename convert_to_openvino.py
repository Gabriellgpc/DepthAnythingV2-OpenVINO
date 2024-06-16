import torch
from pathlib import Path

import openvino as ov

from depth_anything_v2.dpt import DepthAnythingV2

import numpy as np

import utils

if __name__ == '__main__':

    ##########################
    # Load Pre-trained model #
    ##########################
    model_select = "vits"

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    weights_path = f"weights/depth_anything_v2_{model_select}.pth"

    model = DepthAnythingV2(**model_configs[model_select]).eval()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    ########################
    # Get Sample RGB Image #
    ########################

    image_url = "https://images.pexels.com/photos/5740792/pexels-photo-5740792.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    image = np.array(utils.download_image(image_url))

    ########################
    # Preprocess RGB Image #
    ########################
    input_tensor, image_size = utils.image_preprocess(image)

    #######################
    # Convert to OpenVINO #
    #######################

    ov_model_path = Path("models_ov") / Path(Path(weights_path).name.replace(".pth", ".xml"))
    if not ov_model_path.exists():
        ov_model = ov.convert_model(model, example_input=input_tensor, input=[1, 3, 518, 518])
        ov.save_model(ov_model, ov_model_path)