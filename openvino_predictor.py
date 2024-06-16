import openvino as ov
import utils

class DepthAnythingV2OpenVINO:
    def __init__(self, model_type="vits", device="AUTO"):
        self.ov_model_path = f"models_ov/depth_anything_v2_{model_type}_int8/depth_anything_v2_{model_type}_int8.xml"
        self.core = ov.Core()
        self.compiled_model = self.core.compile_model(self.ov_model_path, device)

    def predict(self, image):
        """depth estimation prediction method from a RGB Image.
        Args:
            image (numpy): RGB Image of shape (height, width, 3)
        """
        input_tensor, image_size = utils.image_preprocess(image)
        out = self.compiled_model(input_tensor)[0]
        depth = utils.postprocess(out, image_size)
        return depth