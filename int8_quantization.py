import datasets
from tqdm import tqdm
import numpy as np

import nncf
import openvino as ov

import utils

def main():
    subset_size = 300
    dataset_hf  = "depth-anything/DA-2K"
    # input model
    ov_model_path = "models_ov/depth_anything_v2_vits.xml"

    #######################
    # Calibration dataset #
    #######################
    print("[INFO] Making Calibration dataset ...")

    calibration_data = []
    dataset = datasets.load_dataset(dataset_hf, split="train", streaming=True).shuffle(seed=42).take(subset_size)
    for batch in tqdm(dataset):
        image = np.array(batch["image"])[...,:3]
        input_tensor, _ = utils.image_preprocess(image)
        calibration_data.append(input_tensor)

    # output path
    ov_model_int8_path = ov_model_path.replace(".xml", "_int8.xml")

    print("[INFO] Reading input ov model ...")
    core = ov.Core()
    model = core.read_model(ov_model_path)

    print("[INFO] Running quantization process ...")
    quantized_model = nncf.quantize(
        model=model,
        subset_size=subset_size,
        model_type=nncf.ModelType.TRANSFORMER,
        calibration_dataset=nncf.Dataset(calibration_data),
    )

    print("[INFO] Saving quantized model at {} ...".format(ov_model_int8_path))
    ov.save_model(quantized_model, ov_model_int8_path)

    print("[INFO] Done!")

if __name__ == "__main__":
    main()