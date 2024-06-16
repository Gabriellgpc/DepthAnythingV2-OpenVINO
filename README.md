# DepthAnythingV2 OpenVINO

## How to make the OpenVINO IR from the original Pytorch Model

1. Download Official weights from the authors
Example to download `Depth-Anything-V2-Large`
```bash
cd weights
git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Large
```

Full original collection of [models at Hugginface](https://huggingface.co/collections/depth-anything/depth-anything-v2-666b22412f18a6dbfde23a93)

Pre-trained Models

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |

## Envirnoment Setup

conda environment with Python 3.11.
```bash
conda create -n fastdepthv2 python=3.11 -y
conda activate fastdepthv2
pip install -r requirements-<full/inf>.txt
```

`Note:` In case you want just the inference from the already converted OpenVINO model, you can install only the `requirements-inf.txt`.

# Convert and Quantized Model to OpenVINO IR

| Model | Params | OpenVINO IR |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [FP16](https://drive.google.com/drive/folders/1jghPSOjJPiXDSP_RNJQ3NE2iqRJd-ih3?usp=sharing) [INT8](https://drive.google.com/drive/folders/1xB4UpU0wFDFPqnCf-kDUfAv08k8gNv1l?usp=sharing) |
| Depth-Anything-V2-Base | 97.5M | [FP16](https://drive.google.com/drive/folders/1Lr6K6qiiKG9jtl65RnvdvrkdE1ivnVLb?usp=sharing) [INT8](https://drive.google.com/drive/folders/1MT5yCO2MUe0mYaknpt5gRzMCfWi_S2GT?usp=sharing) |
| Depth-Anything-V2-Large | 335.3M | [FP16](https://drive.google.com/drive/folders/1-GSR1RDgvwGW0zHQZqB9L-EvAm9EkRvi?usp=sharing) [INT8](https://drive.google.com/drive/folders/1QtXVHYNCdp2imyC6GrXZFShqafjv3iJ6?usp=sharing) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |


# Acknowledgements
This work is heavily based on the Depth Anything notebook by the OpenVINO Toolkit team. Special thanks to Yang et al. (2024) for their groundbreaking research.

# References

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [DPT Depth Results](https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md)
- [OpenVINO](https://docs.openvino.ai/2024/home.html)
- [ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2)
- [Export Pytorch to ONNX](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [Convert to OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/convert-to-openvino/convert-to-openvino.ipynb)
- [OpenVINO Model Conversion API](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/convert-to-openvino/legacy-mo-convert-to-openvino.ipynb)