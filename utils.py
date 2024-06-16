from PIL import Image
import requests
from io import BytesIO

import numpy as np
import cv2


def download_image(url):
    """
    Download an image from a given URL and return it as a PIL Image.

    Parameters:
    url (str): The URL of the image.

    Returns:
    PIL.Image: The downloaded image.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

def image_preprocess(image):
    """
        Input:
            image: RGB image, [Height, Width, Channels] as numpy array
        Output:
            input_tensor, (h_o, w_o)
        input_tensor -> ready to feed the model
        and original height and width of the given image
    """
    # save original shape
    image_size = image.shape[:2]
    # normalize [0, 1]
    input_tensor = image / 255.0
    # Resize to [518, 518]
    input_tensor = cv2.resize(input_tensor, dsize=[518, 518], interpolation=cv2.INTER_CUBIC)

    # mean and std
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    input_tensor = (input_tensor - mean) / std

    # turn it channels first.
    # (h, w, c) -> (c, h, w)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))

    # add batch size
    input_tensor = np.expand_dims(input_tensor, 0)

    # force dtype to float32
    input_tensor = input_tensor.astype("float32")
    return input_tensor, image_size

def postprocess(model_output, image_size):
    depth = model_output[0]
    h, w = image_size
    depth = cv2.resize(depth, dsize=(w, h), interpolation=cv2.INTER_AREA)

    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    depth = cv2.applyColorMap(depth, colormap=cv2.COLORMAP_INFERNO)
    return depth

def download_video(url, save_path):
    """
    Download a video from the given URL and save it to the local disk.

    Parameters:
    url (str): The URL of the video to download.
    save_path (str): The local file path to save the downloaded video.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Video downloaded successfully and saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")