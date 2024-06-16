from PIL import Image
import requests
from io import BytesIO

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