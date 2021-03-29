import cv2
from PIL import Image
import numpy as np
import matplotlib.pylab as plt


def cv2array2pil(img):
    """
    Take a numpy array from OpenCV BGR format and output it to PIL format to display (RGB).

    Though this is a simple one-liner one often forgets which underlying format the libraries use.

    Especially useful for displaying PIL images in Jupyter Notebooks, since OpenCV doesn't work there since
    it requires a window manager (e.g. libGTK).

    :param img: Numpy Array containing image in OpenCV BGR format
    :return: Image in RGB format
    """

    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil_image


_disable_img_display = False


def disable_cv2_display():
    """
        Globally disables displaying images with functions internal to this module.
        Eg: cv2_display_img(...)
    """
    global _disable_img_display
    _disable_img_display = True


def enable_cv2_display():
    """
        Globally enables displaying images with functions internal to this module.
        Eg: cv2_display_img(...)
    """
    global _disable_img_display
    _disable_img_display = False


def cv2_display_img(img, window_title="Test Image", force_display=False):
    """
    Displays a cv2 window with an image. Ensures you don't forget to "waitKey(0)" after displaying
    """
    global _disable_img_display
    if not _disable_img_display:
        cv2.imshow(window_title, img)
        cv2.waitKey(0)


def display_img_notebook(img_path : str):
    """

    Args:
        img_path: Path to an image

    Returns: An image in PIL format which will automatically be displayed in an IPython notebook

    """
    return display_img(img_path, separate_window=False)


def display_img(img_path: str, separate_window=True):
    """
    Loads an image via PIL and displays it. By default displays it in a new window. It can be displayed in a notebook
    by changing the separate_window attribute
    :param img_path: Path to image
    :param separate_window:  Whether to display the image in a separate window or just return it
    :return: The Image
    """
    img = Image.open(img_path, 'r')
    if separate_window:
        img.show()
    return img


def display_np_img(np_img, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = np_img.shape[0]
    x = np_img.shape[1]
    w = (y / x) * h
    plt.figure(figsize=(w, h))
    plt.imshow(np_img, interpolation="none", **kwargs)
    plt.axis('off')
    plt.show()
