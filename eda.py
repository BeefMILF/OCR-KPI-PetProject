import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import keras_ocr
from PIL import Image
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KERAS_OCR_CACHE_DIR"] = 'models'


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()
image = Image.open("test1.jpg")


def inference(pipeline, images: List[Image.Image]) -> List[Image.Image]:
    if not isinstance(images, list):
        # pack in list if single image
        images = [images]

    for i, image in enumerate(images):
        # convert to numpy array.
        if isinstance(image, Image.Image):
            images[i] = np.array(image)

    # # Each list of predictions in prediction_groups is a list of
    # # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)

    processed_images = []
    for image, predictions in zip(images, prediction_groups):

        fig, ax = plt.subplots(figsize=(20, 20))
        keras_ocr.tools.drawAnnotations(image, predictions, ax)
        fig.canvas.draw()
        # convert canvas to image
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        image = Image.fromarray(image)
        processed_images.append(image)
    return processed_images


inference(pipeline, image)