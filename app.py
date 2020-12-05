from flask import Flask, request, jsonify
import keras_ocr
from typing import List, Dict
from PIL import Image
import numpy as np
import base64
import io
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KERAS_OCR_CACHE_DIR"] = 'models'

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

app = Flask(__name__)


# def inference(pipeline, images: List[Image.Image]) -> List[Image.Image]:
#     if not isinstance(images, list):
#         # pack in list if single image
#         images = [images]
#
#     for i, image in enumerate(images):
#         # convert to numpy array.
#         if isinstance(image, Image.Image):
#             images[i] = np.array(image)
#
#     # # Each list of predictions in prediction_groups is a list of
#     # # (word, box) tuples.
#     prediction_groups = pipeline.recognize(images)
#
#     processed_images = []
#     for image, predictions in zip(images, prediction_groups):
#         fig, ax = plt.subplots(figsize=(20, 20))
#         keras_ocr.tools.drawAnnotations(image, predictions, ax)
#         fig.canvas.draw()
#         # convert canvas to image
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         # img is rgb, convert to opencv's default bgr
#         image = Image.fromarray(image)
#         processed_images.append(image)
#     return processed_images


def inference(pipeline, images: List[Image.Image]) -> List[Dict]:
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
    results = []
    for image, prediction in zip(images, prediction_groups):
        result = []
        for data in prediction:
            result.append({
                'name': str(data[0]),
                'bbox': data[1].astype(int).tolist()
            })
        results.append(result)
    return results


@app.route('/api/', methods=["POST"])
def main_interface():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"

    image = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(image))

    if img.mode != 'RGB':
        img = img.convert("RGB")
    # img = img.resize((1280, 960))

    # convert to numpy array.
    img_arr = np.array(img)
    print(img_arr.shape)

    # do object detection in inference function.
    results = inference(pipeline, img_arr)
    results = {'results': results[0]}
    print(results)

    return jsonify(results)


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
