import base64
import io
import time
import glob
import cv2
import os
import pybase64
import requests
import numpy as np
from numpy import float32, float64
from PIL import Image as PILImage
from rvai.types import Image, Inputs,Mask
import re

def submit_inference_task(session, request_data, API_KEY, ENDPOINT_URL):
    """
    This function will submit an inference task (in our case an object detection task)
    to the API by sending an HTTP request.
    """
    # The HTTP headers we need for the API to understand us and let us in
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    # Actually send an HTTP POST request
    task_status = session.post(
        url=f"{ENDPOINT_URL}/api/v1/prediction",
        json=request_data,
        headers=headers,
        # The number of seconds we want to wait before the server responds,
        # if the server hasn't responded by then, we throw an error.
        timeout=60 * 3
    )
    return task_status


def get_inference_result(session, task_id, API_KEY, ENDPOINT_URL):
    """
    This function will request the result of our inference tasks, i.e. which
    objects where detected. The task_id is something that is given when
    submitting the task.
    """
    # The HTTP headers we need for the API to understand us and let us in
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    # Actually send an HTTP POST request
    response = requests.get(
        url=f"{ENDPOINT_URL}/api/v1/prediction/{task_id}",
        headers=headers,
        timeout=60 * 3  # Same time-out logic as above.
    )
    while response.status_code != 200:
        time.sleep(0.1)
        # Actually send an HTTP POST request
        response = requests.get(
            url=f"{ENDPOINT_URL}/api/v1/prediction/{task_id}",
            headers=headers,
            timeout=60 * 3  # Same time-out logic as above.
        )
    return response

# Self defined function for decoding base64
def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    # data = re.sub('[^a-zA-Z0-9%s]+' % altchars, '', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += '='* (4 - missing_padding)
    return base64.b64decode(data)

def inference(img, API_KEY, ENDPOINT_URL):

    # Get dimensions of image for using in 'shape' argument in request
    # imgdata = base64.b64decode(img) # standard function
    imgdata = decode_base64(img) # self defined function
    im = PILImage.open(io.BytesIO(imgdata))
    dimensions = np.array(im).shape



    # Here we create a session that keeps track of all our requests.
    # It also allows us to reuse the connection to the API server
    # for multiple request (e.g. submit a task, then get it's result)
    session = requests.Session()

    # Encode the image in the correct format
    # request_data = Inputs(image=Image(img)).to_json_struct()
    # Now only base64 string was passed by API
    request_data = {'$type': 'Inputs',
                        'entries': {'image': {'$type': 'Image',
                                                '$encoding': {'data': 'png+base64'},
                                                'data': {'buffer': img,
                                                         'kind': '|u1',
                                                         'shape': dimensions,
                                                         'scaled': 1}
                                                
                                                }}
                    }


    response_submit = submit_inference_task(session, request_data, API_KEY, ENDPOINT_URL)
    response_get = get_inference_result(
        session, response_submit.json()['taskID'], API_KEY, ENDPOINT_URL)

    return response_get.json()
