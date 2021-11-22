from flask import Flask, jsonify, request

import logging

import datetime
import inference_script
import metrics_script

# Initiate flask app
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello world'

@app.route('/predict_api', methods=["GET"])
def chipping():

    # Read in json input file
    json_data = request.json

    # Unpack json input file
    Snapid = json_data.get('Snapid')
    Ritszaag = json_data.get('Ritszaag')
    Chippinglimietwaarde = json_data.get('Chipping limietwaarde')
    Bovenpapierid = json_data.get('Bovenpapier id')
    Plaatnummerinstapel = json_data.get('Plaatnummer in stapel')
    Image = json_data.get('Image')


    # Instantiate variables for inference
    # The API key grants our script acces for talking to the Robovision API
    API_KEY = "179f3366-0692-4120-8584-348038a92e3c"
    # ENDPOINT_URL = "https://halley.robovision.ai/pipelines/ray-cpu-worker-bfx9j/7000"
    ENDPOINT_URL = "https://halley.robovision.ai/pipelines/ray-cpu-worker-dbj5c/7000"

    # Instantiate variables for calculating metrics
    pixels_per_mm_x = 23.17
    pixels_per_mm_y = 23.1875
    nr_pixels_peak = Chippinglimietwaarde
    common_pixels = 1

    ## PERFORM INFERENCE
    print("Start inference : {}".format(datetime.datetime.now()))
    result_inference = inference_script.inference(Image, API_KEY, ENDPOINT_URL)
    print("End inference :  {}".format(datetime.datetime.now()))

    ## CALCULATE METRICS
    print("Start calculating metrics : {}".format(datetime.datetime.now()))
    message, total_chip_pix, total_chip_mm2, nr_peaks = metrics_script.calculate_metrics(result_inference, pixels_per_mm_x, pixels_per_mm_y, nr_pixels_peak, common_pixels, Snapid, Ritszaag, Chippinglimietwaarde, Bovenpapierid, Plaatnummerinstapel, Image)
    print(message)
    print("End calculating metrics : {}".format(datetime.datetime.now()))

    # Construct final result in JSON
    result_function = {'Snapid': Snapid,
                        'Ritszaag': Ritszaag,
                        'Chipping limietwaarde': Chippinglimietwaarde,
                        'Bovenpapier id': Bovenpapierid,
                        'Plaatnummer in stapel': Plaatnummerinstapel,
                        'Chipping resultaat mm²': total_chip_mm2,
                        'Chipping fouten': nr_peaks
    }


    return result_function
    

if __name__ == "__main__":
    app.run()
    # app.run(host='0.0.0.0', port=5000)