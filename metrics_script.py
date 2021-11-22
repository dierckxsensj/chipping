# imports
import base64
import io
import time
import glob
import cv2
import os
import pybase64
import requests
import numpy as np
import pandas as pd
import json
from numpy import float32, float64
from rvai.types import Image, Inputs,Mask
import datetime
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont # For plotting the results
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from matplotlib.pyplot import figure
from azure.storage.blob import BlobServiceClient

# define possible fit-function (in this case only linear since we want a straight line)
def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

def calculate_metrics(result_API, pixels_per_mm_x, pixels_per_mm_y, nr_pixels_peak, common_pixels, snapid, ritszaag, chippinglimietwaarde, bovenpapierid, plaatnummerinstapel, image_string):

    if not result_API['entries']['masks']['items']:
        message = 'No mask detected'
        total_chip_pix = 0
        total_chip_mm2 = 0
        nr_peaks = 0
        print('There are {nr} peaks larger than {pix} pixels per peak'.format(nr=nr_peaks, pix = nr_pixels_peak))
        print("Total Chipping [#p]:{chip}".format(chip=total_chip_pix))
        print("Total Chipping [mm2]:{chip}".format(chip=total_chip_mm2))
    else:
        message = 'Mask detected successfully'

        # Unpack result to mask
        result_unpack = result_API['entries']['masks']['items'][0]
        # Remove version parameters
        del result_unpack['$version']
        del result_unpack['$class']['$version']
        del result_unpack['$class']['$attributes']['$version']
        del result_unpack['$class']['$attributes']['entries']['class_metadata']['$version']
        del result_unpack['$class']['$attributes']['entries']['class_metadata']['value']['$version']
        
        result_mask = Mask.from_json_struct(result_unpack)

        # Convert mask to array
        mask_array = np.array(result_mask)
        # Convert mask_array to mask on image
        mask_image = PILImage.fromarray(np.uint8(255*mask_array)) # value smaller than 255 gives a particular grey color
        
        # Calculate image dimensions
        height = mask_image.size[1] 
        width = mask_image.size[0] 
        # print('The format of the image has \n height:{h} \n width:{w}'.format(h = height, w = width))


        # ## Create folder for image to store results
        # image = image_file.split('/')[-1].split('.')[0]
        # folder_output_image = image_folder + '_output_detected/' + image # CHANGED '/output_detected/' to '_output_detected/'
        # if not os.path.exists(folder_output_image):
        #     os.makedirs(folder_output_image)

        # # Save detected mask
        # mask_image.save(folder_output_image + '/' + image + '_detected_mask.jpg')

        # # Save detected mask pasted on image
        # mask_image_transp = PILImage.fromarray(np.uint8(100*mask_array)) # value smaller than 255 --> particular transparency
        # image_chipping_mask = PILImage.open(image_file)
        # image_chipping_mask = image_chipping_mask.convert('RGB') # convert to RGB scale to plot colored mask om image
        # red_im = PILImage.new("RGBA", image_chipping_mask.size, color = 'red')
        # image_chipping_mask.paste(im = red_im, box = (0,0), mask = mask_image_transp) # paste colored mask on image
        # image_chipping_mask.save(folder_output_image + '/' + image + '_image_with_detected_mask.jpg') # save image with transparent mask





        ## Find best fitting straight line for mask of pixels to determine intercept

        # Get coordinates of pixels corresponding to marked pixels
        X = np.argwhere(mask_array)
        pixels = pd.DataFrame(data = X)
        # Convert pixel numbers to x and y coordinates
        pixels['x'] = pixels[1]
        pixels['y'] = height - pixels[0] # origin is in top right corner in image terminology
        
        # if less then 5000 pixels detected, no postprocessing is needed, because probably no edge was in the image
        if pixels.shape[0] < 5000:
          message = 'No significant chipping detected'
#           total_chip_pix = pixels.shape[0]
          total_chip_pix = 0
#           total_chip_mm2 = total_chip_pix / pixels_per_mm / pixels_per_mm
          total_chip_mm2 = 0
          nr_peaks = 0
          print('There are {nr} peaks larger than {pix} pixels per peak'.format(nr=nr_peaks, pix = nr_pixels_peak))
          print("Total Chipping [#p]:{chip}".format(chip=total_chip_pix))
          print("Total Chipping [mm2]:{chip}".format(chip=total_chip_mm2))
        else:
        
          # Get pixel coordinates of lowest pixels in detected mask
          lowest_pixels = pixels.groupby('x', as_index=False)['y'].min()

        # fit straight line to lowest pixel coordinates
          popt, pcov = curve_fit(f, xdata = lowest_pixels['x'], ydata = lowest_pixels['y']) # your data x, y to fit
          coeff = popt[0]
          intercept = popt[1]
          # print('The coefficient is {}'.format(coeff))
          # print('The intercept is {}'.format(intercept))





          ## Start with straight line a number of pixels below fitted line and slide to top until at least 1 pixel of top curve is covered

          # create dataframe with pixels of curve surfacing on the top of the mask
          top_curve = pixels.groupby('x', as_index=False)['y'].max()

          # Move straight line 1 pixel up at a time, until straight line has at least 1 pixel in common with top curve
          overlap = 0
          nr_pixels_below_line = 50
          while overlap < common_pixels:
              intercept_new = intercept - nr_pixels_below_line
              new_line = pd.DataFrame({'x' : range(1, width + 1)})
              new_line['y'] = round(intercept_new + new_line['x'] * coeff).astype('int64')
              # calculate number of pixels 'top_curve' (top border) and 'new_line' (new straight line) have in common
              intersection = pd.merge(top_curve, new_line, how = 'inner', on = ['x', 'y'])
              overlap = intersection.shape[0]
              # print('Number of common pixels : {}'.format(overlap))
              nr_pixels_below_line -= 1 # go 1 pixel closer to the mask
          # print('The details of the straight line are \n intercept : {i} \n coefficient : {c}'.format(i = intercept_new, c = coeff))

          # Set new_line to be the final straight line we'll use to determine the intermediary chipping pixels
          straight_line = new_line






          ## All black surfaces between determined straight line and white surfaces (intermediary chipping pixels) --> consider as chipping as well

        #   # plot top curve and bottom straight line and save
        #   figure(figsize=(30,4))
        #   plt.plot(straight_line['x'], straight_line['y']);
        #   plt.plot(top_curve['x'], top_curve['y']);
        #   plt.savefig(folder_output_image + '/' + image + '_top_and_bottom_borders.jpg')
        #   plt.close()

          # create dataframe 'all_pixels' that contain all pixels in the bottom straight line, the top curve and in between
          all_pixels = []
          for idx in straight_line.index:
              x_sl = straight_line._get_value(idx, 'x')
              y_sl = straight_line._get_value(idx, 'y')

              # try catch in case the x value is not found in the top curve
              # try: 
              if x_sl in top_curve['x'].unique():
                  y_tc = top_curve.loc[top_curve['x'] == x_sl]['y'].values[0] # unpack y-value
                  y = y_sl # instantiate y value for initiating while loop --> start from below
                  while y <= y_tc:
                      all_pixels.append([x_sl, y]) # add pixels if they are on straight line, between sl and tc or on top curve
                      y += 1 # go 1 pixel up
              # except:
              #     print('X value {} from straight line cannot be found in top curve'.format(x_sl))
          ap = pd.DataFrame(all_pixels, columns = ['x', 'y']) # convert list to dataframe






          ## Infer full mask and plot on image

          # convert pixels to coordinates and regenerate mask
          ap['x_coord'] = ap['x']
          ap['y_coord'] = height - ap['y']
          mask_inferred = np.zeros(shape = (height,width), dtype = bool)
          mask_inferred[ap['y_coord'],ap['x_coord']] = True

        #   # Create mask image with intermediary surfaces and paste on image
        #   mask_inferred_image = PILImage.fromarray(np.uint8(100*mask_inferred), mode = 'L') # change factor of 'mask_inferred' for changing opacity
        #   red_im = PILImage.new("RGBA", mask_image.size, color = 'red') # Create red image that will be pasted on chipping image to indicate the mask
        #   image_chipping = PILImage.open(image_file) # Open chipping image
        #   image_chipping = image_chipping.convert('RGB') # convert to RGB scale to plot colored mask om image
        #   image_chipping.paste(im = red_im, box = (0,0), mask = mask_inferred_image) # Paste colored mask on image
        #   image_chipping.save(folder_output_image + '/' + image + '_with_calculated_chipping.jpg') # Save image and calculated chipping in image directory
        #   image_chipping.save(image_folder + '_output_detected/' + image + '_with_calculated_chipping.jpg') # Save image and calculated chipping in 'output_detected' directory
        #   # CHANGED '/output_detected/' to '_output_detected/'



          ## Calculate all performance metrics

          # count number of pixels for each x coordinate
          highest_peaks = ap.groupby('x', as_index = False)['y'].count().rename(columns = {'x' : 'x', 'y': 'nr_pixels'})
          # how many peaks larger than 'nr_pixels_peak' (=input)
          nr_peaks = highest_peaks[highest_peaks['nr_pixels'] > nr_pixels_peak].shape[0]
          print('There are {nr} peaks larger than {pix} pixels per peak'.format(nr=nr_peaks, pix = nr_pixels_peak))

          # calculate the amount of chipping (after inclusion of intermediary black surfaces)
          total_chip_pix = ap.shape[0]
          total_chip_mm2 = total_chip_pix / pixels_per_mm_x / pixels_per_mm_y
          print("Total Chipping [#p]:{chip}".format(chip=total_chip_pix))
          print("Total Chipping [mm2]:{chip}".format(chip=total_chip_mm2))




          ## Upload images & json result to Azure Storage

          # Connect to Azure Storage account
          # Define connection string
          connection_string = "DefaultEndpointsProtocol=https;AccountName=asrowadladhoc001;AccountKey=r59QAo/Ap2Nn15D7EMych2E3GMdd3kvNxVUvVLxXHli+CnEpswSc1yhspsAC/E7mBEHdzSUnMFaOjLZY2e1z2w==;EndpointSuffix=core.windows.net"
          
          # Define blob service client
          blob_service_client =  BlobServiceClient.from_connection_string(connection_string)

          # Specify folders for this productionorder
          main_container = "production-chipping"
          productionorder = snapid[:9] # Production order number are the first 9 characters from Snapid
          sub_container = main_container + '/' + productionorder
          sub_container_raw = sub_container + '/raw'
          sub_container_results = sub_container + '/results'


          ## RAW IMAGE
          # Generate raw image 
          # Convert raw image base64 string to bytes and then to image file
          imgdata = base64.b64decode(image_string)
          imageStream_raw = io.BytesIO(imgdata)
          # define raw image
          imageRaw = PILImage.open(imageStream_raw)

          # Upload raw image to Azure storage
          blob_client_raw = blob_service_client.get_blob_client(container=sub_container_raw, blob=snapid+'.jpg')
          blob_client_raw.upload_blob(imageStream_raw.read(), blob_type="BlockBlob", overwrite=True) 

          ## RAW IMAGE test
          # Generate raw image 
          # Convert raw image base64 string to bytes and then to image file
          imgdata = base64.b64decode(image_string)
          imageStream_raw = io.BytesIO(imgdata)
          # define raw image
          imageRaw = PILImage.open(imageStream_raw)
          imageRaw.save(imageStream_raw,'png')
          # reset stream's position to 0
          imageStream_raw.seek(0)

          # Upload raw image to Azure storage
          blob_client_raw = blob_service_client.get_blob_client(container=sub_container_raw, blob=snapid+'.jpg')
          blob_client_raw.upload_blob(imageStream_raw.read(), blob_type="BlockBlob", overwrite=True) 


          ## RESULT IMAGE
          # Generate image with mask
          # Create mask image with intermediary surfaces and paste on image
          mask_inferred_image = PILImage.fromarray(np.uint8(100*mask_inferred), mode = 'L') # change factor of 'mask_inferred' for changing opacity
          red_im = PILImage.new("RGBA", mask_image.size, color = 'red') # Create red image that will be pasted on chipping image to indicate the mask
          image_chipping = imageRaw # Open previously saved chipping image
          image_chipping = image_chipping.convert('RGB') # convert to RGB scale to plot colored mask om image
          image_chipping.paste(im = red_im, box = (0,0), mask = mask_inferred_image) # Paste colored mask on image

          imageStream_results = io.BytesIO()
          image_chipping.save(imageStream_results,'png')
          # reset stream's position to 0
          imageStream_results.seek(0)

          # Upload image with mask to Azure storage
          blob_client_results = blob_service_client.get_blob_client(container=sub_container_results, blob=snapid+'.jpg')
          blob_client_results.upload_blob(imageStream_results.read(), blob_type="BlockBlob", overwrite=True) 


          ## RESULT JSON
          result = {'Snapid': snapid,
                        'Ritszaag': ritszaag,
                        'Chipping limietwaarde': chippinglimietwaarde,
                        'Bovenpapier id': bovenpapierid,
                        'Plaatnummer in stapel': plaatnummerinstapel,
                        'Chipping resultaat mm²': total_chip_mm2,
                        'Chipping fouten': nr_peaks
                  }
          result_json = json.dumps(result)

          # Upload JSON with results to Azure storage
          blob_client_results_json = blob_service_client.get_blob_client(container=sub_container_results, blob=snapid+'.json')
          blob_client_results_json.upload_blob(result_json, overwrite=True) 


    return message, total_chip_pix, total_chip_mm2, nr_peaks

