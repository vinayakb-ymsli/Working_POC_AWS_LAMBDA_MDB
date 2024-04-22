import urllib.parse
import boto3
import sys
import numpy as np
# import matplotlib.pyplot as plt
# from glob import glob
# from tqdm import tqdm
from tifffile import imread
# from csbdeep.utils import Path, normalize
import tensorflow as tf
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D
import random 
import cv2
import os

os.chdir("/tmp/")



s3_client = boto3.client('s3')

bucket_name = 'cicdlambdanew'
output_folder = 'OutputImages/'
model_file_name = 'mymodel_Dec13_keras_new_dataset.keras'
prefix = 'Model_files'
model_key = f'{prefix}/{model_file_name}'


# Define the local path where the model file will be downloaded
local_model_path = '/tmp/' + model_file_name


def download_model_from_s3(bucket_name, key, local_path):
    try:
        s3_client.download_file(bucket_name, key, local_path)
        print(f"Model file downloaded successfully from S3: {local_path}")
        return True
    except Exception as e:
        s3_file_exists = lambda prefix: bool(list(bucket_name.objects.filter(Prefix=prefix)))
        print("this is test s3_file_exists",s3_file_exists)
        print(f"Error downloading model file from S3: {e}")
        return False
                    


def encode_image(image):
    _, encoded_image = cv2.imencode('.jpg', image)
    return encoded_image.tobytes()
    

def upload_image_to_s3(image_data, bucket, outputkey):
    try:
        s3_client.put_object(Body=image_data, Bucket=bucket, Key=outputkey)
        print(f"Image uploaded to s3://{bucket}/{outputkey}")
        return True
    except Exception as e:
        print(f"Error uploading image to S3: {e}")
        return False

#lambda_handler start
def lambda_handler(event, context):
    # Check if the event is an S3 event
    print("this is event",event)
    if 'Records' in event and len(event['Records']) > 0 and 's3' in event['Records'][0]:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        print("this is key",key)
        image_name = key.split("/")[-1]
        print("this is image_name",image_name)
        
        # Check if the uploaded object is in the validationimages folder
        if key.startswith('ValidationImages/'):
            # Process the uploaded image here
            print("line no 67")
            np.random.seed(42)
            image_path = '/tmp/'+image_name
            s3_client.download_file(bucket_name, key, image_path)
            # lbl_cmap = random_label_cmap()
            
            def imreadReshape(key):
                if ".tif" in image_name:
                    imageRead = imread(image_path)
                    if np.ndim(imageRead) == 2:
                        return imageRead
                    imageRead = np.array(imageRead)
                    imageRead = cv2.resize(imageRead,(768,768))
                    return imageRead[:,:,0]
                else:
                    print("line no 80")
                    imageRead = cv2.imread(image_path)
                    print("line no 82")
                    if np.ndim(imageRead) == 2:
                        return imageRead
                    print("line no 85")
                    print("Image shape before resizing:", imageRead.shape)  # Add this line for debugging
                    #imageRead = np.array(imageRead)
                    imageRead = cv2.resize(imageRead,(768,768))
                    print("line no 88")
                    return imageRead[:,:,0]
            
            print("I am readreshape function complete")
            X_val = [image_name]
            X_val = list(map(imreadReshape,X_val))
            n_channel = 1 if X_val[0].ndim == 2 else X_val[0].shape[-1]  #If no third dim. then number of channels = 1. Otherwise get the num channels from the last dim.
            axis_norm = (0,1)  
            if n_channel > 1:
                print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
                sys.stdout.flush()
            
            X_val = [x/255 for x in X_val]
            rng = np.random.RandomState(42)
            
            print(Config2D.__doc__)
            gputools_available()
            
            n_rays = 32 #ok
            use_gpu = True and gputools_available() #ok
            
            grid = (2,2) # ok
            
            conf = Config2D (
                n_rays       = n_rays,
                grid         = grid,
                use_gpu      = use_gpu,
                n_channel_in = n_channel,
                train_patch_size = (768,768)
            )
            print("this is prefix below")
            # prefix_exists(bucket_name, prefix)
            if download_model_from_s3(bucket_name, model_key, local_model_path):
            ## Load the model
                new_model = tf.keras.models.load_model(local_model_path,compile=False)
            print("Load Model Complete")
            # Serialize the 'conf' object and write it to a file in the /tmp directory
            import pickle
            with open('/tmp/conf.pkl', 'wb') as f:
                pickle.dump(conf, f)

            # Load the serialized configuration from the file
            with open('/tmp/conf.pkl', 'rb') as f:
                loaded_conf = pickle.load(f)

            # Now you can use 'loaded_conf' as your configuration object
            model_loaded = StarDist2D(loaded_conf)
            ## new_model = tf.keras.models.load_model(r'C:\Users\ve00ym767\Downloads\mymodel_Dec13_keras_new_dataset.keras')
            model_loaded.keras_model = new_model
            model_loaded.thresholds = {"nms":0.3,"prob":0.5590740124765305}
            prediction_second_list = [model_loaded.predict_instances(x, n_tiles=model_loaded._guess_n_tiles(x), show_tile_progress=False)[0]
                          for x in X_val]
            print("this is prediction_second_list",prediction_second_list)
            for idx, image in enumerate(prediction_second_list):
                # Encode the image to bytes
                encoded_image = encode_image(image)
    
                # Construct the key for the S3 object
                outputkey = f"{output_folder}processed_image_{image_name}"
                
                # Upload the encoded image to S3
                if upload_image_to_s3(encoded_image, bucket_name, outputkey):
                    print(f"Processed image {image_name} uploaded successfully")
                else:
                    print(f"Failed to upload processed image {image_name}")
            # os.makedirs("ResultInferenceawstesting",exist_ok=True)
            
        else:
            print(f"Image uploaded to a different folder: {outputkey}")
    else:
        print("Unsupported event format or not an S3 event")