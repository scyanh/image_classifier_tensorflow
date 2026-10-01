#!/usr/bin/env python3
'''
pip install -q -U "tensorflow-gpu==2.0.0b1"
pip install -q -U tensorflow_hub
'''
# Silensing import warnings
import warnings
warnings.filterwarnings('ignore')
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd 
import json
import sys
from PIL import Image
import os.path
import logging
# Silencing errors and warnings from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Argument parser
parser = argparse.ArgumentParser(
       description='Flower Classification app',
        )
parser.add_argument('--category_names'
                    , action = "store"
                    , dest = 'class_names'
                    , default = 'label_map.json'
                    , help = 'path to json file with class names'
                   )

parser.add_argument('--top_k'
                    , action = "store"
                    , dest = 'topk'
                    , type = int
                    , default = 5
                    , help = 'Number of top records to print'
                   )
# Loading and checking for image and model file presence
try:

	image = str(sys.argv[1])
	model = str(sys.argv[2])
except Exception as error1:
	print(error1)
	sys.exit()
if os.path.isfile(image) == False:
    print('Image does not exist')
    sys.exit()
if os.path.isfile(model) == False:
    print('Model does not exist')
    sys.exit()
# Getting additional parameters
commands = parser.parse_args(sys.argv[3:])
classes = commands.class_names
top_k = commands.topk
# Checking the json file exists
try:
    with open(classes, 'r') as f:
        class_names = json.load(f)
except Exception as error:
    print(error)
    sys.exit()
# Loading the model
loaded_model = tf.keras.models.load_model(model, custom_objects = {'KerasLayer': hub.KerasLayer})
#loaded_model = tf.keras.models.load_model('./flower.h5', custom_objects = {'KerasLayer': hub.KerasLayer})

#loaded_model = tf.keras.models.load_model('./f2model.h5')
#loaded_model = tf.keras.models.load_model('fmodel.h5')

# Functiones to process the image and predict the flower
def process_image(img):
    '''
    Converts the image format to the on needed by the model
    Returns an image
    '''
    img = np.squeeze(img)
    image = tf.image.resize(img, (224, 224))   
    image = (image/255)
    return image
    
def predict(image_path, model, top_k = 5):
    '''
    Loads the image and runs the model to predict the flower
    Returns a df with the predicted category and the probability
    '''
    # Loading the image and formating it
    im = Image.open(image_path)
    prediction_image = np.asarray(im)
    prediction_image = process_image(prediction_image)
    # Running the model 
    prediction = model.predict(np.expand_dims(prediction_image, 0))
    # Formating the results as a pandas df
    r_df = pd.DataFrame(prediction[0]).reset_index().rename(columns = {'index': 'class_code', 
                                            0: 'Probability'})\
                                            .sort_values(by='Probability', ascending = False)\
                                            .head(top_k).reset_index(drop=True)
    r_df.loc[:,'Probability'] = np.round(r_df.Probability * 100, 2)

    r_df.loc[:,'class_code'] = r_df.class_code + 1
    r_df.loc[:,'Flower'] = np.nan
    # Add the flower name based on the number/code
    for value in r_df['class_code'].values:
        class_name = class_names[str(value)]
        r_df.loc[:,'Flower'] = np.where(r_df.class_code==value, class_name.title(), r_df.Flower)
    return r_df[['Flower', 'Probability']]
# Running the functions with the parameters provided
results = predict(image, loaded_model, top_k)
# Printing the result
print(results)