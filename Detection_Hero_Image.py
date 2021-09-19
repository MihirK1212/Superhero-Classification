import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.utils import generic_utils
from tensorflow.keras import models
from tensorflow.keras import layers
import cv2
import numpy as np
import matplotlib as plt
from os import system
import warnings  
warnings.filterwarnings("ignore")

Classes=["BlackWidow","Cap","DStrange","Hulk","IronMan","Loki","SpiderMan","Thanos"]

model=keras.models.load_model('hero_predictor.h5')


cls = lambda: system('cls')
cls()
path=os.path.join('Test')
for file_name in os.listdir(path):
    sample_test_img=cv2.imread(os.path.join(path,file_name))
    final_sample_test_img=cv2.resize(sample_test_img,(48,48))
    final_sample_test_img=np.expand_dims(final_sample_test_img,axis=0) 
    final_sample_test_img=final_sample_test_img/255.0

    Predictions=model.predict(final_sample_test_img)
    label=np.argmax(Predictions,axis=1)[0]
    prob=str((Predictions[0][label])*100)
    class_label=Classes[label]
    print(file_name,class_label)

 
