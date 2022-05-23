import streamlit as st
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from tempfile import NamedTemporaryFile

global model

#@st.cache(allow_output_mutation=True)

# def load_model():
#   model=tf.keras.models.load_model('final_model.h5')
#   return model

# with st.spinner('Model is being loaded..'):
#     model=load_model()

st.write("""
          # Fashion Classification
          """
          )

file = st.file_uploader("Please upload the photo shooting the cloths", type=["jpg", "png"])

def run_example():
    # load the image
    img = load_image(temp_file.name)
    # load model
    model = load_model('final_model.h5')
    # predict the class
    #result = model.predict_classes(img)
    predict_x=model.predict(img) 
    classes_x=np.argmax(predict_x,axis=1)
    dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'}
    return dict[classes_x[0]]

def load_image(filename):   
    # load the image
    img = load_img(filename, target_size=(28, 28), color_mode = "grayscale")
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


if file is None:
    st.text("Please upload an image file")
else:

    image = Image.open(file)
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(file.getvalue())
    st.image(image, use_column_width=True)
    st.title("Tag: "+ run_example())
    st.caption("Based on the inputted "+  run_example() + ", here are some more similar product in shop" )
    images = ['10_diff_pic/CMU12G860I-FI.jpeg', '10_diff_pic/download.jpeg', '10_diff_pic/hmgoepprod.jpeg']
    st.image(images, width= 100)
