import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras import backend as K
import os
import time
import io
from PIL import Image
import plotly.express as px

MODELSPATH = './models/'
DATAPATH = './data/'




title_container = st.container() 
col1, col2 = st.columns([1, 5]) 
image = Image.open('logo.jpg') 
with title_container: 
    with col1: st.image(image, width=100) 
    with col2: st.markdown('<h1 style="color: white;">Veersa DL Labs</h1>', unsafe_allow_html=True)








# Lung Cancer 

@st.cache
def data_gen_lc(x):
    img = np.asarray(Image.open(x).resize((256, 256)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 256, 256, 3)

    return x_validate


@st.cache
def data_gen_Lc(img):
    img = img.reshape(256, 256)
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1,256, 256, 3)

    return x_validate




def load_models_lc():
    model = load_model(MODELSPATH + 'VGG16finalized_model.h5')
    return model


@st.cache
def predict_lc(x_test, model):
    Y_pred = model.predict(x_test)
    # predict_prob=model.predict(x_test)
    # predict_classes=np.argmax(predict_prob,axis=1)
    ynew = model.predict(x_test)
    K.clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew*100
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return y_new,Y_pred_classes


@st.cache
def display_prediction_lc(y_new):
    """Display image and preditions from model"""
    result = pd.DataFrame({'Probability': y_new}, index=np.arange(3))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability Percentage']
    lung_cancer_type = {2: 'Squamous Cell Carcinoma', 1: 'Normal', 0: 'Adenocarcinoma'}
    result["Classes"] = result["Classes"].map(lung_cancer_type)
    return result


# Skin Cancer 


def load_models_sc():
    model1 = load_model(MODELSPATH + 'model.h5')
    return model1

@st.cache
def data_gen_sc(x):
    img = np.asarray(Image.open(x).resize((100, 75)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)
    return x_validate

@st.cache
def data_gen_Sc(img):
    img = img.reshape(100, 75)
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate

@st.cache
def predict_sc(x_test, model):
    Y_pred = model.predict(x_test)
    # predict_prob=model.predict(x_test)
    # predict_classes=np.argmax(predict_prob,axis=1)
    ynew = model.predict(x_test)
    K.clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew*100
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return y_new,Y_pred_classes

def display_prediction_sc(y_new):
    """Display image and preditions from model"""

    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {2: 'Benign keratosis-like lesions', 4: 'Melanocytic nevi', 3: 'Dermatofibroma',
                        5: 'Melanoma', 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result






def main():
    st.sidebar.header('Cancer Analyzer')
    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Upload Your Image for Lung Cancer","Upload Your Image for Skin Cancer"])
    st.sidebar.write('''This model predicts 2 types of cancer :\n 1. Skin Cancer: \n It detects 7 different classes of skin cancer which are following :  
    \n Melanocytic nevi , Melanoma, Benign keratosis-like lesions, Basal cell carcinoma, Actinic keratoses, Vascular lesions, Dermatofibroma''')

    st.sidebar.write(''' 2. Lung Cancer : \n It detects 3 types of Lung Cancer which are following :
    \n Squamous Cell Carcinoma, Normal , Adenocarcinoma ''')

    # st.sidebar.write('1. Skin Cancer \n 2. Lung Cancer')

    if page == "Upload Your Image for Skin Cancer":

        st.header("Upload Your Image for Skin Cancer")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])

        if file_path is not None:
            x_test = data_gen_sc(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)

            st.success('File Upload Success!!')
        else:
            st.info('Please upload Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            st.subheader("Choose Training Algorithm!")
            if st.checkbox('Keras'):
                model = load_models_sc()
                st.success("Hooray !! Keras Model Loaded!")
                if st.checkbox('Show Prediction Probablity for Uploaded Image'):
                    y_new, Y_pred_classes = predict_sc(x_test, model)
                    result = display_prediction_sc(y_new)
                    st.write(result)
                    if st.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                        st.plotly_chart(fig, use_container_width=True)

    if page == "Upload Your Image for Lung Cancer":

        st.header("Upload Your Image")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg','jpeg'])

        if file_path is not None:
            x_test = data_gen_lc(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)

            st.success('File Upload Success!!')
        else:
            st.info('Please upload Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            st.subheader("Choose Training Algorithm!")
            if st.checkbox('Keras'):
                model = load_models_lc()
                st.success("Keras Model Loaded!")
                if st.checkbox('Show Prediction Probablity for Uploaded Image'):
                    y_new, Y_pred_classes = predict_lc(x_test, model)
                    result = display_prediction_lc(y_new)
                    st.write(result)
                    if st.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability Percentage", color='Classes')
                        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
