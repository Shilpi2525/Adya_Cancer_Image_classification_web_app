
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

#CONSTANTS
MODEL_NAME = "mobilenetv2_ep_30_lr_0.001.h5"
LABELS = ['High squamous intra-epithelial lesion','Low squamous intra-epithelial lesion','Negative for Intraepithelial malignancy','Squamous cell carcinoma']
IMAGE_URL = "https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=HSIL%20of%20the%20Cervix%2C%20Nuclear-to-Cytoplasmic%20Ratio&p=BOOKS&id=430728_HSIL01.jpg"

##functions#
@st.cache_resource
def prediction(modelname, sample_image, IMG_SIZE = (224,224)):

   #sort the labels
    LABELS.sort()

    try:
        #loading the .h5 model
        load_model = tf.keras.models.load_model(modelname)

        sample_image = Image.open(sample_image).convert('RGB') #ensuring to convert into RGB as model expects the image to be in 3 channel
        img_array = sample_image.resize(IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis = 0)
        image_batch = img_batch.astype(np.float32)
        image_batch = preprocess_input(image_batch)
        prediction = load_model.predict(img_batch)
        return labels[int(np.argmax(prediction, axis = 1))]


    except Exception as e:
        st.write("ERROR: {}".format(str(e)))


#Building the website

#title of the web page
st.title("Cancer Cell Image Classifictaion")

#setting the main picture
st.image(IMAGE_URL, caption = "Cancer Cell Classification"

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Cancer Cell Predictions")
    st.write("""My app is designed to predict and classify cancer cell images into one of the following categories :
    1.High squamous intra-epithelial lesion
    2.Low squamous intra-epithelial lesion
    3.Negative for Intraepithelial malignancy
    4.Squamous cell carcinoma""")

#setting file uploader
image =st.file_uploader("Upload a cancer cell image",type = ['jpg','png','jpeg'])
if image:
    
    #displaying the image
    st.image(image, caption = "Uploaded Image")

    #get prediction
    label=prediction(MODEL_NAME,image)

    #displaying the predicted label
    st.subheader("Prediction:  **{}**".format(label))


            
