
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image


##functions#
@st.cache_resource
def prediction(modelname, sample_image, IMG_SIZE = (224,224)):

    #labels
    labels = ['High squamous intra-epithelial lesion','Low squamous intra-epithelial lesion','Negative for Intraepithelial malignancy','Squamous cell carcinoma']
    labels.sort()

    try:
        #loading the .h5 model
        load_model = tf.keras.models.load_model(modelname)

        sample_image = Image.open(sample_image)
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
#st.image(
    "https://thoneh.my/wp-content/uploads/2021/11/retina-detachment1-scaled-e1637129967126.jpg", 
    caption = "Cancer cell classification")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Cancer Cell Predictions")
    st.write("My app is designed to predict and classify cancer cell images into one of the following categories :High squamous intra-epithelial lesion,Low squamous intra-epithelial lesion,Negative for Intraepithelial malignancy,Squamous cell carcinoma")

#setting file uploader
#you can change the label name as your preference
image = st.file_uploader(label="Upload a cancer cell image",accept_multiple_files=False, help="Upload an image to classify them")

if image:
     #validating the image type
    image_type = image.type.split("/")[-1]
    if image_type not in ['jpg','jpeg','png','jfif']:
        st.error("Invalid file type : {}".format(image.type), icon="🚨")
    else:
        #displaying the image
        st.image(image, caption = "Uploaded Image")

        #getting the predictions
        label = prediction("Ishita_best_model_mobilenetV2.h5", image)

        #displaying the predicted label
        st.subheader("Prediction  **{}**".format(label))


            
