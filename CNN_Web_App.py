import streamlit as st
from PIL import Image
from CNN_CLF import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Assignment 2 simple Web App")
st.write("")

image_input = st.file_uploader("Upload your image", type="jpg")

if image_input is not None:
    image = Image.open(image_input)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Loading...")
    labels = predict(image_input)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])