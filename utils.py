# utils.py
import base64
import streamlit as st
from pathlib import Path


root_path = Path(__file__).parent.parent
media_path = root_path.joinpath("media")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def apply_background(image_path="/media/book4.jpg"):
    """
    Apply a background image to the Streamlit app.
    
    Args:
    - image_path (str): The file path to the local image you want to use as a background.
    """
    # Encode the image to base64
    background_base64 = get_base64_image(image_path)

    # Define the CSS for the background image
    background_image_css = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{background_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
    }}
    </style>
    '''
    
    # Inject the CSS into the Streamlit app
    st.markdown(background_image_css, unsafe_allow_html=True)
    # Customizing the navigation bar
def custom_navbar():
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #fdf2df;
    </style>
    """,
    unsafe_allow_html=True,
    )


