import streamlit as st
import importlib
import os

from PIL import Image
from streamlit_option_menu import option_menu
from utils import apply_background,custom_navbar
from st_pages import Page, show_pages, add_page_title
from pathlib import Path

root_path = Path(__file__).parent
media_path = root_path.joinpath("media")
pages_path = root_path.joinpath('Pages')

# Set page configuration
st.set_page_config(
    page_title="Book Info Hub",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main content (previously in the sidebar)
st.subheader("Tools Available")
st.write("""

 **Book Chatbot:** Ask any book-related questions, and get insightful responses from our AI-powered assistant.
""")
# Add links to book resources or related pages
st.markdown("""
For more information and about the project, check out my profile:

- [LinkedIn](https://www.linkedin.com/in/swathy-ramakrishnan/)
- [GitHub](https://github.com/swaty06)
""")

# Footer
st.markdown("---")
st.markdown(
    """
    **About:**
    Welcome to the Book-Based Chatbot!  I developed chatbots using Google Gemini, which are designed to answer questions based on a predefined set of questions and answers stored in a CSV file. 
    These chatbots leverage the capabilities of Google Gemini's advanced language models to provide accurate and contextually relevant responses. Stay tuned for more updates!
    """
)

show_pages(
    [
        Page(str(root_path.joinpath("main.py")), "Home", "🏠"),
       
        Page(str(pages_path.joinpath("page1.py")), "BookBot", "🌿")
       
    ]
)

add_page_title(layout="wide")

custom_navbar()
apply_background()
