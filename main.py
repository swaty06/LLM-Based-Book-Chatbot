import streamlit as st
import importlib
import os

from PIL import Image
from streamlit_option_menu import option_menu
from utils import apply_background,custom_navbar
from st_pages import Page, add_page_title
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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Page 1", "Page 2"])

# Function to dynamically load and execute page modules
def load_page(page_name):
    try:
        module_name = f"Pages.{page_name.lower().replace(' ', '_')}"
        module = importlib.import_module(module_name)
        return module
    except ModuleNotFoundError:
        st.error(f" {page_name} not found.")
        return None

# Display the selected page with content
if page == "Home":
    st.title("Home")
    st.write("Welcome to the Home Page!")
else:
    page_module = load_page(page)
    if page_module:
        page_module.app()  # Call the app function from the module

# Main content (previously in the sidebar)
st.subheader("Tools Available")
st.write("""
1. **URL Summarization:** Provide a URL related to any book or literary content, and our tool will summarize the key points for you.
2. **Book Chatbot:** Ask any book-related questions, and get insightful responses from our AI-powered assistant.
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
    **About:** This application is designed to help you explore books, get recommendations, and find your next favorite read. Stay tuned for more updates!
    """
)


[[pages]]
path = "page1.py"
name = "st-pages"
icon = "📄"

add_page_title(layout="wide")
