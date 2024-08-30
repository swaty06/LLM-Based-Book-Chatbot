import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from utils import apply_background,custom_navbar
from st_pages import add_page_title
from pathlib import Path

root_path = Path(__file__).parent.parent
media_path = root_path.joinpath("media")

def app():

    # Set the page configuration
    st.set_page_config(
        page_title="Book Bot 🌱",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # Add a custom CSS style for better visuals
    st.markdown(
        """
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            height: 50px;
            width: 100%;
        }
        .stTextInput input {
            font-size: 18px;
        }
        .stHeader, .stWrite {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Title and introductory message
    st.title("Book Bot 📚🌱")
    st.markdown("""
    Welcome to **Book Bot**, your personal assistant for book-related queries. 
    Click the button below to initialize the knowledge base, then ask any question you have about books!
    """)
    # Apply the background image
    apply_background(image_path="./app/media/book4.jpg")
    # Button to create a knowledge base
    btn = st.button("Create Knowledgebase 🛠️")
    
    if btn:
        with st.spinner("Building the knowledge base... Please wait."):
            # Simulate some processing time
            create_vector_db()
            st.success("Knowledge base created successfully!")
    
    # Input box for user questions
    question = st.text_input("Ask your question below:", placeholder="What is the best book for learning Python?")
    
    # Process the user's question
    if question:
        with st.spinner("Fetching the answer..."):
            chain = get_qa_chain()
            response = chain(question)
    
            # Display the answer
            st.header("Answer")
            st.write(response["result"])

set_custom_bg()
custom_navbar()
apply_background()
add_page_title(layout="wide")
app()


            
        
        

