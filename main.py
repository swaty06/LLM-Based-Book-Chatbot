import streamlit as st
import importlib
import os
from PIL import Image
from streamlit_option_menu import option_menu
from utils import apply_background, custom_navbar
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

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #666;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Social links */
    .social-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .social-link {
        display: inline-block;
        padding: 0.8rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-decoration: none;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .social-link:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        color: white;
        text-decoration: none;
    }
    
    /* About section */
    .about-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
    }
    
    .about-title {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .about-text {
        font-size: 1.1rem;
        color: #555;
        line-height: 1.8;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📚 Book Info Hub</div>
        <div class="hero-subtitle">Your AI-Powered Literary Companion</div>
    </div>
""", unsafe_allow_html=True)

# Features Section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">Intelligent Book Chatbot</div>
            <div class="feature-description">
                Ask any book-related questions and receive insightful, AI-powered responses 
                tailored to your literary curiosity. Powered by Google Gemini's advanced language models.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📖</div>
            <div class="feature-title">Comprehensive Knowledge Base</div>
            <div class="feature-description">
                Access a rich database of book information, curated from carefully selected 
                sources to provide accurate and contextually relevant answers.
            </div>
        </div>
    """, unsafe_allow_html=True)

# Custom divider
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Connect Section
st.markdown("""
    <div style="text-align: center;">
        <h2 style="color: #333; margin-bottom: 1rem;">Connect With Me</h2>
        <p style="color: #666; font-size: 1.1rem;">Learn more about this project and explore my other work</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="social-links">
        <a href="https://www.linkedin.com/in/swathy-ramakrishnan/" target="_blank" class="social-link">
            💼 LinkedIn
        </a>
        <a href="https://github.com/swaty06" target="_blank" class="social-link">
            💻 GitHub
        </a>
    </div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
    <div class="about-section">
        <div class="about-title">About This Project</div>
        <div class="about-text">
            Welcome to the Book-Based Chatbot! This innovative application leverages 
            <strong>Google Gemini's</strong> advanced language models to create intelligent chatbots 
            that answer questions based on a curated CSV database of book information. 
            <br><br>
            The system combines natural language processing with vector databases to provide 
            accurate, contextually relevant responses to your literary queries. Whether you're 
            looking for book recommendations, summaries, or specific information, our AI assistant 
            is here to help.
            <br><br>
            <strong>🚀 Stay tuned for exciting updates and new features!</strong>
        </div>
    </div>
""", unsafe_allow_html=True)

# Page setup
show_pages(
    [
        Page(str(root_path.joinpath("main.py")), "Home", "🏠"),
        Page(str(pages_path.joinpath("page1.py")), "BookBot", "🌿")
    ]
)

add_page_title(layout="wide")
custom_navbar()
apply_background()
