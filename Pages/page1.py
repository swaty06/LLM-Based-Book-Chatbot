import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from utils import apply_background, custom_navbar
from st_pages import add_page_title
from pathlib import Path

root_path = Path(__file__).parent.parent
media_path = root_path.joinpath("media")

def app():
    # Enhanced custom CSS
    st.markdown("""
        <style>
        /* Main styling */
        .main {
            padding: 2rem;
        }
        
        /* Hero header */
        .book-header {
            background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
            padding: 2.5rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(46, 204, 113, 0.3);
        }
        
        .book-header h1 {
            font-size: 2.8rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .book-header p {
            font-size: 1.2rem;
            opacity: 0.95;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
            color: white;
            font-size: 1.1rem;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            padding: 0.8rem 2rem;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(46, 204, 113, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(46, 204, 113, 0.4);
            background: linear-gradient(135deg, #27AE60 0%, #229954 100%);
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            font-size: 1.1rem;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: border-color 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #2ECC71;
            box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.2);
        }
        
        /* Answer box styling */
        .answer-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 12px;
            border-left: 5px solid #2ECC71;
            margin-top: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        .answer-box h2 {
            color: #2ECC71;
            margin-bottom: 1rem;
            font-size: 1.8rem;
        }
        
        .answer-box p {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #333;
        }
        
        /* Info box */
        .info-box {
            background: #e8f5e9;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #2ECC71;
            margin-bottom: 2rem;
        }
        
        .info-box p {
            margin: 0;
            color: #1b5e20;
            font-size: 1rem;
        }
        
        /* KB Section styling */
        .kb-section {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .kb-section h3 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .kb-section p {
            color: #666;
            margin-bottom: 1.5rem;
        }
        
        /* Question section */
        .question-section {
            margin-top: 2rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Spinner customization */
        .stSpinner > div {
            border-top-color: #2ECC71 !important;
        }
        
        /* Success message */
        .element-container:has(.stSuccess) {
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Question input label */
        .stTextInput > label {
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
        <div class="book-header">
            <h1>📚 Book Bot 🌱</h1>
            <p>Your Personal AI Assistant for All Book-Related Queries</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            <p>
                💡 <strong>Getting Started:</strong> First, initialize the knowledge base by clicking the button below. 
                Then, feel free to ask any questions about books!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Apply the background image
    apply_background(image_path="media/book4.jpg")
    
    # Knowledge Base Section - FIXED
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="kb-section">
                <h3>🛠️ Initialize Knowledge Base</h3>
                <p>Click the button below to build the AI's knowledge database</p>
            </div>
        """, unsafe_allow_html=True)
        
        btn = st.button("🚀 Create Knowledgebase", key="create_kb")
    
    if btn:
        with st.spinner("🔄 Building the knowledge base... Please wait."):
            try:
                create_vector_db()
                st.success("✅ Knowledge base created successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"❌ An error occurred while creating the knowledge base: {e}")
    
    # Question input section
    st.markdown("---")
    st.markdown('<div class="section-title">💬 Ask Your Question</div>', unsafe_allow_html=True)
    
    question = st.text_input(
        "Type your book-related question here:",
        placeholder="e.g., What are the main themes in '1984'?",
        label_visibility="collapsed"
    )
    
    # Process the user's question
    if question:
        with st.spinner("🔍 Searching for the answer..."):
            try:
                chain = get_qa_chain()
                response = chain({"query": question})
                
                # Display the answer in a beautiful box
                st.markdown(f"""
                    <div class="answer-box">
                        <h2>💡 Answer</h2>
                        <p>{response["result"]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.info("💡 Tip: Make sure you've created the knowledge base first!")

custom_navbar()
apply_background()
add_page_title(layout="wide")
app()
