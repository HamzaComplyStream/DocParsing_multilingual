import streamlit as st
import json
import base64
import time
import pandas as pd
from datetime import datetime
import os
import io
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Import our custom compatibility client instead of Anthropic directly
from compatible_client import ClaudeClient

# Load environment variables
load_dotenv()

# Constants
MODEL_ID = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 100000
BATCH_SIZE = 250000
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB in bytes

# Get Claude API key
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', st.secrets.get("CLAUDE_API_KEY", ""))

# Page configuration
st.set_page_config(
    page_title="ComplyStream Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ComplyStream logo
LOGO_URL = "https://www.complystream.com/wp-content/uploads/2023/05/COMPLY_STREAM.png"
LOGO_DATA = "data:image/png;base64,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"

# Custom CSS to match ComplyStream theme
st.markdown("""
<style>
    /* Global styles */
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #FFFFFF;
        color: #333333;
    }
    
    /* Main container styling */
    .main {
        background-color: white;
    }
    
    .block-container {
        padding-top: 1rem;
    }
    
    /* Header and text styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: #0A1D3F;
        font-weight: 600;
    }
    
    /* Document section styling */
    .documents-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .documents-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0A1D3F;
    }
    
    /* Upload container styling */
    .upload-container {
        background-color: white;
        border-radius: 8px;
        padding: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        max-width: 550px;
        margin: 0 auto;
    }
    
    .header {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 24px;
        color: #0A1D3F;
    }
    
    /* Empty state styling */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px 0;
        background-color: #F7F8FA;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .empty-state-icon {
        margin-bottom: 20px;
        color: #6B7280;
    }
    
    .empty-state-text {
        color: #6B7280;
        font-size: 16px;
        text-align: center;
    }
    
    /* Button styling */
    .primary-button {
        background-color: #0A1D3F;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .primary-button:hover {
        background-color: #1A336B;
    }
    
    /* Fields styling */
    .field-label {
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
        color: #374151;
        display: block;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 40px 20px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #F9FAFB;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #0A1D3F;
        background-color: #F0F4F8;
    }
    
    .upload-icon {
        color: #6B7280;
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    .upload-text {
        color: #6B7280;
        font-size: 14px;
    }
    
    .file-type-text {
        color: #9CA3AF;
        font-size: 12px;
        margin-top: 8px;
    }
    
    /* File size warning */
    .file-warning {
        color: #DC2626;
        font-size: 14px;
        margin-top: 5px;
    }
    
    /* Results styling */
    .results-container {
        background-color: white;
        border-radius: 8px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .section-header {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 16px;
        color: #0A1D3F;
        padding-bottom: 8px;
        border-bottom: 1px solid #E5E7EB;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom sidebar styling */
    .sidebar .sidebar-content {
        background-color: white;
    }
    
    /* Logo in sidebar */
    .logo-container {
        padding: 20px 0;
        display: flex;
        justify-content: center;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        height: 38px;
        transition: all 0.3s ease;
    }
    
    /* Primary button */
    .stButton.primary>button {
        background-color: #0A1D3F;
        color: white;
        border: none;
    }
    
    .stButton.primary>button:hover {
        background-color: #1A336B;
    }
    
    /* Secondary button */
    .stButton.secondary>button {
        background-color: white;
        color: #374151;
        border: 1px solid #D1D5DB;
    }
    
    .stButton.secondary>button:hover {
        background-color: #F7F8FA;
    }
    
    /* Upload modal button container */
    .modal-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    
    /* Document type dropdown */
    .stSelectbox label {
        font-weight: 500;
        color: #374151;
    }
    
    /* Validation badges */
    .badge-pass {
        display: inline-block;
        padding: 4px 10px;
        background-color: #D1FAE5;
        color: #065F46;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .badge-fail {
        display: inline-block;
        padding: 4px 10px;
        background-color: #FEE2E2;
        color: #991B1B;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# PDF extraction functions
def extract_text_from_pdf(pdf_data: bytes):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    try:
        # Open the PDF from memory
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
        # Extract text from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        
        # Check if we got minimal text - might be a scanned document
        if len(text.strip()) < 100:
            # Try OCR for potential scanned document
            text = perform_ocr(pdf_data)
            
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        raise

def perform_ocr(pdf_data: bytes):
    """Perform OCR on PDF using Tesseract"""
    text = ""
    temp_pdf_path = None
    pdf_document = None
    
    try:
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name
            temp_pdf.write(pdf_data)
        
        # Open PDF document for conversion
        pdf_document = fitz.open(temp_pdf_path)
        
        # Process each page
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document.load_page(page_num)
            
            # Convert page to image
            pix = page.get_pixmap(alpha=False)
            img_data = pix.tobytes("png")
            
            # Use PIL to open the image
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error performing OCR: {str(e)}")
        raise
    finally:
        # Ensure resources are properly closed
        if pdf_document:
            pdf_document.close()
        
        # Try to remove the temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                pass

# Function to process document
def process_document_data(file_data, filename):
    """Process document data and return analysis results"""
    try:
        start_time = time.time()
        
        # Extract text from the document
        document_text = extract_text_from_pdf(file_data)
        
        # Process the document using our compatibility client
        processor = ClaudeClient(CLAUDE_API_KEY)
        result = processor.process_document_in_batches(document_text)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        result["metadata"] = {
            "filename": filename,
            "processing_time_seconds": round(processing_time, 2),
            "processed_at": datetime.now().isoformat(),
            "text_length": len(document_text)
        }
        
        return result
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        raise

# Sidebar with ComplyStream logo and navigation
with st.sidebar:
    st.markdown(f'<div class="logo-container"><img src="{LOGO_DATA}" width="180"></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation menu
    st.markdown("### Navigation")
    st.markdown("- Overview")
    st.markdown("- Entity")
    st.markdown("- Ownership")
    st.markdown("- Directors")
    st.markdown("- Financial Activity")
    st.markdown("- Review")
    st.markdown("- **Documents**")
    st.markdown("- Activity")
    st.markdown("- Comments")
    
    st.markdown("---")
    
    if not CLAUDE_API_KEY:
        st.warning("Claude API key is not set. Please configure it to enable document analysis.")

# Main page content
# Document header with upload button
st.markdown('<div class="documents-header">', unsafe_allow_html=True)
st.markdown('<div class="documents-title">Documents</div>', unsafe_allow_html=True)

# Create a placeholder for the "Upload files" button in the header
upload_button_placeholder = st.empty()

st.markdown('</div>', unsafe_allow_html=True)

# Check if we're in upload mode
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = False

# Toggle upload mode when the button is clicked
if upload_button_placeholder.button("Upload files", key="header_upload_button"):
    st.session_state.upload_mode = True

# Show upload modal when in upload mode
if st.session_state.upload_mode:
    # Upload modal
    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="header">Upload document</h1>', unsafe_allow_html=True)

        # Document Type dropdown
        st.markdown('<label class="field-label">Document Type</label>', unsafe_allow_html=True)
        doc_type = st.selectbox(
            'Document Type',
            ['Select from the list', 'Passport', 'Driver License', 'Utility Bill', 'Bank Statement',
             'Registration Certificate', 'Ownership Certificate', 'Tax Return', 'Financial Statement'],
            label_visibility="collapsed"
        )

        # Document upload
        st.markdown('<label class="field-label">Document</label>', unsafe_allow_html=True)

        # Create a container for the upload area
        upload_container = st.container()
        
        with upload_container:
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload PDF", 
                type=["pdf"],
                label_visibility="collapsed"
            )

            # Check file size
            if uploaded_file is not None:
                file_size = len(uploaded_file.getvalue())
                if file_size > MAX_FILE_SIZE:
                    st.markdown(f'<div class="file-warning">File exceeds the 20MB size limit. Current size: {file_size/1024/1024:.2f}MB</div>', unsafe_allow_html=True)
                    uploaded_file = None
                    
            # If no file is uploaded, show the dashed upload area
            if uploaded_file is None:
                st.markdown("""
                <div class="upload-area">
                    <div class="upload-icon">📄</div>
                    <div class="upload-text">Click to browse or drag and drop here</div>
                    <div class="file-type-text">Supported file types: .pdf (max 20MB)</div>
                </div>
                """, unsafe_allow_html=True)

        # Buttons row
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cancel", type="secondary", key="cancel_upload", use_container_width=True):
                st.session_state.upload_mode = False
                st.rerun()
                
        with col2:
            analyze_clicked = st.button("Upload", type="primary", key="upload_document", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process document when analyze is clicked
        if uploaded_file is not None and analyze_clicked:
            if not CLAUDE_API_KEY:
                st.error("Claude API key is required for document analysis")
            elif doc_type == 'Select from the list':
                st.error("Please select a Document Type")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        # Read the file data
                        file_data = uploaded_file.getvalue()
                        
                        # Process the document
                        result = process_document_data(file_data, uploaded_file.name)
                        
                        # Store result in session state
                        st.session_state.analysis_result = result
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.doc_type = doc_type
                        
                        # Exit upload mode and show results
                        st.session_state.upload_mode = False
                        st.session_state.show_results = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
else:
    # Show results or empty state
    if "analysis_result" in st.session_state and st.session_state.get("show_results", False):
        # Show results
        result = st.session_state.analysis_result
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="section-header">Document Analysis Results: {st.session_state.document_name}</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Document Type:** {st.session_state.doc_type}")
        
        with col2:
            st.markdown(f"**Processed On:** {st.session_state.analysis_time}")
            processing_time = result.get("metadata", {}).get("processing_time_seconds", 0)
            st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
        
        # Summary section
        st.markdown('<h3 class="section-header">Summary</h3>', unsafe_allow_html=True)
        
        if "analysis" in result and "summary" in result["analysis"]:
            summary_data = result["analysis"]["summary"]
            if isinstance(summary_data, dict) and "value" in summary_data:
                st.markdown(f"{summary_data['value']}")
            else:
                st.markdown(f"{summary_data}")
        else:
            st.info("No summary information available")
        
        # Document details
        st.markdown('<h3 class="section-header">Document Details</h3>', unsafe_allow_html=True)
        
        if "analysis" in result and "document_details" in result["analysis"]:
            doc_details = result["analysis"]["document_details"]
            if isinstance(doc_details, dict) and "value" in doc_details:
                if isinstance(doc_details["value"], dict):
                    # Create two columns for better layout
                    col1, col2 = st.columns(2)
                    keys = list(doc_details["value"].keys())
                    half = len(keys) // 2
                    
                    with col1:
                        for key in keys[:half]:
                            st.markdown(f"**{key.replace('_', ' ').title()}**: {doc_details['value'][key]}")
                    
                    with col2:
                        for key in keys[half:]:
                            st.markdown(f"**{key.replace('_', ' ').title()}**: {doc_details['value'][key]}")
                else:
                    st.markdown(doc_details["value"])
            else:
                st.json(doc_details)
        else:
            st.info("No detailed document information available")
        
        # Validation Checks
        st.markdown('<h3 class="section-header">Validation Checks</h3>', unsafe_allow_html=True)
        
        if "analysis" in result and "validation" in result["analysis"]:
            validation = result["analysis"]["validation"]
            if isinstance(validation, dict):
                # Filter out the confidence score
                check_items =
