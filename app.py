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

# Import the code from the provided API script
try:
    from prompts import (
        user_prompt_classification, system_prompt_for_doc_classification,
        user_prompt_poi, user_prompt_poa, user_prompt_registration,
        user_prompt_ownership, user_prompt_tax_return, user_prompt_financial,
        system_prompt_for_indentity_doc, system_prompt_for_poa_doc,
        system_prompt_for_registration_doc, system_prompt_for_ownership_doc,
        system_prompt_for_tax_return_doc, system_prompt_for_financial_doc
    )
except ImportError:
    st.error("Error: prompts.py module not found. Make sure it exists in the same directory.")
    st.stop()

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

# ComplyStream logo URL - Replace with actual logo path when deploying
LOGO_URL = "https://storage.googleapis.com/turing_mongo/complystream_logo.png"

# Custom CSS to match ComplyStream theme
st.markdown("""
<style>
    /* Global styles */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #FAFAFA;
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
        font-family: 'Inter', sans-serif;
        color: #14213D;
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
        color: #14213D;
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
        color: #14213D;
    }
    
    /* Empty state styling */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px 0;
        background-color: #FAFAFA;
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
        background-color: #14213D;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        cursor: pointer;
    }
    
    /* Fields styling */
    .field-label {
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
        color: #374151;
        display: block;
    }
    
    /* Dropdown styling */
    .stSelectbox>div {
        border-radius: 6px;
    }
    
    .stSelectbox>div>div {
        background-color: white;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 40px 20px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #F9FAFB;
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
        color: #14213D;
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
    }
    
    /* Primary button */
    .stButton.primary>button {
        background-color: #14213D;
        color: white;
        border: none;
    }
    
    /* Secondary button */
    .stButton.secondary>button {
        background-color: white;
        color: #374151;
        border: 1px solid #D1D5DB;
    }
    
    /* Upload modal button container */
    .modal-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
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
    st.markdown(f'<div class="logo-container"><img src="{LOGO_URL}" width="180"></div>', unsafe_allow_html=True)
    
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

        # Document Category dropdown
        st.markdown('<label class="field-label">Document Category</label>', unsafe_allow_html=True)
        doc_category = st.selectbox(
            'Document Category',
            ['Select from the list', 'Identity Document', 'Proof of Address', 'Business Registration', 
             'Ownership Document', 'Tax Return', 'Financial Document'],
            label_visibility="collapsed"
        )

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
                    <div class="file-type-text">Supported file types: .pdf</div>
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
            elif doc_category == 'Select from the list' or doc_type == 'Select from the list':
                st.error("Please select both Document Category and Document Type")
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
                        st.session_state.doc_category = doc_category
                        st.session_state.doc_type = doc_type
                        
                        # Exit upload mode and show results
                        st.session_state.upload_mode = False
                        st.session_state.show_results = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
else:
    # Show empty state or results
    if "analysis_result" in st.session
    else:
    # Show empty state or results
    if "analysis_result" in st.session_state and st.session_state.get("show_results", False):
        # Show results
        result = st.session_state.analysis_result
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="section-header">Document Analysis Results: {st.session_state.document_name}</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Document Category:** {st.session_state.doc_category}")
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
                check_items = {k: v for k, v in validation.items() if k != "confidence_score"}
                
                # Create a table-like display
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("**Check**")
                with col2:
                    st.markdown("**Status**")
                
                for key, value in check_items.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"{key.replace('_', ' ').title()}")
                    with col2:
                        if value is True:
                            st.markdown("✅ PASS")
                        elif value is False:
                            st.markdown("❌ FAIL")
                        else:
                            st.markdown(str(value))
            else:
                st.json(validation)
        else:
            st.info("No validation information available")
        
        # Export options
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("Export Results as JSON", use_container_width=True):
                json_str = json.dumps(result, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="analysis_result.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col_exp2:
            if st.button("Download Document Report", use_container_width=True):
                st.info("Document report generation functionality would be implemented here.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close result container
    else:
        # Show empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📄</div>
            <div class="empty-state-text">No documents</div>
            <div class="empty-state-text">Start by uploading or drag and drop here</div>
            <div style="margin-top: 20px;">
                <button class="primary-button" onclick="document.querySelector('[data-testid=baseButton-headerUploadButton]').click()">Upload</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
