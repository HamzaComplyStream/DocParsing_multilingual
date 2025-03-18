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
import asyncio
from anthropic import Anthropic
from dotenv import load_dotenv

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
    page_title="Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the provided UI
st.markdown("""
<style>
    /* Modal-like appearance */
    .upload-container {
        background-color: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 550px;
        margin: 0 auto;
    }
    
    /* Main header */
    .header {
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 30px;
        color: #1F2937;
    }
    
    /* Label styles */
    .field-label {
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 10px;
        color: #1F2937;
        display: block;
    }
    
    /* Dropdown styling */
    .stSelectbox>div>div {
        border-radius: 8px !important;
        border: 1px solid #D1D5DB !important;
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
        font-size: 16px;
    }
    
    .file-type-text {
        color: #9CA3AF;
        font-size: 14px;
        margin-top: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 16px;
        font-size: 16px;
    }
    
    .cancel-button {
        background-color: white;
        color: #1F2937;
        border: 1px solid #D1D5DB;
    }
    
    .upload-button {
        background-color: #14213D;
        color: white;
    }
    
    /* Analysis results section */
    .results-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .section-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
        color: #1F2937;
    }
    
    /* File size warning */
    .file-warning {
        color: #B91C1C;
        font-size: 14px;
        margin-top: 5px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fix container width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Two-column layout for buttons */
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    
    .button-container .stButton {
        width: 48%;
    }
</style>
""", unsafe_allow_html=True)

# Document processor class (simplified to use synchronous Anthropic client)
class DocumentProcessor:
    def __init__(self, api_key):
        self.today = datetime.today().strftime('%Y-%m-%d')
        self.client = Anthropic(api_key=api_key)
    
    def classify_document(self, document_text: str):
        """Classify the document type using Claude API"""
        try:
            user_prompt = user_prompt_classification(document_text)
            
            response = self.client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                system=system_prompt_for_doc_classification,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract JSON from response
            result_text = response.content[0].text
            # Clean up the result (sometimes Claude returns extra text)
            if '{' in result_text and '}' in result_text:
                json_str = result_text[result_text.find('{'):result_text.rfind('}')+1]
                result = json.loads(json_str)
                return result
            else:
                return {"error": "Could not parse classification result", "raw_response": result_text}
        except Exception as e:
            st.error(f"Error classifying document: {str(e)}")
            raise
    
    def process_document(self, document_text: str, document_class: int):
        """Process the document based on its classification using Claude API"""
        # Map document class to the appropriate prompt functions and system prompts
        prompt_map = {
            0: (user_prompt_poi, system_prompt_for_indentity_doc),
            1: (user_prompt_poa, system_prompt_for_poa_doc),
            2: (user_prompt_registration, system_prompt_for_registration_doc),
            3: (user_prompt_ownership, system_prompt_for_ownership_doc),
            4: (user_prompt_tax_return, system_prompt_for_tax_return_doc),
            5: (user_prompt_financial, system_prompt_for_financial_doc),
        }
        
        try:
            # Get the appropriate prompt function and system prompt
            user_prompt_func, system_prompt = prompt_map.get(document_class, 
                                                            (user_prompt_financial, system_prompt_for_financial_doc))
            
            # Generate the user prompt
            user_prompt = user_prompt_func(document_text, self.today)
            
            response = self.client.messages.create(
                model=MODEL_ID,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract JSON from response
            result_text = response.content[0].text
            try:
                # Try to extract JSON if Claude added extra text
                if '{' in result_text and '}' in result_text:
                    json_str = result_text[result_text.find('{'):result_text.rfind('}')+1]
                    result = json.loads(json_str)
                    return result
                else:
                    return {"raw_response": result_text}
            except json.JSONDecodeError:
                return {"raw_response": result_text}
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            raise

    def process_document_in_batches(self, document_text: str):
        """Process large documents by splitting into batches"""
        if len(document_text) <= BATCH_SIZE:
            # If document is small enough, classify and process it directly
            classification = self.classify_document(document_text)
            document_class = classification.get("class", 5)  # Default to financial if classification fails
            result = self.process_document(document_text, document_class)
            return {
                "classification": classification,
                "analysis": result
            }
        else:
            # For large documents, first try to classify using a sample
            sample_size = min(BATCH_SIZE, len(document_text))
            # Take beginning, middle and end portions to create a representative sample
            start_portion = document_text[:sample_size//3]
            middle_start = max(0, (len(document_text) // 2) - (sample_size//6))
            middle_portion = document_text[middle_start:middle_start + (sample_size//3)]
            end_start = max(0, len(document_text) - (sample_size//3))
            end_portion = document_text[end_start:]
            sample_text = start_portion + "\n\n" + middle_portion + "\n\n" + end_portion
            
            # Classify using the sample
            classification = self.classify_document(sample_text)
            document_class = classification.get("class", 5)  # Default to financial if classification fails
            
            # Use first batch for detailed analysis
            analysis_text = document_text[:BATCH_SIZE]
            result = self.process_document(analysis_text, document_class)
            
            return {
                "classification": classification,
                "analysis": result,
                "note": "Document was processed using a representative sample due to size."
            }

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
        
        # Process the document
        processor = DocumentProcessor(CLAUDE_API_KEY)
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

# Display document upload UI in a modal-like container
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

# Create a file uploader that matches the design
uploaded_file = st.file_uploader(
    "Upload PDF", 
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Check file size
    file_size = len(uploaded_file.getvalue())
    if file_size > MAX_FILE_SIZE:
        st.markdown(f'<div class="file-warning">File exceeds the 20MB size limit. Current size: {file_size/1024/1024:.2f}MB</div>', unsafe_allow_html=True)
        uploaded_file = None

# Custom buttons in a two-column layout
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    cancel_button = st.button("Cancel", type="secondary", use_container_width=True)

with col2:
    analyze_button = st.button("Upload", type="primary", use_container_width=True)
    
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # Close upload container

# Process document when analyze is clicked
if uploaded_file is not None and analyze_button:
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
                
                # Display success message
                st.success("Document uploaded and analyzed successfully!")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Results Section (displayed after analysis)
if "analysis_result" in st.session_state:
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
