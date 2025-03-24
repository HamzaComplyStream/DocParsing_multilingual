import os
import io
import json
import logging
import time
import streamlit as st
from typing import Dict, Any, Optional
from pydantic import BaseModel
from anthropic import Anthropic
import fitz  # PyMuPDF for PDF processing
from datetime import datetime
import pandas as pd
import pytesseract
from PIL import Image
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "claude-3-5-sonnet-20240620"  # Update with the latest Claude model
MAX_TOKENS = 100000
BATCH_SIZE = 250000
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB in bytes

# Get environment variables
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
if not CLAUDE_API_KEY:
    logger.warning("CLAUDE_API_KEY environment variable not set. Make sure it's set in your environment.")

# Import the prompts module
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
    logger.error("prompts.py module not found. Make sure it exists in the same directory.")
    st.error("prompts.py module not found. Make sure it exists in the same directory.")
    st.stop()

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Document Analysis Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #E5E7EB;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
    .success-message {
        padding: 15px;
        border-radius: 6px;
        background-color: #D1FAE5;
        color: #065F46;
        margin: 10px 0;
    }
    .error-message {
        padding: 15px;
        border-radius: 6px;
        background-color: #FEE2E2;
        color: #991B1B;
        margin: 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
    }
    .badge-blue {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .badge-green {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-red {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .badge-orange {
        background-color: #FFEDD5;
        color: #9A3412;
    }
    .upload-container {
        border: 2px dashed #CBD5E1;
        border-radius: 8px;
        padding: 40px 20px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

class DocumentProcessor:
    def __init__(self, api_key):
        self.today = datetime.today().strftime('%Y-%m-%d')
        self.client = Anthropic(api_key=api_key)
    
    def classify_document(self, document_text: str) -> Dict[str, Any]:
        """Classify the document type using Claude API"""
        try:
            with st.status("Classifying document type...", expanded=True) as status:
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
                    status.update(label="Document classification complete!", state="complete", expanded=False)
                    return result
                else:
                    status.update(label="Classification failed", state="error")
                    return {"error": "Could not parse classification result", "raw_response": result_text}
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            st.error(f"Error classifying document: {str(e)}")
            raise
    
    def process_document(self, document_text: str, document_class: int) -> Dict[str, Any]:
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
            
            with st.status(f"Analyzing document details (class {document_class})...", expanded=True) as status:
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
                        status.update(label="Document analysis complete!", state="complete", expanded=False)
                        return result
                    else:
                        status.update(label="Analysis parsing failed", state="error")
                        return {"raw_response": result_text}
                except json.JSONDecodeError:
                    logger.warning("Could not parse response as JSON, returning raw text")
                    status.update(label="JSON parsing failed", state="error")
                    return {"raw_response": result_text}
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error(f"Error processing document: {str(e)}")
            raise

    def process_document_in_batches(self, document_text: str) -> Dict[str, Any]:
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

def extract_text_from_pdf(pdf_data: bytes) -> str:
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
        logger.error(f"Error extracting text from PDF: {str(e)}")
        st.error(f"Error extracting text from PDF: {str(e)}")
        raise

def perform_ocr(pdf_data: bytes) -> str:
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
        
        # Status indicator for OCR
        st.text("Document appears to be scanned. Performing OCR...")
        progress_bar = st.progress(0)
        
        # Process each page
        for page_num in range(len(pdf_document)):
            # Update progress
            progress = (page_num + 1) / len(pdf_document)
            progress_bar.progress(progress)
            
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
        
        # Clear progress after completion
        progress_bar.empty()
        
        return text
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
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
                logger.warning(f"Could not remove temporary file {temp_pdf_path}: {str(e)}")

def display_document_type(doc_class):
    """Display document type with appropriate badge"""
    doc_types = {
        0: ("Identity Document", "badge-blue"),
        1: ("Proof of Address", "badge-green"),
        2: ("Business Registration", "badge-orange"),
        3: ("Ownership Document", "badge-blue"),
        4: ("Tax Return", "badge-red"),
        5: ("Financial Document", "badge-green")
    }
    
    name, badge_class = doc_types.get(doc_class, ("Unknown Document", "badge-blue"))
    st.markdown(f'<span class="badge {badge_class}">{name}</span>', unsafe_allow_html=True)
    return name

def display_results(result):
    """
    Display the analysis results in a structured format
    
    Args:
        result: Analysis results dictionary
    """
    if not result:
        return
    
    st.markdown('<div class="sub-header">📄 Document Analysis Results</div>', unsafe_allow_html=True)
    
    # Document type from classification
    classification = result.get("classification", {})
    if "class" in classification:
        doc_class = classification["class"]
        doc_type = display_document_type(doc_class)
        
        if "confidence" in classification:
            st.write(f"**Confidence:** {classification.get('confidence', 0):.2f}%")
        
        if "class_description" in classification:
            st.write(f"**Description:** {classification['class_description']}")
    
    # Analysis data
    analysis = result.get("analysis", {})
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Summary", "Detailed Analysis"])
    
    with tab1:
        # Summary section
        if "summary" in analysis:
            st.markdown("### Key Points")
            summary_data = analysis["summary"]
            if isinstance(summary_data, dict) and "value" in summary_data:
                st.markdown(summary_data["value"])
            else:
                st.markdown(str(summary_data))
        
        # Validation results if available
        if "validation" in analysis:
            st.markdown("### Document Validity")
            validation = analysis["validation"]
            
            if isinstance(validation, dict):
                # Show confidence score if available
                if "confidence_score" in validation:
                    st.markdown(f"**Confidence Score:** {validation['confidence_score']}%")
                
                # Filter out the confidence score for other checks
                check_items = {k: v for k, v in validation.items() if k != "confidence_score"}
                
                # Display validation checks in a more compact format
                for key, value in check_items.items():
                    if value is True:
                        st.markdown(f"✅ **{key.replace('_', ' ').title()}**: Pass")
                    elif value is False:
                        st.markdown(f"❌ **{key.replace('_', ' ').title()}**: Fail")
                    else:
                        st.markdown(f"ℹ️ **{key.replace('_', ' ').title()}**: {value}")
    
    with tab2:
        # Document details
        if "document_details" in analysis:
            st.markdown("### Document Details")
            doc_details = analysis["document_details"]
            
            if isinstance(doc_details, dict) and "value" in doc_details:
                if isinstance(doc_details["value"], dict):
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(list(doc_details["value"].items()), columns=["Field", "Value"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.markdown(str(doc_details["value"]))
            else:
                st.json(doc_details)
        
        # Display other sections
        for key, value in analysis.items():
            if key not in ["summary", "validation", "document_details"]:
                st.markdown(f"### {key.replace('_', ' ').title()}")
                
                if isinstance(value, dict):
                    if "value" in value and isinstance(value["value"], dict):
                        # Create two columns for better layout of key-value pairs
                        col1, col2 = st.columns(2)
                        keys = list(value["value"].keys())
                        half = len(keys) // 2
                        
                        with col1:
                            for k in keys[:half]:
                                st.markdown(f"**{k.replace('_', ' ').title()}**: {value['value'][k]}")
                        
                        with col2:
                            for k in keys[half:]:
                                st.markdown(f"**{k.replace('_', ' ').title()}**: {value['value'][k]}")
                    else:
                        st.json(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")
                        else:
                            st.text(item)
                else:
                    st.text(str(value))
    
    # Add metadata if available
    if "metadata" in result:
        with st.expander("Processing Metadata", expanded=False):
            st.markdown(f"**Filename:** {result['metadata'].get('filename', 'Unknown')}")
            st.markdown(f"**Processing Time:** {result['metadata'].get('processing_time_seconds', 0):.2f} seconds")
            st.markdown(f"**Processed At:** {result['metadata'].get('processed_at', 'Unknown')}")
            st.markdown(f"**Text Length:** {result['metadata'].get('text_length', 0)} characters")
    
    # Download results button
    st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
    
    # Convert the result to JSON string
    json_str = json.dumps(result, indent=2)
    
    # Create a download button for the JSON
    st.download_button(
        label="Download Analysis (JSON)",
        data=json_str,
        file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="download_json"
    )

def process_document(uploaded_file):
    """Process the uploaded document and return analysis results"""
    try:
        st.session_state.processing = True
        
        start_time = time.time()
        
        # Read file content
        file_data = uploaded_file.getvalue()
        
        # Extract text from the document
        with st.status("Extracting text from document...", expanded=True) as status:
            document_text = extract_text_from_pdf(file_data)
            
            # Show text preview
            with st.expander("Preview extracted text", expanded=False):
                st.text_area("Extracted Text", document_text[:10000] + ("..." if len(document_text) > 10000 else ""), height=200)
            
            status.update(label="Text extraction complete!", state="complete", expanded=False)
        
        # Process the document
        processor = DocumentProcessor(CLAUDE_API_KEY)
        result = processor.process_document_in_batches(document_text)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        result["metadata"] = {
            "filename": uploaded_file.name,
            "processing_time_seconds": round(processing_time, 2),
            "processed_at": datetime.now().isoformat(),
            "text_length": len(document_text)
        }
        
        st.session_state.processing = False
        return result
        
    except Exception as e:
        st.markdown(f'<div class="error-message">Error processing document: {str(e)}</div>', unsafe_allow_html=True)
        logger.error(f"Error processing document: {str(e)}")
        st.session_state.processing = False
        return None

def main():
    # Page title
    st.markdown('<div class="main-header">Document Analysis Tool</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Upload a PDF document to generate an automated summary and analysis. This tool identifies document type and extracts key information.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool analyzes various types of documents:
        
        - Proof of Identity Documents
        - Proof of Address Documents
        - Registration Documents
        - Ownership Documents
        - Tax Returns
        - Financial Documents
        
        The analysis includes document classification, key information extraction, and validity assessment.
        
        **Note:** The tool attempts to process scanned documents with OCR, but results may vary based on scan quality.
        """)
    
    # Main content area - File upload section
    st.markdown('<div class="section-header">Upload Document</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="document_uploader", help="Maximum file size is 20MB")
    
    # Check if file is uploaded
    if uploaded_file is not None:
        # Check file size
        file_size = len(uploaded_file.getvalue())
        if file_size > MAX_FILE_SIZE:
            st.markdown(f'<div class="error-message">File exceeds the 20MB size limit. Current size: {file_size/1024/1024:.2f}MB</div>', unsafe_allow_html=True)
        else:
            # Show file info
            st.markdown(f"**File:** {uploaded_file.name} ({file_size/1024/1024:.2f}MB)")
            
            # API key check
            if not CLAUDE_API_KEY:
                st.markdown('<div class="error-message">Claude API key is not set in environment. Please set the CLAUDE_API_KEY environment variable.</div>', unsafe_allow_html=True)
            else:
                # Analyze button
                if st.button("Analyze Document", type="primary"):
                    with st.spinner("Processing document..."):
                        st.session_state.results = process_document(uploaded_file)
    else:
        # Show empty state
        st.markdown("""
        <div class="upload-container">
            <div style="font-size: 48px; margin-bottom: 20px;">📄</div>
            <div style="font-size: 18px; color: #4B5563; margin-bottom: 10px;">Drag and drop a PDF file here</div>
            <div style="font-size: 14px; color: #6B7280;">or click to browse</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results if available
    if st.session_state.results:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        display_results(st.session_state.results)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display processing message
    if st.session_state.processing:
        st.info("Document analysis in progress...")

if __name__ == "__main__":
    main()
