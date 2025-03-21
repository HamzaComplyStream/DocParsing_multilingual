import os
import json
import base64
import logging
import time
import streamlit as st
from typing import Dict, Any, Optional
from pydantic import BaseModel
from anthropic import Anthropic
import fitz  # PyMuPDF for PDF processing
from datetime import datetime
import pytesseract
from PIL import Image
import io
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
    logger.warning("CLAUDE_API_KEY environment variable not set. You'll need to provide it in the app.")

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
    page_title="Document Analysis Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #4B5563;
        margin-bottom: 40px;
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
    .result-container {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid #E5E7EB;
    }
    .info-text {
        font-size: 16px;
        color: #4B5563;
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
        # No proxies parameter used here
    
    def classify_document(self, document_text: str) -> Dict[str, Any]:
        """Classify the document type using Claude API"""
        try:
            st.text("Classifying document type...")
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
            
            # Generate the user prompt
            user_prompt = user_prompt_func(document_text, self.today)
            
            st.text(f"Analyzing document details (class {document_class})...")
            
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
                logger.warning("Could not parse response as JSON, returning raw text")
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

def format_validation_results(validation):
    """Format validation results with colored badges"""
    if not isinstance(validation, dict):
        st.write(validation)
        return
    
    # Filter out the confidence score
    check_items = {k: v for k, v in validation.items() if k != "confidence_score"}
    
    # Create a table-like display with colored badges
    for key, value in check_items.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{key.replace('_', ' ').title()}**")
        with col2:
            if value is True:
                st.markdown('<span class="badge badge-green">✓ PASS</span>', unsafe_allow_html=True)
            elif value is False:
                st.markdown('<span class="badge badge-red">✗ FAIL</span>', unsafe_allow_html=True)
            else:
                st.write(str(value))
    
    # Show confidence score if available
    if "confidence_score" in validation:
        st.write(f"**Overall Confidence Score:** {validation['confidence_score']}%")

def main():
    # Page title
    st.markdown('<div class="title">Document Analysis Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a document for AI-powered analysis</div>', unsafe_allow_html=True)
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Claude API Key", value=CLAUDE_API_KEY or "", type="password")
        st.caption("Your API key is required to use Claude's document analysis capabilities")
        
        st.divider()
        
        st.subheader("About")
        st.markdown("""
        This demo showcases document analysis using Claude AI.
        
        **Supported Document Types:**
        - Identity Documents
        - Proof of Address
        - Business Registration
        - Ownership Documents
        - Tax Returns
        - Financial Documents
        
        **Limitations:**
        - PDF files only
        - Max file size: 20MB
        - Processing may take some time depending on document complexity
        """)
    
    # Main content area - File upload section
    st.markdown('<div class="section-header">Upload Document</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">Select a PDF document to analyze. The system will extract text, classify the document type, and provide a detailed analysis.</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="document_uploader", help="Maximum file size is 20MB")
    
    # Check if file is uploaded
    if uploaded_file is not None:
        # Check file size
        file_size = len(uploaded_file.getvalue())
        if file_size > MAX_FILE_SIZE:
            st.markdown(f'<div class="error-message">File exceeds the 20MB size limit. Current size: {file_size/1024/1024:.2f}MB</div>', unsafe_allow_html=True)
        else:
            # API key validation
            if not api_key:
                st.markdown('<div class="error-message">Claude API key is required to process documents. Please enter it in the sidebar.</div>', unsafe_allow_html=True)
            else:
                # Show file info
                st.markdown(f"**File:** {uploaded_file.name} ({file_size/1024/1024:.2f}MB)")
                
                # Analyze button
                if st.button("Analyze Document", type="primary"):
                    with st.spinner("Processing document..."):
                        try:
                            start_time = time.time()
                            
                            # Read file content
                            file_data = uploaded_file.getvalue()
                            
                            # Extract text from the document
                            st.text("Extracting text from document...")
                            document_text = extract_text_from_pdf(file_data)
                            
                            # Process the document
                            processor = DocumentProcessor(api_key)
                            result = processor.process_document_in_batches(document_text)
                            
                            # Add processing metadata
                            processing_time = time.time() - start_time
                            result["metadata"] = {
                                "filename": uploaded_file.name,
                                "processing_time_seconds": round(processing_time, 2),
                                "processed_at": datetime.now().isoformat(),
                                "text_length": len(document_text)
                            }
                            
                            # Display results
                            st.markdown(f'<div class="success-message">Document analyzed successfully in {result["metadata"]["processing_time_seconds"]:.2f} seconds</div>', unsafe_allow_html=True)
                            
                            # Classification result
                            st.markdown('<div class="section-header">Document Classification</div>', unsafe_allow_html=True)
                            
                            classification = result.get("classification", {})
                            if "class" in classification:
                                doc_class = classification["class"]
                                doc_type = display_document_type(doc_class)
                                
                                if "confidence" in classification:
                                    st.write(f"**Confidence:** {classification['confidence']:.2f}%")
                                
                                if "class_description" in classification:
                                    st.write(f"**Description:** {classification['class_description']}")
                            else:
                                st.write("Could not determine document class")
                            
                            # Document details
                            st.markdown('<div class="section-header">Document Analysis</div>', unsafe_allow_html=True)
                            
                            analysis = result.get("analysis", {})
                            
                            # Check if analysis contains a summary
                            if "summary" in analysis:
                                st.markdown("### Summary")
                                summary_data = analysis["summary"]
                                if isinstance(summary_data, dict) and "value" in summary_data:
                                    st.write(summary_data["value"])
                                else:
                                    st.write(summary_data)
                            
                            # Document details
                            if "document_details" in analysis:
                                st.markdown("### Document Details")
                                doc_details = analysis["document_details"]
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
                                        st.write(doc_details["value"])
                                else:
                                    st.json(doc_details)
                            
                            # Validation results
                            if "validation" in analysis:
                                st.markdown("### Validation Checks")
                                format_validation_results(analysis["validation"])
                            
                            # Export options
                            st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
                            
                            # Convert the result to JSON string
                            json_str = json.dumps(result, indent=2)
                            
                            # Create a download button for the JSON
                            st.download_button(
                                label="Download JSON Results",
                                data=json_str,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analysis.json",
                                mime="application/json",
                                key="download_json"
                            )
                            
                        except Exception as e:
                            st.markdown(f'<div class="error-message">Error processing document: {str(e)}</div>', unsafe_allow_html=True)
                            logger.error(f"Error processing document: {str(e)}")
    else:
        # Show empty state
        st.markdown("""
        <div class="upload-container">
            <div style="font-size: 48px; margin-bottom: 20px;">📄</div>
            <div style="font-size: 18px; color: #4B5563; margin-bottom: 10px;">Drag and drop a PDF file here</div>
            <div style="font-size: 14px; color: #6B7280;">or click to browse</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
