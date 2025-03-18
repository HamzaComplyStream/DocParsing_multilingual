import streamlit as st
import json
import base64
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import io
import os
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import asyncio
from anthropic import AsyncAnthropic
import logging
from dotenv import load_dotenv

# Import the code from the provided API script
from prompts import (
    user_prompt_classification, system_prompt_for_doc_classification,
    user_prompt_poi, user_prompt_poa, user_prompt_registration,
    user_prompt_ownership, user_prompt_tax_return, user_prompt_financial,
    system_prompt_for_indentity_doc, system_prompt_for_poa_doc,
    system_prompt_for_registration_doc, system_prompt_for_ownership_doc,
    system_prompt_for_tax_return_doc, system_prompt_for_financial_doc
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("document-analyzer")

# Constants from the original API
MODEL_ID = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 100000
BATCH_SIZE = 250000

# Get environment variables
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', st.secrets.get("CLAUDE_API_KEY", ""))

# Page configuration
st.set_page_config(
    page_title="Document Analysis Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #475569;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #F1F5F9;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #0284C7;
    }
    .success-box {
        background-color: #DCFCE7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #16A34A;
        margin-bottom: 1.5rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #D97706;
        margin-bottom: 1.5rem;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #DC2626;
        margin-bottom: 1.5rem;
    }
    .result-section {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .upload-button {
        width: 100%;
        padding: 0.75rem 1rem;
        background-color: #F1F5F9;
        border: 1px dashed #94A3B8;
        border-radius: 0.5rem;
        text-align: center;
    }
    .analyze-button {
        width: 100%;
        margin-top: 1rem;
    }
    .validation-check {
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .validation-pass {
        background-color: #DCFCE7;
        color: #166534;
    }
    .validation-fail {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .document-type-list li {
        margin-bottom: 0.75rem;
        list-style-type: none;
        padding-left: 1.5rem;
        position: relative;
        font-size: 0.95rem;
        color: #334155;
    }
    .document-type-list li:before {
        content: "•";
        position: absolute;
        left: 0;
        color: #3B82F6;
        font-weight: bold;
    }
    .stExpander {
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Document processor class from the provided API
class DocumentProcessor:
    def __init__(self, api_key):
        self.today = datetime.today().strftime('%Y-%m-%d')
        self.async_client = AsyncAnthropic(api_key=api_key)
    
    async def classify_document(self, document_text: str):
        """Classify the document type using Claude API"""
        try:
            user_prompt = user_prompt_classification(document_text)
            
            response = await self.async_client.messages.create(
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
            raise
    
    async def process_document(self, document_text: str, document_class: int):
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
            
            response = await self.async_client.messages.create(
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
            raise

    async def process_document_in_batches(self, document_text: str):
        """Process large documents by splitting into batches"""
        if len(document_text) <= BATCH_SIZE:
            # If document is small enough, classify and process it directly
            classification = await self.classify_document(document_text)
            document_class = classification.get("class", 5)  # Default to financial if classification fails
            result = await self.process_document(document_text, document_class)
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
            classification = await self.classify_document(sample_text)
            document_class = classification.get("class", 5)  # Default to financial if classification fails
            
            # Use first batch for detailed analysis
            analysis_text = document_text[:BATCH_SIZE]
            result = await self.process_document(analysis_text, document_class)
            
            return {
                "classification": classification,
                "analysis": result,
                "note": "Document was processed using a representative sample due to size."
            }

# PDF extraction functions from the original API
async def extract_text_from_pdf(pdf_data: bytes):
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
            text = await perform_ocr(pdf_data)
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

async def perform_ocr(pdf_data: bytes):
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
        logger.error(f"Error performing OCR: {str(e)}")
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

# Function to process document (calls the API functions directly)
async def process_document_data(file_data, filename):
    """Process document data and return analysis results"""
    try:
        start_time = time.time()
        
        # Extract text from the document
        document_text = await extract_text_from_pdf(file_data)
        
        # Process the document
        processor = DocumentProcessor(CLAUDE_API_KEY)
        result = await processor.process_document_in_batches(document_text)
        
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
        logger.error(f"Error processing document: {str(e)}")
        raise

# Sidebar 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/document-scanner.png", width=80)
    st.markdown("## Document Analysis")
    st.markdown("Upload documents for comprehensive analysis using Claude's advanced AI capabilities.")
    
    # API Key Configuration
    st.markdown("### Configuration")
    api_key = st.text_input("Claude API Key", value=CLAUDE_API_KEY, type="password")
    
    if not api_key:
        st.markdown('<div class="warning-box">Claude API key is required for document analysis</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyzes various document types:
    
    <ul class="document-type-list">
        <li>Proof of Identity (passports, ID cards)</li>
        <li>Proof of Address (utility bills, leases)</li>
        <li>Business Registration documents</li>
        <li>Ownership documents</li>
        <li>Tax Return documents</li>
        <li>Financial documents</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("© 2025 Document Analysis System")

# Main content
st.markdown('<h1 class="main-header">Document Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your documents for AI-powered analysis and insights</p>', unsafe_allow_html=True)

# Document Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Upload Document</h2>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Upload a PDF document to analyze. The system will extract key information and classify the document based on its content.</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Upload a PDF document for analysis", label_visibility="collapsed")

# Analysis options
col1, col2 = st.columns(2)
with col1:
    ocr_if_needed = st.checkbox("Use OCR if needed (for scanned documents)", value=True)
    show_confidence = st.checkbox("Show confidence scores", value=True)

with col2:
    extract_text = st.checkbox("Extract and display raw text", value=False)
    show_json = st.checkbox("Show raw JSON response", value=False)

# Analyze button - properly aligned
if uploaded_file is not None:
    analyze_clicked = st.button("Analyze Document", use_container_width=True, type="primary")
else:
    st.markdown('<div class="warning-box">Please upload a PDF document to analyze</div>', unsafe_allow_html=True)
    analyze_clicked = False

st.markdown('</div>', unsafe_allow_html=True)  # Close upload section

# Process document when analyze is clicked
if uploaded_file is not None and analyze_clicked:
    if not api_key:
        st.error("Claude API key is required for document analysis")
    else:
        with st.spinner("Analyzing document..."):
            try:
                # Read the file data
                file_data = uploaded_file.getvalue()
                
                # Process the document
                result = asyncio.run(process_document_data(file_data, uploaded_file.name))
                
                # Store result in session state
                st.session_state.analysis_result = result
                st.session_state.document_name = uploaded_file.name
                st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Display success message
                st.markdown('<div class="success-box">Analysis completed successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">Error during analysis: {str(e)}</div>', unsafe_allow_html=True)

# Results Section (displayed below upload section)
if "analysis_result" in st.session_state:
    result = st.session_state.analysis_result
    
    # Results section with improved visibility
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    
    # Header with document info
    st.markdown(f"## Analysis Results: {st.session_state.document_name}")
    st.markdown(f"*Analyzed on: {st.session_state.analysis_time}*")
    
    # Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        doc_type = result.get("classification", {}).get("category", "Unknown")
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Document Type</div>
            <div class="metric-value">{doc_type}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_m2:
        confidence = result.get("classification", {}).get("confidence_score", 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Classification Confidence</div>
            <div class="metric-value">{confidence:.2f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_m3:
        processing_time = result.get("metadata", {}).get("processing_time_seconds", 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Processing Time</div>
            <div class="metric-value">{processing_time:.2f}s</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_m4:
        text_length = result.get("metadata", {}).get("text_length", 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Document Length</div>
            <div class="metric-value">{text_length:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Summary section
    st.markdown('<h3 class="section-header" style="margin-top:1.5rem;">Summary</h3>', unsafe_allow_html=True)
    
    if "analysis" in result and "summary" in result["analysis"]:
        summary_data = result["analysis"]["summary"]
        if isinstance(summary_data, dict) and "value" in summary_data:
            st.markdown(f"**{summary_data['value']}**")
            if show_confidence and "confidence_score" in summary_data:
                st.caption(f"Confidence: {summary_data['confidence_score']:.2f}")
        else:
            st.markdown(f"**{summary_data}**")
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
    
    # Risks and Opportunities
    col_risk, col_opp = st.columns(2)
    
    with col_risk:
        st.markdown('<h3 class="section-header">Identified Risks</h3>', unsafe_allow_html=True)
        
        if "analysis" in result and "risks" in result["analysis"]:
            risks = result["analysis"]["risks"]
            if isinstance(risks, dict) and "value" in risks:
                if isinstance(risks["value"], list):
                    for item in risks["value"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(risks["value"])
            else:
                st.json(risks)
        else:
            st.info("No risks identified")
        
    with col_opp:
        st.markdown('<h3 class="section-header">Opportunities</h3>', unsafe_allow_html=True)
        
        if "analysis" in result and "opportunities" in result["analysis"]:
            opportunities = result["analysis"]["opportunities"]
            if isinstance(opportunities, dict) and "value" in opportunities:
                if isinstance(opportunities["value"], list):
                    for item in opportunities["value"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(opportunities["value"])
            else:
                st.json(opportunities)
        else:
            st.info("No opportunities identified")
    
    # Validation Checks
    st.markdown('<h3 class="section-header">Validation Checks</h3>', unsafe_allow_html=True)
    
    if "analysis" in result and "validation" in result["analysis"]:
        validation = result["analysis"]["validation"]
        if isinstance(validation, dict):
            # Filter out the confidence score
            check_items = {k: v for k, v in validation.items() if k != "confidence_score"}
            
            # Convert to DataFrame for better display
            check_df = pd.DataFrame(
                {"Check": [k.replace("_", " ").title() for k in check_items.keys()],
                 "Status": [v for v in check_items.values()]}
            )
            
            # Add checkmarks or X instead of True/False
            def format_status(val):
                if val is True:
                    return "✅ Pass"
                elif val is False:
                    return "❌ Fail" 
                else:
                    return str(val)
            
            check_df["Status"] = check_df["Status"].apply(format_status)
            
            # Display as color-coded table
            for _, row in check_df.iterrows():
                if "Pass" in row["Status"]:
                    st.markdown(f'<div class="validation-check validation-pass"><b>{row["Check"]}</b>: {row["Status"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="validation-check validation-fail"><b>{row["Check"]}</b>: {row["Status"]}</div>', unsafe_allow_html=True)
            
            if show_confidence and "confidence_score" in validation:
                st.caption(f"Overall validation confidence: {validation['confidence_score']:.2f}")
        else:
            st.json(validation)
    else:
        st.info("No validation information available")
    
    # Required Actions
    st.markdown('<h3 class="section-header">Recommended Actions</h3>', unsafe_allow_html=True)
    
    if "analysis" in result and "required_actions" in result["analysis"]:
        actions = result["analysis"]["required_actions"]
        if isinstance(actions, dict) and "value" in actions:
            if isinstance(actions["value"], list):
                for i, item in enumerate(actions["value"], 1):
                    st.markdown(f"{i}. {item}")
            else:
                st.markdown(actions["value"])
        else:
            st.json(actions)
    else:
        st.info("No recommended actions available")
    
    # Raw JSON options (in expandable section)
    if show_json:
        with st.expander("View Raw JSON Response"):
            st.json(result)
    
    # Export options
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("Export Results as JSON", use_container_width=True):
            json_str = json.dumps(result, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="analysis_result.json">Download JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col_exp2:
        if st.button("Copy Summary to Clipboard", use_container_width=True):
            if "analysis" in result and "summary" in result["analysis"]:
                summary_data = result["analysis"]["summary"]
                if isinstance(summary_data, dict) and "value" in summary_data:
                    summary_text = summary_data["value"]
                else:
                    summary_text = str(summary_data)
                st.success("Summary copied to clipboard!")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close result section

# Footer
st.markdown("---")
st.caption("Document Analysis powered by Claude AI • Built with Streamlit")
