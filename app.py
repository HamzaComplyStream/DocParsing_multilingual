import streamlit as st
import requests
import json
import base64
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = st.secrets["url"]
CLAUDE_API_KEY = st.secrets["claude_api_key"]
# Page configuration
st.set_page_config(
    page_title="Document Analysis Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FFC107;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F44336;
    }
    .result-section {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: 1px solid #E0E0E0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/document-scanner.png", width=80)
    st.markdown("## Document Analysis")
    st.markdown("Upload your documents for comprehensive analysis using Claude's advanced AI capabilities.")
    
    # API Configuration
    st.markdown("### API Configuration")
    api_url = st.text_input("API URL", value=API_URL)
    
    # API Key input with proper masking
    api_key = st.text_input("Claude API Key", value=CLAUDE_API_KEY, type="password")
    
    # Health check
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{api_url}/")
            if response.status_code == 200:
                st.markdown('<div class="success-box">API is running ✅</div>', unsafe_allow_html=True)
                st.json(response.json())
            else:
                st.markdown('<div class="error-box">API is not responding properly ❌</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">Error connecting to API: {str(e)} ❌</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard connects to the Document Analysis API for processing various document types.
    
    - Extract key information
    - Classify documents
    - Generate insights
    """)
    st.markdown("---")
    st.markdown("© 2025 Document Analysis System")

# Main content
st.markdown('<h1 class="main-header">Document Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your documents for AI-powered analysis and insights</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📄 Document Upload", "🔍 Analysis Results"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Document")
        st.markdown('<div class="info-box">Upload a PDF document to analyze. The system will extract key information and classify the document based on its content.</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        # Options
        st.markdown("### Analysis Options")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            analyze_immediately = st.checkbox("Analyze immediately after upload", value=True)
            show_raw_text = st.checkbox("Extract and display raw text", value=False)
        
        with col_opt2:
            ocr_if_needed = st.checkbox("Use OCR if needed (for scanned documents)", value=True)
            display_confidence = st.checkbox("Show confidence scores", value=True)
    
    with col2:
        st.markdown("### Document Types")
        st.markdown("""
        The system can identify and analyze these document types:
        
        - **Proof of Identity** (passports, ID cards)
        - **Proof of Address** (utility bills, lease agreements)
        - **Business Registration** documents
        - **Ownership** documents
        - **Tax Return** documents
        - **Financial** documents
        """)
        
        st.markdown("### Processing Status")
        if uploaded_file is None:
            st.markdown('<div class="warning-box">No document uploaded yet</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">Document "{uploaded_file.name}" uploaded ✅</div>', unsafe_allow_html=True)
            
            if analyze_immediately:
                with st.spinner("Analyzing document..."):
                    try:
                        # Prepare the file for upload
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {}
                        
                        if api_key:
                            data["api_key"] = api_key
                        
                        # Make the API request
                        response = requests.post(f"{api_url}/analyze", files=files, data=data)
                        
                        if response.status_code == 200:
                            # Store the result in session state for display in the results tab
                            st.session_state.analysis_result = response.json()
                            st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.document_name = uploaded_file.name
                            
                            st.markdown('<div class="success-box">Analysis completed successfully! View the results in the "Analysis Results" tab.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-box">Error: {response.status_code} - {response.text}</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.markdown(f'<div class="error-box">Error during analysis: {str(e)}</div>', unsafe_allow_html=True)
            else:
                if st.button("Analyze Document"):
                    with st.spinner("Analyzing document..."):
                        try:
                            # Prepare the file for upload
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                            data = {}
                            
                            if api_key:
                                data["api_key"] = api_key
                            
                            # Make the API request
                            response = requests.post(f"{api_url}/analyze", files=files, data=data)
                            
                            if response.status_code == 200:
                                # Store the result in session state for display in the results tab
                                st.session_state.analysis_result = response.json()
                                st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.document_name = uploaded_file.name
                                
                                st.markdown('<div class="success-box">Analysis completed successfully! View the results in the "Analysis Results" tab.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="error-box">Error: {response.status_code} - {response.text}</div>', unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.markdown(f'<div class="error-box">Error during analysis: {str(e)}</div>', unsafe_allow_html=True)
                            
        if show_raw_text and uploaded_file is not None:
            st.markdown("### Extracted Text Preview")
            st.info("This functionality would require additional backend integration to extract text separately.")

with tab2:
    if "analysis_result" not in st.session_state:
        st.markdown('<div class="warning-box">No analysis results yet. Please upload and analyze a document first.</div>', unsafe_allow_html=True)
    else:
        result = st.session_state.analysis_result
        
        # Header with document info
        st.markdown(f"## Analysis Results: {st.session_state.document_name}")
        st.markdown(f"*Analyzed on: {st.session_state.analysis_time}*")
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            doc_type = result.get("classification", {}).get("category", "Unknown")
            st.markdown(f'<div class="metric-card"><h3>Document Type</h3><h2>{doc_type}</h2></div>', unsafe_allow_html=True)
        
        with col_m2:
            confidence = result.get("classification", {}).get("confidence_score", 0)
            st.markdown(f'<div class="metric-card"><h3>Classification Confidence</h3><h2>{confidence:.2f}</h2></div>', unsafe_allow_html=True)
        
        with col_m3:
            processing_time = result.get("metadata", {}).get("processing_time_seconds", 0)
            st.markdown(f'<div class="metric-card"><h3>Processing Time</h3><h2>{processing_time:.2f}s</h2></div>', unsafe_allow_html=True)
        
        with col_m4:
            text_length = result.get("metadata", {}).get("text_length", 0)
            st.markdown(f'<div class="metric-card"><h3>Document Length</h3><h2>{text_length} chars</h2></div>', unsafe_allow_html=True)
        
        # Visualize confidence if relevant
        if display_confidence and "classification" in result:
            st.markdown("### Classification Confidence")
            confidence = result["classification"].get("confidence_score", 0)
            
            # Create a gauge chart
            fig = px.pie(values=[confidence, 1-confidence], 
                          names=["Confidence", "Uncertainty"],
                          hole=0.7, 
                          color_discrete_sequence=["#1E88E5", "#ECEFF1"])
            
            fig.update_layout(
                annotations=[dict(text=f"{confidence:.2f}", x=0.5, y=0.5, font_size=20, showarrow=False)],
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary section
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### Summary")
        
        if "analysis" in result and "summary" in result["analysis"]:
            summary_data = result["analysis"]["summary"]
            if isinstance(summary_data, dict) and "value" in summary_data:
                st.markdown(f"**{summary_data['value']}**")
                if display_confidence and "confidence_score" in summary_data:
                    st.caption(f"Confidence: {summary_data['confidence_score']:.2f}")
            else:
                st.markdown(f"**{summary_data}**")
        else:
            st.info("No summary information available")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Document details
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### Document Details")
        
        if "analysis" in result and "document_details" in result["analysis"]:
            doc_details = result["analysis"]["document_details"]
            if isinstance(doc_details, dict) and "value" in doc_details:
                if isinstance(doc_details["value"], dict):
                    for key, value in doc_details["value"].items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
                else:
                    st.markdown(doc_details["value"])
            else:
                st.json(doc_details)
        else:
            st.info("No detailed document information available")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risks and Opportunities
        col_risk, col_opp = st.columns(2)
        
        with col_risk:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown("### Identified Risks")
            
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
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_opp:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown("### Opportunities")
            
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
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation Checks
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### Validation Checks")
        
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
                
                # Add styling
                def color_status(val):
                    if val is True:
                        return 'background-color: #E8F5E9; color: #2E7D32'
                    elif val is False:
                        return 'background-color: #FFEBEE; color: #C62828'
                    else:
                        return ''
                
                styled_df = check_df.style.applymap(color_status, subset=['Status'])
                
                # Display
                st.dataframe(styled_df, use_container_width=True)
                
                if display_confidence and "confidence_score" in validation:
                    st.caption(f"Overall validation confidence: {validation['confidence_score']:.2f}")
            else:
                st.json(validation)
        else:
            st.info("No validation information available")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Required Actions
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("### Recommended Actions")
        
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
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Raw JSON options
        with st.expander("View Raw JSON Response"):
            st.json(result)
            
        # Export options
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("Export Results as JSON"):
                json_str = json.dumps(result, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="analysis_result.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col_exp2:
            if st.button("Generate PDF Report"):
                st.info("PDF report generation would require additional integration with a PDF generation library.")

# Footer
st.markdown("---")
st.caption("Document Analysis powered by Claude AI • Built with Streamlit")
