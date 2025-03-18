# Document Analysis Dashboard

A sleek, user-friendly Streamlit dashboard for analyzing documents using the Document Analysis API powered by Claude.

## Features

- 📄 Upload PDF documents for AI-powered analysis
- 🔍 Automatic document classification and information extraction
- 📊 Visual representation of analysis results
- 📝 Detailed insights including risks, opportunities, and recommended actions
- 🔄 Real-time API status checking
- 📱 Responsive design for both desktop and mobile use

## Prerequisites

- Python 3.8+
- Tesseract OCR (for the backend API)
- Claude API key
- Document Analysis API running locally or remotely

## Installation

1. Clone this repository:
```bash
https://github.com/HamzaComplyStream/DocParsing_multilingual.git
cd DocParsing_multilingual
```

2. Create and configure your environment variables:
```bash
cp .env.template .env
```
Then edit the `.env` file to add your API URL and Claude API key.

3. Run the setup script:
```bash
chmod +x run.sh
./run.sh
```

Or manually set up your environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

## Usage

1. Ensure your Document Analysis API is running (see the API repository for instructions)
2. Access the dashboard at http://localhost:8501
3. Configure your API URL and Claude API key in the sidebar
4. Upload a PDF document and view the analysis results

## Dashboard Sections

### 📄 Document Upload
- Upload PDF files
- Configure analysis options
- View document status and type information

### 🔍 Analysis Results
- Document classification details
- Key information extraction
- Risk and opportunity assessment 
- Validation checks
- Recommended actions
- Export options

## Configuration Options

You can configure the following options:
- API URL: The URL of your Document Analysis API
- Claude API Key: Your Anthropic Claude API key
- Analysis options: OCR usage, confidence display, etc.

## Connecting to the API

The dashboard connects to the Document Analysis API which should be running separately. Make sure the API is accessible at the configured URL.

## Screenshots

![Dashboard Screenshot](streamlit-demo-screenshot.svg)

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
