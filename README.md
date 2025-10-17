# Knowledge-base Search Engine
A powerful Document Analysis System that uses Groq's Llama-3.1-8b-instant model for intelligent document processing and question answering. Upload documents and get AI-powered insights instantly!
link for work video : https://drive.google.com/file/d/1vlVpean_z_WrvS4o7JJlRai0sOh6cq44/view?usp=sharing



# Groq RAG Knowledge-base Search Engine

A powerful Document Analysis System that uses Groq's Llama-3.1-8b-instant model for intelligent document processing and question answering. Upload documents and get AI-powered insights instantly!

##  Features

- ** Multi-Format Support**: PDF, DOCX, DOC, TXT files
- ** Intelligent Q&A**: Ask questions about your documents
- ** Fast Responses**: Powered by Groq's accelerated inference
- ** Two Answer Modes**: Basic RAG and Exact Answer modes
- ** Real-time Status**: Live system monitoring
- ** Easy Management**: Upload, reset, and monitor documents
- ** Beautiful UI**: Modern, responsive design

##  Tech Stack

### Backend
- **Flask** - Python web framework
- **LlamaIndex** - RAG framework
- **Groq API** - LLM inference (Llama-3.1-8b-instant)
- **HuggingFace Embeddings** - Text embeddings
- **PyTorch** - Machine learning

### Frontend
- **HTML5/CSS3** - Modern web standards
- **Tailwind CSS** - Utility-first CSS framework
- **JavaScript** - Frontend logic
- **Fetch API** - HTTP requests

##  Prerequisites

- Python 3.8 or higher
- Groq API key
- 2GB+ RAM free

## Quick Start

### 1. Clone/Download the Project
```bash
# Create project directory
mkdir groq-rag-project
cd groq-rag-project
```

### 2. Set Up Backend


#### Option : Using Conda
```bash
# Create conda environment
conda create -n groq-rag python=3.9 -y
conda activate groq-rag

# Install packages
pip install flask flask-cors llama-index llama-index-llms-groq llama-index-embeddings-huggingface pypdf python-docx docx2txt groq sentence-transformers torch transformers
```

### 3. Configure API Key
Edit `app.py` and replace with your Groq API key:
```python
GROQ_API_KEY = "your-actual-groq-api-key-here"
```

### 4. Start Backend Server
```bash
python app.py
```
You should see:
```
 GROQ RAG BACKEND SERVER STARTING...
 API Endpoints:
   GET  /          - Web interface
   GET  /status    - System status
   POST /upload    - Upload document
   POST /ask       - Ask question
   POST /ask-exact - Ask exact question
   GET  /reset     - Reset system
 Server running on: http://localhost:5000
```

### 5. Launch Frontend
Open `index.html` in your web browser:
- Double-click the file, OR
- Use a local server: `python -m http.server 8000` then visit `http://localhost:8000`

##  Project Structure

```
groq-rag-project/
├── app.py                 # Flask backend server
├── index.html             # Frontend web interface
├── uploads/               # Uploaded documents (auto-created)
├── templates/             # HTML templates (auto-created)
└── README.md              # This file
```

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/status` | System status |
| `POST` | `/upload` | Upload document |
| `POST` | `/ask` | Ask basic question |
| `POST` | `/ask-exact` | Ask exact question |
| `GET` | `/reset` | Reset system |

##  Usage Guide

### 1. Upload Documents
- Click "Upload Document"
- Select PDF, DOCX, DOC, or TXT file
- Click "Upload & Process Document"
- Wait for processing confirmation

### 2. Ask Questions
- **Basic Question**: General questions about the document
- **Exact Answer**: Precise, document-based answers
- **Quick Actions**: Pre-defined common questions

### 3. View Results
- **Response Tab**: Question and answer
- **Context Tab**: Retrieved document context
- **Details Tab**: Response metadata

##  Troubleshooting

### Common Issues

**Backend Connection Failed**
- Ensure Flask server is running on port 5000
- Check if `python app.py` is executing without errors
- Verify no other services are using port 5000

**Document Upload Fails**
- Check file size (max 16MB)
- Verify supported formats: PDF, DOCX, DOC, TXT
- Ensure sufficient disk space

**API Key Issues**
- Verify Groq API key is correct
- Check internet connection
- Ensure API key has sufficient credits

**Module Import Errors**
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version (requires 3.8+)
- Verify virtual environment is activated

### Performance Tips
- Use smaller documents for faster processing
- Close other resource-intensive applications
- Ensure stable internet connection for Groq API calls

##  Development

### Adding New Features
1. Backend modifications: Edit `app.py`
2. Frontend modifications: Edit `index.html`
3. Test locally before deployment

### File Processing
The system uses:
- `pypdf` for PDF files
- `python-docx` for Word documents
- `docx2txt` for text extraction
- Custom text processing for TXT files

### Model Configuration
- **LLM**: Groq Llama-3.1-8b-instant
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 512 tokens
- **Temperature**: 0.1 (for consistent responses)

##  Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production, consider:
- Using Gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- Setting up Nginx reverse proxy
- Using environment variables for API keys
- Implementing proper error handling

##  System Requirements

- **Minimum**: 2GB RAM, 1GB storage
- **Recommended**: 4GB+ RAM, 2GB+ storage
- **Network**: Stable internet connection for Groq API

##  Security Notes

- API keys are embedded in code (for development)
- For production, use environment variables
- File uploads are sanitized but validate externally
- No authentication implemented (add for production)

##  License

This project is for educational and development purposes. Please comply with Groq's API terms of service.

