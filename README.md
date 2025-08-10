# 🏛️ AI Legal Assistant

A powerful AI-powered legal document analysis tool that helps users understand contracts, identify risks, and extract key information from legal documents.

![AI Legal Assistant](https://img.shields.io/badge/AI-Legal%20Assistant-blue?style=for-the-badge&logo=scales)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## ✨ Features

- **📄 Document Processing**: Upload and analyze PDF, DOCX, and TXT files
- **⚠️ Risk Detection**: Identify liability, termination, payment, and other high-risk clauses
- **🏷️ Entity Extraction**: Extract companies, dates, money amounts, and legal terms
- **📋 Plain English Summaries**: LLM-generated explanations in simple language
- **📊 Interactive Visualizations**: Beautiful charts and risk distribution analysis
- **🎯 Targeted Analysis**: Choose specific analysis types (Risk Only, Entities Only, etc.)
- **🎨 Modern UI**: Clean, professional interface with gradient styling

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-legal-assistant.git
cd ai-legal-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## 📋 Usage

1. **Upload Document**: Choose a PDF, DOCX, or TXT legal document
2. **Select Analysis Type**: 
   - Full Analysis (everything)
   - Risk Assessment Only
   - Entity Extraction Only  
   - Summary Only
3. **Adjust Risk Sensitivity**: Use the slider to control detection sensitivity
4. **Analyze**: Click "Analyze Document" to process
5. **Review Results**: View risks, entities, and plain English summaries

## 🛠️ Technical Architecture

- **Frontend**: Streamlit with custom CSS styling
- **Document Processing**: PyPDF2 and python-docx for text extraction
- **Risk Detection**: Pattern-matching algorithms for legal clause analysis
- **Entity Recognition**: Regex-based extraction for legal entities
- **Summarization**: Custom LLM-style plain English generation
- **Visualization**: Plotly for interactive charts and graphs

## 📁 Project Structure

```
ai-legal-assistant/
├── app.py                 # Main Streamlit application (all-in-one)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
├── document_processor.py  # Document text extraction (legacy)
├── legal_analyzer.py     # NLP and entity extraction (legacy)
├── risk_detector.py      # Risk analysis and detection (legacy)
├── llm_summarizer.py     # Document summarization (legacy)
└── rag_system.py         # Retrieval-augmented generation (legacy)
```

## 🎯 Key Components

### Risk Detection
- Liability exclusion clauses
- Unfavorable termination rights
- Strict payment terms
- Broad indemnification obligations
- Intellectual property assignments
- Confidentiality requirements

### Entity Extraction
- Company names and legal entities
- Important dates and deadlines
- Financial amounts and terms
- Legal terminology
- Jurisdictions and locations

### Plain English Summaries
- Document overview and purpose
- Key parties involved
- Financial terms and obligations
- Risk assessment and recommendations
- Next steps and legal advice

## 🔧 Configuration

The app includes several customizable settings:

- **Risk Sensitivity**: Adjust detection threshold (0.1 - 1.0)
- **Analysis Types**: Choose specific analysis focus
- **File Support**: PDF, DOCX, and TXT formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Uses PyPDF2 for PDF processing
- Plotly for data visualization
- Custom NLP algorithms for legal text analysis

## 📞 Support

For questions or support, please open an issue on GitHub or contact [your-email@example.com].

---

**⚠️ Disclaimer**: This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for legal matters.

### 🧠 **AI-Powered Analysis**
- **NLP & NER**: Extract legal entities (parties, dates, monetary amounts, legal terms)
- **Risk Detection**: Identify unusual and high-risk clauses using pattern matching and ML
- **Clause Classification**: Categorize different types of legal provisions

### ⚠️ **Risk Assessment**
- **Multi-Level Risk Scoring**: High, Medium, and Low risk categorization
- **Pattern-Based Detection**: Identify risky clauses using legal expertise patterns
- **Keyword Analysis**: Advanced risk scoring based on legal terminology
- **Unusual Clause Detection**: ML-based identification of non-standard provisions

### 📚 **RAG Integration**
- **Legal Precedents**: Search through curated legal case database
- **Contextual Recommendations**: Find relevant case law and precedents
- **Case-Based Explanations**: Understand how similar clauses have been interpreted

### 🤖 **LLM Summarization**
- **Plain English Summaries**: Convert complex legal language to understandable text
- **Document Overview**: Comprehensive analysis of contract structure and key points
- **Clause Explanations**: Individual clause breakdowns in simple terms
- **Risk Explanations**: Clear explanations of why certain clauses are risky

### 📊 **Interactive Dashboard**
- **Modern Web Interface**: Built with Streamlit for easy use
- **Visual Risk Analysis**: Charts and graphs showing risk distribution
- **Tabbed Results**: Organized presentation of analysis results
- **Real-time Processing**: Live analysis with progress indicators

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   cd "AI LAWYER"
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## Usage Guide

### 📄 **Document Upload**
1. Click "Browse files" or drag & drop your legal document
2. Supported formats: PDF, DOCX
3. Click "Analyze Document" to start processing

### ⚙️ **Analysis Options**
- **Full Analysis**: Complete document analysis with all features
- **Risk Assessment Only**: Focus on identifying risky clauses
- **Entity Extraction**: Extract parties, dates, and legal terms
- **Plain English Summary**: Generate readable document summary

### 🔧 **Settings**
- **Risk Sensitivity**: Adjust the threshold for risk detection (0.1 - 1.0)
- **Include Legal Precedents**: Enable/disable RAG-based case law search

### 📊 **Results Interpretation**

#### **Summary Tab**
- Document overview and key metrics
- Risk count and severity distribution
- Entity extraction summary

#### **Risk Analysis Tab**
- Visual risk distribution chart
- Detailed risk explanations
- Specific recommendations for each risk

#### **Entities Tab**
- Extracted parties, dates, monetary amounts
- Legal terms and jurisdictions
- Organized by entity type

#### **Precedents Tab**
- Relevant legal cases and precedents
- Similarity scores and key points
- Case-based explanations

#### **Full Text Tab**
- Complete document text
- Reference for detailed review

## Technical Architecture

### Core Components

1. **DocumentProcessor** (`document_processor.py`)
   - PDF and DOCX text extraction
   - Document metadata analysis
   - Text cleaning and normalization

2. **LegalAnalyzer** (`legal_analyzer.py`)
   - spaCy-based NER for legal entities
   - Pattern-based legal term extraction
   - Sentence structure analysis

3. **RiskDetector** (`risk_detector.py`)
   - Multi-pattern risk detection
   - Keyword-based risk scoring
   - ML-based unusual clause detection

4. **RAGSystem** (`rag_system.py`)
   - ChromaDB vector database
   - Sentence transformer embeddings
   - Legal precedent search and retrieval

5. **LLMSummarizer** (`llm_summarizer.py`)
   - Hugging Face transformer models
   - BART/DistilBART for summarization
   - Rule-based fallback system

### AI Models Used

- **spaCy**: `en_core_web_sm` for NER and linguistic analysis
- **Sentence Transformers**: `all-MiniLM-L6-v2` for embeddings
- **BART**: `facebook/bart-large-cnn` for summarization
- **DistilBART**: `sshleifer/distilbart-cnn-12-6` as fallback
- **ChromaDB**: Vector database for RAG functionality

## Risk Categories Detected

### 🚨 **High Risk**
- **Liability Exclusions**: Broad liability limitations or disclaimers
- **Indemnification**: Extensive indemnification obligations
- **IP Assignments**: Broad intellectual property transfers
- **Warranty Disclaimers**: Extensive warranty exclusions
- **Damage Limitations**: Caps on recoverable damages

### ⚠️ **Medium Risk**
- **Termination Rights**: Unfavorable termination provisions
- **Payment Terms**: Strict payment requirements or penalties
- **Confidentiality**: Broad confidentiality obligations

### ℹ️ **Low Risk**
- **Force Majeure**: Standard force majeure clauses
- **Governing Law**: Jurisdiction and choice of law provisions

## Legal Precedents Database

The system includes a curated database of legal precedents covering:

- **Liability Limitation** cases and enforceability standards
- **Termination Rights** precedents and notice requirements
- **Indemnification** clause interpretations
- **Intellectual Property** work-for-hire doctrines
- **Confidentiality** obligation enforceability
- **Force Majeure** event interpretations
- **Governing Law** choice enforceability
- **Warranty Disclaimer** requirements
- **Damages Limitation** enforceability standards
- **Payment Terms** penalty enforceability

## Customization

### Adding Custom Risk Patterns
Edit `risk_detector.py` to add new risk detection patterns:

```python
self.risk_patterns['new_risk_type'] = {
    'patterns': [r'your_regex_pattern'],
    'severity': 'High',
    'description': 'Description of the risk'
}
```

### Adding Legal Precedents
Use the RAG system to add custom precedents:

```python
rag_system.add_custom_precedent(
    title="Case Title",
    summary="Case summary",
    key_points="Key legal points",
    category="risk_category",
    risk_level="high",
    text="Full case text"
)
```

## Limitations

- **Not Legal Advice**: This tool provides analysis but not legal advice
- **Model Limitations**: AI models may miss nuanced legal interpretations
- **Jurisdiction Specific**: Precedents may not apply to all jurisdictions
- **Document Quality**: Analysis quality depends on document text extraction

## Security & Privacy

- **Local Processing**: All analysis happens locally on your machine
- **No Data Storage**: Documents are not permanently stored
- **Temporary Files**: Uploaded files are automatically cleaned up
- **No External APIs**: No document content sent to external services

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using for commercial purposes.

## Support

For issues or questions:
1. Check the troubleshooting section below
2. Review the code documentation
3. Create an issue with detailed information

## Troubleshooting

### Common Issues

**Model Download Errors**:
```bash
python -m spacy download en_core_web_sm
pip install torch torchvision torchaudio
```

**Memory Issues**:
- Reduce document size
- Lower risk sensitivity threshold
- Disable RAG analysis for large documents

**Slow Performance**:
- Ensure adequate RAM (8GB+ recommended)
- Use GPU if available for transformer models
- Process smaller document sections

**Installation Issues**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

**⚖️ Disclaimer**: This AI Legal Assistant is a tool for document analysis and should not replace professional legal advice. Always consult with qualified legal professionals for important legal matters.
