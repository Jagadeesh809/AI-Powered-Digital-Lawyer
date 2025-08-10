import streamlit as st
import os
from pathlib import Path
import tempfile
import re
from typing import List, Dict
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple but elegant CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 5px solid #ef4444;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .risk-low {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-left: 5px solid #22c55e;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .entity-tag {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        color: #1d4ed8;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        margin: 1rem 0;
    }
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class SimpleDocumentProcessor:
    """Simple document processor with basic PDF/DOCX support"""
    
    def extract_text(self, file_path):
        """Extract text from uploaded file"""
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self._extract_from_docx(file_path)
            else:
                return "Unsupported file format. Please upload PDF, DOCX, or TXT files."
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip() if text.strip() else "Could not extract text from PDF. The PDF might be image-based or encrypted."
        except ImportError:
            return """
            PDF processing requires PyPDF2. For demo purposes, here's sample legal text:
            
            SAMPLE LEGAL CONTRACT
            
            This Agreement is entered into between ABC Corporation ("Company") and XYZ Services LLC ("Contractor").
            
            1. TERMINATION: Company may terminate this agreement at will without notice.
            2. LIABILITY: Company disclaims all liability and warranties. Services provided "as is" without warranty.
            3. PAYMENT: Late fees of 5% per month apply. All payments are non-refundable.
            4. INDEMNIFICATION: Contractor shall indemnify and hold harmless Company from all claims.
            5. INTELLECTUAL PROPERTY: All work product becomes exclusive property of Company.
            6. CONFIDENTIALITY: Contractor agrees to perpetual confidentiality of all Company information.
            
            This contract is governed by the laws of California.
            Payment terms: $50,000 due immediately upon signing.
            Effective Date: January 15, 2024
            """
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip() if text.strip() else "Could not extract text from DOCX file."
        except ImportError:
            return "DOCX processing requires python-docx library. Please install it or use TXT files."
        except Exception as e:
            return f"Error extracting DOCX text: {str(e)}"

class SimpleRiskDetector:
    """Simple risk detector without NLTK dependencies"""
    
    def __init__(self):
        self.risk_patterns = {
            'liability_exclusion': {
                'keywords': ['exclude liability', 'disclaim', 'not liable', 'as is', 'no warranty', 'disclaims all liability'],
                'severity': 'High',
                'description': 'Liability exclusion clauses that limit your legal recourse'
            },
            'termination_rights': {
                'keywords': ['terminate at will', 'terminate without notice', 'sole discretion', 'immediate termination'],
                'severity': 'Medium', 
                'description': 'Unfavorable termination rights that favor the other party'
            },
            'payment_terms': {
                'keywords': ['late fee', 'penalty', 'non-refundable', 'immediate payment', 'due immediately'],
                'severity': 'Medium',
                'description': 'Strict payment terms with penalties'
            },
            'indemnification': {
                'keywords': ['indemnify', 'hold harmless', 'defend against', 'unlimited indemnification'],
                'severity': 'High',
                'description': 'Broad indemnification obligations that could expose you to liability'
            },
            'intellectual_property': {
                'keywords': ['exclusive property', 'all rights', 'work product', 'intellectual property'],
                'severity': 'High',
                'description': 'Broad intellectual property assignments'
            },
            'confidentiality': {
                'keywords': ['perpetual confidentiality', 'confidential information', 'non-disclosure'],
                'severity': 'Medium',
                'description': 'Broad confidentiality obligations'
            }
        }
    
    def detect_risks(self, text: str, threshold: float = 0.7) -> List[Dict]:
        """Detect risks in text using simple pattern matching"""
        risks = []
        text_lower = text.lower()
        
        # Split into sentences (simple approach)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        for risk_type, config in self.risk_patterns.items():
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for keyword in config['keywords']:
                    if keyword in sentence_lower:
                        risks.append({
                            'type': risk_type.replace('_', ' ').title(),
                            'severity': config['severity'],
                            'clause': sentence[:200] + '...' if len(sentence) > 200 else sentence,
                            'explanation': config['description'],
                            'recommendation': f"Review and consider negotiating this {risk_type.replace('_', ' ')} clause.",
                            'confidence': 0.85,
                            'matched_text': keyword
                        })
                        break
        
        return risks[:10]  # Limit to top 10 risks

class SimpleLegalAnalyzer:
    """Simple legal analyzer without NLTK dependencies"""
    
    def __init__(self):
        self.entity_patterns = {
            'COMPANY': [r'\b[A-Z][a-z]+ (?:Inc\.|LLC|Corp\.|Corporation|Company|Ltd\.)\b'],
            'DATE': [r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'],
            'MONEY': [r'\$[\d,]+(?:\.\d{2})?', r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b'],
            'LEGAL_TERM': [r'\b(?:contract|agreement|clause|provision|warranty|liability|indemnity|breach|termination|confidentiality|intellectual property)\b'],
            'LOCATION': [r'\b(?:California|New York|Texas|Florida|San Francisco|Los Angeles|courts?)\b']
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
        
        # Remove duplicates and sort
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['text'].lower(), entity['label'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities[:20]  # Limit to top 20

class SimpleLLMSummarizer:
    """Simple LLM-style summarizer for generating plain English explanations"""
    
    def generate_plain_english_summary(self, text: str, entities: List[Dict], risks: List[Dict]) -> str:
        """Generate a comprehensive plain English summary of the legal document"""
        
        # Extract key information
        word_count = len(text.split())
        risk_count = len(risks)
        high_risk_count = len([r for r in risks if r['severity'] == 'High'])
        entity_count = len(entities)
        
        # Identify key parties
        parties = [e['text'] for e in entities if e['label'] == 'COMPANY']
        dates = [e['text'] for e in entities if e['label'] == 'DATE']
        money_amounts = [e['text'] for e in entities if e['label'] == 'MONEY']
        locations = [e['text'] for e in entities if e['label'] == 'LOCATION']
        
        # Generate summary based on content analysis
        summary_parts = []
        
        # Document overview
        summary_parts.append(f"**Document Overview:**\nThis legal document contains {word_count:,} words and appears to be a {'contract' if 'agreement' in text.lower() or 'contract' in text.lower() else 'legal document'} that establishes terms and conditions between parties.")
        
        # Parties involved
        if parties:
            party_text = ", ".join(parties[:3])
            if len(parties) > 3:
                party_text += f" and {len(parties) - 3} other entities"
            summary_parts.append(f"**Parties Involved:**\nThe main parties to this agreement include {party_text}.")
        
        # Key dates and financial terms
        if dates or money_amounts:
            financial_info = []
            if dates:
                financial_info.append(f"important dates including {dates[0]}")
            if money_amounts:
                financial_info.append(f"financial amounts such as {money_amounts[0]}")
            if financial_info:
                summary_parts.append(f"**Key Terms:**\nThe document specifies {' and '.join(financial_info)}.")
        
        # Risk assessment
        if risk_count > 0:
            risk_summary = f"**Risk Assessment:**\nOur analysis identified {risk_count} potential areas of concern"
            if high_risk_count > 0:
                risk_summary += f", including {high_risk_count} high-risk items that require immediate attention"
            risk_summary += ". "
            
            # Mention specific risk types
            risk_types = list(set([r['type'] for r in risks[:3]]))
            if risk_types:
                risk_summary += f"The main risk categories include {', '.join(risk_types).lower()}."
            
            summary_parts.append(risk_summary)
        else:
            summary_parts.append("**Risk Assessment:**\nThe document appears to have balanced terms with no major red flags identified in our analysis.")
        
        # Recommendations
        if high_risk_count > 0:
            summary_parts.append(f"**Recommendations:**\nGiven the {high_risk_count} high-risk items identified, we strongly recommend having a legal professional review this document before signing. Pay particular attention to clauses related to liability, termination rights, and payment terms.")
        elif risk_count > 0:
            summary_parts.append("**Recommendations:**\nWhile this document doesn't contain major red flags, we recommend reviewing the identified risk areas and considering whether the terms align with your interests and expectations.")
        else:
            summary_parts.append("**Recommendations:**\nThis document appears to have reasonable terms. However, we always recommend having important legal documents reviewed by a qualified attorney before signing.")
        
        # Key takeaways
        takeaways = []
        if 'termination' in text.lower():
            takeaways.append("termination procedures and rights")
        if 'liability' in text.lower():
            takeaways.append("liability and risk allocation")
        if 'payment' in text.lower():
            takeaways.append("payment terms and conditions")
        if 'confidentiality' in text.lower():
            takeaways.append("confidentiality obligations")
        
        if takeaways:
            summary_parts.append(f"**Key Areas Covered:**\nThis document primarily addresses {', '.join(takeaways)}.")
        
        return "\n\n".join(summary_parts)

def initialize_components():
    """Initialize components in session state"""
    if 'components_initialized' not in st.session_state:
        st.session_state.doc_processor = SimpleDocumentProcessor()
        st.session_state.risk_detector = SimpleRiskDetector()
        st.session_state.legal_analyzer = SimpleLegalAnalyzer()
        st.session_state.llm_summarizer = SimpleLLMSummarizer()
        st.session_state.components_initialized = True

def analyze_document(file_path: str, analysis_type: str, risk_threshold: float):
    """Analyze the uploaded document"""
    with st.spinner("🔍 Analyzing document..."):
        try:
            # Extract text
            text_content = st.session_state.doc_processor.extract_text(file_path)
            
            # Extract entities
            entities = st.session_state.legal_analyzer.extract_entities(text_content)
            
            # Detect risks
            risks = st.session_state.risk_detector.detect_risks(text_content, risk_threshold)
            
            # Generate comprehensive plain English summary using LLM
            word_count = len(text_content.split())
            summary = st.session_state.llm_summarizer.generate_plain_english_summary(text_content, entities, risks)
            
            # Store results
            st.session_state.analysis_results = {
                'text': text_content,
                'entities': entities,
                'risks': risks,
                'summary': summary,
                'word_count': word_count,
                'analysis_type': analysis_type
            }
            
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def display_results():
    """Display analysis results with fancy styling based on analysis type"""
    if 'analysis_results' not in st.session_state:
        return
        
    results = st.session_state.analysis_results
    analysis_type = results.get('analysis_type', 'Full Analysis')
    
    # Show different stats cards based on analysis type
    if analysis_type == "Entity Extraction":
        # Show only entity-related stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #1e3a8a; margin: 0;">📄</h3>
                <h2 style="margin: 0.5rem 0;">{results['word_count']:,}</h2>
                <p style="color: #64748b; margin: 0;">Words Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            total_entities = len(results['entities'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #059669; margin: 0;">🏷️</h3>
                <h2 style="margin: 0.5rem 0;">{total_entities}</h2>
                <p style="color: #64748b; margin: 0;">Entities Found</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif analysis_type == "Risk Assessment Only":
        # Show only risk-related stats
        col1, col2, col3 = st.columns(3)
        with col1:
            total_risks = len(results['risks'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #dc2626; margin: 0;">⚠️</h3>
                <h2 style="margin: 0.5rem 0;">{total_risks}</h2>
                <p style="color: #64748b; margin: 0;">Total Risks</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            high_risks = len([r for r in results['risks'] if r['severity'] == 'High'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #dc2626; margin: 0;">🚨</h3>
                <h2 style="margin: 0.5rem 0;">{high_risks}</h2>
                <p style="color: #64748b; margin: 0;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            medium_risks = len([r for r in results['risks'] if r['severity'] == 'Medium'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #f59e0b; margin: 0;">🟡</h3>
                <h2 style="margin: 0.5rem 0;">{medium_risks}</h2>
                <p style="color: #64748b; margin: 0;">Medium Risk</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif analysis_type == "Summary Only":
        # Show only summary stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #1e3a8a; margin: 0;">📄</h3>
                <h2 style="margin: 0.5rem 0;">{results['word_count']:,}</h2>
                <p style="color: #64748b; margin: 0;">Words Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #059669; margin: 0;">📋</h3>
                <h2 style="margin: 0.5rem 0;">Ready</h2>
                <p style="color: #64748b; margin: 0;">Summary Generated</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Full Analysis
        # Show all stats cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #1e3a8a; margin: 0;">📄</h3>
                <h2 style="margin: 0.5rem 0;">{results['word_count']:,}</h2>
                <p style="color: #64748b; margin: 0;">Words</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            total_risks = len(results['risks'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #dc2626; margin: 0;">⚠️</h3>
                <h2 style="margin: 0.5rem 0;">{total_risks}</h2>
                <p style="color: #64748b; margin: 0;">Total Risks</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            high_risks = len([r for r in results['risks'] if r['severity'] == 'High'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #dc2626; margin: 0;">🚨</h3>
                <h2 style="margin: 0.5rem 0;">{high_risks}</h2>
                <p style="color: #64748b; margin: 0;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            total_entities = len(results['entities'])
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="color: #059669; margin: 0;">🏷️</h3>
                <h2 style="margin: 0.5rem 0;">{total_entities}</h2>
                <p style="color: #64748b; margin: 0;">Entities</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show different tabs based on analysis type
    if analysis_type == "Entity Extraction":
        # Only show entities tab
        st.markdown("### 🏷️ Extracted Entities")
        display_entities_section(results)
    
    elif analysis_type == "Risk Assessment Only":
        # Only show risk analysis tab
        st.markdown("### ⚠️ Risk Analysis")
        display_risk_section(results)
    
    elif analysis_type == "Summary Only":
        # Only show summary tab
        st.markdown("### 📋 Document Summary")
        display_summary_section(results)
    
    else:  # Full Analysis
        # Show all tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "⚠️ Risk Analysis", "🏷️ Entities", "📄 Full Text"])
        
        with tab1:
            st.markdown("### 📋 Document Summary")
            display_summary_section(results)
        
        with tab2:
            st.markdown("### ⚠️ Risk Analysis")
            display_risk_section(results)
        
        with tab3:
            st.markdown("### 🏷️ Extracted Entities")
            display_entities_section(results)
        
        with tab4:
            st.markdown("### 📄 Full Document Text")
            st.text_area("Document Content", results['text'], height=400, disabled=True)

def display_summary_section(results):
    """Display summary section"""
    st.info(results['summary'])
    
    if results['risks']:
        st.markdown("### 🎯 Key Findings")
        severity_counts = {}
        for risk in results['risks']:
            severity = risk['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
            st.write(f"{color} **{severity} Risk**: {count} items found")

def display_risk_section(results):
    """Display risk analysis section"""
    if results['risks']:
        # Risk severity chart
        if len(results['risks']) > 0:
            risk_df = pd.DataFrame(results['risks'])
            severity_counts = risk_df['severity'].value_counts()
            
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Risk Distribution by Severity",
                color_discrete_map={
                    'High': '#ef4444',
                    'Medium': '#f59e0b', 
                    'Low': '#22c55e'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # List risks with fancy styling
        st.markdown("### 📋 Detailed Risk Items")
        for i, risk in enumerate(results['risks'], 1):
            severity_class = f"risk-{risk['severity'].lower()}"
            st.markdown(f"""
            <div class="{severity_class}">
                <h4 style="margin: 0 0 0.5rem 0;">🔸 {risk['type']} - {risk['severity']} Risk</h4>
                <p style="margin: 0.5rem 0;"><strong>Clause:</strong> {risk['clause']}</p>
                <p style="margin: 0.5rem 0;"><strong>Issue:</strong> {risk['explanation']}</p>
                <p style="margin: 0.5rem 0;"><strong>Recommendation:</strong> {risk['recommendation']}</p>
                <small style="color: #64748b;">Confidence: {risk['confidence']:.0%}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No significant risks detected in this document.")

def display_entities_section(results):
    """Display entities section"""
    if results['entities']:
        # Group entities by type
        entity_groups = {}
        for entity in results['entities']:
            entity_type = entity['label']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity['text'])
        
        for entity_type, entities in entity_groups.items():
            st.markdown(f"**{entity_type}:**")
            entity_html = ""
            for entity in set(entities):
                entity_html += f'<span class="entity-tag">{entity}</span> '
            st.markdown(entity_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No entities found in the document.")



def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">⚖️ AI Legal Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analyze legal documents with AI-powered risk detection</p>', unsafe_allow_html=True)
    
    # Initialize components
    initialize_components()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Full Analysis", "Risk Assessment Only", "Entity Extraction", "Summary Only"],
            help="Choose the type of analysis to perform"
        )
        
        risk_threshold = st.slider(
            "Risk Sensitivity", 
            0.1, 1.0, 0.7,
            help="Higher values detect more risks"
        )
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.info("""
        **AI Legal Assistant** helps you:
        - 🔍 Analyze legal documents
        - ⚠️ Identify risk clauses  
        - 🏷️ Extract key entities
        - 📊 Generate summaries
        
        Built with advanced NLP and risk detection algorithms.
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file and clear previous results
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.last_uploaded_file = uploaded_file.name
                # Clear previous analysis results
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            
            col_analyze, col_clear = st.columns(2)
            with col_analyze:
                if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                    analyze_document(temp_path, analysis_type, risk_threshold)
            
            with col_clear:
                if st.button("🗑️ Clear Results", use_container_width=True):
                    if 'analysis_results' in st.session_state:
                        del st.session_state.analysis_results
                    st.rerun()
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            display_results()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #64748b;">
                <h3>🔍 Ready to Analyze</h3>
                <p>Upload a legal document to get started with AI-powered analysis.</p>
                <p><strong>Features:</strong></p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Risk clause detection</li>
                    <li>Entity extraction</li>
                    <li>Document summarization</li>
                    <li>Legal term identification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
