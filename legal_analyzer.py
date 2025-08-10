import spacy
import re
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os

class LegalAnalyzer:
    """Handles NLP and NER for legal document analysis"""
    
    def __init__(self):
        self.nlp = None
        self._initialize_nlp()
        self._download_nltk_data()
        
        # Legal entity patterns
        self.legal_patterns = {
            'PARTY': [
                r'\b(?:party|parties|plaintiff|defendant|appellant|appellee|petitioner|respondent)\b',
                r'\b[A-Z][a-z]+ (?:Inc\.|LLC|Corp\.|Corporation|Company|Ltd\.|Limited)\b',
                r'\b(?:The )?[A-Z][A-Za-z\s]+ (?:Inc\.|LLC|Corp\.|Corporation|Company|Ltd\.|Limited)\b'
            ],
            'DATE': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
                r'\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b'
            ],
            'MONEY': [
                r'\$[\d,]+(?:\.\d{2})?',
                r'\b(?:dollars?|USD|cents?)\b',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b'
            ],
            'LEGAL_TERM': [
                r'\b(?:contract|agreement|clause|provision|term|condition|warranty|guarantee|liability|indemnity|breach|default|termination|force majeure|arbitration|jurisdiction|governing law|confidentiality|non-disclosure|intellectual property|copyright|trademark|patent)\b',
                r'\b(?:shall|must|may|will|should|ought to|required to|obligated to|entitled to|authorized to)\b'
            ],
            'JURISDICTION': [
                r'\b(?:United States|USA|US|California|New York|Texas|Florida|Illinois|Pennsylvania|Ohio|Georgia|North Carolina|Michigan|New Jersey|Virginia|Washington|Arizona|Massachusetts|Tennessee|Indiana|Missouri|Maryland|Wisconsin|Colorado|Minnesota|South Carolina|Alabama|Louisiana|Kentucky|Oregon|Oklahoma|Connecticut|Utah|Iowa|Nevada|Arkansas|Mississippi|Kansas|New Mexico|Nebraska|West Virginia|Idaho|Hawaii|New Hampshire|Maine|Montana|Rhode Island|Delaware|South Dakota|North Dakota|Alaska|Vermont|Wyoming)\b',
                r'\b(?:federal|state|local|municipal|county|district)\s+(?:court|jurisdiction|law|statute|regulation)\b'
            ]
        }
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model"""
        # Skip spaCy loading for now to avoid hanging
        self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
        except Exception as e:
            # If NLTK download fails, continue without it
            print(f"NLTK download failed: {e}")
    
    def extract_entities(self, text):
        """Extract legal entities from text using NER and pattern matching"""
        entities = []
        
        # Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'DATE', 'MONEY', 'GPE', 'LAW']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8  # Default confidence for spaCy
                    })
        
        # Add pattern-based entity extraction
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x['start'])
        
        return entities
    
    def _extract_pattern_entities(self, text):
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.7  # Pattern-based confidence
                    })
        
        return entities
    
    def _deduplicate_entities(self, entities):
        """Remove duplicate entities based on text and position overlap"""
        unique_entities = []
        
        for entity in entities:
            is_duplicate = False
            for existing in unique_entities:
                # Check for overlap
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start'] and
                    entity['text'].lower() == existing['text'].lower()):
                    # Keep the one with higher confidence
                    if entity['confidence'] > existing['confidence']:
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_key_clauses(self, text):
        """Extract important legal clauses from the document"""
        clauses = []
        sentences = sent_tokenize(text)
        
        # Keywords that indicate important clauses
        important_keywords = [
            'liability', 'indemnity', 'termination', 'breach', 'default',
            'force majeure', 'confidentiality', 'non-disclosure', 'warranty',
            'guarantee', 'limitation', 'exclusion', 'arbitration', 'jurisdiction',
            'governing law', 'intellectual property', 'payment', 'penalty',
            'damages', 'remedy', 'cure', 'notice', 'assignment', 'amendment'
        ]
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains important keywords
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    clauses.append({
                        'text': sentence.strip(),
                        'type': keyword.replace('_', ' ').title(),
                        'sentence_index': i,
                        'importance_score': self._calculate_importance_score(sentence)
                    })
                    break
        
        # Sort by importance score
        clauses.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return clauses
    
    def _calculate_importance_score(self, sentence):
        """Calculate importance score for a sentence"""
        score = 0
        sentence_lower = sentence.lower()
        
        # High-impact words
        high_impact_words = [
            'shall not', 'must not', 'prohibited', 'forbidden', 'liable',
            'penalty', 'damages', 'terminate', 'breach', 'default',
            'indemnify', 'hold harmless', 'exclusive', 'sole', 'entire'
        ]
        
        # Medium-impact words
        medium_impact_words = [
            'may', 'should', 'warranty', 'guarantee', 'represent',
            'covenant', 'agree', 'acknowledge', 'consent'
        ]
        
        for word in high_impact_words:
            if word in sentence_lower:
                score += 3
        
        for word in medium_impact_words:
            if word in sentence_lower:
                score += 1
        
        # Length penalty for very long sentences
        if len(sentence.split()) > 50:
            score -= 1
        
        return max(score, 0)
    
    def analyze_sentence_structure(self, text):
        """Analyze sentence structure for legal complexity"""
        sentences = sent_tokenize(text)
        analysis = {
            'total_sentences': len(sentences),
            'avg_sentence_length': 0,
            'complex_sentences': 0,
            'passive_voice_count': 0,
            'modal_verb_count': 0
        }
        
        if sentences:
            total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
            analysis['avg_sentence_length'] = total_words / len(sentences)
            
            # Analyze each sentence
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                
                # Complex sentence detection (more than 25 words)
                if len(words) > 25:
                    analysis['complex_sentences'] += 1
                
                # Passive voice detection (simple heuristic)
                if any(word in words for word in ['was', 'were', 'been', 'being']) and \
                   any(word.endswith('ed') for word in words):
                    analysis['passive_voice_count'] += 1
                
                # Modal verbs
                modal_verbs = ['shall', 'should', 'must', 'may', 'might', 'will', 'would', 'can', 'could']
                analysis['modal_verb_count'] += sum(1 for word in words if word in modal_verbs)
        
        return analysis
