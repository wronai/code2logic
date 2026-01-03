"""
Enhanced Intent Generator with NLP support.

Uses lemmatization, pattern matching, and docstring extraction
to generate human-readable intent descriptions for functions.
"""

import re
from typing import Optional, List, Tuple

# Optional NLP imports with graceful degradation
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class EnhancedIntentGenerator:
    """
    Generator intencji z NLP - lemmatyzacja, ekstrakcja z docstringów.
    
    Supports both English and Polish intent generation.
    Falls back gracefully if NLP libraries are not available.
    
    Example:
        >>> gen = EnhancedIntentGenerator(lang='en')
        >>> gen.generate("getUserById", "Fetches a user by their ID")
        'retrieves user by id'
        >>> gen.generate("validateEmail")
        'validates email'
    """
    
    # Extended verb patterns (PL + EN)
    VERB_PATTERNS: dict[tuple[str, ...], tuple[str, str]] = {
        # CRUD operations
        ('get', 'fetch', 'retrieve', 'load', 'find', 'query', 'read', 'select'): 
            ('pobiera', 'retrieves'),
        ('set', 'update', 'modify', 'change', 'edit', 'put', 'patch'): 
            ('aktualizuje', 'updates'),
        ('create', 'make', 'build', 'generate', 'new', 'add', 'insert', 'post', 'init'): 
            ('tworzy', 'creates'),
        ('delete', 'remove', 'clear', 'destroy', 'drop', 'erase'): 
            ('usuwa', 'deletes'),
        
        # Validation
        ('is', 'has', 'can', 'should', 'check', 'test', 'assert'): 
            ('sprawdza', 'checks'),
        ('validate', 'verify', 'confirm', 'authenticate'): 
            ('waliduje', 'validates'),
        
        # Transformation
        ('convert', 'transform', 'map', 'translate', 'cast', 'to'): 
            ('konwertuje', 'converts'),
        ('parse', 'extract', 'decode', 'deserialize'): 
            ('parsuje', 'parses'),
        ('format', 'render', 'serialize', 'encode', 'stringify'): 
            ('formatuje', 'formats'),
        
        # Communication
        ('send', 'emit', 'dispatch', 'publish', 'notify', 'push'): 
            ('wysyła', 'sends'),
        ('receive', 'listen', 'subscribe', 'on', 'handle'): 
            ('obsługuje', 'handles'),
        
        # Lifecycle
        ('init', 'initialize', 'setup', 'configure', 'bootstrap'): 
            ('inicjalizuje', 'initializes'),
        ('start', 'run', 'execute', 'launch', 'begin', 'open'): 
            ('uruchamia', 'starts'),
        ('stop', 'end', 'finish', 'close', 'shutdown', 'terminate'): 
            ('kończy', 'stops'),
        
        # Data operations
        ('process', 'compute', 'calculate', 'evaluate', 'analyze'): 
            ('przetwarza', 'processes'),
        ('filter', 'search', 'match', 'lookup'): 
            ('filtruje', 'filters'),
        ('sort', 'order', 'arrange', 'rank'): 
            ('sortuje', 'sorts'),
        ('merge', 'combine', 'join', 'concat'): 
            ('łączy', 'merges'),
        ('split', 'divide', 'separate', 'partition'): 
            ('dzieli', 'splits'),
        
        # Logging
        ('log', 'print', 'write', 'output', 'display'): 
            ('loguje', 'logs'),
        
        # Registration
        ('register', 'bind', 'attach', 'connect', 'hook'): 
            ('rejestruje', 'registers'),
        
        # Caching
        ('cache', 'memoize', 'store', 'save', 'persist'): 
            ('cachuje', 'caches'),
    }
    
    def __init__(self, lang: str = 'en'):
        """
        Initialize the intent generator.
        
        Args:
            lang: Language for intent output ('en' or 'pl')
        """
        self.lang = lang
        self.lemmatizer = None
        self.nlp = None
        
        # Initialize NLTK lemmatizer if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                try:
                    nltk.download('wordnet', quiet=True)
                    self.lemmatizer = WordNetLemmatizer()
                except Exception:
                    pass
        
        # Initialize spaCy if available (for more advanced NLP)
        if SPACY_AVAILABLE:
            try:
                model = 'pl_core_news_sm' if lang == 'pl' else 'en_core_web_sm'
                self.nlp = spacy.load(model)
            except OSError:
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                except OSError:
                    pass
    
    def generate(self, name: str, docstring: Optional[str] = None) -> str:
        """
        Generate intent from function name and optional docstring.
        
        Args:
            name: Function or method name
            docstring: Optional docstring to extract intent from
            
        Returns:
            Human-readable intent description
            
        Example:
            >>> gen = EnhancedIntentGenerator()
            >>> gen.generate("calculateTotalPrice")
            'processes total price'
        """
        # Try docstring first
        if docstring:
            intent = self._extract_from_docstring(docstring)
            if intent and len(intent) >= 10:
                return intent[:80]
        
        # Parse function name
        words = self._split_name(name)
        if not words:
            return name
        
        first_word = words[0].lower()
        rest = ' '.join(words[1:]).lower() if len(words) > 1 else ''
        
        # Lemmatize if available
        if self.lemmatizer:
            try:
                first_word = self.lemmatizer.lemmatize(first_word, pos='v')
            except Exception:
                pass
        
        # Match against verb patterns
        intent_idx = 0 if self.lang == 'pl' else 1
        for verbs, intents in self.VERB_PATTERNS.items():
            if first_word in verbs:
                intent = intents[intent_idx]
                return f"{intent} {rest}" if rest else intent
        
        # Fallback - join words
        return ' '.join(words).lower()
    
    def _extract_from_docstring(self, docstring: str) -> Optional[str]:
        """Extract intent from docstring's first line."""
        if not docstring:
            return None
        
        first_line = docstring.split('\n')[0].strip()
        
        # Remove common prefixes
        prefixes = [
            'Returns', 'Return', 'Gets', 'Get', 'Sets', 'Set',
            'Creates', 'Create', 'Deletes', 'Delete', 
            'The', 'A', 'An'
        ]
        for prefix in prefixes:
            if first_line.startswith(prefix + ' '):
                first_line = first_line[len(prefix)+1:]
                break
        
        return first_line[:80] if first_line else None
    
    def _split_name(self, name: str) -> List[str]:
        """
        Split function name into words.
        
        Handles:
        - camelCase
        - PascalCase
        - snake_case
        - kebab-case
        - ACRONYMS (e.g., XMLParser -> XML Parser)
        """
        # Remove private prefixes
        name = name.lstrip('_').lstrip('#')
        
        # Handle kebab-case
        name = name.replace('-', '_')
        
        # snake_case
        if '_' in name:
            return [w for w in name.split('_') if w]
        
        # camelCase/PascalCase with acronym support
        # XMLParser -> XML Parser, parseXML -> parse XML
        words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
        words = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', words)
        
        return [w.strip() for w in words.split() if w.strip()]
    
    @classmethod
    def get_available_features(cls) -> dict[str, bool]:
        """
        Get dictionary of available NLP features.
        
        Returns:
            Dict with feature names and availability status
        """
        return {
            'nltk_lemmatizer': NLTK_AVAILABLE,
            'spacy': SPACY_AVAILABLE,
        }
