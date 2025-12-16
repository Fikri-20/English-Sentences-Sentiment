import re
import string
import emoji
import pandas as pd
import numpy as np
from typing import List, Dict
import nltk
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar
from camel_tools.utils.dediac import dediac_ar

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline for English sentiment analysis"""
    
    def __init__(self, 
                 remove_diacritics=True,
                 normalize_text=True,
                 remove_urls=True,
                 remove_emails=True,
                 remove_mentions=True,
                 remove_hashtags=False,
                 remove_emojis=False,
                 remove_english=False,
                 remove_numbers=False,
                 remove_punctuation=True,
                 remove_extra_spaces=True):
        """
        Initialize preprocessor with configuration
        
        Args:
            remove_diacritics: Remove Arabic diacritics (تشكيل)
            normalize_text: Normalize Arabic characters
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_emojis: Remove emojis
            remove_english: Remove English characters
            remove_numbers: Remove numbers
            remove_punctuation: Remove punctuation
            remove_extra_spaces: Remove extra whitespaces
        """
        self.remove_diacritics = remove_diacritics
        self.normalize_text = normalize_text
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emojis = remove_emojis
        self.remove_english = remove_english
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_extra_spaces = remove_extra_spaces
        
        # English punctuation marks
        self.punctuation = string.punctuation
        
        # Common English stopwords
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def remove_diacritics_func(self, text: str) -> str:
        """Remove diacritics (not typically used for English)"""
        # For English, this mainly handles accented characters
        return text
    
    def normalize_text_func(self, text: str) -> str:
        """Normalize English text"""
        # Normalize common contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        return text
    
    def remove_urls_func(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails_func(self, text: str) -> str:
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def remove_mentions_func(self, text: str) -> str:
        """Remove @mentions"""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags_func(self, text: str) -> str:
        """Remove #hashtags"""
        return re.sub(r'#\w+', '', text)
    
    def remove_emojis_func(self, text: str) -> str:
        """Remove emojis"""
        return emoji.replace_emoji(text, replace='')
    
    def remove_english_func(self, text: str) -> str:
        """Remove English characters (not used for English sentiment analysis)"""
        # Keep this as placeholder but return text as-is for English analysis
        return text
    
    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text"""
        # Remove all numbers
        text = re.sub(r'\d+', '', text)
        return text
    
    def remove_punctuation_func(self, text: str) -> str:
        """Remove punctuation marks"""
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def remove_extra_spaces_func(self, text: str) -> str:
        """Remove extra whitespaces"""
        return ' '.join(text.split())
    
    def remove_stopwords(self, text: str) -> str:
        """Remove English stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Apply preprocessing steps in order
        if self.remove_urls:
            text = self.remove_urls_func(text)
        
        if self.remove_emails:
            text = self.remove_emails_func(text)
        
        if self.remove_mentions:
            text = self.remove_mentions_func(text)
        
        if self.remove_hashtags:
            text = self.remove_hashtags_func(text)
        
        if self.remove_emojis:
            text = self.remove_emojis_func(text)
        
        if self.remove_diacritics:
            text = self.remove_diacritics_func(text)
        
        if self.normalize_text:
            text = self.normalize_text_func(text)
        
        if self.remove_english:
            text = self.remove_english_func(text)
        
        if self.remove_numbers:
            text = self.remove_numbers_func(text)
        
        if self.remove_punctuation:
            text = self.remove_punctuation_func(text)
        
        if self.remove_extra_spaces:
            text = self.remove_extra_spaces_func(text)
        
        return text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


def preprocess_dataset(input_path: str, output_path: str, text_column: str = 'text', **kwargs):
    """
    Preprocess a dataset from CSV
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save preprocessed CSV
        text_column: Name of the text column
        **kwargs: Additional arguments for ArabicTextPreprocessor
    """
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Initialize preprocessor
    preprocessor = ArabicTextPreprocessor(**kwargs)
    
    # Preprocess text column
    print(f"Preprocessing {len(df)} texts...")
    df[f'{text_column}_clean'] = preprocessor.preprocess_batch(df[text_column].tolist())
    
    # Remove empty texts
    original_len = len(df)
    df = df[df[f'{text_column}_clean'].str.len() > 0]
    print(f"Removed {original_len - len(df)} empty texts")
    
    # Save preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to {output_path}")
    
    return df


def analyze_text_stats(texts: List[str]) -> Dict:
    """
    Analyze statistics of text data
    
    Args:
        texts: List of texts
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_texts': len(texts),
        'avg_length': np.mean([len(text.split()) for text in texts]),
        'max_length': max([len(text.split()) for text in texts]),
        'min_length': min([len(text.split()) for text in texts]),
        'total_words': sum([len(text.split()) for text in texts]),
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    sample_text = "مرحبا بك في تطبيق تحليل المشاعر العربية! 😊 http://example.com"
    
    preprocessor = ArabicTextPreprocessor()
    clean_text = preprocessor.preprocess(sample_text)
    
    print("Original:", sample_text)
    print("Cleaned:", clean_text)
