#!/usr/bin/env python3
"""
Simple English text preprocessor
"""
import re
import string

def preprocess_english(text):
    """Simple English text preprocessing"""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra spaces
    text = ' '.join(text.split())
    return text.strip()

if __name__ == "__main__":
    # Test
    test_text = "This Product is AMAZING! I love it :) @customer #great"
    print(f"Original: {test_text}")
    print(f"Cleaned: {preprocess_english(test_text)}")
