"""
Preprocessing Module - Shared by both chatbots
Contains all text preprocessing functions
"""

import nltk
import string
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize lemmatizer
lemmer = WordNetLemmatizer()

# Punctuation removal dictionary
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# Stop words
stop_words = set(stopwords.words('english'))


def tokenize_sentences(text):
    """
    Convert text into list of sentences
    """
    return nltk.sent_tokenize(text)


def tokenize_words(text):
    """
    Convert text into list of words
    """
    return nltk.word_tokenize(text)


def remove_noise(text):
    """
    Remove everything that isn't a letter or number
    """
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def remove_stopwords(tokens):
    """
    Remove stopwords from tokenized text
    Note: Be careful with words like 'no', 'not' which can change meaning
    """
    return [word for word in tokens if word.lower() not in stop_words]


def lemmatize_tokens(tokens):
    """
    Reduce words to their root form
    """
    return [lemmer.lemmatize(token) for token in tokens]


def preprocess_text(text, remove_stops=True):
    """
    Complete preprocessing pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Remove noise
    4. Tokenize
    5. Remove stopwords (optional)
    6. Lemmatize
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(remove_punct_dict)
    
    # Remove noise
    text = remove_noise(text)
    
    # Tokenize
    tokens = tokenize_words(text)
    
    # Remove stopwords if requested
    if remove_stops:
        tokens = remove_stopwords(tokens)
    
    # Lemmatize
    tokens = lemmatize_tokens(tokens)
    
    return tokens


def lem_normalize(text):
    """
    Normalize text for TF-IDF vectorization
    Used specifically for the simple chatbot
    """
    return lemmatize_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


if __name__ == "__main__":
    # Test the preprocessing
    test_text = "Hello! How are you doing today? I'm working on my chatbot project."
    print("Original text:", test_text)
    print("\nPreprocessed tokens:", preprocess_text(test_text))
