"""
Simple Chatbot - Bot #1
Uses TF-IDF Vectorization and Cosine Similarity
Based on Wikipedia Chatbot article
"""

import nltk
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import lem_normalize

# Greeting inputs and responses for keyword matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


class SimpleChatbot:
    def __init__(self, corpus_file):
        """
        Initialize the chatbot with a corpus file
        """
        print("Initializing Simple Chatbot...")
        
        # Read the corpus file
        with open(corpus_file, 'r', errors='ignore') as f:
            self.raw_text = f.read()
        
        # Convert to lowercase
        self.raw_text = self.raw_text.lower()
        
        # Tokenize into sentences
        self.sent_tokens = nltk.sent_tokenize(self.raw_text)
        
        print(f"Loaded {len(self.sent_tokens)} sentences from corpus")
        print("Simple Chatbot ready!\n")
    
    def greeting(self, sentence):
        """
        Check if user input is a greeting and return appropriate response
        """
        for word in sentence.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)
        return None
    
    def generate_response(self, user_input):
        """
        Generate response using TF-IDF and Cosine Similarity
        """
        # Add user input to sentence tokens
        self.sent_tokens.append(user_input)
        
        # Create TF-IDF vectorizer with our custom tokenizer
        tfidf_vectorizer = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
        
        # Generate TF-IDF matrix
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.sent_tokens)
        
        # Calculate cosine similarity between user input and all sentences
        # User input is the last item (index -1)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
        
        # Get the index of the most similar sentence (excluding the user input itself)
        # argsort() returns indices that would sort the array
        # [-2] gets the second-to-last (most similar sentence, not counting user input)
        similarity_scores_sorted = similarity_scores.argsort()[0]
        most_similar_idx = similarity_scores_sorted[-2]
        
        # Get the similarity score
        similarity_score = similarity_scores.flatten()
        similarity_score.sort()
        required_score = similarity_score[-2]
        
        # Remove user input from sent_tokens
        self.sent_tokens.pop()
        
        # If similarity is too low, we don't have a good match
        if required_score == 0:
            return "I apologize, I don't understand. Can you rephrase that?"
        else:
            return self.sent_tokens[most_similar_idx]
    
    def chat(self):
        """
        Main chat loop
        """
        print("=" * 60)
        print("SIMPLE CHATBOT (TF-IDF + Cosine Similarity)")
        print("=" * 60)
        print("Bot: Hello! I am a chatbot. I will answer your queries about chatbots.")
        print("     Type 'bye' to exit.\n")
        
        while True:
            user_input = input("You: ").lower()
            
            # Exit condition
            if user_input in ['bye', 'goodbye', 'exit', 'quit']:
                print("Bot: Goodbye! Have a great day!")
                break
            
            # Check for thanks
            if user_input in ['thanks', 'thank you', 'thanks!']:
                print("Bot: You're welcome!")
                continue
            
            # Check for greeting
            greeting_response = self.greeting(user_input)
            if greeting_response:
                print(f"Bot: {greeting_response}")
            else:
                # Generate response using TF-IDF and cosine similarity
                response = self.generate_response(user_input)
                print(f"Bot: {response}")


if __name__ == "__main__":
    # Make sure we have the required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Initialize and run the chatbot
    try:
        bot = SimpleChatbot('data/chatbots.txt')
        bot.chat()
    except FileNotFoundError:
        print("Error: Could not find 'data/chatbots.txt'")
        print("Please make sure you have created the data folder and added the Wikipedia content.")
