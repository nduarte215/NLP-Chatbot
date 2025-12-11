"""
Deep Learning Chatbot - Bot #2
Uses Neural Network for Intent Classification
Based on intents.json with patterns and responses
"""

import json
import random
import numpy as np
import nltk
from preprocessing import preprocess_text

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import pickle


class DeepLearningChatbot:
    def __init__(self, intents_file='data/intents.json'):
        """
        Initialize the deep learning chatbot
        """
        print("Initializing Deep Learning Chatbot...")
        
        # Load intents
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.words = []
        self.classes = []
        self.training_data = []
        
        print("Deep Learning Chatbot initialized!\n")
    
    def prepare_training_data(self):
        """
        Prepare training data from intents
        """
        print("Preparing training data...")
        
        all_words = []
        all_classes = []
        documents = []
        
        # Loop through each intent
        for intent in self.intents['intents']:
            # Get the tag (class/label)
            tag = intent['tag']
            all_classes.append(tag)
            
            # Loop through each pattern
            for pattern in intent['patterns']:
                # Preprocess the pattern
                tokens = preprocess_text(pattern, remove_stops=False)
                all_words.extend(tokens)
                documents.append((tokens, tag))
        
        # Remove duplicates and sort
        self.words = sorted(set(all_words))
        self.classes = sorted(set(all_classes))
        
        print(f"Found {len(self.words)} unique words")
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Create training data
        training_sentences = []
        training_labels = []
        
        for document in documents:
            tokens, tag = document
            
            # Create bag of words
            bag = []
            for word in self.words:
                bag.append(1 if word in tokens else 0)
            
            training_sentences.append(bag)
            training_labels.append(tag)
        
        # Convert to numpy arrays
        self.training_data = np.array(training_sentences)
        
        # Encode labels
        self.training_labels_encoded = self.label_encoder.fit_transform(training_labels)
        
        print(f"Training data shape: {self.training_data.shape}")
        print("Training data prepared!\n")
        
        return self.training_data, self.training_labels_encoded
    
    def build_model(self):
        """
        Build the 4-layer neural network model
        """
        print("Building neural network model...")
        
        input_shape = len(self.words)
        output_shape = len(self.classes)
        
        self.model = Sequential([
            Dense(128, input_shape=(input_shape,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(output_shape, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        print()
    
    def train(self, epochs=200, batch_size=8):
        """
        Train the model
        """
        print(f"Training model for {epochs} epochs...")
        
        # Prepare training data
        X_train, y_train = self.prepare_training_data()
        
        # Build model
        self.build_model()
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        print("\nTraining complete!")
        print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return history
    
    def predict_intent(self, user_input):
        """
        Predict the intent of user input
        """
        # Preprocess input
        tokens = preprocess_text(user_input, remove_stops=False)
        
        # Create bag of words
        bag = []
        for word in self.words:
            bag.append(1 if word in tokens else 0)
        
        # Convert to numpy array
        bag = np.array([bag])
        
        # Predict
        prediction = self.model.predict(bag, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]
        
        # Get the intent tag
        predicted_intent = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_intent, confidence
    
    def get_response(self, intent):
        """
        Get a random response for the given intent
        """
        for i in self.intents['intents']:
            if i['tag'] == intent:
                return random.choice(i['responses'])
        
        return "I'm not sure how to respond to that."
    
    def save_model(self, filename='chatbot_model'):
        """
        Save the trained model and preprocessors
        """
        self.model.save(f'{filename}.keras')
        
        # Save words, classes, and label encoder
        with open(f'{filename}_data.pkl', 'wb') as f:
            pickle.dump({
                'words': self.words,
                'classes': self.classes,
                'label_encoder': self.label_encoder
            }, f)
        
        print(f"Model saved as {filename}.keras")
    
    def load_model(self, filename='chatbot_model'):
        """
        Load a trained model
        """
        self.model = keras.models.load_model(f'{filename}.keras')
        
        with open(f'{filename}_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.words = data['words']
            self.classes = data['classes']
            self.label_encoder = data['label_encoder']
        
        print(f"Model loaded from {filename}.keras")
    
    def chat(self):
        """
        Main chat loop
        """
        print("=" * 60)
        print("DEEP LEARNING CHATBOT (Neural Network Intent Classifier)")
        print("=" * 60)
        print("Bot: Hello! I'm an AI-powered chatbot. How can I help you?")
        print("     Type 'bye' to exit.\n")
        
        while True:
            user_input = input("You: ")
            
            # Exit condition
            if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
                print("Bot: Goodbye! Have a great day!")
                break
            
            # Predict intent
            intent, confidence = self.predict_intent(user_input)
            
            print(f"[Intent: {intent}, Confidence: {confidence:.2f}]")
            
            # Get response
            if confidence > 0.5:  # Confidence threshold
                response = self.get_response(intent)
            else:
                response = "I'm not quite sure what you mean. Could you rephrase that?"
            
            print(f"Bot: {response}\n")


if __name__ == "__main__":
    # Make sure we have the required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        # Initialize chatbot
        bot = DeepLearningChatbot('data/intents.json')
        
        # Train the model
        print("Would you like to train a new model? (yes/no)")
        choice = input().lower()
        
        if choice in ['yes', 'y']:
            bot.train(epochs=200, batch_size=8)
            bot.save_model('chatbot_model')
        else:
            try:
                bot.load_model('chatbot_model')
            except:
                print("No saved model found. Training new model...")
                bot.train(epochs=200, batch_size=8)
                bot.save_model('chatbot_model')
        
        # Start chatting
        bot.chat()
        
    except FileNotFoundError:
        print("Error: Could not find 'data/intents.json'")
        print("Please make sure you have created the intents file.")
