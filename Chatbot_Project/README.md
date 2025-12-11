# Chatbot Project - Week 11 Assignment
**AI for Workforce - Natural Language Processing & Computer Vision**

This project implements two different chatbot approaches:
1. **Simple TF-IDF Chatbot** - Uses cosine similarity for response generation
2. **Deep Learning Chatbot** - Uses neural networks for intent classification

---

## 📁 Project Structure

```
chatbot-project/
├── data/
│   ├── chatbots.txt          # Wikipedia corpus for Bot #1
│   └── intents.json          # Intent patterns for Bot #2
├── preprocessing.py          # Shared preprocessing functions
├── simple_chatbot.py         # Bot #1: TF-IDF implementation
├── deep_learning_chatbot.py  # Bot #2: Neural network implementation
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies

Open Command Prompt and navigate to your project folder:

```bash
pip install -r requirements.txt
```

### Step 2: Download NLTK Data

Run Python and execute:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

---

## 🤖 Bot #1: Simple TF-IDF Chatbot

### How It Works
- Uses Wikipedia article content as knowledge base
- Preprocesses text: lowercase, tokenization, lemmatization
- Vectorizes using TF-IDF (Term Frequency-Inverse Document Frequency)
- Matches user input to corpus using cosine similarity
- Returns most similar sentence as response

### Run Bot #1

```bash
python simple_chatbot.py
```

### Example Interaction
```
You: what is a chatbot
Bot: A chatbot is a software application or web interface designed to have textual or spoken conversations.

You: tell me about ELIZA
Bot: ELIZA's key method of operation involves the recognition of clue words or phrases in the input...

You: bye
Bot: Goodbye! Have a great day!
```

---

## 🧠 Bot #2: Deep Learning Chatbot

### How It Works
- Uses intent-based classification
- Neural network with 4 layers (128 → Dropout → 64 → Dropout → Output)
- Classifies user input into intents (greeting, goodbye, thanks, etc.)
- Returns appropriate response for detected intent
- Learns from training data in intents.json

### Run Bot #2

```bash
python deep_learning_chatbot.py
```

**First Time Setup:**
- Program will ask if you want to train a new model
- Type `yes` to train (takes ~2-3 minutes)
- Model saves automatically for future use

**Subsequent Runs:**
- Type `no` to load existing trained model
- Starts chatting immediately

### Example Interaction
```
You: hello
[Intent: greeting, Confidence: 0.95]
Bot: Hi there! What can I do for you?

You: tell me about AI
[Intent: AI, Confidence: 0.92]
Bot: Artificial Intelligence (AI) is the simulation of human intelligence by machines...

You: bye
[Intent: goodbye, Confidence: 0.98]
Bot: Goodbye! Have a great day!
```

---

## 📊 Understanding the Preprocessing Pipeline

Both chatbots use the same preprocessing steps (from `preprocessing.py`):

1. **Lowercase Conversion** - Normalize text case
2. **Tokenization** - Split into sentences and words
3. **Noise Removal** - Remove special characters, keep alphanumeric
4. **Stopword Removal** - Remove common words (optional)
5. **Lemmatization** - Reduce words to root form (e.g., "running" → "run")

---

## 🎯 Key Differences Between Both Bots

| Feature | Simple Chatbot | Deep Learning Chatbot |
|---------|---------------|----------------------|
| **Approach** | Pattern matching | Machine learning |
| **Data Source** | Wikipedia article | Intent patterns |
| **Training** | No training needed | Requires training |
| **Response Method** | Returns similar sentence | Generates from intent |
| **Flexibility** | Limited to corpus | Learns patterns |
| **Speed** | Fast | Moderate |

---

## 🔧 Customization

### Adding More Intents (Bot #2)

Edit `data/intents.json`:

```json
{
  "tag": "your_intent_name",
  "patterns": [
    "User input example 1",
    "User input example 2"
  ],
  "responses": [
    "Bot response option 1",
    "Bot response option 2"
  ]
}
```

### Changing Wikipedia Content (Bot #1)

Replace content in `data/chatbots.txt` with any text corpus you want the bot to learn from.

---

## 📝 Assignment Requirements Met

✅ **Preprocessing:**
- Lowercase conversion
- Tokenization (sentences and words)
- Noise removal
- Stopword removal
- Lemmatization

✅ **Vectorization:**
- TF-IDF implementation (Bot #1)
- Bag of Words (Bot #2)

✅ **Similarity/Classification:**
- Cosine similarity (Bot #1)
- Neural network classification (Bot #2)

✅ **Response Generation:**
- Keyword matching for greetings
- Document similarity matching
- Intent-based responses

---

## 🎓 Concepts Demonstrated

1. **Natural Language Processing Pipeline**
   - Text preprocessing
   - Feature extraction
   - Similarity computation

2. **Machine Learning**
   - Supervised learning
   - Neural network architecture
   - Training and evaluation

3. **Information Retrieval**
   - TF-IDF vectorization
   - Cosine similarity
   - Document ranking

4. **Intent Classification**
   - Pattern recognition
   - Multi-class classification
   - Confidence scoring

---

## 🐛 Troubleshooting

### Error: "No module named 'nltk'"
```bash
pip install nltk
```

### Error: "Resource punkt not found"
```python
import nltk
nltk.download('punkt')
```

### Error: "No such file or directory: 'data/chatbots.txt'"
- Make sure you're running the script from the project root directory
- Check that the `data` folder exists with both files

### Bot #2 Takes Long to Train
- This is normal! Neural network training takes 2-3 minutes
- Model saves after training for faster future use

---

## 📚 References

- Week 11 Slides: Building a Chatbot
- NLTK Documentation: https://www.nltk.org/
- TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html
- TensorFlow: https://www.tensorflow.org/

---

## 👩‍💻 Author

**Nas Duarte**
- Course: CAI2840C - Natural Language Processing & Computer Vision
- Institution: Miami Dade College
- Date: December 2024

---

## 💡 Future Enhancements

- Add speech recognition (Google Cloud Speech-to-Text)
- Deploy on web interface using Flask
- Integrate with messaging platforms
- Add multilingual support
- Implement conversation history
- Add sentiment analysis

---

**Need help?** Check the code comments in each Python file for detailed explanations!
