"""
Setup script to download required NLTK data
Run this ONCE before using the chatbots
"""

import nltk

print("Downloading NLTK data...")
print("This may take a few minutes...\n")

# Download required NLTK datasets
try:
    nltk.download('punkt', quiet=False)
    print("✓ punkt downloaded")
except Exception as e:
    print(f"✗ Error downloading punkt: {e}")

try:
    nltk.download('wordnet', quiet=False)
    print("✓ wordnet downloaded")
except Exception as e:
    print(f"✗ Error downloading wordnet: {e}")

try:
    nltk.download('stopwords', quiet=False)
    print("✓ stopwords downloaded")
except Exception as e:
    print(f"✗ Error downloading stopwords: {e}")

try:
    nltk.download('punkt_tab', quiet=False)
    print("✓ punkt_tab downloaded")
except Exception as e:
    print(f"✗ Error downloading punkt_tab: {e}")

print("\n" + "="*50)
print("Setup complete! You can now run the chatbots.")
print("="*50)
