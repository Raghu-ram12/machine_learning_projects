import pandas as pd
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# 6 Intents
intents = ['time', 'date', 'weather', 'open_app', 'search', 'volume','exit','greet']

# Base phrases + smart variations
def generate_phrases(intent, n_phrases=100):
    base_phrases = {
        'time': ['what time is it', 'tell me the time', 'current time', 'show time', 'what hour', 'clock time'],
        'date': ['what date is it', 'today date', 'current date', 'show date', 'what day today', 'calendar date'],
        'weather': ['what is weather', 'current weather', 'weather today', 'show weather', 'temperature now', 'weather report'],
        'open_app': ['open notepad', 'launch chrome', 'start calculator', 'open browser', 'run vscode', 'open terminal'],
        'search': ['search python', 'google machine learning', 'find nlp', 'search leetcode', 'what is sklearn', 'google tutorial'],
        'volume': ['increase volume', 'volume up', 'decrease volume', 'volume down', 'mute sound', 'make louder'],
        'exit':['good bye','go to sleep','shut down','bye','sleep',"leave"],
        'greet':['hello','hi','whats up','good morning','good afternoon','good afternoon']
    }
    
    phrases = base_phrases[intent]
    variations = []
    
    # Generate variations
    prefixes = ['please ', 'show me ', 'tell me ', 'what\'s ', 'now ', '']
    suffixes = [' now', ' please', '', '?']
    synonyms = {
        'open': ['launch', 'start', 'run'],
        'volume': ['sound', 'audio'],
        'what': ['tell me', 'show',"who"],
        'exit':['close','shut'],
        'greet':['hello','hi']
    }
    
    while len(variations) < n_phrases:
        base = random.choice(phrases)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        # Apply synonyms
        phrase = prefix + base + suffix
        if random.random() < 0.3:  # 30% chance
            for word, syns in synonyms.items():
                if word in phrase.lower():
                    phrase = phrase.replace(word, random.choice(syns))
        
        if phrase not in variations:
            variations.append(phrase)
    
    return variations

# Generate dataset
data = []
for intent in intents:
    phrases = generate_phrases(intent, 100)
    data.extend([(phrase, intent) for phrase in phrases])

# Create DataFrame (exactly 600 rows, 100 per class)
df = pd.DataFrame(data, columns=['text', 'intent'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV
df.to_csv('training_data.csv', index=False)
print("✅ Generated desktop_assistant_600.csv")
print(f"Shape: {df.shape}")
print("\nClass Balance:")
print(df['intent'].value_counts())
print("\nFirst 10 samples:")
print(df.head(10))

