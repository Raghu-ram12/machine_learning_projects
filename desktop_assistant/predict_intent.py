import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("training_data.csv")


data = data.dropna(subset=['text', 'intent'])
data['text'] = data['text'].astype(str).str.strip()
data['intent'] = data['intent'].astype(str).str.strip()

print(f"Cleaned data: {len(data)} samples")
print(data['intent'].value_counts())

# SPLIT
x_train, x_test, y_train, y_test = train_test_split(  # Fixed order!
    data['text'], data['intent'], 
    test_size=0.2, 
    stratify=data['intent'], 
    random_state=42
)

# converting text int numerical values

vectorizer = TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# converting text labels into numerical values
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# training a logistic regression multi classifier 

model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model.fit(x_train_vec, y_train_enc)

y_predict_enc = model.predict(x_test_vec)

print(model.classes_)

def predict_intent(text):

    text=text.lower().strip()
    clean_input=[text]
    text_vec=vectorizer.transform(clean_input)

    predict_enc=model.predict(text_vec)

    intent=le.inverse_transform(predict_enc)

    print(intent)


