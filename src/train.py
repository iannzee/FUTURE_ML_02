import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_ticket_model(df: pd.DataFrame):
    # 1. Prepare Data
    X = df['cleaned_text']
    y_category = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_category, test_size=0.2, random_state=42
    )
    
    # 2. Convert Text to Numbers (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. Train Classification Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # 4. Evaluate
    predictions = model.predict(X_test_vec)
    print("\n--- CATEGORY CLASSIFICATION REPORT ---")
    print(classification_report(y_test, predictions))
    
    return vectorizer, model