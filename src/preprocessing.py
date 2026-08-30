import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Automatically download stopwords to prevent NLTK errors
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Cleans text by lowercasing, removing punctuation, and dropping stopwords."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    words = text.split()
    cleaned_words = [word for word in words if word not in STOP_WORDS]
    
    return " ".join(cleaned_words)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning to the dataframe."""
    df = df.copy()
    df['cleaned_text'] = df['ticket_text'].apply(clean_text)
    return df