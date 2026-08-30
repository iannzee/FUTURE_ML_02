import pandas as pd
from src.preprocessing import preprocess_data
from src.train import train_ticket_model

def create_mock_tickets():
    """Generates sample customer support tickets."""
    data = {
        'ticket_text': [
            "I can't log into my account, password reset is broken.",
            "My credit card was charged twice this month!",
            "The app crashes when I try to export my report to PDF.",
            "Can I get a copy of my last invoice for my taxes?",
            "How do I change my profile picture?",
            "Urgent: Servers are down and we cannot access the platform."
        ] * 10,  # Multiplying to create enough data for training
        'category': [
            'Account', 'Billing', 'Technical', 'Billing', 'General', 'Technical'
        ] * 10
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    print("Loading support tickets...")
    df = create_mock_tickets()
    
    print("Cleaning text data...")
    processed_df = preprocess_data(df)
    
    print("Training NLP models...")
    vectorizer, model = train_ticket_model(processed_df)
    
    print("Pipeline Complete! Models trained successfully.")