import pandas as pd
import os
from src.preprocessing import clean_text
from src.topic_model import TopicModelPipeline

def train():
    print("Loading data...")
    data_path = 'data/student_feedback.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    if 'text' not in df.columns:
        print("Data must contain a 'text' column.")
        return
        
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Training LDA model...")
    pipeline = TopicModelPipeline(n_topics=5, n_words=5)
    pipeline.fit(df['clean_text'])
    
    os.makedirs('models', exist_ok=True)
    pipeline.save('models/lda_model.pkl')
    print("Model saved to models/lda_model.pkl")

if __name__ == "__main__":
    train()
