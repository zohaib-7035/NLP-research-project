import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return text
    # lowercase everything
    text = str(text).lower()
    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def main():
    # Load the data
    df = pd.read_csv("data/Roman Urdu reviews Dataset with English translation.csv", encoding_errors="replace")
    
    # Rename columns to text and sentiment
    # Assuming we want the Roman Urdu text as the 'text' column based on the dataset name
    df = df.rename(columns={"ROMAN URDU REVIEWS": "text", "SENTIMENT": "sentiment"})
    
    # Drop the unused translation column to keep just 'text' and 'sentiment'
    if "TRANSLATED IN ENGLISH " in df.columns:
        df = df.drop(columns=["TRANSLATED IN ENGLISH "])
    elif "TRANSLATED IN ENGLISH" in df.columns:
        df = df.drop(columns=["TRANSLATED IN ENGLISH"])
        
    # Keep only target columns if they exist
    if 'text' in df.columns and 'sentiment' in df.columns:
         df = df[['text', 'sentiment']]
    else:
         print("Warning: Expected columns 'text' and 'sentiment' were not found after renaming.")
         
    # Standardize labels
    label_mapping = {
        'very positive': 1,
        'Positive': 1,
        'positive': 1,
        'very negative': 0,
        'Negative': 0,
        'negative': 0,
        'neutral': 2
    }
    
    # Apply mapping
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].astype(str).str.strip().map(label_mapping)
        
    # Clean the text: Remove special characters and lowercase everything.
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
        
    # Save this cleaned version as cleaned_reviews.csv
    df.to_csv('cleaned_reviews.csv', index=False)
    print("Cleaned data successfully saved as cleaned_reviews.csv")

if __name__ == "__main__":
    main()
