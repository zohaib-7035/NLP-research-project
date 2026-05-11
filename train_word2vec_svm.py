import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def get_average_word2vec(tokens, model, vector_size):
    # Initialize an empty vector
    vector = np.zeros(vector_size)
    num_words = 0
    for word in tokens:
        if word in model.wv.key_to_index:
            vector += model.wv[word]
            num_words += 1
    if num_words > 0:
        vector /= num_words
    return vector

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv('cleaned_reviews.csv')
    df = df.dropna(subset=['text', 'sentiment'])
    
    # Tokenize the reviews by splitting on spaces
    print("Tokenizing texts...")
    texts = df['text'].astype(str).apply(lambda x: x.split()).tolist()
    labels = df['sentiment']
    
    # Train Word2Vec model
    print("Training Word2Vec model (epochs=30)...")
    vector_size = 100
    w2v_model = Word2Vec(sentences=texts, vector_size=vector_size, window=5, min_count=2, workers=4, epochs=30)
    
    # Convert reviews to average word vectors
    print("Averaging word vectors for each review...")
    X = np.array([get_average_word2vec(tokens, w2v_model, vector_size) for tokens in texts])
    y = np.array(labels)
    
    # Train test split
    print("Splitting data 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM Classifier
    print("Training SVM Classifier (this might take a few moments)...")
    clf = SVC()
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = clf.predict(X_test_scaled)
    
    # We use macro average for F1 score, or weighted. 
    # Weighted handles class imbalance seamlessly which matches the previous default report behavior.
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # Save to file
    with open('baseline2_results.txt', 'w') as f:
        f.write("Word2Vec + SVM Baseline Results\n")
        f.write("=================================\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        
    print("Metrics successfully saved to baseline2_results.txt")

if __name__ == '__main__':
    main()
