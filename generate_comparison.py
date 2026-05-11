import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from gensim.models import Word2Vec

def get_average_word2vec(tokens, model, vector_size):
    vector = np.zeros(vector_size)
    num_words = 0
    for word in tokens:
        if word in model.wv.key_to_index:
            vector += model.wv[word]
            num_words += 1
    if num_words > 0:
        vector /= num_words
    return vector

def compute_metrics_dict(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv('cleaned_reviews.csv')
    df = df.dropna(subset=['text', 'sentiment'])
    
    # 80/20 train-test split
    X_raw = df['text'].astype(str)
    y = df['sentiment']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
    results = []
    
    # --- Baseline 1: TF-IDF + Logistic Regression ---
    print("\nRunning Baseline 1: TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    
    results.append(compute_metrics_dict(y_test, lr_pred, "TF-IDF + Logistic Regression (Baseline 1)"))
    
    # --- Baseline 2: Word2Vec + SVM ---
    print("\nRunning Baseline 2: Word2Vec + SVM...")
    # Tokenize
    texts_list = df['text'].astype(str).tolist()
    tokenized_texts = [text.split() for text in texts_list]
    
    vector_size = 100
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=2, workers=4, epochs=30)
    
    # Needs purely tokenized text for training/test logic matching the overall dataset approach
    X_train_tokens = X_train_raw.apply(lambda x: x.split()).tolist()
    X_test_tokens = X_test_raw.apply(lambda x: x.split()).tolist()
    
    X_train_w2v = np.array([get_average_word2vec(tokens, w2v_model, vector_size) for tokens in X_train_tokens])
    X_test_w2v = np.array([get_average_word2vec(tokens, w2v_model, vector_size) for tokens in X_test_tokens])
    
    scaler = StandardScaler()
    X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
    X_test_w2v_scaled = scaler.transform(X_test_w2v)
    
    svm_model = SVC()
    svm_model.fit(X_train_w2v_scaled, y_train)
    svm_pred = svm_model.predict(X_test_w2v_scaled)
    
    results.append(compute_metrics_dict(y_test, svm_pred, "Word2Vec + SVM (Baseline 2)"))
    
    # --- Proposed Model: XLM-RoBERTa ---
    print("\nAdding Fine-tuned XLM-RoBERTa results (from proposed_model_results.txt)...")
    # Using weighted avg metrics parsed directly from user's result file
    results.append({
        "Model": "XLM-RoBERTa (Fine-Tuned)",
        "Accuracy": 0.7305,
        "Precision": 0.73,
        "Recall": 0.73,
        "F1-Score": 0.73
    })
    
    # --- Generate CSV ---
    print("\nSaving final results to model_comparison.csv...")
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)
    
    print("\nFinal Comparison Table:")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
