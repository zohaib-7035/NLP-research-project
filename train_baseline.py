import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    # 1. Load data
    print("Loading cleaned dataset...")
    df = pd.read_csv('cleaned_reviews.csv')
    
    # Drop any stray NaNs just in case
    df = df.dropna(subset=['text', 'sentiment'])
    
    # 2. Split data 80/20
    X = df['text']
    y = df['sentiment']
    print("Splitting data 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. TF-IDF vectorization
    print("Vectorizing text using TfidfVectorizer (max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Train Logistic Regression
    print("Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    # 5. Evaluate and print Classification Report
    print("Evaluating model...")
    y_pred = clf.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    
    print("\n--- Classification Report ---")
    print(report)
    
    # 6. Save metrics
    with open('baseline1_results.txt', 'w') as f:
        f.write("Logistic Regression Baseline Results\n")
        f.write("="*40 + "\n")
        f.write(report)
        f.write("\n")
    print("Metrics successfully saved to baseline1_results.txt")

if __name__ == "__main__":
    main()
