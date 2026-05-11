import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv('cleaned_reviews.csv')
    df = df.dropna(subset=['text', 'sentiment'])
    
    X = df['text'].astype(str)
    y = df['sentiment'].astype(int)
    
    # Needs exact same split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training rapid TF-IDF + Logistic Regression model to find Hard Cases...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    
    lr_preds = lr.predict(X_test_vec)
    
    id2label = {0: "negative", 1: "positive", 2: "neutral"}
    
    # Create DataFrame for analysis
    test_df = pd.DataFrame({
        "Review_Text": X_test,
        "true_label": y_test,
        "predicted_label": lr_preds
    }).reset_index(drop=True)
    
    test_df["True_Label"] = test_df["true_label"].map(id2label)
    test_df["Predicted_Label"] = test_df["predicted_label"].map(id2label)
    
    # Find errors
    wrong_preds = test_df[test_df["true_label"] != test_df["predicted_label"]].copy()
    
    # Take 15 examples
    sample_hard_cases = wrong_preds.head(15)[["Review_Text", "True_Label", "Predicted_Label"]]
    sample_hard_cases.to_csv("error_analysis.csv", index=False)
    
    print("\n" + "="*50)
    print("SUCCESS: 15 Hard Cases saved to error_analysis.csv")
    print("="*50)
    
    # Food keyword filtering
    food_keywords = ["pizza", "delivery", "taste"]
    def contains_food_word(text):
        text_lower = str(text).lower()
        return any(keyword in text_lower for keyword in food_keywords)
        
    filtered_errors = wrong_preds[wrong_preds["Review_Text"].apply(contains_food_word)]
    
    print(f"\nFound {len(filtered_errors)} total errors across the test set containing food keywords ('pizza', 'delivery', 'taste').")
    
    if len(filtered_errors) > 0:
        print("\nHere are a few examples of food-context errors for your report:")
        samples = filtered_errors.head(5)[["Review_Text", "True_Label", "Predicted_Label"]]
        for _, row in samples.iterrows():
            print(f"- Text: '{row['Review_Text'][:100]}...'")
            print(f"  True: {row['True_Label']} | Predicted: {row['Predicted_Label']}\n")

if __name__ == "__main__":
    main()
