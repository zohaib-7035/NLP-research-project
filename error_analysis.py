import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def main():
    # 1. Load Data & Do Same Split as Before
    print("Loading cleaned dataset...")
    df = pd.read_csv("cleaned_reviews.csv")
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    # 20% test split, state 42. So test set is exactly the same
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=42
    )

    test_df = pd.DataFrame({"text": X_test, "true_label": y_test}).reset_index(drop=True)

    # 2. Load Model & Tokenizer
    try:
        from google.colab import drive
        # Only attempt to mount if running in Colab
        drive.mount('/content/drive')
        model_dir = "/content/drive/MyDrive/xlm_roberta_output_v3"
        print(f"Mounted Google Drive. Looking for model at: {model_dir}")
    except ImportError:
        model_dir = "./xlm_roberta_output_v3"
        print(f"Not via Colab. Looking for model locally at: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found!")
        print("Please ensure this script runs where the fine-tuned model weights are saved (e.g., Google Colab).")
        return

    print("Loading Fine-Tuned Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Enable eval mode
    model.eval()

    # 3. Create Dataset and Trainer for Prediction
    print("Preparing test dataset and running predictions...")
    test_dataset = ReviewDataset(test_df["text"].tolist(), test_df["true_label"].tolist(), tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=16,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )
    
    preds_output = trainer.predict(test_dataset)
    predicted_labels = np.argmax(preds_output.predictions, axis=-1)
    
    test_df["predicted_label"] = predicted_labels

    id2label = {0: "negative", 1: "positive", 2: "neutral"}

    # Map target numbers to text labels
    test_df["True_Label"] = test_df["true_label"].map(id2label)
    test_df["Predicted_Label"] = test_df["predicted_label"].map(id2label)
    test_df = test_df.rename(columns={"text": "Review_Text"})

    # 3.5 Generate Confusion Matrix Plot
    print("\nGenerating Confusion Matrix Plot...")
    cm = confusion_matrix(test_df["true_label"], test_df["predicted_label"])
    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["negative", "positive", "neutral"], 
                yticklabels=["negative", "positive", "neutral"])
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.title('XLM-RoBERTa Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("SUCCESS: Saved confusion_matrix.png!")

    # 4. Find Hard Cases (Wrong Predictions)
    print("Extracting hard cases...")
    wrong_preds = test_df[test_df["true_label"] != test_df["predicted_label"]].copy()

    # Get exactly 15 hard cases
    sample_hard_cases = wrong_preds.head(15)[["Review_Text", "True_Label", "Predicted_Label"]]

    # Save to CSV
    sample_hard_cases.to_csv("error_analysis.csv", index=False)
    print("\nSaved 15 hard cases to error_analysis.csv")
    print(sample_hard_cases.to_string())

    # 5. Filter for Specific Food Keywords
    food_keywords = ["pizza", "delivery", "taste"]
    
    print("\nFiltering hard cases for food keywords: 'pizza', 'delivery', or 'taste'...")
    def contains_food_word(text):
        text_lower = str(text).lower()
        return any(keyword in text_lower for keyword in food_keywords)
        
    filtered_errors = wrong_preds[wrong_preds["Review_Text"].apply(contains_food_word)]
    
    print(f"\nFound {len(filtered_errors)} total errors across the test set containing food keywords.")
    if len(filtered_errors) > 0:
        print("Here are some examples of food-related errors:")
        print(filtered_errors.head()[["Review_Text", "True_Label", "Predicted_Label"]].to_string())

if __name__ == "__main__":
    main()
