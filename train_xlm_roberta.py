import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ── Dataset wrapper ──────────────────────────────────────────────────────────

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


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import f1_score, accuracy_score
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv("cleaned_reviews.csv")
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    # 2. Split data 80/20 (same seed as baselines for a fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=42
    )

    # Keep index so we can identify examples later
    test_df = X_test.to_frame().join(y_test)
    test_df = test_df.reset_index(drop=True)

    # ── Logistic Regression baseline (TF-IDF) ────────────────────────────────
    print("\nTraining TF-IDF + Logistic Regression for comparison...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tr_vec = tfidf.fit_transform(X_train)
    X_te_vec = tfidf.transform(X_test)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr_vec, y_train)
    lr_preds = lr.predict(X_te_vec)
    test_df["lr_pred"] = lr_preds

    # ── Tokeniser & model ─────────────────────────────────────────────────────
    MODEL_NAME = "xlm-roberta-base"
    num_labels = 3  # 0=negative, 1=positive, 2=neutral
    id2label = {0: "negative", 1: "positive", 2: "neutral"}
    label2id = {v: k for k, v in id2label.items()}

    print(f"\nLoading tokenizer & model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ── Tokenise splits ───────────────────────────────────────────────────────
    print("Tokenizing...")
    train_dataset = ReviewDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = ReviewDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    # ── Colab Google Drive Mount ──────────────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        output_directory = "/content/drive/MyDrive/xlm_roberta_output_v3"
        print("Google Drive mounted! Training checkpoints will be saved securely to Drive.")
    except ImportError:
        output_directory = "./xlm_roberta_output_v3"
        print("Not in Google Colab. Using local output directory.")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,      # CRITICAL: Prevents disk from filling up!
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    xlm_preds = np.argmax(preds_output.predictions, axis=-1)
    test_df["xlm_pred"] = xlm_preds

    acc = accuracy_score(y_test, xlm_preds)
    report = classification_report(y_test, xlm_preds, target_names=["negative", "positive", "neutral"])

    print(f"\nAccuracy: {acc:.4f}")
    print("\n--- Classification Report ---")
    print(report)

    # ── Save proposed model results ───────────────────────────────────────────
    with open("proposed_model_results.txt", "w") as f:
        f.write("XLM-RoBERTa Fine-Tuned Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")

    print("\nResults saved to proposed_model_results.txt")

    # ── Find 10 examples where XLM-RoBERTa ✓ but LR ✗ ───────────────────────
    test_df["true_label"] = y_test.values
    xlm_correct = test_df["xlm_pred"] == test_df["true_label"]
    lr_wrong = test_df["lr_pred"] != test_df["true_label"]
    interesting = test_df[xlm_correct & lr_wrong].head(10).copy()

    result_lines = ["\n\n10 Reviews: XLM-RoBERTa Correct, Logistic Regression Wrong\n"]
    result_lines.append("=" * 60 + "\n")
    for i, (_, row) in enumerate(interesting.iterrows(), 1):
        result_lines.append(f"\nExample {i}:")
        result_lines.append(f"  Review    : {row['text'][:120]}")
        result_lines.append(f"  True Label: {id2label[row['true_label']]}")
        result_lines.append(f"  XLM-R Pred: {id2label[row['xlm_pred']]}")
        result_lines.append(f"  LR Pred   : {id2label[row['lr_pred']]}")

    comparison_text = "\n".join(result_lines)
    print(comparison_text)

    with open("proposed_model_results.txt", "a") as f:
        f.write(comparison_text)

    print("\nAll done! Results appended to proposed_model_results.txt")


if __name__ == "__main__":
    main()
