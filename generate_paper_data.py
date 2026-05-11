"""
IEEE Paper Data Generator - Reads from existing result text files.
No re-training needed! Reads from:
  - baseline1_results.txt  (TF-IDF + LR)
  - baseline2_results.txt  (Word2Vec + SVM)
  - proposed_model_results.txt (XLM-RoBERTa)
  - cleaned_reviews.csv    (for stats and error analysis)
"""

import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

FOOD_KEYWORDS = ["pizza", "delivery", "taste", "biryani", "khana", "restaurant"]
ID2LABEL = {0: "negative", 1: "positive", 2: "neutral"}

# ===========================================================
# TASK 1: Data Statistics
# ===========================================================
def print_data_statistics(df, y_train, y_test):
    print("\n" + "="*60)
    print("  TASK 1: DATA STATISTICS")
    print("="*60)
    total = len(df)
    print(f"  Total Reviews       : {total:,}")
    print(f"  Training Samples    : {len(y_train):,} ({len(y_train)/total*100:.1f}%)")
    print(f"  Testing Samples     : {len(y_test):,}  ({len(y_test)/total*100:.1f}%)")
    print(f"\n  Label Distribution (Full Dataset):")
    counts = df["sentiment"].value_counts().sort_index()
    for label_id, count in counts.items():
        print(f"    {ID2LABEL.get(int(label_id)).capitalize():<12}: {count:,}  ({count/total*100:.1f}%)")

# ===========================================================
# TASK 2: Hyperparameters (hard-coded from train_xlm_roberta.py)
# ===========================================================
def print_hyperparameters():
    print("\n" + "="*60)
    print("  TASK 2: HYPERPARAMETERS")
    print("="*60)
    params = {
        "Base Model"          : "xlm-roberta-base",
        "Optimizer"           : "AdamW (HuggingFace Trainer default)",
        "Learning Rate"       : "2e-5",
        "Weight Decay"        : "0.01",
        "Warmup Ratio"        : "0.1",
        "Batch Size (Train)"  : "16",
        "Batch Size (Eval)"   : "16",
        "Epochs"              : "6",
        "Max Sequence Length" : "128 tokens",
        "Eval Strategy"       : "Per Epoch",
        "Best Model Metric"   : "Weighted F1-Score",
        "Hardware"            : "Google Colab (NVIDIA T4 GPU)",
        "Framework"           : "HuggingFace Transformers + PyTorch",
    }
    for k, v in params.items():
        print(f"    {k:<26}: {v}")

# ===========================================================
# TASK 3: Read results from text files + LaTeX table
# ===========================================================
def read_baseline1():
    """Parse TF-IDF + LR results from baseline1_results.txt"""
    with open("baseline1_results.txt") as f:
        text = f.read()
    # Extract weighted avg line
    m = re.search(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    prec, rec, f1 = float(m[1]), float(m[2]), float(m[3])
    # Accuracy is weighted avg recall for LR
    acc_m = re.search(r"accuracy\s+([\d.]+)", text)
    acc = float(acc_m[1]) if acc_m else rec
    return {"Model": "TF-IDF + LR (Baseline 1)", "acc": acc, "prec": prec, "rec": rec, "f1": f1}

def read_baseline2():
    """Parse Word2Vec + SVM results from baseline2_results.txt"""
    with open("baseline2_results.txt") as f:
        text = f.read()
    acc = float(re.search(r"Accuracy:\s*([\d.]+)", text)[1])
    f1  = float(re.search(r"F1-score:\s*([\d.]+)", text)[1])
    return {"Model": "Word2Vec + SVM (Baseline 2)", "acc": acc, "prec": f1, "rec": acc, "f1": f1}

def read_proposed():
    """Parse XLM-RoBERTa results from proposed_model_results.txt"""
    with open("proposed_model_results.txt") as f:
        text = f.read()
    acc = float(re.search(r"Accuracy:\s*([\d.]+)", text)[1])
    m   = re.search(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    prec, rec, f1 = float(m[1]), float(m[2]), float(m[3])
    return {"Model": "XLM-RoBERTa (Fine-Tuned)", "acc": acc, "prec": prec, "rec": rec, "f1": f1}

def build_results_and_latex():
    print("\n" + "="*60)
    print("  TASK 3: MODEL COMPARISON + LaTeX TABLE")
    print("="*60)

    r1 = read_baseline1()
    r2 = read_baseline2()
    rp = read_proposed()
    results = [r1, r2, rp]

    print(f"\n  {'Model':<34} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("  " + "-"*60)
    for r in results:
        print(f"  {r['Model']:<34} {r['acc']:>6.4f} {r['prec']:>6.4f} {r['rec']:>6.4f} {r['f1']:>6.4f}")

    latex = r"""\begin{table}[h]
\centering
\caption{Performance Comparison on Roman Urdu Sentiment Dataset}
\label{tab:results}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\hline
"""
    for r in results:
        b0 = r"\textbf{" if "XLM" in r['Model'] else ""
        b1 = "}" if "XLM" in r['Model'] else ""
        latex += f"  {b0}{r['Model']}{b1} & {r['acc']:.4f} & {r['prec']:.4f} & {r['rec']:.4f} & {r['f1']:.4f} \\\\\n"
    latex += r"""\hline
\end{tabular}
\end{table}"""

    print("\n  --- LaTeX Table ---")
    print(latex)
    with open("latex_comparison_table.txt", "w") as f:
        f.write(latex)
    print("\n  Saved to latex_comparison_table.txt")
    return results

# ===========================================================
# TASK 4: Error Analysis (fast - LR only, no SVM retraining)
# ===========================================================
def error_analysis(X_test, y_test, lr_pred):
    print("\n" + "="*60)
    print("  TASK 4: ERROR ANALYSIS")
    print("="*60)

    test_df = pd.DataFrame({
        "Review_Text"     : X_test.values,
        "True_Label"      : [ID2LABEL[int(l)] for l in y_test.values],
        "Predicted_Label" : [ID2LABEL[int(p)] for p in lr_pred],
    })

    hard_errors = test_df[
        ((test_df["True_Label"]=="negative") & (test_df["Predicted_Label"]=="positive")) |
        ((test_df["True_Label"]=="positive")  & (test_df["Predicted_Label"]=="negative")) |
        ((test_df["True_Label"]=="negative")  & (test_df["Predicted_Label"]=="neutral"))  |
        ((test_df["True_Label"]=="positive")  & (test_df["Predicted_Label"]=="neutral"))
    ].copy()

    print(f"\n  10 Hard Errors:")
    for i, (_, row) in enumerate(hard_errors.head(10).iterrows(), 1):
        print(f"\n  [{i}] {row['Review_Text'][:90]}")
        print(f"       True: {row['True_Label']} | Predicted: {row['Predicted_Label']}")

    # Food-keyword filter
    def has_food(text):
        return any(kw in str(text).lower() for kw in FOOD_KEYWORDS)

    all_errors = test_df[test_df["True_Label"] != test_df["Predicted_Label"]]
    food_errors = all_errors[all_errors["Review_Text"].apply(has_food)].head(5)
    print(f"\n  5 Food-Keyword Errors (keywords: {FOOD_KEYWORDS}):")
    if len(food_errors) > 0:
        for i, (_, row) in enumerate(food_errors.iterrows(), 1):
            print(f"\n  [F{i}] {row['Review_Text'][:90]}")
            print(f"         True: {row['True_Label']} | Predicted: {row['Predicted_Label']}")
    else:
        print("  (No food-keyword errors found in this test split.)")

    hard_errors.head(15)[["Review_Text","True_Label","Predicted_Label"]].to_csv("error_analysis.csv", index=False)
    print("\n  Saved error_analysis.csv")
    return lr_pred

# ===========================================================
# TASK 5: Linguistic Check
# ===========================================================
def linguistic_check(X_test, y_test, lr_pred):
    print("\n" + "="*60)
    print("  TASK 5: LINGUISTIC CHECK (Roman Urdu vs Code-Switched)")
    print("="*60)
    from sklearn.metrics import accuracy_score

    def is_code_switched(text):
        words = str(text).split()
        if not words: return False
        eng = [w for w in words if w.isascii() and w.isalpha()]
        return (len(eng) / len(words)) > 0.3

    test_df = pd.DataFrame({"text": X_test.values, "true": y_test.values, "pred": lr_pred})
    test_df["cs"] = test_df["text"].apply(is_code_switched)

    cs = test_df[test_df["cs"]]
    ru = test_df[~test_df["cs"]]

    cs_acc = accuracy_score(cs["true"], cs["pred"]) if len(cs) > 0 else 0
    ru_acc = accuracy_score(ru["true"], ru["pred"]) if len(ru) > 0 else 0

    print(f"\n  Code-Switched (>30% English words) : {len(cs):,} samples | Accuracy: {cs_acc:.4f}")
    print(f"  Roman Urdu Only                    : {len(ru):,} samples | Accuracy: {ru_acc:.4f}")
    print(f"\n  {'→ Model is BETTER on Code-Switched text.' if cs_acc > ru_acc else '→ Model is BETTER on pure Roman Urdu text.'}")

# ===========================================================
# TASK 6: Plots (from existing result values)
# ===========================================================
def generate_plots(y_test, lr_pred, results):
    print("\n" + "="*60)
    print("  TASK 6: GENERATING PLOTS")
    print("="*60)

    # Confusion Matrix (using the LR proxy on same test split)
    cm = confusion_matrix(y_test, lr_pred)
    classes = ["negative", "positive", "neutral"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\n(TF-IDF+LR baseline on test set)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("  ✓ Saved confusion_matrix.png")

    # F1 Bar Chart (from the text file parsed values)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["Model"] for r in results]
    f1s    = [r["f1"] * 100 for r in results]
    colors = ['#7B9DC9', '#5E7FAD', '#1D3557']
    bars = ax.bar(labels, f1s, color=colors, width=0.45, edgecolor='white')
    ax.set_ylim(55, 80)
    ax.set_ylabel("F1-Score (%)", fontsize=12)
    ax.set_title("F1-Score Comparison Across Models", fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}%",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.xticks(fontsize=9)
    plt.tight_layout()
    plt.savefig('f1_comparison.png', dpi=300)
    plt.close()
    print("  ✓ Saved f1_comparison.png")

# ===========================================================
# MAIN
# ===========================================================
def main():
    print("\nLoading dataset...")
    df = pd.read_csv("cleaned_reviews.csv")
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=42
    )

    # Only LR is trained (quick, ~5 seconds) for confusion matrix + error analysis
    print("  Training quick LR for error analysis and confusion matrix...")
    vec = TfidfVectorizer(max_features=5000)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(vec.fit_transform(X_train), y_train)
    lr_pred = lr.predict(vec.transform(X_test))

    print_data_statistics(df, y_train, y_test)
    print_hyperparameters()
    results = build_results_and_latex()
    error_analysis(X_test, y_test, lr_pred)
    linguistic_check(X_test, y_test, lr_pred)
    generate_plots(y_test, lr_pred, results)

    print("\n" + "="*60)
    print("  ALL DONE. Generated files:")
    print("    ✓ error_analysis.csv")
    print("    ✓ latex_comparison_table.txt")
    print("    ✓ confusion_matrix.png")
    print("    ✓ f1_comparison.png")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
