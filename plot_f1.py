import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_f1_plot():
    # Set standard IEEE style plotting (clean, formal)
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    # Load the CSV generated earlier
    try:
        df = pd.read_csv('model_comparison.csv')
    except FileNotFoundError:
        print("Error: model_comparison.csv not found.")
        return

    # Extract Model names and F1-Scores
    models = df['Model']
    f1_scores = df['F1-Score'] * 100 # Convert to percentage

    # Create Bar Chart
    colors = ['#8EA4D2', '#6279B8', '#49516F'] # Professional blues/grays
    bars = plt.bar(models, f1_scores, color=colors, width=0.5)

    # Decorate plot
    plt.title('Final Model F1-Score Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('F1-Score (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(fontsize=10)
    
    # Add exact value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('f1_comparison.png', dpi=300) # Save high res for paper!
    print("SUCCESS: F1 Comparison chart saved as f1_comparison.png")

if __name__ == "__main__":
    generate_f1_plot()
