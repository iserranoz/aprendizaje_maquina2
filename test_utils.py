from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score,  precision_recall_curve, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_density(prediction, y_test):
    df = pd.DataFrame({'Score': prediction, 'Status': y_test})
    palette = {0: 'orange', 1: 'blue'}
    
    g = sns.FacetGrid(df, hue="Status", palette=palette, height=5, aspect=1.5)
    g.map(sns.kdeplot, 'Score', fill=True, common_norm=True, alpha=0.5, bw_adjust=0.6).add_legend()  
    g.set(xlim=(0, 1))
    
    g.set_axis_labels("Score", "Density")
    plt.title("Score Distribution by Status", fontsize=16)
    plt.show()

def plot_precision_recall_curve(prediction, y_test):
    
    precision, recall, thresholds = precision_recall_curve(y_test, prediction)
    
    ap_score = average_precision_score(y_test, prediction)
    df_pr = pd.DataFrame({'Precision': precision[:-1], 'Recall': recall[:-1], 'Threshold': thresholds})

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Recall', y='Precision', data=df_pr, label=f'Precision-Recall Curve (AVG PRECISION SCORE = {ap_score:.4f})', color='blue', lw=2)
    plt.fill_between(df_pr['Recall'], df_pr['Precision'], alpha=0.2, color='blue')
    plt.title("Precision-Recall Curve", fontsize=16, weight='bold')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    
    plt.show()

def plot_roc_curve(prediction, y_test):
    
    fpr, tpr, thresholds = roc_curve(y_test, prediction)
    auc_score = roc_auc_score(y_test, prediction)
    df_roc = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Threshold': thresholds})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=df_roc, label=f'ROC Curve (AUC = {auc_score:.4f})', color='blue', lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    plt.fill_between(df_roc['False Positive Rate'], df_roc['True Positive Rate'], alpha=0.2, color='blue')
    plt.title("ROC Curve", fontsize=16, weight='bold')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    plt.show()

def calculate_dr_with_ar(y_true, y_pred, AR):
    threshold = np.quantile(y_pred, (100 - AR) / 100)
    dr = 100*(1 - y_true[y_pred >= threshold].mean())
    return np.round(dr, 4), threshold

def plot_ar_vs_dr(prediction, y_test):
    AR_values = range(0, 101, 10)

    dr_current = []

    for i in AR_values:
        dr_current_value, _ = calculate_dr_with_ar(y_test,prediction, i)
        
        dr_current.append(dr_current_value)

    df_comparison = pd.DataFrame({
        'AR': AR_values,
        'Current DR': dr_current
    })
    plt.figure(figsize=(20, 12))
    plt.plot(df_comparison['AR'], df_comparison['Current DR'], marker='o', label='Default rate')

    for i, dr in enumerate(dr_current):
        plt.text(AR_values[i], dr + 0.05, f'{dr:.2f}%', ha='center', fontsize=9)
    plt.title('Comparison of Default Rate by Approval Rate')
    plt.ylabel('Default Rate (DR)')
    plt.xlabel('Approval Rate (AR)')
    plt.grid(True)
    plt.legend(loc='upper left')