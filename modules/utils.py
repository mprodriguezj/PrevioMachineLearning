# modules/utils.py
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        # Try with different separators
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')
    return df


def df_info_to_string(df):
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    return s


def train_test_split_df(df, target_col, test_size=0.2, random_state=42, stratify=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Encode categorical target if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    if stratify:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def plot_correlation_heatmap(df):
    plt.figure(figsize=(8,6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.tight_layout()
    return plt


def plot_histograms(df, max_cols=6):
    num = df.select_dtypes(include=[np.number])
    ncols = min(max_cols, num.shape[1] if num.shape[1]>0 else 1)
    nrows = int(np.ceil(num.shape[1]/ncols)) if num.shape[1]>0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols,3*nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(num.columns):
        sns.histplot(num[col].dropna(), ax=axes[i], kde=True)
        axes[i].set_title(col)
    for j in range(i+1, axes.shape[0]):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig


def plot_roc(y_test, y_score, n_classes=None):
    # y_test must be binarized if multiclass
    plt.figure(figsize=(7,6))
    if n_classes is None or n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
    else:
        # y_test shape: (n_samples, n_classes)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'class {i} (AUC={auc_val:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    return plt