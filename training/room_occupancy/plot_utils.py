import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_and_save_roc(y_test, y_logits, metrics):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_logits)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    return plt
