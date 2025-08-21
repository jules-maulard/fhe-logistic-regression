import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from typing import Tuple

def print_classification_metrics(
    y_true, y_proba, 
    show_report=True,
    show_matrix=True,  
    show_mse_nmse=True,
):
    y_pred = (y_proba > 0.5).astype(int).flatten()

    accuracy = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_proba)
    print(f"\nACC: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    if show_report:
        print(classification_report(y_true, y_pred))
    if show_matrix:
        print(confusion_matrix(y_true, y_pred))
    if show_mse_nmse:
        mse = mean_squared_error(y_true, y_pred)
        nmse = mse / np.var(y_true)
        print(f"\nMSE: {mse:.4f}")
        print(f"NMSE: {nmse:.4f}\n")

def calculate_acc_auc_idash_style(z_data: np.ndarray, w_data: np.ndarray) -> Tuple[float, float]:
    sample_dim, factor_dim = z_data.shape
    
    TN = 0
    FP = 0
    theta_TN = []
    theta_FP = []
    
    def true_ip(x, w, dim):
        return np.dot(x[:dim], w[:dim])
    
    for i in range(sample_dim):
        if z_data[i, 0] > 0:
            if true_ip(z_data[i], w_data, factor_dim) < 0:
                TN += 1
            margin = z_data[i, 0] * true_ip(z_data[i, 1:], w_data[1:], factor_dim - 1)
            theta_TN.append(margin)
            
        else: 
            if true_ip(z_data[i], w_data, factor_dim) < 0:
                FP += 1             
            margin = z_data[i, 0] * true_ip(z_data[i, 1:], w_data[1:], factor_dim - 1)
            theta_FP.append(margin)
    
    correctness = 100.0 - (100.0 * (FP + TN) / sample_dim)
    print(f"ACC: {correctness:.4f} %.")
    
    if len(theta_FP) == 0 or len(theta_TN) == 0:
        print("n_test_yi = 0 : cannot compute AUC")
        auc = 0.0
    else:
        auc = 0.0
        for i in range(len(theta_TN)):
            for j in range(len(theta_FP)):
                if theta_FP[j] <= theta_TN[i]:
                    auc += 1
        auc /= len(theta_TN) * len(theta_FP)
        print(f"AUC: {auc:.4f}")
    
    return correctness, auc


def predict_proba(Z_train, beta, sigmoid):
    linear_pred = np.dot(Z_train, beta)
    y_pred = 1 - sigmoid(linear_pred)
    return y_pred

def beta_to_proba_history(beta_history, Z_train, sigmoid):
    proba_history = []
    
    for beta in beta_history:
        y_pred = predict_proba(Z_train, beta, sigmoid)
        proba_history.append(y_pred)
    
    return proba_history

def compute_loss(y_proba, y_train):
    y_pred = np.clip(y_proba, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
    return loss
    

def beta_to_loss_history(beta_history, Z_train, y_train, sigmoid):
    loss_history = []
    proba_history = beta_to_proba_history(beta_history, Z_train, sigmoid)
    
    for proba in proba_history:
        loss = compute_loss(proba, y_train)
        loss_history.append(loss)
    
    return loss_history

def plot_loss_evolution(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss evolution')
    plt.grid(True, alpha=0.7)
    plt.show()

def plot_proba_distrib(proba):
    plt.figure(figsize=(12, 5))
    plt.hist(proba, bins=50, alpha=0.7)
    plt.xlabel("Predicted probabilities")
    plt.title("Predictions distribution")
    plt.show()