# src/utils/helpers.py
"""
Funções auxiliares e utilitários
"""
import numpy as np
import matplotlib.pyplot as plt
from Config.config import logger

def plot_training_history(history, save_path=None):
    """Plota histórico de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss durante Treinamento')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Acurácia durante Treinamento')
    axes[1].legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_medical_metrics(y_true, y_pred):
    """Calcula métricas médicas específicas"""
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, 
                                 target_names=['AVC', 'Normal', 'Desconhecido'])
    
    # Métricas específicas para medicina
    sensitivity = cm[0,0] / (cm[0,0] + cm[0,1] + cm[0,2])  # True Positive Rate
    specificity = (cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2]) / (cm[1:3, :].sum())  # True Negative Rate
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'sensitivity': sensitivity,
        'specificity': specificity
    }