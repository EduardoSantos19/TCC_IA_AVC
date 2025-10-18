# validar_modelo.py
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from keras.models import load_model
from Src.data_loader import prepare_dataset_for_regression
from Config.config import logger

def validar_modelo_completo():
    """Valida o modelo em todo o dataset de teste"""
    logger.info("🧪 VALIDAÇÃO COMPLETA DO MODELO")
    
    # 1. Carrega dados
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    
    if X_test is None:
        logger.error("❌ Falha ao carregar dados de teste")
        return
    
    # 2. Carrega modelo
    model_path = "Results/models/brain_stroke_model.keras"
    model = load_model(model_path)
    
    # 3. Faz predições
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    y_true_class = (y_test > 0.5).astype(int)
    
    # 4. Métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class)
    recall = recall_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)
    
    print(f"\n📊 MÉTRICAS DE CLASSIFICAÇÃO (threshold 0.5):")
    print(f"   ✅ Acurácia: {accuracy:.4f}")
    print(f"   ✅ Precisão: {precision:.4f}")
    print(f"   ✅ Recall: {recall:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")
    
    # 5. Matriz de confusão
    cm = confusion_matrix(y_true_class, y_pred_class)
    print(f"\n🎯 MATRIZ DE CONFUSÃO:")
    print(f"   Verdadeiros Negativos: {cm[0,0]}")  # Saudável → Saudável
    print(f"   Falsos Positivos: {cm[0,1]}")       # Saudável → AVC
    print(f"   Falsos Negativos: {cm[1,0]}")       # AVC → Saudável  
    print(f"   Verdadeiros Positivos: {cm[1,1]}")  # AVC → AVC
    
    # 6. Análise de probabilidades
    print(f"\n📈 ANÁLISE DE PROBABILIDADES:")
    print(f"   Probabilidade média AVC real: {y_pred_proba[y_true_class==1].mean():.4f}")
    print(f"   Probabilidade média Saudável real: {y_pred_proba[y_true_class==0].mean():.4f}")
    
    # 7. Exemplos de erros
    wrong_indices = np.where(y_pred_class != y_true_class)[0]
    if len(wrong_indices) > 0:
        print(f"\n🔍 EXEMPLOS DE ERROS ({len(wrong_indices)} casos):")
        for i in wrong_indices[:5]:
            true_label = "AVC" if y_true_class[i] == 1 else "Saudável"
            pred_label = "AVC" if y_pred_class[i] == 1 else "Saudável"
            confidence = y_pred_proba[i] * 100 if y_pred_class[i] == 1 else (1 - y_pred_proba[i]) * 100
            print(f"   Amostra {i}: Real={true_label}, Predito={pred_label} ({confidence:.1f}% confiança)")

if __name__ == "__main__":
    validar_modelo_completo()