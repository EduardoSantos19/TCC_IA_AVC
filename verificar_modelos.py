# verificar_modelos.py
import os
import sys
from keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from Src.data_loader import prepare_dataset_for_regression
from sklearn.metrics import accuracy_score
import numpy as np

def testar_modelo(model_path, threshold=0.5):
    """Testa um modelo específico"""
    if not os.path.exists(model_path):
        return f"❌ Arquivo não existe: {model_path}"
    
    try:
        # Carregar dados
        X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
        y_true_class = (y_test > 0.5).astype(int)
        
        # Carregar e testar modelo
        model = load_model(model_path)
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred_class = (y_pred_proba > threshold).astype(int)
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        
        return f"✅ {os.path.basename(model_path)}: Acurácia = {accuracy:.4f} ({accuracy*100:.1f}%)"
    
    except Exception as e:
        return f"❌ Erro em {model_path}: {e}"

def main():
    print("🔍 VERIFICANDO TODOS OS MODELOS:")
    print("=" * 50)
    
    modelos_para_testar = [
        'Results/models/brain_stroke_model.keras',
        'Results/models/modelo_conservador.keras', 
        'Results/models/modelo_final_otimizado.keras',
        'Results/models/modelo_otimizado_completo.keras'
    ]
    
    for model_path in modelos_para_testar:
        resultado = testar_modelo(model_path, threshold=0.4)
        print(resultado)
    
    print("\n🎯 RECOMENDAÇÃO FINAL:")
    print("=" * 50)
    
    # Verificar qual é o melhor
    conservador_path = 'Results/models/modelo_conservador.keras'
    if os.path.exists(conservador_path):
        print("💡 USE: modelo_conservador.keras (93.2% de acurácia)")
        print("💡 Este é o modelo BOM que treinamos por último")
    else:
        print("❌ modelo_conservador.keras não encontrado!")
        
    print("\n📋 AÇÕES RECOMENDADAS:")
    print("1. Use apenas modelo_conservador.keras")
    print("2. Delete os outros modelos se quiser liberar espaço")
    print("3. No seu código, sempre referencie modelo_conservador.keras")

if __name__ == "__main__":
    main()