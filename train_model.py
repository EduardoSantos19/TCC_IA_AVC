# train_model.py (na pasta raiz)
#!/usr/bin/env python3
"""
Script principal para treinar o modelo de REGRESSÃO para probabilidade de AVC
"""
import os
import sys

# 🔥 ADICIONE ESTAS LINHAS para corrigir o path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

try:
    from Src.data_loader import prepare_dataset_for_regression
    from Src.Models.models import create_regression_model, train_regression_model, evaluate_regression_model
    from Config.config import logger
    
    print("✅ Todos os módulos importados com sucesso!")
    
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
    sys.exit(1)

def main():
    logger.info("🎯 Iniciando treinamento do modelo de REGRESSÃO para probabilidade de AVC")
    
    # 1. Preparar dataset PARA REGRESSÃO
    logger.info("📦 Preparando dataset para REGRESSÃO...")
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    
    if X_train is None:
        logger.error("❌ Falha ao preparar dataset")
        return
    
    # 2. Criar modelo DE REGRESSÃO
    logger.info("🧠 Criando modelo de REGRESSÃO...")
    model = create_regression_model(input_shape=X_train.shape[1:])
    
    # 3. Treinar modelo DE REGRESSÃO
    logger.info("🔥 Iniciando treinamento de REGRESSÃO...")
    history, trained_model = train_regression_model(
        model, X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32
    )
    
    # 4. Avaliar modelo DE REGRESSÃO
    logger.info("📊 Avaliando modelo de REGRESSÃO...")
    metrics, y_pred = evaluate_regression_model(trained_model, X_test, y_test)
    
    logger.info("🎉 Treinamento de REGRESSÃO concluído com sucesso!")
    
    # 5. Mostrar algumas predições de exemplo
    logger.info("🔍 Exemplos de predições:")
    for i in range(min(5, len(y_test))):
        true_prob = y_test[i] * 100
        pred_prob = y_pred[i] * 100
        logger.info(f"   Amostra {i+1}: Real={true_prob:.1f}%, Predito={pred_prob:.1f}%")

if __name__ == "__main__":
    main()