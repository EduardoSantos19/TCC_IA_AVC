# train_model.py (na pasta raiz)
#!/usr/bin/env python3
"""
Script principal para treinar o modelo de classificação de AVC
"""
import sys
import os

# 🔥 ADICIONE ESTA LINHA para corrigir o path
sys.path.append('.')

from Src.data_loader import prepare_dataset
from Src.Models.models import create_cnn_model, train_model, evaluate_model, plot_training_history  # ← CORRIGIDO
from Config.config import logger

def main():
    logger.info("🎯 Iniciando treinamento do modelo de classificação de AVC")
    
    # 1. Preparar dataset
    logger.info("📦 Preparando dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    if X_train is None:
        logger.error("❌ Falha ao preparar dataset")
        return
    
    # 2. Criar modelo
    logger.info("🧠 Criando modelo CNN...")
    model = create_cnn_model(input_shape=X_train.shape[1:])
    
    # 3. Treinar modelo
    logger.info("🔥 Iniciando treinamento...")
    history, trained_model = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32
    )
    
    # 4. Avaliar modelo
    logger.info("📊 Avaliando modelo...")
    metrics = evaluate_model(trained_model, X_test, y_test)
    
    # 5. Plotar resultados
    logger.info("📈 Gerando gráficos...")
    plot_training_history(history, save_plots=True)
    
    logger.info("🎉 Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()