# src/models/train.py
"""
Script principal de treinamento - coordena todo o pipeline
"""
import os
import numpy as np
from Config.config import logger
from Src.data_loader import prepare_dataset
from Src.Models.cnn_model import build_model, compile_model, train_with_cross_validation

def main_training_pipeline(model_type='small_cnn', use_cross_validation=True):
    """Pipeline completo de treinamento"""
    logger.info("🚀 Iniciando pipeline de treinamento")
    
    # 1. Carrega dados
    X_train, X_test, y_train, y_test = prepare_dataset()
    logger.info(f"📊 Dados carregados: {X_train.shape} treino, {X_test.shape} teste")
    
    # 2. Constroi modelo
    model = build_model(model_type)
    model = compile_model(model)
    
    # 3. Treinamento
    if use_cross_validation:
        logger.info("🔁 Treinamento com Validação Cruzada")
        histories, model_paths = train_with_cross_validation(
            lambda: build_model(model_type), 
            np.concatenate([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            k_folds=5,
            epochs=30
        )
    else:
        logger.info("🎯 Treinamento simples")
        history = model.fit(X_train, y_train,
                          validation_data=(X_test, y_test),
                          epochs=30,
                          batch_size=32,
                          verbose=1)
    
    logger.info("✅ Treinamento concluído!")
    return model

if __name__ == "__main__":
    main_training_pipeline()