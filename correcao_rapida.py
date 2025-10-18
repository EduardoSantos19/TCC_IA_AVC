# voltar_melhorado.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from Src.Models.models import create_regression_model, train_regression_model
from Src.data_loader import prepare_dataset_for_regression
from Config.config import logger

def treinar_modelo_conservador():
    """
    🎯 MODELO CONSERVADOR - EVITA OVER-REGULARIZAÇÃO
    """
    logger.info("🧠 Treinando modelo CONSERVADOR...")
    
    # Carregar dados
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    
    # Usar modelo original (que funcionava)
    from Src.Models.models import create_regression_model
    model = create_regression_model(input_shape=X_train.shape[1:])
    
    # Treinar com parâmetros conservadores
    history, trained_model = train_regression_model(
        model, X_train, y_train, X_test, y_test,
        epochs=50,  # Menos épocas
        batch_size=32  # Batch normal
    )
    
    # Salvar modelo
    trained_model.save('Results/models/modelo_conservador.keras')
    logger.info("💾 Modelo conservador salvo!")
    
    return trained_model

if __name__ == "__main__":
    treinar_modelo_conservador()