"""
Módulo de modelos de rede neural para REGRESSÃO de probabilidade de AVC
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                   Dense, Dropout, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Config.config import logger, MODEL_PATH, PLOTS_DIR
import matplotlib.pyplot as plt
import numpy as np
import os

def create_regression_model(input_shape=(128, 128, 1)):
    """
    Cria uma arquitetura CNN para REGRESSÃO de probabilidade de AVC (0% a 100%)
    """
    logger.info(f"Criando modelo de REGRESSÃO com input_shape={input_shape}")
    
    model = Sequential()
    
    # Bloco 1
    model.add(Conv2D(32, (3, 3), activation='relu', 
                    input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Bloco 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Bloco 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Camadas fully connected
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # 🔥 CAMADA FINAL DE REGRESSÃO - 1 neurônio com sigmoid (0 a 1 = 0% a 100%)
    model.add(Dense(1, activation='sigmoid'))
    
    # 🔥 COMPILAÇÃO PARA REGRESSÃO
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # Para regressão entre 0-1
        metrics=['mae', 'mse']  # Mean Absolute Error, Mean Squared Error
    )
    
    logger.info("✅ Modelo de REGRESSÃO criado e compilado com sucesso")
    model.summary(print_fn=logger.info)
    
    return model

def train_regression_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Treina o modelo de REGRESSÃO e salva o melhor weights
    """
    logger.info("Iniciando treinamento do modelo de REGRESSÃO...")
    
    # 🔥 CALLBACKS PARA REGRESSÃO - Monitora val_loss
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # 🔥 MUDOU: monitora loss em vez de accuracy
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',  # 🔥 MUDOU: salva baseado na loss
            save_best_only=True,
            mode='min',  # 🔥 MUDOU: minimiza a loss
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Treinamento COM CALLBACKS
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"✅ Treinamento de REGRESSÃO concluído. Melhor modelo salvo em: {MODEL_PATH}")
    return history, model

def evaluate_regression_model(model, X_test, y_test):
    """
    Avalia o modelo de REGRESSÃO nos dados de teste
    
    Args:
        model: Modelo treinado
        X_test, y_test: Dados de teste
    
    Returns:
        metrics: Dicionário com métricas de avaliação
    """
    logger.info("Avaliando modelo de REGRESSÃO nos dados de teste...")
    
    results = model.evaluate(X_test, y_test, verbose=0)
    
    # 🔥 MÉTRICAS PARA REGRESSÃO
    metrics = {
        'loss': results[0],
        'mae': results[1],   # Mean Absolute Error
        'mse': results[2]    # Mean Squared Error
    }
    
    # Predições para calcular métricas adicionais
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calcula R² score
    from sklearn.metrics import r2_score
    metrics['r2'] = r2_score(y_test, y_pred)
    
    # Calcula acurácia baseada em threshold (opcional)
    threshold = 0.5
    y_pred_class = (y_pred > threshold).astype(int)
    y_true_class = (y_test > threshold).astype(int)
    metrics['accuracy'] = np.mean(y_pred_class == y_true_class)
    
    logger.info(f"📊 Métricas de REGRESSÃO - Loss: {metrics['loss']:.4f}, "
               f"MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")
    logger.info(f"📊 R² Score: {metrics['r2']:.4f}, "
               f"Acurácia (threshold 0.5): {metrics['accuracy']:.4f}")
    
    return metrics, y_pred

def plot_regression_training_history(history, save_plots=True):
    """
    Plota gráficos do histórico de treinamento para REGRESSÃO
    
    Args:
        history: Histórico retornado pelo model.fit()
        save_plots: Se deve salvar os gráficos
    """
    plt.figure(figsize=(15, 5))
    
    # Gráfico de Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss do Modelo (Regressão)')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.legend()
    
    # Gráfico de MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE do Modelo')
    plt.ylabel('MAE')
    plt.xlabel('Época')
    plt.legend()
    
    # Gráfico de MSE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mse'], label='Train MSE')
    if 'val_mse' in history.history:
        plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('MSE do Modelo')
    plt.ylabel('MSE')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(PLOTS_DIR, 'regression_training_history.png')
        plt.savefig(plot_path)
        logger.info(f"📈 Gráficos de REGRESSÃO salvos em: {plot_path}")
    
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, save_plots=True):
    """
    Plota predições vs valores reais para análise de regressão
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)  # Linha de referência perfeita
    
    plt.xlabel('Valor Real (Probabilidade AVC)')
    plt.ylabel('Predição (Probabilidade AVC)')
    plt.title('Predições vs Valores Reais - Regressão')
    plt.grid(True, alpha=0.3)
    
    if save_plots:
        plot_path = os.path.join(PLOTS_DIR, 'predictions_vs_actual.png')
        plt.savefig(plot_path)
        logger.info(f"📈 Gráfico predições salvo em: {plot_path}")
    
    plt.show()

# 🔥 MANTÉM as funções auxiliares (com pequenos ajustes)
def load_trained_model():
    """
    Carrega um modelo treinado salvo
    
    Returns:
        model: Modelo carregado
    """
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Modelo não encontrado em: {MODEL_PATH}")
        return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Modelo de REGRESSÃO carregado de: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return None

# 🔥 FUNÇÃO DE COMPATIBILIDADE (se precisar manter temporariamente)
def create_cnn_model(input_shape=(128, 128, 1), num_classes=3):
    """
    Função legada - mantida para compatibilidade
    """
    logger.warning("⚠️  Usando modelo LEGADO de classificação. Use create_regression_model()")
    return create_regression_model(input_shape)

# Teste do módulo
if __name__ == "__main__":
    # Teste de criação do modelo de REGRESSÃO
    model = create_regression_model()
    print("✅ Módulo de modelos de REGRESSÃO carregado com sucesso!")