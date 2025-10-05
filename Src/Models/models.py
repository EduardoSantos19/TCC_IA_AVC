"""
Módulo de modelos de rede neural para classificação de imagens médicas
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

def create_cnn_model(input_shape=(128, 128, 1), num_classes=3):
    """
    Cria uma arquitetura CNN para classificação de imagens médicas
    """
    logger.info(f"Criando modelo CNN com input_shape={input_shape}, classes={num_classes}")
    
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
    model.add(Dense(256, activation='relu'))  # Reduzi de 512 para 256
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))  # Reduzi de 256 para 128
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # 🔥 CORREÇÃO: Use métricas mais simples
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # ← Só accuracy por enquanto
    
    logger.info("✅ Modelo CNN criado e compilado com sucesso")
    model.summary(print_fn=logger.info)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Treina o modelo e salva o melhor weights
    """
    logger.info("Iniciando treinamento do modelo...")
    
    # 🔥 CALLBACKS PARA SALVAR MELHOR MODELO
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=15,  # Para após 15 épocas sem melhoria
            restore_best_weights=True,  # Restaura pesos do melhor modelo
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,  # Usa o MODEL_PATH do config.py
            monitor='val_accuracy', 
            save_best_only=True,  # Salva apenas se melhorar
            mode='max',  # Maximiza a validação
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # Reduz LR pela metade
            patience=10,  # Após 10 épocas sem melhoria
            min_lr=1e-7,  # LR mínimo
            verbose=1
        )
    ]
    
    # Treinamento COM CALLBACKS
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,  # 🔥 ADICIONADO AQUI
        verbose=1
    )
    
    logger.info(f"✅ Treinamento concluído. Melhor modelo salvo em: {MODEL_PATH}")
    return history, model

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo nos dados de teste
    
    Args:
        model: Modelo treinado
        X_test, y_test: Dados de teste
    
    Returns:
        metrics: Dicionário com métricas de avaliação
    """
    logger.info("Avaliando modelo nos dados de teste...")
    
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3]
    }
    
    logger.info(f"📊 Métricas de teste - Loss: {metrics['loss']:.4f}, "
               f"Acurácia: {metrics['accuracy']:.4f}, "
               f"Precisão: {metrics['precision']:.4f}, "
               f"Recall: {metrics['recall']:.4f}")
    
    return metrics

def plot_training_history(history, save_plots=True):
    """
    Plota gráficos do histórico de treinamento
    
    Args:
        history: Histórico retornado pelo model.fit()
        save_plots: Se deve salvar os gráficos
    """
    plt.figure(figsize=(12, 4))
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()
    
    # Gráfico de loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss do Modelo')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(PLOTS_DIR, 'training_history.png')
        plt.savefig(plot_path)
        logger.info(f"📈 Gráficos salvos em: {plot_path}")
    
    plt.show()

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
        logger.info(f"✅ Modelo carregado de: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return None

# Teste do módulo
if __name__ == "__main__":
    # Teste de criação do modelo
    model = create_cnn_model()
    print("✅ Módulo de modelos carregado com sucesso!")