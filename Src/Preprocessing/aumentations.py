# src/preprocessing/augmentations.py
"""
Camadas e funções de data augmentation para imagens médicas
"""
import numpy as np
import tensorflow as tf
from keras import layers
from Config.config import logger

def get_medical_augmentations():
    """
    Retorna augmentations específicas para imagens médicas
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),  # ±36 graus
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.GaussianNoise(0.01),  # Ruído leve
    ], name="medical_augmentations")

def apply_test_time_augmentation(model, image, n_augmentations=5):
    """
    Aplica TTA (Test Time Augmentation) para melhorar previsões
    """
    augmentations = get_medical_augmentations()
    predictions = []
    
    for _ in range(n_augmentations):
        augmented_image = augmentations(tf.expand_dims(image, 0))
        pred = model.predict(augmented_image, verbose=0)
        predictions.append(pred[0])
    
    return np.mean(predictions, axis=0)