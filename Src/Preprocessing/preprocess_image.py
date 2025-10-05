# src/preprocessing/preprocess_image.py
"""
Pré-processamento individual de imagens
"""
import cv2
import os
import numpy as np
from Config.config import logger, IMG_HEIGHT, IMG_WIDTH

def preprocess_single_image(image_path, output_path=None):
    """
    Pré-processa uma única imagem médica
    """
    try:
        # Carrega imagem
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Falha ao carregar imagem: {image_path}")
        
        # Redimensiona
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Equalização de histograma para melhor contraste
        img = cv2.equalizeHist(img)
        
        # Filtro Gaussiano para reduzir ruído
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Normalização
        img = img.astype(np.float32) / 255.0
        
        if output_path:
            cv2.imwrite(output_path, (img * 255).astype(np.uint8))
            
        return img
        
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        return None

def preprocess_batch(input_dir, output_dir):
    """
    Pré-processa todas as imagens de um diretório
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # CORREÇÃO: Verifica se não é None em vez de truth value
            processed_img = preprocess_single_image(input_path, output_path)
            if processed_img is not None:  # ← LINHA CORRIGIDA
                processed_count += 1
                
    logger.info(f"✅ Pré-processamento concluído: {processed_count} imagens")
    return processed_count