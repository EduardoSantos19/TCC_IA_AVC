import os
import cv2
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from Config.config import (
    logger, PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, 
    PROCESSED_UNKNOWN_DIR, IMG_HEIGHT, IMG_WIDTH,
    RAW_POSITIVE_DIR, RAW_NEGATIVE_DIR, RAW_UNKNOWN_DIR
)

def check_if_processed_images_exist():
    """Verifica se existem imagens nas pastas processed"""
    processed_dirs = [PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR]
    
    for dir_path in processed_dirs:
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm'))]
            if len(images) > 0:
                return True
    return False

def load_images_from_folder(folder, label, image_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Carrega imagens de uma pasta, converte para grayscale e redimensiona."""
    images = []
    labels = []
    ignored_count = 0

    logger.info(f"Carregando imagens da pasta: {folder}")

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm")

    try:
        # Verifica se a pasta existe
        if not os.path.exists(folder):
            logger.warning(f"Pasta não encontrada: {folder}")
            return images, labels

        for filename in os.listdir(folder):
            if not filename.lower().endswith(valid_extensions):
                continue

            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                ignored_count += 1
                continue

            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)

        logger.info(f"✅ {len(images)} imagens carregadas de {folder}")
        if ignored_count > 0:
            logger.info(f"⚠ {ignored_count} imagens ignoradas em {folder}")

    except Exception as e:
        logger.error(f"❌ Erro ao carregar imagens de {folder}: {e}")

    return images, labels

def preprocess_if_needed():
    """Executa o pré-processamento se as pastas processed estiverem vazias"""
    if not check_if_processed_images_exist():
        logger.info("Nenhuma imagem processada encontrada. Executando pré-processamento...")
        try:
            # CORREÇÃO: Adiciona o caminho correto para importação
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(current_dir, '..')
            if src_path not in sys.path:
                sys.path.append(src_path)
            
            # Tenta importar e executar o pré-processamento
            from Preprocessing.preprocess_image import preprocess_batch
            
            # Processa cada classe
            classes_to_process = {
                'positive': (RAW_POSITIVE_DIR, PROCESSED_POSITIVE_DIR),
                'negative': (RAW_NEGATIVE_DIR, PROCESSED_NEGATIVE_DIR),
                'unknown': (RAW_UNKNOWN_DIR, PROCESSED_UNKNOWN_DIR)
            }
            
            total_processed = 0
            for class_name, (input_dir, output_dir) in classes_to_process.items():
                if os.path.exists(input_dir) and len(os.listdir(input_dir)) > 0:
                    logger.info(f"Processando imagens {class_name}...")
                    count = preprocess_batch(input_dir, output_dir)
                    total_processed += count
                    logger.info(f"✅ {count} imagens {class_name} processadas")
                else:
                    logger.warning(f"Pasta vazia ou não encontrada: {input_dir}")
            
            if total_processed == 0:
                logger.error("❌ Nenhuma imagem foi processada. Verifique as pastas raw/")
                return False
                
            return True
                    
        except ImportError as e:
            logger.error(f"Erro ao importar módulo de pré-processamento: {e}")
            logger.error(f"sys.path: {sys.path}")
            logger.error(f"Diretório atual: {os.getcwd()}")
            return False
        except Exception as e:
            logger.error(f"Erro durante pré-processamento: {e}")
            return False
    return True

def prepare_dataset(test_size=0.2, image_size=(IMG_HEIGHT, IMG_WIDTH), include_unknown=True):
    """Prepara o dataset completo com 3 classes: positivo, negativo e desconhecido"""
    logger.info("Iniciando preparação do dataset com 3 classes...")

    # Verifica e executa pré-processamento se necessário
    if not preprocess_if_needed():
        logger.error("❌ Falha no pré-processamento. Verifique as pastas raw/")
        return None, None, None, None

    # Carrega imagens das três classes
    pos_images, pos_labels = load_images_from_folder(PROCESSED_POSITIVE_DIR, 0, image_size)  # Classe 0: AVC
    neg_images, neg_labels = load_images_from_folder(PROCESSED_NEGATIVE_DIR, 1, image_size)  # Classe 1: Saudável
    
    if include_unknown:
        unknown_images, unknown_labels = load_images_from_folder(PROCESSED_UNKNOWN_DIR, 2, image_size)  # Classe 2: Desconhecido
    else:
        unknown_images, unknown_labels = [], []

    # Combina datasets
    all_images = pos_images + neg_images + unknown_images
    all_labels = pos_labels + neg_labels + unknown_labels
    
    if not all_images:
        logger.error("❌ Nenhuma imagem foi carregada. Verifique os diretórios.")
        return None, None, None, None

    X = np.array(all_images, dtype="float32")
    y = np.array(all_labels)

    logger.info(f"Dataset combinado: {X.shape[0]} imagens totais.")
    logger.info(f"Distribuição de classes: {np.unique(y, return_counts=True)}")

    # Normalização
    try:
        X = X / 255.0
        logger.info("✅ Normalização concluída (valores entre 0 e 1).")
    except Exception as e:
        logger.error(f"❌ Erro na normalização: {e}")
        return None, None, None, None

    # Expande dimensões para CNN
    try:
        X = np.expand_dims(X, axis=-1)
        logger.info(f"✅ Ajuste de formato concluído: {X.shape}")
    except Exception as e:
        logger.error(f"❌ Erro ao ajustar formato do dataset: {e}")
        return None, None, None, None

    # Divide treino e teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info(f"✅ Divisão treino/teste concluída.")
        logger.info(f"Treino: {X_train.shape[0]} amostras")
        logger.info(f"Teste: {X_test.shape[0]} amostras")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"❌ Erro na divisão treino/teste: {e}")
        return None, None, None, None

# Teste rápido
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_dataset()
    if X_train is not None:
        logger.info("✅ Dataset preparado com sucesso!")
    else:
        logger.error("❌ Falha ao preparar dataset.")