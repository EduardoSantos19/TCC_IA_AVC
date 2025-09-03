import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# sys.path para encontrar a pasta config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Config.config import logger, PROCESSED_DIR

# ----------------------------
# Carregar imagens
# ----------------------------
def load_images_from_folder(folder, label, image_size=(128, 128)):
    """Carrega imagens de uma pasta, converte para grayscale e redimensiona."""
    images = []
    labels = []
    ignored_count = 0

    logger.info(f"Iniciando carregamento das imagens da pasta: {folder}")

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm")

    try:
        for filename in os.listdir(folder):
            if not filename.lower().endswith(valid_extensions):
                logger.warning(f"Ignorado arquivo não suportado: {filename}")
                continue

            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                ignored_count += 1
                logger.warning(f"Imagem inválida ignorada: {file_path}")
                continue

            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)

        logger.info(f"✅ {len(images)} imagens carregadas de {folder}")
        if ignored_count > 0:
            logger.info(f"⚠️ {ignored_count} imagens foram ignoradas em {folder}")

    except Exception as e:
        logger.error(f"❌ Erro ao carregar imagens da pasta {folder}: {e}")

    return images, labels

# ----------------------------
# Preparação dataset
# ----------------------------
def prepare_dataset(test_size=0.2, image_size=(128, 128)):
    """Prepara o dataset completo com imagens positivas e negativas (a partir de processed)."""
    logger.info("Iniciando preparação do dataset...")

    pos_folder = os.path.join(PROCESSED_DIR, "positivas")
    neg_folder = os.path.join(PROCESSED_DIR, "negativas")

    # Carrega imagens
    pos_images, pos_labels = load_images_from_folder(pos_folder, 1, image_size)
    neg_images, neg_labels = load_images_from_folder(neg_folder, 0, image_size)

    # Combina datasets
    X = np.array(pos_images + neg_images, dtype="float32")
    y = np.array(pos_labels + neg_labels)
    logger.info(f"Dataset combinado: {X.shape[0]} imagens totais.")

    # Normalização
    try:
        X = X / 255.0
        logger.info("✅ Normalização concluída (valores entre 0 e 1).")
    except Exception as e:
        logger.error(f"❌ Erro na normalização: {e}")

    # Expande dimensões para CNN
    try:
        X = np.expand_dims(X, axis=-1)
        logger.info(f"✅ Ajuste de formato concluído: {X.shape}")
    except Exception as e:
        logger.error(f"❌ Erro ao ajustar formato do dataset: {e}")

    # Divide treino e teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info(f"✅ Divisão treino/teste concluída.")
        logger.info(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
    except Exception as e:
        logger.error(f"❌ Erro na divisão treino/teste: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    return X_train, X_test, y_train, y_test

# ----------------------------
# Teste rápido
# ----------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_dataset()
    logger.info("Dataset preparado com sucesso!")