import os
import cv2
import numpy as np
import sys

# 🔥 CORREÇÃO: Adiciona o caminho do projeto ao sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Agora importa os módulos
try:
    from Config.config import (
        logger, PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, 
        PROCESSED_UNKNOWN_DIR, IMG_HEIGHT, IMG_WIDTH,
        RAW_POSITIVE_DIR, RAW_NEGATIVE_DIR, RAW_UNKNOWN_DIR
    )
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"📁 sys.path: {sys.path}")
    sys.exit(1)

from sklearn.model_selection import train_test_split

# 🔥 RESTANTE DO CÓDIGO (mantenha todas as funções que mostrei anteriormente)
def fix_encoding_path(file_path):
    """
    🔥 CORREÇÃO: Corrige problemas de encoding em caminhos de arquivo
    Substitui caracteres corrompidos como '├ü' por 'Á'
    """
    try:
        # Corrige encoding comum no Windows/OneDrive
        corrected_path = file_path.replace('├ü', 'Á')
        corrected_path = corrected_path.replace('├í', 'á')
        corrected_path = corrected_path.replace('├©', 'è')
        corrected_path = corrected_path.replace('├®', 'é')
        corrected_path = corrected_path.replace('├¡', 'í')
        corrected_path = corrected_path.replace('├│', 'ó')
        corrected_path = corrected_path.replace('├║', 'ú')
        corrected_path = corrected_path.replace('├▒', 'ñ')
        corrected_path = corrected_path.replace('├º', 'ç')
        
        # Tenta o caminho corrigido primeiro
        if os.path.exists(corrected_path):
            return corrected_path
        
        # Se não existir, tenta o caminho original
        if os.path.exists(file_path):
            return file_path
            
        # Última tentativa: usa caminho absoluto
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            return abs_path
            
    except Exception as e:
        logger.warning(f"Erro ao corrigir encoding: {e}")
    
    return file_path

def safe_imread(file_path):
    """
    🔥 NOVO: Leitura segura de imagens com tratamento de encoding
    """
    # Corrige encoding do caminho
    file_path = fix_encoding_path(file_path)
    
    # Tentativa 1: Leitura normal com OpenCV
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    
    # Tentativa 2: Leitura com caminho absoluto
    try:
        abs_path = os.path.abspath(file_path)
        if abs_path != file_path:
            img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    except:
        pass
    
    # Tentativa 3: Leitura via bytes (mais robusta)
    try:
        with open(file_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    except Exception as e:
        logger.warning(f"Falha na leitura por bytes: {file_path} - {e}")
    
    # Tentativa 4: Usa PIL como fallback
    try:
        from PIL import Image
        img_pil = Image.open(file_path).convert('L')  # Converte para grayscale
        img = np.array(img_pil)
        return img
    except ImportError:
        logger.warning("PIL não disponível para fallback")
    except Exception as e:
        logger.warning(f"Falha na leitura com PIL: {file_path} - {e}")
    
    return None

def load_images_from_folder(folder, label, image_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Carrega imagens com tratamento robusto de erros e encoding"""
    images = []
    labels = []
    ignored_count = 0
    corrupted_count = 0

    logger.info(f"Carregando imagens da pasta: {folder}")

    # Corrige encoding da pasta também
    folder = fix_encoding_path(folder)

    if not os.path.exists(folder):
        logger.warning(f"Pasta não encontrada: {folder}")
        return images, labels

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(folder):
        if not filename.lower().endswith(valid_extensions):
            ignored_count += 1
            continue

        file_path = os.path.join(folder, filename)
        
        try:
            # Verifica se arquivo não está vazio
            if os.path.getsize(file_path) == 0:
                logger.warning(f"Arquivo vazio: {filename}")
                corrupted_count += 1
                continue
                
            # 🔥 USA LEITURA SEGURA
            img = safe_imread(file_path)
            
            if img is None:
                corrupted_count += 1
                logger.warning(f"Não conseguiu ler: {filename}")
                continue
                
            # Redimensiona
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            logger.warning(f"Erro ao processar {filename}: {e}")
            corrupted_count += 1
            continue

    logger.info(f"✅ {len(images)} imagens carregadas de {folder}")
    if ignored_count > 0:
        logger.info(f"⚠ {ignored_count} arquivos ignorados (extensão)")
    if corrupted_count > 0:
        logger.info(f"🚨 {corrupted_count} arquivos corrompidos/inválidos")

    return images, labels

def check_if_processed_images_exist():
    """Verifica se existem imagens nas pastas processed com encoding corrigido"""
    processed_dirs = [
        fix_encoding_path(PROCESSED_POSITIVE_DIR),
        fix_encoding_path(PROCESSED_NEGATIVE_DIR), 
        fix_encoding_path(PROCESSED_UNKNOWN_DIR)
    ]
    
    for dir_path in processed_dirs:
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if len(images) > 0:
                return True
    return False

def preprocess_if_needed():
    """Executa o pré-processamento se as pastas processed estiverem vazias"""
    if not check_if_processed_images_exist():
        logger.info("Nenhuma imagem processada encontrada. Executando pré-processamento...")
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(current_dir, '..')
            if src_path not in sys.path:
                sys.path.append(src_path)
            
            from Preprocessing.preprocess_image import preprocess_batch
            
            classes_to_process = {
                'positive': (RAW_POSITIVE_DIR, PROCESSED_POSITIVE_DIR),
                'negative': (RAW_NEGATIVE_DIR, PROCESSED_NEGATIVE_DIR),
                'unknown': (RAW_UNKNOWN_DIR, PROCESSED_UNKNOWN_DIR)
            }
            
            total_processed = 0
            for class_name, (input_dir, output_dir) in classes_to_process.items():
                # 🔥 CORRIGE ENCODING DOS DIRETÓRIOS TAMBÉM
                input_dir = fix_encoding_path(input_dir)
                output_dir = fix_encoding_path(output_dir)
                
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
            return False
        except Exception as e:
            logger.error(f"Erro durante pré-processamento: {e}")
            return False
    return True

def prepare_dataset_for_regression(test_size=0.2, image_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Prepara dataset para REGRESSÃO de probabilidade de AVC
    """
    logger.info("🔥 Preparando dataset para REGRESSÃO de probabilidade de AVC...")

    # Verifica e executa pré-processamento se necessário
    if not preprocess_if_needed():
        logger.error("❌ Falha no pré-processamento. Verifique as pastas raw/")
        return None, None, None, None

    # Carrega apenas positive e negative para treinamento
    pos_images, pos_labels = load_images_from_folder(PROCESSED_POSITIVE_DIR, 1.0, image_size)
    neg_images, neg_labels = load_images_from_folder(PROCESSED_NEGATIVE_DIR, 0.0, image_size)
    
    logger.info("⚠️  Imagens 'unknown' serão usadas apenas para validação")

    # Combina datasets
    all_images = pos_images + neg_images
    all_labels = pos_labels + neg_labels
    
    if not all_images:
        logger.error("❌ Nenhuma imagem foi carregada. Verifique os diretórios.")
        return None, None, None, None

    X = np.array(all_images, dtype="float32")
    y = np.array(all_labels, dtype="float32")

    logger.info(f"📊 Dataset regressão: {X.shape[0]} imagens totais")
    logger.info(f"📈 Distribuição - AVC (1.0): {len(pos_images)} imagens, Saudável (0.0): {len(neg_images)} imagens")

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
        y_binary = (y > 0.5).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        logger.info(f"✅ Divisão treino/teste para REGRESSÃO concluída.")
        logger.info(f"📊 Treino: {X_train.shape[0]} amostras")
        logger.info(f"📊 Teste: {X_test.shape[0]} amostras")
        logger.info(f"📈 Treino - AVC: {np.sum(y_train == 1.0)}, Saudável: {np.sum(y_train == 0.0)}")
        logger.info(f"📈 Teste - AVC: {np.sum(y_test == 1.0)}, Saudável: {np.sum(y_test == 0.0)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"❌ Erro na divisão treino/teste: {e}")
        return None, None, None, None

def test_image_loading():
    """Testa se as imagens estão sendo carregadas corretamente"""
    print("🧪 TESTANDO CARREGAMENTO DE IMAGENS")
    
    test_dirs = [
        PROCESSED_POSITIVE_DIR,
        PROCESSED_NEGATIVE_DIR,
        PROCESSED_UNKNOWN_DIR
    ]
    
    for dir_path in test_dirs:
        print(f"\n📁 Testando: {os.path.basename(dir_path)}")
        dir_path = fix_encoding_path(dir_path)
        
        if not os.path.exists(dir_path):
            print("❌ Pasta não existe")
            continue
            
        images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]
        
        for img_file in images:
            file_path = os.path.join(dir_path, img_file)
            print(f"  📄 {img_file}: ", end="")
            
            img = safe_imread(file_path)
            if img is not None:
                print(f"✅ Carregada - Shape: {img.shape}")
            else:
                print("❌ Falha")

if __name__ == "__main__":
    test_image_loading()