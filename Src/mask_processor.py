import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Config.config import logger, RAW_DIR, PROCESSED_DIR, IMG_HEIGHT, IMG_WIDTH

# Cria diretório para máscaras segmentadas
SEGMENTATION_DIR = os.path.join(PROCESSED_DIR, "segmentation_maps")
os.makedirs(SEGMENTATION_DIR, exist_ok=True)

def extract_red_areas(image_path, output_path=None, sensitivity=40):
    """
    EXTRAI ÁREAS VERMELHAS de imagens médicas com marcações de AVC.
    
    Esta função é CRUCIAL porque converte as marcas vermelhas que os médicos
    fizeram nas imagens em MÁSCARAS BINÁRIAS que a IA pode usar para aprender
    exatamente onde estão os AVCs.
    
    Args:
        image_path (str): Caminho para a imagem com marcações vermelhas
        output_path (str, optional): Onde salvar a máscara binária resultante
        sensitivity (int): Sensibilidade para detecção do vermelho (0-100)
    
    Returns:
        numpy.ndarray: Máscara binária onde 1 = área com AVC, 0 = resto
    """
    try:
        # Carrega a imagem colorida (com as marcas vermelhas)
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Erro ao carregar imagem: {image_path}")
            return None
        
        logger.info(f"Processando marcações vermelhas em: {image_path}")
        
        # Converte para HSV - melhor espaço de cor para detecção de cores
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # DEFINIÇÃO DOS INTERVALOS DE COR VERMELHA EM HSV // IMPORTANTE ALTERAR ESSAS INFORMAÇÕES CASO A DETECÇÃO NAO ESTIVER BOA
        # O vermelho é complicado porque fica no limite do espectro HSV
        # Por isso precisamos de DOIS intervalos
        lower_red1 = np.array([0, 70, 50])         # Vermelho-alaranjado
        upper_red1 = np.array([10, 255, 255])      # Vermelho-vivo
        
        lower_red2 = np.array([170, 70, 50])       # Vermelho-arroxeado  
        upper_red2 = np.array([180, 255, 255])     # Vermelho-intenso
        
        # Cria máscaras para ambos os ranges de vermelho
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combina as máscaras - áreas vermelhas de qualquer tom
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # OPERAÇÕES MORFOLÓGICAS para limpar a máscara
        kernel = np.ones((5, 5), np.uint8)
        
        # Fecha pequenos buracos dentro das áreas vermelhas
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove ruídos e pequenas manchas isoladas
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # ENCONTRAR CONTORNOS das áreas marcadas
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Preenche áreas internas dos contornos
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Ignora áreas muito pequenas (ruído)
                cv2.drawContours(red_mask, [contour], 0, 255, -1)  # -1 = preenche interior
        
        # Redimensiona para o tamanho padrão do projeto
        red_mask = cv2.resize(red_mask, (IMG_WIDTH, IMG_HEIGHT))
        
        # Salva a máscara binária se um caminho foi fornecido
        if output_path:
            cv2.imwrite(output_path, red_mask)
            logger.info(f"Máscara de AVC salva em: {output_path}")
        
        return red_mask
        
    except Exception as e:
        logger.error(f"Erro ao processar {image_path}: {str(e)}")
        return None

def process_all_avc_masks(input_dir=None, output_dir=SEGMENTATION_DIR):
    """
    PROCESSA EM LOTE todas as imagens com marcações de AVC.
    
    Esta função automatiza o processamento de todas as imagens anotadas
    que você tem. Ela busca em vários locais possíveis onde suas imagens
    com marcas vermelhas podem estar.
    
    Args:
        input_dir (str, optional): Diretório com imagens anotadas
        output_dir (str): Onde salvar as máscaras processadas
    
    Returns:
        bool: True se processou com sucesso, False caso contrário
    """
    # Busca automática do diretório de imagens anotadas
    if input_dir is None:
        possible_dirs = [
            os.path.join(RAW_DIR, "masks"),
            os.path.join(RAW_DIR, "annotated"),
            os.path.join(RAW_DIR, "red_marks"),
            os.path.join(RAW_DIR, "avc_marks"),
            os.path.join(RAW_DIR, "positive")  # Talvez as marcas estejam aqui
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                input_dir = dir_path
                logger.info(f"Encontrado diretório de máscaras: {dir_path}")
                break
        else:
            logger.error("Não encontrei diretório com imagens anotadas!")
            logger.info("Procurei em: " + ", ".join(possible_dirs))
            return False
    
    # Verifica se o diretório existe
    if not os.path.exists(input_dir):
        logger.error(f"Diretório não existe: {input_dir}")
        return False
    
    logger.info(f"Iniciando processamento em lote de: {input_dir}")
    
    # Processa todas as imagens do diretório
    processed_count = 0
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_dir, filename)
            
            # Nome do arquivo de saída
            output_filename = f"avc_mask_{os.path.splitext(filename)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Processa a imagem e extrai as áreas vermelhas
            if extract_red_areas(input_path, output_path):
                processed_count += 1
    
    logger.info(f"Processamento concluído: {processed_count} máscaras de AVC extraídas")
    return processed_count > 0

def preview_red_extraction(image_path, save_path=None):
    """
    Gera uma visualização de como a extração de áreas vermelhas está funcionando.
    
    Esta função é ÚTIL PARA AJUSTAR os parâmetros de detecção. Ela mostra
    lado a lado: imagem original, máscara binária e áreas detectadas.
    
    Args:
        image_path (str): Caminho para a imagem de teste
        save_path (str, optional): Onde salvar a visualização
    
    Returns:
        bool: True se bem-sucedido, False caso contrário
    """
    try:
        # Carrega e converte imagem
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extrai máscara vermelha (sem salvar)
        red_mask = extract_red_areas(image_path, None)
        
        if red_mask is not None:
            # Aplica máscara na imagem original para visualização
            masked_image = cv2.bitwise_and(image_rgb, image_rgb, 
                                         mask=red_mask.astype(np.uint8))
            
            # Cria visualização com subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Imagem original
            axes[0].imshow(image_rgb)
            axes[0].set_title('Imagem Original\n(com marcas médicas)')
            axes[0].axis('off')
            
            # Máscara binária
            axes[1].imshow(red_mask, cmap='gray')
            axes[1].set_title('Máscara de AVC Extraída\n(áreas brancas = AVC)')
            axes[1].axis('off')
            
            # Áreas detectadas sobrepostas
            axes[2].imshow(masked_image)
            axes[2].set_title('Áreas de AVC Detectadas\n(visualização)')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Salva ou mostra a visualização
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualização salva em: {save_path}")
                plt.close()
            else:
                plt.show()
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Erro na visualização: {str(e)}")
        return False

def create_segmentation_dataset(image_dir, mask_dir, output_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    CRIA DATASET DE TREINAMENTO para segmentação de AVC.
    
    Esta função prepara os dados para treinar uma IA que não apenas
    classifica mas também SEGMENTA onde está o AVC.
    
    Args:
        image_dir (str): Diretório com imagens originais
        mask_dir (str): Diretório com máscaras de AVC
        output_size (tuple): Tamanho para redimensionamento
    
    Returns:
        tuple: (X_seg, y_seg) - Imagens e máscaras para treinamento
    """
    X_seg = []  # Imagens de treinamento
    y_seg = []  # Máscaras correspondentes (ground truth)
    
    # Garante correspondência entre imagens e máscaras
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    mask_files = sorted([f for f in os.listdir(mask_dir) 
                        if f.startswith('avc_mask_') and f.endswith('.png')])
    
    logger.info(f"Encontradas {len(image_files)} imagens e {len(mask_files)} máscaras")
    
    for img_file in image_files:
        # Encontra máscara correspondente
        corresponding_mask = f"avc_mask_{os.path.splitext(img_file)[0]}.png"
        
        if corresponding_mask in mask_files:
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, corresponding_mask)
            
            # Carrega e redimensiona
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                img = cv2.resize(img, output_size)
                mask = cv2.resize(mask, output_size)
                
                # Normalização
                img = img.astype(np.float32) / 255.0
                mask = (mask > 127).astype(np.float32)  # Binariza (0 ou 1)
                
                X_seg.append(img)
                y_seg.append(mask)
    
    # Converte para arrays numpy
    X_seg = np.array(X_seg)
    y_seg = np.array(y_seg)
    
    # Adiciona dimensão de canal (exigida pelo TensorFlow)
    X_seg = np.expand_dims(X_seg, axis=-1)  # (batch, height, width, 1)
    y_seg = np.expand_dims(y_seg, axis=-1)  # (batch, height, width, 1)
    
    logger.info(f"Dataset de segmentação criado: {X_seg.shape} imagens, {y_seg.shape} máscaras")
    return X_seg, y_seg

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    logger.info("🚀 Iniciando processamento de máscaras de AVC...")
    logger.info("📁 Estrutura de diretórios:")
    logger.info(f"   RAW_DIR: {RAW_DIR}")
    logger.info(f"   SEGMENTATION_DIR: {SEGMENTATION_DIR}")
    
    # 1. Processa todas as máscaras vermelhas disponíveis
    success = process_all_avc_masks()
    
    if success:
        logger.info("✅ Processamento de máscaras concluído com sucesso!")
        
        # 2. Tenta criar dataset de segmentação se tiver imagens correspondentes
        possible_image_dirs = [
            os.path.join(RAW_DIR, "images"),
            os.path.join(RAW_DIR, "positive"),
            os.path.join(PROCESSED_DIR, "positive")
        ]
        
        for image_dir in possible_image_dirs:
            if os.path.exists(image_dir):
                X_seg, y_seg = create_segmentation_dataset(
                    image_dir, 
                    SEGMENTATION_DIR,
                    output_size=(IMG_HEIGHT, IMG_WIDTH)
                )
                
                if len(X_seg) > 0:
                    logger.info(f"🎯 Dataset de segmentação pronto com {len(X_seg)} amostras!")
                    logger.info("💡 Use esses dados para treinar IA que localiza AVCs!")
                    break
        else:
            logger.warning("⚠️  Diretório de imagens não encontrado. Dataset de segmentação não criado.")
            
    else:
        logger.error("❌ Falha no processamento das máscaras. Verifique os diretórios.")
    
    logger.info("📋 Para testar com uma imagem individual, use:")
    logger.info("   preview_red_extraction('caminho/para/sua/imagem.jpg')")