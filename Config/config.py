import os
import random
import numpy as np
import torch
import logging
from datetime import datetime

# Criar pasta de logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Nome do arquivo de log por data
log_filename = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configuração do logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Criar logger global
logger = logging.getLogger("app")

# Seed para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Diretórios do Projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Diretórios de dados
DATA_DIR = os.path.join(PROJECT_DIR, 'Data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Subpastas para as três classes
RAW_POSITIVE_DIR = os.path.join(RAW_DIR, 'positive')
RAW_NEGATIVE_DIR = os.path.join(RAW_DIR, 'negative')
RAW_UNKNOWN_DIR = os.path.join(RAW_DIR, 'unknown')

PROCESSED_POSITIVE_DIR = os.path.join(PROCESSED_DIR, 'positive')
PROCESSED_NEGATIVE_DIR = os.path.join(PROCESSED_DIR, 'negative')
PROCESSED_UNKNOWN_DIR = os.path.join(PROCESSED_DIR, 'unknown')

# Diretórios de resultados
RESULTS_DIR = os.path.join(PROJECT_DIR, 'Results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
MODEL_PATH = os.path.join(MODELS_DIR, 'brain_stroke_model.keras')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Parâmetros do modelo
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Agora são 3 classes: Positivo, Negativo, Desconhecido

# Dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Função auxiliar para criar as pastas se necessário
def create_dirs():
    """Cria todas as pastas necessárias para o projeto"""
    directories = [
        RAW_POSITIVE_DIR, RAW_NEGATIVE_DIR, RAW_UNKNOWN_DIR,
        PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR,
        MODELS_DIR, PLOTS_DIR, LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório garantido: {directory}")

# Cria os diretórios ao importar o config
create_dirs()

