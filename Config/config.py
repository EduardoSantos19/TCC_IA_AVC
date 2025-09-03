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
    level=logging.DEBUG,  # DEBUG mostra tudo (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()  # também mostra no terminal
    ]
)

# Criar logger global
logger = logging.getLogger("app")


#Seed para reprodutibilidade e tirar a redundancia nos testes e sempre ter os mesmos pesos iniciais, divisão de dados e etc

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Diretórios do Projeto // Definição dos caminhos para os dados, resultados, modelos e plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR,'..'))

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

#Parâmetros do modelo // caso precise mudar algo voltado a parametros de epocas, tamanho de lote, taxa de aprendizado.

IMG_HEIGHT = 128 # ALTURA DAS IMAGENS
IMG_WIDTH = 128  # LARGURA
CHANNELS = 1     # ESCALA DE FILTROS (CINZA)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

#Parametros diversos

NUM_CLASSES = 2    # Positivo (AVC) ou Negativo (saudável)
DEVICE = torch.device('cuda' if torch.cuda.is_available()else 'cpu') # Usado pra detecão de placa de video em sistema / caso não usar o proprio processador

#Função auxiliar para criar as pastas se necessário

def create_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

# VALIDAÇÃO DE PASTAS CRIADAS

folders = [
    "data/raw",
    "data/processed",
    "results/models",
    "results/plots"
]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[CRIADA] Pasta criada: {folder}")
    else:
        print(f"[EXISTE] pasta já existente em sistema: {folder}")


