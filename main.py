# main.py
"""
Ponto de entrada principal do projeto
"""
import argparse
from Src.Models.train import main_training_pipeline
from Src.mask_processor import process_all_avc_masks
from Config.config import logger

def main():
    parser = argparse.ArgumentParser(description="Sistema de Detecção de AVC via IA")
    parser.add_argument('--mode', choices=['train', 'process_masks', 'evaluate'], 
                       default='train', help='Modo de operação')
    parser.add_argument('--model', choices=['small_cnn', 'efficientnet'], 
                       default='small_cnn', help='Tipo de modelo')
    parser.add_argument('--cross_val', action='store_true', 
                       help='Usar validação cruzada')
    
    args = parser.parse_args()
    
    if args.mode == 'process_masks':
        logger.info("🔴 Processando máscaras de AVC...")
        process_all_avc_masks()
        
    elif args.mode == 'train':
        logger.info("🤖 Iniciando treinamento...")
        main_training_pipeline(args.model, args.cross_val)
        
    elif args.mode == 'evaluate':
        logger.info("📊 Modo avaliação - Em desenvolvimento")
        # Implementar avaliação de modelos treinados
        
    logger.info("✅ Operação concluída!")

if __name__ == "__main__":
    main()