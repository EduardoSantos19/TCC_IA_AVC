# test_batch.py
import os
from test_model import test_single_image

def test_model_on_dataset(model_path, dataset_dir, true_class):
    """Testa o modelo em várias imagens de uma classe"""
    
    correct = 0
    total = 0
    
    print(f'🧪 TESTANDO CLASSE: {true_class}')
    
    for img_file in os.listdir(dataset_dir)[:10]:  # Testa 10 imagens
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_dir, img_file)
            
            try:
                pred_class, confidence = test_single_image(model_path, img_path, true_class)
                if pred_class == true_class:
                    correct += 1
                total += 1
                print('---')
                
            except Exception as e:
                print(f'❌ Erro em {img_file}: {e}')
    
    accuracy = correct / total if total > 0 else 0
    print(f'\n📊 RESULTADO FINAL: {correct}/{total} acertos ({accuracy:.2%})')
    return accuracy

# Teste todas as classes
if __name__ == "__main__":
    from Config.config import PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR
    
    model_path = 'Results/models/modelo_v3_5020imagens.h5'
    
    print('🚀 TESTE COMPLETO DO MODELO\n')
    
    # Testa cada classe
    test_model_on_dataset(model_path, PROCESSED_POSITIVE_DIR, 0)
    test_model_on_dataset(model_path, PROCESSED_NEGATIVE_DIR, 1) 
    test_model_on_dataset(model_path, PROCESSED_UNKNOWN_DIR, 2)