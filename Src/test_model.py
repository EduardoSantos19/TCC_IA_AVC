# test_model.py
import sys
sys.path.append('.')
import cv2
import numpy as np
from keras.models import load_model
from Src.Preprocessing.preprocess_image import preprocess_single_image
from Config.config import IMG_HEIGHT, IMG_WIDTH

def test_single_image(
        model_path='Results/models/modelo_v2_3000imagens.h5',
        image_path='Data/processed/positive/10002.jpg',   # Altere para o nome desejado
        true_class=0  # 0=Positive, 1=Negative, 2=Unknown
    ):
    
    # Carrega modelo
    model = load_model(model_path)
    print(f'✅ Modelo carregado: {model_path}')
    
    # Pré-processa imagem
    processed_img = preprocess_single_image(image_path)
    if processed_img is None:
        print('❌ Erro no pré-processamento')
        return
    
    # Prepara para predição
    input_img = processed_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    # Faz predição
    prediction = model.predict(input_img, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Mapeia classes
    class_names = {0: 'AVC (Positive)', 1: 'Saudável (Negative)', 2: 'Desconhecido (Unknown)'}
    
    print(f'\n🎯 RESULTADO DA PREDIÇÃO:')
    print(f'📁 Imagem: {image_path}')
    print(f'🔍 Classe prevista: {class_names[predicted_class]}')
    print(f'📊 Confiança: {confidence:.2%}')
    
    if true_class is not None:
        print(f'✅ Classe real: {class_names[true_class]}')
        print(f'🎯 Acerto: {predicted_class == true_class}')
    
    # Mostra probabilidades
    print(f'\n📈 Probabilidades:')
    for i, prob in enumerate(prediction[0]):
        print(f'   {class_names[i]}: {prob:.2%}')
    
    return predicted_class, confidence

# Teste rápido
if __name__ == "__main__":
    test_single_image(
        'Results/models/modelo_v2_3000imagens.h5',
        'Data/processed/positive/01.jpg',  # Altere o caminho
        true_class=0  # 0=Positive, 1=Negative, 2=Unknown
    )