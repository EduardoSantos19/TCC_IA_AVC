# test_probabilidade.py
import os
import sys
import numpy as np
import glob

# Adiciona o caminho raiz
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

try:
    from keras.models import load_model
    from Src.data_loader import safe_imread, fix_encoding_path
    from Config.config import IMG_HEIGHT, IMG_WIDTH, logger, PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR
    
    print("✅ Módulos importados com sucesso!")
    
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    sys.exit(1)

class StrokeRiskAssessor:
    def __init__(self):
        self.risk_thresholds = {
            'risco_minimo': (0, 15),
            'risco_moderado': (16, 35), 
            'risco_alto': (36, 60),
            'risco_muito_alto': (61, 85),
            'emergencia': (86, 100)
        }
        
        self.recommendations = {
            'risco_minimo': "Paciente dentro da normalidade. Acompanhamento de rotina.",
            'risco_moderado': "Alterações sutis detectadas. Recomenda-se reavaliação em 3-6 meses.",
            'risco_alto': "Alterações significativas. Encaminhar para neurologista.",
            'risco_muito_alto': "Alta probabilidade de AVC. Solicitar exames complementares urgentemente.",
            'emergencia': "Evidências claras de AVC. Atendimento médico imediato necessário."
        }
    
    def assess_risk(self, probability):
        """Avalia o risco baseado na probabilidade"""
        percentage = probability * 100
        
        for risk_level, (min_val, max_val) in self.risk_thresholds.items():
            if min_val <= percentage <= max_val:
                return {
                    'probabilidade_percentual': percentage,
                    'nivel_risco': risk_level,
                    'recomendacao': self.recommendations[risk_level],
                    'faixa': f"{min_val}%-{max_val}%"
                }
        
        return {
            'probabilidade_percentual': percentage,
            'nivel_risco': 'indeterminado',
            'recomendacao': 'Necessária avaliação médica adicional',
            'faixa': 'Fora das faixas padrão'
        }

def predict_stroke_probability(model, image):
    """Prediz probabilidade de AVC para uma imagem"""
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    probability = model.predict(image, verbose=0)[0][0]
    return probability

def test_single_image_with_probability(model, image_path):
    """Testa uma imagem e retorna probabilidade de AVC"""
    try:
        # Carrega e pré-processa imagem
        processed_img = safe_imread(image_path)
        if processed_img is None:
            print(f'❌ Erro no carregamento da imagem: {image_path}')
            return None
        
        # Redimensiona se necessário
        if processed_img.shape != (IMG_HEIGHT, IMG_WIDTH):
            import cv2
            processed_img = cv2.resize(processed_img, (IMG_HEIGHT, IMG_WIDTH))
        
        # Normaliza
        processed_img = processed_img.astype("float32") / 255.0
        processed_img = np.expand_dims(processed_img, axis=-1)
        
        # Prediz probabilidade
        probability = predict_stroke_probability(model, processed_img)
        
        # Avalia risco
        assessor = StrokeRiskAssessor()
        resultado = assessor.assess_risk(probability)
        
        print(f"\n🎯 RESULTADO DA ANÁLISE:")
        print(f"📁 Imagem: {os.path.basename(image_path)}")
        print(f"📊 Probabilidade de AVC: {resultado['probabilidade_percentual']:.2f}%")
        print(f"⚠️  Nível de Risco: {resultado['nivel_risco'].upper()}")
        print(f"📋 Faixa: {resultado['faixa']}")
        print(f"💡 Recomendação: {resultado['recomendacao']}")
        
        return resultado
        
    except Exception as e:
        print(f'❌ Erro ao testar imagem {image_path}: {e}')
        return None

def find_test_images():
    """Encontra imagens reais para teste"""
    test_images = []
    
    # Procura por imagens existentes
    folders = [
        (PROCESSED_POSITIVE_DIR, "AVC"),
        (PROCESSED_NEGATIVE_DIR, "Saudável"), 
        (PROCESSED_UNKNOWN_DIR, "Desconhecido")
    ]
    
    for folder, label in folders:
        folder = fix_encoding_path(folder)
        if os.path.exists(folder):
            # Procura por qualquer imagem
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images = glob.glob(os.path.join(folder, ext))
                if images:
                    test_images.extend(images[:2])  # Pega até 2 imagens de cada
                    break
    
    return test_images

def main():
    model_path = "Results/models/brain_stroke_model.keras"
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        print("💡 Execute primeiro: python train_model.py")
        return
    
    try:
        # Carrega modelo uma vez
        model = load_model(model_path)
        print(f'✅ Modelo carregado: {model_path}')
        
        # Encontra imagens para teste
        test_images = find_test_images()
        
        if not test_images:
            print("❌ Nenhuma imagem encontrada para teste")
            print("💡 Verifique se as pastas processed têm imagens")
            return
        
        print("🧪 TESTE DE PREDIÇÃO COM PROBABILIDADE")
        print("=" * 60)
        print(f"📊 Encontradas {len(test_images)} imagens para teste")
        
        resultados = []
        for img_path in test_images:
            print(f"\n📄 Testando: {os.path.basename(img_path)}")
            resultado = test_single_image_with_probability(model, img_path)
            if resultado:
                resultados.append(resultado)
            print("-" * 50)
        
        # Estatísticas
        if resultados:
            print("\n📈 ESTATÍSTICAS DOS TESTES:")
            probabilidades = [r['probabilidade_percentual'] for r in resultados]
            print(f"   📊 Média: {np.mean(probabilidades):.2f}%")
            print(f"   📊 Mínimo: {np.min(probabilidades):.2f}%")
            print(f"   📊 Máximo: {np.max(probabilidades):.2f}%")
            
            # Distribuição de risco
            riscos = {}
            for r in resultados:
                nivel = r['nivel_risco']
                riscos[nivel] = riscos.get(nivel, 0) + 1
            
            print(f"   📋 Distribuição de Risco:")
            for nivel, count in riscos.items():
                print(f"      {nivel}: {count} imagens")
    
    except Exception as e:
        print(f'❌ Erro ao carregar modelo: {e}')

# Teste rápido com imagens específicas
def test_specific_images():
    """Testa com imagens específicas se existirem"""
    specific_images = [
        "Data/processed/positive/10002.png",
        "Data/processed/negative/10000.png", 
        "Data/processed/unknown/10.jpg"
    ]
    
    model_path = "Results/models/brain_stroke_model.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    model = load_model(model_path)
    print(f'✅ Modelo carregado: {model_path}')
    
    print("🧪 TESTE COM IMAGENS ESPECÍFICAS")
    print("=" * 50)
    
    for img_path in specific_images:
        if os.path.exists(img_path):
            print(f"\n📄 Testando: {os.path.basename(img_path)}")
            test_single_image_with_probability(model, img_path)
            print("-" * 50)
        else:
            print(f"❌ Imagem não encontrada: {img_path}")

if __name__ == "__main__":
    # Tenta primeiro com imagens específicas, depois busca automaticamente
    test_specific_images()
    
    # Se não encontrou imagens específicas, busca automaticamente
    print("\n" + "="*60)
    print("🔍 BUSCANDO IMAGENS AUTOMATICAMENTE...")
    main()