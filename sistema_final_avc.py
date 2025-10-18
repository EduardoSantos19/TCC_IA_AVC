# sistema_final_avc.py
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from keras.models import load_model
from Src.data_loader import safe_imread
from Config.config import IMG_HEIGHT, IMG_WIDTH, logger
import cv2

class SistemaDetecçãoAVC:
    def __init__(self, model_path='Results/models/modelo_conservador.keras', threshold=0.4):
        """
        🏥 SISTEMA FINAL DE DETECÇÃO DE AVC
        """
        self.model = load_model(model_path)
        self.threshold = threshold
        logger.info("✅ Sistema de Detecção de AVC carregado!")
    
    def classificar_imagem(self, image_path):
        """
        🎯 CLASSIFICA uma imagem e retorna probabilidade de AVC
        """
        try:
            # Carregar imagem
            processed_img = safe_imread(image_path)
            if processed_img is None:
                return {"erro": f"Falha ao carregar imagem: {image_path}"}
            
            # Pré-processamento
            processed_img = cv2.resize(processed_img, (IMG_HEIGHT, IMG_WIDTH))
            processed_img = processed_img.astype("float32") / 255.0
            processed_img = np.expand_dims(processed_img, axis=-1)
            processed_img = np.expand_dims(processed_img, axis=0)
            
            # Predição
            probabilidade = self.model.predict(processed_img, verbose=0)[0][0]
            percentual = probabilidade * 100
            
            # Classificação e confiança
            tem_avc = probabilidade > self.threshold
            classificacao = "AVC" if tem_avc else "Saudável"
            confianca = percentual if tem_avc else (100 - percentual)
            
            # Nível de risco
            nivel_risco, recomendacao = self._calcular_risco(percentual)
            
            return {
                "imagem": os.path.basename(image_path),
                "probabilidade_avc": f"{percentual:.1f}%",
                "classificacao": classificacao,
                "confianca": f"{confianca:.1f}%",
                "nivel_risco": nivel_risco,
                "recomendacao": recomendacao,
                "threshold_utilizado": self.threshold
            }
            
        except Exception as e:
            return {"erro": f"Erro na classificação: {str(e)}"}
    
    def _calcular_risco(self, percentual):
        """
        🚨 CALCULA nível de risco baseado na probabilidade
        """
        if percentual < 15:
            return "Risco Mínimo", "Paciente dentro da normalidade. Acompanhamento de rotina."
        elif percentual < 35:
            return "Risco Moderado", "Alterações sutis detectadas. Recomenda-se reavaliação em 3-6 meses."
        elif percentual < 60:
            return "Risco Alto", "Alterações significativas. Encaminhar para neurologista."
        elif percentual < 85:
            return "Risco Muito Alto", "Alta probabilidade de AVC. Solicitar exames complementares urgentemente."
        else:
            return "Emergência", "Evidências claras de AVC. Atendimento médico imediato necessário."
    
    def avaliar_lote(self, lista_imagens):
        """
        📊 AVALIA várias imagens de uma vez
        """
        resultados = []
        for img_path in lista_imagens:
            resultado = self.classificar_imagem(img_path)
            resultados.append(resultado)
        return resultados

def main():
    """
    🎯 EXEMPLO DE USO DO SISTEMA FINAL
    """
    print("🏥 SISTEMA DE DETECÇÃO DE AVC - VERSÃO FINAL")
    print("=" * 60)
    
    # Inicializar sistema
    sistema = SistemaDetecçãoAVC()
    
    # Exemplo de uso
    from Config.config import PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR
    import glob
    
    # Encontrar algumas imagens para demonstração
    imagens_teste = []
    for pasta in [PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR]:
        if os.path.exists(pasta):
            imagens = glob.glob(os.path.join(pasta, "*.png"))[:2]
            imagens_teste.extend(imagens)
    
    print(f"🔍 Analisando {len(imagens_teste)} imagens...")
    print("=" * 60)
    
    for img_path in imagens_teste:
        resultado = sistema.classificar_imagem(img_path)
        
        if "erro" not in resultado:
            print(f"\n📄 IMAGEM: {resultado['imagem']}")
            print(f"📊 Probabilidade de AVC: {resultado['probabilidade_avc']}")
            print(f"🏷️  Classificação: {resultado['classificacao']}")
            print(f"💪 Confiança: {resultado['confianca']}")
            print(f"🚨 Nível de Risco: {resultado['nivel_risco']}")
            print(f"💡 Recomendação: {resultado['recomendacao']}")
            print("-" * 50)
        else:
            print(f"❌ Erro: {resultado['erro']}")

if __name__ == "__main__":
    main()