"""
Sistema de avaliação de risco baseado em probabilidade
"""
import numpy as np
from Config.config import logger

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
        """
        Avalia o risco baseado na probabilidade
        """
        percentage = probability * 100  # Converte para porcentagem
        
        for risk_level, (min_val, max_val) in self.risk_thresholds.items():
            if min_val <= percentage <= max_val:
                return {
                    'probabilidade': percentage,
                    'nivel_risco': risk_level,
                    'recomendacao': self.recommendations[risk_level],
                    'faixa': f"{min_val}%-{max_val}%"
                }
        
        return {
            'probabilidade': percentage,
            'nivel_risco': 'indeterminado',
            'recomendacao': 'Necessária avaliação médica adicional',
            'faixa': 'Fora das faixas padrão'
        }

def predict_stroke_probability(model, image):
    """
    Prediz probabilidade de AVC para uma imagem
    """
    # Preprocessamento da imagem
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Predição
    probability = model.predict(image, verbose=0)[0][0]
    
    return probability

# Função para testar uma imagem
def analyze_stroke_risk(model, image_path, preprocess_function):
    """
    Analisa risco de AVC para uma imagem específica
    """
    # Pré-processa a imagem
    processed_img = preprocess_function(image_path)
    if processed_img is None:
        return None
    
    # Prediz probabilidade
    probability = predict_stroke_probability(model, processed_img)
    
    # Avalia risco
    assessor = StrokeRiskAssessor()
    risk_assessment = assessor.assess_risk(probability)
    
    return risk_assessment