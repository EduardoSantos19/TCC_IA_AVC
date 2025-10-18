# testar_modelo_conservador.py
import os
import sys
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from Src.data_loader import prepare_dataset_for_regression
from keras.models import load_model
from Config.config import logger

def obter_imagens_aleatorias(pasta, quantidade=5):
    """
    🎲 Obtém imagens aleatórias de uma pasta
    """
    if not os.path.exists(pasta):
        return []
    
    # Lista todos os arquivos de imagem
    extensoes = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    imagens = []
    
    for extensao in extensoes:
        imagens.extend([f for f in os.listdir(pasta) if f.lower().endswith(extensao[1:])])
    
    # Seleciona aleatoriamente
    if len(imagens) > quantidade:
        return random.sample(imagens, quantidade)
    else:
        return imagens

def testar_modelo_conservador():
    """
    🧪 TESTE COMPLETO DO MODELO CONSERVADOR
    """
    logger.info("🧪 Testando modelo conservador...")
    
    # 1. Carregar dados
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    y_true_class = (y_test > 0.5).astype(int)
    
    # 2. Carregar modelo conservador
    model_path = 'Results/models/modelo_conservador.keras'
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo não encontrado: {model_path}")
        return
    
    model = load_model(model_path)
    logger.info("✅ Modelo conservador carregado!")
    
    # 3. Fazer predições
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    # 4. Testar diferentes thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("🎯 COMPARAÇÃO DE THRESHOLDS:")
    print("=" * 70)
    
    melhor_resultado = None
    melhor_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_class = (y_pred_proba > threshold).astype(int)
        
        # Métricas
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class)
        recall = recall_score(y_true_class, y_pred_class)
        f1 = f1_score(y_true_class, y_pred_class)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true_class, y_pred_class)
        tn, fp, fn, tp = cm.ravel()
        
        # Custo clínico (FN mais caro)
        custo_clinico = fn * 3 + fp * 1  # FN 3x mais custoso
        
        print(f"\n📊 Threshold: {threshold:.1f}")
        print(f"   ✅ Acurácia:  {accuracy:.4f}")
        print(f"   🎯 Precisão: {precision:.4f}")
        print(f"   🎯 Recall:   {recall:.4f}")
        print(f"   🎯 F1-Score: {f1:.4f}")
        print(f"   📈 TN: {tn:3d} | FP: {fp:3d} | FN: {fn:3d} | TP: {tp:3d}")
        print(f"   💰 Custo Clínico: {custo_clinico}")
        
        # Avaliar se é o melhor
        if melhor_resultado is None or custo_clinico < melhor_resultado['custo_clinico']:
            melhor_resultado = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fp': fp,
                'fn': fn,
                'custo_clinico': custo_clinico
            }
            melhor_threshold = threshold
    
    # 5. Resultado final
    print("\n" + "=" * 70)
    print("🎊 MELHOR CONFIGURAÇÃO ENCONTRADA:")
    print(f"   🎯 Threshold Ideal: {melhor_threshold:.1f}")
    print(f"   ✅ Acurácia:  {melhor_resultado['accuracy']:.4f}")
    print(f"   🎯 Precisão: {melhor_resultado['precision']:.4f}") 
    print(f"   🎯 Recall:   {melhor_resultado['recall']:.4f}")
    print(f"   🎯 F1-Score: {melhor_resultado['f1']:.4f}")
    print(f"   🚨 Falsos Positivos: {melhor_resultado['fp']}")
    print(f"   🚨 Falsos Negativos: {melhor_resultado['fn']}")
    print(f"   💰 Custo Clínico: {melhor_resultado['custo_clinico']}")
    
    # 6. Comparar com modelo anterior
    print("\n" + "=" * 70)
    print("📊 COMPARAÇÃO COM MODELO ANTERIOR:")
    print("   Modelo 'Otimizado': Acurácia 51.5% | FP: 419 | FN: 6")
    print(f"   Modelo Conservador: Acurácia {melhor_resultado['accuracy']:.1%} | FP: {melhor_resultado['fp']} | FN: {melhor_resultado['fn']}")
    
    return melhor_threshold, melhor_resultado

def testar_imagens_aleatorias(threshold=0.5, imagens_por_pasta=3):
    """
    🎲 TESTE COM IMAGENS ALEATÓRIAS DE TODAS AS PASTAS
    """
    logger.info("🎲 Testando com imagens aleatórias...")
    
    from keras.models import load_model
    
    model_path = 'Results/models/modelo_conservador.keras'
    model = load_model(model_path)
    
    # Configurar pastas
    from Config.config import PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR
    
    pastas = {
        "AVC (Positive)": PROCESSED_POSITIVE_DIR,
        "Saudável (Negative)": PROCESSED_NEGATIVE_DIR,
        "Desconhecido (Unknown)": PROCESSED_UNKNOWN_DIR
    }
    
    print(f"\n🎲 TESTANDO IMAGENS ALEATÓRIAS ({imagens_por_pasta} por pasta):")
    print("=" * 60)
    
    resultados_por_classe = {
        "AVC (Positive)": {"total": 0, "corretos": 0},
        "Saudável (Negative)": {"total": 0, "corretos": 0},
        "Desconhecido (Unknown)": {"total": 0, "corretos": 0}
    }
    
    for classe, pasta in pastas.items():
        print(f"\n📁 CLASSE: {classe}")
        print("-" * 40)
        
        # Obter imagens aleatórias
        imagens_aleatorias = obter_imagens_aleatorias(pasta, imagens_por_pasta)
        
        if not imagens_aleatorias:
            print(f"   ⚠️  Nenhuma imagem encontrada em {pasta}")
            continue
        
        for imagem_nome in imagens_aleatorias:
            caminho_imagem = os.path.join(pasta, imagem_nome)
            
            if os.path.exists(caminho_imagem):
                # Carregar e processar imagem
                from Src.data_loader import safe_imread
                import cv2
                from Config.config import IMG_HEIGHT, IMG_WIDTH
                
                processed_img = safe_imread(caminho_imagem)
                if processed_img is not None:
                    # Pré-processamento
                    processed_img = cv2.resize(processed_img, (IMG_HEIGHT, IMG_WIDTH))
                    processed_img = processed_img.astype("float32") / 255.0
                    processed_img = np.expand_dims(processed_img, axis=-1)
                    processed_img = np.expand_dims(processed_img, axis=0)
                    
                    # Predição
                    probabilidade = model.predict(processed_img, verbose=0)[0][0]
                    percentual = probabilidade * 100
                    
                    # Classificação
                    classificacao = "AVC" if probabilidade > threshold else "Saudável"
                    confianca = percentual if classificacao == "AVC" else 100 - percentual
                    
                    # Determinar se a classificação está correta
                    correto = False
                    if classe == "AVC (Positive)":
                        correto = (classificacao == "AVC")
                    elif classe == "Saudável (Negative)":
                        correto = (classificacao == "Saudável")
                    else:  # Unknown - não temos label verdadeiro
                        correto = None
                    
                    # Atualizar estatísticas
                    resultados_por_classe[classe]["total"] += 1
                    if correto is not None:
                        if correto:
                            resultados_por_classe[classe]["corretos"] += 1
                    
                    # Exibir resultado
                    status = "✅ CORRETO" if correto else "❌ ERRADO" if correto is not None else "🔍 ANALISADO"
                    print(f"   {status} {imagem_nome}:")
                    print(f"      📊 Probabilidade: {percentual:.1f}%")
                    print(f"      🏷️  Classificação: {classificacao}")
                    print(f"      💪 Confiança: {confianca:.1f}%")
                    
                    # Adicionar análise de risco para casos interessantes
                    if percentual > 70 or percentual < 30:
                        risco = "ALTO RISCO" if percentual > 70 else "BAIXO RISCO"
                        print(f"      🚨 {risco}")
                    
                    print("      " + "-" * 25)
                else:
                    print(f"   ❌ Falha ao carregar: {imagem_nome}")
            else:
                print(f"   ❌ Imagem não existe: {imagem_nome}")
    
    # 7. Estatísticas finais das imagens aleatórias
    print("\n" + "=" * 60)
    print("📈 ESTATÍSTICAS DAS IMAGENS ALEATÓRIAS:")
    print("=" * 60)
    
    total_imagens = 0
    total_corretos = 0
    
    for classe, stats in resultados_por_classe.items():
        if stats["total"] > 0:
            if classe != "Desconhecido (Unknown)":  # Não temos label verdadeiro para unknown
                acuracia_classe = (stats["corretos"] / stats["total"]) * 100
                print(f"   {classe}:")
                print(f"      📊 {stats['corretos']}/{stats['total']} corretos ({acuracia_classe:.1f}%)")
                total_imagens += stats["total"]
                total_corretos += stats["corretos"]
            else:
                print(f"   {classe}:")
                print(f"      🔍 {stats['total']} imagens analisadas (sem label verdadeiro)")
    
    if total_imagens > 0:
        acuracia_geral = (total_corretos / total_imagens) * 100
        print(f"\n   🎯 ACURÁCIA GERAL: {total_corretos}/{total_imagens} ({acuracia_geral:.1f}%)")

def analisar_casos_limite(threshold=0.5, quantidade=5):
    """
    🔍 ANALISA CASOS LIMITE (probabilidades próximas ao threshold)
    """
    logger.info("🔍 Analisando casos limite...")
    
    from keras.models import load_model
    from Src.data_loader import prepare_dataset_for_regression
    import numpy as np
    
    model_path = 'Results/models/modelo_conservador.keras'
    model = load_model(model_path)
    
    # Carregar dados
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    y_true_class = (y_test > 0.5).astype(int)
    
    # Fazer predições
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    # Encontrar casos próximos ao threshold
    margem = 0.15  # ±15% do threshold
    indices_limite = np.where(
        (y_pred_proba > threshold - margem) & 
        (y_pred_proba < threshold + margem)
    )[0]
    
    if len(indices_limite) > 0:
        # Selecionar aleatoriamente alguns casos limite
        if len(indices_limite) > quantidade:
            indices_selecionados = np.random.choice(indices_limite, quantidade, replace=False)
        else:
            indices_selecionados = indices_limite
        
        print(f"\n🔍 CASOS LIMITE (próximos ao threshold {threshold}):")
        print("=" * 60)
        
        for i, idx in enumerate(indices_selecionados):
            prob = y_pred_proba[idx]
            real = "AVC" if y_true_class[idx] == 1 else "Saudável"
            predito = "AVC" if prob > threshold else "Saudável"
            correto = (real == predito)
            
            status = "✅" if correto else "❌"
            print(f"   {status} Caso {i+1}:")
            print(f"      📊 Probabilidade: {prob*100:.1f}%")
            print(f"      🏷️  Real: {real} | Predito: {predito}")
            print(f"      📏 Distância do threshold: {abs(prob-threshold)*100:.1f}%")
            print("      " + "-" * 25)

if __name__ == "__main__":
    # 1. Teste completo do modelo
    threshold_ideal, resultados = testar_modelo_conservador()
    
    # 2. Teste com imagens aleatórias
    print("\n" + "=" * 70)
    testar_imagens_aleatorias(threshold_ideal, imagens_por_pasta=5)
    
    # 3. Análise de casos limite
    print("\n" + "=" * 70)
    analisar_casos_limite(threshold_ideal, quantidade=3)
    
    # 4. Salvar configuração ideal
    print(f"\n💡 CONFIGURAÇÃO RECOMENDADA:")
    print(f"   Use threshold: {threshold_ideal:.1f}")
    print(f"   Modelo: modelo_conservador.keras")
    print(f"   Acurácia esperada: {resultados['accuracy']:.1%}")
    print(f"   Falsos positivos: {resultados['fp']}")
    print(f"   Falsos negativos: {resultados['fn']}")