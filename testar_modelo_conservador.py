# testar_modelo_conservador.py
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from Src.data_loader import prepare_dataset_for_regression
from keras.models import load_model
from Config.config import logger

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

def testar_imagens_individuais(threshold=0.5):
    """
    🖼️ TESTE COM IMAGENS INDIVIDUAIS
    """
    logger.info("🖼️ Testando com imagens individuais...")
    
    from test_probabilidade import test_single_image_with_probability
    from keras.models import load_model
    
    model_path = 'Results/models/modelo_conservador.keras'
    model = load_model(model_path)
    
    # Encontrar algumas imagens para teste
    from Config.config import PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR
    import os
    import glob
    
    test_images = []
    for folder in [PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR]:
        if os.path.exists(folder):
            images = glob.glob(os.path.join(folder, "*.png"))[:2]  # 2 de cada
            test_images.extend(images)
    
    print(f"\n🧪 TESTANDO {len(test_images)} IMAGENS INDIVIDUAIS:")
    print("=" * 50)
    
    for img_path in test_images:
        if os.path.exists(img_path):
            # Usar a função do test_probabilidade mas com nosso modelo
            from Src.data_loader import safe_imread
            import cv2
            import numpy as np
            from Config.config import IMG_HEIGHT, IMG_WIDTH
            
            # Carregar e processar imagem
            processed_img = safe_imread(img_path)
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
                
                print(f"📄 {os.path.basename(img_path)}:")
                print(f"   📊 Probabilidade: {percentual:.1f}%")
                print(f"   🏷️  Classificação: {classificacao}")
                print(f"   💪 Confiança: {confianca:.1f}%")
                print("   " + "-" * 30)

if __name__ == "__main__":
    # 1. Teste completo do modelo
    threshold_ideal, resultados = testar_modelo_conservador()
    
    # 2. Teste com imagens individuais
    print("\n" + "=" * 70)
    testar_imagens_individuais(threshold_ideal)
    
    # 3. Salvar configuração ideal
    print(f"\n💡 CONFIGURAÇÃO RECOMENDADA:")
    print(f"   Use threshold: {threshold_ideal:.1f}")
    print(f"   Modelo: modelo_conservador.keras")
    print(f"   Acurácia esperada: {resultados['accuracy']:.1%}")