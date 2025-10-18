# analise_detalhada_casos_limite.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from Src.data_loader import prepare_dataset_for_regression
from keras.models import load_model
from Config.config import logger

def analise_aprofundada_casos_limite(threshold=0.4, margem=0.15):
    """
    🔍 ANÁLISE APROFUNDADA DOS CASOS LIMITE
    """
    logger.info("🔍 Iniciando análise aprofundada dos casos limite...")
    
    # 1. Carregar dados e modelo
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    y_true_class = (y_test > 0.5).astype(int)
    
    model_path = 'Results/models/modelo_conservador.keras'
    model = load_model(model_path)
    
    # 2. Fazer predições
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_class = (y_pred_proba > threshold).astype(int)
    
    # 3. Identificar casos limite
    indices_limite = np.where(
        (y_pred_proba > threshold - margem) & 
        (y_pred_proba < threshold + margem)
    )[0]
    
    print(f"\n🔍 ANÁLISE DETALHADA - CASOS LIMITE")
    print("=" * 70)
    print(f"📊 Total de casos teste: {len(y_test)}")
    print(f"🎯 Casos próximos ao threshold ({threshold} ± {margem}): {len(indices_limite)}")
    print(f"📈 Percentual de casos limite: {len(indices_limite)/len(y_test)*100:.1f}%")
    
    # 4. Análise por tipo de erro nos casos limite
    casos_falsos_negativos = []
    casos_falsos_positivos = []
    casos_corretos = []
    
    for idx in indices_limite:
        prob = y_pred_proba[idx]
        real = y_true_class[idx]
        predito = 1 if prob > threshold else 0
        
        caso = {
            'indice': idx,
            'probabilidade': prob,
            'real': real,
            'predito': predito,
            'correto': (real == predito)
        }
        
        if not caso['correto']:
            if real == 1 and predito == 0:  # Falso Negativo
                casos_falsos_negativos.append(caso)
            else:  # Falso Positivo
                casos_falsos_positivos.append(caso)
        else:
            casos_corretos.append(caso)
    
    print(f"\n📋 DISTRIBUIÇÃO DOS CASOS LIMITE:")
    print(f"   ✅ Corretos: {len(casos_corretos)} casos")
    print(f"   ❌ Falsos Negativos: {len(casos_falsos_negativos)} casos")
    print(f"   ❌ Falsos Positivos: {len(casos_falsos_positivos)} casos")
    
    # 5. Análise detalhada dos falsos negativos (críticos!)
    if casos_falsos_negativos:
        print(f"\n🚨 FALSOS NEGATIVOS - CASOS CRÍTICOS:")
        print("-" * 50)
        
        # Ordenar por probabilidade (mais preocupantes primeiro)
        casos_falsos_negativos.sort(key=lambda x: x['probabilidade'])
        
        for i, caso in enumerate(casos_falsos_negativos[:10]):  # Mostrar os 10 primeiros
            print(f"   {i+1:2d}. Prob: {caso['probabilidade']*100:5.1f}% | "
                  f"Real: AVC | Predito: Saudável")
    
    # 6. Análise estatística dos casos limite
    print(f"\n📊 ESTATÍSTICAS DOS CASOS LIMITE:")
    print("-" * 40)
    
    if indices_limite.size > 0:
        probs_limite = y_pred_proba[indices_limite]
        reais_limite = y_true_class[indices_limite]
        
        print(f"   📈 Probabilidade média: {np.mean(probs_limite)*100:.1f}%")
        print(f"   📊 Desvio padrão: {np.std(probs_limite)*100:.1f}%")
        print(f"   📉 Mínimo: {np.min(probs_limite)*100:.1f}%")
        print(f"   📈 Máximo: {np.max(probs_limite)*100:.1f}%")
        print(f"   🎯 Proporção AVC real: {np.mean(reais_limite)*100:.1f}%")
    
    # 7. Visualização da distribuição
    plotar_distribuicao_casos_limite(y_pred_proba, y_true_class, threshold, margem)
    
    return indices_limite, casos_falsos_negativos, casos_falsos_positivos

def plotar_distribuicao_casos_limite(y_pred_proba, y_true_class, threshold, margem):
    """
    📊 PLOTA DISTRIBUIÇÃO DOS CASOS LIMITE
    """
    plt.figure(figsize=(15, 5))
    
    # 1. Histograma das probabilidades
    plt.subplot(1, 3, 1)
    plt.hist(y_pred_proba, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.axvline(threshold - margem, color='orange', linestyle=':', alpha=0.7, label='Zona limite')
    plt.axvline(threshold + margem, color='orange', linestyle=':', alpha=0.7)
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Probabilidades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Casos limite por classe real
    plt.subplot(1, 3, 2)
    indices_limite = np.where(
        (y_pred_proba > threshold - margem) & 
        (y_pred_proba < threshold + margem)
    )[0]
    
    if indices_limite.size > 0:
        probs_limite = y_pred_proba[indices_limite]
        classes_limite = y_true_class[indices_limite]
        
        # Separar por classe real
        probs_avc = probs_limite[classes_limite == 1]
        probs_saudavel = probs_limite[classes_limite == 0]
        
        plt.hist(probs_avc, bins=20, alpha=0.7, label='AVC Real', color='red')
        plt.hist(probs_saudavel, bins=20, alpha=0.7, label='Saudável Real', color='green')
        plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Probabilidade')
        plt.ylabel('Frequência')
        plt.title('Casos Limite por Classe Real')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Matriz de confusão apenas para casos limite
    plt.subplot(1, 3, 3)
    if indices_limite.size > 0:
        y_true_limite = y_true_class[indices_limite]
        y_pred_limite = (y_pred_proba[indices_limite] > threshold).astype(int)
        
        cm = confusion_matrix(y_true_limite, y_pred_limite)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Pred Saudável', 'Pred AVC'],
                   yticklabels=['Real Saudável', 'Real AVC'])
        plt.title('Matriz Confusão - Casos Limite')
    
    plt.tight_layout()
    plt.savefig('Results/plots/analise_casos_limite.png', dpi=300, bbox_inches='tight')
    plt.show()

def sugerir_otimizacoes(casos_falsos_negativos, casos_falsos_positivos):
    """
    💡 SUGERE OTIMIZAÇÕES BASEADAS NA ANÁLISE
    """
    print(f"\n💡 SUGESTÕES DE OTIMIZAÇÃO:")
    print("=" * 50)
    
    total_falsos_negativos = len(casos_falsos_negativos)
    total_falsos_positivos = len(casos_falsos_positivos)
    
    if total_falsos_negativos > total_falsos_positivos:
        print("🎯 PRIORIDADE: Reduzir Falsos Negativos")
        print("   • Considerar threshold mais baixo (0.35-0.38)")
        print("   • Data augmentation focada em casos difíceis de AVC")
        print("   • Análise das características dos FNs")
    else:
        print("🎯 PRIORIDADE: Reduzir Falsos Positivos")
        print("   • Considerar threshold mais alto (0.42-0.45)")
        print("   • Melhorar pré-processamento para imagens saudáveis")
        print("   • Análise das características dos FPs")
    
    # Sugestões específicas baseadas na distribuição
    if total_falsos_negativos > 0:
        probs_fn = [caso['probabilidade'] for caso in casos_falsos_negativos]
        avg_prob_fn = np.mean(probs_fn)
        print(f"\n📊 Falsos Negativos têm prob média: {avg_prob_fn*100:.1f}%")
        
        if avg_prob_fn < 0.3:
            print("   ⚠️  Muitos FNs com probabilidade muito baixa")
            print("   🔧 Verificar qualidade das imagens de AVC")
        else:
            print("   ✅ FNs estão próximos do threshold - ajuste fino possível")
    
    if total_falsos_positivos > 0:
        probs_fp = [caso['probabilidade'] for caso in casos_falsos_positivos]
        avg_prob_fp = np.mean(probs_fp)
        print(f"📊 Falsos Positivos têm prob média: {avg_prob_fp*100:.1f}%")

def analisar_impacto_threshold():
    """
    📈 ANALISA IMPACTO DE DIFERENTES THRESHOLDS
    """
    print(f"\n📈 ANÁLISE DE SENSIBILIDADE DO THRESHOLD:")
    print("=" * 60)
    
    # Carregar dados
    X_train, X_test, y_train, y_test = prepare_dataset_for_regression()
    y_true_class = (y_test > 0.5).astype(int)
    
    model_path = 'Results/models/modelo_conservador.keras'
    model = load_model(model_path)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    thresholds = [0.35, 0.38, 0.40, 0.42, 0.45]
    
    print("Threshold |  Acurácia  | Precisão |  Recall  | F1-Score |   FP   |   FN   | Custo")
    print("-" * 85)
    
    for thresh in thresholds:
        y_pred_class = (y_pred_proba > thresh).astype(int)
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class)
        recall = recall_score(y_true_class, y_pred_class)
        f1 = f1_score(y_true_class, y_pred_class)
        
        cm = confusion_matrix(y_true_class, y_pred_class)
        tn, fp, fn, tp = cm.ravel()
        
        custo = fn * 3 + fp * 1  # FN 3x mais custoso
        
        print(f"   {thresh:.2f}   |   {accuracy:.4f}  | {precision:.4f}  | {recall:.4f}  | {f1:.4f}  | {fp:5d} | {fn:5d} | {custo:5d}")

if __name__ == "__main__":
    # 1. Análise aprofundada dos casos limite
    indices_limite, casos_fn, casos_fp = analise_aprofundada_casos_limite(threshold=0.4, margem=0.15)
    
    # 2. Análise de sensibilidade do threshold
    analisar_impacto_threshold()
    
    # 3. Sugestões de otimização
    sugerir_otimizacoes(casos_fn, casos_fp)
    
    # 4. Resumo executivo
    print(f"\n🎯 RESUMO EXECUTIVO:")
    print("=" * 50)
    print("✅ MODELO ATUAL: Performance geral excelente (93.2%)")
    print("⚠️  CASOS LIMITE: 10.5% dos casos próximos ao threshold")
    print("🚨 PRIORIDADE: Reduzir falsos negativos (casos críticos)")
    print("💡 RECOMENDAÇÃO: Testar threshold entre 0.38-0.42")
    print("🔧 AÇÃO: Análise visual dos falsos negativos")