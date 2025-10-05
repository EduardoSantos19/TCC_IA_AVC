import os
import sys
sys.path.append('C:/Projetos/TCC_EEG_IA')

from Config.config import (
    RAW_POSITIVE_DIR, RAW_NEGATIVE_DIR, RAW_UNKNOWN_DIR,
    PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR
)

def verificar_pastas():
    """Verifica o conteúdo de todas as pastas"""
    print("🔍 VERIFICANDO FLUXO RAW → PROCESSED")
    print("=" * 50)
    
    # Verifica pastas RAW
    print("\n📁 PASTA RAW (ORIGINAL):")
    for classe, pasta in [("positive", RAW_POSITIVE_DIR), 
                         ("negative", RAW_NEGATIVE_DIR),
                         ("unknown", RAW_UNKNOWN_DIR)]:
        if os.path.exists(pasta):
            arquivos = os.listdir(pasta)
            imagens = [f for f in arquivos if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"   {classe}: {len(imagens)} imagens / {len(arquivos)} arquivos")
        else:
            print(f"   {classe}: ❌ PASTA NÃO EXISTE")

    # Verifica pastas PROCESSED
    print("\n📁 PASTA PROCESSED (PRÉ-PROCESSADA):")
    for classe, pasta in [("positive", PROCESSED_POSITIVE_DIR), 
                         ("negative", PROCESSED_NEGATIVE_DIR),
                         ("unknown", PROCESSED_UNKNOWN_DIR)]:
        if os.path.exists(pasta):
            arquivos = os.listdir(pasta)
            imagens = [f for f in arquivos if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"   {classe}: {len(imagens)} imagens / {len(arquivos)} arquivos")
        else:
            print(f"   {classe}: ❌ PASTA NÃO EXISTE")

def executar_preprocessamento():
    """Executa o pré-processamento manualmente"""
    print("\n🔄 EXECUTANDO PRÉ-PROCESSAMENTO")
    print("=" * 50)
    
    try:
        from Preprocessing.preprocess_image import preprocess_batch
        
        total_processadas = 0
        classes = {
            'positive': (RAW_POSITIVE_DIR, PROCESSED_POSITIVE_DIR),
            'negative': (RAW_NEGATIVE_DIR, PROCESSED_NEGATIVE_DIR),
            'unknown': (RAW_UNKNOWN_DIR, PROCESSED_UNKNOWN_DIR)
        }
        
        for classe, (input_dir, output_dir) in classes.items():
            if os.path.exists(input_dir):
                print(f"   Processando {classe}...")
                count = preprocess_batch(input_dir, output_dir)
                total_processadas += count
                print(f"   ✅ {count} imagens {classe} processadas")
            else:
                print(f"   ❌ {classe}: Pasta raw não existe")
        
        print(f"\n🎉 TOTAL: {total_processadas} imagens processadas")
        return total_processadas
        
    except ImportError as e:
        print(f"❌ ERRO DE IMPORTAÇÃO: {e}")
        return 0
    except Exception as e:
        print(f"❌ ERRO NO PROCESSAMENTO: {e}")
        return 0

def main():
    """Fluxo completo de teste"""
    print("🧪 TESTE DO FLUXO RAW → PROCESSED")
    print("=" * 50)
    
    # 1. Verifica estado inicial
    verificar_pastas()
    
    # 2. Executa pré-processamento
    input("\n⏰ Pressione Enter para executar o pré-processamento...")
    total = executar_preprocessamento()
    
    # 3. Verifica resultado
    input("\n⏰ Pressione Enter para verificar o resultado...")
    verificar_pastas()
    
    # 4. Conclusão
    print("\n" + "=" * 50)
    if total > 0:
        print(f"✅ SUCESSO! {total} imagens processadas")
        print("📊 Fluxo: RAW → PROCESSED ✓")
    else:
        print("❌ FALHA! Nenhuma imagem foi processada")
        print("💡 Verifique:")
        print("   - Pastas RAW existem e têm imagens")
        print("   - Importação do módulo Preprocessing")
        print("   - Permissões de escrita nas pastas PROCESSED")

if __name__ == "__main__":
    main()