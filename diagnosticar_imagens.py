import os
import cv2
from Config.config import PROCESSED_POSITIVE_DIR, PROCESSED_NEGATIVE_DIR, PROCESSED_UNKNOWN_DIR

def diagnosticar_pasta(pasta):
    """Diagnostica problemas em uma pasta de imagens"""
    print(f"\n🔍 DIAGNÓSTICO: {pasta}")
    
    if not os.path.exists(pasta):
        print(f"❌ PASTA NÃO EXISTE: {pasta}")
        return
    
    arquivos = os.listdir(pasta)
    imagens = [f for f in arquivos if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📁 Total de arquivos: {len(arquivos)}")
    print(f"🖼️  Arquivos de imagem: {len(imagens)}")
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada")
        return
    
    # Testa as primeiras 5 imagens
    for i, img_file in enumerate(imagens[:5]):
        caminho = os.path.join(pasta, img_file)
        print(f"\n  📄 Imagem {i+1}: {img_file}")
        
        # Verifica se o arquivo existe
        if not os.path.exists(caminho):
            print("     ❌ Arquivo não existe")
            continue
            
        # Verifica tamanho do arquivo
        tamanho = os.path.getsize(caminho)
        print(f"     📏 Tamanho: {tamanho} bytes")
        
        if tamanho == 0:
            print("     ❌ Arquivo vazio (0 bytes)")
            continue
            
        # Tenta ler com OpenCV
        img = cv2.imread(caminho)
        if img is None:
            print("     ❌ OpenCV não conseguiu ler")
            # Tenta ler como binário para ver o conteúdo
            try:
                with open(caminho, 'rb') as f:
                    header = f.read(10)
                    print(f"     🔍 Cabeçalho: {header[:10]}")
            except Exception as e:
                print(f"     ❌ Erro ao ler arquivo: {e}")
        else:
            print(f"     ✅ OpenCV leu: {img.shape}")
    
    return len(imagens)

# Diagnóstico completo
if __name__ == "__main__":
    print("🎯 DIAGNÓSTICO COMPLETO DO DATASET")
    print("=" * 50)
    
    pastas = {
        "POSITIVE": PROCESSED_POSITIVE_DIR,
        "NEGATIVE": PROCESSED_NEGATIVE_DIR, 
        "UNKNOWN": PROCESSED_UNKNOWN_DIR
    }
    
    totais = {}
    for nome, pasta in pastas.items():
        total = diagnosticar_pasta(pasta)
        totais[nome] = total
    
    print("\n📊 RESUMO:")
    for nome, total in totais.items():
        status = "✅" if total and total > 0 else "❌"
        print(f"   {nome}: {total} imagens {status}")