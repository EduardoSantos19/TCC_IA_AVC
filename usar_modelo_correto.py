# usar_modelo_correto.py
import os
import shutil

def garantir_modelo_correto():
    """
    🎯 GARANTE que o modelo correto seja o padrão
    """
    modelo_bom = 'Results/models/modelo_conservador.keras'
    modelo_padrao = 'Results/models/modelo_final_avc.keras'
    
    if os.path.exists(modelo_bom):
        # Copiar o modelo bom para o nome padrão
        shutil.copy2(modelo_bom, modelo_padrao)
        print(f"✅ Modelo correto copiado para: {modelo_padrao}")
        
        # Listar todos os modelos
        print("\n📁 MODELOS EXISTENTES:")
        modelos_dir = 'Results/models/'
        for arquivo in os.listdir(modelos_dir):
            if arquivo.endswith('.keras'):
                status = "✅ BOM" if "conservador" in arquivo else "❌ RUIM" 
                print(f"   {status} {arquivo}")
                
        print(f"\n💡 USE SEMPRE: {modelo_padrao}")
    else:
        print("❌ Modelo conservador não encontrado!")

if __name__ == "__main__":
    garantir_modelo_correto()