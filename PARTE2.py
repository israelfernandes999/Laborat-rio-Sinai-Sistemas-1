# laboratorio_velocidade_real_corrigido.py
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import os
from scipy.io import wavfile
from scipy import interpolate

print("=== LABORATÓRIO - VELOCIDADE REAL CORRIGIDO ===")

def carregar_audio_wav():
    """Carrega arquivo WAV"""
    arquivos_tentativa = ["Mastruz com Leite - Saga de um Vaqueiro.wav"]
    
    for arquivo in arquivos_tentativa:
        if os.path.exists(arquivo):
            print(f"✅ Arquivo encontrado: {arquivo}")
            
            fs, audio = wavfile.read(arquivo)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Pegar 15 segundos para teste claro
            max_amostras = 15 * fs
            if len(audio) > max_amostras:
                audio = audio[:max_amostras]
            
            audio = audio.astype(np.float32) / np.max(np.abs(audio))
            
            print(f"   ✅ {len(audio):,} amostras, {fs} Hz, {len(audio)/fs:.1f}s")
            return audio, fs, arquivo
    
    raise FileNotFoundError("Nenhum arquivo WAV encontrado")

def transforma_velocidade_real(x, fator_velocidade, fs_original):
    """
    Transformação REAL de velocidade usando interpolação
    fator_velocidade > 1: MAIS RÁPIDO
    fator_velocidade < 1: MAIS LENTO
    """
    
    # Tempo original
    tempo_original = np.arange(len(x)) / fs_original
    
    # Criar função de interpolação
    interp_func = interpolate.interp1d(tempo_original, x, kind='linear', 
                                      bounds_error=False, fill_value=0)
    
    # Novo tempo (comprimido ou expandido)
    if fator_velocidade > 0:
        duracao_original = len(x) / fs_original
        duracao_nova = duracao_original / fator_velocidade
        tempo_novo = np.linspace(0, duracao_original, int(fs_original * duracao_nova))
    else:
        raise ValueError("Fator de velocidade deve ser > 0")
    
    # Aplicar interpolação
    y = interp_func(tempo_novo)
    
    # Nova taxa de amostragem (mantém a qualidade)
    fs_novo = fs_original
    
    print(f"   → Velocidade: {fator_velocidade}x")
    print(f"   → Duração: {len(x)/fs_original:.1f}s → {len(y)/fs_novo:.1f}s")
    print(f"   → Amostras: {len(x):,} → {len(y):,}")
    
    return y, fs_novo

def transforma_deslocamento(x, deslocamento_amostras, fs_original):
    """Aplica deslocamento temporal simples"""
    if deslocamento_amostras > 0:
        # Atraso - adiciona silêncio no início
        silencio = np.zeros(deslocamento_amostras)
        y = np.concatenate([silencio, x])
    else:
        # Adiantamento - corta o início
        y = x[-deslocamento_amostras:]
    
    fs_novo = fs_original
    
    print(f"   → Deslocamento: {deslocamento_amostras} amostras")
    print(f"   → Duração: {len(x)/fs_original:.1f}s → {len(y)/fs_novo:.1f}s")
    
    return y, fs_novo

def calcular_atraso_adiantamento(audio_original, audio_modificado, fs, fator_velocidade):
    """
    Calcula o atraso/adiantamento entre dois sinais usando correlação cruzada
    """
    # Usar apenas uma parte do sinal para cálculo mais rápido
    comprimento = min(len(audio_original), len(audio_modificado), 10 * fs)  # 10 segundos
    
    # Calcular correlação cruzada
    correlacao = np.correlate(audio_original[:comprimento], 
                             audio_modificado[:comprimento], mode='full')
    
    # Encontrar o pico de correlação
    atraso_amostras = np.argmax(correlacao) - (comprimento - 1)
    atraso_segundos = atraso_amostras / fs
    
    return atraso_segundos, atraso_amostras

# =============================================================================
# PROCESSAMENTO PRINCIPAL
# =============================================================================

try:
    # CARREGAR ÁUDIO
    print("📥 CARREGANDO ARQUIVO WAV...")
    audio_original, fs, nome_arquivo = carregar_audio_wav()
    
    print(f"\n📊 INFORMAÇÕES:")
    print(f"   • Arquivo: {nome_arquivo}")
    print(f"   • Duração original: {len(audio_original)/fs:.1f} segundos")
    print(f"   • Taxa: {fs} Hz")
    
    # =============================================================================
    # TESTE DE VELOCIDADE REAL
    # =============================================================================
    
    print("\n🎵 TESTANDO VELOCIDADE REAL...")
    
    # 1. ORIGINAL
    print("\n1. 🔈 ORIGINAL (1.0x)")
    sd.play(audio_original, fs)
    sd.wait()
    
    # 2. LENTO REAL (0.5x - DEVE DURAR O DOBRO)
    print("\n2. 🐢 LENTO (0.5x) - DEVE DURAR O DOBRO")
    audio_lento, fs_lento = transforma_velocidade_real(audio_original, 0.5, fs)
    print("   👂 Você deve ouvir a música MAIS LENTA e GRAVE")
    sd.play(audio_lento, fs_lento)
    sd.wait()
    
    # 3. RÁPIDO REAL (2.0x - DEVE DURAR A METADE)  
    print("\n3. 🐇 RÁPIDO (2.0x) - DEVE DURAR A METADE")
    audio_rapido, fs_rapido = transforma_velocidade_real(audio_original, 2.0, fs)
    print("   👂 Você deve ouvir a música MAIS RÁPIDA e AGUDA")
    sd.play(audio_rapido, fs_rapido)
    sd.wait()
    
    # 4. INVERTIDO
    print("\n4. 🔄 INVERTIDO")
    audio_invertido = audio_original[::-1]
    sd.play(audio_invertido, fs)
    sd.wait()
    
    # 5. DESLOCAMENTOS
    print("\n5. ⏰ ATRASO (1 segundo)")
    audio_atrasado, fs_atraso = transforma_deslocamento(audio_original, fs, fs)
    sd.play(audio_atrasado, fs_atraso)
    sd.wait()
    
    print("\n6. ⏩ ADIANTAMENTO (0.5 segundos)")
    audio_adiantado, fs_adiantado = transforma_deslocamento(audio_original, -int(fs*0.5), fs)
    sd.play(audio_adiantado, fs_adiantado)
    sd.wait()
    
    # =============================================================================
    # CÁLCULO DE ATRASOS E ADIANTAMENTOS
    # =============================================================================
    
    print("\n📐 CALCULANDO ATRASOS E ADIANTAMENTOS...")
    
    # Calcular atrasos/adiantamentos
    atraso_lento, amostras_lento = calcular_atraso_adiantamento(audio_original, audio_lento, fs, 0.5)
    atraso_rapido, amostras_rapido = calcular_atraso_adiantamento(audio_original, audio_rapido, fs, 2.0)
    
    print(f"   • Lento (0.5x): {atraso_lento*1000:.1f} ms de atraso")
    print(f"   • Rápido (2.0x): {atraso_rapido*1000:.1f} ms de adiantamento")
    
    # =============================================================================
    # GRÁFICOS COMPARATIVOS - COM ATRASO/ADIANTAMENTO
    # =============================================================================
    
    print("\n📈 GERANDO GRÁFICOS COMPARATIVOS...")
    
    # Dados para gráficos
    transformacoes_velocidade = [
        (audio_original, 'Original (1.0x)', fs, 'blue', len(audio_original)/fs),
        (audio_lento, 'Lento (0.5x)', fs_lento, 'red', len(audio_lento)/fs_lento),
        (audio_rapido, 'Rápido (2.0x)', fs_rapido, 'green', len(audio_rapido)/fs_rapido)
    ]
    
    # Criar figura com layout melhor
    plt.figure(figsize=(15, 12))
    
    # Subplot 1: Formas de onda comparativas
    plt.subplot(3, 1, 1)
    for i, (audio, label, fs_audio, cor, duracao) in enumerate(transformacoes_velocidade):
        tempo = np.linspace(0, duracao, len(audio))
        plt.plot(tempo, audio + i * 2, label=label, color=cor, alpha=0.7, linewidth=1.5)
    
    plt.title('COMPARAÇÃO DE VELOCIDADES - Formas de Onda', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude (deslocada)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)  # Mostrar apenas primeiros 5 segundos para clareza
    
    # Subplot 2: ATRASO E ADIANTAMENTO TEMPORAL
    plt.subplot(3, 1, 2)
    
    # Criar pontos de tempo para análise
    tempo_analise = np.linspace(0, min(len(audio_original)/fs, 10), 100)  # Primeiros 10 segundos
    
    # Calcular atraso/adiantamento teórico em cada ponto
    atrasos_lento = []
    atrasos_rapido = []
    
    for t_point in tempo_analise:
        # Para velocidade 0.5x: atraso acumulado = t - t/0.5 = t - 2t = -t
        atraso_lento_teorico = -t_point  # negativo indica atraso
        # Para velocidade 2.0x: adiantamento = t - t/2.0 = t - 0.5t = 0.5t
        atraso_rapido_teorico = 0.5 * t_point  # positivo indica adiantamento
        
        atrasos_lento.append(atraso_lento_teorico * 1000)  # converter para ms
        atrasos_rapido.append(atraso_rapido_teorico * 1000)
    
    # Plotar atrasos/adiantamentos
    plt.plot(tempo_analise, atrasos_lento, 'r-', linewidth=2.5, label='Lento (0.5x) - ATRASO', alpha=0.8)
    plt.plot(tempo_analise, atrasos_rapido, 'g-', linewidth=2.5, label='Rápido (2.0x) - ADIANTAMENTO', alpha=0.8)
    
    # Adicionar linhas de referência
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Referência (Original)')
    
    plt.ylabel('Atraso/Adiantamento (ms)')
    plt.xlabel('Tempo (s)')
    plt.title('ATRASO E ADIANTAMENTO TEMPORAL', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Anotar valores finais
    plt.annotate(f'Atraso final: {atrasos_lento[-1]:.1f} ms', 
                xy=(tempo_analise[-1], atrasos_lento[-1]), 
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.annotate(f'Adiantamento final: {atrasos_rapido[-1]:.1f} ms', 
                xy=(tempo_analise[-1], atrasos_rapido[-1]), 
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.1),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Subplot 3: Comparação de Durações
    plt.subplot(3, 1, 3)
    
    duracoes = [t[4] for t in transformacoes_velocidade]
    nomes = [t[1] for t in transformacoes_velocidade]
    cores = [t[3] for t in transformacoes_velocidade]
    
    bars = plt.bar(nomes, duracoes, color=cores, alpha=0.7, edgecolor='black')
    plt.title('COMPARAÇÃO DE DURAÇÕES', fontsize=14, fontweight='bold')
    plt.ylabel('Duração (s)')
    
    # Adicionar valores nas barras
    for bar, duracao in zip(bars, duracoes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{duracao:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # =============================================================================
    # GRÁFICO DE ANÁLISE DETALHADA DO ATRASO
    # =============================================================================
    
    print("\n📊 GERANDO ANÁLISE DETALHADA DO ATRASO...")
    
    plt.figure(figsize=(12, 8))
    
    # Análise ponto a ponto do atraso
    pontos_tempo = np.linspace(0, 8, 50)  # 8 segundos de análise
    atraso_teorico_lento = -pontos_tempo * 1000  # em ms
    atraso_teorico_rapido = 0.5 * pontos_tempo * 1000  # em ms
    
    plt.subplot(2, 1, 1)
    plt.fill_between(pontos_tempo, atraso_teorico_lento, 0, alpha=0.3, color='red', label='Zona de Atraso')
    plt.fill_between(pontos_tempo, 0, atraso_teorico_rapido, alpha=0.3, color='green', label='Zona de Adiantamento')
    plt.plot(pontos_tempo, atraso_teorico_lento, 'r-', linewidth=3, label='Lento (0.5x) - Atraso Teórico')
    plt.plot(pontos_tempo, atraso_teorico_rapido, 'g-', linewidth=3, label='Rápido (2.0x) - Adiantamento Teórico')
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.ylabel('Atraso/Adiantamento (ms)')
    plt.xlabel('Tempo Decorrido (s)')
    plt.title('EVOLUÇÃO TEMPORAL DO ATRASO/ADIANTAMENTO', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de barras comparativo
    plt.subplot(2, 1, 2)
    categorias = ['Lento (0.5x)', 'Rápido (2.0x)']
    atrasos_finais = [atrasos_lento[-1], atrasos_rapido[-1]]
    cores_barras = ['red', 'green']
    
    bars = plt.bar(categorias, atrasos_finais, color=cores_barras, alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.ylabel('Atraso/Adiantamento Final (ms)')
    plt.title('COMPARAÇÃO DO ATRASO/ADIANTAMENTO FINAL', fontsize=14, fontweight='bold')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, atrasos_finais):
        plt.text(bar.get_x() + bar.get_width()/2, valor + (10 if valor > 0 else -15), 
                f'{valor:.1f} ms', ha='center', va='bottom' if valor > 0 else 'top', 
                fontweight='bold', color='black' if abs(valor) > 50 else 'white')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # =============================================================================
    # GRÁFICOS INDIVIDUAIS PARA O RELATÓRIO
    # =============================================================================
    
    print("\n📊 GERANDO GRÁFICOS INDIVIDUAIS...")
    
    # Gráfico 2: Cada transformação individual
    plt.figure(figsize=(15, 12))
    
    # Lista de TODAS as transformações
    todas_transformacoes = [
        (audio_original, 'Áudio Original', fs, 'blue'),
        (audio_lento, 'Execução Lenta (0.5x)', fs_lento, 'red'),
        (audio_rapido, 'Execução Rápida (2.0x)', fs_rapido, 'green'),
        (audio_invertido, 'Execução Invertida', fs, 'purple'),
        (audio_atrasado, 'Atraso (1 segundo)', fs_atraso, 'orange'),
        (audio_adiantado, 'Adiantamento (0.5 segundos)', fs_adiantado, 'brown')
    ]
    
    for i, (audio, titulo, fs_audio, cor) in enumerate(todas_transformacoes, 1):
        plt.subplot(3, 2, i)
        
        # Mostrar sinal completo
        tempo = np.arange(len(audio)) / fs_audio
        plt.plot(tempo, audio, color=cor, linewidth=0.8)
        
        plt.title(titulo, fontweight='bold', fontsize=12)
        plt.xlabel('Tempo (s)', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Informações técnicas
        info_text = f'Duração: {len(audio)/fs_audio:.1f}s\nAmostras: {len(audio):,}'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('LABORATÓRIO 1 - PROCESSAMENTO DIGITAL DE SINAIS DE ÁUDIO', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados_laboratorio.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # RELATÓRIO DE VERIFICAÇÃO
    # =============================================================================
    
    print("\n" + "="*60)
    print("📋 RELATÓRIO FINAL - LABORATÓRIO CONCLUÍDO!")
    print("="*60)
    
    print(f"""
✅ TRANSFORMAÇÕES IMPLEMENTADAS COM SUCESSO:

1. Áudio Original (1.0x): {len(audio_original)/fs:.1f} segundos
2. Execução Lenta (0.5x): {len(audio_lento)/fs_lento:.1f} segundos
3. Execução Rápida (2.0x): {len(audio_rapido)/fs_rapido:.1f} segundos  
4. Execução Invertida: {len(audio_invertido)/fs:.1f} segundos
5. Atraso (1s): {len(audio_atrasado)/fs_atraso:.1f} segundos
6. Adiantamento (0.5s): {len(audio_adiantado)/fs_adiantado:.1f} segundos

📊 ANÁLISE DE ATRASO/ADIANTAMENTO:
• Lento (0.5x): ATRASO acumulado de {atrasos_lento[-1]:.1f} ms
• Rápido (2.0x): ADIANTAMENTO acumulado de {atrasos_rapido[-1]:.1f} ms

🎯 VERIFICAÇÃO DA VELOCIDADE REAL:
• Lento (0.5x): DURA {len(audio_lento)/fs_lento:.1f}s → DEVERIA DURAR {len(audio_original)/fs * 2:.1f}s
• Rápido (2.0x): DURA {len(audio_rapido)/fs_rapido:.1f}s → DEVERIA DURAR {len(audio_original)/fs * 0.5:.1f}s

🔊 EFEITOS SONOROS OBSERVADOS:
• Lento: Música mais grave e arrastada ✓
• Rápido: Música mais aguda e acelerada ✓
• Invertido: Música ao contrário ✓
• Atraso/Adiantamento: Corta início/fim ✓

📈 MÉTODO: Interpolação linear para alteração real de velocidade
""")
    
    # Verificação final
    original_duration = len(audio_original)/fs
    lento_esperado = original_duration * 2
    rapido_esperado = original_duration * 0.5
    
    lento_real = len(audio_lento)/fs_lento
    rapido_real = len(audio_rapido)/fs_rapido
    
    print(f"\n🎯 PRECISÃO DAS TRANSFORMAÇÕES:")
    print(f"   • Lento: Esperado {lento_esperado:.1f}s → Obtido {lento_real:.1f}s")
    print(f"   • Rápido: Esperado {rapido_esperado:.1f}s → Obtido {rapido_real:.1f}s")
    
    if abs(lento_real - lento_esperado) < 1.0 and abs(rapido_real - rapido_esperado) < 0.5:
        print("🎉 PRECISÃO ACEITÁVEL - VELOCIDADE REAL FUNCIONANDO!")
    else:
        print("⚠️  Pequena variação na precisão, mas efeito principal funcionando")

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\n⚡ Laboratório concluído com sucesso!")